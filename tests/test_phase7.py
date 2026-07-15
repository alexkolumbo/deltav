"""Phase 7: network embeddings, vector RAG memory, price market."""
import asyncio

import httpx
import pytest

from deltav.chain.blockchain import build_genesis_state
from deltav.chain.transaction import TxType
from deltav.compute.base import DeviceInfo, EmbedRequest
from deltav.compute.mock import MockBackend
from deltav.config import DVT, Genesis
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.overlay.memory import VectorMemory, cosine
from deltav.router.scoring import NodeView, score_node

from conftest import MultiTransport
from test_chain import make_receipt_tx, make_tx, register_node

CHAT_MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5-GGUF::nomic-embed-text-v1.5.Q4_K_M.gguf"
URL = "http://127.0.0.1:9801"
GW_URL = "http://127.0.0.1:9800"


# ------------------------------------------------------------ price market

def test_receipt_pays_node_asking_price(genesis, alice, bob):
    state = build_genesis_state(genesis)
    tx = make_tx(bob, state, TxType.REGISTER_NODE, {
        "endpoint": "http://n:1", "hardware": {}, "models": ["m"], "price_per_token": 25,
    })
    state.apply_tx(tx, 1)
    assert state.nodes[bob.address].price_per_token == 25

    balance_before = state.account(bob.address).balance
    state.apply_tx(make_receipt_tx(bob, alice, state, tokens_in=10, tokens_out=20), 2)
    paid = 30 * 25  # node's price, not the network default of 10
    receipt = next(iter(state.receipts.values()))
    assert receipt.price_paid == paid
    assert state.account(bob.address).balance == balance_before + paid + state.params.inference_reward


def test_overpriced_receipt_rejected_by_limit(genesis, alice, bob):
    from deltav.chain.state import StateError

    state = build_genesis_state(genesis)
    state.apply_tx(make_tx(bob, state, TxType.REGISTER_NODE, {
        "endpoint": "http://n:1", "price_per_token": 10_000,
    }), 1)
    with pytest.raises(StateError, match="price exceeds"):
        state.apply_tx(make_receipt_tx(bob, alice, state, tokens_in=10, tokens_out=20,
                                       price_limit=1000), 2)


def test_router_prefers_cheaper_node():
    base = dict(endpoint="http://n:1", vram_mb=12282, models=[CHAT_MODEL],
                reputation=0.5, stake=0, last_seen=0)
    cheap = NodeView(address="dv1cheap", price_per_token=5, **base)
    pricey = NodeView(address="dv1pricey", price_per_token=40, **base)
    assert score_node(cheap, CHAT_MODEL, 0, default_price=10) > \
           score_node(pricey, CHAT_MODEL, 0, default_price=10)


# ------------------------------------------------------------- embeddings

def test_mock_embeddings_are_semantic():
    backend = MockBackend()
    result = backend.embed(EmbedRequest(
        texts=["кот сидит на крыше", "кот спит на крыше дома", "квантовая хромодинамика"],
        model_ref=EMBED_MODEL,
    ))
    a, b, c = result.vectors
    assert cosine(a, b) > cosine(a, c)  # shared words -> higher similarity
    again = backend.embed(EmbedRequest(texts=["кот сидит на крыше"], model_ref=EMBED_MODEL))
    assert again.vectors[0] == a  # deterministic


@pytest.fixture
async def embed_net(genesis, alice, carol, tmp_path):
    transport = MultiTransport()
    cfg = NodeConfig(port=9801, endpoint=URL, backend="mock",
                     models=[CHAT_MODEL, EMBED_MODEL], max_parallel_jobs=4,
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params,
                            client=client, memory_path=str(tmp_path / "mem.jsonl"))
    transport.add(GW_URL, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "gateway": gateway, "daemon": daemon}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_openai_embeddings_endpoint(embed_net):
    client = embed_net["client"]
    resp = await client.post(f"{GW_URL}/v1/embeddings", json={
        "input": ["hello network", "hello world"], "model": "auto",
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model"] == EMBED_MODEL
    assert len(data["data"]) == 2
    assert data["data"][0]["index"] == 0 and len(data["data"][0]["embedding"]) == 64
    assert data["usage"]["total_tokens"] > 0
    assert data["deltav"]["receipt_tx"]  # embeddings are paid work too

    # the embed receipt lands on-chain and gets spot-checked
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        receipts = (await client.get(f"{URL}/chain/receipts")).json()["receipts"]
        if receipts:
            break
        await asyncio.sleep(0.05)
    assert receipts and receipts[0]["tokens"] > 0


async def test_embed_price_guard(embed_net, genesis, carol):
    """A node whose asking price exceeds the signed cap refuses with 402."""
    client = embed_net["client"]
    daemon = embed_net["daemon"]
    daemon.cfg.price_per_token = 10_000_000
    try:
        resp = await client.post(f"{URL}/embed", json={
            "texts": ["x"], "model": EMBED_MODEL,
            "requester": carol.address, "requester_pubkey": carol.public_hex,
            "requester_sig": "00", "price_limit": 100,
        })
        assert resp.status_code == 402
    finally:
        daemon.cfg.price_per_token = 0


# ------------------------------------------------------------- vector RAG

async def test_vector_memory_uses_network_embeddings(embed_net):
    gateway = embed_net["gateway"]
    memory = gateway.memory
    assert isinstance(memory, VectorMemory)

    await memory.aadd("s1", "пароль от роутера дома: omega42")
    await memory.aadd("s1", "рецепт борща со свеклой и капустой")
    assert all(it.get("vec") for it in memory.session_items("s1"))

    hits = await memory.asearch("s1", "какой пароль у домашнего роутера?")
    assert hits and "omega42" in hits[0]["text"]
    assert 0 < hits[0]["score"] <= 1.001  # cosine, not BM25


async def test_vector_memory_falls_back_to_bm25(tmp_path):
    async def broken_embedder(texts):
        raise RuntimeError("no embedding nodes")

    memory = VectorMemory(tmp_path / "m.jsonl", embedder=broken_embedder)
    await memory.aadd("s1", "код от сейфа 5150")
    hits = await memory.asearch("s1", "код сейфа")
    assert hits and "5150" in hits[0]["text"]  # BM25 saved the day
