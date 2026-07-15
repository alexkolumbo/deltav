"""Phase 5: state checkpoints, end-to-end token streaming, Groq backend."""
import asyncio
import json

import httpx
import pytest

from deltav.chain.blockchain import Blockchain
from deltav.compute.base import DeviceInfo, InferRequest
from deltav.compute.groq import GroqBackend
from deltav.compute.mock import MockBackend
from deltav.config import DVT, Genesis
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.router import Catalog, SmartRouter

from conftest import MultiTransport
from test_phase2 import keys_map, produce

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
GROQ_MODEL = "groq/llama-3.3-70b-versatile"
URL = "http://127.0.0.1:9601"
GW_URL = "http://127.0.0.1:9600"


# ------------------------------------------------------------ checkpoints

def test_state_at_matches_block_roots(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(40):
        produce(chain, keys)
    assert any(h >= Blockchain.SNAPSHOT_INTERVAL for h in chain._snapshots)
    for height in (0, 5, 31, 32, 37, 40):
        assert chain._state_at(height).state_root() == chain.blocks[height].state_root


def test_sibling_reorg_uses_fast_path(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(30):
        primary = chain.next_proposer(slot=0)
        backup = chain.next_proposer(slot=1)
        if primary != backup:
            break
        produce(chain, keys)
    slot0 = chain.build_block(keys[primary], [], 100.0, slot=0)
    slot1 = chain.build_block(keys[backup], [], 101.0, slot=1)
    chain.add_block(slot1)
    assert chain.replace_sibling(slot0)
    assert chain.metrics["sibling_fast"] == 1
    # the swapped-in state is exactly what full validation would produce
    assert chain.state.state_root() == slot0.state_root


def test_deep_fork_replaces_from_checkpoint(genesis, alice, bob):
    keys = keys_map(alice, bob)
    ours = Blockchain(genesis)
    for i in range(40):
        produce(ours, keys, timestamp=100.0 + i)

    theirs = Blockchain(genesis)
    for block in ours.blocks[1:36]:          # shared prefix up to height 35
        theirs.add_block(block)
    for i in range(11):                       # then a longer divergent tail
        produce(theirs, keys, timestamp=500.0 + i)
    assert theirs.height > ours.height

    assert ours.replace([b.to_dict() for b in theirs.blocks])
    assert ours.head.hash == theirs.head.hash
    assert ours.state.state_root() == theirs.state.state_root()
    # validation restarted from the checkpoint at height 32, not genesis
    assert ours.metrics["replace_base_height"] == 32


# ------------------------------------------------------------- streaming

def test_mock_infer_stream_matches_infer():
    backend = MockBackend()
    request = InferRequest(prompt="stream please", model_ref=MODEL, max_tokens=24, seed=9)
    items = list(backend.infer_stream(request))
    pieces, final = items[:-1], items[-1]
    assert len(pieces) >= 2
    assert "".join(pieces) == final.text == backend.infer(request).text


@pytest.fixture
async def stream_net(genesis, alice, carol):
    transport = MultiTransport()
    cfg = NodeConfig(port=9601, endpoint=URL, backend="mock", models=[MODEL],
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params, client=client)
    transport.add(GW_URL, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "daemon": daemon, "gateway": gateway}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_route_stream_end_to_end(stream_net, genesis, carol):
    client = stream_net["client"]
    router = SmartRouter(Catalog(), carol, client, genesis.params.price_per_token)
    await router.refresh([URL])

    events = []
    async for event in router.route_stream("hello stream", model="auto",
                                           max_tokens=32, seed=3):
        events.append(event)
    tokens = [v for k, v in events if k == "token"]
    finals = [v for k, v in events if k == "final"]
    assert len(tokens) >= 2 and len(finals) == 1
    final = finals[0]
    assert "".join(tokens) == final.text
    assert final.receipt_tx

    # the streamed job's receipt lands on-chain like any other
    deadline = asyncio.get_event_loop().time() + 8.0
    receipts = []
    while asyncio.get_event_loop().time() < deadline:
        receipts = (await client.get(f"{URL}/chain/receipts")).json()["receipts"]
        if receipts:
            break
        await asyncio.sleep(0.05)
    assert receipts and receipts[0]["node"] == final.node


async def test_gateway_stream_is_true_passthrough(stream_net):
    client = stream_net["client"]
    body = {"model": "auto", "max_tokens": 32, "seed": 4,
            "messages": [{"role": "user", "content": "stream through"}]}

    ref = (await client.post(f"{GW_URL}/v1/chat/completions", json=body, timeout=30.0)).json()
    ref_text = ref["choices"][0]["message"]["content"]

    content_chunks, final_chunk, done = [], None, False
    async with client.stream("POST", f"{GW_URL}/v1/chat/completions",
                             json={**body, "stream": True}, timeout=30.0) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"]
            if delta.get("content"):
                content_chunks.append(delta["content"])
            if chunk["choices"][0]["finish_reason"] == "stop":
                final_chunk = chunk

    assert done and len(content_chunks) >= 2
    assert "".join(content_chunks) == ref_text
    assert final_chunk["usage"]["completion_tokens"] > 0
    assert final_chunk["deltav"]["receipt_tx"]  # the paid trail survives streaming


# ------------------------------------------------------------------ groq

def test_groq_backend_relays_to_lpu_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "llama-3.3-70b-versatile"  # groq/ prefix stripped
        assert payload["messages"] == [{"role": "user", "content": "hi lpu"}]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "groq says hi"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        })

    backend = GroqBackend(api_key="test-key", base_url="http://groq.test/v1",
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = backend.infer(InferRequest(prompt="hi lpu", model_ref=GROQ_MODEL, max_tokens=64))
    assert result.text == "groq says hi"
    assert result.tokens_in == 7 and result.tokens_out == 3
    assert result.deterministic is False  # fuzzy spot checks apply


def test_groq_available_only_with_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert not GroqBackend.is_available()
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert GroqBackend.is_available()


async def test_router_synthesizes_spec_for_announced_api_model(carol):
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/chain/nodes"):
            return httpx.Response(200, json={"height": 5, "nodes": [{
                "address": "dv1groqrelay", "endpoint": "http://relay:1",
                "models": [GROQ_MODEL], "hardware": {"vendor": "groq", "vram_mb": 0},
                "reputation": 0.7, "stake": 0, "last_seen": 5, "active": True,
            }]})
        if url.endswith("/health"):
            return httpx.Response(200, json={"load": 0.0})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    router = SmartRouter(Catalog(), carol, client, price_per_token=10)
    await router.refresh(["http://chain:1"])

    spec = router.resolve_model(GROQ_MODEL)
    assert spec.ref == GROQ_MODEL and spec.file_mb == 0
    assert [n.address for n in router.rank_nodes(spec)] == ["dv1groqrelay"]

    # a vram-0 relay must never be asked to cold-load a GPU model
    gpu_spec = Catalog().by_ref("Qwen/Qwen2.5-7B-Instruct-GGUF")
    assert not router._servable(gpu_spec, router.nodes[0])
