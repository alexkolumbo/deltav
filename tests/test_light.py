"""Light client: header verification, quorum, payment-charge audit."""
import asyncio

import httpx
import pytest

from deltav.chain.blockchain import Blockchain
from deltav.chain.transaction import Tx, TxType, receipt_auth_bytes
from deltav.config import DVT
from deltav.crypto import KeyPair
from deltav.light import LightClient, verify_charges, verify_header_chain
from deltav.node import NodeConfig, NodeDaemon
from deltav.compute.base import DeviceInfo

from conftest import MultiTransport
from test_chain import make_receipt_tx
from test_phase2 import keys_map, produce

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"


# --------------------------------------------------------- header chain

def test_verify_clean_header_chain(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(6):
        produce(chain, keys)
    headers = [b.to_dict() for b in chain.blocks]
    verdict = verify_header_chain(genesis, headers)
    assert verdict.ok and verdict.height == 6 and verdict.checked == 7


def test_wrong_genesis_rejected(genesis, params, alice):
    from deltav.config import Genesis
    other = Genesis(params=params, alloc={alice.address: 1 * DVT})
    chain = Blockchain(genesis)
    verdict = verify_header_chain(other, [b.to_dict() for b in chain.blocks])
    assert not verdict.ok and "genesis" in verdict.error


def test_tampered_header_signature_detected(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(4):
        produce(chain, keys)
    headers = [b.to_dict() for b in chain.blocks]
    headers[3]["signature"] = "00" * 64  # forge a block
    verdict = verify_header_chain(genesis, headers)
    assert not verdict.ok
    assert "signature" in verdict.error and verdict.height == 3


def test_broken_prev_hash_link_detected(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    for _ in range(4):
        produce(chain, keys)
    headers = [b.to_dict() for b in chain.blocks]
    del headers[2]  # splice out a block -> chain no longer links
    verdict = verify_header_chain(genesis, headers)
    assert not verdict.ok


# ------------------------------------------------------------ charge audit

def _receipt_block(node_kp, requester_kp, state, tokens=30):
    tx = make_receipt_tx(node_kp, requester_kp, state, tokens_in=10, tokens_out=20)
    return {"height": 2, "txs": [tx.to_dict()]}


def test_audit_confirms_authorized_charge(genesis, alice, bob):
    from deltav.chain.blockchain import build_genesis_state
    from test_chain import register_node
    state = build_genesis_state(genesis)
    register_node(state, bob)
    block = _receipt_block(bob, alice, state)
    audit = verify_charges([block], alice.address, alice.public_hex)
    assert len(audit.charges) == 1
    assert audit.charges[0].authorized and audit.all_authorized
    assert audit.charges[0].node == bob.address


def test_audit_flags_forged_charge(genesis, alice, bob, carol):
    """A gateway forges a receipt charging carol, but can't sign as carol."""
    from deltav.chain.blockchain import build_genesis_state
    from test_chain import register_node
    state = build_genesis_state(genesis)
    register_node(state, bob)
    # bob builds a receipt claiming carol authorized it, signing with his own key
    req_hash = "de" * 32
    forged_auth = bob.sign(receipt_auth_bytes(req_hash, bob.address, MODEL, 10_000))
    tx = Tx(type=TxType.INFERENCE_RECEIPT.value, sender=bob.address, nonce=0, payload={
        "requester": carol.address, "requester_pubkey": carol.public_hex,
        "requester_sig": forged_auth,  # signed by bob, not carol
        "request_hash": req_hash, "output_hash": "ab" * 16, "model": MODEL,
        "seed": 0, "tokens_in": 10, "tokens_out": 20, "price_limit": 10_000,
    }).sign(bob)
    audit = verify_charges([{"height": 2, "txs": [tx.to_dict()]}],
                           carol.address, carol.public_hex)
    assert len(audit.charges) == 1
    assert not audit.charges[0].authorized  # caught: sig doesn't verify
    assert not audit.all_authorized


# ------------------------------------------------------------- quorum e2e

@pytest.fixture
async def two_node_net(genesis, alice, bob):
    transport = MultiTransport()
    urls = ["http://127.0.0.1:9601", "http://127.0.0.1:9602"]
    keys = [alice, bob]
    daemons = []
    for i, kp in enumerate(keys):
        cfg = NodeConfig(port=9601 + i, endpoint=urls[i], peers=[urls[1 - i]],
                         backend="mock", models=[MODEL],
                         device=DeviceInfo(vendor="nvidia", name="t", vram_mb=12282))
        d = NodeDaemon(kp, genesis, cfg, client=httpx.AsyncClient(transport=transport))
        transport.add(urls[i], d.app)
        daemons.append(d)
    client = httpx.AsyncClient(transport=transport)
    for d in daemons:
        await d.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        heights = [(await client.get(f"{u}/chain/head")).json()["height"] for u in urls]
        if all(h >= 3 for h in heights):
            break
        await asyncio.sleep(0.05)
    try:
        yield {"urls": urls, "client": client}
    finally:
        for d in daemons:
            await d.stop()
        await client.aclose()


async def test_light_client_verifies_live_network(two_node_net, genesis):
    lc = LightClient(genesis, two_node_net["urls"], client=two_node_net["client"])
    verdict = await lc.verify_headers()
    assert verdict.ok and verdict.height >= 3

    head_hash, height, votes = await lc.quorum_head()
    assert votes == 2  # both nodes agree on the head


async def test_quorum_ignores_a_lying_node(two_node_net, genesis):
    """A third 'node' that lies about the head is outvoted."""
    real = two_node_net["urls"]

    def liar(request: httpx.Request) -> httpx.Response:
        if str(request.url).endswith("/chain/head"):
            return httpx.Response(200, json={"hash": "ff" * 32, "height": 999})
        return httpx.Response(404)

    # mix the real transport nodes with a lying one via a routing transport
    base_transport = two_node_net["client"]._transport

    class Mixed(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if "9699" in str(request.url):
                return liar(request)
            return await base_transport.handle_async_request(request)

    client = httpx.AsyncClient(transport=Mixed())
    lc = LightClient(genesis, real + ["http://127.0.0.1:9699"], client=client)
    head_hash, height, votes = await lc.quorum_head()
    assert votes == 2 and head_hash != "ff" * 32  # honest majority wins
    await client.aclose()
