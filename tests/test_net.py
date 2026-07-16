"""External connectivity: reachability self-test, circuit relay, hardening.

All in-process over ASGITransport — no sockets, no WebSocket.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx
import pytest
from fastapi import FastAPI

from deltav.compute.base import DeviceInfo
from deltav.config import DVT, ChainParams, Genesis
from deltav.crypto import KeyPair
from deltav.net import RelayClient, check_direct, mount_reach, probe_public_ip
from deltav.net.relay import RelayServer, _attach_message
from deltav.net.security import install_guards, verify_peer
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport


@dataclass
class _StubNode:
    address: str
    chain_id: str
    client: httpx.AsyncClient


def _reach_app(node) -> FastAPI:
    app = FastAPI()
    mount_reach(app, node)
    return app


# ---------------------------------------------------------------- reachability
async def test_whoami_and_echo_roundtrip():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    node = _StubNode(address="dv1abc", chain_id="deltav-test-1", client=client)
    transport.add("http://b", _reach_app(node))

    who = (await client.get("http://b/whoami")).json()
    assert "ip" in who
    echo = (await client.get("http://b/net/echo", params={"nonce": "xyz"})).json()
    assert echo == {"nonce": "xyz", "address": "dv1abc", "chain_id": "deltav-test-1"}
    await client.aclose()


# The candidate URL is SSRF-screened inside reachcheck, so it must be a
# resolvable public IP (a fake hostname would be rejected before routing).
# getaddrinfo on a numeric literal is offline. Peer URLs posted to directly
# aren't screened, so they can stay symbolic.
_PUB_B = "http://93.184.216.34"    # example.com's IP — public, numeric, offline
_PUB_C = "http://93.184.216.35"


async def test_check_direct_true_when_reachable():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    # A performs the callback; B is the node proving it's reachable.
    a = _StubNode("dv1a", "chain-x", client)
    b = _StubNode("dv1b", "chain-x", client)
    transport.add("http://a", _reach_app(a))
    transport.add(_PUB_B, _reach_app(b))

    assert await check_direct(client, ["http://a"], _PUB_B, "chain-x") is True
    await client.aclose()


async def test_check_direct_false_when_unroutable():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    a = _StubNode("dv1a", "chain-x", client)
    transport.add("http://a", _reach_app(a))
    # public candidate that passes the SSRF screen but has no route -> NAT.
    assert await check_direct(client, ["http://a"], _PUB_C, "chain-x") is False
    await client.aclose()


async def test_check_direct_false_on_chain_mismatch():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    a = _StubNode("dv1a", "chain-x", client)
    b = _StubNode("dv1b", "chain-OTHER", client)
    transport.add("http://a", _reach_app(a))
    transport.add(_PUB_B, _reach_app(b))
    # reachable, but wrong chain -> not a valid direct peer for us.
    assert await check_direct(client, ["http://a"], _PUB_B, "chain-x") is False
    await client.aclose()


# --------------------------------------------------------------- relay handshake
def _relay_app(server: RelayServer) -> FastAPI:
    app = FastAPI()
    server.mount(app)
    return app


async def test_relay_attach_requires_valid_signature():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    server = RelayServer("http://relay", poll_timeout=0.3)
    transport.add("http://relay", _relay_app(server))
    kp = KeyPair.from_seed_hex("ab" * 32)

    nonce = (await client.get("http://relay/relay/challenge",
                              params={"node_id": kp.address})).json()["nonce"]
    # Bad signature is rejected.
    bad = await client.post("http://relay/relay/attach", json={
        "node_id": kp.address, "pubkey": kp.public_hex, "signature": "00" * 64})
    assert bad.status_code == 403
    # Correct signature over the fresh nonce attaches and yields a via URL.
    nonce = (await client.get("http://relay/relay/challenge",
                              params={"node_id": kp.address})).json()["nonce"]
    sig = kp.sign(_attach_message(kp.address, nonce))
    ok = await client.post("http://relay/relay/attach", json={
        "node_id": kp.address, "pubkey": kp.public_hex, "signature": sig})
    assert ok.status_code == 200
    assert ok.json()["via_url"] == f"http://relay/via/{kp.address}"
    assert ok.json()["token"]
    await client.aclose()


async def test_relay_attach_rejects_impersonation():
    """Signing with key B while claiming node_id A must fail."""
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    server = RelayServer("http://relay")
    transport.add("http://relay", _relay_app(server))
    victim = KeyPair.from_seed_hex("11" * 32)
    attacker = KeyPair.from_seed_hex("22" * 32)

    nonce = (await client.get("http://relay/relay/challenge",
                              params={"node_id": victim.address})).json()["nonce"]
    sig = attacker.sign(_attach_message(victim.address, nonce))
    resp = await client.post("http://relay/relay/attach", json={
        "node_id": victim.address, "pubkey": attacker.public_hex, "signature": sig})
    assert resp.status_code == 403  # pubkey doesn't derive victim's node_id
    await client.aclose()


async def test_relay_pull_requires_token():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    server = RelayServer("http://relay", poll_timeout=0.2)
    transport.add("http://relay", _relay_app(server))
    kp = KeyPair.from_seed_hex("cd" * 32)
    resp = await client.get(f"http://relay/relay/pull/{kp.address}")
    assert resp.status_code == 401
    await client.aclose()


# --------------------------------------------------------------- full tunnel
async def test_relay_tunnel_roundtrip():
    """A NAT'd origin app is reached end-to-end through the relay's /via."""
    transport = MultiTransport()
    server = RelayServer("http://relay", poll_timeout=0.3, via_timeout=5.0)
    transport.add("http://relay", _relay_app(server))

    # The origin app is NOT registered under any public URL — only reachable
    # via the relay. It replays locally inside the RelayClient.
    origin = FastAPI()

    @origin.get("/health")
    async def health() -> dict:
        return {"address": "dv1origin", "ok": True}

    @origin.post("/infer")
    async def infer(body: dict) -> dict:
        return {"echo": body.get("prompt", ""), "n": len(body.get("prompt", ""))}

    kp = KeyPair.from_seed_hex("ef" * 32)
    relay_client = RelayClient(kp, origin, "http://relay",
                               httpx.AsyncClient(transport=transport), concurrency=2)
    via = await relay_client.start()
    assert via == f"http://relay/via/{kp.address}"
    await asyncio.sleep(0.1)  # let pullers begin long-polling

    caller = httpx.AsyncClient(transport=transport)
    got = await caller.get(f"{via}/health")
    assert got.status_code == 200 and got.json() == {"address": "dv1origin", "ok": True}

    posted = await caller.post(f"{via}/infer", json={"prompt": "hello"})
    assert posted.status_code == 200 and posted.json() == {"echo": "hello", "n": 5}

    # Unknown node id -> 502.
    miss = await caller.get("http://relay/via/dv1nope/health")
    assert miss.status_code == 502

    await relay_client.stop()
    await caller.aclose()


async def test_auto_connect_falls_back_to_relay_behind_nat():
    """Full daemon: a node that can't be reached directly attaches to a
    relay and announces its /via URL as the on-chain endpoint."""
    params = ChainParams(block_time=0.05)
    relay_kp = KeyPair.from_seed_hex("a1" * 32)
    nat_kp = KeyPair.from_seed_hex("b2" * 32)
    genesis = Genesis(
        params=params,
        alloc={relay_kp.address: 100_000 * DVT, nat_kp.address: 100_000 * DVT},
        stakes={relay_kp.address: 10_000 * DVT},
    )
    transport = MultiTransport()

    relay_cfg = NodeConfig(port=9310, endpoint="http://relay", relay=True,
                           relay_public_url="http://relay", backend="mock",
                           produce=False, spot_check=False, auto_register=False,
                           device=DeviceInfo(vendor="cpu", name="relay", vram_mb=0))
    relay = NodeDaemon(relay_kp, genesis, relay_cfg,
                       client=httpx.AsyncClient(transport=transport))
    transport.add("http://relay", relay.app)

    nat_cfg = NodeConfig(port=9320, connect="auto", peers=["http://relay"],
                         backend="mock", produce=False, spot_check=False,
                         auto_register=False,
                         device=DeviceInfo(vendor="cpu", name="nat", vram_mb=0))
    nat = NodeDaemon(nat_kp, genesis, nat_cfg,
                     client=httpx.AsyncClient(transport=transport))
    # deliberately NOT added to transport under its public candidate -> unreachable

    await relay.start()
    await nat.start()
    try:
        async def announced_via():
            return nat.cfg.endpoint.startswith("http://relay/via/")
        for _ in range(200):
            if await announced_via():
                break
            await asyncio.sleep(0.05)
        assert nat.cfg.endpoint == f"http://relay/via/{nat_kp.address}"

        await asyncio.sleep(0.1)
        caller = httpx.AsyncClient(transport=transport)
        health = await caller.get(f"{nat.cfg.endpoint}/health")
        assert health.status_code == 200
        assert health.json()["address"] == nat_kp.address
        await caller.aclose()
    finally:
        await nat.stop()
        await relay.stop()
        await relay.client.aclose()


async def test_probe_public_ip_ignores_loopback():
    """A same-host seed sees us as 127.0.0.1 — never a valid cross-host
    endpoint, so probe must skip it (else connect=auto false-positives)."""
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)

    loop_app = FastAPI()

    @loop_app.get("/whoami")
    async def whoami_loop() -> dict:
        return {"ip": "127.0.0.1"}

    transport.add("http://local-seed", loop_app)
    assert await probe_public_ip(client, ["http://local-seed"]) == ""

    real_app = FastAPI()

    @real_app.get("/whoami")
    async def whoami_real() -> dict:
        return {"ip": "203.0.113.7"}

    transport.add("http://real-seed", real_app)
    assert await probe_public_ip(
        client, ["http://local-seed", "http://real-seed"]) == "203.0.113.7"
    await client.aclose()


async def test_standalone_relay_app():
    """`deltav relay` app: health + full tunnel, no chain required."""
    from deltav.net.relay import build_relay_app

    transport = MultiTransport()
    transport.add("http://relay", build_relay_app("http://relay"))
    client = httpx.AsyncClient(transport=transport)

    info = (await client.get("http://relay/health")).json()
    assert info["relay"] is True and info["origins"] == 0

    origin = FastAPI()

    @origin.get("/ping")
    async def ping() -> dict:
        return {"pong": True}

    kp = KeyPair.from_seed_hex("dd" * 32)
    rc = RelayClient(kp, origin, "http://relay",
                     httpx.AsyncClient(transport=transport), concurrency=1)
    via = await rc.start()
    await asyncio.sleep(0.1)
    got = await client.get(f"{via}/ping")
    assert got.status_code == 200 and got.json() == {"pong": True}
    await rc.stop()
    await client.aclose()


# ------------------------------------------------------------------- hardening
async def test_verify_peer_rejects_wrong_chain():
    transport = MultiTransport()
    client = httpx.AsyncClient(transport=transport)
    params = ChainParams(chain_id="chain-A", block_time=0.05)
    kp = KeyPair.from_seed_hex("77" * 32)
    genesis = Genesis(params=params, alloc={kp.address: DVT})
    # verify_peer SSRF-screens the URL, so use a resolvable public IP host.
    node_url = "http://93.184.216.34"
    node = NodeDaemon(kp, genesis, NodeConfig(endpoint=node_url, backend="mock"),
                      client=client)
    transport.add(node_url, node.app)

    assert await verify_peer(client, node_url, "chain-A") is True
    assert await verify_peer(client, node_url, "chain-B") is False
    assert await verify_peer(client, "http://ghost", "chain-A") is False  # unresolvable
    await client.aclose()


async def test_guards_rate_limit_and_body_cap():
    app = FastAPI()
    install_guards(app, max_body_mb=0.001, gossip_rate=1.0, gossip_burst=2.0)

    @app.post("/tx")
    async def tx(body: dict) -> dict:
        return {"ok": True}

    @app.post("/infer")
    async def infer(body: dict) -> dict:
        return {"ok": True}

    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://n")
    # Body cap (0.001 MB ~ 1 KB) rejects a large payload with 413.
    big = await client.post("/infer", json={"x": "z" * 5000})
    assert big.status_code == 413
    # Gossip burst of 2 then rate-limited.
    codes = [(await client.post("/tx", json={})).status_code for _ in range(5)]
    assert codes[:2] == [200, 200]
    assert 429 in codes[2:]
    await client.aclose()
