"""The public portal: aggregation endpoints, /metrics, HTML, and the
gateway making itself externally reachable through a relay.

Portal surfaces expose only public chain/infra data — never user content.
"""
from __future__ import annotations

import httpx
import pytest
from fastapi import FastAPI

from deltav.config import ChainParams, DVT
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.net.relay import RelayServer

from conftest import MultiTransport

CHAIN = "deltav-portal-test"


def health_stub() -> FastAPI:
    """Answers /health on any path — stands in for a node reachable at its
    own host (MultiTransport routes by host, ignoring the /via/... path)."""
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET"])
    async def any_health(path: str) -> dict:
        return {"height": 1234, "load": 0.2, "peers": 2, "backend": "llamaserver",
                "max_parallel_jobs": 1}

    return app


# Three nodes exercise every connectivity class: relay-host (public + relays),
# relayed (reached via /via/), and plain direct.
NODES = [
    {"address": "dv1relayhost00", "endpoint": "http://node",
     "hardware": {"vendor": "amd", "name": "RX 6600M", "vram_mb": 8176,
                  "backend": "llamaserver", "relay": True},
     "models": ["org/Qwen2.5-7B::q4.gguf"], "reputation": 7, "stake": 10_000 * DVT,
     "jobs_done": 42, "tokens_served": 1000, "price_per_token": 9,
     "last_seen": 5, "active": True, "jailed_until": 0},
    {"address": "dv1relayed0000", "endpoint": "http://relay/via/dv1relayed0000",
     "hardware": {"vendor": "nvidia", "name": "RTX 4070", "vram_mb": 12282,
                  "backend": "llamaserver", "relay": False},
     "models": ["org/Llama-3.2-3B::q4.gguf"], "reputation": 3, "stake": 5_000 * DVT,
     "jobs_done": 8, "tokens_served": 200, "price_per_token": 10,
     "last_seen": 4, "active": True, "jailed_until": 0},
    {"address": "dv1direct00000", "endpoint": "http://nodec",
     "hardware": {"vendor": "intel", "name": "Arc A770", "vram_mb": 16000,
                  "backend": "llamaserver", "relay": False},
     "models": ["org/Qwen2.5-7B::q4.gguf"], "reputation": 5, "stake": 5_000 * DVT,
     "jobs_done": 0, "tokens_served": 0, "price_per_token": 10,
     "last_seen": 3, "active": True, "jailed_until": 0},
]


def wire(transport) -> None:
    """Register the seed node plus health stubs for the other hosts."""
    transport.add("http://node", fake_node())
    transport.add("http://relay", health_stub())    # relayed node's live probe
    transport.add("http://nodec", health_stub())    # direct node's live probe


def fake_node() -> FastAPI:
    """A minimal seed node serving the chain surface the portal reads."""
    app = FastAPI()
    nodes = NODES

    @app.get("/chain/stats")
    async def stats() -> dict:
        return {"chain_id": CHAIN, "height": 1234, "supply": 5_000_000, "pool": 3_000_000,
                "nodes": 3, "validators": 1, "receipts": 50, "unchecked_receipts": 4}

    @app.get("/chain/nodes")
    async def chain_nodes() -> dict:
        return {"height": 1234, "nodes": nodes}

    @app.get("/health")
    async def health() -> dict:
        return {"height": 1234, "load": 0.5, "peers": 3, "backend": "llamaserver",
                "max_parallel_jobs": 2}

    @app.get("/chain/head")
    async def head() -> dict:
        return {"height": 1234}

    @app.get("/chain/headers")
    async def headers(start: int = 0, count: int = 41) -> dict:
        return {"headers": [{"height": h, "proposer": "dv1direct0000", "tx_count": 1,
                             "hash": f"hash{h:04d}"} for h in range(start, start + 3)], "height": 1234}

    @app.get("/chain/receipts")
    async def receipts() -> dict:
        return {"receipts": [
            {"receipt_hash": "r1", "node": "dv1direct0000", "model": "org/Qwen2.5-7B::q4.gguf",
             "tokens": 120, "price_paid": 1080, "height": 1200, "deterministic": True,
             "checked": True, "check_ok": True}]}

    return app


def _gateway(transport):
    gw_key = KeyPair.from_seed_hex("99" * 32)
    return GatewayDaemon(gw_key, node_urls=["http://node"],
                         params=ChainParams(chain_id=CHAIN),
                         client=httpx.AsyncClient(transport=transport)), gw_key


async def test_portal_overview_aggregates():
    transport = MultiTransport()
    wire(transport)
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    o = (await gwc.get("/portal/overview")).json()
    assert o["chain_id"] == CHAIN and o["height"] == 1234
    assert o["nodes_total"] == 3 and o["nodes_online"] == 3
    assert o["direct_nodes"] == 1 and o["relayed_nodes"] == 1
    assert o["relays"] == 1                       # the relay-host advertises relay:true
    assert o["tokens_served_total"] == 1200
    assert set(m.split("::")[0].split("/")[-1] for m in o["models_served"]) == {"Qwen2.5-7B", "Llama-3.2-3B"}
    conns = {n["address"]: n["connectivity"] for n in o["nodes"]}
    assert conns["dv1relayhost00"] == "relay-host"  # public + relay-capable
    assert conns["dv1relayed0000"] == "relay"       # reached via /via/
    assert conns["dv1direct00000"] == "direct"      # plain public
    await gwc.aclose(); await gw.close()


async def test_portal_metrics_prometheus():
    transport = MultiTransport()
    wire(transport)
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    text = (await gwc.get("/metrics")).text
    assert "deltav_chain_height 1234" in text
    assert "deltav_nodes_online 3" in text
    assert 'deltav_node_load{node="dv1relayhost00"' in text
    assert "# TYPE deltav_relays gauge" in text
    await gwc.aclose(); await gw.close()


async def test_portal_html_and_config():
    transport = MultiTransport()
    transport.add("http://node", fake_node())
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    html = (await gwc.get("/")).text
    assert "<title>Delta V" in html and 'data-tab="explorer"' in html
    # /explorer serves the same SPA (deep-links to the tab client-side).
    assert (await gwc.get("/explorer")).status_code == 200
    cfg = (await gwc.get("/portal/config")).json()
    assert cfg["chain_id"] == CHAIN and cfg["origin"].startswith("http")
    await gwc.aclose(); await gw.close()


async def test_portal_receipts_and_blocks():
    transport = MultiTransport()
    transport.add("http://node", fake_node())
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    r = (await gwc.get("/portal/receipts")).json()
    assert r["receipts"] and r["receipts"][0]["node"] == "dv1direct0000"
    b = (await gwc.get("/portal/blocks")).json()
    assert b["height"] == 1234 and len(b["headers"]) >= 1
    await gwc.aclose(); await gw.close()


async def test_gateway_makes_itself_reachable_via_relay():
    """connect=relay: the gateway attaches to a relay and its portal/API
    become reachable at a public /via URL — external access for the client
    side with zero config, through the same decentralized relay."""
    transport = MultiTransport()
    relay_app = FastAPI()
    RelayServer("http://relay", poll_timeout=0.3).mount(relay_app)

    @relay_app.get("/chain/nodes")   # discover_relay probes this first
    async def chain_nodes() -> dict:
        return {"nodes": []}

    transport.add("http://relay", relay_app)

    gw_key = KeyPair.from_seed_hex("88" * 32)
    gw = GatewayDaemon(gw_key, node_urls=["http://relay"], params=ChainParams(chain_id=CHAIN),
                       client=httpx.AsyncClient(transport=transport),
                       connect="relay", port=9000)
    await gw.start()
    try:
        assert gw.public_origin == f"http://relay/via/{gw_key.address}"
        assert gw.relay_client is not None
    finally:
        await gw.stop()
        await gw.close()
