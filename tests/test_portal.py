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

    @app.get("/chain/blocks")
    async def blocks(start: int = 0, count: int = 30) -> dict:
        out = []
        for h in range(start, min(1234, start + count - 1) + 1):
            txs = []
            if h == 1200:
                txs = [{"type": "inference_receipt", "sender": "dv1relayhost00",
                        "hash": "aa" * 16,
                        "payload": {"model": "org/Qwen2.5-7B::q4.gguf",
                                    "tokens_in": 20, "tokens_out": 100}}]
            elif h == 1210:
                txs = [{"type": "transfer", "sender": "dv1direct00000", "hash": "bb" * 16,
                        "payload": {"to": "dv1relayed0000", "amount": 5 * DVT}}]
            elif h == 1220:
                txs = [{"type": "spot_check", "sender": "dv1relayhost00", "hash": "cc" * 16,
                        "payload": {"receipt_hash": "e1" * 16, "ok": True}}]
            out.append({"height": h, "timestamp": 1_700_000_000.0 + h,
                        "proposer": "dv1relayhost00", "slot": 0,
                        "hash": f"{'0' * 8}{h:056d}", "txs": txs})
        return {"blocks": out}

    @app.get("/chain/receipts")
    async def receipts() -> dict:
        # Two models, mixed check verdicts, spread across heights → feeds
        # the series buckets, top-models and per-node quality stats.
        rows = []
        for i in range(10):
            rows.append({"receipt_hash": f"e{i}{'0' * 30}", "node": "dv1relayhost00",
                         "model": "org/Qwen2.5-7B::q4.gguf", "tokens": 100 + i,
                         "price_paid": 900 + i, "height": 100 + i * 100,
                         "deterministic": True, "checked": i % 2 == 0,
                         "check_ok": i % 4 != 2})
        rows.append({"receipt_hash": "f" * 32, "node": "dv1relayed0000",
                     "model": "org/Llama-3.2-3B::q4.gguf", "tokens": 50,
                     "price_paid": 500, "height": 1230, "deterministic": True,
                     "checked": True, "check_ok": False})
        return {"receipts": rows}

    @app.get("/chain/account/{address}")
    async def account(address: str) -> dict:
        return {"address": address, "balance": 42 * DVT, "nonce": 7, "stake": 10_000 * DVT}

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
    assert r["receipts"] and r["receipts"][0]["node"] == "dv1relayed0000"  # newest first
    b = (await gwc.get("/portal/blocks?limit=10")).json()
    assert b["height"] == 1234 and len(b["headers"]) == 10
    assert b["headers"][0]["height"] == 1234           # newest first
    assert b["headers"][0]["timestamp"] > 1_700_000_000
    one = (await gwc.get("/portal/block/1200")).json()["block"]
    assert one["height"] == 1200 and one["txs"][0]["type"] == "inference_receipt"
    await gwc.aclose(); await gw.close()


async def test_portal_series_models_txs():
    """The analytics endpoints powering charts, top-models and the tx feed."""
    transport = MultiTransport()
    wire(transport)
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    s = (await gwc.get("/portal/series?buckets=10")).json()
    assert len(s["series"]) == 10 and s["height"] == 1234
    assert sum(r["requests"] for r in s["series"]) == 11        # all receipts bucketed
    assert sum(r["fail"] for r in s["series"]) == 3             # i=2,6 + the llama fail
    assert sum(r["tokens"] for r in s["series"]) == sum(100 + i for i in range(10)) + 50

    m = (await gwc.get("/portal/models")).json()["models"]
    assert m[0]["name"] == "Qwen2.5-7B" and m[0]["requests"] == 10
    assert m[0]["share_pct"] > 80 and m[0]["nodes_serving"] >= 1
    assert m[1]["name"] == "Llama-3.2-3B" and m[1]["requests"] == 1

    x = (await gwc.get("/portal/txs?limit=10")).json()["txs"]
    types = {tx["type"] for tx in x}
    assert {"inference_receipt", "transfer", "spot_check"} <= types
    tr = next(tx for tx in x if tx["type"] == "transfer")
    assert tr["to"] == "dv1relayed0000" and tr["amount_udvt"] == 5 * DVT
    assert x[0]["height"] >= x[-1]["height"]                    # newest first
    await gwc.aclose(); await gw.close()


async def test_portal_search_and_address():
    transport = MultiTransport()
    wire(transport)
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    # numeric → block
    r = (await gwc.get("/portal/search?q=1200")).json()
    assert r["type"] == "block" and r["block"]["height"] == 1200
    # dv1 address → account + node + its receipts
    r = (await gwc.get("/portal/search?q=dv1relayhost00")).json()
    assert r["type"] == "address"
    assert r["result"]["account"]["balance"] == 42 * DVT
    assert r["result"]["node"]["address"] == "dv1relayhost00"
    assert r["result"]["receipts"]
    # receipt-hash prefix → receipt
    r = (await gwc.get("/portal/search?q=" + "f" * 12)).json()
    assert r["type"] == "receipt" and r["receipt"]["node"] == "dv1relayed0000"
    # model-name substring → models
    r = (await gwc.get("/portal/search?q=qwen")).json()
    assert r["type"] == "models" and r["models"][0]["requests"] == 10
    # garbage → not_found
    r = (await gwc.get("/portal/search?q=zzzz-nope")).json()
    assert r["type"] == "not_found"
    await gwc.aclose(); await gw.close()


async def test_portal_overview_epoch_and_quality():
    transport = MultiTransport()
    wire(transport)
    gw, _ = _gateway(transport)
    gwc = httpx.AsyncClient(transport=httpx.ASGITransport(app=gw.app), base_url="http://gw")

    o = (await gwc.get("/portal/overview")).json()
    assert o["epoch"] == 1234 // o["epoch_blocks"]
    assert o["epoch_next_in_s"] > 0
    host = next(n for n in o["nodes"] if n["address"] == "dv1relayhost00")
    # receipts: i even → checked; i%4==2 (i=2,6) → check_ok False
    assert host["checks_ok"] == 3 and host["checks_fail"] == 2
    assert host["checks_pending"] == 5
    assert host["fail_pct"] == 40.0
    assert host["weight_pct"] > 0 and sum(n["weight_pct"] for n in o["nodes"]) > 99
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
