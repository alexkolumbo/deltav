"""The Delta V portal: one public surface for explorer, monitoring and
onboarding, served by the gateway.

Everything here is **read-only and public** — chain-ledger and node-infra
data only (blocks, receipts, node health, pool). It never touches user
content (companion memory, chats, API keys): those stay own-identity-scoped
behind the billing layer, by policy.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..config import DVT


async def _first_json(client: httpx.AsyncClient, seeds, path: str, timeout: float = 6.0):
    """GET `path` from the first seed node that answers."""
    for seed in seeds:
        try:
            resp = await client.get(f"{seed.rstrip('/')}{path}", timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPError:
            continue
    return None


def _connectivity(endpoint: str, hardware: dict) -> str:
    if "/via/" in (endpoint or ""):
        return "relay"
    if isinstance(hardware, dict) and hardware.get("relay"):
        return "relay-host"
    return "direct"


async def gather_network(gw) -> dict:
    """Aggregate the whole network from the gateway's seed node(s), then
    probe each node's live /health. Public data only."""
    seeds = gw.node_urls
    stats = await _first_json(gw.client, seeds, "/chain/stats") or {}
    reg = await _first_json(gw.client, seeds, "/chain/nodes") or {"nodes": []}
    chain_nodes = reg.get("nodes", [])

    async def probe(node: dict) -> dict:
        endpoint = node.get("endpoint", "")
        live = {}
        if endpoint:
            try:
                resp = await gw.client.get(f"{endpoint.rstrip('/')}/health", timeout=4.0)
                resp.raise_for_status()
                live = resp.json()
            except httpx.HTTPError:
                live = {}
        hardware = node.get("hardware") or {}
        price = int(node.get("price_per_token") or gw.params.price_per_token)
        return {
            "address": node.get("address", ""),
            "endpoint": endpoint,
            "connectivity": _connectivity(endpoint, hardware),
            "vendor": hardware.get("vendor", "?"),
            "device": hardware.get("name", ""),
            "vram_mb": hardware.get("vram_mb", 0),
            "backend": hardware.get("backend", live.get("backend", "?")),
            "is_relay": bool(hardware.get("relay")),
            "models": node.get("models", []),
            "reputation": node.get("reputation", 0),
            "stake": node.get("stake", 0),
            "price_per_token": price,
            "jobs_done": node.get("jobs_done", 0),
            "tokens_served": node.get("tokens_served", 0),
            "earnings_udvt": int(node.get("tokens_served", 0)) * price,
            "active": bool(node.get("active")),
            "jailed": bool(node.get("jailed_until", 0)) and node.get("jailed_until", 0) > stats.get("height", 0),
            # live
            "alive": bool(live),
            "load": live.get("load", 0.0),
            "active_jobs": round(live.get("load", 0.0) * live.get("max_parallel_jobs", 1)),
            "height": live.get("height", 0),
            "peers": live.get("peers", 0),
            "max_parallel_jobs": live.get("max_parallel_jobs", 1),
        }

    nodes = await asyncio.gather(*(probe(n) for n in chain_nodes)) if chain_nodes else []
    online = [n for n in nodes if n["alive"]]
    relays = [n for n in nodes if n["is_relay"]]
    relayed = [n for n in nodes if n["connectivity"] == "relay"]
    served_models = sorted({m for n in online for m in n["models"]})
    return {
        "chain_id": stats.get("chain_id", ""),
        "height": stats.get("height", 0),
        "supply_udvt": stats.get("supply", 0),
        "pool_udvt": stats.get("pool", 0),
        "price_per_token": gw.params.price_per_token,
        "nodes_total": len(nodes),
        "nodes_online": len(online),
        "validators": stats.get("validators", 0),
        "relays": len(relays),
        "relayed_nodes": len(relayed),
        "direct_nodes": len([n for n in nodes if n["connectivity"] == "direct"]),
        "receipts": stats.get("receipts", 0),
        "unchecked_receipts": stats.get("unchecked_receipts", 0),
        "models_served": served_models,
        "tokens_served_total": sum(n["tokens_served"] for n in nodes),
        "jobs_done_total": sum(n["jobs_done"] for n in nodes),
        "nodes": nodes,
    }


def prometheus(overview: dict) -> str:
    """Render aggregate + per-node metrics in Prometheus text format."""
    lines: list[str] = []

    def g(name: str, value, help_: str, labels: str = ""):
        lines.append(f"# HELP deltav_{name} {help_}")
        lines.append(f"# TYPE deltav_{name} gauge")
        lines.append(f"deltav_{name}{labels} {value}")

    g("chain_height", overview["height"], "current chain height")
    g("nodes_total", overview["nodes_total"], "registered nodes")
    g("nodes_online", overview["nodes_online"], "nodes answering /health")
    g("validators", overview["validators"], "staked validators")
    g("relays", overview["relays"], "nodes serving as relays")
    g("relayed_nodes", overview["relayed_nodes"], "nodes reachable via a relay")
    g("pool_udvt", overview["pool_udvt"], "chain pool balance (udvt)")
    g("receipts_total", overview["receipts"], "inference receipts on-chain")
    g("tokens_served_total", overview["tokens_served_total"], "tokens served network-wide")
    for n in overview["nodes"]:
        lab = f'{{node="{n["address"][:16]}",vendor="{n["vendor"]}",conn="{n["connectivity"]}"}}'
        g("node_load", round(n["load"], 3), "per-node load 0..1", lab)
        g("node_tokens_served", n["tokens_served"], "per-node tokens served", lab)
        g("node_reputation", n["reputation"], "per-node reputation", lab)
        g("node_alive", int(n["alive"]), "1 if node answered /health", lab)
    return "\n".join(lines) + "\n"


def mount_portal(app: FastAPI, gw) -> None:
    here = Path(__file__).parent

    @app.get("/", response_class=HTMLResponse)
    @app.get("/portal", response_class=HTMLResponse)
    async def portal_page() -> str:
        return (here / "portal.html").read_text(encoding="utf-8")

    @app.get("/explorer", response_class=HTMLResponse)
    async def explorer_alias() -> str:
        # Same SPA — deep-links open on the Explorer tab client-side.
        return (here / "portal.html").read_text(encoding="utf-8")

    @app.get("/portal/overview")
    async def portal_overview() -> dict:
        return await gather_network(gw)

    @app.get("/portal/nodes")
    async def portal_nodes() -> dict:
        data = await gather_network(gw)
        return {"height": data["height"], "nodes": data["nodes"]}

    @app.get("/portal/receipts")
    async def portal_receipts() -> dict:
        data = await _first_json(gw.client, gw.node_urls, "/chain/receipts") or {"receipts": []}
        return {"receipts": data.get("receipts", [])[-60:][::-1]}

    @app.get("/portal/blocks")
    async def portal_blocks() -> dict:
        head = await _first_json(gw.client, gw.node_urls, "/chain/head") or {}
        h = int(head.get("height", 0))
        start = max(0, h - 40)
        data = await _first_json(gw.client, gw.node_urls, f"/chain/headers?start={start}&count=41") \
            or {"headers": []}
        return {"headers": data.get("headers", [])[::-1], "height": h}

    @app.get("/portal/config")
    async def portal_config(request: Request) -> dict:
        # What the onboarding tab needs: the public origin clients should use
        # and the install one-liners. Origin is taken from how you reached us,
        # so a relayed gateway shows its public /via URL automatically.
        origin = str(request.base_url).rstrip("/")
        return {
            "origin": gw.public_origin or origin,
            "chain_id": gw.params.chain_id,
            "repo": "https://github.com/alexkolumbo/deltav",
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics() -> str:
        return prometheus(await gather_network(gw))
