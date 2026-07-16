"""The Delta V portal: one public surface for explorer, monitoring and
onboarding, served by the gateway.

Everything here is **read-only and public** — chain-ledger and node-infra
data only (blocks, receipts, node health, pool). It never touches user
content (companion memory, chats, API keys): those stay own-identity-scoped
behind the billing layer, by policy.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..config import DVT

# Address shape — validated before interpolating into an upstream node URL.
_ADDR_RE = re.compile(r"^dv1[0-9a-z]{6,64}$")


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


def _model_short(ref: str) -> str:
    return (ref or "").split("::")[0].split("/")[-1]


class PortalData:
    """Chain-wide aggregation with a short TTL cache, so a busy portal
    doesn't hammer the seed nodes with repeated full-receipt pulls."""

    def __init__(self, gw, ttl: float = 4.0):
        self.gw = gw
        self.ttl = ttl
        self._cache: dict[str, tuple[float, object]] = {}

    async def _cached(self, key: str, coro_fn):
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < self.ttl:
            return hit[1]
        value = await coro_fn()
        self._cache[key] = (now, value)
        return value

    async def stats(self) -> dict:
        return await self._cached("stats", lambda: self._stats())

    async def _stats(self) -> dict:
        return await _first_json(self.gw.client, self.gw.node_urls, "/chain/stats") or {}

    async def receipts(self) -> list[dict]:
        return await self._cached("receipts", lambda: self._receipts())

    async def _receipts(self) -> list[dict]:
        data = await _first_json(self.gw.client, self.gw.node_urls, "/chain/receipts",
                                 timeout=15.0) or {"receipts": []}
        return data.get("receipts", [])

    async def chain_nodes(self) -> list[dict]:
        return await self._cached("nodes", lambda: self._chain_nodes())

    async def _chain_nodes(self) -> list[dict]:
        data = await _first_json(self.gw.client, self.gw.node_urls, "/chain/nodes") or {}
        return data.get("nodes", [])

    async def recent_blocks(self, count: int = 30) -> list[dict]:
        return await self._cached(f"blocks{count}", lambda: self._recent_blocks(count))

    async def _recent_blocks(self, count: int) -> list[dict]:
        head = await _first_json(self.gw.client, self.gw.node_urls, "/chain/head") or {}
        h = int(head.get("height", 0))
        start = max(0, h - count + 1)
        data = await _first_json(
            self.gw.client, self.gw.node_urls,
            f"/chain/blocks?start={start}&count={count}", timeout=15.0) or {"blocks": []}
        return data.get("blocks", [])


# ------------------------------------------------------------------ overview
async def gather_network(gw) -> dict:
    """Aggregate the whole network. Public data only."""
    data: PortalData = gw.portal_data
    stats, chain_nodes, receipts = await asyncio.gather(
        data.stats(), data.chain_nodes(), data.receipts())
    height = int(stats.get("height", 0))
    params = gw.params

    # Per-node spot-check quality from the receipt ledger.
    per_node: dict[str, dict] = {}
    for r in receipts:
        acc = per_node.setdefault(r.get("node", ""), {"ok": 0, "fail": 0, "unchecked": 0})
        if not r.get("checked"):
            acc["unchecked"] += 1
        elif r.get("check_ok"):
            acc["ok"] += 1
        else:
            acc["fail"] += 1

    total_stake = sum(int(n.get("stake") or 0) for n in chain_nodes) or 1

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
        price = int(node.get("price_per_token") or params.price_per_token)
        checks = per_node.get(node.get("address", ""), {"ok": 0, "fail": 0, "unchecked": 0})
        judged = checks["ok"] + checks["fail"]
        last_seen = int(node.get("last_seen") or 0)
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
            "stake": int(node.get("stake") or 0),
            "weight_pct": round(100.0 * int(node.get("stake") or 0) / total_stake, 2),
            "price_per_token": price,
            "jobs_done": node.get("jobs_done", 0),
            "tokens_served": node.get("tokens_served", 0),
            "earnings_udvt": int(node.get("tokens_served", 0)) * price,
            "checks_ok": checks["ok"],
            "checks_fail": checks["fail"],
            "checks_pending": checks["unchecked"],
            "fail_pct": round(100.0 * checks["fail"] / judged, 2) if judged else 0.0,
            "active": bool(node.get("active")),
            "jailed": bool(node.get("jailed_until", 0)) and node.get("jailed_until", 0) > height,
            "last_seen_height": last_seen,
            "last_seen_blocks_ago": max(0, height - last_seen),
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
    served_models = sorted({m for n in online for m in n["models"]})
    epoch_blocks = max(1, getattr(params, "epoch_blocks", 600))
    into_epoch = height % epoch_blocks
    return {
        "chain_id": stats.get("chain_id", ""),
        "height": height,
        "supply_udvt": stats.get("supply", 0),
        "pool_udvt": stats.get("pool", 0),
        "price_per_token": params.price_per_token,
        "block_time": params.block_time,
        "epoch": height // epoch_blocks,
        "epoch_blocks": epoch_blocks,
        "epoch_progress": into_epoch,
        "epoch_next_in_s": int((epoch_blocks - into_epoch) * params.block_time),
        "nodes_total": len(nodes),
        "nodes_online": len(online),
        "validators": stats.get("validators", 0),
        "relays": len([n for n in nodes if n["is_relay"]]),
        "relayed_nodes": len([n for n in nodes if n["connectivity"] == "relay"]),
        "direct_nodes": len([n for n in nodes if n["connectivity"] == "direct"]),
        "receipts": stats.get("receipts", 0),
        "unchecked_receipts": stats.get("unchecked_receipts", 0),
        "models_served": served_models,
        "tokens_served_total": sum(n["tokens_served"] for n in nodes),
        "jobs_done_total": sum(n["jobs_done"] for n in nodes),
        "total_stake": total_stake,
        "nodes": nodes,
    }


# ------------------------------------------------------------------- series
def bucket_series(receipts: list[dict], height: int, buckets: int = 30) -> dict:
    """Bucket the receipt ledger by height ranges → time-series for charts:
    AI tokens, request counts, validated vs failed spot-checks, DVT paid."""
    height = max(1, height)
    size = max(1, (height + buckets - 1) // buckets)
    rows = [{"from": i * size, "to": min(height, (i + 1) * size - 1),
             "requests": 0, "tokens": 0, "paid_udvt": 0, "ok": 0, "fail": 0}
            for i in range(buckets)]
    for r in receipts:
        i = min(buckets - 1, int(r.get("height", 0)) // size)
        row = rows[i]
        row["requests"] += 1
        row["tokens"] += int(r.get("tokens", 0))
        row["paid_udvt"] += int(r.get("price_paid", 0))
        if r.get("checked"):
            row["ok" if r.get("check_ok") else "fail"] += 1
    return {"bucket_blocks": size, "height": height, "series": rows}


def top_models(receipts: list[dict], nodes: list[dict]) -> list[dict]:
    """Aggregate the receipt ledger by model — the gonka-style 'top models'."""
    serving: dict[str, int] = {}
    for n in nodes:
        if not n.get("alive"):
            continue
        for m in n.get("models", []):
            serving[m] = serving.get(m, 0) + 1
    agg: dict[str, dict] = {}
    for r in receipts:
        ref = r.get("model", "")
        a = agg.setdefault(ref, {"model": ref, "name": _model_short(ref),
                                 "requests": 0, "tokens": 0, "paid_udvt": 0})
        a["requests"] += 1
        a["tokens"] += int(r.get("tokens", 0))
        a["paid_udvt"] += int(r.get("price_paid", 0))
    total_req = sum(a["requests"] for a in agg.values()) or 1
    out = sorted(agg.values(), key=lambda a: -a["requests"])
    for a in out:
        a["share_pct"] = round(100.0 * a["requests"] / total_req, 1)
        a["nodes_serving"] = serving.get(a["model"], 0)
    return out


def flatten_txs(blocks: list[dict], limit: int = 40) -> list[dict]:
    """Newest-first typed transaction feed from full blocks."""
    out: list[dict] = []
    for b in reversed(blocks):
        for tx in reversed(b.get("txs", [])):
            payload = tx.get("payload", {})
            row = {"type": tx.get("type", "?"), "sender": tx.get("sender", ""),
                   "hash": tx.get("hash", ""), "height": b.get("height", 0),
                   "timestamp": b.get("timestamp", 0)}
            if row["type"] == "transfer":
                row["to"] = payload.get("to", "")
                row["amount_udvt"] = int(payload.get("amount", 0))
            elif row["type"] == "inference_receipt":
                row["model"] = _model_short(payload.get("model", ""))
                row["tokens"] = int(payload.get("tokens_in", 0)) + int(payload.get("tokens_out", 0))
            elif row["type"] == "spot_check":
                row["ok"] = bool(payload.get("ok"))
            elif row["type"] == "stake":
                row["amount_udvt"] = int(payload.get("amount", 0))
            out.append(row)
            if len(out) >= limit:
                return out
    return out


# ------------------------------------------------------------------ metrics
def prometheus(overview: dict) -> str:
    """Render aggregate + per-node metrics in Prometheus text format."""
    lines: list[str] = []

    def g(name: str, value, help_: str, labels: str = ""):
        lines.append(f"# HELP deltav_{name} {help_}")
        lines.append(f"# TYPE deltav_{name} gauge")
        lines.append(f"deltav_{name}{labels} {value}")

    g("chain_height", overview["height"], "current chain height")
    g("epoch", overview["epoch"], "current epoch")
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
        g("node_checks_failed", n["checks_fail"], "per-node failed spot-checks", lab)
        g("node_alive", int(n["alive"]), "1 if node answered /health", lab)
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- mount
def mount_portal(app: FastAPI, gw) -> None:
    here = Path(__file__).parent
    gw.portal_data = PortalData(gw)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/portal", response_class=HTMLResponse)
    @app.get("/explorer", response_class=HTMLResponse)
    async def portal_page() -> str:
        # One SPA; deep-links (#explorer, #nodes, …) open the tab client-side.
        return (here / "portal.html").read_text(encoding="utf-8")

    @app.get("/portal/overview")
    async def portal_overview() -> dict:
        return await gather_network(gw)

    @app.get("/portal/series")
    async def portal_series(buckets: int = 30) -> dict:
        data: PortalData = gw.portal_data
        stats, receipts = await asyncio.gather(data.stats(), data.receipts())
        return bucket_series(receipts, int(stats.get("height", 0)), max(4, min(120, buckets)))

    @app.get("/portal/models")
    async def portal_models() -> dict:
        o = await gather_network(gw)
        receipts = await gw.portal_data.receipts()
        return {"models": top_models(receipts, o["nodes"])}

    @app.get("/portal/txs")
    async def portal_txs(limit: int = 40) -> dict:
        blocks = await gw.portal_data.recent_blocks(60)
        return {"txs": flatten_txs(blocks, max(1, min(200, limit)))}

    @app.get("/portal/receipts")
    async def portal_receipts(limit: int = 60) -> dict:
        receipts = await gw.portal_data.receipts()
        return {"receipts": receipts[-max(1, min(500, limit)):][::-1]}

    @app.get("/portal/blocks")
    async def portal_blocks(limit: int = 30) -> dict:
        blocks = await gw.portal_data.recent_blocks(max(1, min(200, limit)))
        out = [{"height": b.get("height"), "timestamp": b.get("timestamp"),
                "proposer": b.get("proposer"), "slot": b.get("slot", 0),
                "txs": len(b.get("txs", [])), "hash": b.get("hash", "")}
               for b in reversed(blocks)]
        return {"headers": out, "height": out[0]["height"] if out else 0}

    @app.get("/portal/block/{height}")
    async def portal_block(height: int) -> dict:
        data = await _first_json(gw.client, gw.node_urls,
                                 f"/chain/blocks?start={height}&count=1") or {"blocks": []}
        blocks = data.get("blocks", [])
        return {"block": blocks[0] if blocks else None}

    @app.get("/portal/address/{address}")
    async def portal_address(address: str) -> dict:
        if not _ADDR_RE.match(address):
            raise HTTPException(400, "invalid address")
        acc = await _first_json(gw.client, gw.node_urls, f"/chain/account/{address}") or {}
        nodes = await gw.portal_data.chain_nodes()
        node = next((n for n in nodes if n.get("address") == address), None)
        receipts = [r for r in await gw.portal_data.receipts()
                    if r.get("node") == address][-25:][::-1]
        return {"account": acc, "node": node, "receipts": receipts}

    @app.get("/portal/search")
    async def portal_search(q: str = "") -> dict:
        q = q.strip()
        if not q:
            return {"type": "none"}
        if q.isdigit():
            data = await _first_json(gw.client, gw.node_urls,
                                     f"/chain/blocks?start={int(q)}&count=1") or {"blocks": []}
            blocks = data.get("blocks", [])
            return {"type": "block", "block": blocks[0] if blocks else None}
        if q.startswith("dv1"):
            return {"type": "address", "result": await portal_address(q)}
        low = q.lower()
        if len(low) >= 12 and all(c in "0123456789abcdef" for c in low):
            for r in await gw.portal_data.receipts():
                if r.get("receipt_hash", "").startswith(low):
                    return {"type": "receipt", "receipt": r}
            for b in await gw.portal_data.recent_blocks(200):
                if b.get("hash", "").startswith(low):
                    return {"type": "block", "block": b}
            return {"type": "not_found"}
        receipts = await gw.portal_data.receipts()
        models = [m for m in top_models(receipts, []) if low in m["model"].lower()]
        return {"type": "models", "models": models} if models else {"type": "not_found"}

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
