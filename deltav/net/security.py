"""Edge hardening for a node exposed to the open internet.

The chain is already safe at the message layer — every tx, block and
receipt is signed, and payment needs the requester's signature over a
price cap. What an internet-facing node still needs is protection of the
*transport*: don't trust a peer URL until it proves it's on our chain,
don't let one IP flood gossip, and don't accept unbounded bodies.
"""
from __future__ import annotations

import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Endpoints that mutate shared state from untrusted callers — rate-limited.
_GUARDED_PREFIXES = ("/tx", "/p2p/", "/relay/attach", "/relay/challenge")


async def verify_peer(client: httpx.AsyncClient, url: str, chain_id: str) -> bool:
    """True only if `url` serves a node on our chain — call before trusting
    a peer URL learned from gossip (so junk/hostile URLs never enter the
    peer set)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    try:
        resp = await client.get(f"{url.rstrip('/')}/health", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return False
    return resp.json().get("chain_id") == chain_id


class _Bucket:
    """Per-IP token bucket: `rate` requests/sec, burst `capacity`."""

    def __init__(self, rate: float, capacity: float, clock):
        self.rate = rate
        self.capacity = capacity
        self._clock = clock
        self._tokens: dict[str, tuple[float, float]] = {}  # ip -> (tokens, ts)

    def allow(self, ip: str) -> bool:
        now = self._clock()
        tokens, ts = self._tokens.get(ip, (self.capacity, now))
        tokens = min(self.capacity, tokens + (now - ts) * self.rate)
        if tokens < 1.0:
            self._tokens[ip] = (tokens, now)
            return False
        self._tokens[ip] = (tokens - 1.0, now)
        return True


def install_guards(app: FastAPI, *, max_body_mb: float = 16.0,
                   gossip_rate: float = 50.0, gossip_burst: float = 100.0,
                   clock=time.monotonic) -> None:
    """Add a body-size cap and per-IP rate limit on gossip endpoints."""
    max_body = int(max_body_mb * 1024 * 1024)
    bucket = _Bucket(gossip_rate, gossip_burst, clock)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        path = request.url.path
        if request.method == "POST" and any(path.startswith(p) for p in _GUARDED_PREFIXES):
            ip = request.client.host if request.client else "?"
            if not bucket.allow(ip):
                return JSONResponse({"error": "rate limited"}, status_code=429)
        return await call_next(request)
