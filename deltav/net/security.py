"""Edge hardening for a node exposed to the open internet.

The chain is already safe at the message layer — every tx, block and
receipt is signed, and payment needs the requester's signature over a
price cap. What an internet-facing node still needs is protection of the
*transport*: don't trust a peer URL until it proves it's on our chain,
don't let one IP flood gossip, and don't accept unbounded bodies.
"""
from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import time
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Endpoints that mutate shared state from untrusted callers — rate-limited.
_GUARDED_PREFIXES = ("/tx", "/p2p/", "/relay/attach", "/relay/challenge")


async def screen_url(url: str, *, allow_private: bool = False,
                     allow_ports: set[int] | None = None) -> str:
    """SSRF screen for a caller-supplied URL. Returns "" if safe to fetch,
    else a short reason.

    Always refuses loopback / link-local (incl. cloud-metadata 169.254.169.254)
    / multicast / reserved / unspecified targets. Private RFC1918 ranges are
    refused too unless `allow_private` — the node/peer reachability paths pass
    allow_private=True because the network legitimately runs on a LAN, while
    the agent web-fetch tool passes allow_private=False (it only fetches the
    public web).
    """
    try:
        u = urlparse(url)
    except ValueError:
        return "bad url"
    if u.scheme not in ("http", "https"):
        return "only http(s) urls"
    host = u.hostname
    if not host:
        return "no host"
    port = u.port or (443 if u.scheme == "https" else 80)
    if allow_ports is not None and port not in allow_ports:
        return "port not allowed"
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError, UnicodeError):
        return "dns resolution failed"
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return "unresolvable address"
        if (ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_unspecified or ip.is_reserved):
            return "blocked address"
        if not allow_private and ip.is_private:
            return "blocked private address"
    return ""


async def verify_peer(client: httpx.AsyncClient, url: str, chain_id: str) -> bool:
    """True only if `url` serves a node on our chain — call before trusting
    a peer URL learned from gossip (so junk/hostile URLs never enter the
    peer set)."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    # A gossiped URL is attacker-influenced — screen it (LAN allowed) so it
    # can't be aimed at loopback / cloud-metadata as an SSRF oracle.
    if await screen_url(url, allow_private=True):
        return False
    try:
        resp = await client.get(f"{url.rstrip('/')}/health", timeout=5.0)
        resp.raise_for_status()
    except httpx.HTTPError:
        return False
    return resp.json().get("chain_id") == chain_id


def _trusted_proxies() -> set[str]:
    return {p.strip() for p in os.environ.get("DELTAV_TRUSTED_PROXIES", "").split(",")
            if p.strip()}


def client_ip(request: Request) -> str:
    """The caller's IP for rate-limiting. Honours X-Forwarded-For only when
    the immediate peer is a configured trusted proxy — otherwise a spoofed
    header could evade or poison the per-IP buckets."""
    peer = request.client.host if request.client else "?"
    if peer in _trusted_proxies():
        xff = request.headers.get("x-forwarded-for", "")
        first = xff.split(",")[0].strip()
        if first:
            return first
    return peer


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
        # Content-Length is a fast reject; but a chunked/streamed request may
        # omit it, so also enforce a hard cap while draining the body.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_body:
            return JSONResponse({"error": "request body too large"}, status_code=413)
        if length is None and request.method in ("POST", "PUT", "PATCH"):
            body = b""
            async for chunk in request.stream():
                body += chunk
                if len(body) > max_body:
                    return JSONResponse({"error": "request body too large"},
                                        status_code=413)
            # Re-expose the buffered body to downstream handlers.
            request._body = body
        path = request.url.path
        if request.method == "POST" and any(path.startswith(p) for p in _GUARDED_PREFIXES):
            if not bucket.allow(client_ip(request)):
                return JSONResponse({"error": "rate limited"}, status_code=429)
        return await call_next(request)
