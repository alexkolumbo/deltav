"""Self-endpoint discovery + reachability self-test (STUN-equivalent).

A node learns its own public address from any peer and verifies it is
directly reachable by having a peer call it back — with no third-party
STUN/echo service, using only the node's own HTTP surface.
"""
from __future__ import annotations

import ipaddress
import secrets
from dataclasses import dataclass

import httpx
from fastapi import FastAPI, Request


def mount_reach(app: FastAPI, node) -> None:
    """Register the reachability endpoints on a node's app.

    `node` must expose `.address`, `.chain_id` and `.client` (httpx).
    """

    @app.get("/whoami")
    async def whoami(request: Request) -> dict:
        # The source IP we saw you connect from — a node behind NAT learns
        # its own public address this way, no external STUN server needed.
        return {"ip": request.client.host if request.client else ""}

    @app.get("/net/echo")
    async def net_echo(nonce: str = "") -> dict:
        # Cheap liveness beacon used by reachcheck: proves both identity
        # (address) and that this is the right chain (chain_id).
        return {"nonce": nonce, "address": node.address, "chain_id": node.chain_id}

    @app.post("/net/reachcheck")
    async def net_reachcheck(body: dict) -> dict:
        # A peer asks us to prove *they* are reachable: we call their
        # /net/echo from the outside and confirm the nonce round-trips.
        # The url is attacker-supplied, so screen it (LAN allowed, but not
        # loopback / cloud-metadata) and never echo the raw transport error —
        # otherwise this is an SSRF port-scan oracle for the node's own host.
        from .security import screen_url

        url = str(body.get("url", "")).rstrip("/")
        nonce = str(body.get("nonce", ""))
        if await screen_url(url, allow_private=True):
            return {"reachable": False, "reason": "unreachable"}
        try:
            resp = await node.client.get(f"{url}/net/echo", params={"nonce": nonce}, timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError:
            return {"reachable": False, "reason": "unreachable"}
        ok = data.get("nonce") == nonce and data.get("chain_id") == node.chain_id
        return {"reachable": bool(ok), "address": data.get("address")}


@dataclass
class ReachResult:
    mode: str            # "direct" | "relay"
    endpoint: str = ""   # public URL when directly reachable
    public_ip: str = ""


async def probe_public_ip(client: httpx.AsyncClient, peers) -> str:
    """Ask peers what public IP they see us coming from (first answer wins).

    Loopback answers are ignored: a same-host peer sees us as 127.0.0.1,
    which can never be a valid cross-host endpoint — without this filter
    `connect=auto` behind a localhost seed would false-positive as
    "directly reachable" at 127.0.0.1.
    """
    for peer in peers:
        try:
            resp = await client.get(f"{peer.rstrip('/')}/whoami", timeout=5.0)
            resp.raise_for_status()
            ip = str(resp.json().get("ip", "")).strip()
        except httpx.HTTPError:
            continue
        if not ip:
            continue
        try:
            if ipaddress.ip_address(ip).is_loopback:
                continue
        except ValueError:
            continue
        return ip
    return ""


async def check_direct(client: httpx.AsyncClient, peers, candidate: str, chain_id: str) -> bool:
    """True if at least one peer can reach us at `candidate` from outside.

    Uses a fresh nonce per attempt so a stale/cached echo can't spoof
    reachability.
    """
    nonce = secrets.token_hex(16)
    for peer in peers:
        try:
            resp = await client.post(
                f"{peer.rstrip('/')}/net/reachcheck",
                json={"url": candidate, "nonce": nonce},
                timeout=8.0,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            continue
        data = resp.json()
        if data.get("reachable") and data.get("address"):
            return True
    return False
