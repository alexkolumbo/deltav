"""Decentralized circuit relay — reach a NAT'd node with no inbound port.

Any node with a public address can advertise `relay: true`. A node behind
NAT opens an *outbound* HTTP long-poll to such a relay, proves it owns its
chain identity, and is handed a public URL `https://<relay>/via/<node-id>`.
Other nodes then talk to it through that URL; the relay forwards each
request down the long-poll and streams the reply back.

Pure HTTP (no WebSocket) so it survives restrictive proxies, and testable
in-process. Ownership is signed, so no one can attach under another node's
id or hijack its public URL.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import secrets
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response

from ..crypto import KeyPair, address_from_public, verify_signature

log = logging.getLogger("deltav.relay")

# Request headers we never forward through the tunnel (recomputed per hop).
_HOP_HEADERS = {"host", "connection", "content-length", "keep-alive",
                "transfer-encoding", "upgrade", "accept-encoding"}


def _attach_message(node_id: str, nonce: str) -> bytes:
    return f"deltav-relay-attach:{node_id}:{nonce}".encode()


def _filter_headers(items) -> dict:
    return {k: v for k, v in items if k.lower() not in _HOP_HEADERS}


# --------------------------------------------------------------- relay server
@dataclass
class _Origin:
    token: str
    queue: "asyncio.Queue" = field(default_factory=asyncio.Queue)
    pending: dict = field(default_factory=dict)   # req_id -> asyncio.Future
    last_seen: float = 0.0


class RelayServer:
    """Mountable on any public node to relay for NAT'd peers.

    poll_timeout: how long a /relay/pull long-poll blocks before returning
    204 (the origin immediately re-polls). via_timeout: how long an inbound
    /via request waits for the origin to answer before 504.
    """

    def __init__(self, public_url: str, *, poll_timeout: float = 20.0,
                 via_timeout: float = 30.0, max_origins: int = 256,
                 clock=time.monotonic):
        self.public_url = public_url.rstrip("/")
        self.poll_timeout = poll_timeout
        self.via_timeout = via_timeout
        self.max_origins = max_origins
        self._clock = clock
        self._origins: dict[str, _Origin] = {}
        self._challenges: dict[str, tuple[str, float]] = {}  # node_id -> (nonce, expiry)

    @property
    def origin_count(self) -> int:
        return len(self._origins)

    def _auth(self, node_id: str, authorization: str | None) -> _Origin:
        origin = self._origins.get(node_id)
        token = (authorization or "").removeprefix("Bearer ").strip()
        if origin is None or not token or not secrets.compare_digest(token, origin.token):
            raise HTTPException(401, "not attached")
        origin.last_seen = self._clock()
        return origin

    def mount(self, app: FastAPI) -> None:
        @app.get("/relay/info")
        async def relay_info() -> dict:
            return {"relay": True, "public_url": self.public_url,
                    "origins": self.origin_count, "capacity": self.max_origins}

        @app.get("/relay/challenge")
        async def relay_challenge(node_id: str) -> dict:
            # A one-time nonce the node signs to prove it owns node_id.
            nonce = secrets.token_hex(16)
            self._challenges[node_id] = (nonce, self._clock() + 60.0)
            return {"nonce": nonce}

        @app.post("/relay/attach")
        async def relay_attach(body: dict) -> dict:
            node_id = str(body.get("node_id", ""))
            pubkey = str(body.get("pubkey", ""))
            signature = str(body.get("signature", ""))
            entry = self._challenges.pop(node_id, None)
            if entry is None or entry[1] < self._clock():
                raise HTTPException(400, "no valid challenge — GET /relay/challenge first")
            nonce = entry[0]
            # Ownership: the signer's key must derive exactly this node_id,
            # and the signature must cover our fresh nonce.
            if address_from_public(pubkey) != node_id:
                raise HTTPException(403, "pubkey does not match node_id")
            if not verify_signature(pubkey, _attach_message(node_id, nonce), signature):
                raise HTTPException(403, "bad signature")
            if node_id not in self._origins and self.origin_count >= self.max_origins:
                raise HTTPException(503, "relay at capacity")
            token = secrets.token_hex(24)
            self._origins[node_id] = _Origin(token=token, last_seen=self._clock())
            return {"via_url": f"{self.public_url}/via/{node_id}",
                    "token": token, "poll_timeout": self.poll_timeout}

        @app.post("/relay/detach")
        async def relay_detach(node_id: str, authorization: str | None = Header(None)) -> dict:
            self._auth(node_id, authorization)
            self._origins.pop(node_id, None)
            return {"detached": True}

        @app.get("/relay/pull/{node_id}")
        async def relay_pull(node_id: str, authorization: str | None = Header(None)):
            origin = self._auth(node_id, authorization)
            try:
                job = await asyncio.wait_for(origin.queue.get(), timeout=self.poll_timeout)
            except asyncio.TimeoutError:
                return Response(status_code=204)
            return job

        @app.post("/relay/push/{node_id}")
        async def relay_push(node_id: str, body: dict,
                             authorization: str | None = Header(None)) -> dict:
            origin = self._auth(node_id, authorization)
            fut = origin.pending.get(str(body.get("req_id", "")))
            if fut is not None and not fut.done():
                fut.set_result(body)
            return {"ok": True}

        @app.api_route("/via/{node_id}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        @app.api_route("/via/{node_id}/{path:path}",
                       methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
        async def via(node_id: str, request: Request, path: str = ""):
            origin = self._origins.get(node_id)
            if origin is None:
                raise HTTPException(502, "node not attached to this relay")
            body = await request.body()
            req_id = secrets.token_hex(12)
            loop = asyncio.get_running_loop()
            fut = loop.create_future()
            origin.pending[req_id] = fut
            job = {
                "req_id": req_id,
                "method": request.method,
                "path": "/" + path,
                "query": str(request.url.query),
                "headers": _filter_headers(request.headers.items()),
                "body_b64": base64.b64encode(body).decode(),
            }
            await origin.queue.put(job)
            try:
                result = await asyncio.wait_for(fut, timeout=self.via_timeout)
            except asyncio.TimeoutError:
                raise HTTPException(504, "relayed node did not respond")
            finally:
                origin.pending.pop(req_id, None)
            content = base64.b64decode(result.get("body_b64", ""))
            headers = {k: v for k, v in (result.get("headers") or {}).items()
                       if k.lower() not in _HOP_HEADERS}
            return Response(content=content, status_code=int(result.get("status", 502)),
                            headers=headers)


# --------------------------------------------------------------- relay client
class RelayClient:
    """Runs inside a NAT'd node: keeps an outbound tunnel to a relay open
    and replays each inbound request against the node's own app in-process.
    """

    def __init__(self, keypair: KeyPair, local_app: FastAPI,
                 relay_url: str, client: httpx.AsyncClient, *, concurrency: int = 4):
        self.keypair = keypair
        self.local_app = local_app
        self.relay_url = relay_url.rstrip("/")
        self.client = client
        self.concurrency = concurrency
        self.via_url = ""
        self.token = ""
        self.poll_timeout = 20.0
        self._local = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=local_app), base_url="http://node.local")
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def attach(self) -> str:
        node_id = self.keypair.address
        resp = await self.client.get(
            f"{self.relay_url}/relay/challenge", params={"node_id": node_id}, timeout=10.0)
        resp.raise_for_status()
        nonce = resp.json()["nonce"]
        signature = self.keypair.sign(_attach_message(node_id, nonce))
        resp = await self.client.post(f"{self.relay_url}/relay/attach", json={
            "node_id": node_id, "pubkey": self.keypair.public_hex, "signature": signature,
        }, timeout=10.0)
        resp.raise_for_status()
        data = resp.json()
        self.via_url = data["via_url"]
        self.token = data["token"]
        self.poll_timeout = float(data.get("poll_timeout", 20.0))
        log.info("attached to relay %s as %s", self.relay_url, self.via_url)
        return self.via_url

    async def _handle(self, job: dict) -> None:
        # Replay the tunneled request against our own app, in-process.
        body = base64.b64decode(job.get("body_b64", ""))
        url = job.get("path", "/")
        if job.get("query"):
            url = f"{url}?{job['query']}"
        try:
            resp = await self._local.request(
                job.get("method", "GET"), url,
                headers=job.get("headers") or {}, content=body, timeout=self.poll_timeout + 30.0)
            out = {"req_id": job["req_id"], "status": resp.status_code,
                   "headers": dict(resp.headers), "body_b64": base64.b64encode(resp.content).decode()}
        except Exception as exc:  # never leave the caller hanging on a 504
            out = {"req_id": job["req_id"], "status": 502,
                   "headers": {"content-type": "application/json"},
                   "body_b64": base64.b64encode(
                       f'{{"error":"relay replay failed: {exc}"}}'.encode()).decode()}
        try:
            await self.client.post(
                f"{self.relay_url}/relay/push/{self.keypair.address}",
                json=out, headers={"Authorization": f"Bearer {self.token}"}, timeout=15.0)
        except httpx.HTTPError:
            pass

    async def _puller(self) -> None:
        node_id = self.keypair.address
        while self._running:
            try:
                resp = await self.client.get(
                    f"{self.relay_url}/relay/pull/{node_id}",
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=self.poll_timeout + 10.0)
            except httpx.HTTPError:
                await asyncio.sleep(1.0)
                if not await self._reattach():
                    await asyncio.sleep(3.0)
                continue
            if resp.status_code == 204:
                continue
            if resp.status_code == 401:
                if not await self._reattach():
                    await asyncio.sleep(3.0)
                continue
            if resp.status_code != 200:
                await asyncio.sleep(1.0)
                continue
            asyncio.create_task(self._handle(resp.json()))

    async def _reattach(self) -> bool:
        try:
            await self.attach()
            return True
        except httpx.HTTPError:
            return False

    async def start(self) -> str:
        self._running = True
        if not self.token:
            await self.attach()
        self._tasks = [asyncio.create_task(self._puller()) for _ in range(self.concurrency)]
        return self.via_url

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        try:
            await self.client.post(
                f"{self.relay_url}/relay/detach", params={"node_id": self.keypair.address},
                headers={"Authorization": f"Bearer {self.token}"}, timeout=5.0)
        except httpx.HTTPError:
            pass
        await self._local.aclose()
