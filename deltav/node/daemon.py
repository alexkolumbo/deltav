"""Delta V node daemon.

One process = one network participant: it keeps a full copy of the
chain, gossips txs/blocks with peers, proposes blocks when the PoS
lottery picks it, serves inference on its compute backend, and — if it
has validator stake — spot-checks other nodes' receipts by re-executing
their jobs.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, HTTPException

from ..chain import Blockchain, Mempool, Tx, TxType
from ..chain.block import Block
from ..chain.consensus import ConsensusError
from ..compute import DeviceInfo, InferRequest, detect_device, make_backend
from ..config import Genesis
from ..crypto import KeyPair, canonical_json, sha256_hex

log = logging.getLogger("deltav.node")


@dataclass
class NodeConfig:
    host: str = "127.0.0.1"
    port: int = 9100
    endpoint: str = ""            # public URL; defaults to http://host:port
    peers: list[str] = field(default_factory=list)
    backend: str = "mock"
    models: list[str] = field(default_factory=list)  # model refs to announce
    stake: int = 0                 # udvt to stake at startup
    produce: bool = True
    spot_check: bool = True
    auto_register: bool = True
    device: DeviceInfo | None = None

    def public_url(self) -> str:
        return self.endpoint or f"http://{self.host}:{self.port}"


class NodeDaemon:
    def __init__(
        self,
        keypair: KeyPair,
        genesis: Genesis,
        cfg: NodeConfig,
        client: httpx.AsyncClient | None = None,
    ):
        self.keypair = keypair
        self.cfg = cfg
        self.chain = Blockchain(genesis)
        self.mempool = Mempool()
        self.backend = make_backend(cfg.backend)
        self.device = cfg.device or detect_device()
        self.client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self.jobs: dict[str, dict] = {}      # request_hash -> job params (for spot checkers)
        self.active_jobs = 0
        self._seen_blocks: set[str] = set()
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self.app = self._build_app()

    # --------------------------------------------------------------- utils
    @property
    def address(self) -> str:
        return self.keypair.address

    def _next_nonce(self) -> int:
        base = self.chain.state.account(self.address).nonce
        pending = [tx.nonce for tx in self.mempool.txs.values() if tx.sender == self.address]
        return max([base] + [n + 1 for n in pending])

    def _make_tx(self, tx_type: TxType, payload: dict) -> Tx:
        tx = Tx(type=tx_type.value, sender=self.address, nonce=self._next_nonce(), payload=payload)
        return tx.sign(self.keypair)

    async def submit_tx(self, tx: Tx) -> bool:
        if not self.mempool.add(tx):
            return False
        await self._gossip("/tx", tx.to_dict())
        return True

    async def _gossip(self, path: str, body: dict) -> None:
        async def send(peer: str) -> None:
            try:
                await self.client.post(f"{peer}{path}", json=body, timeout=5.0)
            except httpx.HTTPError:
                pass
        await asyncio.gather(*(send(p) for p in self.cfg.peers), return_exceptions=True)

    # ----------------------------------------------------------------- api
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Delta V node", version="0.1.0")

        @app.get("/health")
        async def health() -> dict:
            return {
                "address": self.address,
                "chain_id": self.chain.genesis.params.chain_id,
                "height": self.chain.height,
                "head": self.chain.head.hash,
                "backend": self.backend.name,
                "device": self.device.to_dict(),
                "models": self.cfg.models,
                "load": min(1.0, self.active_jobs / 4.0),
                "mempool": len(self.mempool),
            }

        @app.post("/tx")
        async def receive_tx(body: dict) -> dict:
            try:
                tx = Tx.from_dict(body)
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, f"malformed tx: {exc}")
            accepted = await self.submit_tx(tx)
            return {"accepted": accepted, "hash": tx.hash}

        @app.post("/p2p/block")
        async def receive_block(body: dict) -> dict:
            block_dict = body.get("block", {})
            from_url = body.get("from_url", "")
            try:
                block = Block.from_dict(block_dict)
            except (KeyError, ValueError) as exc:
                raise HTTPException(400, f"malformed block: {exc}")
            if block.hash in self._seen_blocks:
                return {"status": "seen"}
            self._seen_blocks.add(block.hash)
            try:
                self.chain.add_block(block)
                self.mempool.prune(self.chain.state)
                await self._gossip("/p2p/block", body)
                return {"status": "accepted", "height": self.chain.height}
            except ConsensusError:
                if block.height > self.chain.height + 1 and from_url:
                    asyncio.get_running_loop().create_task(self._sync_from(from_url))
                    return {"status": "syncing"}
                return {"status": "rejected"}

        @app.get("/chain/head")
        async def chain_head() -> dict:
            return self.chain.head.to_dict()

        @app.get("/chain/blocks")
        async def chain_blocks(start: int = 0, count: int = 500) -> dict:
            return {"blocks": self.chain.blocks_from(start, count)}

        @app.get("/chain/nodes")
        async def chain_nodes() -> dict:
            state = self.chain.state
            nodes = []
            for addr, node in sorted(state.nodes.items()):
                acc = state.account(addr)
                nodes.append({
                    "address": addr,
                    "endpoint": node.endpoint,
                    "hardware": node.hardware,
                    "models": node.models,
                    "reputation": node.reputation,
                    "stake": acc.stake,
                    "jobs_done": node.jobs_done,
                    "tokens_served": node.tokens_served,
                    "last_seen": node.last_seen,
                    "active": node.active,
                })
            return {"height": state.height, "nodes": nodes}

        @app.get("/chain/account/{address}")
        async def chain_account(address: str) -> dict:
            acc = self.chain.state.account(address)
            return {"address": address, "balance": acc.balance, "nonce": acc.nonce, "stake": acc.stake}

        @app.get("/chain/receipts")
        async def chain_receipts() -> dict:
            receipts = [
                {
                    "receipt_hash": r.receipt_hash, "node": r.node, "model": r.model,
                    "tokens": r.tokens_in + r.tokens_out, "price_paid": r.price_paid,
                    "height": r.height, "checked": r.checked, "check_ok": r.check_ok,
                }
                for r in sorted(self.chain.state.receipts.values(), key=lambda x: x.height)
            ]
            return {"receipts": receipts}

        @app.get("/job/{request_hash}")
        async def get_job(request_hash: str) -> dict:
            job = self.jobs.get(request_hash)
            if job is None:
                raise HTTPException(404, "unknown job")
            return job

        @app.post("/infer")
        async def infer(body: dict) -> dict:
            return await self._handle_infer(body)

        return app

    # ------------------------------------------------------------- serving
    async def _handle_infer(self, body: dict) -> dict:
        for key in ("prompt", "model", "requester", "requester_pubkey", "requester_sig", "price_limit"):
            if key not in body:
                raise HTTPException(400, f"missing field {key!r}")
        prompt = body["prompt"]
        model_ref = body["model"]
        max_tokens = int(body.get("max_tokens", 256))
        temperature = float(body.get("temperature", 0.0))
        seed = int(body.get("seed", 0))

        request_hash = sha256_hex(canonical_json(
            {"prompt": prompt, "model": model_ref, "max_tokens": max_tokens, "seed": seed}
        ))
        request = InferRequest(prompt=prompt, model_ref=model_ref,
                               max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.active_jobs += 1
        try:
            result = await asyncio.to_thread(self.backend.infer, request)
        except Exception as exc:
            raise HTTPException(500, f"inference failed: {exc}")
        finally:
            self.active_jobs -= 1

        output_hash = sha256_hex(result.text.encode())
        self.jobs[request_hash] = {
            "prompt": prompt, "model": model_ref, "max_tokens": max_tokens,
            "temperature": temperature, "seed": seed, "output_hash": output_hash,
        }
        receipt = self._make_tx(TxType.INFERENCE_RECEIPT, {
            "requester": body["requester"],
            "requester_pubkey": body["requester_pubkey"],
            "requester_sig": body["requester_sig"],
            "request_hash": request_hash,
            "output_hash": output_hash,
            "model": model_ref,
            "seed": seed,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "price_limit": int(body["price_limit"]),
        })
        await self.submit_tx(receipt)
        return {
            "text": result.text,
            "tokens_in": result.tokens_in,
            "tokens_out": result.tokens_out,
            "output_hash": output_hash,
            "request_hash": request_hash,
            "receipt_tx": receipt.hash,
            "node": self.address,
            "backend": self.backend.name,
        }

    # --------------------------------------------------------------- loops
    async def start(self) -> None:
        self._running = True
        if self.cfg.auto_register:
            self._tasks.append(asyncio.create_task(self._register_self()))
        self._tasks.append(asyncio.create_task(self._producer_loop()))
        self._tasks.append(asyncio.create_task(self._sync_loop()))
        if self.cfg.spot_check:
            self._tasks.append(asyncio.create_task(self._checker_loop()))

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._owns_client:
            await self.client.aclose()

    async def _register_self(self) -> None:
        register = self._make_tx(TxType.REGISTER_NODE, {
            "endpoint": self.cfg.public_url(),
            "hardware": self.device.to_dict() | {"backend": self.backend.name},
            "models": self.cfg.models,
        })
        await self.submit_tx(register)
        if self.cfg.stake > 0:
            stake = self._make_tx(TxType.STAKE, {"amount": self.cfg.stake})
            await self.submit_tx(stake)

    async def _producer_loop(self) -> None:
        params = self.chain.genesis.params
        while self._running:
            await asyncio.sleep(params.block_time)
            if not self.cfg.produce:
                continue
            if self.chain.next_proposer() != self.address:
                continue
            block = self.chain.build_block(self.keypair, self.mempool.collect(), time.time())
            try:
                self.chain.add_block(block)
            except ConsensusError as exc:  # e.g. lost a race with an incoming block
                log.debug("own block rejected: %s", exc)
                continue
            self._seen_blocks.add(block.hash)
            self.mempool.prune(self.chain.state)
            await self._gossip("/p2p/block", {"block": block.to_dict(), "from_url": self.cfg.public_url()})

    async def _sync_loop(self) -> None:
        params = self.chain.genesis.params
        while self._running:
            await asyncio.sleep(params.block_time)
            for peer in self.cfg.peers:
                try:
                    resp = await self.client.get(f"{peer}/chain/head", timeout=5.0)
                    resp.raise_for_status()
                except httpx.HTTPError:
                    continue
                if int(resp.json().get("height", 0)) > self.chain.height:
                    await self._sync_from(peer)

    async def _sync_from(self, peer: str) -> None:
        try:
            resp = await self.client.get(
                f"{peer}/chain/blocks", params={"start": 0, "count": 10_000}, timeout=30.0
            )
            resp.raise_for_status()
            blocks = resp.json().get("blocks", [])
        except httpx.HTTPError:
            return
        if self.chain.replace(blocks):
            self.mempool.prune(self.chain.state)
            log.info("synced to height %s from %s", self.chain.height, peer)

    # --------------------------------------------------------- spot checks
    def _my_check_duty(self, receipt_hash: str) -> bool:
        """Deterministic per-validator sampling of receipts to re-execute."""
        h = hashlib.sha256(f"{receipt_hash}:{self.address}".encode()).digest()
        return (int.from_bytes(h[:4], "big") % 1000) / 1000.0 < self.chain.state.params.spot_check_rate

    async def _checker_loop(self) -> None:
        params = self.chain.genesis.params
        while self._running:
            await asyncio.sleep(params.block_time * 1.5)
            state = self.chain.state
            if state.account(self.address).stake < params.min_validator_stake:
                continue
            if not self.backend.deterministic:
                continue
            for receipt in state.unchecked_receipts():
                if receipt.node == self.address or not self._my_check_duty(receipt.receipt_hash):
                    continue
                node = state.nodes.get(receipt.node)
                if node is None:
                    continue
                verdict = await self._verify_receipt(node.endpoint, receipt)
                if verdict is None:
                    continue  # couldn't fetch the job — don't punish on network noise
                check = self._make_tx(TxType.SPOT_CHECK, {
                    "receipt_hash": receipt.receipt_hash,
                    "ok": verdict,
                })
                await self.submit_tx(check)

    async def _verify_receipt(self, endpoint: str, receipt) -> bool | None:
        try:
            resp = await self.client.get(f"{endpoint}/job/{receipt.request_hash}", timeout=10.0)
            resp.raise_for_status()
            job = resp.json()
        except httpx.HTTPError:
            return None
        request = InferRequest(
            prompt=job["prompt"], model_ref=job["model"],
            max_tokens=int(job["max_tokens"]),
            temperature=float(job.get("temperature", 0.0)),
            seed=int(job["seed"]),
        )
        try:
            result = await asyncio.to_thread(self.backend.infer, request)
        except Exception:
            return None
        return sha256_hex(result.text.encode()) == receipt.output_hash
