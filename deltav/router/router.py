"""SmartRouter: pick the best (model, node) pair on the network and dispatch.

Model resolution:
  * "auto"  — the highest-quality catalog model that at least one live
              node can actually serve (announced it, or fits its VRAM);
  * a repo id / ref — routed to nodes serving it, VRAM-fit as fallback.

Dispatch signs a payment authorization (request_hash + node + price cap)
with the requester's key, so the serving node can claim payment on-chain
with an INFERENCE_RECEIPT and nothing else.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from ..chain.transaction import receipt_auth_bytes
from ..crypto import KeyPair, canonical_json, sha256_hex
from .catalog import Catalog, ModelSpec, estimate_vram_mb
from .scoring import NodeView, score_node


class RouteError(Exception):
    pass


@dataclass
class RouteResult:
    text: str
    model_ref: str
    node: str
    endpoint: str
    tokens_in: int
    tokens_out: int
    receipt_tx: str | None
    attempts: int


def request_hash_for(prompt: str, model_ref: str, max_tokens: int, seed: int) -> str:
    return sha256_hex(canonical_json(
        {"prompt": prompt, "model": model_ref, "max_tokens": max_tokens, "seed": seed}
    ))


class SmartRouter:
    def __init__(
        self,
        catalog: Catalog,
        requester: KeyPair,
        client: httpx.AsyncClient,
        price_per_token: int,
    ):
        self.catalog = catalog
        self.requester = requester
        self.client = client
        self.price_per_token = price_per_token
        self.nodes: list[NodeView] = []
        self.chain_height = 0

    # ------------------------------------------------------------ registry
    async def refresh(self, node_urls: list[str]) -> None:
        """Pull the node registry from the chain plus live health data."""
        registry: dict[str, NodeView] = {}
        for url in node_urls:
            try:
                resp = await self.client.get(f"{url}/chain/nodes", timeout=5.0)
                resp.raise_for_status()
            except httpx.HTTPError:
                continue
            data = resp.json()
            self.chain_height = max(self.chain_height, int(data.get("height", 0)))
            for n in data.get("nodes", []):
                registry[n["address"]] = NodeView(
                    address=n["address"],
                    endpoint=n["endpoint"],
                    vram_mb=int(n.get("hardware", {}).get("vram_mb", 0)),
                    models=list(n.get("models", [])),
                    reputation=float(n.get("reputation", 0.5)),
                    stake=int(n.get("stake", 0)),
                    last_seen=int(n.get("last_seen", 0)),
                    active=bool(n.get("active", True)),
                )
            break  # one healthy chain source is enough for the registry
        for node in registry.values():
            try:
                health = await self.client.get(f"{node.endpoint}/health", timeout=3.0)
                health.raise_for_status()
                node.load = float(health.json().get("load", 0.0))
                node.alive = True
            except httpx.HTTPError:
                node.alive = False
        self.nodes = list(registry.values())

    # ------------------------------------------------------------- resolve
    def _servable(self, spec: ModelSpec, node: NodeView) -> bool:
        if spec.ref in node.models or spec.repo_id in node.models:
            return True
        # API-relayed models (Groq etc.) have no file size — only nodes
        # that explicitly announced them can serve them.
        if spec.file_mb <= 0:
            return False
        return node.vram_mb > 0 and estimate_vram_mb(spec) <= node.vram_mb

    def resolve_model(self, model: str) -> ModelSpec:
        live = [n for n in self.nodes if n.alive and n.active]
        if not live:
            raise RouteError("no live nodes on the network")
        if model in ("auto", "", None):
            ranked = sorted(self.catalog.specs, key=lambda s: -s.quality)
            # Prefer models some node already announced (warm, no multi-GB
            # cold download); fall back to anything that fits a node's VRAM.
            for spec in ranked:
                if any(spec.ref in n.models or spec.repo_id in n.models for n in live):
                    return spec
            for spec in ranked:
                if any(self._servable(spec, n) for n in live):
                    return spec
            raise RouteError("no catalog model is servable by any live node")
        spec = self.catalog.by_ref(model)
        if spec is not None:
            return spec
        # Not in the catalog, but if live nodes announce exactly this ref
        # (e.g. a Groq relay), synthesize a spec for it.
        if any(model in n.models for n in live):
            repo, _, filename = model.partition("::")
            return ModelSpec(repo_id=repo, filename=filename, family="external",
                             params_b=0.0, quant="api", file_mb=0, quality=0.5)
        raise RouteError(f"model {model!r} is not in the catalog and no live node announces it")

    def rank_nodes(self, spec: ModelSpec) -> list[NodeView]:
        candidates = [n for n in self.nodes if self._servable(spec, n)]
        ranked = sorted(
            candidates,
            key=lambda n: score_node(n, spec.ref, self.chain_height),
            reverse=True,
        )
        return [n for n in ranked if score_node(n, spec.ref, self.chain_height) > float("-inf")]

    # ------------------------------------------------------------ dispatch
    def _infer_body(self, node: NodeView, spec: ModelSpec, prompt: str,
                    max_tokens: int, temperature: float, seed: int) -> dict[str, Any]:
        req_hash = request_hash_for(prompt, spec.ref, max_tokens, seed)
        price_limit = (max_tokens + len(prompt.split()) + 64) * self.price_per_token * 2
        auth_sig = self.requester.sign(
            receipt_auth_bytes(req_hash, node.address, spec.ref, price_limit)
        )
        return {
            "prompt": prompt,
            "model": spec.ref,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "requester": self.requester.address,
            "requester_pubkey": self.requester.public_hex,
            "requester_sig": auth_sig,
            "price_limit": price_limit,
        }

    async def route(
        self,
        prompt: str,
        model: str = "auto",
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> RouteResult:
        spec = self.resolve_model(model)
        candidates = self.rank_nodes(spec)
        if not candidates:
            raise RouteError(f"no node can serve {spec.ref}")

        errors: list[str] = []
        for attempt, node in enumerate(candidates, start=1):
            body = self._infer_body(node, spec, prompt, max_tokens, temperature, seed)
            try:
                resp = await self.client.post(f"{node.endpoint}/infer", json=body, timeout=300.0)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                errors.append(f"{node.address[:12]}: {exc}")
                node.alive = False  # local penalty; chain reputation moves via spot checks
                continue
            return RouteResult(
                text=data["text"],
                model_ref=spec.ref,
                node=node.address,
                endpoint=node.endpoint,
                tokens_in=int(data.get("tokens_in", 0)),
                tokens_out=int(data.get("tokens_out", 0)),
                receipt_tx=data.get("receipt_tx"),
                attempts=attempt,
            )
        raise RouteError("all candidate nodes failed: " + "; ".join(errors))

    async def route_stream(
        self,
        prompt: str,
        model: str = "auto",
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
    ):
        """End-to-end token streaming: yields ("token", str) pieces as the
        node generates them, then ("final", RouteResult). Failover happens
        only before the first token — a stream broken mid-generation is an
        error (the receipt already committed to that output)."""
        spec = self.resolve_model(model)
        candidates = self.rank_nodes(spec)
        if not candidates:
            raise RouteError(f"no node can serve {spec.ref}")

        errors: list[str] = []
        for attempt, node in enumerate(candidates, start=1):
            body = self._infer_body(node, spec, prompt, max_tokens, temperature, seed)
            body["stream"] = True
            pieces: list[str] = []
            try:
                async with self.client.stream(
                    "POST", f"{node.endpoint}/infer", json=body, timeout=300.0
                ) as resp:
                    if resp.status_code != 200:
                        errors.append(f"{node.address[:12]}: http {resp.status_code}")
                        node.alive = False
                        continue
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[len("data: "):])
                        if "error" in event:
                            raise RouteError(f"node error mid-stream: {event['error']}")
                        if event.get("done"):
                            yield ("final", RouteResult(
                                text=event.get("text", "".join(pieces)),
                                model_ref=spec.ref,
                                node=node.address,
                                endpoint=node.endpoint,
                                tokens_in=int(event.get("tokens_in", 0)),
                                tokens_out=int(event.get("tokens_out", 0)),
                                receipt_tx=event.get("receipt_tx"),
                                attempts=attempt,
                            ))
                            return
                        token = event.get("token", "")
                        pieces.append(token)
                        yield ("token", token)
            except httpx.HTTPError as exc:
                if pieces:
                    raise RouteError(f"stream broke mid-generation: {exc}") from exc
                errors.append(f"{node.address[:12]}: {exc}")
                node.alive = False
                continue
            raise RouteError("node stream ended without a final event")
        raise RouteError("all candidate nodes failed: " + "; ".join(errors))
