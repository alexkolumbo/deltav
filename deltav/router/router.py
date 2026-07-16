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


@dataclass
class EmbedRouteResult:
    vectors: list[list[float]]
    model_ref: str
    node: str
    tokens: int
    receipt_tx: str | None
    attempts: int


@dataclass
class ImageRouteResult:
    images: list
    model_ref: str
    node: str
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
                    price_per_token=int(n.get("price_per_token", 0)),
                    dynamic=bool(n.get("hardware", {}).get("dynamic_models", True)),
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
        # Fixed-model nodes (llama-server) can't cold-load anything else.
        if not node.dynamic:
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
            key=lambda n: score_node(n, spec.ref, self.chain_height, self.price_per_token),
            reverse=True,
        )
        return [n for n in ranked
                if score_node(n, spec.ref, self.chain_height, self.price_per_token) > float("-inf")]

    # ------------------------------------------------------------ dispatch
    def estimate_price_limit(self, prompt: str, max_tokens: int) -> int:
        return (max_tokens + len(prompt.split()) + 64) * self.price_per_token * 2

    def _infer_body(self, node: NodeView, spec: ModelSpec, prompt: str,
                    max_tokens: int, temperature: float, seed: int,
                    requester: KeyPair | None = None,
                    images: list | None = None) -> dict[str, Any]:
        payer = requester or self.requester
        req_hash = request_hash_for(prompt, spec.ref, max_tokens, seed)
        price_limit = self.estimate_price_limit(prompt, max_tokens)
        auth_sig = payer.sign(
            receipt_auth_bytes(req_hash, node.address, spec.ref, price_limit)
        )
        body = {
            "prompt": prompt,
            "model": spec.ref,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "requester": payer.address,
            "requester_pubkey": payer.public_hex,
            "requester_sig": auth_sig,
            "price_limit": price_limit,
        }
        if images:
            body["images"] = images
        return body

    async def route(
        self,
        prompt: str,
        model: str = "auto",
        max_tokens: int = 256,
        temperature: float = 0.0,
        seed: int = 0,
        requester: KeyPair | None = None,
        images: list | None = None,
    ) -> RouteResult:
        spec = self.resolve_model(model)
        candidates = self.rank_nodes(spec)
        if not candidates:
            raise RouteError(f"no node can serve {spec.ref}")

        errors: list[str] = []
        for attempt, node in enumerate(candidates, start=1):
            body = self._infer_body(node, spec, prompt, max_tokens, temperature, seed,
                                    requester, images)
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
        requester: KeyPair | None = None,
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
            body = self._infer_body(node, spec, prompt, max_tokens, temperature, seed,
                                    requester)
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

    # ---------------------------------------------------------- embeddings
    def resolve_embedding_model(self, model: str = "auto") -> ModelSpec:
        live = [n for n in self.nodes if n.alive and n.active]
        if not live:
            raise RouteError("no live nodes on the network")
        if model in ("auto", "", None):
            for spec in self.catalog.embedding_specs():
                if any(self._servable(spec, n) for n in live):
                    return spec
            raise RouteError("no embedding model is servable by any live node")
        spec = self.catalog.by_ref(model)
        if spec is not None and spec.kind == "embedding":
            return spec
        if any(model in n.models for n in live):
            repo, _, filename = model.partition("::")
            return ModelSpec(repo_id=repo, filename=filename, family="external",
                             params_b=0.0, quant="api", file_mb=0, quality=0.5,
                             kind="embedding")
        raise RouteError(f"no embedding model {model!r} on the network")

    async def route_embed(self, texts: list[str], model: str = "auto",
                          requester: KeyPair | None = None) -> EmbedRouteResult:
        if not texts:
            raise RouteError("no texts to embed")
        payer = requester or self.requester
        spec = self.resolve_embedding_model(model)
        candidates = self.rank_nodes(spec)
        if not candidates:
            raise RouteError(f"no node can serve {spec.ref}")

        errors: list[str] = []
        for attempt, node in enumerate(candidates, start=1):
            req_hash = sha256_hex(canonical_json({"texts": texts, "model": spec.ref}))
            total_words = sum(len(t.split()) for t in texts)
            price_limit = (total_words + 16 * len(texts) + 64) * self.price_per_token * 2
            auth_sig = payer.sign(
                receipt_auth_bytes(req_hash, node.address, spec.ref, price_limit)
            )
            body = {
                "texts": texts,
                "model": spec.ref,
                "requester": payer.address,
                "requester_pubkey": payer.public_hex,
                "requester_sig": auth_sig,
                "price_limit": price_limit,
            }
            try:
                resp = await self.client.post(f"{node.endpoint}/embed", json=body, timeout=120.0)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                errors.append(f"{node.address[:12]}: {exc}")
                node.alive = False
                continue
            return EmbedRouteResult(
                vectors=data["vectors"],
                model_ref=spec.ref,
                node=node.address,
                tokens=int(data.get("tokens", 0)),
                receipt_tx=data.get("receipt_tx"),
                attempts=attempt,
            )
        raise RouteError("all candidate nodes failed: " + "; ".join(errors))

    # -------------------------------------------------------------- images
    def resolve_image_model(self, model: str = "auto") -> ModelSpec:
        live = [n for n in self.nodes if n.alive and n.active]
        if not live:
            raise RouteError("no live nodes on the network")
        if model in ("auto", "", None):
            for spec in self.catalog.image_specs():
                if any(spec.ref in n.models or spec.repo_id in n.models for n in live):
                    return spec
            raise RouteError("no image model is served by any live node")
        spec = self.catalog.by_ref(model)
        if spec is not None and spec.kind == "image":
            return spec
        if any(model in n.models for n in live):
            repo, _, filename = model.partition("::")
            return ModelSpec(repo_id=repo, filename=filename, family="external",
                             params_b=0.0, quant="api", file_mb=0, quality=0.5, kind="image")
        raise RouteError(f"no image model {model!r} on the network")

    async def route_image(self, prompt: str, model: str = "auto", width: int = 512,
                          height: int = 512, steps: int = 20, seed: int = 0,
                          requester: KeyPair | None = None) -> ImageRouteResult:
        payer = requester or self.requester
        spec = self.resolve_image_model(model)
        candidates = [n for n in self.nodes
                      if spec.ref in n.models or spec.repo_id in n.models]
        candidates.sort(key=lambda n: score_node(n, spec.ref, self.chain_height,
                                                 self.price_per_token), reverse=True)
        if not candidates:
            raise RouteError(f"no node serves {spec.ref}")
        errors: list[str] = []
        for attempt, node in enumerate(candidates, start=1):
            req_hash = sha256_hex(canonical_json(
                {"prompt": prompt, "model": spec.ref, "w": width, "h": height,
                 "steps": steps, "seed": seed}))
            price_limit = (width * height // 4096 + 64) * self.price_per_token * 4
            auth = payer.sign(receipt_auth_bytes(req_hash, node.address, spec.ref, price_limit))
            body = {"prompt": prompt, "model": spec.ref, "width": width, "height": height,
                    "steps": steps, "seed": seed, "requester": payer.address,
                    "requester_pubkey": payer.public_hex, "requester_sig": auth,
                    "price_limit": price_limit}
            try:
                resp = await self.client.post(f"{node.endpoint}/image", json=body, timeout=300.0)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                errors.append(f"{node.address[:12]}: {exc}")
                node.alive = False
                continue
            return ImageRouteResult(images=data["images"], model_ref=spec.ref,
                                    node=node.address, receipt_tx=data.get("receipt_tx"),
                                    attempts=attempt)
        raise RouteError("all candidate nodes failed: " + "; ".join(errors))
