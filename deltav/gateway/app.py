"""Delta V gateway — the user-facing edge of the network.

Speaks the OpenAI chat-completions dialect and routes every request
through the SmartRouter onto the best (model, node) pair. The gateway
holds a funded wallet: it is the on-chain requester that authorizes and
pays for each job.
"""
from __future__ import annotations

import json
import re
import time
import uuid

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from ..config import ChainParams
from ..crypto import KeyPair
from ..router import Catalog, RouteError, SmartRouter
from ..router.catalog import estimate_vram_mb


def render_prompt(messages: list[dict]) -> str:
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    return "\n".join(lines) + "\nassistant:"


class GatewayDaemon:
    def __init__(
        self,
        keypair: KeyPair,
        node_urls: list[str],
        params: ChainParams | None = None,
        catalog: Catalog | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.keypair = keypair
        self.node_urls = node_urls
        self.params = params or ChainParams()
        self.client = client or httpx.AsyncClient()
        self._owns_client = client is None
        self.catalog = catalog or Catalog()
        self.router = SmartRouter(
            catalog=self.catalog,
            requester=self.keypair,
            client=self.client,
            price_per_token=self.params.price_per_token,
        )
        self.app = self._build_app()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Delta V gateway", version="0.1.0")

        @app.get("/health")
        async def health() -> dict:
            return {"gateway": self.keypair.address, "nodes": self.node_urls}

        @app.get("/network")
        async def network() -> dict:
            await self.router.refresh(self.node_urls)
            return {
                "height": self.router.chain_height,
                "nodes": [
                    {
                        "address": n.address, "endpoint": n.endpoint, "vram_mb": n.vram_mb,
                        "models": n.models, "reputation": n.reputation, "stake": n.stake,
                        "alive": n.alive, "load": n.load,
                    }
                    for n in self.router.nodes
                ],
            }

        @app.get("/v1/models")
        async def models() -> dict:
            await self.router.refresh(self.node_urls)
            data = []
            for spec in sorted(self.catalog.specs, key=lambda s: -s.quality):
                servers = [n.address for n in self.router.nodes if self.router._servable(spec, n) and n.alive]
                data.append({
                    "id": spec.ref,
                    "object": "model",
                    "owned_by": "deltav",
                    "deltav": {
                        "family": spec.family, "params_b": spec.params_b, "quant": spec.quant,
                        "vram_needed_mb": estimate_vram_mb(spec), "quality": spec.quality,
                        "served_by": servers,
                    },
                })
            return {"object": "list", "data": data}

        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict):
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            prompt = render_prompt(messages)
            await self.router.refresh(self.node_urls)
            try:
                result = await self.router.route(
                    prompt=prompt,
                    model=body.get("model", "auto"),
                    max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
                    temperature=float(body.get("temperature") or 0.0),
                    seed=int(body.get("seed") or 0),
                )
            except RouteError as exc:
                raise HTTPException(503, str(exc))

            completion_id = f"dvcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            usage = {
                "prompt_tokens": result.tokens_in,
                "completion_tokens": result.tokens_out,
                "total_tokens": result.tokens_in + result.tokens_out,
            }
            meta = {
                "node": result.node,
                "endpoint": result.endpoint,
                "receipt_tx": result.receipt_tx,
                "attempts": result.attempts,
            }

            if body.get("stream"):
                return StreamingResponse(
                    _sse_chunks(completion_id, created, result.model_ref, result.text, usage),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": result.model_ref,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": "stop",
                }],
                "usage": usage,
                "deltav": meta,
            }

        return app


async def _sse_chunks(completion_id: str, created: int, model: str, text: str, usage: dict):
    """OpenAI-compatible SSE stream.

    The node protocol returns the full completion (it has to be hashed
    into the on-chain receipt), so the gateway re-chunks it for clients
    that require streaming. True token-level passthrough is a later phase.
    """
    def chunk(delta: dict, finish: str | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    for piece in re.findall(r"\S+\s*", text) or [text]:
        yield chunk({"content": piece})
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": usage,
    }
    yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
