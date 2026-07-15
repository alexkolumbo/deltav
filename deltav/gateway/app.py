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
from ..overlay import (
    Agent,
    SearchEngine,
    ToolRegistry,
    build_tool_system_prompt,
    builtin_registry,
    parse_tool_calls,
    render_conversation,
    strip_tool_calls,
    to_openai_tool_calls,
)
from ..router import Catalog, RouteError, SmartRouter
from ..router.catalog import estimate_vram_mb


def render_prompt(messages: list[dict], tools: list[dict] | None = None) -> str:
    if tools:
        system = {"role": "system", "content": build_tool_system_prompt(tools)}
        return render_conversation([system, *messages])
    return render_conversation(messages)


class GatewayDaemon:
    def __init__(
        self,
        keypair: KeyPair,
        node_urls: list[str],
        params: ChainParams | None = None,
        catalog: Catalog | None = None,
        client: httpx.AsyncClient | None = None,
        tools: ToolRegistry | None = None,
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
        # Overlay: search engine + tool registry shared by tool-calling and agents.
        self.tools = tools or builtin_registry(self.client)
        self.search = SearchEngine(self.client)
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

        @app.get("/v1/search")
        async def search(q: str, max_results: int = 5) -> dict:
            """The network's internet search surface (also a built-in tool)."""
            results = await self.search.search(q, max_results)
            return {"query": q, "results": results}

        @app.post("/v1/agents/run")
        async def agents_run(body: dict) -> dict:
            task = body.get("task", "").strip()
            if not task:
                raise HTTPException(400, "task required")
            model = body.get("model", "auto")
            max_steps = min(int(body.get("max_steps", 6)), 16)
            await self.router.refresh(self.node_urls)

            async def complete(prompt: str) -> tuple[str, dict]:
                result = await self.router.route(
                    prompt=prompt, model=model,
                    max_tokens=int(body.get("max_tokens", 512)),
                    temperature=float(body.get("temperature") or 0.0),
                    seed=int(body.get("seed") or 0),
                )
                return result.text, {"node": result.node, "receipt_tx": result.receipt_tx}

            agent = Agent(complete, self.tools, max_steps=max_steps)
            try:
                result = await agent.run(task)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            return {
                "task": task,
                "answer": result.answer,
                "finished": result.finished,
                "model_calls": result.model_calls,
                "steps": [
                    {
                        "tool": s.tool, "arguments": s.arguments,
                        "result": s.result[:2000],
                        "node": s.node, "receipt_tx": s.receipt_tx,
                    }
                    for s in result.steps
                ],
                "tools_available": self.tools.names(),
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict):
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            tools = body.get("tools") or None
            prompt = render_prompt(messages, tools)
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

            # OpenAI tools dialect: a <tool_call> in the completion becomes
            # a tool_calls message; the client executes and calls us back.
            tool_calls = to_openai_tool_calls(parse_tool_calls(result.text)) if tools else []
            if tool_calls:
                message = {
                    "role": "assistant",
                    "content": strip_tool_calls(result.text) or None,
                    "tool_calls": tool_calls,
                }
                finish = "tool_calls"
            else:
                message = {"role": "assistant", "content": result.text}
                finish = "stop"

            if body.get("stream"):
                return StreamingResponse(
                    _sse_chunks(completion_id, created, result.model_ref,
                                message, finish, usage),
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
                    "message": message,
                    "finish_reason": finish,
                }],
                "usage": usage,
                "deltav": meta,
            }

        return app


async def _sse_chunks(completion_id: str, created: int, model: str,
                      message: dict, finish: str, usage: dict):
    """OpenAI-compatible SSE stream.

    The node protocol returns the full completion (it has to be hashed
    into the on-chain receipt), so the gateway re-chunks it for clients
    that require streaming. True token-level passthrough is a later phase.
    """
    def chunk(delta: dict, finish_reason: str | None = None, extra: dict | None = None) -> str:
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extra:
            payload.update(extra)
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield chunk({"role": "assistant", "content": ""})
    if message.get("tool_calls"):
        deltas = [
            {**c, "index": i, "function": dict(c["function"])}
            for i, c in enumerate(message["tool_calls"])
        ]
        yield chunk({"tool_calls": deltas})
    else:
        text = message.get("content") or ""
        for piece in re.findall(r"\S+\s*", text) or [text]:
            yield chunk({"content": piece})
    yield chunk({}, finish, extra={"usage": usage})
    yield "data: [DONE]\n\n"
