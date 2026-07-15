"""Delta V gateway — the user-facing edge of the network.

Speaks the OpenAI chat-completions dialect and routes every request
through the SmartRouter onto the best (model, node) pair. The gateway
holds a funded wallet: it is the on-chain requester that authorizes and
pays for each job.
"""
from __future__ import annotations

import asyncio
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
from ..overlay.memory import VectorMemory
from ..overlay.tools import ToolSpec
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
        memory_path: str | None = None,
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
        # Memory embeds through the network's own paid embedding jobs and
        # falls back to BM25 when no embedding node is live.
        self.memory = VectorMemory(memory_path, embedder=self._embed_texts)
        self.app = self._build_app()

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.router.nodes:
            await self.router.refresh(self.node_urls)
        try:
            result = await self.router.route_embed(texts=texts, model="auto")
        except RouteError:
            # registry may be stale (nodes marked dead) — one refresh, one retry
            await self.router.refresh(self.node_urls)
            result = await self.router.route_embed(texts=texts, model="auto")
        return result.vectors

    def _agent_registry(self, complete, session_id: str | None, max_steps: int) -> ToolRegistry:
        """Per-run registry: built-ins + session memory + parallel sub-agents."""
        registry = ToolRegistry(self.tools.specs())

        if session_id:
            async def remember(text: str) -> str:
                item = await self.memory.aadd(session_id, str(text))
                return f"remembered as {item['id']}"

            async def recall(query: str, k: int = 4) -> str:
                hits = await self.memory.asearch(session_id, str(query), int(k))
                if not hits:
                    return "no matching memories"
                return "\n".join(f"[{h['id']}] {h['text']}" for h in hits)

            registry.register(ToolSpec(
                name="remember",
                description="Store a fact in this session's long-term memory.",
                parameters={"type": "object", "properties": {"text": {"type": "string"}},
                            "required": ["text"]},
                handler=remember,
            ))
            registry.register(ToolSpec(
                name="recall",
                description="Search this session's long-term memory.",
                parameters={"type": "object",
                            "properties": {"query": {"type": "string"},
                                           "k": {"type": "integer", "default": 4}},
                            "required": ["query"]},
                handler=recall,
            ))

        async def spawn_agents(tasks: list) -> str:
            """Run up to 4 sub-agents in parallel, each with a fresh context."""
            sub_registry = ToolRegistry(registry.specs())  # snapshot WITHOUT spawn_agents
            subs = [Agent(complete, sub_registry, max_steps=max_steps)
                    for _ in tasks[:4]]
            results = await asyncio.gather(
                *(agent.run(str(task)) for agent, task in zip(subs, tasks[:4])))
            return json.dumps(
                [{"task": str(t), "answer": r.answer, "steps": len(r.steps)}
                 for t, r in zip(tasks[:4], results)],
                ensure_ascii=False)

        registry.register(ToolSpec(
            name="spawn_agents",
            description=("Delegate independent subtasks to up to 4 parallel sub-agents "
                         "(fresh context each). Returns their answers as JSON."),
            parameters={"type": "object",
                        "properties": {"tasks": {"type": "array", "items": {"type": "string"}}},
                        "required": ["tasks"]},
            handler=spawn_agents,
        ))
        return registry

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

        @app.get("/v1/plan")
        async def plan_endpoint(vram_mb: int, objective: str = "balanced") -> dict:
            """Model planner enriched with live network state: which of the
            recommended models are already served (warm) right now."""
            from ..router.planner import launch_hint, plan

            await self.router.refresh(self.node_urls)
            options = plan(vram_mb, objective=objective, catalog=self.catalog)
            served = {m for n in self.router.nodes if n.alive for m in n.models}
            return {
                "vram_mb": vram_mb,
                "objective": objective,
                "options": [
                    o.to_dict() | {
                        "launch": launch_hint(o),
                        "already_served_on_network": o.ref in served,
                    }
                    for o in options
                ],
            }

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
            session_id = (body.get("session_id") or "").strip() or None
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

            registry = self._agent_registry(complete, session_id, max_steps)

            memory_used: list[dict] = []
            agent_task = task
            if session_id:
                memory_used = await self.memory.asearch(session_id, task, k=3)
                if memory_used:
                    context = "\n".join(f"- {h['text']}" for h in memory_used)
                    agent_task = f"Relevant memory from earlier sessions:\n{context}\n\nTask: {task}"

            agent = Agent(complete, registry, max_steps=max_steps)
            try:
                result = await agent.run(agent_task)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            return {
                "task": task,
                "session_id": session_id,
                "answer": result.answer,
                "finished": result.finished,
                "model_calls": result.model_calls,
                "memory_used": [{"id": h["id"], "text": h["text"], "score": h["score"]}
                                for h in memory_used],
                "steps": [
                    {
                        "tool": s.tool, "arguments": s.arguments,
                        "result": s.result[:2000],
                        "node": s.node, "receipt_tx": s.receipt_tx,
                    }
                    for s in result.steps
                ],
                "tools_available": registry.names(),
            }

        @app.get("/v1/memory")
        async def memory_view(session: str, q: str = "", k: int = 8) -> dict:
            if q:
                items = await self.memory.asearch(session, q, k)
            else:
                items = self.memory.session_items(session)[-k:]
            return {"session": session,
                    "items": [{key: v for key, v in it.items() if key != "vec"}
                              for it in items]}

        @app.post("/v1/embeddings")
        async def embeddings(body: dict) -> dict:
            raw = body.get("input")
            texts = [raw] if isinstance(raw, str) else [str(t) for t in (raw or [])]
            if not texts:
                raise HTTPException(400, "input required")
            await self.router.refresh(self.node_urls)
            try:
                result = await self.router.route_embed(texts, model=body.get("model", "auto"))
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            return {
                "object": "list",
                "model": result.model_ref,
                "data": [
                    {"object": "embedding", "index": i, "embedding": vec}
                    for i, vec in enumerate(result.vectors)
                ],
                "usage": {"prompt_tokens": result.tokens, "total_tokens": result.tokens},
                "deltav": {"node": result.node, "receipt_tx": result.receipt_tx,
                           "attempts": result.attempts},
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict):
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            tools = body.get("tools") or None
            prompt = render_prompt(messages, tools)
            await self.router.refresh(self.node_urls)

            if body.get("stream") and not tools:
                # True end-to-end streaming: tokens flow node -> gateway ->
                # client as they are generated. (With tools we must see the
                # full text to parse <tool_call>, so that path buffers.)
                try:
                    spec = self.router.resolve_model(body.get("model", "auto"))
                    upstream = self.router.route_stream(
                        prompt=prompt, model=spec.ref,
                        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
                        temperature=float(body.get("temperature") or 0.0),
                        seed=int(body.get("seed") or 0),
                    )
                    first = await upstream.__anext__()  # fail as HTTP 503, not mid-stream
                except (RouteError, StopAsyncIteration) as exc:
                    raise HTTPException(503, str(exc))
                return StreamingResponse(
                    _sse_passthrough(f"dvcmpl-{uuid.uuid4().hex[:24]}", int(time.time()),
                                     spec.ref, first, upstream),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )

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


def _chunk_line(completion_id: str, created: int, model: str,
                delta: dict, finish: str | None = None, extra: dict | None = None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if extra:
        payload.update(extra)
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _sse_passthrough(completion_id: str, created: int, model: str, first, upstream):
    """Relay live ("token", str) / ("final", RouteResult) events as OpenAI chunks."""
    yield _chunk_line(completion_id, created, model, {"role": "assistant", "content": ""})

    async def events():
        yield first
        async for event in upstream:
            yield event

    async for kind, value in events():
        if kind == "token":
            if value:
                yield _chunk_line(completion_id, created, model, {"content": value})
            continue
        usage = {
            "prompt_tokens": value.tokens_in,
            "completion_tokens": value.tokens_out,
            "total_tokens": value.tokens_in + value.tokens_out,
        }
        meta = {"node": value.node, "endpoint": value.endpoint,
                "receipt_tx": value.receipt_tx, "attempts": value.attempts}
        yield _chunk_line(completion_id, created, model, {}, "stop",
                          extra={"usage": usage, "deltav": meta})
    yield "data: [DONE]\n\n"


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
