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

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from .anthropic import (
    anthropic_text_stream,
    messages_to_prompt,
    to_anthropic_message,
    to_anthropic_tool_stream,
)
from . import ollama as ol
from .keys import KEY_PREFIX, KeyRecord, KeyStore

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
from ..companion import CompanionAgent, UserMemory, resolve_identity
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
        keys_path: str | None = None,
        require_keys: bool = False,
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
        # Billing: API keys are custodial on-chain wallets.
        self.keys = KeyStore(keys_path)
        self.require_keys = require_keys
        self.app = self._build_app()

    # ------------------------------------------------------------- billing
    async def _requester_from(self, request: Request) -> tuple[KeyPair, KeyRecord | None]:
        """Resolve the paying wallet for a request.

        dvk_ tokens must resolve (401 otherwise); anything else — goose
        and friends send placeholder api keys — is anonymous, allowed
        only when require_keys is off (then the gateway wallet pays)."""
        auth = request.headers.get("authorization", "")
        token = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if token.startswith(KEY_PREFIX):
            record = self.keys.resolve(token)
            if record is None:
                raise HTTPException(401, "unknown API key")
            return self.keys.keypair(record), record
        if self.require_keys:
            raise HTTPException(401, "API key required: POST /v1/keys, then fund its address")
        return self.keypair, None

    async def _onchain_balance(self, address: str) -> int:
        for url in self.node_urls:
            try:
                resp = await self.client.get(f"{url}/chain/account/{address}", timeout=5.0)
                resp.raise_for_status()
                return int(resp.json()["balance"])
            except httpx.HTTPError:
                continue
        raise HTTPException(503, "no node reachable to check balance")

    async def _precheck_funds(self, record: KeyRecord | None,
                              prompt: str, max_tokens: int) -> None:
        """Fail with 402 BEFORE inference: an underfunded receipt would be
        rejected on-chain and the node would have worked for free."""
        if record is None:
            return
        needed = self.router.estimate_price_limit(prompt, max_tokens)
        balance = await self._onchain_balance(record.address)
        if balance < needed:
            raise HTTPException(
                402,
                f"insufficient DVT: balance {balance} udvt, need ~{needed}; "
                f"top up {record.address}")

    def _distinct_models(self, n: int) -> list[str]:
        """Up to n distinct chat models currently served by live nodes,
        best-quality first — the basis for spreading a swarm across nodes."""
        served: set[str] = set()
        for node in self.router.nodes:
            if node.alive and node.active:
                served.update(node.models)
        specs = sorted(
            (s for s in self.catalog.specs if s.kind == "chat" and s.ref in served),
            key=lambda s: -s.quality)
        picked = [s.ref for s in specs]
        # include announced models we don't have in the catalog (skip embeddings)
        for ref in sorted(served):
            spec = self.catalog.by_ref(ref)
            if ref not in picked and (spec is None or spec.kind == "chat"):
                picked.append(ref)
        return picked[:n]

    async def _ollama_run(self, payer, record, model: str, tag: str, prompt: str,
                          max_tokens: int, temperature: float, stream: bool, kind: str):
        """Shared chat/generate runner for the Ollama API (NDJSON stream or
        a single JSON object), billed like any request."""
        if stream:
            try:
                spec = self.router.resolve_model(model)
                upstream = self.router.route_stream(
                    prompt=prompt, model=spec.ref, max_tokens=max_tokens,
                    temperature=temperature, requester=payer)
                first = await upstream.__anext__()
            except (RouteError, StopAsyncIteration) as exc:
                raise HTTPException(503, str(exc))
            holder: dict = {}

            async def pieces():
                kind0, value = first
                if kind0 == "token" and value:
                    yield value
                async for k, v in upstream:
                    if k == "token" and v:
                        yield v
                    elif k == "final":
                        total = v.tokens_in + v.tokens_out
                        holder.update(tokens_in=v.tokens_in, tokens_out=v.tokens_out,
                                      meta={"node": v.node, "receipt_tx": v.receipt_tx})
                        if record is not None:
                            self.keys.record_usage(record, total,
                                                   total * self.params.price_per_token)

            gen = (ol.chat_stream if kind == "chat" else ol.generate_stream)(tag, pieces(), holder)
            return StreamingResponse(gen, media_type="application/x-ndjson")

        try:
            result = await self.router.route(prompt=prompt, model=model,
                                             max_tokens=max_tokens, temperature=temperature,
                                             requester=payer)
        except RouteError as exc:
            raise HTTPException(503, str(exc))
        total = result.tokens_in + result.tokens_out
        if record is not None:
            self.keys.record_usage(record, total, total * self.params.price_per_token)
        meta = {"node": result.node, "receipt_tx": result.receipt_tx}
        builder = ol.chat_response if kind == "chat" else ol.generate_response
        return builder(tag, result.text, result.tokens_in, result.tokens_out, meta)

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

        @app.get("/chat", response_class=HTMLResponse)
        async def chat_ui() -> str:
            """Mobile-first chat frontend served straight off the gateway."""
            return (Path(__file__).parent / "chat.html").read_text(encoding="utf-8")

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
                        "kind": spec.kind,
                        "vram_needed_mb": estimate_vram_mb(spec), "quality": spec.quality,
                        "served_by": servers,
                    },
                })
            return {"object": "list", "data": data}

        @app.post("/v1/keys")
        async def keys_create(body: dict | None = None) -> dict:
            api_key, record = self.keys.create(label=(body or {}).get("label", ""))
            return {
                "api_key": api_key,   # shown exactly once
                "address": record.address,
                "note": "fund this address with DVT (deltav send --to <address>); "
                        "requests are paid from it on-chain",
            }

        @app.get("/v1/keys/me")
        async def keys_me(request: Request) -> dict:
            _, record = await self._requester_from(request)
            if record is None:
                raise HTTPException(401, "pass your dvk_ API key as a Bearer token")
            return {
                "label": record.label,
                "address": record.address,
                "balance_udvt": await self._onchain_balance(record.address),
                "requests": record.requests,
                "tokens": record.tokens,
                "spent_udvt_estimate": record.spent_udvt,
            }

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
        async def agents_run(body: dict, request: Request) -> dict:
            payer, record = await self._requester_from(request)
            task = body.get("task", "").strip()
            if not task:
                raise HTTPException(400, "task required")
            model = body.get("model", "auto")
            session_id = (body.get("session_id") or "").strip() or None
            max_steps = min(int(body.get("max_steps", 6)), 16)
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, task, int(body.get("max_tokens", 512)) * 2)

            spent = {"tokens": 0}

            async def complete(prompt: str) -> tuple[str, dict]:
                result = await self.router.route(
                    prompt=prompt, model=model,
                    max_tokens=int(body.get("max_tokens", 512)),
                    temperature=float(body.get("temperature") or 0.0),
                    seed=int(body.get("seed") or 0),
                    requester=payer,
                )
                spent["tokens"] += result.tokens_in + result.tokens_out
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
            if record is not None:
                self.keys.record_usage(record, spent["tokens"],
                                       spent["tokens"] * self.params.price_per_token)
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

        @app.post("/v1/companion/chat")
        async def companion_chat(body: dict, request: Request) -> dict:
            """Persistent, per-user, self-improving agent. The user's identity
            comes from their key — memory is strictly isolated per user."""
            payer, record = await self._requester_from(request)
            message = (body.get("message") or "").strip()
            if not message:
                raise HTTPException(400, "message required")
            identity = resolve_identity(
                address=record.address if record else "",
                fallback_user=body.get("user_id", ""))
            memory = UserMemory(self.memory, identity)
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, message, int(body.get("max_tokens", 512)) * 3)

            async def complete(prompt: str):
                r = await self.router.route(prompt=prompt, model=body.get("model", "auto"),
                                            max_tokens=int(body.get("max_tokens", 512)),
                                            requester=payer)
                if record is not None:
                    used = r.tokens_in + r.tokens_out
                    self.keys.record_usage(record, used, used * self.params.price_per_token)
                return r.text, {"node": r.node, "receipt_tx": r.receipt_tx}

            agent = CompanionAgent(complete, self.tools,
                                   max_steps=min(int(body.get("max_steps", 6)), 12),
                                   reflect=body.get("reflect", True))
            try:
                result = await agent.run(memory, message, body.get("history"))
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            return {
                "answer": result.answer,
                "user": identity.user_id,
                "authenticated": identity.authenticated,
                "memory_used": result.memory_used,
                "learned": result.learned,
                "model_calls": result.model_calls,
                "steps": [{"tool": s.tool, "arguments": s.arguments,
                           "result": s.result[:800], "receipt_tx": s.receipt_tx}
                          for s in result.steps],
            }

        @app.post("/v1/companion/feedback")
        async def companion_feedback(body: dict, request: Request) -> dict:
            _, record = await self._requester_from(request)
            note = (body.get("note") or "").strip()
            if not note:
                raise HTTPException(400, "note required")
            identity = resolve_identity(record.address if record else "",
                                        body.get("user_id", ""))
            memory = UserMemory(self.memory, identity)
            item = await CompanionAgent(None, self.tools).feedback(memory, note)
            return {"stored": item["id"], "user": identity.user_id}

        @app.get("/v1/companion/memory")
        async def companion_memory(request: Request) -> dict:
            """A user sees only their OWN memory — identity from the key."""
            _, record = await self._requester_from(request)
            identity = resolve_identity(record.address if record else "")
            memory = UserMemory(self.memory, identity)
            items = [{k: v for k, v in it.items() if k != "vec"}
                     for it in memory.all_items()]
            return {"user": identity.user_id, "authenticated": identity.authenticated,
                    "items": items}

        @app.post("/v1/swarm")
        async def swarm(body: dict, request: Request) -> dict:
            """Fan a task across several distinct models in parallel — each
            lands on whichever node serves it, so the work spreads across
            the network. Modes: fanout (same task, diverse answers),
            map (one task per worker), vote (same task + synthesized pick)."""
            payer, record = await self._requester_from(request)
            task = (body.get("task") or "").strip()
            tasks = body.get("tasks") or ([task] if task else [])
            if not tasks:
                raise HTTPException(400, "task or tasks required")
            mode = body.get("mode", "fanout")
            max_tokens = int(body.get("max_tokens", 512))
            await self.router.refresh(self.node_urls)

            models = body.get("models") or self._distinct_models(int(body.get("n", 3)))
            if not models:
                raise HTTPException(503, "no live models to swarm across")
            await self._precheck_funds(record, " ".join(tasks), max_tokens * (len(models) + 1))

            # Build the (worker -> task) plan.
            if mode == "map":
                plan_items = [(models[i % len(models)], t) for i, t in enumerate(tasks)]
            else:  # fanout / vote: same task to each distinct model
                plan_items = [(m, tasks[0]) for m in models]

            async def one(model: str, t: str) -> dict:
                try:
                    r = await self.router.route(
                        prompt=f"user: {t}\nassistant:", model=model,
                        max_tokens=max_tokens, requester=payer)
                    used = r.tokens_in + r.tokens_out
                    if record is not None:
                        self.keys.record_usage(record, used, used * self.params.price_per_token)
                    return {"model": model.split("::")[0], "node": r.node,
                            "answer": r.text, "receipt_tx": r.receipt_tx}
                except RouteError as exc:
                    return {"model": model.split("::")[0], "error": str(exc)}

            workers = await asyncio.gather(*(one(m, t) for m, t in plan_items))
            answers = [w for w in workers if "answer" in w]

            synthesized = None
            if mode == "vote" and answers:
                joined = "\n\n".join(f"[{i+1}] {w['answer']}" for i, w in enumerate(answers))
                syn_prompt = (f"user: {tasks[0]}\n\n"
                              f"Independent answers from {len(answers)} models:\n{joined}\n\n"
                              "Synthesize the single best, correct answer.\nassistant:")
                try:
                    r = await self.router.route(prompt=syn_prompt, model="auto",
                                                max_tokens=max_tokens, requester=payer)
                    synthesized = r.text
                    if record is not None:
                        used = r.tokens_in + r.tokens_out
                        self.keys.record_usage(record, used, used * self.params.price_per_token)
                except RouteError:
                    pass

            return {"mode": mode, "workers": workers,
                    "models_used": [m for m, _ in plan_items],
                    "answer": synthesized}

        @app.get("/v1/memory")
        async def memory_view(session: str, q: str = "", k: int = 8) -> dict:
            if q:
                items = await self.memory.asearch(session, q, k)
            else:
                items = self.memory.session_items(session)[-k:]
            return {"session": session,
                    "items": [{key: v for key, v in it.items() if key != "vec"}
                              for it in items]}

        @app.post("/v1/images/generations")
        async def images_generations(body: dict, request: Request) -> dict:
            """OpenAI-compatible text-to-image; routes to diffusion nodes."""
            payer, record = await self._requester_from(request)
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                raise HTTPException(400, "prompt required")
            size = str(body.get("size", "512x512"))
            try:
                w, h = (int(x) for x in size.lower().split("x"))
            except ValueError:
                w, h = 512, 512
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, prompt, w * h // 4096 + 64)
            try:
                result = await self.router.route_image(
                    prompt, model=body.get("model", "auto"), width=w, height=h,
                    seed=int(body.get("seed", 0)), requester=payer)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            if record is not None:
                units = w * h // 4096 + 64
                self.keys.record_usage(record, units, units * self.params.price_per_token)
            return {
                "created": 0,
                "data": [{"b64_json": img} for img in result.images],
                "model": result.model_ref,
                "deltav": {"node": result.node, "receipt_tx": result.receipt_tx},
            }

        @app.post("/v1/embeddings")
        async def embeddings(body: dict, request: Request) -> dict:
            payer, record = await self._requester_from(request)
            raw = body.get("input")
            texts = [raw] if isinstance(raw, str) else [str(t) for t in (raw or [])]
            if not texts:
                raise HTTPException(400, "input required")
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, " ".join(texts), 16 * len(texts))
            try:
                result = await self.router.route_embed(texts, model=body.get("model", "auto"),
                                                       requester=payer)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            if record is not None:
                self.keys.record_usage(record, result.tokens,
                                       result.tokens * self.params.price_per_token)
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

        # ------------------------------------------------------ Ollama API
        @app.get("/api/version")
        async def ollama_version() -> dict:
            return {"version": "deltav-0.1.0"}

        @app.get("/api/tags")
        async def ollama_tags() -> dict:
            await self.router.refresh(self.node_urls)
            served = {m for n in self.router.nodes if n.alive for m in n.models}
            return ol.tags_payload(self.catalog.specs, served)

        def _served_refs() -> list[str]:
            return [m for n in self.router.nodes if n.alive and n.active for m in n.models]

        @app.post("/api/chat")
        async def ollama_chat(body: dict, request: Request):
            payer, record = await self._requester_from(request)
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            prompt = ol.chat_messages_to_prompt(messages)
            opts = body.get("options") or {}
            max_tokens = int(opts.get("num_predict", 512) if opts.get("num_predict", 512) > 0 else 512)
            await self.router.refresh(self.node_urls)
            model = ol.resolve_model(body.get("model", "auto"), _served_refs(), self.catalog)
            tag = body.get("model", "auto")
            await self._precheck_funds(record, prompt, max_tokens)
            stream = body.get("stream", True)  # Ollama defaults to streaming
            return await self._ollama_run(payer, record, model, tag, prompt, max_tokens,
                                          float(opts.get("temperature", 0.0)), stream, kind="chat")

        @app.post("/api/generate")
        async def ollama_generate(body: dict, request: Request):
            payer, record = await self._requester_from(request)
            prompt_in = body.get("prompt", "")
            if not prompt_in:
                raise HTTPException(400, "prompt required")
            prompt = f"user: {prompt_in}\nassistant:"
            opts = body.get("options") or {}
            max_tokens = int(opts.get("num_predict", 512) if opts.get("num_predict", 512) > 0 else 512)
            await self.router.refresh(self.node_urls)
            model = ol.resolve_model(body.get("model", "auto"), _served_refs(), self.catalog)
            tag = body.get("model", "auto")
            await self._precheck_funds(record, prompt, max_tokens)
            stream = body.get("stream", True)
            return await self._ollama_run(payer, record, model, tag, prompt, max_tokens,
                                          float(opts.get("temperature", 0.0)), stream,
                                          kind="generate")

        @app.post("/api/embeddings")
        async def ollama_embeddings(body: dict, request: Request) -> dict:
            payer, record = await self._requester_from(request)
            text = body.get("prompt", "")
            if not text:
                raise HTTPException(400, "prompt required")
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, text, 16)
            try:
                result = await self.router.route_embed([text], model="auto", requester=payer)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            if record is not None:
                self.keys.record_usage(record, result.tokens,
                                       result.tokens * self.params.price_per_token)
            return {"embedding": result.vectors[0]}

        @app.post("/api/embed")
        async def ollama_embed(body: dict, request: Request) -> dict:
            payer, record = await self._requester_from(request)
            raw = body.get("input", "")
            texts = [raw] if isinstance(raw, str) else [str(t) for t in raw]
            if not texts or not texts[0]:
                raise HTTPException(400, "input required")
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, " ".join(texts), 16 * len(texts))
            try:
                result = await self.router.route_embed(texts, model="auto", requester=payer)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            if record is not None:
                self.keys.record_usage(record, result.tokens,
                                       result.tokens * self.params.price_per_token)
            return {"model": body.get("model", "auto"), "embeddings": result.vectors}

        @app.post("/v1/messages")
        async def anthropic_messages(body: dict, request: Request):
            """Anthropic Messages API — Claude-native agents connect directly."""
            payer, record = await self._requester_from(request)
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            tools = body.get("tools") or None
            prompt = messages_to_prompt(body.get("system", ""), messages, tools)
            max_tokens = int(body.get("max_tokens", 1024))
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(record, prompt, max_tokens)

            if body.get("stream") and not tools:
                try:
                    spec = self.router.resolve_model(body.get("model", "auto"))
                    upstream = self.router.route_stream(
                        prompt=prompt, model=spec.ref, max_tokens=max_tokens,
                        temperature=float(body.get("temperature") or 0.0),
                        requester=payer)
                    first = await upstream.__anext__()
                except (RouteError, StopAsyncIteration) as exc:
                    raise HTTPException(503, str(exc))

                holder: dict = {}

                async def pieces():
                    kind, value = first
                    if kind == "token" and value:
                        yield value
                    async for k, v in upstream:
                        if k == "token" and v:
                            yield v
                        elif k == "final":
                            total = v.tokens_in + v.tokens_out
                            holder.update(tokens_in=v.tokens_in, tokens_out=v.tokens_out,
                                          meta={"node": v.node, "receipt_tx": v.receipt_tx})
                            if record is not None:
                                self.keys.record_usage(record, total,
                                                       total * self.params.price_per_token)

                return StreamingResponse(
                    anthropic_text_stream(spec.ref, pieces(), holder),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

            try:
                result = await self.router.route(
                    prompt=prompt, model=body.get("model", "auto"), max_tokens=max_tokens,
                    temperature=float(body.get("temperature") or 0.0), requester=payer)
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            total = result.tokens_in + result.tokens_out
            if record is not None:
                self.keys.record_usage(record, total, total * self.params.price_per_token)
            msg = to_anthropic_message(
                result.text, result.model_ref, result.tokens_in, result.tokens_out,
                {"node": result.node, "receipt_tx": result.receipt_tx}, bool(tools))
            if body.get("stream"):
                return StreamingResponse(
                    to_anthropic_tool_stream(msg), media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"})
            return msg

        @app.post("/v1/chat/completions")
        async def chat_completions(body: dict, request: Request):
            payer, record = await self._requester_from(request)
            messages = body.get("messages") or []
            if not messages:
                raise HTTPException(400, "messages required")
            tools = body.get("tools") or None
            prompt = render_prompt(messages, tools)
            await self.router.refresh(self.node_urls)
            await self._precheck_funds(
                record, prompt,
                int(body.get("max_tokens") or body.get("max_completion_tokens") or 256))

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
                        requester=payer,
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
                    requester=payer,
                )
            except RouteError as exc:
                raise HTTPException(503, str(exc))
            if record is not None:
                total = result.tokens_in + result.tokens_out
                self.keys.record_usage(record, total, total * self.params.price_per_token)

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
