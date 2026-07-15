"""Ollama-compatible API surface.

Open-source local-model tools (Open WebUI, LangChain's Ollama backend,
many desktop apps) speak Ollama's `/api/*` dialect. Exposing it lets the
whole network present itself as a drop-in Ollama server — the natural
shape for the open models we serve. NDJSON streaming, not SSE.

Pure helpers here; the gateway wires them to routing + billing.
"""
from __future__ import annotations

import json

from ..router.catalog import Catalog, ModelSpec


def short_name(ref: str) -> str:
    """Human/ollama-ish short name from a model ref."""
    base = ref.split("::", 1)[0].split("/")[-1]
    for suffix in ("-GGUF", "-gguf"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base.lower()


def ollama_tag(spec: ModelSpec) -> str:
    """e.g. 'qwen2.5-7b-instruct:q4_k_m'."""
    return f"{short_name(spec.ref)}:{spec.quant.lower()}"


def resolve_model(name: str, served_refs: list[str], catalog: Catalog) -> str:
    """Map an Ollama-style model name back to a network model ref.

    Accepts 'auto', a full ref, an ollama tag, or a loose short name —
    prefers something currently served, else the catalog, else 'auto'."""
    if not name or name.lower() == "auto":
        return "auto"
    if name in served_refs or catalog.by_ref(name) is not None:
        return name
    wanted = name.split(":", 1)[0].lower()

    def matches(ref: str) -> bool:
        s = short_name(ref)
        return s == wanted or s.startswith(wanted) or wanted in s

    for ref in served_refs:              # prefer a live-served model
        if matches(ref):
            return ref
    for spec in catalog.specs:           # then the catalog
        if spec.kind == "chat" and matches(spec.ref):
            return spec.ref
    return "auto"


def tags_payload(specs: list[ModelSpec], served_refs: set[str]) -> dict:
    """/api/tags — list models Ollama-style (served ones marked)."""
    models = []
    for spec in specs:
        if spec.kind != "chat":
            continue
        models.append({
            "name": ollama_tag(spec),
            "model": ollama_tag(spec),
            "modified_at": "1970-01-01T00:00:00Z",
            "size": spec.file_mb * 1024 * 1024,
            "digest": "",
            "details": {
                "family": spec.family,
                "parameter_size": f"{spec.params_b}B",
                "quantization_level": spec.quant,
            },
            "deltav": {"ref": spec.ref, "served": spec.ref in served_refs},
        })
    return {"models": models}


def chat_messages_to_prompt(messages: list[dict]) -> str:
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
    return "\n".join(lines) + "\nassistant:"


def chat_response(model_tag: str, text: str, tokens_in: int, tokens_out: int,
                  meta: dict) -> dict:
    return {
        "model": model_tag,
        "created_at": "1970-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": tokens_in,
        "eval_count": tokens_out,
        "deltav": meta,
    }


def generate_response(model_tag: str, text: str, tokens_in: int, tokens_out: int,
                      meta: dict) -> dict:
    return {
        "model": model_tag,
        "created_at": "1970-01-01T00:00:00Z",
        "response": text,
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": tokens_in,
        "eval_count": tokens_out,
        "deltav": meta,
    }


def ndjson(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"


async def chat_stream(model_tag: str, pieces_iter, final_holder: dict):
    """Ollama chat NDJSON stream: one object per chunk, final has done=true."""
    async for piece in pieces_iter:
        if piece:
            yield ndjson({"model": model_tag, "created_at": "1970-01-01T00:00:00Z",
                          "message": {"role": "assistant", "content": piece}, "done": False})
    yield ndjson({
        "model": model_tag, "created_at": "1970-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": ""}, "done": True,
        "done_reason": "stop",
        "prompt_eval_count": final_holder.get("tokens_in", 0),
        "eval_count": final_holder.get("tokens_out", 0),
        "deltav": final_holder.get("meta", {}),
    })


async def generate_stream(model_tag: str, pieces_iter, final_holder: dict):
    async for piece in pieces_iter:
        if piece:
            yield ndjson({"model": model_tag, "created_at": "1970-01-01T00:00:00Z",
                          "response": piece, "done": False})
    yield ndjson({
        "model": model_tag, "created_at": "1970-01-01T00:00:00Z", "response": "",
        "done": True, "done_reason": "stop",
        "prompt_eval_count": final_holder.get("tokens_in", 0),
        "eval_count": final_holder.get("tokens_out", 0),
    })
