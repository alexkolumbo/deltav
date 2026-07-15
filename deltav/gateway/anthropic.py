"""Anthropic Messages API compatibility.

Lets Claude-native agent software (OpenClaw, Claude-Code-style tools,
Hermes when it speaks Anthropic, anything using the `anthropic` SDK)
point straight at a Delta V gateway — no LiteLLM shim. We translate the
Anthropic Messages dialect to/from the network's raw-completion + tool
protocol, including SSE in Anthropic's event format.

Pure functions here; the gateway wires them to routing + billing.
"""
from __future__ import annotations

import json
import time
import uuid

from ..overlay import build_tool_system_prompt, parse_tool_calls, strip_tool_calls


def _block_text(content) -> str:
    """Flatten Anthropic message content (str or list of blocks) to text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        t = block.get("type")
        if t == "text":
            parts.append(block.get("text", ""))
        elif t == "tool_use":
            parts.append("<tool_call>" + json.dumps(
                {"name": block.get("name"), "arguments": block.get("input", {})},
                ensure_ascii=False) + "</tool_call>")
        elif t == "tool_result":
            inner = block.get("content", "")
            if isinstance(inner, list):
                inner = " ".join(b.get("text", "") for b in inner if isinstance(b, dict))
            parts.append(f"tool result: {inner}")
    return "\n".join(parts)


def _tools_openai(tools: list[dict] | None) -> list[dict]:
    """Anthropic tool defs -> the OpenAI-ish shape our tool prompt expects."""
    out = []
    for t in tools or []:
        out.append({"type": "function", "function": {
            "name": t.get("name"),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
        }})
    return out


def messages_to_prompt(system: str, messages: list[dict], tools: list[dict] | None) -> str:
    """Render an Anthropic request into the network's prompt convention."""
    lines: list[str] = []
    sys_text = system or ""
    if tools:
        tool_sys = build_tool_system_prompt(_tools_openai(tools))
        sys_text = (sys_text + "\n\n" + tool_sys).strip() if sys_text else tool_sys
    if sys_text:
        lines.append(f"system: {sys_text}")
    for m in messages:
        role = m.get("role", "user")
        lines.append(f"{role}: {_block_text(m.get('content'))}")
    return "\n".join(lines) + "\nassistant:"


def to_anthropic_message(text: str, model: str, tokens_in: int, tokens_out: int,
                         meta: dict, has_tools: bool) -> dict:
    """Build a non-streaming Anthropic message response from model text."""
    calls = parse_tool_calls(text) if has_tools else []
    if calls:
        content = []
        plain = strip_tool_calls(text)
        if plain:
            content.append({"type": "text", "text": plain})
        for c in calls:
            content.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                "name": c["name"],
                "input": c["arguments"],
            })
        stop_reason = "tool_use"
    else:
        content = [{"type": "text", "text": text}]
        stop_reason = "end_turn"
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": tokens_in, "output_tokens": tokens_out},
        "deltav": meta,
    }


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def anthropic_text_stream(model: str, pieces_iter, final_holder: dict):
    """SSE in Anthropic event format for a text answer.

    `pieces_iter` yields text chunks; `final_holder` is filled with
    {'tokens_in','tokens_out','meta'} by the caller's async generator
    before it completes (so usage lands in message_delta)."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": model,
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": 0, "output_tokens": 0}}})
    yield _sse("content_block_start", {"type": "content_block_start", "index": 0,
                                       "content_block": {"type": "text", "text": ""}})
    yield _sse("ping", {"type": "ping"})
    async for piece in pieces_iter:
        if piece:
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                       "delta": {"type": "text_delta", "text": piece}})
    yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    usage = {"output_tokens": final_holder.get("tokens_out", 0),
             "input_tokens": final_holder.get("tokens_in", 0)}
    yield _sse("message_delta", {"type": "message_delta",
               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
               "usage": usage})
    yield _sse("message_stop", {"type": "message_stop", "deltav": final_holder.get("meta", {})})


def to_anthropic_tool_stream(msg: dict):
    """SSE for a buffered response that turned out to contain tool_use —
    emit each content block as complete Anthropic stream blocks."""
    msg_id = msg["id"]
    yield _sse("message_start", {"type": "message_start", "message": {
        "id": msg_id, "type": "message", "role": "assistant", "model": msg["model"],
        "content": [], "stop_reason": None,
        "usage": {"input_tokens": msg["usage"]["input_tokens"], "output_tokens": 0}}})
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "text_delta", "text": block["text"]}})
        else:  # tool_use
            yield _sse("content_block_start", {"type": "content_block_start", "index": i,
                       "content_block": {"type": "tool_use", "id": block["id"],
                                         "name": block["name"], "input": {}}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": i,
                       "delta": {"type": "input_json_delta",
                                 "partial_json": json.dumps(block["input"], ensure_ascii=False)}})
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    yield _sse("message_delta", {"type": "message_delta",
               "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
               "usage": {"output_tokens": msg["usage"]["output_tokens"]}})
    yield _sse("message_stop", {"type": "message_stop"})
