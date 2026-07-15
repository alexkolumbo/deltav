"""Tool-calling protocol over raw text completions.

Models on the network are plain completion endpoints, so tool calling is
a prompt convention (the Hermes/Qwen `<tool_call>` format most open
instruct models were trained on) plus robust parsing on the way out.
The gateway translates between this and the OpenAI `tools`/`tool_calls`
dialect that client software speaks.
"""
from __future__ import annotations

import json
import re
import uuid

# Content-agnostic inner match: nested braces inside JSON strings would
# defeat a non-greedy {...} pattern; json.loads is the actual validator.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[dict]:
    """Extract {"name", "arguments"} dicts; malformed JSON is skipped."""
    calls = []
    for raw in _TOOL_CALL_RE.findall(text):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        name = data.get("name")
        if not isinstance(name, str):
            continue
        arguments = data.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append({"name": name, "arguments": arguments})
    return calls


def strip_tool_calls(text: str) -> str:
    return _TOOL_CALL_RE.sub("", text).strip()


def to_openai_tool_calls(calls: list[dict]) -> list[dict]:
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:16]}",
            "type": "function",
            "function": {
                "name": c["name"],
                "arguments": json.dumps(c["arguments"], ensure_ascii=False),
            },
        }
        for c in calls
    ]


def build_tool_system_prompt(tools_openai: list[dict]) -> str:
    specs = json.dumps(
        [t.get("function", t) for t in tools_openai], ensure_ascii=False, indent=None
    )
    return (
        "You can call tools. Available tools (JSON Schema):\n"
        f"{specs}\n"
        "To call a tool, reply with EXACTLY this format and nothing else:\n"
        '<tool_call>{"name": "<tool-name>", "arguments": {<args>}}</tool_call>\n'
        "After a tool result arrives, either call another tool or give the final answer "
        "as plain text without any <tool_call> tags."
    )


def render_conversation(messages: list[dict]) -> str:
    """Render an OpenAI-style message list (incl. tool traffic) to a prompt."""
    lines = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            name = m.get("name", m.get("tool_call_id", "tool"))
            lines.append(f"tool ({name}): {m.get('content', '')}")
            continue
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            calls = "".join(
                "<tool_call>"
                + json.dumps({
                    "name": c["function"]["name"],
                    "arguments": json.loads(c["function"].get("arguments") or "{}"),
                }, ensure_ascii=False)
                + "</tool_call>"
                for c in m["tool_calls"]
            )
            content = (content + "\n" + calls).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines) + "\nassistant:"
