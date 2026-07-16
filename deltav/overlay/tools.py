"""Tool registry + built-in tools (web_search, fetch_url, calculator).

Handlers are async callables taking kwargs and returning a string — the
string goes back into the model's context verbatim.
"""
from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import httpx

from .search import SearchEngine, format_results


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON Schema
    handler: Callable[..., Awaitable[str]]


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def specs_openai(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> str:
        spec = self._tools.get(name)
        if spec is None:
            return f"error: unknown tool {name!r} (available: {', '.join(self.names())})"
        try:
            return await spec.handler(**arguments)
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}"
        except Exception as exc:  # tool failures are model-visible, not fatal
            return f"error: {name} failed: {exc}"


# ------------------------------------------------------------ calculator

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        # A tiny expression like 9**9**9 has no long digit-runs but produces an
        # astronomically large int that hangs the event loop — bound the power.
        if type(node.op) is ast.Pow and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:40]}")


def calculate(expression: str) -> str:
    if len(expression) > 200 or "**" in expression and any(
        len(part) > 6 for part in re.findall(r"\d+", expression)
    ):
        raise ValueError("expression too large")
    result = _safe_eval(ast.parse(expression, mode="eval"))
    return repr(result)


# -------------------------------------------------------------- builtins

_TAG_RE = re.compile(r"<script.*?</script>|<style.*?</style>|<[^>]+>", re.S | re.I)
_WS_RE = re.compile(r"\s+")


def builtin_registry(client: httpx.AsyncClient) -> ToolRegistry:
    engine = SearchEngine(client)

    async def web_search(query: str, max_results: int = 5) -> str:
        return format_results(await engine.search(query, int(max_results)))

    async def fetch_url(url: str, max_chars: int = 4000) -> str:
        # SSRF guard: this runs server-side in the agent/companion loop and
        # returns the body to the caller, so it must never reach internal
        # hosts, cloud metadata, or non-web ports. Follow redirects manually,
        # re-screening every hop (an external 302 can point back inside).
        from ..net.security import screen_url

        for _ in range(5):
            reason = await screen_url(url, allow_private=False, allow_ports={80, 443})
            if reason:
                return f"error: url blocked ({reason})"
            resp = await client.get(url, headers={"User-Agent": "deltav-agent/0.1"},
                                    timeout=20.0, follow_redirects=False)
            if resp.is_redirect and resp.headers.get("location"):
                url = str(resp.url.join(resp.headers["location"]))
                continue
            resp.raise_for_status()
            text = _WS_RE.sub(" ", _TAG_RE.sub(" ", resp.text)).strip()
            return text[: int(max_chars)]
        return "error: too many redirects"

    async def calculator(expression: str) -> str:
        return calculate(str(expression))

    return ToolRegistry([
        ToolSpec(
            name="web_search",
            description="Search the internet. Returns titles, URLs and snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            handler=web_search,
        ),
        ToolSpec(
            name="fetch_url",
            description="Fetch a web page and return its readable text.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 4000},
                },
                "required": ["url"],
            },
            handler=fetch_url,
        ),
        ToolSpec(
            name="calculator",
            description="Evaluate an arithmetic expression (+ - * / // % ** and parentheses).",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
            handler=calculator,
        ),
    ])
