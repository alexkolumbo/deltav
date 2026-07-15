"""Overlay services on top of the raw inference network:
tool calling, an internet search engine, and server-side agents."""
from .agent import Agent, AgentResult, AgentStep
from .search import SearchEngine
from .toolcall import (
    build_tool_system_prompt,
    parse_tool_calls,
    render_conversation,
    strip_tool_calls,
    to_openai_tool_calls,
)
from .tools import ToolRegistry, ToolSpec, builtin_registry

__all__ = [
    "Agent", "AgentResult", "AgentStep",
    "SearchEngine",
    "ToolRegistry", "ToolSpec", "builtin_registry",
    "build_tool_system_prompt", "parse_tool_calls", "render_conversation",
    "strip_tool_calls", "to_openai_tool_calls",
]
