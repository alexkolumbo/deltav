"""Phase 4: tool calling, search parsing, agent loop, gateway overlay APIs."""
import asyncio
import json

import httpx
import pytest

from deltav.compute.base import DeviceInfo
from deltav.config import DVT, Genesis
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.overlay import (
    Agent,
    ToolRegistry,
    parse_tool_calls,
    render_conversation,
    strip_tool_calls,
)
from deltav.overlay.search import parse_ddg
from deltav.overlay.tools import ToolSpec, calculate

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9501"
GW_URL = "http://127.0.0.1:9500"


# ------------------------------------------------------------ tool calls

def test_parse_tool_calls():
    text = ('thinking...\n<tool_call>{"name": "web_search", '
            '"arguments": {"query": "deltav"}}</tool_call>')
    calls = parse_tool_calls(text)
    assert calls == [{"name": "web_search", "arguments": {"query": "deltav"}}]
    assert strip_tool_calls(text) == "thinking..."


def test_parse_tool_calls_multiple_and_malformed():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            "<tool_call>{broken json</tool_call>"
            '<tool_call>{"name": "b", "arguments": "{\\"x\\": 1}"}</tool_call>'
            '<tool_call>{"arguments": {}}</tool_call>')  # no name
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == {"x": 1}  # string arguments get parsed


def test_render_conversation_with_tool_traffic():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"q": "x"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "result!"},
    ]
    prompt = render_conversation(messages)
    assert '"name": "lookup"' in prompt
    assert "tool (lookup): result!" in prompt
    assert prompt.endswith("assistant:")


# ------------------------------------------------------------ calculator

def test_calculator():
    assert calculate("2 + 2 * 2") == "6"
    assert calculate("(10 - 4) / 3") == "2.0"
    assert calculate("2 ** 10") == "1024"


def test_calculator_rejects_code():
    with pytest.raises((ValueError, SyntaxError)):
        calculate("__import__('os').system('x')")
    with pytest.raises((ValueError, SyntaxError)):
        calculate("open('f')")


# ------------------------------------------------------------ search parse

DDG_PAGE = """
<div class="result">
<a rel="nofollow" class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Fdv&amp;rut=x">Delta V <b>network</b></a>
<a class="result__snippet" href="/l/?uddg=x">Decentralized <b>AI</b> inference.</a>
</div>
"""


def test_parse_ddg_unwraps_and_cleans():
    results = parse_ddg(DDG_PAGE, 5)
    assert results == [{
        "title": "Delta V network",
        "url": "https://example.com/dv",
        "snippet": "Decentralized AI inference.",
    }]


# ------------------------------------------------------------ registry

def stub_registry() -> ToolRegistry:
    async def lookup(query: str) -> str:
        return f"stub result for {query}"

    async def boom() -> str:
        raise RuntimeError("kaput")

    schema = {"type": "object", "properties": {"query": {"type": "string"}},
              "required": ["query"]}
    return ToolRegistry([
        ToolSpec("lookup", "test lookup", schema, lookup),
        ToolSpec("boom", "always fails", {"type": "object", "properties": {}}, boom),
    ])


async def test_registry_execute_and_errors():
    reg = stub_registry()
    assert await reg.execute("lookup", {"query": "dv"}) == "stub result for dv"
    assert "unknown tool" in await reg.execute("nope", {})
    assert "bad arguments" in await reg.execute("lookup", {"wrong": 1})
    assert "kaput" in await reg.execute("boom", {})


# ------------------------------------------------------------ agent loop

async def test_agent_loop_two_steps():
    replies = iter([
        '<tool_call>{"name": "lookup", "arguments": {"query": "delta"}}</tool_call>',
        "The answer is 42.",
    ])

    async def complete(prompt: str):
        return next(replies), {"node": "dv1node", "receipt_tx": "rcpt1"}

    agent = Agent(complete, stub_registry(), max_steps=4)
    result = await agent.run("what is delta?")
    assert result.answer == "The answer is 42."
    assert result.model_calls == 2 and result.finished
    assert len(result.steps) == 1
    step = result.steps[0]
    assert step.tool == "lookup" and step.result == "stub result for delta"
    assert step.receipt_tx == "rcpt1"


async def test_agent_step_limit():
    async def complete(prompt: str):
        return '<tool_call>{"name": "lookup", "arguments": {"query": "x"}}</tool_call>', {}

    agent = Agent(complete, stub_registry(), max_steps=3)
    result = await agent.run("loop forever")
    assert not result.finished
    assert len(result.steps) == 3


# ------------------------------------------------------- gateway overlay e2e

@pytest.fixture
async def overlay_net(genesis, alice, carol):
    transport = MultiTransport()
    cfg = NodeConfig(port=9501, endpoint=URL, backend="mock", models=[MODEL],
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params,
                            client=client, tools=stub_registry())
    transport.add(GW_URL, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "daemon": daemon}
    finally:
        await daemon.stop()
        await client.aclose()


TOOLS_OPENAI = [{
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "test lookup",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]},
    },
}]

SCRIPTED = ('[[reply]]<tool_call>{"name": "lookup", "arguments": {"query": "delta"}}'
            "</tool_call>[[/reply]]"
            "[[reply]]Final answer: 42[[/reply]] what is delta?")


async def test_gateway_tool_calls_roundtrip(overlay_net):
    client = overlay_net["client"]
    body = {"model": "auto", "tools": TOOLS_OPENAI, "max_tokens": 64,
            "messages": [{"role": "user", "content": SCRIPTED}]}
    resp = (await client.post(f"{GW_URL}/v1/chat/completions", json=body, timeout=30.0)).json()
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"query": "delta"}

    # client executes the tool and calls back -> final answer
    body["messages"] += [
        choice["message"],
        {"role": "tool", "tool_call_id": call["id"], "name": "lookup",
         "content": "stub result for delta"},
    ]
    resp2 = (await client.post(f"{GW_URL}/v1/chat/completions", json=body, timeout=30.0)).json()
    choice2 = resp2["choices"][0]
    assert choice2["finish_reason"] == "stop"
    assert choice2["message"]["content"] == "Final answer: 42"


async def test_gateway_agent_run(overlay_net):
    client = overlay_net["client"]
    resp = (await client.post(f"{GW_URL}/v1/agents/run", json={
        "task": SCRIPTED, "max_steps": 4, "max_tokens": 64,
    }, timeout=60.0)).json()
    assert resp["answer"] == "Final answer: 42"
    assert resp["finished"] and resp["model_calls"] == 2
    assert len(resp["steps"]) == 1
    step = resp["steps"][0]
    assert step["tool"] == "lookup"
    assert step["result"] == "stub result for delta"
    # every reasoning step is a paid, spot-checkable inference on the chain
    assert step["receipt_tx"]

    # the receipt actually lands on-chain
    deadline = asyncio.get_event_loop().time() + 8.0
    receipts = []
    while asyncio.get_event_loop().time() < deadline:
        receipts = (await client.get(f"{URL}/chain/receipts")).json()["receipts"]
        if len(receipts) >= 2:
            break
        await asyncio.sleep(0.05)
    assert any(r["receipt_hash"] for r in receipts)


async def test_gateway_search_endpoint_shape(overlay_net):
    client = overlay_net["client"]
    resp = await client.get(f"{GW_URL}/v1/search", params={"q": "delta v"})
    assert resp.status_code == 200
    data = resp.json()
    # providers are unreachable through the test transport -> empty but well-formed
    assert data["query"] == "delta v"
    assert isinstance(data["results"], list)
