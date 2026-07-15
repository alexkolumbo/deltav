"""Client side: Anthropic Messages API, swarm, client SDK + profiles."""
import asyncio
import json

import httpx
import pytest

from deltav.compute.base import DeviceInfo
from deltav.config import DVT, Genesis
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.gateway.anthropic import messages_to_prompt, to_anthropic_message
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
MODEL2 = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF::Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
GW = "http://127.0.0.1:9700"


# ------------------------------------------------------ anthropic translation

def test_messages_to_prompt_flattens_blocks():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "42"}]},
    ]
    prompt = messages_to_prompt("be brief", msgs, None)
    assert "system: be brief" in prompt
    assert "user: hi" in prompt
    assert "assistant: hello" in prompt
    assert "tool result: 42" in prompt
    assert prompt.endswith("assistant:")


def test_messages_to_prompt_injects_tools():
    prompt = messages_to_prompt("", [{"role": "user", "content": "x"}],
                                [{"name": "lookup", "description": "find",
                                  "input_schema": {"type": "object"}}])
    assert "lookup" in prompt and "<tool_call>" in prompt


def test_to_anthropic_message_text():
    msg = to_anthropic_message("plain answer", MODEL, 5, 7, {"node": "n"}, has_tools=False)
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["content"][0] == {"type": "text", "text": "plain answer"}
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_to_anthropic_message_tool_use():
    text = ('thinking <tool_call>{"name": "lookup", "arguments": {"q": "x"}}</tool_call>')
    msg = to_anthropic_message(text, MODEL, 5, 7, {}, has_tools=True)
    assert msg["stop_reason"] == "tool_use"
    blocks = {b["type"] for b in msg["content"]}
    assert "tool_use" in blocks
    tu = next(b for b in msg["content"] if b["type"] == "tool_use")
    assert tu["name"] == "lookup" and tu["input"] == {"q": "x"}


# ------------------------------------------------------------------ e2e

@pytest.fixture
async def net(genesis, alice, bob, carol):
    transport = MultiTransport()
    urls = ["http://127.0.0.1:9701", "http://127.0.0.1:9702"]
    keys = [alice, bob]
    # two nodes, each serving a different model -> a swarm spreads across them
    modelsets = [[MODEL], [MODEL2]]
    daemons = []
    for i, kp in enumerate(keys):
        cfg = NodeConfig(port=9701 + i, endpoint=urls[i], peers=[urls[1 - i]],
                         backend="mock", models=modelsets[i], max_parallel_jobs=4,
                         device=DeviceInfo(vendor="nvidia", name="t", vram_mb=12282))
        d = NodeDaemon(kp, genesis, cfg, client=httpx.AsyncClient(transport=transport))
        transport.add(urls[i], d.app)
        daemons.append(d)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=urls, params=genesis.params, client=client)
    transport.add(GW, gateway.app)
    for d in daemons:
        await d.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        n = (await client.get(f"{urls[0]}/chain/nodes")).json()["nodes"]
        if len(n) == 2:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "gateway": gateway, "urls": urls}
    finally:
        for d in daemons:
            await d.stop()
        await client.aclose()


async def test_anthropic_messages_endpoint(net):
    client = net["client"]
    resp = await client.post(f"{GW}/v1/messages", json={
        "model": "auto", "max_tokens": 32,
        "messages": [{"role": "user", "content": "hello claude-style"}],
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    msg = resp.json()
    assert msg["type"] == "message" and msg["role"] == "assistant"
    assert msg["content"][0]["type"] == "text"
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] > 0
    assert msg["deltav"]["receipt_tx"]  # billed like any request


async def test_anthropic_streaming_events(net):
    client = net["client"]
    events = []
    async with client.stream("POST", f"{GW}/v1/messages", json={
        "model": "auto", "max_tokens": 24, "stream": True,
        "messages": [{"role": "user", "content": "stream anthropic"}],
    }, timeout=30.0) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                events.append(line[len("event: "):])
    # the Anthropic event sequence
    assert events[0] == "message_start"
    assert "content_block_delta" in events
    assert events[-1] == "message_stop"


async def test_swarm_spreads_across_models_and_nodes(net):
    client = net["client"]
    resp = await client.post(f"{GW}/v1/swarm", json={
        "task": "что такое дельта-в?", "n": 2, "mode": "vote", "max_tokens": 32,
    }, timeout=60.0)
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert len(d["workers"]) == 2
    used_models = {w["model"] for w in d["workers"] if "answer" in w}
    used_nodes = {w["node"] for w in d["workers"] if "answer" in w}
    assert len(used_models) == 2      # two distinct models
    assert len(used_nodes) == 2       # served by two distinct nodes
    assert d["answer"]                # vote synthesized a final answer


async def test_swarm_map_mode(net):
    client = net["client"]
    resp = await client.post(f"{GW}/v1/swarm", json={
        "tasks": ["задача А", "задача Б", "задача В"], "mode": "map", "max_tokens": 24,
    }, timeout=60.0)
    d = resp.json()
    assert len(d["workers"]) == 3     # one worker per task


# ------------------------------------------------------------ client SDK

def test_client_failover_skips_dead_gateway():
    from deltav.client import DeltaVClient

    def handler(request: httpx.Request) -> httpx.Response:
        if "dead" in str(request.url):
            raise httpx.ConnectError("down", request=request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"gateway": "dv1live", "nodes": []})
        return httpx.Response(404)

    c = DeltaVClient(base_urls=["http://dead:9000", "http://live:9000"],
                     client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.health()["gateway"] == "dv1live"


def test_profile_roundtrip(tmp_path):
    from deltav.client import Profile, load_profile, save_profile

    p = Profile(base_urls=["http://a:9000", "http://b:9000"], api_key="dvk_x", model="auto")
    path = save_profile(p, tmp_path / "client.json")
    again = load_profile(path)
    assert again.base_urls == p.base_urls and again.api_key == "dvk_x"


def test_profile_migrates_single_base_url(tmp_path):
    from deltav.client import load_profile

    path = tmp_path / "client.json"
    path.write_text(json.dumps({"base_url": "http://old:9000", "api_key": "k"}))
    p = load_profile(path)
    assert p.base_urls == ["http://old:9000"]  # legacy single-URL config still loads
