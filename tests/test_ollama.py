"""Ollama-compatible API surface."""
import asyncio
import json

import httpx
import pytest

from deltav.compute.base import DeviceInfo
from deltav.gateway import GatewayDaemon
from deltav.gateway.ollama import ollama_tag, resolve_model, short_name
from deltav.node import NodeConfig, NodeDaemon
from deltav.router import Catalog

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9711"
GW = "http://127.0.0.1:9710"


# ------------------------------------------------------------- translation

def test_short_name_and_tag():
    spec = Catalog().by_ref("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
    assert short_name(spec.ref) == "meta-llama-3.1-8b-instruct"
    assert ollama_tag(spec) == "meta-llama-3.1-8b-instruct:q4_k_m"


def test_resolve_model_by_short_name():
    catalog = Catalog()
    served = [MODEL]
    assert resolve_model("auto", served, catalog) == "auto"
    assert resolve_model(MODEL, served, catalog) == MODEL
    # loose short name -> the served ref
    assert resolve_model("qwen2.5-14b", served, catalog) == MODEL
    # ollama tag form
    assert resolve_model("qwen2.5-14b-instruct:q4_k_m", served, catalog) == MODEL
    # unknown -> auto
    assert resolve_model("gpt-4", served, catalog) == "auto"


def test_resolve_prefers_served_over_catalog():
    catalog = Catalog()
    # 'llama' matches several catalog specs; served list should win
    served = ["bartowski/Meta-Llama-3.1-8B-Instruct-GGUF::Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"]
    assert resolve_model("meta-llama-3.1-8b", served, catalog) == served[0]


# ------------------------------------------------------------------- e2e

@pytest.fixture
async def net(genesis, alice, carol):
    transport = MultiTransport()
    cfg = NodeConfig(port=9711, endpoint=URL, backend="mock", models=[MODEL],
                     max_parallel_jobs=4,
                     device=DeviceInfo(vendor="nvidia", name="t", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params, client=client)
    transport.add(GW, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_api_version_and_tags(net):
    client = net["client"]
    assert (await client.get(f"{GW}/api/version")).json()["version"].startswith("deltav")
    tags = (await client.get(f"{GW}/api/tags")).json()
    names = [m["name"] for m in tags["models"]]
    assert any("qwen2.5" in n for n in names)
    served = [m for m in tags["models"] if m["deltav"]["served"]]
    assert served and served[0]["details"]["quantization_level"] == "Q4_K_M"


async def test_api_chat_non_stream(net):
    client = net["client"]
    resp = await client.post(f"{GW}/api/chat", json={
        "model": "qwen2.5-14b", "stream": False,
        "messages": [{"role": "user", "content": "hi ollama"}],
        "options": {"num_predict": 24},
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["done"] is True and d["message"]["role"] == "assistant"
    assert d["message"]["content"]
    assert d["eval_count"] > 0
    assert d["deltav"]["receipt_tx"]


async def test_api_chat_stream_ndjson(net):
    client = net["client"]
    lines = []
    async with client.stream("POST", f"{GW}/api/chat", json={
        "model": "auto", "stream": True,
        "messages": [{"role": "user", "content": "stream ollama"}],
        "options": {"num_predict": 24},
    }, timeout=30.0) as resp:
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]
        async for line in resp.aiter_lines():
            if line.strip():
                lines.append(json.loads(line))
    assert len(lines) >= 2
    assert lines[-1]["done"] is True
    text = "".join(l["message"]["content"] for l in lines)
    assert text.strip()


async def test_api_generate(net):
    client = net["client"]
    resp = await client.post(f"{GW}/api/generate", json={
        "model": "auto", "prompt": "one plus one", "stream": False,
        "options": {"num_predict": 16},
    }, timeout=30.0)
    d = resp.json()
    assert d["done"] and d["response"] and d["deltav"]["receipt_tx"]


async def test_api_embeddings(net, genesis):
    # embed model needed; this net has no embed node -> expect 503, not a crash
    client = net["client"]
    resp = await client.post(f"{GW}/api/embeddings", json={
        "model": "auto", "prompt": "vectorize me"})
    assert resp.status_code in (200, 503)
