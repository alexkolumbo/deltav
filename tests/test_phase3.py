"""Phase 3: node bootstrap, explorer/stats endpoints, SSE streaming."""
import asyncio
import json

import httpx
import pytest

from deltav.bootstrap import fetch_genesis, pick_model_for_device
from deltav.compute.base import DeviceInfo
from deltav.config import DVT, Genesis
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9401"


# ------------------------------------------------------------- bootstrap

def test_pick_model_4070():
    spec = pick_model_for_device(DeviceInfo(vendor="nvidia", name="RTX 4070", vram_mb=12282))
    assert spec is not None and spec.params_b > 10


def test_pick_model_small_gpu():
    spec = pick_model_for_device(DeviceInfo(vendor="amd", name="RX 6600", vram_mb=8192))
    assert spec is not None and 6 <= spec.params_b <= 10  # a 7-9B-class model


@pytest.fixture
async def single_node(genesis, alice):
    transport = MultiTransport()
    cfg = NodeConfig(port=9401, endpoint=URL, backend="mock", models=[MODEL],
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    await daemon.start()
    try:
        yield {"daemon": daemon, "client": client, "transport": transport}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_genesis_endpoint_roundtrip(single_node, genesis):
    fetched = await fetch_genesis(URL, client=single_node["client"])
    assert fetched.to_dict() == genesis.to_dict()


async def test_stats_endpoint(single_node, genesis):
    stats = (await single_node["client"].get(f"{URL}/chain/stats")).json()
    assert stats["chain_id"] == genesis.params.chain_id
    assert stats["validators"] == 2
    assert set(stats) >= {"height", "supply", "nodes", "receipts", "mempool", "peers"}


async def test_explorer_served(single_node):
    resp = await single_node["client"].get(f"{URL}/explorer")
    assert resp.status_code == 200
    assert "DELTA V EXPLORER" in resp.text
    assert "text/html" in resp.headers["content-type"]


# ------------------------------------------------------------- streaming

async def test_gateway_sse_streaming(single_node, genesis, carol):
    gw_key = carol  # funded in the genesis fixture, so receipts can be paid
    client = single_node["client"]
    gateway = GatewayDaemon(gw_key, node_urls=[URL], params=genesis.params, client=client)
    gw_url = "http://127.0.0.1:9400"
    single_node["transport"].add(gw_url, gateway.app)

    # wait until the node's REGISTER_NODE tx lands on-chain
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        nodes = (await client.get(f"{URL}/chain/nodes")).json()["nodes"]
        if nodes:
            break
        await asyncio.sleep(0.05)
    assert nodes, "node never registered on-chain"

    body = {"model": "auto", "messages": [{"role": "user", "content": "stream me"}],
            "max_tokens": 32, "seed": 5}

    # non-streamed reference answer (deterministic mock -> same text)
    ref = (await client.post(f"{gw_url}/v1/chat/completions", json=body, timeout=30.0)).json()
    assert "choices" in ref, ref
    ref_text = ref["choices"][0]["message"]["content"]

    chunks, done = [], False
    async with client.stream(
        "POST", f"{gw_url}/v1/chat/completions", json={**body, "stream": True}, timeout=30.0
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))

    assert done
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == ref_text
    finals = [c for c in chunks if c["choices"][0]["finish_reason"] == "stop"]
    assert finals and "usage" in finals[-1]
