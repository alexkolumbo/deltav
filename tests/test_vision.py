"""Vision (image input) passthrough + reasoning thinking-toggle."""
import asyncio

import httpx
import pytest

from deltav.compute.base import InferRequest
from deltav.compute.llamaserver import LlamaServerBackend
from deltav.compute.mock import MockBackend
from deltav.compute.base import DeviceInfo
from deltav.config import DVT
from deltav.gateway import GatewayDaemon
from deltav.gateway.app import extract_images
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9741"
GW = "http://127.0.0.1:9740"


# ------------------------------------------------------- gateway extraction

def test_extract_images_flattens_content():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "что на картинке?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        {"type": "image_url", "image_url": {"url": "https://x/y.jpg"}},
    ]}]
    imgs = extract_images(messages)
    assert imgs == ["data:image/png;base64,AAA", "https://x/y.jpg"]
    assert messages[0]["content"] == "что на картинке?"   # flattened to text


def test_extract_images_plain_text_unchanged():
    messages = [{"role": "user", "content": "просто текст"}]
    assert extract_images(messages) == []
    assert messages[0]["content"] == "просто текст"


# ---------------------------------------------------------- backend body

def test_llamaserver_thinking_off_by_default():
    b = LlamaServerBackend(base_url="http://x")
    body = b._chat_body(InferRequest(prompt="hi", model_ref=MODEL))
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    on = LlamaServerBackend(base_url="http://x", think=True)
    assert on._chat_body(InferRequest(prompt="hi", model_ref=MODEL))[
        "chat_template_kwargs"]["enable_thinking"] is True


def test_llamaserver_vision_content_blocks():
    b = LlamaServerBackend(base_url="http://x")
    body = b._chat_body(InferRequest(prompt="what is this?", model_ref=MODEL,
                                     images=["data:image/png;base64,AAA"]))
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "what is this?"}
    assert content[1]["type"] == "image_url"


def test_mock_reflects_images():
    r = MockBackend().infer(InferRequest(prompt="x", model_ref=MODEL,
                                         images=["data:...", "data:..."]))
    assert "saw 2 image(s)" in r.text


# ------------------------------------------------------------------- e2e

@pytest.fixture
async def net(genesis, alice, carol):
    transport = MultiTransport()
    cfg = NodeConfig(port=9741, endpoint=URL, backend="mock", models=[MODEL],
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


async def test_vision_image_flows_through_network(net):
    client = net["client"]
    resp = await client.post(f"{GW}/v1/chat/completions", json={
        "model": "auto", "max_tokens": 32,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "опиши изображение"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,ABCDEF"}},
        ]}],
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    d = resp.json()
    # the image reached the node/backend end to end
    assert "saw 1 image(s)" in d["choices"][0]["message"]["content"]
    assert d["deltav"]["receipt_tx"]
