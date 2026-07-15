"""Multimodal + diffusion groundwork and RL/training scaffolding."""
import asyncio

import httpx
import pytest

from deltav.compute.base import ImageRequest
from deltav.compute.mock import MockBackend
from deltav.compute.base import DeviceInfo
from deltav.config import DVT
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.router import Catalog

from conftest import MultiTransport
from test_chain import make_receipt_tx, register_node

SD_MODEL = "second-state/stable-diffusion-v1-5-GGUF::stable-diffusion-v1-5-Q4_0.gguf"
URL = "http://127.0.0.1:9721"
GW = "http://127.0.0.1:9720"


# ------------------------------------------------------------ catalog + backend

def test_catalog_has_vision_diffusion_grok():
    catalog = Catalog()
    kinds = {s.kind for s in catalog.specs}
    assert "image" in kinds
    assert any(s.vision for s in catalog.specs)                 # multimodal
    assert any(s.family == "grok" for s in catalog.specs)       # xAI open weights
    assert len(catalog.image_specs()) >= 2                      # SD + FLUX


def test_mock_generates_deterministic_image():
    b = MockBackend()
    req = ImageRequest(prompt="a red cube", model_ref=SD_MODEL, seed=1)
    r1 = b.generate_image(req)
    r2 = b.generate_image(req)
    assert r1.images == r2.images and r1.images[0]              # deterministic, non-empty
    diff = b.generate_image(ImageRequest(prompt="a blue cube", model_ref=SD_MODEL, seed=1))
    assert diff.images != r1.images                             # prompt changes output
    assert b.supports_image_gen and b.supports_vision


# ------------------------------------------------------------------- e2e

@pytest.fixture
async def image_net(genesis, alice, carol):
    transport = MultiTransport()
    cfg = NodeConfig(port=9721, endpoint=URL, backend="mock", models=[SD_MODEL],
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
        yield {"client": client, "daemon": daemon}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_image_generation_endpoint(image_net):
    client = image_net["client"]
    resp = await client.post(f"{GW}/v1/images/generations", json={
        "model": "auto", "prompt": "космический корабль", "size": "512x512",
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    d = resp.json()
    assert d["data"][0]["b64_json"]                    # got an image
    assert "stable-diffusion" in d["model"]
    assert d["deltav"]["receipt_tx"]                   # paid + on-chain

    # the diffusion receipt lands on-chain like any inference
    deadline = asyncio.get_event_loop().time() + 8.0
    receipts = []
    while asyncio.get_event_loop().time() < deadline:
        receipts = (await client.get(f"{URL}/chain/receipts")).json()["receipts"]
        if receipts:
            break
        await asyncio.sleep(0.05)
    assert receipts and receipts[0]["model"] == SD_MODEL


async def test_image_request_to_chat_node_is_refused(genesis, alice):
    # a chat-only node must not accept an image job for a model it doesn't serve
    transport = MultiTransport()
    cfg = NodeConfig(port=9722, endpoint="http://127.0.0.1:9722", backend="mock",
                     models=["some/chat-model::c.gguf"],
                     device=DeviceInfo(vendor="nvidia", name="t", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    daemon.backend.dynamic_models = False  # fixed-model node (like llama-server)
    transport.add("http://127.0.0.1:9722", daemon.app)
    client = httpx.AsyncClient(transport=transport)
    await daemon.start()
    try:
        resp = await client.post("http://127.0.0.1:9722/image", json={
            "prompt": "x", "model": SD_MODEL, "requester": alice.address,
            "requester_pubkey": alice.public_hex, "requester_sig": "00", "price_limit": 10**9,
        })
        assert resp.status_code == 409  # image model it doesn't serve
    finally:
        await daemon.stop()
        await client.aclose()


# ----------------------------------------------------- RL / training groundwork

def test_dataset_from_receipts_labels_by_verdict():
    from deltav.training import iter_training_samples

    class R:
        def __init__(self, rh, checked, ok):
            self.request_hash = rh; self.model = "m"; self.output_hash = "h"
            self.tokens_in = 10; self.tokens_out = 20; self.height = 1
            self.checked = checked; self.check_ok = ok
    receipts = [R("a", True, True), R("b", True, False), R("c", False, None)]
    jobs = {"a": {"prompt": "hi", "output": "there"}}
    good = list(iter_training_samples(receipts, jobs, min_reward=1.0))
    assert len(good) == 1 and good[0].request_hash == "a"
    assert good[0].prompt == "hi" and good[0].reward == 1.0


def test_dry_run_coordinator_pipeline():
    from deltav.training.coordinator import DryRunCoordinator
    from deltav.training.dataset import ReceiptSample

    samples = [ReceiptSample("r%d" % i, "m", "prompt", "h", 30, 1, 1.0, "out")
               for i in range(70)]
    coord = DryRunCoordinator("Qwen2.5-7B", min_samples=64)
    report = coord.run_round(1, samples)
    assert report.positive == 70 and report.checkpoint.startswith("Qwen2.5-7B+ft")

    thin = DryRunCoordinator("Qwen2.5-7B", min_samples=64)
    r2 = thin.run_round(2, samples[:10])
    assert not r2.checkpoint and "not enough" in r2.notes[0]
