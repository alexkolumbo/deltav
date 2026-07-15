"""Phase 6: RANDAO, parallel jobs / multi-GPU, agent memory + sub-agents."""
import asyncio
import json
import time

import httpx
import pytest

from deltav.chain.blockchain import Blockchain
from deltav.chain.consensus import ConsensusError, randao_message
from deltav.compute.base import DeviceInfo, InferRequest
from deltav.compute.detect import parse_nvidia_smi
from deltav.compute.mock import MockBackend
from deltav.config import DVT, Genesis
from deltav.crypto import KeyPair, verify_signature
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.overlay.memory import MemoryStore

from conftest import MultiTransport
from test_phase2 import keys_map, produce

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9701"
GW_URL = "http://127.0.0.1:9700"


# ---------------------------------------------------------------- randao

def test_randao_accumulator_advances(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    assert chain.state.randao == ""
    seen = set()
    for _ in range(5):
        produce(chain, keys)
        assert chain.state.randao not in seen
        seen.add(chain.state.randao)
    # the reveal on every block verifies against the proposer's key
    for block in chain.blocks[1:]:
        assert verify_signature(
            block.pubkey,
            randao_message(genesis.params.chain_id, block.height),
            block.randao_reveal,
        )


def test_tampered_randao_reveal_rejected(genesis, alice, bob):
    keys = keys_map(alice, bob)
    chain = Blockchain(genesis)
    proposer = chain.next_proposer()
    block = chain.build_block(keys[proposer], [], 1.0)
    other = alice if proposer != alice.address else bob
    block.randao_reveal = other.sign(randao_message(genesis.params.chain_id, 1))
    block.sign(keys[proposer])  # re-sign the header so only the reveal is wrong
    with pytest.raises(ConsensusError, match="randao"):
        chain.add_block(block)


def test_diverged_histories_give_diverged_randao(genesis, alice, bob):
    keys = keys_map(alice, bob)
    a = Blockchain(genesis)
    b = Blockchain(genesis)
    for _ in range(6):
        produce(a, keys, timestamp=100.0)
    victim = a.blocks[1].proposer  # force different proposers on chain b
    for _ in range(6):
        produce(b, keys, exclude={victim}, timestamp=100.0)
    if [blk.proposer for blk in a.blocks] != [blk.proposer for blk in b.blocks]:
        assert a.state.randao != b.state.randao


# ------------------------------------------------- multi-GPU + parallel jobs

def test_parse_nvidia_smi_multi_gpu():
    out = "NVIDIA GeForce RTX 4070, 12282\nNVIDIA GeForce RTX 4070, 12282\n"
    device = parse_nvidia_smi(out)
    assert device.gpu_count == 2
    assert device.vram_mb == 24564
    assert device.name.endswith("x2")
    assert parse_nvidia_smi("garbage") is None


def test_parse_windows_gpus_picks_discrete_amd():
    from deltav.compute.detect import parse_windows_gpus

    out = ("AMD Radeon(TM) Graphics|536870912\n"        # iGPU carve-out, skipped
           "AMD Radeon RX 6600M|8573157376\n")
    device = parse_windows_gpus(out)
    assert device.vendor == "amd"
    assert device.name == "AMD Radeon RX 6600M"
    assert device.vram_mb == 8176
    assert parse_windows_gpus("") is None


class SlowBackend(MockBackend):
    """Records how many infer() calls overlap."""
    current = 0
    max_seen = 0

    def infer(self, request: InferRequest):
        cls = type(self)
        cls.current += 1
        cls.max_seen = max(cls.max_seen, cls.current)
        try:
            time.sleep(0.08)
            return super().infer(request)
        finally:
            cls.current -= 1


async def test_job_semaphore_caps_concurrency(genesis, alice, carol):
    SlowBackend.current = SlowBackend.max_seen = 0
    transport = MultiTransport()
    cfg = NodeConfig(port=9701, endpoint=URL, backend="mock", models=[MODEL],
                     max_parallel_jobs=2,
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    daemon.backend = SlowBackend()
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    await daemon.start()
    try:
        from deltav.router import Catalog, SmartRouter
        router = SmartRouter(Catalog(), carol, client, genesis.params.price_per_token)

        async def one(i: int):
            spec = Catalog().by_ref(MODEL)
            node_view = type("N", (), {"address": alice.address})
            body = router._infer_body(node_view, spec, f"job {i}", 16, 0.0, i)
            resp = await client.post(f"{URL}/infer", json=body, timeout=30.0)
            assert resp.status_code == 200

        await asyncio.gather(*(one(i) for i in range(5)))
        assert SlowBackend.max_seen <= 2  # semaphore held the line
        health = (await client.get(f"{URL}/health")).json()
        assert health["max_parallel_jobs"] == 2
    finally:
        await daemon.stop()
        await client.aclose()


# ---------------------------------------------------------------- memory

def test_bm25_relevance_and_sessions(tmp_path):
    store = MemoryStore(tmp_path / "mem.jsonl")
    store.add("s1", "код от сейфа в офисе 1234")
    store.add("s1", "любимый цвет пользователя синий")
    store.add("s2", "чужая сессия про сейф 9999")

    hits = store.search("s1", "какой код у сейфа")
    assert hits and "1234" in hits[0]["text"]
    assert all(h["session"] == "s1" for h in hits)

    reopened = MemoryStore(tmp_path / "mem.jsonl")
    assert len(reopened.session_items("s1")) == 2
    assert "1234" in reopened.search("s1", "код сейфа")[0]["text"]


# --------------------------------------------------- gateway agents e2e

@pytest.fixture
async def agent_net(genesis, alice, carol, tmp_path):
    transport = MultiTransport()
    cfg = NodeConfig(port=9701, endpoint=URL, backend="mock", models=[MODEL],
                     max_parallel_jobs=4,
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params,
                            client=client, memory_path=str(tmp_path / "mem.jsonl"))
    transport.add(GW_URL, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "gateway": gateway}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_agent_memory_roundtrip(agent_net):
    client = agent_net["client"]
    remember_task = (
        '[[reply]]<tool_call>{"name": "remember", '
        '"arguments": {"text": "секретный код проекта: 7788"}}</tool_call>[[/reply]]'
        "[[reply]]Запомнил.[[/reply]] запомни код проекта 7788"
    )
    r1 = (await client.post(f"{GW_URL}/v1/agents/run", json={
        "task": remember_task, "session_id": "sess-1", "max_steps": 4,
    }, timeout=60.0)).json()
    assert r1["steps"][0]["tool"] == "remember"
    assert "remembered" in r1["steps"][0]["result"]

    recall_task = (
        '[[reply]]<tool_call>{"name": "recall", '
        '"arguments": {"query": "код проекта"}}</tool_call>[[/reply]]'
        "[[reply]]Код 7788.[[/reply]] какой код проекта?"
    )
    r2 = (await client.post(f"{GW_URL}/v1/agents/run", json={
        "task": recall_task, "session_id": "sess-1", "max_steps": 4,
    }, timeout=60.0)).json()
    assert "7788" in r2["steps"][0]["result"]
    # auto-recall injected the stored fact before the agent even asked
    assert any("7788" in h["text"] for h in r2["memory_used"])

    # other sessions can't see it
    view = (await client.get(f"{GW_URL}/v1/memory",
                             params={"session": "sess-2", "q": "код"})).json()
    assert view["items"] == []


async def test_spawn_agents_parallel_subtasks(agent_net):
    client = agent_net["client"]
    # Sub-agents run with fresh, unscripted prompts — the deterministic mock
    # answers each one directly (no tool calls), like a plain completion.
    spawn_call = json.dumps({"name": "spawn_agents",
                             "arguments": {"tasks": ["subtask alpha", "subtask beta"]}},
                            ensure_ascii=False)
    task = (f"[[reply]]<tool_call>{spawn_call}</tool_call>[[/reply]]"
            "[[reply]]Оба подагента отработали.[[/reply]] разбей работу на двоих")
    resp = (await client.post(f"{GW_URL}/v1/agents/run", json={
        "task": task, "max_steps": 4,
    }, timeout=60.0)).json()
    assert resp["answer"] == "Оба подагента отработали."
    step = resp["steps"][0]
    assert step["tool"] == "spawn_agents"
    payload = json.loads(step["result"])
    assert [p["task"] for p in payload] == ["subtask alpha", "subtask beta"]
    assert all(p["answer"] for p in payload)
    assert payload[0]["answer"] != payload[1]["answer"]  # independent contexts
    assert "spawn_agents" in resp["tools_available"]