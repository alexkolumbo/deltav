"""Companion agent: strict per-user isolation, per-user memory, self-improvement."""
import asyncio

import httpx
import pytest

from deltav.companion import CompanionAgent, UserMemory, resolve_identity
from deltav.compute.base import DeviceInfo
from deltav.config import DVT
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon
from deltav.overlay.memory import VectorMemory
from deltav.overlay.tools import ToolRegistry

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9731"
GW = "http://127.0.0.1:9730"


# --------------------------------------------------------------- identity

def test_identity_from_key_beats_body():
    # an authenticated key defines the identity; a body-supplied user is ignored
    ident = resolve_identity(address="dv1alice", fallback_user="pretend-to-be-bob")
    assert ident.address == "dv1alice" and ident.authenticated
    assert "dv1alice" in ident.user_id and "bob" not in ident.user_id


def test_keyless_identity_is_local_namespace():
    ident = resolve_identity(address="", fallback_user="")
    assert not ident.authenticated
    assert ident.user_id.startswith("companion:local")


# ------------------------------------------------------ isolation (security)

async def test_users_cannot_read_each_others_memory():
    store = VectorMemory(embedder=None)  # BM25 fallback, no network needed
    alice = UserMemory(store, resolve_identity("dv1alice"))
    bob = UserMemory(store, resolve_identity("dv1bob"))

    await alice.remember("моя банковская карта заканчивается на 4242")
    await bob.remember("я люблю котов")

    alice_hits = await alice.recall("карта")
    bob_hits = await bob.recall("карта")
    assert any("4242" in h["text"] for h in alice_hits)      # alice sees her own
    assert all("4242" not in h["text"] for h in bob_hits)    # bob sees NOTHING of alice's
    assert bob.all_items() and all("4242" not in it["text"] for it in bob.all_items())


# ---------------------------------------------------- memory + reflection

async def test_companion_recalls_and_self_improves():
    # scripted model: an answer, then a reflection producing a learning
    replies = iter(["Готово, встреча в 15:00.",
                    "Пользователь предпочитает встречи после обеда."])

    async def complete(prompt: str):
        return next(replies), {"node": "n", "receipt_tx": "r"}

    store = VectorMemory(embedder=None)
    mem = UserMemory(store, resolve_identity("dv1alice"))
    agent = CompanionAgent(complete, ToolRegistry(), reflect=True)

    r = await agent.run(mem, "поставь встречу")
    assert r.answer == "Готово, встреча в 15:00."
    assert r.learned == "Пользователь предпочитает встречи после обеда."
    # the learning persisted for this user
    assert any("после обеда" in it["text"] for it in mem.learnings())


async def test_reflection_none_stores_nothing():
    replies = iter(["Ответ.", "NONE"])

    async def complete(prompt: str):
        return next(replies), {}

    mem = UserMemory(VectorMemory(embedder=None), resolve_identity("dv1x"))
    r = await CompanionAgent(complete, ToolRegistry()).run(mem, "привет")
    assert r.learned is None and not mem.learnings()


async def test_learnings_injected_into_next_turn():
    captured = {}

    async def complete(prompt: str):
        captured["prompt"] = prompt
        return "ок", {}

    store = VectorMemory(embedder=None)
    mem = UserMemory(store, resolve_identity("dv1alice"))
    await mem.learn("пользователь говорит по-русски и любит кратко")
    agent = CompanionAgent(complete, ToolRegistry(), reflect=False)
    await agent.run(mem, "расскажи о погоде")
    assert "learned about this user" in captured["prompt"]
    assert "кратко" in captured["prompt"]


async def test_feedback_becomes_high_priority_learning():
    mem = UserMemory(VectorMemory(embedder=None), resolve_identity("dv1alice"))
    await CompanionAgent(None, ToolRegistry()).feedback(mem, "не используй смайлики")
    learn = mem.learnings()
    assert learn and "смайлики" in learn[-1]["text"]
    assert learn[-1]["meta"]["weight"] >= 3.0     # feedback outranks self-notes


# ------------------------------------------------------------------ e2e

@pytest.fixture
async def companion_net(genesis, alice, bob, carol, tmp_path):
    transport = MultiTransport()
    cfg = NodeConfig(port=9731, endpoint=URL, backend="mock", models=[MODEL],
                     max_parallel_jobs=4,
                     device=DeviceInfo(vendor="nvidia", name="t", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params, client=client,
                            keys_path=str(tmp_path / "keys.json"))
    transport.add(GW, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "gateway": gateway, "bob": bob}
    finally:
        await daemon.stop()
        await client.aclose()


async def test_gateway_companion_isolated_by_key(companion_net):
    client = companion_net["client"]
    # two funded keys = two users
    async def make_key(label):
        k = (await client.post(f"{GW}/v1/keys", json={"label": label})).json()
        return k["api_key"], k["address"]
    from deltav.chain.transaction import Tx, TxType
    k1, a1 = await make_key("u1")
    k2, a2 = await make_key("u2")
    # fund both from bob
    bob = companion_net["bob"]
    for addr in (a1, a2):
        acc = (await client.get(f"{URL}/chain/account/{bob.address}")).json()
        tx = Tx(type=TxType.TRANSFER.value, sender=bob.address, nonce=acc["nonce"],
                payload={"to": addr, "amount": 5 * DVT}).sign(bob)
        await client.post(f"{URL}/tx", json=tx.to_dict())
        deadline = asyncio.get_event_loop().time() + 8.0
        while asyncio.get_event_loop().time() < deadline:
            if (await client.get(f"{URL}/chain/account/{addr}")).json()["balance"] > 0:
                break
            await asyncio.sleep(0.05)

    # user 1 stores a secret via feedback; a chat turn works end to end too
    await client.post(f"{GW}/v1/companion/feedback",
                      headers={"Authorization": f"Bearer {k1}"},
                      json={"note": "мой пароль alpha-777"}, timeout=30.0)
    r1 = (await client.post(f"{GW}/v1/companion/chat",
                            headers={"Authorization": f"Bearer {k1}"},
                            json={"message": "привет", "max_tokens": 24,
                                  "reflect": False}, timeout=30.0)).json()
    assert r1["answer"] and r1["authenticated"]

    # user 1 sees their own memory
    m1 = (await client.get(f"{GW}/v1/companion/memory",
                           headers={"Authorization": f"Bearer {k1}"})).json()
    assert any("alpha-777" in it["text"] for it in m1["items"])
    # user 2 sees NOTHING of user 1 — even trying to name user 1 can't help
    m2 = (await client.get(f"{GW}/v1/companion/memory",
                           headers={"Authorization": f"Bearer {k2}"})).json()
    assert all("alpha-777" not in it["text"] for it in m2["items"])
    assert m2["user"] != m1["user"]
    # and user 2's companion cannot recall user 1's secret in a turn
    r2 = (await client.post(f"{GW}/v1/companion/chat",
                            headers={"Authorization": f"Bearer {k2}"},
                            json={"message": "какой у меня пароль?", "max_tokens": 24,
                                  "reflect": False}, timeout=30.0)).json()
    assert all("alpha-777" not in m for m in r2["memory_used"])
