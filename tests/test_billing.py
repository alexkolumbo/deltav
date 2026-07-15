"""Billing: API keys as custodial on-chain wallets."""
import asyncio

import httpx
import pytest

from deltav.chain.transaction import Tx, TxType
from deltav.compute.base import DeviceInfo
from deltav.config import DVT, Genesis
from deltav.gateway import GatewayDaemon
from deltav.gateway.keys import KeyStore
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URL = "http://127.0.0.1:9901"
GW_URL = "http://127.0.0.1:9900"


# -------------------------------------------------------------- keystore

def test_keystore_create_resolve_persist(tmp_path):
    store = KeyStore(tmp_path / "keys.json")
    api_key, record = store.create(label="phone")
    assert api_key.startswith("dvk_")
    assert store.resolve(api_key) is record
    assert store.resolve("dvk_wrong") is None
    assert store.keypair(record).address == record.address

    store.record_usage(record, tokens=100, spent_udvt=900)
    reopened = KeyStore(tmp_path / "keys.json")
    again = reopened.resolve(api_key)
    assert again.address == record.address
    assert again.tokens == 100 and again.spent_udvt == 900
    # plaintext key is never stored
    assert api_key not in (tmp_path / "keys.json").read_text(encoding="utf-8")


# ------------------------------------------------------------ end to end

@pytest.fixture
async def billing_net(genesis, alice, carol, tmp_path):
    transport = MultiTransport()
    cfg = NodeConfig(port=9901, endpoint=URL, backend="mock", models=[MODEL],
                     max_parallel_jobs=4,
                     device=DeviceInfo(vendor="nvidia", name="test", vram_mb=12282))
    daemon = NodeDaemon(alice, genesis, cfg, client=httpx.AsyncClient(transport=transport))
    transport.add(URL, daemon.app)
    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(carol, node_urls=[URL], params=genesis.params, client=client,
                            keys_path=str(tmp_path / "keys.json"))
    transport.add(GW_URL, gateway.app)
    await daemon.start()
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if (await client.get(f"{URL}/chain/nodes")).json()["nodes"]:
            break
        await asyncio.sleep(0.05)
    try:
        yield {"client": client, "gateway": gateway, "daemon": daemon,
               "transport": transport}
    finally:
        await daemon.stop()
        await client.aclose()


async def balance_of(client, address: str) -> int:
    return (await client.get(f"{URL}/chain/account/{address}")).json()["balance"]


async def fund(client, from_kp, to_address: str, amount: int) -> None:
    acc = (await client.get(f"{URL}/chain/account/{from_kp.address}")).json()
    tx = Tx(type=TxType.TRANSFER.value, sender=from_kp.address, nonce=acc["nonce"],
            payload={"to": to_address, "amount": amount}).sign(from_kp)
    await client.post(f"{URL}/tx", json=tx.to_dict())
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if await balance_of(client, to_address) >= amount:
            return
        await asyncio.sleep(0.05)
    raise AssertionError("funding transfer never landed")


async def test_key_lifecycle_charges_consumer_not_gateway(billing_net, bob, carol):
    client = billing_net["client"]

    # 1. create a key -> a fresh on-chain wallet
    created = (await client.post(f"{GW_URL}/v1/keys", json={"label": "test"})).json()
    api_key, address = created["api_key"], created["address"]
    headers = {"Authorization": f"Bearer {api_key}"}

    # 2. unfunded -> 402 BEFORE any inference happens
    body = {"model": "auto", "max_tokens": 32,
            "messages": [{"role": "user", "content": "платный запрос"}]}
    resp = await client.post(f"{GW_URL}/v1/chat/completions", json=body, headers=headers)
    assert resp.status_code == 402
    assert address in resp.json()["detail"]

    # 3. fund the key's address from bob's wallet, retry -> 200
    await fund(client, bob, address, 5 * DVT)
    gateway_before = await balance_of(client, carol.address)
    resp = await client.post(f"{GW_URL}/v1/chat/completions", json=body,
                             headers=headers, timeout=30.0)
    assert resp.status_code == 200, resp.text
    usage = resp.json()["usage"]

    # 4. the receipt charges the KEY's wallet on-chain, not the gateway's
    price = usage["total_tokens"] * 10
    deadline = asyncio.get_event_loop().time() + 8.0
    while asyncio.get_event_loop().time() < deadline:
        if await balance_of(client, address) == 5 * DVT - price:
            break
        await asyncio.sleep(0.05)
    assert await balance_of(client, address) == 5 * DVT - price
    assert await balance_of(client, carol.address) == gateway_before

    # 5. usage shows up in /v1/keys/me
    me = (await client.get(f"{GW_URL}/v1/keys/me", headers=headers)).json()
    assert me["address"] == address
    assert me["requests"] == 1 and me["tokens"] == usage["total_tokens"]
    assert me["balance_udvt"] == 5 * DVT - price


async def test_unknown_dvk_key_is_401_even_in_open_mode(billing_net):
    client = billing_net["client"]
    resp = await client.post(f"{GW_URL}/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "x"}],
    }, headers={"Authorization": "Bearer dvk_deadbeef"})
    assert resp.status_code == 401


async def test_placeholder_token_falls_back_to_gateway_wallet(billing_net):
    """goose sends api_key='deltav' — open mode must keep working."""
    client = billing_net["client"]
    resp = await client.post(f"{GW_URL}/v1/chat/completions", json={
        "model": "auto", "max_tokens": 16,
        "messages": [{"role": "user", "content": "анонимно"}],
    }, headers={"Authorization": "Bearer deltav"}, timeout=30.0)
    assert resp.status_code == 200


async def test_require_keys_locks_anonymous_out(billing_net, carol, tmp_path):
    client = billing_net["client"]
    strict = GatewayDaemon(carol, node_urls=[URL], params=None, client=client,
                           keys_path=str(tmp_path / "strict-keys.json"),
                           require_keys=True)
    billing_net["transport"].add("http://127.0.0.1:9902", strict.app)
    resp = await client.post("http://127.0.0.1:9902/v1/chat/completions", json={
        "model": "auto", "messages": [{"role": "user", "content": "x"}],
    })
    assert resp.status_code == 401
    assert "API key required" in resp.json()["detail"]
