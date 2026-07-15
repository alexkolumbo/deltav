"""End-to-end: two node daemons + gateway, in-process (no sockets).

Covers the full trust pipeline: registration txs -> PoS blocks -> smart
routing -> inference -> on-chain receipt with payment -> spot-check
re-execution -> reputation/slashing.
"""
import asyncio

import httpx
import pytest

from deltav.compute.base import DeviceInfo
from deltav.config import DVT, ChainParams, Genesis
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.node import NodeConfig, NodeDaemon

from conftest import MultiTransport

MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
URLS = ["http://127.0.0.1:9101", "http://127.0.0.1:9102"]


async def wait_for(predicate, timeout=8.0, interval=0.05):
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        result = await predicate()
        if result:
            return result
        await asyncio.sleep(interval)
    raise AssertionError("condition not met within timeout")


@pytest.fixture
async def network():
    params = ChainParams(block_time=0.05, spot_check_rate=1.0)
    node_keys = [KeyPair.from_seed_hex("aa" * 32), KeyPair.from_seed_hex("bb" * 32)]
    gw_key = KeyPair.from_seed_hex("cc" * 32)
    genesis = Genesis(
        params=params,
        alloc={kp.address: 100_000 * DVT for kp in node_keys} | {gw_key.address: 50_000 * DVT},
        stakes={kp.address: 10_000 * DVT for kp in node_keys},
    )
    transport = MultiTransport()
    daemons = []
    for i, kp in enumerate(node_keys):
        cfg = NodeConfig(
            port=9101 + i,
            endpoint=URLS[i],
            peers=[URLS[1 - i]],
            backend="mock",
            models=[MODEL],
            device=DeviceInfo(vendor="nvidia", name="RTX 4070 (test)", vram_mb=12282),
        )
        daemon = NodeDaemon(kp, genesis, cfg, client=httpx.AsyncClient(transport=transport))
        transport.add(URLS[i], daemon.app)
        daemons.append(daemon)

    client = httpx.AsyncClient(transport=transport)
    gateway = GatewayDaemon(gw_key, node_urls=URLS, params=params, client=client)
    gw_url = "http://127.0.0.1:9100"
    transport.add(gw_url, gateway.app)

    for d in daemons:
        await d.start()
    try:
        yield {"daemons": daemons, "gateway": gateway, "client": client,
               "gw_url": gw_url, "gw_key": gw_key, "node_keys": node_keys}
    finally:
        for d in daemons:
            await d.stop()
        await client.aclose()


async def test_full_pipeline(network):
    client = network["client"]
    gw_key = network["gw_key"]

    # 1. Both nodes register on-chain and blocks are being produced.
    async def registered():
        resp = await client.get(f"{URLS[0]}/chain/nodes")
        data = resp.json()
        return data if len(data["nodes"]) == 2 and data["height"] > 0 else None
    registry = await wait_for(registered)
    assert all(n["stake"] >= 10_000 * DVT for n in registry["nodes"])

    # 2. Chat through the gateway: auto-routing picks the 14B model on a 4070.
    resp = await client.post(f"{network['gw_url']}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "hello delta v"}],
        "max_tokens": 32,
        "seed": 7,
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model"] == MODEL
    assert data["choices"][0]["message"]["content"]
    assert data["deltav"]["receipt_tx"]
    serving_node = data["deltav"]["node"]

    # 3. The receipt lands on-chain and the requester actually paid for it.
    async def receipt_on_chain():
        r = (await client.get(f"{URLS[1]}/chain/receipts")).json()["receipts"]
        return r if r else None
    receipts = await wait_for(receipt_on_chain)
    assert receipts[0]["node"] == serving_node

    async def gateway_charged():
        acc = (await client.get(f"{URLS[0]}/chain/account/{gw_key.address}")).json()
        return acc if acc["balance"] == 50_000 * DVT - receipts[0]["price_paid"] else None
    await wait_for(gateway_charged)  # node 0 may see the receipt block a beat later

    # 4. The other validator spot-checks the receipt by re-executing the job.
    async def checked():
        r = (await client.get(f"{URLS[0]}/chain/receipts")).json()["receipts"]
        return r if r and r[0]["checked"] else None
    receipts = await wait_for(checked)
    assert receipts[0]["check_ok"] is True

    # 5. Both nodes converge to the same height eventually.
    h0 = (await client.get(f"{URLS[0]}/chain/head")).json()["height"]
    h1 = (await client.get(f"{URLS[1]}/chain/head")).json()["height"]
    assert abs(h0 - h1) <= 2


async def test_cheating_node_gets_slashed(network):
    client = network["client"]
    daemons = network["daemons"]
    gw_key = network["gw_key"]

    async def ready():
        data = (await client.get(f"{URLS[0]}/chain/nodes")).json()
        return data if len(data["nodes"]) == 2 else None
    await wait_for(ready)

    resp = await client.post(f"{network['gw_url']}/v1/chat/completions", json={
        "model": "auto",
        "messages": [{"role": "user", "content": "trust but verify"}],
        "max_tokens": 32,
        "seed": 99,
    }, timeout=30.0)
    assert resp.status_code == 200, resp.text
    serving_addr = resp.json()["deltav"]["node"]
    serving = next(d for d in daemons if d.address == serving_addr)

    # The node lies about what it computed: it serves tampered job params,
    # so the checker's re-execution won't reproduce the receipt's output hash.
    for job in serving.jobs.values():
        job["prompt"] = "something entirely different"

    async def slashed():
        r = (await client.get(f"{URLS[0]}/chain/receipts")).json()["receipts"]
        return r if r and r[0]["checked"] else None
    receipts = await wait_for(slashed)
    assert receipts[0]["check_ok"] is False

    acc = (await client.get(f"{URLS[0]}/chain/account/{serving_addr}")).json()
    assert acc["stake"] < 10_000 * DVT  # part of the stake was burned
    nodes = (await client.get(f"{URLS[0]}/chain/nodes")).json()["nodes"]
    cheater = next(n for n in nodes if n["address"] == serving_addr)
    assert cheater["reputation"] < 0.5
