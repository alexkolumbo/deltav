"""Local network simulation.

Boots N node daemons (mock backend, real HTTP on localhost ports) plus a
gateway, lets the chain produce blocks, pushes a couple of chat requests
through the smart router, waits for receipts and spot checks to land
on-chain, and prints a report.
"""
from __future__ import annotations

import asyncio
import contextlib

import httpx
import uvicorn

from .config import DVT, ChainParams, Genesis
from .crypto import KeyPair
from .gateway import GatewayDaemon
from .node import NodeConfig, NodeDaemon

# 4070-class VRAM so the router picks a 14B model in the demo.
SIM_VRAM_MB = 12282
SIM_MODEL = "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"
SIM_SMALL_MODEL = "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf"


def _dvt(amount_udvt: int) -> str:
    text = f"{amount_udvt / DVT:,.6f}".rstrip("0").rstrip(".")
    return f"{text} DVT"


async def run_simulation(n_nodes: int = 3, duration: float = 25.0, base_port: int = 9100) -> dict:
    from .compute.base import DeviceInfo

    params = ChainParams(block_time=1.0)
    node_keys = [KeyPair.generate() for _ in range(n_nodes)]
    gateway_key = KeyPair.generate()
    genesis = Genesis(
        params=params,
        alloc={kp.address: 100_000 * DVT for kp in node_keys} | {gateway_key.address: 50_000 * DVT},
        stakes={kp.address: 10_000 * DVT for kp in node_keys},
    )

    urls = [f"http://127.0.0.1:{base_port + i}" for i in range(n_nodes)]
    daemons: list[NodeDaemon] = []
    for i, kp in enumerate(node_keys):
        cfg = NodeConfig(
            host="127.0.0.1",
            port=base_port + i,
            peers=[u for j, u in enumerate(urls) if j != i],
            backend="mock",
            models=[SIM_MODEL, SIM_SMALL_MODEL] if i % 2 == 0 else [SIM_SMALL_MODEL],
            stake=0,  # validator stake comes from genesis; extra staking is exercised in tests
            device=DeviceInfo(vendor="nvidia", name="GeForce RTX 4070 (sim)", vram_mb=SIM_VRAM_MB),
        )
        daemons.append(NodeDaemon(kp, genesis, cfg))

    gateway = GatewayDaemon(gateway_key, node_urls=urls, params=params)
    gateway_port = base_port - 1

    servers = [
        uvicorn.Server(uvicorn.Config(d.app, host="127.0.0.1", port=base_port + i, log_level="error"))
        for i, d in enumerate(daemons)
    ]
    servers.append(uvicorn.Server(uvicorn.Config(
        gateway.app, host="127.0.0.1", port=gateway_port, log_level="error")))
    server_tasks = [asyncio.create_task(s.serve()) for s in servers]

    report: dict = {}
    try:
        await asyncio.sleep(1.0)  # let uvicorn bind
        for d in daemons:
            await d.start()

        print(f"[sim] {n_nodes} nodes up, waiting for registration + staking to land on-chain...")
        await asyncio.sleep(params.block_time * 5)

        async with httpx.AsyncClient() as client:
            gw = f"http://127.0.0.1:{gateway_port}"

            for i, prompt in enumerate([
                "Explain what delta-v means in orbital mechanics.",
                "Write a haiku about GPUs mining knowledge.",
            ]):
                resp = await client.post(f"{gw}/v1/chat/completions", json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "seed": 42 + i,
                }, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                meta = data["deltav"]
                print(f"[sim] chat {i + 1}: model={data['model'].split('::')[0]}")
                print(f"[sim]   -> node {meta['node'][:16]}... attempts={meta['attempts']}")
                print(f"[sim]   -> {data['choices'][0]['message']['content'][:100]}")

            print(f"[sim] letting the chain run for {duration:.0f}s (blocks, receipts, spot checks)...")
            await asyncio.sleep(duration)

            head = (await client.get(f"{urls[0]}/chain/head")).json()
            nodes = (await client.get(f"{urls[0]}/chain/nodes")).json()["nodes"]
            receipts = (await client.get(f"{urls[0]}/chain/receipts")).json()["receipts"]
            gw_acc = (await client.get(f"{urls[0]}/chain/account/{gateway_key.address}")).json()

            heights = []
            for u in urls:
                h = (await client.get(f"{u}/chain/head")).json()["height"]
                heights.append(h)

            print("\n========== DELTA V SIM REPORT ==========")
            print(f"chain height: {head['height']}  (per node: {heights})")
            print(f"gateway balance: {_dvt(gw_acc['balance'])} (paid for inference)")
            print(f"receipts on chain: {len(receipts)}")
            for r in receipts:
                status = "unchecked" if not r["checked"] else ("OK" if r["check_ok"] else "SLASHED")
                print(f"  {r['receipt_hash'][:12]}... node={r['node'][:12]}... "
                      f"tokens={r['tokens']} paid={_dvt(r['price_paid'])} [{status}]")
            print("nodes:")
            for n in nodes:
                print(f"  {n['address'][:12]}... rep={n['reputation']:.3f} "
                      f"stake={_dvt(n['stake'])} jobs={n['jobs_done']} "
                      f"vram={n['hardware'].get('vram_mb')}MB models={len(n['models'])}")
            print("========================================")

            report = {
                "height": head["height"],
                "heights": heights,
                "receipts": receipts,
                "nodes": nodes,
                "gateway_balance": gw_acc["balance"],
            }
    finally:
        for d in daemons:
            await d.stop()
        await gateway.close()
        for s in servers:
            s.should_exit = True
        for t in server_tasks:
            with contextlib.suppress(asyncio.CancelledError, asyncio.TimeoutError):
                await asyncio.wait_for(t, timeout=5.0)
    return report
