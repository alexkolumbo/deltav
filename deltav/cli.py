"""Delta V command-line interface.

  deltav wallet new [--file PATH]           create a wallet
  deltav wallet show [--file PATH]          print address
  deltav genesis --alloc addr=DVT ... -o F  write a genesis file
  deltav node --genesis F --wallet W ...    run a node daemon
  deltav gateway --genesis F --wallet W ... run the OpenAI-compatible gateway
  deltav sim [--nodes 3] [--duration 30]    run a local simulated network
  deltav chat "prompt" [--gateway URL]      one-shot chat through the network
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import DVT, ChainParams, Genesis
from .crypto import KeyPair
from .wallet import load_or_create, load_wallet, save_wallet, wallet_path


def _cmd_wallet(args: argparse.Namespace) -> int:
    path = args.file or wallet_path()
    if args.action == "new":
        keypair = KeyPair.generate()
        save_wallet(keypair, path)
        print(f"wallet written to {path}\naddress: {keypair.address}")
    else:
        keypair = load_wallet(path)
        print(f"address: {keypair.address}")
    return 0


def _cmd_genesis(args: argparse.Namespace) -> int:
    def parse_pairs(pairs: list[str]) -> dict[str, int]:
        out: dict[str, int] = {}
        for pair in pairs:
            addr, _, amount = pair.partition("=")
            out[addr] = int(float(amount) * DVT)
        return out

    genesis = Genesis(
        params=ChainParams(chain_id=args.chain_id),
        alloc=parse_pairs(args.alloc),
        stakes=parse_pairs(args.stake),
    )
    genesis.save(args.output)
    print(f"genesis written to {args.output} "
          f"({len(genesis.alloc)} allocations, {len(genesis.stakes)} genesis stakes)")
    return 0


def _run_node(keypair, genesis: Genesis, cfg) -> None:
    import uvicorn

    from .node import NodeDaemon

    daemon = NodeDaemon(keypair, genesis, cfg)
    print(f"node     : {keypair.address}")
    print(f"listen   : {cfg.public_url()}  (explorer: {cfg.public_url()}/explorer)")
    print(f"backend  : {daemon.backend.name}  device: {daemon.device.vendor}/{daemon.device.name}")

    async def run() -> None:
        server = uvicorn.Server(uvicorn.Config(
            daemon.app, host=cfg.host, port=cfg.port, log_level="warning"))
        await daemon.start()
        await server.serve()
        await daemon.stop()

    asyncio.run(run())


def _cmd_node(args: argparse.Namespace) -> int:
    from .node import NodeConfig

    keypair = load_or_create(args.wallet or wallet_path("node"))
    genesis = Genesis.load(args.genesis)
    cfg = NodeConfig(
        host=args.host,
        port=args.port,
        endpoint=args.endpoint,
        peers=args.peer,
        backend=args.backend,
        models=args.model,
        stake=int(args.stake * DVT),
        data_dir=args.data_dir,
    )
    _run_node(keypair, genesis, cfg)
    return 0


def _cmd_join(args: argparse.Namespace) -> int:
    """One command: wallet -> genesis from seed -> hardware -> model -> node."""
    from pathlib import Path

    from .bootstrap import describe_plan, download_model, fetch_genesis, pick_model_for_device
    from .compute import detect_device
    from .node import NodeConfig

    if not args.seed and not args.genesis:
        print("join needs --seed <node-url> or --genesis <file>", file=sys.stderr)
        return 1
    keypair = load_or_create(args.wallet or wallet_path("node"))
    if args.genesis:
        genesis = Genesis.load(args.genesis)
    else:
        print(f"fetching genesis from seed {args.seed} ...")
        genesis = asyncio.run(fetch_genesis(args.seed))
    print(f"chain    : {genesis.params.chain_id}")

    device = detect_device()
    spec = pick_model_for_device(device)
    print(describe_plan(device, spec, args.backend))

    if spec and not args.no_download and args.backend not in ("mock",):
        print(f"downloading {spec.filename} ({spec.file_mb} MB) from HuggingFace ...")
        path = download_model(spec)
        print(f"model at : {path}" if path
              else "huggingface_hub not installed — model will download on first load")

    if keypair.address not in genesis.alloc and args.stake > 0:
        print(f"NOTE: fund {keypair.address} first (deltav send), staking needs balance.")

    data_dir = args.data_dir or str(Path.home() / ".deltav" / "node-data")
    cfg = NodeConfig(
        host=args.host,
        port=args.port,
        endpoint=args.endpoint,
        peers=[args.seed] if args.seed else [],
        backend=args.backend,
        models=[spec.ref] if spec else [],
        stake=int(args.stake * DVT),
        data_dir=data_dir,
    )
    _run_node(keypair, genesis, cfg)
    return 0


def _cmd_gateway(args: argparse.Namespace) -> int:
    import uvicorn

    from .gateway import GatewayDaemon

    keypair = load_or_create(args.wallet or wallet_path("gateway"))
    params = Genesis.load(args.genesis).params if args.genesis else ChainParams()
    daemon = GatewayDaemon(keypair, node_urls=args.node, params=params)
    print(f"gateway {keypair.address} on http://{args.host}:{args.port} -> nodes {args.node}")

    async def run() -> None:
        server = uvicorn.Server(uvicorn.Config(
            daemon.app, host=args.host, port=args.port, log_level="warning"))
        await server.serve()
        await daemon.close()

    asyncio.run(run())
    return 0


def _cmd_sim(args: argparse.Namespace) -> int:
    from .sim import run_simulation

    asyncio.run(run_simulation(n_nodes=args.nodes, duration=args.duration, base_port=args.base_port))
    return 0


def _cmd_balance(args: argparse.Namespace) -> int:
    import httpx

    data = httpx.get(f"{args.node}/chain/account/{args.address}", timeout=10.0).json()
    print(f"address : {args.address}")
    print(f"balance : {data['balance'] / DVT:,.6f} DVT")
    print(f"stake   : {data['stake'] / DVT:,.6f} DVT")
    return 0


def _cmd_send(args: argparse.Namespace) -> int:
    import httpx

    from .chain.transaction import Tx, TxType

    keypair = load_wallet(args.wallet or wallet_path())
    acc = httpx.get(f"{args.node}/chain/account/{keypair.address}", timeout=10.0).json()
    tx = Tx(
        type=TxType.TRANSFER.value,
        sender=keypair.address,
        nonce=int(acc["nonce"]),
        payload={"to": args.to, "amount": int(args.amount * DVT)},
    ).sign(keypair)
    resp = httpx.post(f"{args.node}/tx", json=tx.to_dict(), timeout=10.0).json()
    print(f"tx {tx.hash[:16]}... accepted={resp.get('accepted')}")
    return 0


def _cmd_network(args: argparse.Namespace) -> int:
    import httpx

    stats = httpx.get(f"{args.node}/chain/stats", timeout=10.0).json()
    nodes = httpx.get(f"{args.node}/chain/nodes", timeout=10.0).json()["nodes"]
    print(f"chain {stats['chain_id']}  height={stats['height']}  "
          f"supply={stats['supply'] / DVT:,.0f} DVT  validators={stats['validators']}")
    for n in nodes:
        hw = n["hardware"]
        jailed = " JAILED" if n.get("jailed_until", 0) > stats["height"] else ""
        print(f"  {n['address'][:16]}... {n['endpoint']:<28} "
              f"{hw.get('vendor', '?')}/{hw.get('vram_mb', '?')}MB "
              f"rep={n['reputation']:.3f} stake={n['stake'] / DVT:,.0f} "
              f"jobs={n['jobs_done']}{jailed}")
        for m in n["models"]:
            print(f"      - {m}")
    return 0


def _cmd_models(args: argparse.Namespace) -> int:
    import httpx

    data = httpx.get(f"{args.gateway}/v1/models", timeout=30.0).json()["data"]
    for m in data:
        d = m["deltav"]
        served = f"{len(d['served_by'])} node(s)" if d["served_by"] else "no live nodes"
        print(f"{m['id']}\n    {d['params_b']}B {d['quant']} "
              f"~{d['vram_needed_mb']}MB quality={d['quality']} -> {served}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    import httpx

    data = httpx.get(f"{args.gateway}/v1/search",
                     params={"q": args.query, "max_results": args.max_results},
                     timeout=30.0).json()
    for i, r in enumerate(data["results"], 1):
        print(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    if not data["results"]:
        print("no results")
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    import httpx

    resp = httpx.post(f"{args.gateway}/v1/agents/run", json={
        "task": args.task,
        "model": args.model,
        "max_steps": args.max_steps,
    }, timeout=600.0)
    if resp.status_code != 200:
        print(f"error {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1
    data = resp.json()
    for i, step in enumerate(data["steps"], 1):
        print(f"[step {i}] {step['tool']}({json.dumps(step['arguments'], ensure_ascii=False)})",
              file=sys.stderr)
        print(f"          -> {step['result'][:200]}", file=sys.stderr)
        if step.get("receipt_tx"):
            print(f"          receipt={step['receipt_tx'][:16]} node={step['node'][:16]}",
                  file=sys.stderr)
    print(data["answer"])
    if not data["finished"]:
        print("(step limit reached)", file=sys.stderr)
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    import httpx

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
    }
    resp = httpx.post(f"{args.gateway}/v1/chat/completions", json=body, timeout=300.0)
    if resp.status_code != 200:
        print(f"error {resp.status_code}: {resp.text}", file=sys.stderr)
        return 1
    data = resp.json()
    print(data["choices"][0]["message"]["content"])
    meta = data.get("deltav", {})
    usage = data.get("usage", {})
    print(
        f"\n-- model={data.get('model')} node={meta.get('node', '')[:16]} "
        f"tokens={usage.get('total_tokens')} receipt={str(meta.get('receipt_tx'))[:16]}",
        file=sys.stderr,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deltav", description="Delta V decentralized AI network")
    sub = parser.add_subparsers(dest="command", required=True)

    p_wallet = sub.add_parser("wallet", help="manage wallets")
    p_wallet.add_argument("action", choices=["new", "show"])
    p_wallet.add_argument("--file", default=None)
    p_wallet.set_defaults(func=_cmd_wallet)

    p_gen = sub.add_parser("genesis", help="write a genesis file")
    p_gen.add_argument("--alloc", action="append", default=[], metavar="ADDR=DVT")
    p_gen.add_argument("--stake", action="append", default=[],
                       metavar="ADDR=DVT", help="genesis validator stake")
    p_gen.add_argument("--chain-id", default="deltav-local-1")
    p_gen.add_argument("-o", "--output", default="genesis.json")
    p_gen.set_defaults(func=_cmd_genesis)

    p_node = sub.add_parser("node", help="run a node daemon")
    p_node.add_argument("--genesis", required=True)
    p_node.add_argument("--wallet", default=None)
    p_node.add_argument("--host", default="127.0.0.1")
    p_node.add_argument("--port", type=int, default=9100)
    p_node.add_argument("--endpoint", default="", help="public URL other nodes reach you at")
    p_node.add_argument("--peer", action="append", default=[], help="peer base URL (repeatable)")
    p_node.add_argument("--backend", default="auto", help="auto | llamacpp | mock | groq | asic")
    p_node.add_argument("--model", action="append", default=[], help="model ref to announce (repeatable)")
    p_node.add_argument("--stake", type=float, default=0.0, help="DVT to stake at startup")
    p_node.add_argument("--data-dir", default="", help="persist the chain to this directory")
    p_node.set_defaults(func=_cmd_node)

    p_join = sub.add_parser(
        "join", help="one-command node bootstrap: hardware -> model -> download -> run")
    p_join.add_argument("--seed", default="", help="URL of any existing node (fetches genesis + peers)")
    p_join.add_argument("--genesis", default="", help="local genesis file (instead of --seed fetch)")
    p_join.add_argument("--wallet", default=None)
    p_join.add_argument("--host", default="0.0.0.0")
    p_join.add_argument("--port", type=int, default=9100)
    p_join.add_argument("--endpoint", default="", help="public URL other nodes reach you at")
    p_join.add_argument("--backend", default="auto")
    p_join.add_argument("--stake", type=float, default=0.0)
    p_join.add_argument("--no-download", action="store_true")
    p_join.add_argument("--data-dir", default="")
    p_join.set_defaults(func=_cmd_join)

    p_bal = sub.add_parser("balance", help="show an account")
    p_bal.add_argument("address")
    p_bal.add_argument("--node", default="http://127.0.0.1:9100")
    p_bal.set_defaults(func=_cmd_balance)

    p_send = sub.add_parser("send", help="transfer DVT")
    p_send.add_argument("--to", required=True)
    p_send.add_argument("--amount", type=float, required=True, help="DVT")
    p_send.add_argument("--wallet", default=None)
    p_send.add_argument("--node", default="http://127.0.0.1:9100")
    p_send.set_defaults(func=_cmd_send)

    p_net = sub.add_parser("network", help="network overview from a node")
    p_net.add_argument("--node", default="http://127.0.0.1:9100")
    p_net.set_defaults(func=_cmd_network)

    p_models = sub.add_parser("models", help="models served by the network")
    p_models.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_models.set_defaults(func=_cmd_models)

    p_gw = sub.add_parser("gateway", help="run the OpenAI-compatible gateway")
    p_gw.add_argument("--genesis", default=None)
    p_gw.add_argument("--wallet", default=None)
    p_gw.add_argument("--host", default="127.0.0.1")
    p_gw.add_argument("--port", type=int, default=9000)
    p_gw.add_argument("--node", action="append", required=True, help="node base URL (repeatable)")
    p_gw.set_defaults(func=_cmd_gateway)

    p_sim = sub.add_parser("sim", help="run a local simulated network")
    p_sim.add_argument("--nodes", type=int, default=3)
    p_sim.add_argument("--duration", type=float, default=25.0)
    p_sim.add_argument("--base-port", type=int, default=9100)
    p_sim.set_defaults(func=_cmd_sim)

    p_search = sub.add_parser("search", help="internet search through the gateway")
    p_search.add_argument("query")
    p_search.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_search.add_argument("--max-results", type=int, default=5)
    p_search.set_defaults(func=_cmd_search)

    p_agent = sub.add_parser("agent", help="run a tool-using agent on the network")
    p_agent.add_argument("task")
    p_agent.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_agent.add_argument("--model", default="auto")
    p_agent.add_argument("--max-steps", type=int, default=6)
    p_agent.set_defaults(func=_cmd_agent)

    p_chat = sub.add_parser("chat", help="one-shot chat through the network")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_chat.add_argument("--model", default="auto")
    p_chat.add_argument("--max-tokens", type=int, default=256)
    p_chat.set_defaults(func=_cmd_chat)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
