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


def _cmd_node(args: argparse.Namespace) -> int:
    import uvicorn

    from .node import NodeConfig, NodeDaemon

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
    daemon = NodeDaemon(keypair, genesis, cfg)
    print(f"node {keypair.address} on {cfg.public_url()} backend={daemon.backend.name}")

    async def run() -> None:
        server = uvicorn.Server(uvicorn.Config(
            daemon.app, host=cfg.host, port=cfg.port, log_level="warning"))
        await daemon.start()
        await server.serve()
        await daemon.stop()

    asyncio.run(run())
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
