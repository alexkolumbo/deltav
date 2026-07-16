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
        params=ChainParams(chain_id=args.chain_id, dev_fund=args.dev_fund),
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
        max_parallel_jobs=args.parallel,
        price_per_token=args.price,
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
        max_parallel_jobs=args.parallel,
        price_per_token=args.price,
    )
    _run_node(keypair, genesis, cfg)
    return 0


def _cmd_gateway(args: argparse.Namespace) -> int:
    import uvicorn

    from .gateway import GatewayDaemon

    keypair = load_or_create(args.wallet or wallet_path("gateway"))
    params = Genesis.load(args.genesis).params if args.genesis else ChainParams()
    daemon = GatewayDaemon(keypair, node_urls=args.node, params=params,
                           memory_path=args.memory_file or None,
                           keys_path=args.keys_file or None,
                           require_keys=args.require_keys)
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
        price = n.get("price_per_token", 0) or "default"
        print(f"  {n['address'][:16]}... {n['endpoint']:<28} "
              f"{hw.get('vendor', '?')}/{hw.get('vram_mb', '?')}MB "
              f"rep={n['reputation']:.3f} stake={n['stake'] / DVT:,.0f} "
              f"price={price} jobs={n['jobs_done']}{jailed}")
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


def _cmd_keys(args: argparse.Namespace) -> int:
    import httpx

    if args.action == "create":
        data = httpx.post(f"{args.gateway}/v1/keys",
                          json={"label": args.label}, timeout=15.0).json()
        print(f"api_key : {data['api_key']}   (показывается ОДИН раз)")
        print(f"address : {data['address']}")
        print(f"note    : {data['note']}")
    else:  # me
        if not args.key:
            print("нужен --key dvk_...", file=sys.stderr)
            return 1
        resp = httpx.get(f"{args.gateway}/v1/keys/me",
                         headers={"Authorization": f"Bearer {args.key}"}, timeout=15.0)
        if resp.status_code != 200:
            print(f"error {resp.status_code}: {resp.text}", file=sys.stderr)
            return 1
        d = resp.json()
        print(f"label    : {d['label'] or '—'}")
        print(f"address  : {d['address']}")
        print(f"balance  : {d['balance_udvt'] / DVT:,.6f} DVT")
        print(f"usage    : {d['requests']} запросов, {d['tokens']} токенов, "
              f"~{d['spent_udvt_estimate'] / DVT:,.6f} DVT")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .light import LightClient
    from .wallet import load_wallet

    genesis = Genesis.load(args.genesis)
    nodes = args.node or [args.gateway.replace(":9000", ":9100")]

    async def run() -> int:
        lc = LightClient(genesis, nodes)
        rc = 0
        try:
            print("== целостность цепочки ==")
            hv = await lc.verify_headers()
            print(f"  {'OK' if hv.ok else 'СБОЙ'}: проверено {hv.checked} заголовков "
                  f"до высоты {hv.height}" + (f" — {hv.error}" if hv.error else ""))
            rc |= 0 if hv.ok else 1

            head_hash, height, votes = await lc.quorum_head()
            print(f"== кворум ({len(nodes)} нод) ==")
            print(f"  голова {head_hash[:16]}… высота {height} — {votes}/{len(nodes)} согласны")

            if args.address or args.wallet:
                address = args.address
                pubkey = None
                if args.wallet:
                    kp = load_wallet(args.wallet)
                    address, pubkey = kp.address, kp.public_hex
                bal, bvotes = await lc.quorum_balance(address)
                print(f"== счёт {address[:16]}… ==")
                print(f"  баланс {bal / DVT:,.6f} DVT ({bvotes}/{len(nodes)} согласны)")
                audit = await lc.audit_charges(address, pubkey)
                print(f"== аудит списаний ({len(audit.charges)}) ==")
                for c in audit.charges:
                    mark = "✓" if c.authorized else "✗ НЕ АВТОРИЗОВАНО"
                    print(f"  {mark} h{c.height} {c.receipt_hash[:12]}… "
                          f"{c.model.split('::')[0].split('/')[-1]} "
                          f"лимит {c.price_limit} udvt")
                if not audit.all_authorized:
                    print("  ⚠ ОБНАРУЖЕНЫ НЕАВТОРИЗОВАННЫЕ СПИСАНИЯ")
                    rc |= 1
                elif audit.charges:
                    print("  все списания подписаны вашим ключом ✓")
        finally:
            await lc.close()
        return rc

    return asyncio.run(run())


def _cmd_setup(args: argparse.Namespace) -> int:
    from .setup import run_setup

    return run_setup(home=args.home, seed=args.seed, lang=args.lang,
                     auto_start=not args.no_start)


def _cmd_price(args: argparse.Namespace) -> int:
    from .economics import price_report

    r = price_report(watts=args.watts, tokens_per_sec=args.tps,
                     electricity_usd_kwh=args.kwh_usd, margin=args.margin,
                     usd_per_dvt=args.usd_per_dvt)
    print(f"node profile : {r.watts:.0f} W @ {r.tokens_per_sec:.0f} tok/s")
    print(f"energy       : {r.kwh_per_million} kWh per 1M tokens")
    print(f"electricity  : ${r.electricity_usd_kwh}/kWh (world avg default)")
    print(f"cost         : ${r.cost_usd_per_million} per 1M tokens")
    print(f"+{r.margin:.0%} margin  : ${r.price_usd_per_million} per 1M tokens")
    print(f"peg          : 1 DVT = ${r.usd_per_dvt}")
    print(f"recommended  : --price {r.suggested_price_udvt}   (udvt per token; "
          f"{r.suggested_price_udvt} DVT per 1M)")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    from .compute import detect_device
    from .router.planner import launch_hint, plan

    if args.vram:
        vram, label = args.vram, f"{args.vram} MB (given)"
    else:
        device = detect_device()
        vram = device.vram_mb
        label = f"{device.vendor}/{device.name} {vram} MB (detected)"
    print(f"hardware : {label}\nobjective: {args.objective}\n")
    options = plan(vram, objective=args.objective)
    if not options:
        print("nothing fits this VRAM budget")
        return 1
    for i, o in enumerate(options, 1):
        star = " <== native limit" if any("native" in n for n in o.notes) else ""
        print(f"{i:2}. {o.ref.split('::')[0]}")
        print(f"    {o.params_b}B {o.quant} quality={o.quality} | ctx={o.max_context:,} "
              f"kv={o.kv_type} | ~{o.est_vram_mb:,} MB{star}")
        print(f"    {launch_hint(o)}")
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
        "session_id": args.session,
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


def _cmd_tgbot(args: argparse.Namespace) -> int:
    import os

    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    os.environ["DELTAV_GATEWAY"] = args.gateway
    if args.allow:
        os.environ["DELTAV_ALLOW"] = args.allow
    from .tgbot import main as tg_main

    return tg_main()


def _cmd_connect(args: argparse.Namespace) -> int:
    from .client import DeltaVClient, Profile, load_profile, save_profile

    profile = load_profile()
    if args.url:
        profile.base_urls = [u.strip() for u in args.url.split(",") if u.strip()]
    if args.key:
        profile.api_key = args.key
    if args.model:
        profile.model = args.model
    path = save_profile(profile)
    print(f"сохранено в {path}")
    print(f"  base URLs : {', '.join(profile.base_urls)}")
    print(f"  api key   : {profile.api_key[:12]}{'…' if len(profile.api_key) > 12 else ''}")
    print(f"  model     : {profile.model}")
    try:
        client = DeltaVClient.from_profile(profile)
        h = client.health()
        models = [m["id"].split("::")[0].split("/")[-1]
                  for m in client.models() if m["deltav"]["served_by"]]
        print(f"  ✓ подключено: шлюз {h['gateway'][:14]}…, моделей на сети: {len(models)}")
        if models:
            print(f"    {', '.join(sorted(set(models)))}")
    except Exception as exc:
        print(f"  ! не удалось проверить связь: {exc}", file=sys.stderr)
    return 0


def _cmd_repl(args: argparse.Namespace) -> int:
    from .client import DeltaVClient, load_profile

    profile = load_profile()
    urls = [args.gateway] if args.gateway else profile.base_urls
    client = DeltaVClient(base_urls=urls, api_key=args.key or profile.api_key,
                          model=args.model or profile.model)
    messages: list[dict] = []
    print(f"ΔV REPL — модель {client.model}. /help для команд, /exit для выхода.")
    while True:
        try:
            text = input("\nвы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd in ("/exit", "/quit"):
                break
            if cmd == "/reset":
                messages = []; print("диалог очищен"); continue
            if cmd == "/model":
                client.model = arg.strip() or client.model; print(f"модель: {client.model}"); continue
            if cmd == "/models":
                for m in client.models():
                    if m["deltav"]["served_by"]:
                        print(" ", m["id"].split("::")[0])
                continue
            if cmd == "/agent":
                d = client.agent(arg.strip(), session_id="repl")
                for s in d.get("steps", []):
                    print(f"  🛠 {s['tool']}({s['arguments']})")
                print("ΔV:", d.get("answer", "")); continue
            if cmd == "/swarm":
                d = client.swarm(arg.strip(), mode="vote")
                for w in d["workers"]:
                    print(f"  [{w.get('model')}] {w.get('answer', w.get('error',''))[:80]}")
                if d.get("answer"):
                    print("ΔV (синтез):", d["answer"])
                continue
            print("команды: /model <ref> /models /agent <task> /swarm <task> /reset /exit")
            continue
        messages.append({"role": "user", "content": text})
        print("ΔV: ", end="", flush=True)
        acc = ""
        try:
            for chunk in client.chat_stream(messages, max_tokens=args.max_tokens):
                delta = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                if delta:
                    acc += delta; print(delta, end="", flush=True)
            print()
        except Exception as exc:
            print(f"\n! {exc}"); messages.pop(); continue
        messages.append({"role": "assistant", "content": acc})
    return 0


def _cmd_companion(args: argparse.Namespace) -> int:
    from .client import DeltaVClient, load_profile

    profile = load_profile()
    urls = [args.gateway] if args.gateway else profile.base_urls
    client = DeltaVClient(base_urls=urls, api_key=args.key or profile.api_key,
                          model=args.model or profile.model)
    history: list[dict] = []
    if args.feedback:
        d = client.companion_feedback(args.feedback)
        print(f"запомнено как обратная связь ({d['stored']}) для {d['user']}")
        return 0
    print("ΔV Companion — персональный агент. Помнит вас между сессиями.")
    print("  /memory — что я о вас помню · /feedback <текст> · /exit")
    while True:
        try:
            text = input("\nвы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/memory":
            m = client.companion_memory()
            print(f"  ({m['user']}, {'ключ' if m['authenticated'] else 'локально'})")
            for it in m["items"][-15:]:
                kind = it.get("meta", {}).get("kind", "note")
                print(f"  · [{kind}] {it['text'][:90]}")
            continue
        if text.startswith("/feedback "):
            client.companion_feedback(text[len("/feedback "):])
            print("  учту это на будущее.")
            continue
        d = client.companion(text, history=history)
        for s in d.get("steps", []):
            print(f"  🛠 {s['tool']}({s['arguments']})")
        print("ΔV:", d["answer"])
        if d.get("learned"):
            print(f"  💡 запомнил: {d['learned']}")
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": d["answer"]})
    return 0


def _cmd_swarm(args: argparse.Namespace) -> int:
    from .client import DeltaVClient, load_profile

    profile = load_profile()
    urls = [args.gateway] if args.gateway else profile.base_urls
    client = DeltaVClient(base_urls=urls, api_key=args.key or profile.api_key)
    d = client.swarm(args.task, n=args.n, mode=args.mode, max_tokens=args.max_tokens)
    for w in d["workers"]:
        head = w.get("answer", w.get("error", ""))
        print(f"[{w.get('model')}] node={str(w.get('node',''))[:12]}")
        print(f"  {head[:300].strip()}")
    if d.get("answer"):
        print(f"\n=== синтез ({args.mode}) ===\n{d['answer'].strip()}")
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
    p_gen.add_argument("--dev-fund", default="",
                       help="address receiving the dev share of the chain pool")
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
    p_node.add_argument("--parallel", type=int, default=1,
                        help="concurrent inference jobs (keep 1 per GPU)")
    p_node.add_argument("--price", type=int, default=0,
                        help="asking price in udvt per token (0 = network default)")
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
    p_join.add_argument("--parallel", type=int, default=1,
                        help="concurrent inference jobs (keep 1 per GPU)")
    p_join.add_argument("--price", type=int, default=0,
                        help="asking price in udvt per token (0 = network default)")
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
    p_gw.add_argument("--memory-file", default="",
                      help="persist agent session memory to this jsonl file")
    p_gw.add_argument("--keys-file", default="",
                      help="persist API keys (custodial billing wallets) here")
    p_gw.add_argument("--require-keys", action="store_true",
                      help="reject requests without a funded dvk_ API key")
    p_gw.set_defaults(func=_cmd_gateway)

    p_keys = sub.add_parser("keys", help="gateway API keys (billing wallets)")
    p_keys.add_argument("action", choices=["create", "me"])
    p_keys.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_keys.add_argument("--label", default="")
    p_keys.add_argument("--key", default="", help="dvk_... for 'me'")
    p_keys.set_defaults(func=_cmd_keys)

    p_sim = sub.add_parser("sim", help="run a local simulated network")
    p_sim.add_argument("--nodes", type=int, default=3)
    p_sim.add_argument("--duration", type=float, default=25.0)
    p_sim.add_argument("--base-port", type=int, default=9100)
    p_sim.set_defaults(func=_cmd_sim)

    p_verify = sub.add_parser(
        "verify", help="light-client verification: chain integrity + your charges")
    p_verify.add_argument("--genesis", required=True)
    p_verify.add_argument("--node", action="append", default=[],
                          help="node base URL for quorum (repeatable)")
    p_verify.add_argument("--gateway", default="http://127.0.0.1:9000",
                          help="fallback if no --node given")
    p_verify.add_argument("--address", default="", help="account to audit")
    p_verify.add_argument("--wallet", default="",
                          help="wallet file: audits its address AND checks signatures")
    p_verify.set_defaults(func=_cmd_verify)

    p_setup = sub.add_parser(
        "setup", help="friendly wizard: bare machine -> live node, step by step")
    p_setup.add_argument("--home", default="", help="install directory (default ~/deltav-node)")
    p_setup.add_argument("--seed", default="", help="a live node URL to join")
    p_setup.add_argument("--lang", default="", choices=["", "en", "ru"],
                         help="interface language (default: auto-detect)")
    p_setup.add_argument("--no-start", action="store_true",
                         help="prepare everything but don't launch (writes a start script)")
    p_setup.set_defaults(func=_cmd_setup)

    p_price = sub.add_parser(
        "price", help="cost-anchored pricing: electricity + 50% service margin")
    p_price.add_argument("--watts", type=float, default=150.0,
                         help="system power draw under inference")
    p_price.add_argument("--tps", type=float, default=30.0, help="tokens per second")
    p_price.add_argument("--kwh-usd", type=float, default=0.155,
                         help="your electricity price, USD/kWh")
    p_price.add_argument("--margin", type=float, default=0.5)
    p_price.add_argument("--usd-per-dvt", type=float, default=0.032,
                         help="DVT reference peg")
    p_price.set_defaults(func=_cmd_price)

    p_plan = sub.add_parser(
        "plan", help="hardware-aware model planner: what should this machine run?")
    p_plan.add_argument("--vram", type=int, default=0, help="VRAM MB (default: detect)")
    p_plan.add_argument("--objective", default="balanced",
                        choices=["balanced", "max_context", "max_quality"])
    p_plan.set_defaults(func=_cmd_plan)

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
    p_agent.add_argument("--session", default="",
                         help="session id: enables remember/recall memory tools")
    p_agent.set_defaults(func=_cmd_agent)

    p_tg = sub.add_parser("tgbot", help="Telegram bot bridge to a gateway")
    p_tg.add_argument("--token", default="", help="or TELEGRAM_BOT_TOKEN env")
    p_tg.add_argument("--gateway", default="http://127.0.0.1:9000")
    p_tg.add_argument("--allow", default="", help="comma-separated Telegram user ids")
    p_tg.set_defaults(func=_cmd_tgbot)

    p_conn = sub.add_parser("connect", help="save how to reach the network (base URL + key)")
    p_conn.add_argument("--url", default="", help="gateway base URL(s), comma-separated for failover")
    p_conn.add_argument("--key", default="", help="API key (dvk_... or any)")
    p_conn.add_argument("--model", default="", help="default model (auto)")
    p_conn.set_defaults(func=_cmd_connect)

    p_repl = sub.add_parser("repl", help="interactive multi-turn chat")
    p_repl.add_argument("--gateway", default="", help="override saved base URL")
    p_repl.add_argument("--key", default="")
    p_repl.add_argument("--model", default="")
    p_repl.add_argument("--max-tokens", type=int, default=800)
    p_repl.set_defaults(func=_cmd_repl)

    p_comp = sub.add_parser("companion", help="persistent per-user agent with memory")
    p_comp.add_argument("--gateway", default="")
    p_comp.add_argument("--key", default="", help="your dvk_ key = your identity + memory")
    p_comp.add_argument("--model", default="")
    p_comp.add_argument("--feedback", default="", help="store one feedback note and exit")
    p_comp.set_defaults(func=_cmd_companion)

    p_swarm = sub.add_parser("swarm", help="fan a task across several models/nodes in parallel")
    p_swarm.add_argument("task")
    p_swarm.add_argument("--gateway", default="")
    p_swarm.add_argument("--key", default="")
    p_swarm.add_argument("-n", type=int, default=3, help="number of models")
    p_swarm.add_argument("--mode", default="vote", choices=["fanout", "vote", "map"])
    p_swarm.add_argument("--max-tokens", type=int, default=400)
    p_swarm.set_defaults(func=_cmd_swarm)

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
