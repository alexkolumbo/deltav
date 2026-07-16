# Running a Delta V node (hosting)

[English](HOSTING.md) · [Русский](HOSTING.ru.md)

A node is a computer that answers AI requests for the network and earns
DVT tokens. Any machine works — a gaming PC, a laptop, a server — on
**AMD, NVIDIA, Intel, Apple or CPU**. This guide takes you from nothing to
a live, earning node.

## What you need

- A computer with **Python 3.11+** (Windows, Linux or macOS).
- A **GPU with ≥6 GB VRAM** is ideal (any vendor); a CPU works but is slow.
- The URL of one **live node** to join (a "seed"), or you start the first one.
- ~5–15 GB of disk for the model.

## The easy way — one command

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.sh | sh
# Windows (PowerShell)
irm https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.ps1 | iex
```

This runs the **setup wizard** (bilingual). It will, explaining each step:

1. **Detect your hardware** (GPU or CPU, how much memory).
2. **Recommend a model** that fits your VRAM (ranked from the model DB;
   you can accept it, pick another, or paste your own HuggingFace repo —
   analyzed for fit).
3. **Download the engine** — the right prebuilt `llama.cpp` binary for your
   OS+GPU (Vulkan for AMD/NVIDIA/Intel, Metal for Apple, CPU fallback).
   Nothing is compiled; no CUDA/ROCm needed.
4. **Download the model.**
5. **Create a wallet** — the address your earnings go to. Back up this file.
6. **Join the network** — fetches the genesis from your seed node.
7. **Set a price** — cost-anchored (your electricity + 50%), or your own.
8. **Launch** — starts the engine + node and prints your dashboard URL.

It also writes a `start-node` script so next time you launch with one command.

## If you already have Python

```bash
pip install "deltav-network[gpu,hub] @ git+https://github.com/alexkolumbo/deltav"
deltav setup --seed http://<any-live-node>:9100
```

## Manual control (`deltav join` / `deltav node`)

For full control over model, backend and peers:

```bash
# 1) run a local llama.cpp server holding your model (Vulkan build = any GPU)
llama-server -m <model.gguf> --host 127.0.0.1 --port 8085 -ngl 99 -c 8192
#    vision model? add:  --mmproj <mmproj.gguf>
#    reasoning model?    thinking is off by default; DELTAV_THINK=1 to keep it

# 2) run the node, pointing at that server
deltav node --genesis genesis.json --wallet node.wallet.json \
    --host 0.0.0.0 --port 9100 --endpoint http://<your-ip>:9100 \
    --peer http://<seed>:9100 \
    --backend llamaserver --model "<org/repo::file.gguf>" \
    --data-dir ./data --price 9
```

`deltav join --seed http://<node>:9100` does the hardware→model→download→run
steps automatically if you'd rather not pick everything by hand.

## Starting the very first node (your own network)

```bash
deltav wallet new --file node.wallet.json
deltav genesis --alloc dv1...=100000 --stake dv1...=10000 \
    --dev-fund dv1... -o genesis.json
deltav join --genesis genesis.json --wallet node.wallet.json --port 9100
# then run a gateway so clients can connect (OpenAI/Ollama/Anthropic):
deltav gateway --genesis genesis.json --node http://127.0.0.1:9100 --port 9000
```

## Make it reachable (external access)

Other nodes reach you at a public address. `deltav join` works this out for
you — **no port-forwarding, no manual IP, no TLS setup**:

- **`--connect auto`** (the default for `join`) — the node learns its public
  address from a peer, self-tests whether it's directly reachable, and if so
  announces it. Behind NAT/CGNAT it instead opens an *outbound* tunnel to a
  public node that advertises as a relay and is handed a public URL
  `https://<relay>/via/<node-id>` — reachable with **zero inbound ports**.
- **`--connect direct`** — force the detected public address (you have a
  public IP or a port-forward). Open the ports:

  ```powershell
  # Windows (PowerShell as admin) — node 9100, gateway 9000
  New-NetFirewallRule -DisplayName "DeltaV" -Direction Inbound `
    -Protocol TCP -LocalPort 9000,9100 -Action Allow
  ```
  ```bash
  # Linux (ufw)
  sudo ufw allow 9000,9100/tcp
  ```

- **`--connect relay`** / **`--relay-via <url>`** — always tunnel through a
  relay (useful on CGNAT or a laptop that moves networks).
- **`--connect local`** — LAN / dev only; announce `host:port` as-is.

You can still set `--endpoint http://<addr>:9100` to name your address
explicitly; it's always trusted as-is.

### Run a relay (help NAT'd nodes join through you)

Any node with a public address can volunteer as a relay so nodes behind NAT
reach the network through it — this is what keeps external access
**decentralized**, with no third-party tunnel service:

```bash
deltav node ... --relay --relay-url https://relay.example.com
```

A relay only forwards **signed** traffic; it cannot read or forge chain
messages (every tx, block and receipt is signed), and it only tunnels for a
node that cryptographically proves it owns its identity. Put TLS in front of
a public relay (your existing Caddy / reverse proxy) so relayed nodes serve
over `https://`.

**Security, built in:** gossip is rate-limited per IP and request bodies are
capped; a peer URL learned from gossip is trusted only after it proves it's
on your chain; payment always needs the requester's signature over a price
cap. Watch your node at **`http://<addr>:9100/explorer`**.

## Choosing a model

```bash
deltav plan                       # best models for your hardware + launch commands
deltav registry list --vram 8176  # the full ranked DB (catalog + HuggingFace)
deltav registry sync              # pull fresh models from HuggingFace
deltav registry add <org/repo>    # add a specific HF model to the DB
```

On 8 GB: Qwen2.5-7B (32k context) is a great practical default; Llama-3.2-3B
reaches 128k context; reasoning/vision models (DeepSeek-R1, Qwythos-9B) work
but think a lot and are tighter.

## Pricing & earnings

```bash
deltav price --watts 130 --tps 30 --kwh-usd 0.10   # your recommended price
```

Price is anchored in cost: electricity + a 50% service margin, using the
world-average electricity rate by default. Cheaper power or a faster node →
lower price → more traffic. You're paid in DVT per token via on-chain
receipts; 10% of each payment funds the network pool, distributed each epoch
to working nodes and the dev fund.

## Keeping it running

- The chain persists to `--data-dir`; a restart restores it from disk.
- On Linux, run the `start-node` script under `systemd` or `tmux`.
- On Windows, the wizard's `start-node.bat` reopens the engine + node.
- Watch your node at **`http://<your-ip>:9100/explorer`** — height, load,
  reputation, receipts, earnings.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Wizard: "no prebuilt binary" | Install `llama.cpp` manually and use `deltav node` |
| Node won't earn | Check it's announced: `deltav network`; ensure the gateway routes to it |
| Model gives empty answers | Reasoning model — thinking is disabled by default; raise `--max-tokens` on the client, or `DELTAV_THINK=1` on the server |
| Vision doesn't work | Start `llama-server` with `--mmproj <projector.gguf>` |
| Out of memory on load | Pick a smaller model (`deltav plan`) or lower `-c` (context) |
| Restart lost the chain | It re-syncs from a `--peer`; make sure one is set |
