# Delta V — decentralized AI network

**English** · [Русский](README.ru.md)

A lightweight proof-of-stake blockchain + a network of GPU nodes +
smart routing of open-source models from
[HuggingFace](https://huggingface.co/models). A user sends a request to
an OpenAI/Ollama/Anthropic-compatible gateway; the network picks the best
model that physically fits the VRAM of a live node, runs the inference,
records a receipt on-chain, pays the node in DVT tokens, and randomly
re-verifies its honesty.

```
client ──► gateway (/v1/chat/completions · /api/chat · /v1/messages)
              │  SmartRouter: model ⨯ node (VRAM-fit, reputation, stake, price, load)
              ▼
        node daemon ──► ComputeBackend (llama.cpp/Vulkan on AMD·NVIDIA·Intel·CPU; Groq; …)
              │
              ▼
        Delta V chain: INFERENCE_RECEIPT (payment) ──► SPOT_CHECK (re-run / slashing)
```

Vendor-agnostic by design: the same GGUF runs on **AMD, NVIDIA, Intel,
Apple or CPU** via llama.cpp (Vulkan needs no CUDA/ROCm). Only
**non-proprietary** models are served — Qwen, Llama, Gemma, Mistral,
DeepSeek, xAI's open Grok weights, or your own. The API *shape* is a
translation layer, independent of the model; the network speaks the
surfaces the open ecosystem uses: **OpenAI** (the de-facto standard —
Hermes, grok-build, goose, opencode), **Ollama** (local-model tools), and
**Anthropic** (an extra surface for `anthropic`-SDK clients).

## Get a node running (5 minutes, anyone) — `deltav setup`

A friendly bilingual (EN/RU) wizard takes you from a bare machine to a
live, earning node — explaining each step. It detects your hardware,
**downloads the right prebuilt llama.cpp binary** for your OS+GPU (Vulkan
for AMD/NVIDIA/Intel, Metal for Apple, CPU fallback — no compiling),
picks and downloads a fitting model, creates a wallet, joins the network,
computes a cost-anchored price, and launches everything.

```bash
# Linux / macOS
curl -fsSL https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.sh | sh
# Windows (PowerShell)
irm https://raw.githubusercontent.com/alexkolumbo/deltav/main/install.ps1 | iex

# or, if you already have Python 3.11+:
pip install "deltav-network[gpu,hub] @ git+https://github.com/alexkolumbo/deltav"
deltav setup --seed http://<any-live-node>:9100
```

You can accept the recommended model or paste your own HuggingFace repo —
**analyzed** for fit (verdict + max context) or **forced** as-is.

## Connect a client

The gateway is a full **OpenAI-compatible** endpoint with streaming (SSE),
tool calling and billing, plus **Ollama** and **Anthropic** surfaces.

| Client expects | Point it at | Notes |
|---|---|---|
| OpenAI | `http://<gw>:9000/v1` | Hermes, grok-build, goose, opencode, openai SDK — most tools |
| Ollama | `http://<gw>:9000` | Open WebUI, LangChain-Ollama, desktop apps |
| Anthropic | `http://<gw>:9000` | any `anthropic`-SDK client |

```bash
# OpenAI
curl http://<gw>:9000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"stream":true}'
# Ollama
curl http://<gw>:9000/api/chat \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

**Full guides**: [Hosting a node](docs/HOSTING.md) ([RU](docs/HOSTING.ru.md)) ·
[Using as a client](docs/CLIENT.md) ([RU](docs/CLIENT.ru.md)) ·
per-tool recipes and the `DeltaVClient` SDK in
[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md).

### Delta V's own client tools

```bash
deltav connect --url http://gw1:9000,http://gw2:9000 --key dvk_… --model auto
deltav repl                     # interactive streaming chat; /agent /swarm /model
deltav swarm "compare two approaches" --mode vote -n 3   # fan across models/nodes
deltav companion                # persistent per-user agent with memory + self-improvement
deltav chat "Explain delta-v"   # one-shot
```

**Companion** is a stateful agent with a personal memory layer and
self-improvement, under **strict per-user isolation** — identity comes
from your key, so no user can ever reach another's memory. It recalls your
past facts/learnings each turn and reflects to store new ones, so even
small models get better per user over time (`/v1/companion/*`).

## Model planner

The network tells hardware what to run — exact per-architecture memory
math (weights + KV cache by layers × KV-heads × head_dim × cache type +
buffers):

```bash
deltav plan                          # detect hardware -> ranked options + launch commands
deltav plan --objective max_context  # the longest context this hardware can hold
GET http://<gw>:9000/v1/plan?vram_mb=8176   # + which models are already warm on the net
```

### Auto-updating model database

A unified registry merges the curated catalog, models discovered from
HuggingFace, and models live nodes serve — the wizard ranks from it:

```bash
deltav registry sync                 # HF bot: pull trending GGUF repos into the DB
deltav registry sync --daemon        # keep it fresh on an interval
deltav registry add <org/repo>       # add one HF repo (reads size/quant/arch/vision)
deltav registry list --vram 8176     # ranked models that fit your VRAM
GET http://<gw>:9000/v1/registry?vram_mb=8176   # the unified DB over HTTP
```

**Reasoning models** (DeepSeek-R1, Qwythos…) think before answering. On
weak GPUs that burns the token budget, so the backend disables thinking by
default (direct answers via `enable_thinking:false`); set `DELTAV_THINK=1`
to keep the chain of thought.

**Vision (multimodal)** works end to end: start the model's llama-server
with `--mmproj <projector.gguf>`, and OpenAI vision messages (content
blocks with `image_url`) flow gateway → node → model. Example: Qwythos-9B
on an RX 6600M correctly answered the colour of an image in ~2 s.

On 8 GB (RX 6600M): Qwen2.5-7B fits its native 32k context; Llama-3.2-3B
reaches its full 128k (quantized KV). The catalog ships curated chat
models (0.5B–70B: Qwen2.5, Llama 3.x, Gemma 2, Phi, Mistral-Nemo/Small,
DeepSeek-R1-Distill, xAI's open Grok…) with architecture facts, plus
embedding, **multimodal (vision)** and **diffusion** models.

## Model types beyond text

Groundwork (works via the mock backend; a real engine implements one
`ComputeBackend` interface):

- **Multimodal / vision** — models that take image input
  (`ModelSpec.vision`, `InferRequest.images`); Qwen2.5-VL in the catalog.
- **Diffusion (text → image)** — `kind="image"`, an OpenAI-compatible
  `POST /v1/images/generations` routed to diffusion nodes, paid and
  spot-checked like text; Stable Diffusion 1.5 and FLUX.1 in the catalog
  (via stable-diffusion.cpp / GGUF).
- **A network-native model + RL training** — scaffolding in
  `deltav/training/`: every receipt is a (prompt, output) pair, every
  spot-check is a reward signal (verified / slashed) — so the dataset and
  reward already exist on-chain; `TrainingCoordinator` defines a training
  round for a powerful node to run in a later phase.

## Tokenomics

Price is anchored in cost: **a node operator recovers electricity and
earns a 50% service margin**. Everything derives from three numbers:

```
J/token    = watts / (tokens/sec)
kWh per 1M = watts / (tps × 3.6)
$/1M       = kWh × electricity_price × 1.5
```

The electricity coefficient is the **world-average** household price
($0.155/kWh) — a decentralized network isn't tied to one country's tariff,
and each operator computes their own (`deltav price`) and sets it on the
market (`--price`). Cheaper power → lower price → more traffic. Reference
peg: a 150 W / 30 tok/s node at the world average costs $0.32/1M with the
margin → the default network price of 10 udvt/token = 10 DVT per 1M →
**1 DVT ≈ $0.032**.

**Chain pool**: `pool_fee_bps` (10%) of every inference payment accrues
to an on-chain pool; every `epoch_blocks` it's distributed — `dev_share_bps`
(30%) to a dev fund, the rest to nodes pro-rata to tokens served that epoch.

**Billing**: an API key (`dvk_…`) is a custodial on-chain wallet. Fund its
address; every request bearing the key is paid from it on-chain, so
receipts charge the consumer and the gateway wallet stays untouched.
`POST /v1/keys`, `deltav keys create/me`.

## Trust & consensus

- **Payment auth**: a node can only claim payment with the requester's
  signature over `(request_hash, node, model, price_limit)` — no stealing
  or exceeding the cap.
- **Spot-check**: validators re-run a sampled fraction of jobs and compare
  the output hash; lying burns a slice of stake and reputation. GPU/API
  (non-deterministic) backends use fuzzy token-count verification.
- **Liveness**: stake-weighted proposer with RANDAO randomization; fallback
  slots keep blocks coming if a proposer is silent; absentees are jailed.
- **Light client** (`deltav verify`): from a trusted genesis, verify chain
  integrity (signatures, RANDAO), a quorum of nodes, and that every charge
  against your key was actually authorized by your key.

## Client tools

```bash
deltav explorer   →  http://<node>:9100/explorer     # dashboard: nodes, blocks, receipts, pool
deltav chat "…"                # web chat UI at /chat; Telegram bot in deltav/tgbot.py
deltav network / models / balance / send
```

## Develop

```bash
pip install -e .[dev]
pytest                          # 190 tests: chain, consensus, routing, billing, client, e2e
deltav sim --nodes 3 --duration 20   # local network: nodes + gateway + blocks + spot-checks
```

## Adding a new accelerator

The chain and router only ever see the `ComputeBackend` interface
(`deltav/compute/base.py`): implement `is_available / load / infer`
(+ optional `embed`, `infer_stream`), set `deterministic` and
`dynamic_models`, register the class — nothing else changes. Backends:
`llamacpp` (in-process), `llamaserver` (local llama.cpp over HTTP —
Vulkan for AMD without compiling), `groq` (LPU relay), `mock`, and an
`asic` skeleton for custom chips.

## License

MIT — see [LICENSE](LICENSE).
