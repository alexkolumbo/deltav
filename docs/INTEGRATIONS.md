# Connecting client software to Delta V

The gateway speaks the surfaces the open-source ecosystem expects — the
API *shape* is a translation layer, independent of the model being
served. Any tool with an **OpenAI**, **Ollama**, or **Anthropic** setting
works with the network directly.

## Zero-config: `deltav serve`

The simplest integration — a local proxy that holds your gateway URL and
key, so tools point at `localhost` with **no key and no config**:

```bash
deltav connect --url http://<gw>:9000 --key dvk_…   # once
deltav serve                                          # local proxy on :11434
```

Then any tool just works, credential-free:

| Tool expects | Point it at |
|---|---|
| Ollama | `http://localhost:11434` (its default — often auto-detected) |
| OpenAI | `http://localhost:11434/v1` |
| Anthropic | `http://localhost:11434` |

The proxy attaches your key and fails over across gateways. It's a
transparent catch-all, so every surface (chat, embeddings, images,
companion, swarm) works through it.

## Web app (installable PWA)

Open **`http://<gw>:9000/chat`** in any browser — a mobile-first client
with streaming chat, agent / swarm / companion modes, image upload for
vision, a model picker, and a settings panel (gateway + key, importable
via `#key=…`). On iOS/Android "Add to Home Screen" installs it as an app;
behind HTTPS it's a full PWA.

Common settings:

| Setting | Value |
|---|---|
| OpenAI base URL | `http://<gateway>:9000/v1` |
| Ollama host | `http://<gateway>:9000` |
| Anthropic base URL | `http://<gateway>:9000` |
| API key | any (or a funded `dvk_…` key for on-chain billing) |
| Model | `auto` — the network picks the best fitting model — or a specific ref from `deltav models` |

Inference is paid **at the gateway**: its wallet (or the request's `dvk_`
key) authorizes a price limit per request and the node is paid in DVT via
an on-chain receipt. The client doesn't need to know about it.

## Ollama-compatible tools (Open WebUI, LangChain, desktop apps)

The gateway exposes the Ollama `/api/*` dialect, so the network looks like
a local Ollama server:

```bash
# point any Ollama client at the gateway
export OLLAMA_HOST=http://127.0.0.1:9000

# or call it directly
curl http://127.0.0.1:9000/api/tags
curl http://127.0.0.1:9000/api/chat \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}]}'
```

Supported: `/api/tags`, `/api/chat`, `/api/generate`, `/api/embeddings`,
`/api/embed`, `/api/version`. Streaming is NDJSON (Ollama's format). Model
names are loose — `auto`, a full ref, an ollama tag (`qwen2.5-7b:q4_k_m`),
or a short name all resolve to a served model.

**Open WebUI**: Settings → Connections → Ollama API → `http://<gw>:9000`.
**LangChain**: `ChatOllama(base_url="http://<gw>:9000", model="auto")`.

## Goose (block/goose)

```yaml
# ~/.config/goose/config.yaml
GOOSE_PROVIDER: openai
GOOSE_MODEL: auto
OPENAI_HOST: http://127.0.0.1:9000
OPENAI_API_KEY: deltav
```

## opencode

```jsonc
// opencode.json
{
  "provider": {
    "deltav": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Delta V",
      "options": { "baseURL": "http://127.0.0.1:9000/v1" },
      "models": { "auto": { "name": "Delta V auto-routed" } }
    }
  }
}
```

## Anthropic-API clients

The gateway also serves a native **Anthropic Messages API** —
`POST /v1/messages` with SSE in Anthropic's event format
(`message_start` → `content_block_delta` → `message_stop`) and tool-use
blocks — for any client hardcoded to the `anthropic` SDK. This is just an
extra compatibility surface; the models are open-source (Qwen, Llama,
Gemma, Grok…), and most agents (Hermes, grok-build, goose, opencode) speak
OpenAI, not Anthropic.

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9000
export ANTHROPIC_API_KEY=deltav          # or your dvk_ key
export ANTHROPIC_MODEL=auto
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:9000", api_key="deltav")
msg = c.messages.create(model="auto", max_tokens=256,
                        messages=[{"role": "user", "content": "hello"}])
print(msg.content[0].text)
```

`system`, `tools` (Anthropic input_schema) and `tool_result` are supported;
the model returns `stop_reason: "tool_use"` when it calls a tool.

## Hermes / grok-build / any OPENAI_BASE_URL agent

Nous-Research Hermes (and stacks built on it) and xAI's grok-build are
OpenAI-compatible — point them at the OpenAI surface:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:9000/v1
export OPENAI_API_KEY=deltav
export OPENAI_MODEL=auto
```

## Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:9000/v1", api_key="deltav")
resp = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "hello"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

## Tool calling

The gateway supports the OpenAI `tools` / `tool_calls` dialect (and the
Anthropic equivalent): tool definitions are injected into the prompt (the
`<tool_call>` format most open instruct models were trained on), the
model's reply is parsed back into `tool_calls` with `finish_reason:
"tool_calls"`. The client executes the tool (goose/opencode do this
themselves) and returns the result as a `role: "tool"` message.

## Network overlays

```bash
# internet search (DDG -> Mojeek, no API keys)
GET /v1/search?q=<query>&max_results=5
deltav search "rtx 4070 llm benchmarks"

# server-side agent: a ReAct loop on the network; each reasoning step is a
# paid, spot-checkable receipt (receipt_tx in every step)
POST /v1/agents/run {"task": "...", "model": "auto", "max_steps": 6}
deltav agent "find the latest llama.cpp release and compute 2**20"
```

Built-in agent tools: `web_search`, `fetch_url`, `calculator`
(extensible — `deltav/overlay/tools.py::ToolRegistry`).

## Multi-node — a swarm of agents

```bash
# fan one task across several DISTINCT models in parallel (each on whichever
# node serves it), then synthesize the best answer
POST /v1/swarm {"task": "...", "n": 3, "mode": "vote"}
deltav swarm "assess the risks of this plan" --mode vote -n 3

# modes: fanout (diverse answers), vote (+synthesis), map (one task per worker)
POST /v1/swarm {"tasks": ["A","B","C"], "mode": "map"}
```

Each worker routes independently, so work spreads across live nodes. The
response has `workers[]` (model, node, answer, receipt) and a synthesized
`answer` for vote mode.

## Companion — a persistent per-user agent

A stateful agent with a **personal memory layer** and **self-improvement**,
with **strict per-user isolation**: identity comes from your key, never
from the request body, so one user can never reach another's memory.

```bash
POST /v1/companion/chat     {"message": "..."}   # per-user, remembers you
POST /v1/companion/feedback {"note": "be concise"}   # a durable learning
GET  /v1/companion/memory                          # only YOUR memory
deltav companion            # interactive; /memory, /feedback <text>
```

Each turn recalls your relevant memory + learnings, runs the ReAct loop
(tools available), and reflects to store what it learned — so with small
models it still gets better per user over time. Requests are billed to
your `dvk_` key; the key's wallet address is your isolation identity.

## Your own client and REPL

```bash
deltav connect --url http://gw1:9000,http://gw2:9000 --key dvk_… --model auto
deltav repl                     # interactive streaming chat; /agent /swarm /model
```

`connect` saves a profile (multiple base URLs for failover + key) to
`~/.deltav/client.json`; `deltav repl`, `swarm`, and the Python SDK
`deltav.client.DeltaVClient` all read it.

```python
from deltav.client import DeltaVClient
c = DeltaVClient.from_profile()          # or base_urls=[...], api_key="dvk_…"
print(c.chat([{"role": "user", "content": "hi"}])["choices"][0]["message"]["content"])
for chunk in c.chat_stream([...]): ...   # streaming
c.swarm("compare two approaches", n=2, mode="vote")
```

The `deltav` field in a non-streaming response shows which node served the
request and its on-chain receipt hash — find it in the explorer
(`http://<node>:9100/explorer`).

> Note: exact config field names for goose/opencode change across versions —
> check the tool's docs; the constant part is the gateway base URL and
> `model: auto`.
