# Using Delta V (client)

[English](CLIENT.md) · [Русский](CLIENT.ru.md)

You reach the network through a **gateway** — an OpenAI/Ollama/Anthropic-
compatible endpoint. Pick whichever path fits; the simplest is first.

You'll need: the gateway URL (e.g. `http://10.0.0.223:9000`) and, for
billing, an API key. Model is always `auto` (the network picks the best).

## 1. Web app — nothing to install

Open **`http://<gateway>:9000/chat`** in any browser (phone or laptop):

- Streaming chat, plus **Agent** (web search + tools), **Swarm**
  (several models in parallel), and **Companion** (remembers you) modes.
- **🖼 image upload** for vision models.
- **⚙ settings** — set the gateway URL and your API key once.
- On iOS/Android: *Share → Add to Home Screen* installs it as an app.

## 2. Existing tools (goose, Open WebUI, opencode, your code)

Point the tool at the gateway:

| Tool speaks | Base URL | Key |
|---|---|---|
| OpenAI (goose, opencode, openai SDK, Hermes, grok-build) | `http://<gw>:9000/v1` | your `dvk_…` |
| Ollama (Open WebUI, LangChain, desktop apps) | `http://<gw>:9000` | your `dvk_…` |
| Anthropic (`anthropic` SDK) | `http://<gw>:9000` | your `dvk_…` |

Model: `auto`. Full per-tool recipes: [INTEGRATIONS.md](INTEGRATIONS.md).

## 3. Zero-config — the `deltav serve` proxy

So a tool needs **no key and no config** — point it at `localhost`:

```bash
pip install "deltav-network @ git+https://github.com/alexkolumbo/deltav"
deltav connect --url http://<gw>:9000 --key dvk_…   # save it once
deltav serve                                          # proxy on :11434
```

Now any tool works credential-free:
- Ollama: `http://localhost:11434`
- OpenAI: `http://localhost:11434/v1`
- Anthropic: `http://localhost:11434`

The proxy attaches your key and fails over across gateways.

## 4. Command line

```bash
deltav repl                       # interactive streaming chat; /agent /swarm /model
deltav chat "Explain delta-v"     # one-shot
deltav agent "find the latest llama.cpp release and compute 2**20"
deltav swarm "compare two approaches" --mode vote -n 3
deltav companion                  # persistent agent that remembers you
deltav models / network / balance dv1… / send --to dv1… --amount 100
```

## 5. Telegram — from anywhere

The bot **@DeltaV_ai_bot** works from any network:
- Just type — the agent answers, searching the web when needed, and
  remembers your conversation.
- **Send a photo** 🖼 — a vision model describes it or answers your caption.
- `/agent <task>`, `/models`, `/net`, `/plan`, `/fast`, `/reset`.

## Billing — API keys are wallets

An API key `dvk_…` is a custodial on-chain wallet. Fund its address and
every request is paid from it; you never overpay (the price limit is
signed per request).

```bash
# create a key (shown once) — then fund its address with DVT
curl -X POST http://<gw>:9000/v1/keys
deltav keys me --key dvk_…         # balance + usage
```

Without a key, requests use the gateway's wallet (fine for private
networks). Set `--require-keys` on the gateway to require funded keys.

## Quick test

```bash
curl http://<gw>:9000/v1/chat/completions \
  -H "Authorization: Bearer dvk_…" -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

The `deltav` field in the reply shows which node served you and the
on-chain receipt — find it in the explorer (`http://<node>:9100/explorer`).

## Notes

- **Same network only** for the web app and direct API (LAN). For access
  from anywhere, the host puts a tunnel (Cloudflare/Caddy TLS) in front of
  the gateway — then use its public URL. Telegram works from anywhere.
- **Reasoning models** answer directly by default; if replies look
  truncated, raise `max_tokens`.
- **Vision** needs a vision-capable model served (marked 👁 in the model
  picker / `deltav models`).
