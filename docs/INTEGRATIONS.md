# Подключение клиентского софта к Delta V

Шлюз Delta V говорит на диалекте OpenAI API (`/v1/chat/completions`,
`/v1/models`, стриминг через SSE), поэтому **любой** инструмент с настройкой
"OpenAI-compatible endpoint / base URL" работает с сетью напрямую.

Общие параметры:

| Параметр | Значение |
|---|---|
| Base URL | `http://<gateway-host>:9000/v1` |
| API key | любой (пока не проверяется; позже — привязка к кошельку) |
| Model | `auto` — сеть сама выберет лучшую влезающую модель, либо конкретный ref из `deltav models` |

Оплата инференса происходит **на шлюзе**: его кошелёк подписывает лимит цены
для каждого запроса, нода получает DVT через чек в чейне. Клиентскому софту
об этом знать не нужно.

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

## OpenClaw / Claude-нативные агенты (Anthropic API)

Шлюз отдаёт **нативный Anthropic Messages API** — `POST /v1/messages` со
стримингом в формате событий Anthropic (`message_start` → `content_block_delta`
→ `message_stop`) и tool-use блоками. Софт на `anthropic` SDK подключается
напрямую, без LiteLLM:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9000
export ANTHROPIC_API_KEY=deltav          # или ваш dvk_-ключ
export ANTHROPIC_MODEL=auto
```

```python
import anthropic
c = anthropic.Anthropic(base_url="http://127.0.0.1:9000", api_key="deltav")
msg = c.messages.create(model="auto", max_tokens=256,
                        messages=[{"role": "user", "content": "привет"}])
print(msg.content[0].text)
```

Поддержаны `system`, `tools` (Anthropic input_schema) и `tool_result` —
агент видит инструменты и возвращает `stop_reason: "tool_use"`.

> Старый путь через LiteLLM-прокси (`model: openai/auto`, `api_base:
> …/v1`) тоже работает, если инструмент жёстко ждёт именно его.

## Hermes (и любые OPENAI_BASE_URL-боты)

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

## curl

```bash
curl http://127.0.0.1:9000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

## Tool calling

Шлюз поддерживает OpenAI-диалект `tools` / `tool_calls`: определения
инструментов инжектируются в промпт (формат `<tool_call>` — Hermes/Qwen,
на нём обучено большинство открытых instruct-моделей), ответ модели
парсится обратно в `tool_calls` с `finish_reason: "tool_calls"`.
Инструмент исполняет клиент (goose/opencode это делают сами), результат
возвращается сообщением `role: "tool"` — стандартный цикл OpenAI.

## Встроенные надстройки сети

```bash
# интернет-поиск (DDG -> Mojeek fallback, без API-ключей)
GET /v1/search?q=<query>&max_results=5
deltav search "rtx 4070 llm benchmarks"

# серверный агент: ReAct-цикл прямо на сети; каждый шаг рассуждения —
# оплаченный чек в чейне (receipt_tx в каждом step)
POST /v1/agents/run {"task": "...", "model": "auto", "max_steps": 6}
deltav agent "найди последнюю версию llama.cpp и посчитай 2**20"
```

Встроенные инструменты агента: `web_search`, `fetch_url`, `calculator`
(реестр расширяем — `deltav/overlay/tools.py::ToolRegistry`).

## Мультинода — сворм агентов

```bash
# одну задачу — нескольким РАЗНЫМ моделям параллельно (каждая на своей ноде),
# затем синтез лучшего ответа
POST /v1/swarm {"task": "...", "n": 3, "mode": "vote"}
deltav swarm "оцени риски этого плана" --mode vote -n 3

# режимы: fanout (разные ответы), vote (+синтез), map (по задаче на воркера)
POST /v1/swarm {"tasks": ["A","B","C"], "mode": "map"}
```

Каждый воркер маршрутизируется независимо, поэтому работа расходится по
живым нодам сети. В ответе — `workers[]` (модель, нода, ответ, чек) и
синтезированный `answer` для режима vote.

## Свой клиент и REPL

```bash
deltav connect --url http://gw1:9000,http://gw2:9000 --key dvk_… --model auto
deltav repl                     # интерактивный чат со стримингом; /agent /swarm /model
```

`connect` сохраняет профиль (несколько base URL для failover + ключ) в
`~/.deltav/client.json`; `deltav repl`, `swarm`, а также Python-SDK
`deltav.client.DeltaVClient` берут его оттуда.

```python
from deltav.client import DeltaVClient
c = DeltaVClient.from_profile()          # или base_urls=[...], api_key="dvk_…"
print(c.chat([{"role": "user", "content": "hi"}])["choices"][0]["message"]["content"])
for chunk in c.chat_stream([...]): ...   # стриминг
c.swarm("сравни два подхода", n=2, mode="vote")
```

Поле `deltav` в не-стриминговом ответе показывает, какая нода обслужила
запрос и хэш чека в чейне — его можно найти в эксплорере
(`http://<node>:9100/explorer`).

> Примечание: точные имена полей конфига у goose/opencode меняются от версии
> к версии — сверяйтесь с документацией инструмента; неизменная часть — это
> base URL шлюза и `model: auto`.
