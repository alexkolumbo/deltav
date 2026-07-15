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

## OpenClaw / Claude-подобные агенты

Инструменты, ожидающие Anthropic API, подключаются через LiteLLM-прокси:

```yaml
# litellm-config.yaml
model_list:
  - model_name: deltav-auto
    litellm_params:
      model: openai/auto
      api_base: http://127.0.0.1:9000/v1
      api_key: deltav
```

```bash
litellm --config litellm-config.yaml --port 4000
# агенту указывается http://127.0.0.1:4000
```

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

Поле `deltav` в не-стриминговом ответе показывает, какая нода обслужила
запрос и хэш чека в чейне — его можно найти в эксплорере
(`http://<node>:9100/explorer`).

> Примечание: точные имена полей конфига у goose/opencode меняются от версии
> к версии — сверяйтесь с документацией инструмента; неизменная часть — это
> base URL шлюза и `model: auto`.
