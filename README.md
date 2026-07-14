# Delta V — decentralized AI network

Собственный лёгкий PoS-блокчейн + сеть GPU-нод (класс NVIDIA RTX 4070, 12 ГБ VRAM) +
умный роутинг моделей с [HuggingFace](https://huggingface.co/models). Пользователь шлёт
запрос в OpenAI-совместимый шлюз — сеть сама выбирает лучшую модель, которая физически
влезает в VRAM живых нод, исполняет инференс, записывает чек в чейн, платит ноде токенами
DVT и выборочно перепроверяет её честность.

```
клиент ──► gateway (/v1/chat/completions)
              │  SmartRouter: модель ⨯ нода (VRAM-fit, репутация, стейк, загрузка)
              ▼
        node daemon ──► ComputeBackend (llama.cpp: NVIDIA/AMD/CPU; groq/asic — скелеты)
              │
              ▼
        Delta V chain: INFERENCE_RECEIPT (оплата) ──► SPOT_CHECK (перепроверка / слэшинг)
```

## Компоненты

| Пакет | Что делает |
|---|---|
| `deltav/chain/` | PoS-чейн: блоки, транзакции, детерминированный выбор пропозера по стейку, форк-чойс longest-chain, мемпул |
| `deltav/compute/` | Абстракция вычислений: `llamacpp` (NVIDIA/AMD/Intel/Apple/CPU — один GGUF везде), `mock` (детерминированный, для симуляции), скелеты `groq` и `asic` |
| `deltav/router/` | Каталог GGUF-моделей HF с оценкой VRAM (веса + KV-cache + overhead), скоринг нод, диспетчеризация с failover |
| `deltav/node/` | Демон ноды: полный узел чейна + P2P-госсип + inference-сервер + спот-чекер |
| `deltav/gateway/` | OpenAI-совместимый API; кошелёк шлюза — плательщик за инференс |

## Экономика и доверие

- **DVT** (1 DVT = 10⁶ udvt). Реквестер платит `price_per_token` за каждый токен; нода
  дополнительно получает эмиссию за чек, пропозер — за блок.
- **Авторизация оплаты**: нода может списать деньги только предъявив подпись реквестера
  над `(request_hash, node, model, price_limit)` — украсть оплату или превысить лимит нельзя.
- **Spot-check**: валидаторы детерминированно выбирают долю чеков (`spot_check_rate`),
  скачивают у ноды исходный джоб, перепрокатывают его на своём бэкенде и сверяют хэш
  вывода. Ложь ⇒ сжигание доли стейка (`slash_fraction`) и обвал репутации.
- **Роутинг**: `score = 3·model_ready + 2·reputation + 1.5·(1−load) + 1·stake + 0.5·freshness`;
  «auto» выбирает самую качественную модель каталога, которую живые ноды уже анонсировали
  (без холодной загрузки), иначе — что влезает в их VRAM. На 12 ГБ (4070) это Qwen2.5-14B
  Q4_K_M; 32B/70B честно отсекаются по VRAM-оценке.

## Быстрый старт

```bash
pip install -e .[dev]          # + .[gpu] для llama.cpp, .[hub] для живого каталога HF
pytest                          # 42 теста: чейн, консенсус, роутер, e2e со слэшингом

# локальная сеть: 3 ноды + шлюз + блоки + инференс + спот-чеки
deltav sim --nodes 3 --duration 20
```

## Реальная сеть

```bash
# 1. кошельки
deltav wallet new --file node1.wallet.json

# 2. генезис (стартовые балансы и стейки валидаторов)
deltav genesis --alloc dv1...=100000 --stake dv1...=10000 -o genesis.json

# 3. нода на машине с 4070 (llama.cpp сам увидит CUDA; на AMD — сборка с ROCm/Vulkan)
deltav node --genesis genesis.json --wallet node1.wallet.json \
    --port 9100 --endpoint http://<public-ip>:9100 \
    --peer http://<other-node>:9100 \
    --backend auto \
    --model "Qwen/Qwen2.5-14B-Instruct-GGUF::qwen2.5-14b-instruct-q4_k_m.gguf"

# 4. шлюз
deltav gateway --genesis genesis.json --node http://<node>:9100 --port 9000

# 5. запрос (или любой OpenAI-клиент на http://localhost:9000/v1)
deltav chat "Explain delta-v" --gateway http://localhost:9000
```

## Как добавить новый чип (Groq, кастомный ASIC)

Чейн и роутер видят только интерфейс `ComputeBackend` (`deltav/compute/base.py`):
реализуйте `is_available / load / infer`, укажите `deterministic` (иначе спот-чеки
перейдут в fuzzy-режим) и зарегистрируйте класс — больше ничего менять не нужно.
Скелеты с инструкциями: `deltav/compute/groq.py`, `deltav/compute/asic.py`.

## Статус MVP / что дальше

Работает: PoS-консенсус, P2P-синхронизация, оплата и слэшинг on-chain, VRAM-роутинг,
OpenAI-совместимый API, полная локальная симуляция без GPU.

Осознанные упрощения MVP (следующие фазы):
- синхронизация — наивная полная перевалидация чейна (нужны чекпоинты/снапшоты);
- нет unbonding-периода стейка и наказания за пропуск блока;
- спот-чек детерминированного бэкенда сверяет хэш 1:1; для GPU-недетерминизма нужен
  fuzzy-верификатор (перплексия/токен-каунт);
- госсип полносвязный, без discovery (нужен peer exchange);
- каталог HF курируемый; `Catalog.refresh_from_hf()` уже умеет подтягивать живой топ.
