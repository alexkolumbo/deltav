"""Tiny bilingual (English / Russian) string table for the setup wizard."""
from __future__ import annotations

import locale
import os

LANGS = ("en", "ru")

M: dict[str, dict[str, str]] = {
    "title":        {"en": "Delta V — node setup", "ru": "Delta V — установка ноды"},
    "lang_prompt":  {"en": "Language / Язык (en/ru)", "ru": "Язык / Language (ru/en)"},
    "intro":        {"en": "A node is your computer answering AI requests and earning DVT tokens.\n"
                           "I'll walk you through it step by step:",
                     "ru": "Нода — это ваш компьютер, который отвечает на запросы к ИИ\n"
                           "и зарабатывает за это токены DVT. Я проведу вас по шагам:"},
    "flow":         {"en": "hardware → engine → model → wallet → network → launch",
                     "ru": "железо → движок → модель → кошелёк → сеть → запуск"},
    "install_dir":  {"en": "Everything installs into: {home}", "ru": "Всё установится в: {home}"},

    "s_hardware":   {"en": "Checking your hardware", "ru": "Смотрю, какое у вас железо"},
    "gpu_found":    {"en": "GPU: {name} — {vram} of video memory",
                     "ru": "Видеокарта: {name} — {vram} видеопамяти"},
    "gpu_good":     {"en": "Great — the AI will run fast on the GPU.",
                     "ru": "Отлично — ИИ будет работать быстро на видеокарте."},
    "no_gpu":       {"en": "No GPU found, will run on the CPU ({vram} RAM)",
                     "ru": "Видеокарта не найдена, буду считать на процессоре ({vram} ОЗУ)"},
    "no_gpu_note":  {"en": "It works, but answers are slower. Fine to start with.",
                     "ru": "Заведётся, но ответы будут медленнее. Это нормально для старта."},

    "s_model":      {"en": "Choosing an AI model for your hardware",
                     "ru": "Подбираю модель ИИ под ваше железо"},
    "ports_busy":   {"en": "Ports {ports} are already in use — another node/engine is running here.",
                     "ru": "Порты {ports} уже заняты — на этой машине уже работает узел/движок."},
    "ports_busy2":  {"en": "NOT starting: the launcher self-cleans those ports and would kill it. "
                           "Stop the other node first, or run this one on another machine.",
                     "ru": "Запуск отменён: лаунчер чистит эти порты и убил бы соседа. "
                           "Остановите тот узел или поднимите этот на другой машине."},
    # --- image (draw-only) node ---
    "s_image_model": {"en": "Setting up the image model (FLUX.1-schnell)",
                      "ru": "Настраиваю модель для картинок (FLUX.1-schnell)"},
    "s_image_deps":  {"en": "Checking the image engine (torch + diffusers)",
                      "ru": "Проверяю движок картинок (torch + diffusers)"},
    "image_vram_ok": {"en": "{vram} VRAM — the pipeline fits (CPU offload below 24 GB).",
                      "ru": "{vram} видеопамяти — пайплайн влезет (ниже 24 ГБ включается CPU-offload)."},
    "image_vram_low": {"en": "Only {vram} VRAM — FLUX needs ~8 GB even offloaded; "
                             "generation may fail or be very slow.",
                       "ru": "Всего {vram} видеопамяти — FLUX требует ~8 ГБ даже с офлоадом; "
                             "генерация может не пойти или будет очень медленной."},
    "image_deps_ok": {"en": "torch + diffusers are installed.",
                      "ru": "torch и diffusers установлены."},
    "image_deps_missing": {"en": "torch/diffusers missing — install them, then re-run:",
                           "ru": "нет torch/diffusers — установите и запустите визард заново:"},
    "image_dl_note": {"en": "The model (~24 GB) downloads on first start and the node "
                            "warms it up in the background.",
                      "ru": "Модель (~24 ГБ) скачается при первом запуске, узел прогреет её в фоне."},
    "recommend":    {"en": "Recommended: {name}", "ru": "Рекомендую: {name}"},
    "model_specs":  {"en": "{b}B params, fits ~{ctx} tokens of context",
                     "ru": "{b}B параметров, влезает контекст ~{ctx} токенов"},
    "download_once":{"en": "One-time download: ~{size}.", "ru": "Скачать нужно ~{size} один раз."},
    "show_others":  {"en": "Show other options?", "ru": "Показать другие варианты?"},
    "paste_own":    {"en": "Paste your own model from HuggingFace?",
                     "ru": "Вставить свою модель с HuggingFace?"},
    "recommended_tag": {"en": " (recommended)", "ru": " (рекомендую)"},
    "pick_number":  {"en": "Model number (or 'c' to paste your own)",
                     "ru": "Номер модели (или 'c' — вставить свою)"},
    "model_legend": {"en": "👁 = sees images · 🔥 = already served on this network",
                     "ru": "👁 = видит картинки · 🔥 = уже раздаётся в этой сети"},

    "custom_prompt":{"en": "HuggingFace repo (e.g. org/repo or org/repo::file.gguf)",
                     "ru": "Репозиторий HuggingFace (напр. org/repo или org/repo::file.gguf)"},
    "custom_mode":  {"en": "How to add it?", "ru": "Как добавить?"},
    "custom_analyze":{"en": "  a) analyze — check if it fits your hardware (recommended)",
                      "ru": "  a) анализ — проверить, влезет ли на ваше железо (рекомендую)"},
    "custom_forced":{"en": "  f) forced  — use it as-is, I know what I'm doing",
                     "ru": "  f) форсировать — использовать как есть, я знаю что делаю"},
    "custom_choice":{"en": "Choice (a/f)", "ru": "Выбор (a/f)"},
    "analyzing":    {"en": "Analyzing {ref} …", "ru": "Анализирую {ref} …"},
    "analyze_fail": {"en": "Couldn't read that model from HuggingFace. Check the name/file.",
                     "ru": "Не смог прочитать эту модель с HuggingFace. Проверьте имя/файл."},
    "verdict_great":{"en": "Fits well: ~{size}, up to ~{ctx} tokens of context.",
                     "ru": "Влезает хорошо: ~{size}, до ~{ctx} токенов контекста."},
    "verdict_tight":{"en": "Fits, but tight: ~{size}, only ~{ctx} tokens of context.",
                     "ru": "Влезает впритык: ~{size}, только ~{ctx} токенов контекста."},
    "verdict_cpu":  {"en": "Weights (~{size}) fit but leave no room for context — "
                           "it will be slow (CPU offload).",
                     "ru": "Веса (~{size}) влезают, но контексту места нет — "
                           "будет медленно (выгрузка на CPU)."},
    "verdict_big":  {"en": "Too big for this hardware (~{size} vs {vram} VRAM).",
                     "ru": "Слишком большая для этого железа (~{size} против {vram} видеопамяти)."},
    "use_anyway":   {"en": "Use it anyway?", "ru": "Всё равно использовать?"},
    "use_it":       {"en": "Use this model?", "ru": "Использовать эту модель?"},
    "forced_note":  {"en": "Forced: using {ref} without a fit check.",
                     "ru": "Форсировано: беру {ref} без проверки на влезаемость."},

    "s_engine":     {"en": "Installing the engine (llama.cpp)", "ru": "Ставлю движок (llama.cpp)"},
    "engine_have":  {"en": "Engine already installed.", "ru": "Движок уже установлен."},
    "engine_dl":    {"en": "Downloading a prebuilt binary — nothing to compile.",
                     "ru": "Скачиваю готовый бинарник — компилировать ничего не нужно."},
    "engine_none":  {"en": "No prebuilt binary for your system — install llama.cpp manually.",
                     "ru": "Готового бинарника под вашу систему нет — установите llama.cpp вручную."},
    "engine_ok":    {"en": "Engine installed.", "ru": "Движок установлен."},

    "s_model_dl":   {"en": "Downloading the model", "ru": "Скачиваю модель"},
    "model_have":   {"en": "Model already downloaded.", "ru": "Модель уже скачана."},
    "model_tea":    {"en": "~{size} — time for a coffee.", "ru": "~{size} — можно заварить чай."},
    "model_ok":     {"en": "Model downloaded.", "ru": "Модель скачана."},
    "vision_have":  {"en": "Vision adapter already downloaded.",
                     "ru": "Vision-адаптер уже скачан."},
    "vision_dl":    {"en": "This model can see images — fetching its vision adapter (mmproj).",
                     "ru": "Эта модель умеет видеть картинки — качаю vision-адаптер (mmproj)."},
    "vision_ok":    {"en": "Vision adapter ready — the node will accept images.",
                     "ru": "Vision-адаптер готов — нода будет принимать картинки."},
    "vision_skip":  {"en": "No vision adapter found in the repo — the node will serve text only.",
                     "ru": "Vision-адаптер в репозитории не найден — нода будет только текстовой."},

    "s_wallet":     {"en": "Node wallet", "ru": "Кошелёк ноды"},
    "wallet_note":  {"en": "The wallet is the address your earned DVT flows to.",
                     "ru": "Кошелёк — это адрес, на который капают заработанные DVT."},
    "wallet_addr":  {"en": "Address: {addr}", "ru": "Адрес: {addr}"},
    "wallet_keep":  {"en": "Kept locally. Don't delete it — these are your keys.",
                     "ru": "Файл хранится локально. Не удаляйте его — это ваши ключи."},

    "s_network":    {"en": "Connecting to the network", "ru": "Подключаюсь к сети"},
    "seed_prompt":  {"en": "Any live node URL to join (seed)", "ru": "Адрес любой живой ноды сети (seed)"},
    "connected":    {"en": "Joined network \"{chain}\" via {seed}",
                     "ru": "Подключился к сети «{chain}» через {seed}"},
    "no_network":   {"en": "Couldn't reach the network ({err}).",
                     "ru": "Не достучался до сети ({err})."},
    "no_network2":  {"en": "Check the seed URL and that it's online, then run again.",
                     "ru": "Проверьте адрес seed-ноды и что она включена, затем запустите снова."},

    "s_price":      {"en": "Price for your work", "ru": "Цена за работу"},
    "price_note":   {"en": "At the world-average electricity price + 50% service that's "
                           "~${usd}/million tokens.",
                     "ru": "По среднемировой цене электричества + 50% сервиса выходит "
                           "~${usd}/млн токенов."},
    "price_ask":    {"en": "Set the recommended price of {rec} udvt/token?",
                     "ru": "Поставить рекомендованную цену {rec} udvt/токен?"},
    "price_custom": {"en": "Your price (udvt/token)", "ru": "Своя цена (udvt/токен)"},
    "price_set":    {"en": "Price: {price} udvt per token.", "ru": "Цена: {price} udvt за токен."},

    "s_launch":     {"en": "Launching the node", "ru": "Запускаю ноду"},
    "not_ready":    {"en": "Some steps didn't finish — complete them and run again.",
                     "ru": "Не все шаги завершились — доделайте их и запустите снова."},
    "script_saved": {"en": "Launch script saved: {path}", "ru": "Скрипт запуска сохранён: {path}"},
    "ready_run":    {"en": "All set. Start the node with:", "ru": "Всё готово. Запустите ноду командой:"},
    "starting":     {"en": "Bringing up the engine and node…", "ru": "Поднимаю движок и ноду…"},
    "engine_up":    {"en": "Engine is responding.", "ru": "Движок отвечает."},
    "engine_slow":  {"en": "Engine didn't come up in time. Run the script manually and watch the output.",
                     "ru": "Движок не поднялся вовремя. Запустите скрипт вручную и посмотрите вывод."},
    "node_slow":    {"en": "Node is taking longer than usual — check the explorer in a minute.",
                     "ru": "Нода запускается дольше обычного — проверьте эксплорер через минуту."},

    "relay_wait":   {"en": "Checking the node is reachable from outside…",
                     "ru": "Проверяю, что нода видна снаружи…"},
    "relay_ok":     {"en": "Reachable from the internet — the network can send it work:",
                     "ru": "Нода доступна из интернета — сеть может слать ей запросы:"},
    "relay_slow":   {"en": ("The node is running, but it has not appeared on the relay yet. "
                            "It often attaches a minute later; if it never does, the node stays "
                            "LAN-only and the dashboard will show it offline. Check: {url}/health"),
                     "ru": ("Нода работает, но на релее пока не появилась. Часто цепляется через "
                            "минуту; если так и не появится — она останется только в локальной "
                            "сети, и дашборд будет показывать её офлайн. Проверить: {url}/health")},
    "relay_none":   {"en": ("No relay could be derived, so this node is LAN-ONLY: other nodes "
                            "cannot reach it and it will earn nothing. Re-run with a public "
                            "…/via/<id> seed, or pass --relay <url>."),
                     "ru": ("Релей вывести не из чего — нода останется ТОЛЬКО в локальной сети: "
                            "другие узлы её не увидят и заработка не будет. Перезапустите с "
                            "публичным …/via/<id> сидом или передайте --relay <url>.")},

    "done_title":   {"en": "Your node is live and on the network! 🎉",
                     "ru": "Нода запущена и в сети! 🎉"},
    "done_panel":   {"en": "Node dashboard (open in a browser):",
                     "ru": "Панель ноды (откройте в браузере):"},
    "done_addr":    {"en": "Your earning address:", "ru": "Ваш адрес для заработка:"},
    "done_next":    {"en": "Next time just run: {script}", "ru": "В следующий раз просто запустите: {script}"},
    "done_stop":    {"en": "To stop — close the engine and node windows.",
                     "ru": "Чтобы остановить — закройте окна движка и ноды."},
    "interrupted":  {"en": "Interrupted. Progress saved — run again to continue.",
                     "ru": "Прервано. Прогресс сохранён — запустите снова, чтобы продолжить."},
    "light_model":  {"en": "Modest hardware — taking the lightest model.",
                     "ru": "Железо скромное — беру самую лёгкую модель."},

    "s_schedule":   {"en": "Auto-start & schedule", "ru": "Автозапуск и расписание"},
    "autostart_ask":{"en": "Auto-start the node on login so it stays online without babysitting?",
                     "ru": "Автозапускать ноду при входе в систему, чтобы она держалась в сети сама?"},
    "schedule_ask": {"en": "Run only during set hours? (No = online 24/7)",
                     "ru": "Работать только в заданные часы? (Нет = в сети круглосуточно)"},
    "schedule_start":{"en": "Online from (HH:MM)", "ru": "Онлайн с (ЧЧ:ММ)"},
    "schedule_end": {"en": "Online until (HH:MM)", "ru": "Онлайн до (ЧЧ:ММ)"},
    "schedule_bad": {"en": "Couldn't read the times — defaulting to 24/7.",
                     "ru": "Не разобрал время — оставляю круглосуточно."},
    "autostart_ok": {"en": "Auto-start installed — the node comes up on its own.",
                     "ru": "Автозапуск установлен — нода поднимается сама."},
    "autostart_win":{"en": "Online window {start}–{end}; auto-starts at {start}, stops at {end}.",
                     "ru": "Окно онлайна {start}–{end}; сам стартует в {start}, останавливается в {end}."},
    "autostart_fail":{"en": "Couldn't install auto-start — start manually with the script below.",
                     "ru": "Не удалось поставить автозапуск — запускайте вручную скриптом ниже."},
    "autostart_skip":{"en": "No auto-start. Start the node with: {script}",
                     "ru": "Без автозапуска. Запускайте ноду командой: {script}"},
    "engine_session":{"en": "Note: the GPU engine needs your logged-in desktop session, so keep this "
                            "user logged in (auto-start fires while you're logged in, even away).",
                     "ru": "Важно: GPU-движку нужна ваша залогиненная сессия рабочего стола — держите "
                           "пользователя в системе (автозапуск срабатывает, пока вы залогинены, даже отойдя)."},
}


def detect_lang() -> str:
    for src in (os.environ.get("DELTAV_LANG"), os.environ.get("LANG"),
                os.environ.get("LC_ALL")):
        if src and src.lower().startswith("ru"):
            return "ru"
    try:
        loc = locale.getlocale()[0] or ""
        if loc.lower().startswith(("ru", "russian")):
            return "ru"
    except Exception:
        pass
    return "en"


class T:
    """Translator bound to a language."""

    def __init__(self, lang: str = "en"):
        self.lang = lang if lang in LANGS else "en"

    def __call__(self, key: str, **fmt) -> str:
        entry = M.get(key, {})
        text = entry.get(self.lang) or entry.get("en") or key
        return text.format(**fmt) if fmt else text
