"""Delta V Telegram bot — access the network from anywhere.

Standalone by design: only stdlib + httpx, no deltav imports, so this
single file can be copied to any relay box (a ZimaOS docker container, a
VPS) that can reach a gateway. Long polling — no inbound ports, no TLS,
no domain needed.

Env / CLI:
  TELEGRAM_BOT_TOKEN   (required)  token from @BotFather
  DELTAV_GATEWAY       (default http://127.0.0.1:9000)
  DELTAV_ALLOW         comma-separated Telegram user ids; empty = anyone

Commands: /start /help, /model [ref|auto], /models, /net, /plan [vram],
/agent <task> (web search + session memory), /reset. Plain text chats
with rolling history; every answer carries its on-chain receipt.
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import sys
from collections import OrderedDict, defaultdict

import httpx

log = logging.getLogger("deltav.tgbot")

HISTORY_CHAR_BUDGET = 9000
MAX_TOKENS = 900
# How much recent dialog travels into an agent task as context.
SMART_CONTEXT_MSGS = 4
SMART_CONTEXT_CHARS = 500


class AgentFailed(Exception):
    """The network could not answer. Distinct from an answer ON PURPOSE: when
    failures were returned as ordinary strings they got stored as assistant
    turns and poisoned the next turn's context."""


class TgBot:
    def __init__(self, token: str, gateway: str, allow: set[int] | None = None,
                 client: httpx.AsyncClient | None = None, api_key: str = ""):
        self._token = token
        self.api = f"https://api.telegram.org/bot{token}"
        self.file_api = f"https://api.telegram.org/file/bot{token}"
        self.gateway = gateway.rstrip("/")
        # dvk_ key -> the bot's requests are billed to that on-chain wallet
        self.gw_headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.allow = allow or set()
        self.client = client or httpx.AsyncClient(timeout=330.0)
        self.histories: dict[int, list[dict]] = {}
        self.models: dict[int, str] = {}   # chat_id -> chosen model
        # chat_id -> the model list a picker was last drawn from, so a button
        # press (which can only carry a tiny payload) maps back to a full ref.
        self._model_choices: dict[int, list[str]] = {}
        # "smart" = every message goes through the agent (web search when
        # the model decides it needs it); "fast" = plain chat, no tools.
        self.modes: dict[int, str] = {}
        self._offset = 0
        # Telegram redelivers updates when a long poll breaks mid-flight —
        # without dedupe the same question gets answered twice.
        self._seen: OrderedDict[int, None] = OrderedDict()
        self._locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def _redact(self, text) -> str:
        # The bot token appears in request URLs; keep it out of logs.
        return str(text).replace(self._token, "***") if self._token else str(text)

    # ----------------------------------------------------------- telegram
    async def tg(self, method: str, **params):
        resp = await self.client.post(f"{self.api}/{method}", json=params)
        data = resp.json()
        if not data.get("ok"):
            log.warning("telegram %s failed: %s", method, self._redact(data))
        return data.get("result")

    async def send(self, chat_id: int, text: str) -> None:
        for i in range(0, max(len(text), 1), 4000):  # telegram hard limit 4096
            await self.tg("sendMessage", chat_id=chat_id, text=text[i:i + 4000])

    def allowed(self, user_id: int) -> bool:
        return not self.allow or user_id in self.allow

    # ------------------------------------------------------------ history
    def history(self, chat_id: int) -> list[dict]:
        h = self.histories.setdefault(chat_id, [])
        while sum(len(m["content"]) for m in h) > HISTORY_CHAR_BUDGET and len(h) > 2:
            del h[:2]
        return h

    # ------------------------------------------------------------ actions
    async def do_chat(self, chat_id: int, text: str) -> str:
        history = self.history(chat_id)
        history.append({"role": "user", "content": text})
        resp = await self.client.post(f"{self.gateway}/v1/chat/completions", headers=self.gw_headers, json={
            "model": self.models.get(chat_id, "auto"),
            "messages": history,
            "max_tokens": MAX_TOKENS,
        })
        if resp.status_code != 200:
            history.pop()
            detail = resp.json().get("detail", resp.text[:200])
            return f"⚠ сеть ответила {resp.status_code}: {detail}"
        data = resp.json()
        answer = data["choices"][0]["message"]["content"] or "(пусто)"
        history.append({"role": "assistant", "content": answer})
        meta = data.get("deltav", {})
        usage = data.get("usage", {})
        footer = (f"\n\n·  {data.get('model', '?').split('::')[0].split('/')[-1]}"
                  f" · {usage.get('completion_tokens', '?')} ток"
                  f" · чек {str(meta.get('receipt_tx'))[:8]}")
        return answer + footer

    async def download_photo(self, file_id: str) -> str | None:
        """Telegram photo file_id -> base64 data URI for vision models."""
        info = await self.tg("getFile", file_id=file_id)
        path = (info or {}).get("file_path")
        if not path:
            return None
        try:
            resp = await self.client.get(f"{self.file_api}/{path}", timeout=60.0)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None
        mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        return f"data:{mime};base64," + base64.b64encode(resp.content).decode()

    async def do_vision(self, chat_id: int, caption: str, image_uri: str) -> str:
        """Send an image (+optional caption) to a vision model."""
        prompt = caption.strip() or "Что на этом изображении? Опиши кратко."
        resp = await self.client.post(f"{self.gateway}/v1/chat/completions",
                                      headers=self.gw_headers, json={
            "model": self.models.get(chat_id, "auto"),
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_uri}}]}],
            "max_tokens": MAX_TOKENS,
        })
        if resp.status_code != 200:
            detail = resp.json().get("detail", resp.text[:200])
            return (f"⚠ {resp.status_code}: {detail}\n"
                    "(нужна модель со зрением — сейчас служит "
                    f"{self.models.get(chat_id, 'auto')})")
        data = resp.json()
        answer = data["choices"][0]["message"]["content"] or "(пусто)"
        # keep vision turns in history as a text note so context survives
        self.history(chat_id).append({"role": "user", "content": f"[прислал изображение] {prompt}"})
        self.history(chat_id).append({"role": "assistant", "content": answer})
        meta = data.get("deltav", {})
        return answer + f"\n\n·  👁 {data.get('model','?').split('/')[-1].split('::')[0]} · чек {str(meta.get('receipt_tx'))[:8]}"

    async def do_agent(self, chat_id: int, task: str) -> str:
        """Raises AgentFailed if the network could not answer. Callers must
        NOT treat a failure as an answer — see do_smart."""
        resp = await self.client.post(f"{self.gateway}/v1/agents/run", headers=self.gw_headers, json={
            "task": task,
            "model": self.models.get(chat_id, "auto"),
            "max_steps": 6,
            "session_id": f"tg-{chat_id}",
        })
        if resp.status_code != 200:
            raise AgentFailed(str(resp.json().get("detail", resp.status_code)))
        data = resp.json()
        lines = []
        for s in data.get("steps", []):
            lines.append(f"🛠 {s['tool']}({json.dumps(s['arguments'], ensure_ascii=False)[:120]})")
        if lines:
            lines.append("")
        lines.append(data.get("answer", "(пусто)"))
        lines.append(f"\n· агент: {data.get('model_calls')} вызовов, память сессии tg-{chat_id}")
        return "\n".join(lines)

    async def do_smart(self, chat_id: int, text: str) -> str:
        """Default mode: route through the agent with recent dialog as
        context — the model itself decides whether to hit web_search."""
        history = self.history(chat_id)
        recent = history[-SMART_CONTEXT_MSGS:]
        context = "\n".join(
            f"{m['role']}: {m['content'][:SMART_CONTEXT_CHARS]}" for m in recent)
        task = (f"Контекст диалога:\n{context}\n\nСообщение пользователя: {text}"
                if context else text)
        # A failure must never enter the history. It used to: the error string
        # was appended as an assistant turn, so it became "Контекст диалога"
        # for the next message — a user typed "Привет" and the model politely
        # explained HTTP 500, because that is what the context was about.
        try:
            answer = await self.do_agent(chat_id, task)
        except AgentFailed as exc:
            return f"⚠ агент: {exc}"
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": answer.split("\n\n·")[0][:1500]})
        return answer

    async def served_models(self) -> list[tuple[str, int, bool]]:
        """(ref, how many nodes serve it, vision?) for models the network can
        ACTUALLY answer with right now. The catalog lists many more; offering
        those would just produce 'no node serves X'.

        kind is checked too: the seed also serves an embedding model, and
        picking that one for a chat would fail on every message."""
        data = (await self.client.get(f"{self.gateway}/v1/models",
                                      headers=self.gw_headers)).json()
        out = []
        for m in data.get("data", []):
            dv = m.get("deltav") or {}
            nodes = len(dv.get("served_by") or [])
            if nodes and dv.get("kind", "chat") == "chat":
                out.append((m["id"], nodes, bool(dv.get("vision"))))
        return sorted(out, key=lambda t: -t[1])

    @staticmethod
    def short_name(ref: str) -> str:
        return ref.split("::")[0].split("/")[-1]

    async def do_models(self) -> str:
        models = await self.served_models()
        if not models:
            return "Сейчас сеть не отдаёт ни одной модели — узлы офлайн."
        lines = ["Модели на сети:"]
        for ref, nodes, vision in models:
            lines.append(f"• {self.short_name(ref)}{' 👁' if vision else ''}  ({nodes} нод)")
        lines.append("\n/model — выбрать кнопкой")
        return "\n".join(lines)

    async def model_keyboard(self, chat_id: int) -> tuple[str, dict]:
        """Inline picker built from what the network serves. Telegram caps
        callback_data at 64 bytes and model refs run far longer, so buttons
        carry an INDEX into a per-chat snapshot of the list."""
        models = await self.served_models()
        if not models:
            return "Сейчас сеть не отдаёт ни одной модели — узлы офлайн.", {}
        self._model_choices[chat_id] = [ref for ref, _, _ in models]
        current = self.models.get(chat_id, "auto")
        rows = [[{"text": ("✅ " if current == "auto" else "") + "auto (сеть выберет сама)",
                  "callback_data": "m:auto"}]]
        for i, (ref, nodes, vision) in enumerate(models):
            mark = "✅ " if ref == current else ""
            label = f"{mark}{self.short_name(ref)}{' 👁' if vision else ''} · {nodes}"
            rows.append([{"text": label, "callback_data": f"m:{i}"}])
        return f"Текущая модель: {self.short_name(current)}\nВыберите:", {"inline_keyboard": rows}

    async def do_net(self) -> str:
        data = (await self.client.get(f"{self.gateway}/network", headers=self.gw_headers)).json()
        lines = [f"Высота чейна: {data['height']}"]
        for n in data["nodes"]:
            dot = "🟢" if n["alive"] else "🔴"
            models = ", ".join(m.split("::")[0].split("/")[-1] for m in n["models"]) or "—"
            lines.append(f"{dot} {n['address'][:12]}… · {models} · rep {n['reputation']:.2f}")
        return "\n".join(lines)

    async def do_plan(self, vram: int) -> str:
        data = (await self.client.get(f"{self.gateway}/v1/plan", headers=self.gw_headers, params={"vram_mb": vram})).json()
        lines = [f"План для {vram} MB VRAM:"]
        for o in data["options"][:5]:
            warm = " 🔥" if o.get("already_served_on_network") else ""
            lines.append(f"• {o['ref'].split('::')[0].split('/')[-1]} — ctx {o['max_context']:,},"
                         f" kv {o['kv_type']}{warm}")
        return "\n".join(lines)

    HELP = (
        "ΔV — децентрализованная AI-сеть. Полноценный агент прямо в чате.\n\n"
        "• Просто пишите — по умолчанию отвечает агент: сам ищет в интернете,"
        " когда нужно, помнит сессию (каждый ответ = чек в чейне).\n"
        "• Пришлите фото 🖼 — модель со зрением опишет/ответит по картинке"
        " (подпись к фото = ваш вопрос).\n\n"
        "/fast — быстрый режим без инструментов\n"
        "/smart — вернуть агентский режим (по умолчанию)\n"
        "/agent <задача> — явный запуск агента\n"
        "/model — выбрать модель кнопкой (список берётся из сети)\n"
        "/models — что сеть отдаёт прямо сейчас\n"
        "/net — состояние сети\n"
        "/plan [vram_mb] — что запускать на железе\n"
        "/id — ваш Telegram ID (для allowlist)\n"
        "/reset — забыть диалог"
    )

    # ----------------------------------------------------------- dispatch
    def _duplicate(self, update_id: int) -> bool:
        if update_id in self._seen:
            return True
        self._seen[update_id] = None
        while len(self._seen) > 500:
            self._seen.popitem(last=False)
        return False

    async def handle_callback(self, cq: dict) -> None:
        """A tap on the model picker. Telegram wants every callback answered,
        or the client spins on the button forever."""
        cq_id = cq.get("id")
        data = cq.get("data") or ""
        msg = cq.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        user_id = (cq.get("from") or {}).get("id")
        if not chat_id or not self.allowed(user_id):
            await self.tg("answerCallbackQuery", callback_query_id=cq_id, text="⛔ нет доступа")
            return
        if not data.startswith("m:"):
            await self.tg("answerCallbackQuery", callback_query_id=cq_id)
            return

        key = data[2:]
        if key == "auto":
            chosen = "auto"
        else:
            choices = self._model_choices.get(chat_id) or []
            try:
                chosen = choices[int(key)]
            except (ValueError, IndexError):
                # the list moved on (a node went down) — redraw instead of
                # silently setting the wrong model
                await self.tg("answerCallbackQuery", callback_query_id=cq_id,
                              text="список устарел, открываю заново")
                text_, markup = await self.model_keyboard(chat_id)
                if markup:
                    await self.tg("sendMessage", chat_id=chat_id, text=text_, reply_markup=markup)
                return

        self.models[chat_id] = chosen
        await self.tg("answerCallbackQuery", callback_query_id=cq_id,
                      text=f"выбрано: {self.short_name(chosen)}")
        await self.tg("editMessageText", chat_id=chat_id, message_id=msg.get("message_id"),
                      text=f"✅ модель: {self.short_name(chosen)}")

    async def handle(self, update: dict) -> None:
        if self._duplicate(update.get("update_id", -1)):
            return
        if update.get("callback_query"):
            await self.handle_callback(update["callback_query"])
            return
        msg = update.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        user_id = (msg.get("from") or {}).get("id")
        text = (msg.get("text") or msg.get("caption") or "").strip()
        photo = msg.get("photo") or []
        # image documents (sent as file) also count
        doc = msg.get("document") or {}
        if doc.get("mime_type", "").startswith("image/"):
            photo = photo or [{"file_id": doc["file_id"]}]
        if not chat_id or (not text and not photo):
            return
        if not self.allowed(user_id):
            await self.send(chat_id, "⛔ доступ по allowlist; ваш id: " + str(user_id))
            return

        # a photo -> vision (the served model must support images)
        if photo:
            async with self._locks[chat_id]:
                await self.tg("sendChatAction", chat_id=chat_id, action="typing")
                uri = await self.download_photo(photo[-1]["file_id"])  # largest size
                if not uri:
                    await self.send(chat_id, "⚠ не смог скачать изображение")
                    return
                try:
                    await self.send(chat_id, await self.do_vision(chat_id, text, uri))
                except httpx.HTTPError as exc:
                    await self.send(chat_id, f"⚠ сеть недоступна: {exc}")
            return

        cmd, _, arg = text.partition(" ")
        cmd = cmd.split("@")[0].lower()
        # one message at a time per chat: concurrent replies corrupt history
        async with self._locks[chat_id]:
            try:
                if cmd in ("/start", "/help"):
                    await self.send(chat_id, self.HELP)
                elif cmd == "/id":
                    await self.send(chat_id, f"ваш Telegram ID: {user_id}")
                elif cmd == "/reset":
                    self.histories.pop(chat_id, None)
                    await self.send(chat_id, "🗑 диалог забыт")
                elif cmd == "/fast":
                    self.modes[chat_id] = "fast"
                    await self.send(chat_id, "⚡ быстрый режим: без поиска и инструментов")
                elif cmd == "/smart":
                    self.modes[chat_id] = "smart"
                    await self.send(chat_id, "🛠 агентский режим: поиск и память включены")
                elif cmd == "/model":
                    if arg.strip():                       # explicit ref still works
                        self.models[chat_id] = arg.strip()
                        await self.send(chat_id, f"модель: {self.models[chat_id]}")
                    else:                                  # no arg -> pick a button
                        text_, markup = await self.model_keyboard(chat_id)
                        if markup:
                            await self.tg("sendMessage", chat_id=chat_id, text=text_,
                                          reply_markup=markup)
                        else:
                            await self.send(chat_id, text_)
                elif cmd == "/models":
                    await self.send(chat_id, await self.do_models())
                elif cmd == "/net":
                    await self.send(chat_id, await self.do_net())
                elif cmd == "/plan":
                    await self.send(chat_id, await self.do_plan(int(arg) if arg.strip() else 8176))
                elif cmd == "/agent":
                    if not arg.strip():
                        await self.send(chat_id, "формат: /agent <задача>")
                        return
                    await self.tg("sendChatAction", chat_id=chat_id, action="typing")
                    try:
                        await self.send(chat_id, await self.do_agent(chat_id, arg.strip()))
                    except AgentFailed as exc:
                        await self.send(chat_id, f"⚠ агент: {exc}")
                else:
                    await self.tg("sendChatAction", chat_id=chat_id, action="typing")
                    if self.modes.get(chat_id, "smart") == "fast":
                        await self.send(chat_id, await self.do_chat(chat_id, text))
                    else:
                        await self.send(chat_id, await self.do_smart(chat_id, text))
            except httpx.HTTPError as exc:
                await self.send(chat_id, f"⚠ сеть недоступна: {exc}")

    async def run(self) -> None:
        me = await self.tg("getMe")
        log.info("bot @%s online, gateway %s", (me or {}).get("username"), self.gateway)
        while True:
            try:
                updates = await self.tg("getUpdates", offset=self._offset, timeout=50)
            except httpx.HTTPError as exc:
                log.warning("poll error: %s", self._redact(exc))
                await asyncio.sleep(5)
                continue
            for update in updates or []:
                self._offset = update["update_id"] + 1
                asyncio.create_task(self.handle(update))


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        print("TELEGRAM_BOT_TOKEN is required", file=sys.stderr)
        return 1
    gateway = os.environ.get("DELTAV_GATEWAY", "http://127.0.0.1:9000")
    allow = {int(x) for x in os.environ.get("DELTAV_ALLOW", "").split(",") if x.strip()}
    bot = TgBot(token, gateway, allow, api_key=os.environ.get("DELTAV_API_KEY", ""))
    asyncio.run(bot.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
