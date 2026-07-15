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
import json
import logging
import os
import sys

import httpx

log = logging.getLogger("deltav.tgbot")

HISTORY_CHAR_BUDGET = 9000
MAX_TOKENS = 900


class TgBot:
    def __init__(self, token: str, gateway: str, allow: set[int] | None = None,
                 client: httpx.AsyncClient | None = None):
        self.api = f"https://api.telegram.org/bot{token}"
        self.gateway = gateway.rstrip("/")
        self.allow = allow or set()
        self.client = client or httpx.AsyncClient(timeout=330.0)
        self.histories: dict[int, list[dict]] = {}
        self.models: dict[int, str] = {}  # chat_id -> chosen model
        self._offset = 0

    # ----------------------------------------------------------- telegram
    async def tg(self, method: str, **params):
        resp = await self.client.post(f"{self.api}/{method}", json=params)
        data = resp.json()
        if not data.get("ok"):
            log.warning("telegram %s failed: %s", method, data)
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
        resp = await self.client.post(f"{self.gateway}/v1/chat/completions", json={
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

    async def do_agent(self, chat_id: int, task: str) -> str:
        resp = await self.client.post(f"{self.gateway}/v1/agents/run", json={
            "task": task,
            "model": self.models.get(chat_id, "auto"),
            "max_steps": 6,
            "session_id": f"tg-{chat_id}",
        })
        if resp.status_code != 200:
            return f"⚠ агент: {resp.json().get('detail', resp.status_code)}"
        data = resp.json()
        lines = []
        for s in data.get("steps", []):
            lines.append(f"🛠 {s['tool']}({json.dumps(s['arguments'], ensure_ascii=False)[:120]})")
        lines.append("")
        lines.append(data.get("answer", "(пусто)"))
        lines.append(f"\n· агент: {data.get('model_calls')} вызовов, память сессии tg-{chat_id}")
        return "\n".join(lines)

    async def do_models(self) -> str:
        data = (await self.client.get(f"{self.gateway}/v1/models")).json()
        lines = ["Модели на сети:"]
        for m in data["data"]:
            served = len(m["deltav"]["served_by"])
            if served:
                lines.append(f"• {m['id'].split('::')[0]}  ({served} нод)")
        lines.append("\n/model <repo или auto> — выбрать")
        return "\n".join(lines)

    async def do_net(self) -> str:
        data = (await self.client.get(f"{self.gateway}/network")).json()
        lines = [f"Высота чейна: {data['height']}"]
        for n in data["nodes"]:
            dot = "🟢" if n["alive"] else "🔴"
            models = ", ".join(m.split("::")[0].split("/")[-1] for m in n["models"]) or "—"
            lines.append(f"{dot} {n['address'][:12]}… · {models} · rep {n['reputation']:.2f}")
        return "\n".join(lines)

    async def do_plan(self, vram: int) -> str:
        data = (await self.client.get(f"{self.gateway}/v1/plan",
                                      params={"vram_mb": vram})).json()
        lines = [f"План для {vram} MB VRAM:"]
        for o in data["options"][:5]:
            warm = " 🔥" if o.get("already_served_on_network") else ""
            lines.append(f"• {o['ref'].split('::')[0].split('/')[-1]} — ctx {o['max_context']:,},"
                         f" kv {o['kv_type']}{warm}")
        return "\n".join(lines)

    HELP = (
        "ΔV — децентрализованная AI-сеть.\n\n"
        "Просто пишите — отвечает модель с сети (каждый ответ = чек в чейне).\n\n"
        "/agent <задача> — агент с web-поиском и памятью\n"
        "/model [ref|auto] — выбрать модель\n"
        "/models — что доступно\n"
        "/net — состояние сети\n"
        "/plan [vram_mb] — что запускать на железе\n"
        "/reset — забыть диалог"
    )

    # ----------------------------------------------------------- dispatch
    async def handle(self, update: dict) -> None:
        msg = update.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        user_id = (msg.get("from") or {}).get("id")
        text = (msg.get("text") or "").strip()
        if not chat_id or not text:
            return
        if not self.allowed(user_id):
            await self.send(chat_id, "⛔ доступ по allowlist; ваш id: " + str(user_id))
            return

        cmd, _, arg = text.partition(" ")
        cmd = cmd.split("@")[0].lower()
        try:
            if cmd in ("/start", "/help"):
                await self.send(chat_id, self.HELP)
            elif cmd == "/reset":
                self.histories.pop(chat_id, None)
                await self.send(chat_id, "🗑 диалог забыт")
            elif cmd == "/model":
                self.models[chat_id] = arg.strip() or "auto"
                await self.send(chat_id, f"модель: {self.models[chat_id]}")
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
                await self.send(chat_id, await self.do_agent(chat_id, arg.strip()))
            else:
                await self.tg("sendChatAction", chat_id=chat_id, action="typing")
                await self.send(chat_id, await self.do_chat(chat_id, text))
        except httpx.HTTPError as exc:
            await self.send(chat_id, f"⚠ сеть недоступна: {exc}")

    async def run(self) -> None:
        me = await self.tg("getMe")
        log.info("bot @%s online, gateway %s", (me or {}).get("username"), self.gateway)
        while True:
            try:
                updates = await self.tg("getUpdates", offset=self._offset, timeout=50)
            except httpx.HTTPError as exc:
                log.warning("poll error: %s", exc)
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
    bot = TgBot(token, gateway, allow)
    asyncio.run(bot.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
