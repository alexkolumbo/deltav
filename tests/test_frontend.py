"""Web chat UI + Telegram bot bridge."""
import json

import httpx
import pytest

from deltav.config import ChainParams
from deltav.crypto import KeyPair
from deltav.gateway import GatewayDaemon
from deltav.tgbot import TgBot


async def test_gateway_serves_chat_ui():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500)))
    gateway = GatewayDaemon(KeyPair.generate(), node_urls=["http://n:1"],
                            params=ChainParams(), client=client)
    api = httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                            base_url="http://gw")
    resp = await api.get("/chat")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "v1/chat/completions" in resp.text  # the UI talks to the local API
    await api.aclose()
    await client.aclose()


def make_bot(routes: dict) -> tuple[TgBot, list]:
    """Bot with a fake Telegram API + fake gateway; returns sent messages."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "api.telegram.org" in url:
            if url.endswith("/sendMessage"):
                sent.append(json.loads(request.content))
            return httpx.Response(200, json={"ok": True, "result": []})
        for suffix, respond in routes.items():
            if suffix in url:
                return respond(request)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TgBot("tok", "http://gw:9000", allow=set(), client=client), sent


def update(text: str, chat_id: int = 7, user_id: int = 42) -> dict:
    return {"update_id": 1, "message": {
        "chat": {"id": chat_id}, "from": {"id": user_id}, "text": text}}


async def test_fast_mode_chats_with_history_and_receipt():
    def chat_route(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][-1] == {"role": "user", "content": "привет"}
        return httpx.Response(200, json={
            "model": "m/x::f", "choices": [{"message": {"content": "здравствуй"}}],
            "usage": {"completion_tokens": 5},
            "deltav": {"receipt_tx": "abcdef1234567890"},
        })

    bot, sent = make_bot({"/v1/chat/completions": chat_route})
    bot.modes[7] = "fast"
    await bot.handle(update("привет"))
    assert "здравствуй" in sent[0]["text"]
    assert "abcdef12" in sent[0]["text"]  # the receipt travels to the phone
    assert bot.histories[7][-1]["role"] == "assistant"


async def test_smart_mode_is_default_and_uses_agent_with_context():
    calls = []

    def agent_route(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json={
            "answer": "нашёл в сети: матч 11 июня", "model_calls": 2, "finished": True,
            "steps": [{"tool": "web_search", "arguments": {"query": "fifa 2026"},
                       "result": "...", "node": "n", "receipt_tx": "r"}],
        })

    bot, sent = make_bot({"/v1/agents/run": agent_route})
    await bot.handle(update("когда матч fifa 2026?"))
    assert calls[0]["session_id"] == "tg-7"
    assert "web_search" in sent[0]["text"] and "11 июня" in sent[0]["text"]
    # dialog history now feeds the next smart task as context
    await bot.handle({**update("а где?"), "update_id": 2})
    assert "Контекст диалога" in calls[1]["task"]
    assert "а где?" in calls[1]["task"]


async def test_duplicate_updates_are_ignored():
    def agent_route(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"answer": "ok", "model_calls": 1,
                                         "finished": True, "steps": []})

    bot, sent = make_bot({"/v1/agents/run": agent_route})
    u = update("привет")
    await bot.handle(u)
    await bot.handle(u)  # telegram redelivery after a broken long poll
    assert len(sent) == 1


async def test_allowlist_blocks_strangers():
    bot, sent = make_bot({})
    bot.allow = {999}
    await bot.handle(update("привет", user_id=42))
    assert "⛔" in sent[0]["text"] and "42" in sent[0]["text"]


async def test_agent_command_uses_session_memory():
    def agent_route(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["session_id"] == "tg-7"
        return httpx.Response(200, json={
            "answer": "готово", "model_calls": 2, "finished": True,
            "steps": [{"tool": "web_search", "arguments": {"query": "x"},
                       "result": "...", "node": "n", "receipt_tx": "r"}],
        })

    bot, sent = make_bot({"/v1/agents/run": agent_route})
    await bot.handle(update("/agent найди что-нибудь"))
    assert "web_search" in sent[0]["text"] and "готово" in sent[0]["text"]


def test_history_trimming():
    bot, _ = make_bot({})
    h = bot.histories[1] = []
    for i in range(40):
        h += [{"role": "user", "content": "x" * 400},
              {"role": "assistant", "content": "y" * 400}]
    trimmed = bot.history(1)
    assert sum(len(m["content"]) for m in trimmed) <= 9000
    assert trimmed[-1]["role"] == "assistant"  # newest kept, oldest dropped
