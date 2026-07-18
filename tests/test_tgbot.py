"""Telegram bot: model picking driven by what the NETWORK actually serves.

Both sides are mocked — the Telegram API and the gateway — so nothing here
touches a real bot token or a live node.
"""
import json

import httpx
import pytest

from deltav.tgbot import TgBot

GW = "http://gw.test"

# two served models (one with vision) + one the network does NOT serve
MODELS_PAYLOAD = {
    "data": [
        {"id": "prism-ml/Bonsai-27B-gguf::Bonsai-27B-Q1_0.gguf",
         "deltav": {"served_by": ["dv1a"], "vision": False, "kind": "chat"}},
        {"id": "empero-ai/Qwythos-9B-GGUF::Qwythos-9B-Q4_K_M.gguf",
         "deltav": {"served_by": ["dv1a", "dv1b"], "vision": True, "kind": "chat"}},
        {"id": "bartowski/Llama-3.3-70B-Instruct-GGUF::L-Q4.gguf",
         "deltav": {"served_by": [], "vision": False, "kind": "chat"}},  # nobody serves it
        {"id": "nomic-ai/nomic-embed-text-v1.5-GGUF::nomic-embed.gguf",
         "deltav": {"served_by": ["dv1c"], "vision": False, "kind": "embedding"}},
        {"id": "black-forest-labs/FLUX.1-schnell",
         "deltav": {"served_by": ["dv1d"], "vision": False, "kind": "image"}},
    ]
}


def _bot(sent: list):
    """A bot whose Telegram calls are recorded and whose gateway is faked."""
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.startswith(GW) and "/v1/models" in url:
            return httpx.Response(200, json=MODELS_PAYLOAD)
        if "api.telegram.org" in url:
            method = url.rsplit("/", 1)[-1]
            sent.append((method, json.loads(request.content or b"{}")))
            return httpx.Response(200, json={"ok": True, "result": {}})
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return TgBot(token="T", gateway=GW, client=client)


async def test_served_models_hides_what_no_node_serves():
    """Offering an unserved model would just yield 'no node serves X'."""
    bot = _bot([])
    models = await bot.served_models()
    refs = [ref for ref, _, _ in models]
    assert "bartowski/Llama-3.3-70B-Instruct-GGUF::L-Q4.gguf" not in refs
    assert len(models) == 2
    # busiest first, and vision is flagged
    assert models[0][1] == 2 and models[0][2] is True


async def test_served_models_offers_only_chat_models():
    """The seed serves an embedding model too (and a node may serve an image
    one). Picking either for a chat fails on every single message."""
    bot = _bot([])
    refs = [ref for ref, _, _ in await bot.served_models()]
    assert not any("nomic-embed" in r or "FLUX" in r for r in refs)


async def test_model_keyboard_uses_indexes_not_refs():
    """Telegram caps callback_data at 64 bytes; real refs are far longer, so
    buttons must carry an index into a snapshot."""
    bot = _bot([])
    text, markup = await bot.model_keyboard(chat_id=7)
    rows = markup["inline_keyboard"]
    assert rows[0][0]["callback_data"] == "m:auto"
    payloads = [b["callback_data"] for row in rows[1:] for b in row]
    assert payloads == ["m:0", "m:1"]
    assert all(len(p.encode()) <= 64 for p in payloads)
    assert bot._model_choices[7][0].startswith("empero-ai/")   # busiest first


async def test_tapping_a_button_selects_that_model():
    sent = []
    bot = _bot(sent)
    await bot.model_keyboard(chat_id=7)                 # snapshot the list
    await bot.handle({"update_id": 1, "callback_query": {
        "id": "cb1", "data": "m:1", "from": {"id": 42},
        "message": {"message_id": 5, "chat": {"id": 7}}}})
    assert bot.models[7] == "prism-ml/Bonsai-27B-gguf::Bonsai-27B-Q1_0.gguf"
    methods = [m for m, _ in sent]
    assert "answerCallbackQuery" in methods      # or the button spins forever
    assert "editMessageText" in methods


async def test_auto_button_resets_to_network_choice():
    bot = _bot([])
    bot.models[7] = "something/else"
    await bot.handle({"update_id": 2, "callback_query": {
        "id": "cb2", "data": "m:auto", "from": {"id": 42},
        "message": {"message_id": 5, "chat": {"id": 7}}}})
    assert bot.models[7] == "auto"


async def test_stale_index_redraws_instead_of_picking_the_wrong_model():
    """The served list can change between drawing the picker and the tap (a
    node drops). Never silently resolve to whatever now sits at that index."""
    sent = []
    bot = _bot(sent)
    bot._model_choices[7] = []                    # snapshot went stale/empty
    await bot.handle({"update_id": 3, "callback_query": {
        "id": "cb3", "data": "m:1", "from": {"id": 42},
        "message": {"message_id": 5, "chat": {"id": 7}}}})
    assert 7 not in bot.models                    # nothing was selected
    assert any(m == "sendMessage" for m, _ in sent)   # a fresh picker was sent


async def test_callback_respects_the_allowlist():
    sent = []
    bot = _bot(sent)
    bot.allow = {999}                             # 42 is not allowed
    await bot.handle({"update_id": 4, "callback_query": {
        "id": "cb4", "data": "m:auto", "from": {"id": 42},
        "message": {"message_id": 5, "chat": {"id": 7}}}})
    assert 7 not in bot.models
    assert sent and sent[0][0] == "answerCallbackQuery"


async def test_do_models_lists_only_served_and_points_at_the_picker():
    bot = _bot([])
    text = await bot.do_models()
    assert "Llama-3.3-70B" not in text
    assert "Qwythos" in text and "Bonsai" in text
    assert "/model" in text
