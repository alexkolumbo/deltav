"""Zero-config local proxy: forwarding, key injection, streaming, failover.

Uses a real ASGI upstream (not MockTransport) so true streaming is exercised.
"""
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from deltav.proxy import build_proxy

from conftest import MultiTransport


def fake_gateway(record: dict) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health(request: Request):
        record["auth"] = request.headers.get("authorization")
        return {"gw": "live"}

    @app.get("/api/tags")
    async def tags(request: Request):
        record["auth"] = request.headers.get("authorization")
        return {"models": []}

    @app.get("/v1/plan")
    async def plan(request: Request):
        record["url"] = str(request.url)
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        record["auth"] = request.headers.get("authorization")
        record["body"] = body
        if body.get("stream"):
            async def gen():
                yield 'data: {"a":1}\n\n'
                yield 'data: {"a":2}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(gen(), media_type="text/event-stream")
        return {"ok": True}

    return app


def _proxy_client(gateways: list[str], transport: MultiTransport, key="dvk_x"):
    upstream = httpx.AsyncClient(transport=transport)
    app = build_proxy(gateways, api_key=key, client=upstream)
    api = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy")
    return api, upstream


async def test_injects_key_and_forwards():
    rec = {}
    t = MultiTransport(); t.add("http://gw:9000", fake_gateway(rec))
    api, up = _proxy_client(["http://gw:9000"], t, key="dvk_secret")
    # client sends NO Authorization of its own
    r = await api.post("/v1/chat/completions", json={"model": "auto"})
    assert r.status_code == 200 and r.json()["ok"]
    assert rec["auth"] == "Bearer dvk_secret"          # proxy injected the key
    assert rec["body"]["model"] == "auto"              # body forwarded
    await api.aclose(); await up.aclose()


async def test_client_needs_no_credentials():
    rec = {}
    t = MultiTransport(); t.add("http://gw:9000", fake_gateway(rec))
    api, up = _proxy_client(["http://gw:9000"], t, key="dvk_held")
    await api.get("/api/tags")                          # Ollama surface, no key sent
    assert rec["auth"] == "Bearer dvk_held"
    await api.aclose(); await up.aclose()


async def test_query_string_preserved():
    rec = {}
    t = MultiTransport(); t.add("http://gw:9000", fake_gateway(rec))
    api, up = _proxy_client(["http://gw:9000"], t)
    await api.get("/v1/plan?vram_mb=8176&objective=balanced")
    assert "vram_mb=8176" in rec["url"] and "objective=balanced" in rec["url"]
    await api.aclose(); await up.aclose()


async def test_streaming_passthrough():
    rec = {}
    t = MultiTransport(); t.add("http://gw:9000", fake_gateway(rec))
    api, up = _proxy_client(["http://gw:9000"], t)
    lines = []
    async with api.stream("POST", "/v1/chat/completions", json={"stream": True}) as resp:
        assert "text/event-stream" in resp.headers["content-type"]
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                lines.append(line)
    assert "data: [DONE]" in lines and len(lines) == 3
    await api.aclose(); await up.aclose()


async def test_failover_to_second_gateway():
    rec = {}
    t = MultiTransport(); t.add("http://live:9000", fake_gateway(rec))
    # "dead" isn't registered -> ConnectError -> proxy fails over to live
    api, up = _proxy_client(["http://dead:9000", "http://live:9000"], t)
    r = await api.get("/health")
    assert r.status_code == 200 and r.json()["gw"] == "live"
    await api.aclose(); await up.aclose()


async def test_info_endpoint():
    api, up = _proxy_client(["http://a:9000", "http://b:9000"], MultiTransport())
    d = (await api.get("/_deltav")).json()
    assert d["gateways"] == ["http://a:9000", "http://b:9000"]
    await api.aclose(); await up.aclose()
