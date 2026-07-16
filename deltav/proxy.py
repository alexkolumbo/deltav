"""Zero-config local proxy — the simplest possible integration.

`deltav serve` runs a tiny reverse proxy on localhost that holds your
gateway URL(s) and API key. Any local tool then points at
`http://localhost:11434` (Ollama's default port) or
`http://localhost:11434/v1` (OpenAI) with NO key and NO config — the
proxy attaches your key and forwards to the network, failing over across
gateways. Because it's a transparent catch-all, every surface (OpenAI,
Ollama, Anthropic, companion, swarm) works through it unchanged.
"""
from __future__ import annotations

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

# Hop-by-hop headers we must not forward.
_STRIP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding",
          "authorization", "accept-encoding"}


def build_proxy(base_urls: list[str], api_key: str = "deltav",
                client: httpx.AsyncClient | None = None) -> FastAPI:
    bases = [u.rstrip("/") for u in base_urls]
    app = FastAPI(title="Delta V local proxy", version="0.1.0")
    http = client or httpx.AsyncClient(timeout=None)

    def out_headers(request: Request) -> dict:
        h = {k: v for k, v in request.headers.items() if k.lower() not in _STRIP}
        h["Authorization"] = f"Bearer {api_key}"   # inject the held key
        return h

    @app.get("/_deltav")
    async def info() -> dict:
        return {"proxy": "deltav", "gateways": bases}

    @app.api_route("/{path:path}",
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def forward(path: str, request: Request):
        body = await request.body()
        headers = out_headers(request)
        query = request.url.query
        last_err = "no gateway reachable"
        for base in bases:
            url = f"{base}/{path}" + (f"?{query}" if query else "")
            try:
                req = http.build_request(request.method, url, headers=headers, content=body)
                resp = await http.send(req, stream=True)
            except httpx.HTTPError as exc:
                last_err = str(exc)
                continue
            if resp.status_code >= 500 and base != bases[-1]:
                await resp.aclose()
                continue

            passthrough = {k: v for k, v in resp.headers.items()
                           if k.lower() not in ("content-length", "transfer-encoding",
                                                "content-encoding", "connection")}

            async def stream():
                try:
                    async for chunk in resp.aiter_raw():
                        yield chunk
                finally:
                    await resp.aclose()

            return StreamingResponse(stream(), status_code=resp.status_code,
                                     headers=passthrough,
                                     media_type=resp.headers.get("content-type"))
        return Response(content=f'{{"error":"{last_err}"}}', status_code=502,
                        media_type="application/json")

    return app


def run_proxy(base_urls: list[str], api_key: str, host: str = "127.0.0.1",
              port: int = 11434) -> None:
    import uvicorn

    app = build_proxy(base_urls, api_key)
    print(f"Delta V proxy on http://{host}:{port}  →  {', '.join(base_urls)}")
    print("  point any tool at this address, no key needed:")
    print(f"    OpenAI base URL : http://{host}:{port}/v1")
    print(f"    Ollama host     : http://{host}:{port}")
    print(f"    Anthropic       : http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
