"""LlamaServerBackend: HTTP contract with a llama.cpp server."""
import json

import httpx

from deltav.compute.base import EmbedRequest, InferRequest
from deltav.compute.llamaserver import LlamaServerBackend

MODEL = "bartowski/Llama-3.2-3B-Instruct-GGUF::Llama-3.2-3B-Instruct-Q4_K_M.gguf"


def make_backend(handler) -> LlamaServerBackend:
    return LlamaServerBackend(
        base_url="http://llamasrv.test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_infer_maps_completion_api():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url).endswith("/completion")
        payload = json.loads(request.content)
        assert payload["prompt"] == "hi amd"
        assert payload["n_predict"] == 32
        return httpx.Response(200, json={
            "content": "hello from vulkan",
            "tokens_evaluated": 4,
            "tokens_predicted": 5,
        })

    result = make_backend(handler).infer(
        InferRequest(prompt="hi amd", model_ref=MODEL, max_tokens=32, seed=7))
    assert result.text == "hello from vulkan"
    assert result.tokens_in == 4 and result.tokens_out == 5
    assert result.deterministic is False  # fuzzy spot checks


def test_infer_stream_collects_pieces():
    def handler(request: httpx.Request) -> httpx.Response:
        body = (
            'data: {"content": "hel", "stop": false}\n\n'
            'data: {"content": "lo", "stop": false}\n\n'
            'data: {"content": "", "stop": true, "tokens_evaluated": 3, "tokens_predicted": 2}\n\n'
        )
        return httpx.Response(200, content=body.encode(),
                              headers={"content-type": "text/event-stream"})

    items = list(make_backend(handler).infer_stream(
        InferRequest(prompt="x", model_ref=MODEL, max_tokens=8)))
    pieces, final = items[:-1], items[-1]
    assert pieces == ["hel", "lo"]
    assert final.text == "hello"
    assert final.tokens_in == 3 and final.tokens_out == 2


def test_server_error_detail_is_surfaced():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {
            "message": "request (5402 tokens) exceeds the available context size (4096 tokens)"}})

    import pytest
    with pytest.raises(RuntimeError, match="exceeds the available context"):
        make_backend(handler).infer(InferRequest(prompt="long", model_ref=MODEL))


def test_fixed_models_flag():
    assert LlamaServerBackend.dynamic_models is False


def test_embed_handles_single_and_nested():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[
            {"embedding": [[0.1, 0.2]]},   # nested per-token form
            {"embedding": [0.3, 0.4]},
        ])

    result = make_backend(handler).embed(EmbedRequest(texts=["a", "b"], model_ref=MODEL))
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
