"""llama-server backend: drive a local llama.cpp server over HTTP.

Why this exists: `llama-cpp-python` needs a compiler matched to your
GPU stack, but ggml-org ships prebuilt `llama-server` binaries for every
backend — including **Vulkan**, which runs on AMD (and Intel/NVIDIA)
GPUs on Windows with zero build steps. A node points this backend at a
running server and joins the network:

    llama-server -m model.gguf --port 8085 -ngl 99   # Vulkan build = AMD GPU
    set LLAMA_SERVER_URL=http://127.0.0.1:8085
    deltav node --backend llamaserver --model <ref-you-loaded> ...

The server holds ONE model; announce exactly that ref on-chain.
Sampling on GPU is not bit-reproducible across machines, so
deterministic=False — spot checks use the fuzzy token-count path.
"""
from __future__ import annotations

import json
import os

import httpx

from .base import ComputeBackend, EmbedRequest, EmbedResult, InferRequest, InferResult, register_backend

DEFAULT_URL = "http://127.0.0.1:8085"


@register_backend
class LlamaServerBackend(ComputeBackend):
    name = "llamaserver"
    vendors = ("amd", "nvidia", "intel", "apple", "cpu")
    deterministic = False
    dynamic_models = False  # the server pre-loads ONE model at startup

    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None):
        self.base_url = (base_url or os.environ.get("LLAMA_SERVER_URL") or DEFAULT_URL).rstrip("/")
        self.client = client or httpx.Client(timeout=600.0)

    @classmethod
    def is_available(cls) -> bool:
        url = (os.environ.get("LLAMA_SERVER_URL") or DEFAULT_URL).rstrip("/")
        try:
            return httpx.get(f"{url}/health", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def load(self, model_ref: str) -> None:
        """The server pre-loads its model at startup; nothing to do here."""

    # The network's prompt convention is "role: content" lines — stop the
    # raw completion before the model starts speaking for the other roles.
    STOP = ["\nuser:", "\nsystem:", "\nassistant:", "\ntool ("]

    @staticmethod
    def _raise_with_detail(resp: httpx.Response) -> None:
        """Surface llama-server's error message (e.g. context overflow)
        instead of a bare 500 — the router and logs need the reason."""
        if resp.status_code >= 400:
            raise RuntimeError(f"llama-server {resp.status_code}: {resp.text[:300]}")

    def _chat_body(self, request: InferRequest) -> dict:
        """We send the whole rendered conversation as ONE user message to
        the server's OpenAI endpoint, so llama.cpp applies the model's OWN
        chat template from GGUF metadata (Qwen needs ChatML or it drifts
        into Chinese; Llama needs its header tokens; the server knows —
        we don't have to). STOP stays as a role-play safety net."""
        return {
            "model": "default",
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "stop": self.STOP,
        }

    def infer(self, request: InferRequest) -> InferResult:
        resp = self.client.post(f"{self.base_url}/v1/chat/completions",
                                json=self._chat_body(request))
        self._raise_with_detail(resp)
        data = resp.json()
        usage = data.get("usage") or {}
        return InferResult(
            text=data["choices"][0]["message"].get("content") or "",
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=max(1, int(usage.get("completion_tokens", 1))),
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=False,
        )

    def infer_stream(self, request: InferRequest):
        pieces: list[str] = []
        tokens_in = 0
        tokens_out = 0
        body = self._chat_body(request) | {
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        with self.client.stream("POST", f"{self.base_url}/v1/chat/completions",
                                json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                self._raise_with_detail(resp)
            for line in resp.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[len("data: "):])
                usage = event.get("usage")
                if usage:
                    tokens_in = int(usage.get("prompt_tokens", 0))
                    tokens_out = int(usage.get("completion_tokens", 0))
                choices = event.get("choices") or []
                if not choices:
                    continue
                piece = choices[0].get("delta", {}).get("content") or ""
                if piece:
                    pieces.append(piece)
                    yield piece
        yield InferResult(
            text="".join(pieces),
            tokens_in=tokens_in or max(1, len(request.prompt.split())),
            tokens_out=tokens_out or max(1, len(pieces)),
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=False,
        )

    # Embeddings work when the server was started with --embedding.
    supports_embeddings = True

    def embed(self, request: EmbedRequest) -> EmbedResult:
        resp = self.client.post(f"{self.base_url}/embedding",
                                json={"content": request.texts})
        self._raise_with_detail(resp)
        data = resp.json()
        rows = data if isinstance(data, list) else [data]
        vectors = []
        for row in rows:
            vec = row.get("embedding")
            if vec and isinstance(vec[0], list):  # server may nest per-token
                vec = vec[0]
            vectors.append([float(x) for x in vec])
        return EmbedResult(
            vectors=vectors,
            tokens=sum(max(1, len(t.split())) for t in request.texts),
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=False,
        )
