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

    def infer(self, request: InferRequest) -> InferResult:
        resp = self.client.post(f"{self.base_url}/completion", json={
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "stop": self.STOP,
            "cache_prompt": False,
        })
        resp.raise_for_status()
        data = resp.json()
        return InferResult(
            text=data.get("content", ""),
            tokens_in=int(data.get("tokens_evaluated", 0)),
            tokens_out=max(1, int(data.get("tokens_predicted", 1))),
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=False,
        )

    def infer_stream(self, request: InferRequest):
        pieces: list[str] = []
        tokens_in = 0
        tokens_out = 0
        with self.client.stream("POST", f"{self.base_url}/completion", json={
            "prompt": request.prompt,
            "n_predict": request.max_tokens,
            "temperature": request.temperature,
            "seed": request.seed,
            "stop": self.STOP,
            "cache_prompt": False,
            "stream": True,
        }) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                piece = event.get("content", "")
                if piece:
                    pieces.append(piece)
                    yield piece
                if event.get("stop"):
                    timings = event.get("timings", {})
                    tokens_in = int(event.get("tokens_evaluated", 0)
                                    or timings.get("prompt_n", 0))
                    tokens_out = int(event.get("tokens_predicted", 0)
                                     or timings.get("predicted_n", 0))
                    break
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
        resp.raise_for_status()
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
