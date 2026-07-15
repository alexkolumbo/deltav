"""Groq LPU backend.

Groq exposes its LPU clusters through an OpenAI-compatible HTTP API
rather than local drivers, so a "Groq node" on the Delta V network is a
machine that holds a GROQ_API_KEY and relays jobs to LPU hardware.
Announce the models the key can serve as refs like
"groq/llama-3.3-70b-versatile" (the "groq/" prefix is stripped for the
upstream call).

LPU sampling is not bit-reproducible -> deterministic=False, so spot
checks use fuzzy verification (token-count tolerance) automatically.
"""
from __future__ import annotations

import os

import httpx

from .base import ComputeBackend, InferRequest, InferResult, register_backend

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


def _upstream_model(model_ref: str) -> str:
    ref = model_ref.split("::", 1)[0]
    return ref[len("groq/"):] if ref.startswith("groq/") else ref


@register_backend
class GroqBackend(ComputeBackend):
    name = "groq"
    vendors = ("groq",)
    deterministic = False

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.base_url = (base_url or os.environ.get("GROQ_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.client = client or httpx.Client(timeout=120.0)
        self._loaded: set[str] = set()

    @classmethod
    def is_available(cls) -> bool:
        return bool(os.environ.get("GROQ_API_KEY"))

    def load(self, model_ref: str) -> None:
        self._loaded.add(model_ref)

    def infer(self, request: InferRequest) -> InferResult:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        self.load(request.model_ref)
        resp = self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": _upstream_model(request.model_ref),
                "messages": [{"role": "user", "content": request.prompt}],
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "seed": request.seed,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return InferResult(
            text=data["choices"][0]["message"]["content"],
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=max(1, int(usage.get("completion_tokens", 1))),
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=False,
        )

    def loaded_models(self) -> list[str]:
        return sorted(self._loaded)
