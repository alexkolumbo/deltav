"""Groq LPU backend — skeleton.

Groq exposes its LPU clusters through an OpenAI-compatible HTTP API
rather than local drivers, so a "Groq node" on the Delta V network is a
machine that holds a Groq API key and relays jobs to LPU hardware.

To finish this backend:
  1. `pip install groq` and set GROQ_API_KEY.
  2. Implement `infer` via groq.Groq().chat.completions.create(...).
  3. Mark deterministic=False — LPU sampling is not bit-reproducible, so
     spot checks fall back to fuzzy verification (token counts + length).
"""
from __future__ import annotations

import os

from .base import ComputeBackend, InferRequest, InferResult, register_backend


@register_backend
class GroqBackend(ComputeBackend):
    name = "groq"
    vendors = ("groq",)
    deterministic = False

    @classmethod
    def is_available(cls) -> bool:
        # Requires the SDK and a key; disabled until the TODOs above are done.
        return False

    def load(self, model_ref: str) -> None:
        raise NotImplementedError("Groq backend skeleton — see module docstring")

    def infer(self, request: InferRequest) -> InferResult:
        raise NotImplementedError("Groq backend skeleton — see module docstring")


class _GroqEnv:
    """Helper: where the finished backend will read its credentials from."""

    @staticmethod
    def api_key() -> str | None:
        return os.environ.get("GROQ_API_KEY")
