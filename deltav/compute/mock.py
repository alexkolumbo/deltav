"""Deterministic mock backend for simulation and tests.

Output is a pure function of (model_ref, prompt, max_tokens, seed), so
spot-check re-execution on another node genuinely verifies receipts —
the whole trust pipeline can be exercised without a GPU.
"""
from __future__ import annotations

import hashlib
import re

from .base import ComputeBackend, InferRequest, InferResult, register_backend

# Scripted-reply marker: a prompt containing [[reply]]...[[/reply]] gets that
# text back verbatim (still deterministic). Lets tests and demos drive
# tool-calling/agent flows through the full network without a real LLM.
_REPLY_RE = re.compile(r"\[\[reply\]\](.*?)\[\[/reply\]\]", re.S)

_WORDS = [
    "delta", "vector", "orbit", "thrust", "ion", "burn", "apogee", "node",
    "stake", "block", "tensor", "quant", "layer", "token", "route", "swarm",
]


@register_backend
class MockBackend(ComputeBackend):
    name = "mock"
    vendors = ("cpu",)
    deterministic = True

    def __init__(self) -> None:
        self._loaded: set[str] = set()

    @classmethod
    def is_available(cls) -> bool:
        return True

    def load(self, model_ref: str) -> None:
        self._loaded.add(model_ref)

    def infer(self, request: InferRequest) -> InferResult:
        self.load(request.model_ref)
        scripted = _REPLY_RE.findall(request.prompt)
        if scripted:
            # the N-th model call in a growing agent prompt uses the N-th script
            n_turns = request.prompt.count("tool (")
            text = scripted[min(n_turns, len(scripted) - 1)].strip()
            return InferResult(
                text=text,
                tokens_in=max(1, len(request.prompt.split())),
                tokens_out=max(1, len(text.split())),
                seed=request.seed,
                model_ref=request.model_ref,
                backend=self.name,
            )
        seed_material = f"{request.model_ref}|{request.prompt}|{request.max_tokens}|{request.seed}"
        digest = hashlib.sha256(seed_material.encode()).digest()
        n_out = min(request.max_tokens, 24)
        words = []
        stream = digest
        while len(words) < n_out:
            for byte in stream:
                words.append(_WORDS[byte % len(_WORDS)])
                if len(words) >= n_out:
                    break
            stream = hashlib.sha256(stream).digest()
        return InferResult(
            text="[" + request.model_ref + "] " + " ".join(words),
            tokens_in=max(1, len(request.prompt.split())),
            tokens_out=n_out,
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
        )

    def infer_stream(self, request: InferRequest):
        result = self.infer(request)
        for piece in re.findall(r"\S+\s*", result.text) or [result.text]:
            yield piece
        yield result

    def loaded_models(self) -> list[str]:
        return sorted(self._loaded)
