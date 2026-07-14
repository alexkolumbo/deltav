"""Vendor-agnostic compute abstraction.

A node exposes exactly one ComputeBackend. The interface is deliberately
tiny — load / infer / unload — so that adding a new accelerator (AMD via
ROCm, Groq LPUs, a custom ASIC) means implementing one class, not
touching the chain or the router.

Determinism contract: given the same (model_ref, prompt, max_tokens,
seed), a backend SHOULD produce the same output. Spot-check verification
on the chain relies on it; backends that cannot guarantee it (GPU
non-determinism) are verified with fuzzy token-count checks instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class DeviceInfo:
    vendor: str  # nvidia | amd | intel | apple | groq | asic | cpu
    name: str
    vram_mb: int
    backend: str = ""

    def to_dict(self) -> dict:
        return {"vendor": self.vendor, "name": self.name, "vram_mb": self.vram_mb, "backend": self.backend}


@dataclass
class InferRequest:
    prompt: str
    model_ref: str  # HF repo id (optionally "repo::filename.gguf")
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0


@dataclass
class InferResult:
    text: str
    tokens_in: int
    tokens_out: int
    seed: int
    model_ref: str
    backend: str
    deterministic: bool = True


class ComputeBackend(ABC):
    """One accelerator (or accelerator family) able to run LLM inference."""

    name: str = "abstract"
    # Hardware vendors this backend can drive.
    vendors: tuple[str, ...] = ()
    # Whether identical requests produce identical outputs (spot-check mode).
    deterministic: bool = True

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Can this backend run on the current machine right now?"""

    @abstractmethod
    def load(self, model_ref: str) -> None:
        """Ensure `model_ref` is ready to serve (download/compile/warm up)."""

    @abstractmethod
    def infer(self, request: InferRequest) -> InferResult:
        """Run one completion. Must honor request.seed for determinism."""

    def unload(self, model_ref: str | None = None) -> None:  # noqa: B027 - optional hook
        """Free memory; default no-op."""

    def loaded_models(self) -> list[str]:
        return []


# Ordered by preference: real GPU backends first, mock last.
BACKENDS: list[type[ComputeBackend]] = []


def register_backend(cls: type[ComputeBackend]) -> type[ComputeBackend]:
    BACKENDS.append(cls)
    return cls


def make_backend(name: str = "auto") -> ComputeBackend:
    """Instantiate a backend by name, or the best available one for "auto"."""
    # Imports here so optional heavy deps never load unless needed.
    from . import asic, groq, llamacpp, mock  # noqa: F401  (registration side effect)

    if name != "auto":
        for cls in BACKENDS:
            if cls.name == name:
                if not cls.is_available():
                    raise RuntimeError(f"backend {name!r} is not available on this machine")
                return cls()
        raise ValueError(f"unknown backend {name!r}; known: {[c.name for c in BACKENDS]}")

    for cls in BACKENDS:
        if cls.is_available():
            return cls()
    raise RuntimeError("no compute backend available")
