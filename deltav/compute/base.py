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
from typing import Iterator, Union


@dataclass
class DeviceInfo:
    vendor: str  # nvidia | amd | intel | apple | groq | asic | cpu
    name: str
    # Total memory budget for model planning. For multi-GPU boxes this is
    # the SUM across GPUs — llama.cpp splits layers across them.
    vram_mb: int
    backend: str = ""
    gpu_count: int = 1
    gpus: list = field(default_factory=list)  # [{"name", "vram_mb"}, ...]

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "vram_mb": self.vram_mb,
            "backend": self.backend,
            "gpu_count": self.gpu_count,
            "gpus": self.gpus,
        }


@dataclass
class InferRequest:
    prompt: str
    model_ref: str  # HF repo id (optionally "repo::filename.gguf")
    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 0
    # Multimodal input: image references (data URIs or URLs) for vision
    # models. Empty for plain text. Backends that ignore it stay text-only.
    images: list = field(default_factory=list)


@dataclass
class ImageRequest:
    """A text-to-image (diffusion) job."""
    prompt: str
    model_ref: str
    width: int = 512
    height: int = 512
    steps: int = 20
    seed: int = 0


@dataclass
class ImageResult:
    images: list  # base64 PNG strings (or data descriptors)
    model_ref: str
    backend: str
    seed: int
    deterministic: bool = True


@dataclass
class InferResult:
    text: str
    tokens_in: int
    tokens_out: int
    seed: int
    model_ref: str
    backend: str
    deterministic: bool = True


@dataclass
class EmbedRequest:
    texts: list[str]
    model_ref: str


@dataclass
class EmbedResult:
    vectors: list[list[float]]
    tokens: int
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
    # Whether the backend can load arbitrary model refs on demand.
    # False = it serves a fixed, pre-loaded model set (e.g. llama-server):
    # the node then only accepts jobs for its announced models, and the
    # router never cold-start-routes other models to it.
    dynamic_models: bool = True

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

    def infer_stream(self, request: InferRequest) -> Iterator[Union[str, InferResult]]:
        """Yield text pieces as they are generated, then the final InferResult.

        The final result's `text` MUST equal the concatenated pieces — the
        on-chain receipt hashes the full output. Default: one piece.
        """
        result = self.infer(request)
        if result.text:
            yield result.text
        yield result

    # Extra job types this backend can serve (defaults off).
    supports_embeddings: bool = False
    supports_vision: bool = False       # image input to a chat model
    supports_image_gen: bool = False    # text -> image (diffusion)

    def embed(self, request: EmbedRequest) -> EmbedResult:
        """Embed a batch of texts in one pass (native batching)."""
        raise NotImplementedError(f"backend {self.name} does not support embeddings")

    def generate_image(self, request: "ImageRequest") -> "ImageResult":
        """Text-to-image generation (diffusion). Implement for engines like
        stable-diffusion.cpp; report supports_image_gen=True."""
        raise NotImplementedError(f"backend {self.name} does not generate images")

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
    # Order = "auto" preference: local GPU first (in-process, then a local
    # llama-server), API relays after, mock last.
    from . import llamacpp, llamaserver, groq, asic, mock  # noqa: F401  (registration side effect)

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
