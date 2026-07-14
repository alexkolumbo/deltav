"""llama.cpp backend — the real workhorse.

llama.cpp is itself vendor-agnostic: the same GGUF file runs on NVIDIA
(CUDA), AMD (ROCm/HIP or Vulkan), Intel (SYCL/Vulkan), Apple (Metal) and
plain CPU. That makes it the natural first backend for a heterogeneous
volunteer network: install `llama-cpp-python` built for your hardware
and the node just works.

model_ref format: "org/repo" or "org/repo::filename.gguf".
"""
from __future__ import annotations

import importlib.util

from .base import ComputeBackend, InferRequest, InferResult, register_backend


def _split_ref(model_ref: str) -> tuple[str, str | None]:
    if "::" in model_ref:
        repo, filename = model_ref.split("::", 1)
        return repo, filename
    return model_ref, None


@register_backend
class LlamaCppBackend(ComputeBackend):
    name = "llamacpp"
    vendors = ("nvidia", "amd", "intel", "apple", "cpu")
    deterministic = True  # with temperature=0 and a fixed seed on one machine class

    def __init__(self, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._models: dict[str, object] = {}

    @classmethod
    def is_available(cls) -> bool:
        return importlib.util.find_spec("llama_cpp") is not None

    def load(self, model_ref: str) -> None:
        if model_ref in self._models:
            return
        from llama_cpp import Llama

        repo, filename = _split_ref(model_ref)
        if repo.endswith(".gguf"):
            model = Llama(model_path=repo, n_ctx=self.n_ctx,
                          n_gpu_layers=self.n_gpu_layers, verbose=False)
        else:
            model = Llama.from_pretrained(
                repo_id=repo,
                filename=filename or "*Q4_K_M.gguf",
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
        self._models[model_ref] = model

    def infer(self, request: InferRequest) -> InferResult:
        self.load(request.model_ref)
        model = self._models[request.model_ref]
        out = model.create_completion(
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
        )
        usage = out.get("usage", {})
        return InferResult(
            text=out["choices"][0]["text"],
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            seed=request.seed,
            model_ref=request.model_ref,
            backend=self.name,
            deterministic=request.temperature == 0.0,
        )

    def unload(self, model_ref: str | None = None) -> None:
        if model_ref is None:
            self._models.clear()
        else:
            self._models.pop(model_ref, None)

    def loaded_models(self) -> list[str]:
        return list(self._models)
