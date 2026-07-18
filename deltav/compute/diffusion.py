"""Diffusers text-to-image backend (FLUX.1-schnell and friends).

Why this exists: the network already speaks images end to end — `/image` on the
node, `route_image` in the router, `/v1/images/generations` on the gateway — but
the only implementation was the mock. This is the first REAL image engine.

Default model is **FLUX.1-schnell**, which is Apache-2.0: unlike most strong
image models (FLUX.1-dev, Ideogram v4, …) it may be served commercially, which
matters for a paid inference network.

schnell is *timestep-distilled*, so it has hard requirements that make or break
output quality:
  * guidance_scale MUST be 0.0 (it has no CFG; >0 produces mush)
  * 1-4 sampling steps is the design point (20 is wasted compute, not "better")
  * the T5 prompt is capped at 256 tokens

VRAM: the full pipeline (12B transformer + T5-XXL + CLIP + VAE) is ~24 GB in
bf16, so it does NOT fit a 16 GB card outright — we enable diffusers' model CPU
offload automatically below ~24 GB, which trades a little speed for fitting.
Sampling is not bit-reproducible across different GPUs, so deterministic=False
(spot checks take the fuzzy path, exactly like the llama-server backend).
"""
from __future__ import annotations

import base64
import io
import os

from .base import ComputeBackend, ImageRequest, ImageResult, InferRequest, InferResult, register_backend

DEFAULT_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
# Below this much VRAM we offload components to CPU between stages.
_OFFLOAD_BELOW_GB = 24.0


@register_backend
class DiffusersBackend(ComputeBackend):
    name = "diffusers"
    vendors = ("nvidia", "amd", "intel", "apple", "cpu")
    # Diffusion sampling diverges across GPUs/kernels even with a fixed seed.
    deterministic = False
    # One pipeline is pinned in memory; the node only accepts its announced ref.
    dynamic_models = False
    supports_image_gen = True
    text_capable = False        # draw-only: `auto` must never select this

    def __init__(self, model_ref: str | None = None, offload: bool | None = None):
        self.model_ref = (model_ref or os.environ.get("DELTAV_IMAGE_MODEL")
                          or DEFAULT_IMAGE_MODEL)
        self._offload = offload
        self._pipe = None
        self._loaded_ref = ""

    # ------------------------------------------------------------- capability
    @classmethod
    def is_available(cls) -> bool:
        """Usable when torch + diffusers are installed. We do NOT require CUDA
        here: the node may be starting before the GPU is probed, and a missing
        accelerator surfaces as a slow-but-working run rather than a hard fail."""
        try:
            import diffusers  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False
        return True

    # ----------------------------------------------------------------- device
    @staticmethod
    def _device_and_dtype():
        import torch

        if torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps", torch.bfloat16
        return "cpu", torch.float32

    @staticmethod
    def _vram_gb() -> float:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    # ------------------------------------------------------------------ load
    def load(self, model_ref: str) -> None:
        """Build the pipeline once. Safe to call repeatedly."""
        ref = model_ref or self.model_ref
        if self._pipe is not None and self._loaded_ref == ref:
            return
        import torch
        from diffusers import DiffusionPipeline

        device, dtype = self._device_and_dtype()
        pipe = DiffusionPipeline.from_pretrained(ref, torch_dtype=dtype)
        offload = self._offload
        if offload is None:                       # decide from the actual card
            offload = device == "cuda" and self._vram_gb() < _OFFLOAD_BELOW_GB
        if offload and device == "cuda":
            # Keeps peak VRAM to roughly the largest single component instead of
            # the whole pipeline — this is what makes FLUX fit a 16 GB card.
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()             # cheap peak-memory win
        self._pipe = pipe
        self._loaded_ref = ref
        self.model_ref = ref
        del torch

    def loaded_models(self) -> list[str]:
        return [self._loaded_ref] if self._pipe is not None else []

    def unload(self, model_ref: str | None = None) -> None:
        self._pipe = None
        self._loaded_ref = ""
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------- text (n/a)
    def infer(self, request: InferRequest) -> InferResult:
        # Required by the ABC, but this engine only draws.
        raise NotImplementedError(
            "backend 'diffusers' generates images only — run a text model on a "
            "separate node/backend")

    # ------------------------------------------------------------------ draw
    @staticmethod
    def _tuning(model_ref: str, steps: int) -> tuple[int, float, int]:
        """schnell is distilled: 1-4 steps, NO guidance, 256-token prompts.
        Non-distilled siblings (dev/pro) want real CFG and more steps."""
        ref = (model_ref or "").lower()
        if "schnell" in ref:
            return max(1, min(steps or 4, 8)), 0.0, 256
        return max(1, min(steps or 20, 50)), 3.5, 512

    @staticmethod
    def _generator(seed: int):
        """Seed on CPU: with model-CPU-offload the pipeline's modules move
        around, and a CPU generator gives the same stream regardless of
        placement. (Isolated so the draw path is testable without torch.)"""
        import torch

        return torch.Generator("cpu").manual_seed(int(seed) or 0)

    def generate_image(self, request: ImageRequest) -> ImageResult:
        ref = request.model_ref or self.model_ref
        self.load(ref)
        steps, guidance, max_seq = self._tuning(ref, request.steps)
        generator = self._generator(request.seed)

        kwargs = dict(
            prompt=request.prompt,
            width=int(request.width) or 512,
            height=int(request.height) or 512,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        )
        # Only FLUX-family pipelines take max_sequence_length.
        if "flux" in ref.lower():
            kwargs["max_sequence_length"] = max_seq

        out = self._pipe(**kwargs)
        images = []
        for img in out.images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            images.append(base64.b64encode(buf.getvalue()).decode("ascii"))

        return ImageResult(
            images=images,
            model_ref=ref,
            backend=self.name,
            seed=int(request.seed) or 0,
            deterministic=False,
        )
