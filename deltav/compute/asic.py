"""Custom AI accelerator backend — skeleton.

Target: future in-house or third-party AI ASICs/NPUs that expose a
runtime SDK. The contract a new chip must satisfy to join the network:

  * is_available() — detect the device (driver present, SDK importable).
  * load(model_ref) — pull weights from HuggingFace and compile/quantize
    them into the chip's native format. Cache the compiled artifact.
  * infer(request)  — run generation honoring `seed`; if the chip cannot
    be bit-deterministic, set deterministic=False so the chain uses
    fuzzy spot checks.
  * Report DeviceInfo(vendor="asic", vram_mb=<on-chip/board memory>)
    from deltav.compute.detect so the router can do VRAM-fit planning.

Nothing else in the system needs to change — the chain and the router
only ever see the ComputeBackend interface.
"""
from __future__ import annotations

from .base import ComputeBackend, InferRequest, InferResult, register_backend


@register_backend
class AsicBackend(ComputeBackend):
    name = "asic"
    vendors = ("asic",)
    deterministic = True

    @classmethod
    def is_available(cls) -> bool:
        return False  # flip when a real device/SDK probe is implemented

    def load(self, model_ref: str) -> None:
        raise NotImplementedError("custom accelerator skeleton — see module docstring")

    def infer(self, request: InferRequest) -> InferResult:
        raise NotImplementedError("custom accelerator skeleton — see module docstring")
