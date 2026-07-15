"""Hardware-aware model planner.

Given a VRAM budget, enumerate the (model x KV-cache type x context)
space with exact per-architecture KV math and rank by objective:

  * max_context — the longest context this hardware can hold at all;
  * max_quality — the smartest model that fits with a usable context;
  * balanced    — quality first, with a bonus for extra context.

This is what `deltav plan`, the node's /plan and the gateway's /v1/plan
serve: the network itself tells any hardware what it should run. The
model space is deliberately open — engines like colibri (disk-streamed
MoE experts) can join later as ComputeBackends with their own specs;
the planner only needs memory math, not a specific runtime.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from .catalog import Catalog, KV_BYTES, ModelSpec, estimate_vram_mb

# Leave headroom for the OS/compositor and allocator fragmentation.
SAFETY = 0.95
MIN_USEFUL_CTX = 2048
CTX_GRANULARITY = 1024


@dataclass
class PlanOption:
    ref: str
    family: str
    params_b: float
    quant: str
    quality: float
    kv_type: str
    max_context: int
    est_vram_mb: int
    native_ctx: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def max_context_for(spec: ModelSpec, vram_mb: int, kv_type: str = "f16") -> int:
    """Largest context (multiple of 1K, capped at native max) that fits."""
    budget = int(vram_mb * SAFETY)
    ctx = spec.max_ctx or spec.ctx
    while ctx >= MIN_USEFUL_CTX:
        if estimate_vram_mb(spec, ctx, kv_type) <= budget:
            return ctx
        # linear KV growth -> big steps down are fine
        ctx = (ctx - 1) // CTX_GRANULARITY * CTX_GRANULARITY \
            if ctx <= 16384 else ctx // 2
    return 0


def _score(option: PlanOption, objective: str) -> tuple:
    if objective == "max_context":
        return (option.max_context, option.quality)
    if objective == "max_quality":
        return (option.quality, option.max_context)
    # balanced: quality dominates, longer context breaks the tie upward
    bonus = 0.7 + 0.3 * math.log2(max(option.max_context, 2048)) / 17.0
    return (option.quality * bonus, option.max_context)


def plan(
    vram_mb: int,
    objective: str = "balanced",
    catalog: Catalog | None = None,
    kv_types: tuple[str, ...] = ("f16", "q8_0", "q4_0"),
    top: int = 10,
) -> list[PlanOption]:
    catalog = catalog or Catalog()
    options: list[PlanOption] = []
    for spec in catalog.specs:
        if spec.kind != "chat":
            continue
        for kv_type in kv_types:
            if kv_type not in KV_BYTES:
                continue
            ctx = max_context_for(spec, vram_mb, kv_type)
            if ctx < MIN_USEFUL_CTX:
                continue
            notes = []
            if kv_type != "f16":
                notes.append("needs flash attention (-fa) for quantized KV cache")
            if ctx >= spec.max_ctx:
                notes.append("reaches the model's native context limit")
            options.append(PlanOption(
                ref=spec.ref,
                family=spec.family,
                params_b=spec.params_b,
                quant=spec.quant,
                quality=spec.quality,
                kv_type=kv_type,
                max_context=ctx,
                est_vram_mb=estimate_vram_mb(spec, ctx, kv_type),
                native_ctx=spec.max_ctx,
                notes=notes,
            ))

    options.sort(key=lambda o: _score(o, objective), reverse=True)
    # one line per (model, kv_type) is noisy — keep the best kv per model,
    # but let genuinely different tradeoffs through
    seen: dict[str, int] = {}
    deduped: list[PlanOption] = []
    for option in options:
        if seen.get(option.ref, 0) >= 2:
            continue
        seen[option.ref] = seen.get(option.ref, 0) + 1
        deduped.append(option)
    return deduped[:top]


def launch_hint(option: PlanOption, port: int = 8085) -> str:
    """The llama-server command that realizes a plan option."""
    parts = [
        f"llama-server -m <{option.ref.split('::')[-1] or 'model.gguf'}>",
        f"--port {port} -ngl 99 -c {option.max_context}",
    ]
    if option.kv_type != "f16":
        parts.append(f"-fa on --cache-type-k {option.kv_type} --cache-type-v {option.kv_type}")
    return " ".join(parts)
