"""Model catalog: which HuggingFace models fit which VRAM.

Ships with a curated set of GGUF instruct models (sizes are real
Q4_K_M/Q4 file sizes from HF), and can optionally refresh live from the
HuggingFace Hub when `huggingface_hub` is installed. The VRAM estimator
is intentionally conservative: file size + KV cache + runtime overhead.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# Runtime scratch/buffer overhead on top of weights, MB.
_RUNTIME_OVERHEAD_MB = 600
# Weights rarely map 1:1 — mmap slack, dequant buffers.
_WEIGHTS_FACTOR = 1.05


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str            # HF repo with GGUF files
    filename: str           # concrete quant file inside the repo
    family: str
    params_b: float         # billions of parameters
    quant: str
    file_mb: int            # size of the GGUF file
    quality: float          # rough capability score 0..1 for auto-routing
    ctx: int = 4096         # default SERVING context (planning may raise it)
    kind: str = "chat"      # "chat" | "embedding" | "image" (diffusion)
    vision: bool = False    # chat model that also accepts image input
    # Architecture facts for exact KV-cache math (0 = unknown -> heuristic).
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 128
    max_ctx: int = 4096     # the model's native context limit

    @property
    def ref(self) -> str:
        # API-relayed models (no concrete file) are referenced by repo_id alone.
        return f"{self.repo_id}::{self.filename}" if self.filename else self.repo_id

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ref"] = self.ref
        return d


# Bytes per KV element for llama.cpp cache types (q* need flash attention).
KV_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
# Scratch buffers grow mildly with context.
_PER_TOKEN_OVERHEAD_MB = 0.005


def kv_cache_mb(params_b: float, ctx: int) -> int:
    """Rough fp16 KV-cache estimate scaled by model size (fallback when
    the architecture is unknown)."""
    return int(params_b * ctx / 32)


def kv_bytes_per_token(spec: ModelSpec, kv_type: str = "f16") -> int:
    """Exact per-token KV size: 2 (K+V) x layers x kv_heads x head_dim."""
    if spec.n_layers <= 0 or spec.n_kv_heads <= 0:
        return 0
    return int(2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * KV_BYTES[kv_type])


def estimate_vram_mb(spec: ModelSpec, ctx: int | None = None, kv_type: str = "f16") -> int:
    ctx = ctx or spec.ctx
    per_token = kv_bytes_per_token(spec, kv_type)
    kv_mb = int(per_token * ctx / (1024 * 1024)) if per_token \
        else kv_cache_mb(spec.params_b, ctx)
    return (int(spec.file_mb * _WEIGHTS_FACTOR) + kv_mb
            + _RUNTIME_OVERHEAD_MB + int(ctx * _PER_TOKEN_OVERHEAD_MB))


# Real GGUF quants published on HuggingFace (sizes in MB, Q4_K_M unless
# noted) with architecture facts for exact KV planning.
CURATED_CATALOG: list[ModelSpec] = [
    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf",
              "qwen2.5", 0.5, "Q4_K_M", 491, 0.35,
              n_layers=24, n_kv_heads=2, head_dim=64, max_ctx=32768),
    ModelSpec("bartowski/Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
              "llama3", 1.2, "Q4_K_M", 808, 0.45,
              n_layers=16, n_kv_heads=8, head_dim=64, max_ctx=131072),
    ModelSpec("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
              "llama3", 3.2, "Q4_K_M", 2020, 0.60,
              n_layers=28, n_kv_heads=8, head_dim=128, max_ctx=131072),
    ModelSpec("bartowski/Mistral-7B-Instruct-v0.3-GGUF", "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
              "mistral", 7.2, "Q4_K_M", 4370, 0.72,
              n_layers=32, n_kv_heads=8, head_dim=128, max_ctx=32768),
    ModelSpec("Qwen/Qwen2.5-7B-Instruct-GGUF", "qwen2.5-7b-instruct-q4_k_m.gguf",
              "qwen2.5", 7.6, "Q4_K_M", 4680, 0.75,
              n_layers=28, n_kv_heads=4, head_dim=128, max_ctx=32768),
    # Same weights, single-file quant (the official repo splits q4_k_m).
    ModelSpec("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
              "qwen2.5", 7.6, "Q4_K_M", 4467, 0.75,
              n_layers=28, n_kv_heads=4, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
              "llama3", 8.0, "Q4_K_M", 4920, 0.78,
              n_layers=32, n_kv_heads=8, head_dim=128, max_ctx=131072),
    ModelSpec("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf",
              "qwen2.5", 14.7, "Q4_K_M", 8990, 0.84,
              n_layers=48, n_kv_heads=8, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/phi-4-GGUF", "phi-4-Q4_K_M.gguf",
              "phi", 14.7, "Q4_K_M", 9050, 0.85,
              n_layers=40, n_kv_heads=10, head_dim=128, max_ctx=16384),
    ModelSpec("Qwen/Qwen2.5-32B-Instruct-GGUF", "qwen2.5-32b-instruct-q4_k_m.gguf",
              "qwen2.5", 32.8, "Q4_K_M", 19900, 0.90,
              n_layers=64, n_kv_heads=8, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/Llama-3.3-70B-Instruct-GGUF", "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
              "llama3", 70.6, "Q4_K_M", 42500, 0.95,
              n_layers=80, n_kv_heads=8, head_dim=128, max_ctx=131072),
    # --- more curated instruct models across the size/quality spectrum ---
    ModelSpec("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
              "qwen2.5", 3.1, "Q4_K_M", 1930, 0.58,
              n_layers=36, n_kv_heads=2, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/gemma-2-2b-it-GGUF", "gemma-2-2b-it-Q4_K_M.gguf",
              "gemma2", 2.6, "Q4_K_M", 1710, 0.55,
              n_layers=26, n_kv_heads=4, head_dim=256, max_ctx=8192),
    ModelSpec("bartowski/Phi-3.5-mini-instruct-GGUF", "Phi-3.5-mini-instruct-Q4_K_M.gguf",
              "phi", 3.8, "Q4_K_M", 2390, 0.70,
              n_layers=32, n_kv_heads=32, head_dim=96, max_ctx=131072),
    ModelSpec("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
              "qwen2.5", 7.6, "Q4_K_M", 4680, 0.76,
              n_layers=28, n_kv_heads=4, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/gemma-2-9b-it-GGUF", "gemma-2-9b-it-Q4_K_M.gguf",
              "gemma2", 9.2, "Q4_K_M", 5760, 0.80,
              n_layers=42, n_kv_heads=8, head_dim=256, max_ctx=8192),
    ModelSpec("bartowski/Mistral-Nemo-Instruct-2407-GGUF", "Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
              "mistral", 12.2, "Q4_K_M", 7480, 0.82,
              n_layers=40, n_kv_heads=8, head_dim=128, max_ctx=131072),
    ModelSpec("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF", "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
              "deepseek-r1", 7.6, "Q4_K_M", 4680, 0.79,
              n_layers=28, n_kv_heads=4, head_dim=128, max_ctx=131072),
    ModelSpec("bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
              "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
              "deepseek-r1", 14.8, "Q4_K_M", 8990, 0.86,
              n_layers=48, n_kv_heads=8, head_dim=128, max_ctx=131072),
    ModelSpec("bartowski/Mistral-Small-24B-Instruct-2501-GGUF",
              "Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf",
              "mistral", 23.6, "Q4_K_M", 14300, 0.88,
              n_layers=40, n_kv_heads=8, head_dim=128, max_ctx=32768),
    ModelSpec("bartowski/gemma-2-27b-it-GGUF", "gemma-2-27b-it-Q4_K_M.gguf",
              "gemma2", 27.2, "Q4_K_M", 16600, 0.89,
              n_layers=46, n_kv_heads=16, head_dim=128, max_ctx=8192),
    # Reasoning + vision 9B (Qwen-based); thinks in reasoning_content.
    ModelSpec("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
              "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
              "qwythos", 8.7, "Q4_K_M", 5368, 0.80, vision=True,
              n_layers=48, n_kv_heads=8, head_dim=128, max_ctx=32768),
    # xAI's open Grok weights (big — for powerful nodes; still open, still
    # llama.cpp-servable once converted to GGUF).
    ModelSpec("xai-org/grok-1-GGUF", "grok-1-Q4_K_M.gguf",
              "grok", 314.0, "Q4_K_M", 170000, 0.92,
              n_layers=64, n_kv_heads=8, head_dim=128, max_ctx=8192),
    # --- multimodal (vision: image input) ---
    ModelSpec("bartowski/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
              "qwen2.5-vl", 8.3, "Q4_K_M", 5300, 0.77, vision=True,
              n_layers=28, n_kv_heads=4, head_dim=128, max_ctx=32768),
    # --- diffusion (text -> image), served via stable-diffusion.cpp (GGUF) ---
    ModelSpec("second-state/stable-diffusion-v1-5-GGUF", "stable-diffusion-v1-5-Q4_0.gguf",
              "sd", 0.98, "Q4_0", 1700, 0.60, kind="image", max_ctx=0),
    ModelSpec("second-state/FLUX.1-schnell-GGUF", "flux1-schnell-Q4_0.gguf",
              "flux", 12.0, "Q4_0", 6800, 0.80, kind="image", max_ctx=0),
    # Embedding models (routed by kind, never picked for chat).
    ModelSpec("nomic-ai/nomic-embed-text-v1.5-GGUF", "nomic-embed-text-v1.5.Q4_K_M.gguf",
              "nomic", 0.14, "Q4_K_M", 84, 0.60, ctx=2048, kind="embedding",
              n_layers=12, n_kv_heads=12, head_dim=64, max_ctx=8192),
    ModelSpec("CompendiumLabs/bge-small-en-v1.5-gguf", "bge-small-en-v1.5-q4_k_m.gguf",
              "bge", 0.03, "Q4_K_M", 24, 0.50, ctx=512, kind="embedding",
              n_layers=12, n_kv_heads=12, head_dim=32, max_ctx=512),
]


class Catalog:
    def __init__(self, specs: list[ModelSpec] | None = None):
        self.specs: list[ModelSpec] = list(specs if specs is not None else CURATED_CATALOG)

    def by_ref(self, ref: str) -> ModelSpec | None:
        for spec in self.specs:
            if spec.ref == ref or spec.repo_id == ref:
                return spec
        return None

    def fitting(self, vram_mb: int, ctx: int | None = None, kind: str = "chat") -> list[ModelSpec]:
        """Models of `kind` that fit the given VRAM, best quality first."""
        fits = [s for s in self.specs
                if s.kind == kind and estimate_vram_mb(s, ctx) <= vram_mb]
        return sorted(fits, key=lambda s: (-s.quality, s.file_mb))

    def best_for(self, vram_mb: int, ctx: int | None = None, kind: str = "chat") -> ModelSpec | None:
        fits = self.fitting(vram_mb, ctx, kind)
        return fits[0] if fits else None

    def embedding_specs(self) -> list[ModelSpec]:
        return sorted((s for s in self.specs if s.kind == "embedding"),
                      key=lambda s: -s.quality)

    def image_specs(self) -> list[ModelSpec]:
        return sorted((s for s in self.specs if s.kind == "image"),
                      key=lambda s: -s.quality)

    def refresh_from_hf(self, limit: int = 30) -> int:
        """Optionally enrich the catalog from the live HuggingFace Hub.

        Requires `huggingface_hub`; adds trending GGUF repos with file
        sizes read from repo metadata. Returns how many specs were added.
        Purely additive — the curated list keeps working offline.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            return 0
        api = HfApi()
        added = 0
        known = {s.repo_id for s in self.specs}
        for model in api.list_models(filter="gguf", sort="downloads", limit=limit):
            if model.id in known:
                continue
            try:
                files = api.list_repo_files(model.id)
            except Exception:
                continue
            q4 = [f for f in files if f.lower().endswith(".gguf") and "q4_k_m" in f.lower()]
            if not q4:
                continue
            try:
                info = api.get_paths_info(model.id, q4[:1])
                size_mb = int((info[0].size or 0) / (1024 * 1024))
            except Exception:
                continue
            if size_mb <= 0:
                continue
            params_b = max(0.5, size_mb / 615)  # ~0.615 GB per B params at Q4_K_M
            self.specs.append(ModelSpec(
                repo_id=model.id, filename=q4[0], family="hub",
                params_b=round(params_b, 1), quant="Q4_K_M", file_mb=size_mb,
                quality=min(0.9, 0.3 + params_b / 40),
            ))
            added += 1
        return added
