"""Analyze an arbitrary HuggingFace GGUF model for a given machine.

Two paths the wizard offers:
  * forced  — use the pasted repo as-is (operator knows what they're doing);
  * analyze — fetch the GGUF's size from HF, estimate VRAM need, and advise
              whether it fits this hardware and at what context.

Kept dependency-light: uses huggingface_hub if present, else a plain HTTP
HEAD for the file size. Architecture is unknown for a pasted repo, so the
VRAM estimate uses the size-based KV heuristic (conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

from ..router.catalog import ModelSpec, estimate_vram_mb
from ..router.planner import max_context_for

# Q4_K_M weights are ~0.62 GB per billion params; invert to estimate params.
_MB_PER_B_Q4 = 615


@dataclass
class ModelAnalysis:
    ref: str
    repo: str
    filename: str
    file_mb: int
    est_params_b: float
    fits: bool
    verdict: str            # "great" | "tight" | "cpu_offload" | "too_big" | "unknown"
    max_context: int
    est_vram_mb: int
    spec: ModelSpec | None


def parse_ref(ref: str) -> tuple[str, str]:
    """Split 'org/repo::file.gguf' or 'org/repo' -> (repo, filename)."""
    repo, _, filename = ref.partition("::")
    return repo.strip().strip("/"), filename.strip()


def _hf_file_size_mb(repo: str, filename: str, client: httpx.Client) -> int | None:
    """Best-effort file size in MB via the hub API or an HTTP HEAD."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().get_paths_info(repo, [filename])
        if info and getattr(info[0], "size", None):
            return int(info[0].size / (1024 * 1024))
    except Exception:
        pass
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
    try:
        resp = client.head(url, follow_redirects=True, timeout=20.0)
        if resp.status_code < 400 and "content-length" in resp.headers:
            return int(int(resp.headers["content-length"]) / (1024 * 1024))
    except httpx.HTTPError:
        pass
    return None


def _pick_gguf_filename(repo: str, client: httpx.Client) -> str | None:
    """If no filename was given, try to find a Q4_K_M GGUF in the repo."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo)
    except Exception:
        return None
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    if not ggufs:
        return None
    for f in ggufs:
        if "q4_k_m" in f.lower():
            return f
    return ggufs[0]


def analyze_model(ref: str, vram_mb: int, client: httpx.Client | None = None) -> ModelAnalysis:
    owns = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        repo, filename = parse_ref(ref)
        if not filename:
            filename = _pick_gguf_filename(repo, client) or ""
        if not filename:
            return ModelAnalysis(ref, repo, "", 0, 0.0, False, "unknown", 0, 0, None)

        file_mb = _hf_file_size_mb(repo, filename, client)
        if not file_mb:
            return ModelAnalysis(ref, repo, filename, 0, 0.0, False, "unknown", 0, 0, None)

        est_params = round(file_mb / _MB_PER_B_Q4, 1)
        spec = ModelSpec(repo_id=repo, filename=filename, family="custom",
                         params_b=est_params, quant="?", file_mb=file_mb,
                         quality=0.5, max_ctx=32768)  # unknown arch -> KV heuristic
        budget = int(vram_mb * 0.95)
        min_ctx_vram = estimate_vram_mb(spec, ctx=4096)
        max_ctx = max_context_for(spec, vram_mb)
        weights_only = int(file_mb * 1.05)

        if max_ctx >= 8192:
            verdict, fits = "great", True
        elif max_ctx >= 2048:
            verdict, fits = "tight", True
        elif weights_only <= budget:
            verdict, fits = "cpu_offload", False  # weights fit but no room for context
        else:
            verdict, fits = "too_big", False

        return ModelAnalysis(
            ref=f"{repo}::{filename}", repo=repo, filename=filename, file_mb=file_mb,
            est_params_b=est_params, fits=fits, verdict=verdict,
            max_context=max_ctx, est_vram_mb=min_ctx_vram, spec=spec)
    finally:
        if owns:
            client.close()
