"""Model registry: discover from HuggingFace, persist, rank.

Kept dependency-light — uses httpx against the public HF API (no key). A
discovered entry carries enough to plan VRAM: file size, quant, estimated
params, and architecture from config.json when available (else the
size-based KV heuristic applies, like any custom model).
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from ..router.catalog import Catalog, ModelSpec, estimate_vram_mb
from ..router.planner import max_context_for

DEFAULT_PATH = Path.home() / ".deltav" / "registry.json"
HF_API = "https://huggingface.co/api/models"
_MB_PER_B_Q4 = 615
_QUANT_RE = re.compile(r"(Q\d(?:_K)?(?:_[MSL])?|BF16|F16|F32|Q8_0|IQ\d\w*)", re.I)


@dataclass
class DiscoveredModel:
    repo_id: str
    filename: str
    file_mb: int
    quant: str
    params_b: float
    family: str = "hub"
    kind: str = "chat"
    downloads: int = 0
    likes: int = 0
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 128
    max_ctx: int = 32768
    vision: bool = False
    discovered_at: float = 0.0

    @property
    def ref(self) -> str:
        return f"{self.repo_id}::{self.filename}"

    def to_spec(self) -> ModelSpec:
        return ModelSpec(
            repo_id=self.repo_id, filename=self.filename, family=self.family,
            params_b=self.params_b, quant=self.quant, file_mb=self.file_mb,
            quality=min(0.9, 0.3 + self.params_b / 40), kind=self.kind,
            vision=self.vision, n_layers=self.n_layers, n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim, max_ctx=self.max_ctx or 32768)


def _quant_of(filename: str) -> str:
    m = _QUANT_RE.search(filename)
    return m.group(1).upper() if m else "?"


class ModelRegistry:
    def __init__(self, path: str | Path | None = None, catalog: Catalog | None = None):
        self.path = Path(path) if path else DEFAULT_PATH
        self.catalog = catalog or Catalog()
        self.discovered: dict[str, DiscoveredModel] = {}
        if self.path.exists():
            for d in json.loads(self.path.read_text(encoding="utf-8")).get("models", []):
                m = DiscoveredModel(**d)
                self.discovered[m.ref] = m

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {"updated": 0.0, "models": [asdict(m) for m in self.discovered.values()]},
            indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ HF bot
    def sync_from_hf(self, limit: int = 40, prefer_quant: str = "Q4_K_M",
                     client: httpx.Client | None = None, now: float = 0.0) -> int:
        """Pull trending GGUF repos and register a fitting quant from each.
        Returns how many NEW models were added. `now` timestamps entries
        (pass a real time; the chain-free code stays deterministic)."""
        owns = client is None
        client = client or httpx.Client(timeout=30.0)
        added = 0
        known = set(self.discovered) | {s.ref for s in self.catalog.specs}
        try:
            resp = client.get(HF_API, params={"filter": "gguf", "sort": "downloads",
                                              "direction": -1, "limit": limit}, timeout=30.0)
            resp.raise_for_status()
            for entry in resp.json():
                repo = entry.get("id") or entry.get("modelId")
                if not repo:
                    continue
                model = self._discover_one(repo, prefer_quant, client, now,
                                           downloads=int(entry.get("downloads", 0)),
                                           likes=int(entry.get("likes", 0)))
                if model and model.ref not in known:
                    self.discovered[model.ref] = model
                    known.add(model.ref)
                    added += 1
        except httpx.HTTPError:
            pass
        finally:
            if owns:
                client.close()
        if added:
            self.save()
        return added

    def add_repo(self, repo: str, prefer_quant: str = "Q4_K_M",
                 client: httpx.Client | None = None, now: float = 0.0) -> DiscoveredModel | None:
        """Explicitly add one repo (a user/bot pasting a HF link)."""
        owns = client is None
        client = client or httpx.Client(timeout=30.0)
        try:
            m = self._discover_one(repo, prefer_quant, client, now)
            if m:
                self.discovered[m.ref] = m
                self.save()
            return m
        finally:
            if owns:
                client.close()

    def _discover_one(self, repo: str, prefer_quant: str, client: httpx.Client,
                      now: float, downloads: int = 0, likes: int = 0) -> DiscoveredModel | None:
        try:
            tree = client.get(f"{HF_API}/{repo}/tree/main", timeout=30.0).json()
        except httpx.HTTPError:
            return None
        ggufs = [(e["path"], int(e.get("size", 0))) for e in tree
                 if isinstance(e, dict) and str(e.get("path", "")).lower().endswith(".gguf")]
        if not ggufs:
            return None
        # prefer the requested quant, avoid MTP/mmproj/split shards
        def score(item):
            name = item[0].lower()
            s = 0
            if prefer_quant.lower() in name:
                s += 10
            if "mtp" in name or "mmproj" in name or "-of-" in name:
                s -= 20
            return s
        path, size = max(ggufs, key=score)
        vision = any("mmproj" in p.lower() for p, _ in ggufs)
        file_mb = int(size / (1024 * 1024)) if size else 0
        if file_mb <= 0:
            return None
        arch = self._read_arch(repo, client)
        params = round(file_mb / _MB_PER_B_Q4, 1)
        return DiscoveredModel(
            repo_id=repo, filename=path, file_mb=file_mb, quant=_quant_of(path),
            params_b=params, family=arch.get("family", "hub"),
            kind="image" if "stable-diffusion" in repo.lower() or "flux" in repo.lower() else "chat",
            downloads=downloads, likes=likes, vision=vision,
            n_layers=arch.get("n_layers", 0), n_kv_heads=arch.get("n_kv_heads", 0),
            head_dim=arch.get("head_dim", 128), max_ctx=arch.get("max_ctx", 32768),
            discovered_at=now)

    @staticmethod
    def _read_arch(repo: str, client: httpx.Client) -> dict:
        """Best-effort architecture from the base repo's config.json."""
        base = repo.replace("-GGUF", "").replace("-gguf", "")
        try:
            cfg = client.get(f"https://huggingface.co/{base}/resolve/main/config.json",
                             timeout=15.0)
            if cfg.status_code >= 400:
                return {}
            c = cfg.json()
        except (httpx.HTTPError, ValueError):
            return {}
        heads = c.get("num_attention_heads", 0)
        hidden = c.get("hidden_size", 0)
        return {
            "family": (c.get("model_type") or "hub"),
            "n_layers": int(c.get("num_hidden_layers", 0)),
            "n_kv_heads": int(c.get("num_key_value_heads", heads) or heads),
            "head_dim": int(hidden / heads) if heads else 128,
            "max_ctx": int(c.get("max_position_embeddings", 32768)),
        }

    # -------------------------------------------------------------- query
    def all_specs(self, kind: str = "chat", served: set[str] | None = None) -> list[ModelSpec]:
        specs = {s.ref: s for s in self.catalog.specs if s.kind == kind}
        for m in self.discovered.values():
            if m.kind == kind and m.ref not in specs:
                specs[m.ref] = m.to_spec()
        return list(specs.values())

    @staticmethod
    def _quant_bonus(quant: str) -> float:
        """Higher-bit quants keep more of the model's quality — prefer them
        among candidates that all fit (fit is already filtered by VRAM)."""
        q = (quant or "").lower()
        for prefix, bonus in (("f16", 0.04), ("bf16", 0.04), ("q8", 0.04),
                              ("q6", 0.03), ("q5", 0.02), ("q4_k", 0.01),
                              ("iq4", 0.0), ("q4", 0.0),
                              ("q3", -0.03), ("iq3", -0.03),
                              ("q2", -0.06), ("iq2", -0.06)):
            if q.startswith(prefix):
                return bonus
        return 0.0

    @staticmethod
    def score(row: dict) -> float:
        """Composite ranking score, in quality units (~0..1):

        capability (params/family quality) + usable context (log-scaled — a
        32k model beats an equal one stuck at 3k) + modality (vision serves
        more request types) + quant fidelity + already-served-on-network
        (warm model, immediate demand) + a small popularity nudge.
        """
        ctx_term = 0.03 * math.log2(max(1024, row["max_context"]) / 4096.0)
        ctx_term = max(-0.06, min(0.12, ctx_term))
        dl_term = min(0.03, 0.01 * math.log10(1 + row.get("downloads", 0)))
        return (row["quality"]
                + ctx_term
                + (0.05 if row.get("vision") else 0.0)
                + ModelRegistry._quant_bonus(row.get("quant", ""))
                + (0.15 if row.get("served") else 0.0)
                + dl_term)

    def rank(self, vram_mb: int, kind: str = "chat", top: int = 20,
             served: set[str] | None = None) -> list[dict]:
        """Rank models that fit `vram_mb` by the composite score (context,
        modality, params, quant, served, popularity — see `score`)."""
        served = served or set()
        rows = []
        for spec in self.all_specs(kind):
            ctx = max_context_for(spec, vram_mb)
            if ctx < 2048 and spec.file_mb and estimate_vram_mb(spec) > vram_mb:
                continue  # doesn't fit at all
            is_served = spec.ref in served or spec.repo_id in served
            disc = self.discovered.get(spec.ref)
            row = {
                "ref": spec.ref, "repo": spec.repo_id, "family": spec.family,
                "params_b": spec.params_b, "quant": spec.quant, "vision": spec.vision,
                "quality": spec.quality, "file_mb": spec.file_mb, "max_context": ctx,
                "served": is_served,
                "downloads": disc.downloads if disc else 0,
                "source": "hub" if disc else "catalog",
            }
            row["score"] = round(self.score(row), 4)
            rows.append(row)
        rows.sort(key=lambda r: r["score"], reverse=True)
        # The catalog and the HF-discovered DB often hold the same model from
        # different uploaders (Qwen/… vs bartowski/…) — showing both as #7 and
        # #8 only confuses the wizard user. Keep the best-ranked copy.
        seen: set[tuple] = set()
        unique = []
        for r in rows:
            key = (r["repo"].split("/")[-1].lower(), (r["quant"] or "").lower())
            if key in seen:
                continue
            seen.add(key)
            unique.append(r)
        return unique[:top]
