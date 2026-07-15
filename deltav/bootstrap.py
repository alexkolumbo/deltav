"""One-command node bootstrap: detect hardware, pick a model, join the chain.

Used by `deltav join`. Everything here is also usable programmatically.
"""
from __future__ import annotations

import httpx

from .compute.base import DeviceInfo
from .config import Genesis
from .router.catalog import Catalog, ModelSpec, estimate_vram_mb


def pick_model_for_device(device: DeviceInfo, catalog: Catalog | None = None) -> ModelSpec | None:
    """The best-quality catalog model that fits this device's memory."""
    catalog = catalog or Catalog()
    return catalog.best_for(device.vram_mb)


def download_model(spec: ModelSpec) -> str | None:
    """Pre-download the GGUF so the first inference doesn't pay for it.

    Needs `huggingface_hub`; returns the local path, or None when the hub
    client isn't installed (llama.cpp will then lazy-download on load).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    return hf_hub_download(repo_id=spec.repo_id, filename=spec.filename)


async def fetch_genesis(seed_url: str, client: httpx.AsyncClient | None = None) -> Genesis:
    """Pull the network's genesis from a seed node."""
    owns = client is None
    client = client or httpx.AsyncClient()
    try:
        resp = await client.get(f"{seed_url.rstrip('/')}/genesis", timeout=10.0)
        resp.raise_for_status()
        return Genesis.from_dict(resp.json())
    finally:
        if owns:
            await client.aclose()


def describe_plan(device: DeviceInfo, spec: ModelSpec | None, backend_name: str) -> str:
    lines = [
        f"hardware : {device.vendor} / {device.name} ({device.vram_mb} MB)",
        f"backend  : {backend_name}",
    ]
    if spec is None:
        lines.append("model    : none fits this device — node will relay only")
    else:
        lines.append(
            f"model    : {spec.ref}\n"
            f"           {spec.params_b}B {spec.quant}, ~{estimate_vram_mb(spec)} MB VRAM needed"
        )
    return "\n".join(lines)
