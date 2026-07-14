"""Hardware detection: figure out which accelerator this node has."""
from __future__ import annotations

import shutil
import subprocess

from .base import DeviceInfo


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _detect_nvidia() -> DeviceInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    first = out.splitlines()[0]
    try:
        name, mem = [p.strip() for p in first.rsplit(",", 1)]
        return DeviceInfo(vendor="nvidia", name=name, vram_mb=int(float(mem)))
    except ValueError:
        return None


def _detect_amd() -> DeviceInfo | None:
    if not shutil.which("rocm-smi"):
        return None
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if not out:
        return None
    name, vram_mb = "AMD GPU", 0
    for line in out.splitlines():
        low = line.lower()
        if "card series" in low or "card model" in low:
            name = line.split(",")[-1].strip() or name
        if "vram total" in low:
            try:
                vram_mb = int(int(line.split(",")[-1].strip()) / (1024 * 1024))
            except ValueError:
                pass
    if vram_mb == 0:
        vram_mb = 8192  # conservative default when rocm-smi hides memory info
    return DeviceInfo(vendor="amd", name=name, vram_mb=vram_mb)


def detect_device() -> DeviceInfo:
    """Best-effort detection; falls back to CPU with a nominal 8 GB budget."""
    for probe in (_detect_nvidia, _detect_amd):
        device = probe()
        if device is not None:
            return device
    return DeviceInfo(vendor="cpu", name="cpu", vram_mb=8192)
