"""Hardware detection: figure out which accelerator this node has."""
from __future__ import annotations

import shutil
import subprocess
import sys

from .base import DeviceInfo


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def parse_nvidia_smi(output: str) -> DeviceInfo | None:
    """Parse `nvidia-smi --query-gpu=name,memory.total` CSV — all GPUs."""
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        try:
            name, mem = [p.strip() for p in line.rsplit(",", 1)]
            gpus.append({"name": name, "vram_mb": int(float(mem))})
        except ValueError:
            continue
    if not gpus:
        return None
    label = gpus[0]["name"] + (f" x{len(gpus)}" if len(gpus) > 1 else "")
    return DeviceInfo(
        vendor="nvidia",
        name=label,
        vram_mb=sum(g["vram_mb"] for g in gpus),
        gpu_count=len(gpus),
        gpus=gpus,
    )


def _detect_nvidia() -> DeviceInfo | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
    if not out:
        return None
    return parse_nvidia_smi(out)


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


# GPU class GUID in the Windows registry; each 00NN subkey is an adapter
# with DriverDesc + HardwareInformation.qwMemorySize (true VRAM, unlike
# Win32_VideoController.AdapterRAM which caps at 4 GB).
_WIN_GPU_PS = (
    r"Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class"
    r"\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' -ErrorAction SilentlyContinue | "
    r"ForEach-Object { if ($_.'HardwareInformation.qwMemorySize') { "
    r"'{0}|{1}' -f $_.DriverDesc, $_.'HardwareInformation.qwMemorySize' } }"
)


def parse_windows_gpus(output: str) -> DeviceInfo | None:
    """Parse 'name|vram_bytes' lines; the biggest adapter is the compute GPU."""
    gpus = []
    for line in output.splitlines():
        name, _, raw = line.strip().rpartition("|")
        try:
            vram_mb = int(int(raw) / (1024 * 1024))
        except ValueError:
            continue
        if name and vram_mb >= 512:  # skip iGPU stubs reporting tiny carve-outs
            gpus.append({"name": name, "vram_mb": vram_mb})
    if not gpus:
        return None
    gpus.sort(key=lambda g: -g["vram_mb"])
    best = gpus[0]
    lowered = best["name"].lower()
    vendor = ("amd" if "amd" in lowered or "radeon" in lowered
              else "nvidia" if "nvidia" in lowered or "geforce" in lowered
              else "intel" if "intel" in lowered or "arc" in lowered
              else "gpu")
    return DeviceInfo(vendor=vendor, name=best["name"], vram_mb=best["vram_mb"],
                      gpu_count=1, gpus=[best])


def _detect_windows_gpu() -> DeviceInfo | None:
    if sys.platform != "win32":
        return None
    out = _run(["powershell", "-NoProfile", "-Command", _WIN_GPU_PS])
    if not out:
        return None
    return parse_windows_gpus(out)


def detect_device() -> DeviceInfo:
    """Best-effort detection; falls back to CPU with a nominal 8 GB budget."""
    for probe in (_detect_nvidia, _detect_amd, _detect_windows_gpu):
        device = probe()
        if device is not None:
            return device
    return DeviceInfo(vendor="cpu", name="cpu", vram_mb=8192)
