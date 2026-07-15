"""Pick the right prebuilt llama.cpp binary for this machine.

ggml-org ships `llama-server` binaries for every OS and backend. We
prefer a GPU build (Vulkan runs on AMD/NVIDIA/Intel with no drivers to
compile; Metal on Apple Silicon) and fall back to CPU. Kept pure and
data-driven so the choice is unit-testable without hitting the network.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LlamaAsset:
    filename: str
    backend: str   # "vulkan" | "metal" | "cpu"
    reason: str    # plain-language explanation for the wizard


# Ordered preference per platform key. First asset whose substring matches
# a release filename wins. Platform key = (os, is_apple_silicon).
_PREFERENCE = {
    "windows": [
        ("bin-win-vulkan-x64", "vulkan",
         "использует видеокарту (AMD/NVIDIA/Intel) через Vulkan — быстро, без драйверов CUDA"),
        ("bin-win-cpu-x64", "cpu",
         "работает на процессоре — медленнее, но заведётся на любой машине"),
    ],
    "linux": [
        ("bin-ubuntu-vulkan-x64", "vulkan",
         "использует видеокарту через Vulkan — быстро, без CUDA/ROCm"),
        ("bin-ubuntu-x64", "cpu",
         "работает на процессоре — медленнее, но универсально"),
    ],
    "macos-arm": [
        ("bin-macos-arm64", "metal",
         "использует чип Apple (Metal) — быстро и энергоэффективно"),
    ],
    "macos-x64": [
        ("bin-macos-x64", "cpu",
         "работает на процессоре Intel Mac"),
    ],
}


def platform_key(system: str, machine: str) -> str:
    """Map platform.system()/platform.machine() to a preference bucket."""
    system = (system or "").lower()
    machine = (machine or "").lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "macos-arm" if machine in ("arm64", "aarch64") else "macos-x64"
    return "linux"


def resolve_llama_asset(release_assets: list[str], system: str, machine: str,
                        prefer_gpu: bool = True) -> LlamaAsset | None:
    """Choose the best asset filename from a release's asset list.

    `release_assets` is the list of filenames in the GitHub release.
    Returns None if nothing matches (the wizard then tells the user how to
    install llama.cpp manually)."""
    key = platform_key(system, machine)
    prefs = _PREFERENCE.get(key, _PREFERENCE["linux"])
    ordered = prefs if prefer_gpu else [p for p in prefs if p[1] == "cpu"] or prefs
    for substr, backend, reason in ordered:
        for name in release_assets:
            if substr in name.lower():
                return LlamaAsset(filename=name, backend=backend, reason=reason)
    return None
