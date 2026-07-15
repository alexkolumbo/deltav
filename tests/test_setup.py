"""Setup wizard: asset resolution, model planning, launcher generation."""
import json

import httpx
import pytest

from deltav.setup.assets import platform_key, resolve_llama_asset
from deltav.setup.wizard import SetupWizard

WIN_ASSETS = [
    "llama-b10015-bin-win-cpu-arm64.zip",
    "llama-b10015-bin-win-cpu-x64.zip",
    "llama-b10015-bin-win-vulkan-x64.zip",
    "llama-b10015-bin-ubuntu-vulkan-x64.zip",
    "llama-b10015-bin-macos-arm64.zip",
]


# ------------------------------------------------------------- asset pick

def test_platform_key():
    assert platform_key("Windows", "AMD64") == "windows"
    assert platform_key("Darwin", "arm64") == "macos-arm"
    assert platform_key("Darwin", "x86_64") == "macos-x64"
    assert platform_key("Linux", "x86_64") == "linux"


def test_windows_gpu_prefers_vulkan():
    a = resolve_llama_asset(WIN_ASSETS, "Windows", "AMD64", prefer_gpu=True)
    assert a.backend == "vulkan" and "vulkan" in a.filename
    assert "видеокарт" in a.reason


def test_windows_cpu_only_falls_back():
    a = resolve_llama_asset(WIN_ASSETS, "Windows", "AMD64", prefer_gpu=False)
    assert a.backend == "cpu" and "cpu" in a.filename


def test_apple_silicon_gets_metal():
    a = resolve_llama_asset(WIN_ASSETS, "Darwin", "arm64")
    assert a.backend == "metal"


def test_linux_vulkan():
    a = resolve_llama_asset(WIN_ASSETS, "Linux", "x86_64")
    assert a.backend == "vulkan" and "ubuntu" in a.filename


def test_no_matching_asset_returns_none():
    assert resolve_llama_asset(["llama-b1-bin-freebsd.zip"], "Windows", "AMD64") is None


# ------------------------------------------------------------ wizard logic

def test_pick_model_records_a_servable_spec(tmp_path):
    wiz = SetupWizard(home=tmp_path)
    from deltav.compute.base import DeviceInfo
    wiz.device = DeviceInfo(vendor="amd", name="RX 6600M", vram_mb=8176)
    # non-interactive: default 'no' to the "show others" prompt
    import builtins
    orig = builtins.input
    builtins.input = lambda *a: ""
    try:
        wiz.pick_model(8)
    finally:
        builtins.input = orig
    assert wiz.spec is not None
    assert wiz.state["model"] == wiz.spec.ref
    assert wiz.spec.params_b >= 3  # 8 GB fits a 7-8B model


def test_launcher_script_is_runnable(tmp_path):
    wiz = SetupWizard(home=tmp_path)
    wiz.state = {
        "server": str(tmp_path / "llama-server"),
        "model": "org/repo::model.gguf",
        "model_path": str(tmp_path / "model.gguf"),
        "genesis": str(tmp_path / "genesis.json"),
        "wallet": str(tmp_path / "node.wallet.json"),
        "seed": "http://seed:9100",
        "price": 9,
    }
    script = wiz.write_launcher()
    assert script.exists()
    body = script.read_text(encoding="utf-8")
    assert "llama-server" in body
    assert "deltav.cli node" in body
    assert "--price 9" in body
    assert "http://seed:9100" in body


def test_connect_network_saves_genesis(tmp_path):
    genesis_dict = {
        "params": {"chain_id": "deltav-test", "price_per_token": 10},
        "alloc": {"dv1abc": 1000}, "stakes": {}, "timestamp": 0.0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/genesis":
            return httpx.Response(200, json=genesis_dict)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    wiz = SetupWizard(home=tmp_path, seed="http://seed:9100", client=client)
    import builtins
    orig = builtins.input
    builtins.input = lambda *a: ""
    try:
        wiz.connect_network(8)
    finally:
        builtins.input = orig
    assert wiz.state["genesis"]
    from deltav.config import Genesis
    g = Genesis.load(wiz.state["genesis"])
    assert g.params.chain_id == "deltav-test"


def test_connect_network_unreachable_is_graceful(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down", request=request)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    wiz = SetupWizard(home=tmp_path, seed="http://dead:9100", client=client)
    wiz.connect_network(8)  # must not raise
    assert wiz.state["genesis"] == ""
