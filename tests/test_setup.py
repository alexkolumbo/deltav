"""Setup wizard: asset resolution, model planning, custom models, i18n."""
import json
import os

import httpx
import pytest

from deltav.setup.assets import platform_key, resolve_llama_asset
from deltav.setup.custom import ModelAnalysis, analyze_model, parse_ref
from deltav.setup.i18n import T, detect_lang
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

def test_pick_model_records_a_servable_spec(tmp_path, monkeypatch):
    wiz = SetupWizard(home=tmp_path, lang="en")
    from deltav.compute.base import DeviceInfo
    wiz.device = DeviceInfo(vendor="amd", name="RX 6600M", vram_mb=8176)
    monkeypatch.setattr("builtins.input", lambda *a: "")  # decline "show others"
    wiz.pick_model(8)
    assert wiz.spec is not None
    assert wiz.state["model"] == wiz.spec.ref
    assert wiz.spec.params_b >= 3  # 8 GB fits a 7-8B model


# ------------------------------------------------------------------ i18n

def test_translator_falls_back_to_english():
    t = T("ru")
    assert t("s_hardware") == "Смотрю, какое у вас железо"
    assert T("en")("s_hardware") == "Checking your hardware"
    assert T("xx")("s_hardware") == T("en")("s_hardware")  # unknown -> en
    assert t("nonexistent_key") == "nonexistent_key"       # missing -> key


def test_translator_formats():
    assert "5 GB" in T("en")("gpu_found", name="X", vram="5 GB")


def test_detect_lang_from_env(monkeypatch):
    monkeypatch.setenv("DELTAV_LANG", "ru")
    assert detect_lang() == "ru"
    monkeypatch.setenv("DELTAV_LANG", "en_US")
    assert detect_lang() == "en"


# ---------------------------------------------------------- custom models

def test_parse_ref():
    assert parse_ref("org/repo::file.gguf") == ("org/repo", "file.gguf")
    assert parse_ref("org/repo") == ("org/repo", "")
    assert parse_ref(" org/repo/ ") == ("org/repo", "")


def _hf_head_handler(size_bytes: int):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD" and request.url.path.endswith(".gguf"):
            return httpx.Response(200, headers={"content-length": str(size_bytes)})
        return httpx.Response(404)
    return handler


def test_analyze_model_fits_verdict():
    # a ~2 GB model on 8 GB VRAM should fit well
    client = httpx.Client(transport=httpx.MockTransport(_hf_head_handler(2_000_000_000)))
    a = analyze_model("some/repo::m-Q4_K_M.gguf", vram_mb=8176, client=client)
    assert a.file_mb > 1500
    assert a.verdict in ("great", "tight") and a.fits
    assert a.spec is not None and a.max_context >= 2048


def test_analyze_model_too_big_verdict():
    # a ~40 GB model on 8 GB VRAM cannot fit
    client = httpx.Client(transport=httpx.MockTransport(_hf_head_handler(40_000_000_000)))
    a = analyze_model("big/repo::m-Q4_K_M.gguf", vram_mb=8176, client=client)
    assert a.verdict == "too_big" and not a.fits


def test_analyze_model_unreadable():
    def handler(request):
        return httpx.Response(404)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    a = analyze_model("nope/none::x.gguf", vram_mb=8176, client=client)
    assert a.verdict == "unknown" and a.spec is None


# ------------------------------------------------------------- expanded catalog

def test_catalog_covers_more_families():
    from deltav.router import Catalog
    families = {s.family for s in Catalog().specs}
    assert {"gemma2", "mistral", "phi", "deepseek-r1", "qwen2.5", "llama3"} <= families
    chat = [s for s in Catalog().specs if s.kind == "chat"]
    assert len(chat) >= 15


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
        "ctx": 7168,
        "mmproj_path": str(tmp_path / "mmproj-F16.gguf"),
    }
    script = wiz.write_launcher()
    assert script.exists()
    body = script.read_text(encoding="utf-8")
    assert "llama-server" in body
    assert "deltav.cli" in body and "node" in body
    assert "--price" in body and "http://seed:9100" in body
    # External access is automatic — a wizard node must not stay LAN-only.
    assert "--connect" in body and "auto" in body
    # Engine context respects the computed fit (7168 -> 7168, never 8192).
    assert "7168" in body and "8192" not in body
    # A vision model launches with its projector, or images silently fail.
    assert "--mmproj" in body and "mmproj-F16.gguf" in body
    # Uses the real interpreter by absolute path, not bare "python" (which on
    # Windows can resolve to the Microsoft Store stub).
    import sys
    assert sys.executable in body
    # Waits for the engine's health, never a fixed sleep for readiness.
    assert "8085/health" in body
    # The robust pattern: detached launch (survives the window/session) with
    # logs — NOT `start "llama-server"` (reaped, no logs).
    if os.name == "nt":
        assert "Start-Process" in body and 'start "llama-server"' not in body
        assert "engine.log" in body and "node.log" in body


def test_launcher_ctx_capped_and_defaulted(tmp_path):
    wiz = SetupWizard(home=tmp_path)
    base = {
        "server": "llama-server", "model": "o/r::m.gguf", "model_path": "m.gguf",
        "genesis": "g.json", "wallet": "w.json", "seed": "http://s:9100", "price": 1,
    }
    wiz.state = dict(base, ctx=131072)          # huge fit -> practical cap
    assert "8192" in wiz.write_launcher().read_text(encoding="utf-8")
    wiz.state = dict(base, ctx=2500)            # rounded down to 1024 multiple
    assert "2048" in wiz.write_launcher().read_text(encoding="utf-8")
    wiz.state = dict(base)                      # unknown -> safe default
    body = wiz.write_launcher().read_text(encoding="utf-8")
    assert "4096" in body and "--mmproj" not in body


def test_launcher_adds_relay_via_for_via_seed(tmp_path):
    """A remote node MUST be told a relay or it stays LAN-only. Derive it from
    a `…/via/<id>` seed URL automatically; a plain LAN seed adds none."""
    wiz = SetupWizard(home=tmp_path)
    base = {"server": "llama-server", "model": "o/r::m.gguf", "model_path": "m.gguf",
            "genesis": "g.json", "wallet": "w.json", "price": 1}
    wiz.state = dict(base, seed="http://relay.example:9200/via/dv1abc")
    body = wiz.write_launcher().read_text(encoding="utf-8")
    assert "--relay-via" in body and "http://relay.example:9200" in body
    wiz.state = dict(base, seed="http://lan:9100")     # plain LAN seed
    assert "--relay-via" not in wiz.write_launcher().read_text(encoding="utf-8")
    # explicit --relay wins even without a /via seed
    wiz2 = SetupWizard(home=tmp_path, relay="http://r:9200")
    wiz2.state = dict(base, seed="http://lan:9100")
    assert "http://r:9200" in wiz2.write_launcher().read_text(encoding="utf-8")


def test_rank_score_weighs_context_modality_quant_and_served():
    """The wizard's ordering follows the composite score: usable context,
    vision, quant fidelity and being already-served all matter — not just
    parameter count."""
    from deltav.registry import ModelRegistry

    base = {"quality": 0.75, "max_context": 4096, "vision": False,
            "quant": "Q4_K_M", "served": False, "downloads": 0}
    s = ModelRegistry.score
    # More usable context wins between equal-quality models.
    assert s(dict(base, max_context=32768)) > s(dict(base, max_context=3072))
    # A 32k 7B outranks a slightly "better" model stuck at 3k context.
    assert s(dict(base, max_context=32768)) > s(dict(base, quality=0.76, max_context=3072))
    # Vision is a capability bonus.
    assert s(dict(base, vision=True)) > s(base)
    # Higher-bit quants rank above lower-bit at equal quality.
    assert s(dict(base, quant="Q6_K")) > s(dict(base, quant="Q4_K_M")) > s(dict(base, quant="Q2_K"))
    # Already-served models get a strong practical boost.
    assert s(dict(base, served=True)) > s(dict(base, quality=0.85))


def test_rank_returns_a_dozen_with_scores():
    from deltav.registry import ModelRegistry
    from deltav.router import Catalog

    ranked = ModelRegistry(catalog=Catalog()).rank(24_000, kind="chat", top=12)
    assert len(ranked) >= 10, "the wizard should offer a real choice, not one model"
    assert all("score" in r for r in ranked)
    assert ranked == sorted(ranked, key=lambda r: -r["score"])


def test_rank_dedupes_same_model_from_different_uploaders():
    from deltav.registry import ModelRegistry
    from deltav.registry.registry import DiscoveredModel
    from deltav.router import Catalog

    catalog = Catalog()
    reg = ModelRegistry(catalog=catalog)
    # A HF-discovered copy of a curated catalog model (different uploader).
    dup_repo = next(s for s in catalog.specs if s.kind == "chat" and s.params_b <= 8)
    reg.discovered[f"bartowski-dup/{dup_repo.repo_id.split('/')[-1]}::x.gguf"] = DiscoveredModel(
        repo_id=f"bartowski-dup/{dup_repo.repo_id.split('/')[-1]}",
        filename="x.gguf", params_b=dup_repo.params_b, quant=dup_repo.quant,
        file_mb=dup_repo.file_mb - 100, downloads=10, kind="chat")
    ranked = reg.rank(24_000, kind="chat", top=50)
    names = [(r["repo"].split("/")[-1].lower(), (r["quant"] or "").lower()) for r in ranked]
    assert len(names) == len(set(names)), "duplicate model shown twice in the wizard list"


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


def test_connect_network_auto_uses_public_seed_without_prompting(tmp_path, monkeypatch):
    """No --seed → the node connects to the public seed automatically; the
    wizard must NOT prompt for a seed address (that last step is gone)."""
    genesis_dict = {
        "params": {"chain_id": "deltav-alpha-3", "price_per_token": 10},
        "alloc": {}, "stakes": {}, "timestamp": 0.0,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/genesis"):            # seed is a /via/<id> URL
            return httpx.Response(200, json=genesis_dict)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    wiz = SetupWizard(home=tmp_path, client=client)          # no seed passed
    def _no_prompt(*a):
        raise AssertionError("wizard must not prompt for a seed")
    monkeypatch.setattr("builtins.input", _no_prompt)
    wiz.connect_network(8)
    assert wiz.state["seed"].startswith("http://5.78.65.237:9200/via/")
    assert wiz.state["genesis"]


def test_connect_network_unreachable_is_graceful(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down", request=request)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    wiz = SetupWizard(home=tmp_path, seed="http://dead:9100", client=client)
    wiz.connect_network(8)  # must not raise
    assert wiz.state["genesis"] == ""
