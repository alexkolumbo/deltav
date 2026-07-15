"""Interactive node-setup wizard: bare machine -> live earning node.

Bilingual (EN/RU), every step explained, safe defaults, visible progress,
resumable. Lets the operator accept the recommended model or paste their
own HuggingFace repo — analyzed for fit, or forced as-is.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx

from ..compute import detect_device
from ..config import Genesis
from ..economics import price_report
from ..router import Catalog, plan
from ..wallet import load_or_create
from .assets import resolve_llama_asset
from .custom import analyze_model, parse_ref
from .i18n import T, detect_lang

LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
DEFAULT_HOME = Path.home() / "deltav-node"

C_OK, C_WARN, C_DIM, C_ACCENT, C_RESET = (
    ("\033[32m", "\033[33m", "\033[90m", "\033[36m", "\033[0m")
    if sys.stdout.isatty() else ("", "", "", "", "")
)


def say(msg: str = "") -> None:
    print(msg)


def human_mb(mb: int) -> str:
    return f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb} MB"


def download(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, last = 0, 0.0
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if total and (now - last > 0.2 or done == total):
                    pct = done / total
                    bar = "█" * int(pct * 28) + "·" * (28 - int(pct * 28))
                    print(f"\r  {label} [{bar}] {pct*100:4.0f}%  {done/1e6:6.0f} MB",
                          end="", flush=True)
                    last = now
    tmp.replace(dest)
    print()


class SetupWizard:
    def __init__(self, home: Path | None = None, seed: str = "",
                 lang: str = "", client: httpx.Client | None = None):
        self.home = Path(home) if home else DEFAULT_HOME
        self.seed = seed
        self.t = T(lang or detect_lang())
        self.client = client or httpx.Client(timeout=30.0)
        self.state_file = self.home / "setup.json"
        self.state: dict = {}
        self.device = None
        self.spec = None
        self.llama_dir = self.home / "llama"
        self.models_dir = self.home / "models"

    # ---------------------------------------------------------- presentation
    def step(self, n: int, total: int, key: str) -> None:
        say(f"\n{C_ACCENT}[{n}/{total}] {self.t(key)}{C_RESET}")

    def note(self, text: str) -> None:
        say(f"  {C_DIM}{text}{C_RESET}")

    def ok(self, text: str) -> None:
        say(f"  {C_OK}✓{C_RESET} {text}")

    def warn(self, text: str) -> None:
        say(f"  {C_WARN}!{C_RESET} {text}")

    def ask(self, text: str, default: str = "") -> str:
        suffix = f" [{default}]" if default else ""
        try:
            answer = input(f"  {text}{suffix}: ").strip()
        except EOFError:
            answer = ""
        return answer or default

    def ask_yes(self, key: str, default: bool = True, **fmt) -> bool:
        d = "Y/n" if default else "y/N"
        answer = self.ask(f"{self.t(key, **fmt)} ({d})").lower()
        if not answer:
            return default
        return answer.startswith(("y", "д"))

    # --------------------------------------------------------------- state
    def _load_state(self) -> None:
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))
            if self.state.get("lang"):
                self.t = T(self.state["lang"])

    def _save_state(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.state["lang"] = self.t.lang
        self.state_file.write_text(json.dumps(self.state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    # --------------------------------------------------------------- steps
    def welcome(self) -> None:
        choice = self.ask(self.t("lang_prompt"), self.t.lang).lower()
        if choice.startswith("ru"):
            self.t = T("ru")
        elif choice.startswith("en"):
            self.t = T("en")
        say(f"\n{C_ACCENT}╔══════════════════════════════════════════╗{C_RESET}")
        say(f"{C_ACCENT}║   ΔV   {self.t('title'):<34}║{C_RESET}")
        say(f"{C_ACCENT}╚══════════════════════════════════════════╝{C_RESET}\n")
        say(self.t("intro"))
        self.note(self.t("flow"))
        say()
        self.note(self.t("install_dir", home=self.home))

    def detect_hardware(self, total: int) -> None:
        self.step(1, total, "s_hardware")
        self.device = detect_device()
        d = self.device
        if d.vendor in ("nvidia", "amd", "intel"):
            self.ok(self.t("gpu_found", name=d.name, vram=human_mb(d.vram_mb)))
            self.note(self.t("gpu_good"))
        else:
            self.warn(self.t("no_gpu", vram=human_mb(d.vram_mb)))
            self.note(self.t("no_gpu_note"))
        self.state["device"] = d.to_dict()

    # ------------------------------------------------------------ model pick
    def pick_model(self, total: int) -> None:
        self.step(2, total, "s_model")
        catalog = Catalog()
        options = plan(self.device.vram_mb, objective="balanced", catalog=catalog)
        if not options:
            self.spec = catalog.specs[0]
            self.warn(self.t("light_model"))
        else:
            best = options[0]
            self.spec = catalog.by_ref(best.ref)
            self._present_recommended(best)
            if self.ask_yes("show_others", default=False):
                self._choose_from_list(options, catalog)
        self.state["model"] = self.spec.ref

    def _present_recommended(self, best) -> None:
        self.ok(self.t("recommend", name=self.spec.repo_id.split("/")[-1]))
        self.note(self.t("model_specs", b=self.spec.params_b, ctx=f"{best.max_context:,}"))
        self.note(self.t("download_once", size=human_mb(self.spec.file_mb)))

    def _choose_from_list(self, options, catalog) -> None:
        for i, o in enumerate(options[:8], 1):
            tag = self.t("recommended_tag") if i == 1 else ""
            say(f"    {i}. {o.ref.split('/')[-1].split('::')[0]} "
                f"· {o.params_b}B · ctx {o.max_context:,}{tag}")
        choice = self.ask(self.t("pick_number"), "1").lower()
        if choice == "c":
            self._custom_model()
            return
        try:
            self.spec = catalog.by_ref(options[int(choice) - 1].ref)
        except (ValueError, IndexError):
            pass

    def _custom_model(self) -> None:
        ref = self.ask(self.t("custom_prompt"))
        if not ref:
            return
        say(f"  {self.t('custom_analyze')}")
        say(f"  {self.t('custom_forced')}")
        mode = self.ask(self.t("custom_choice"), "a").lower()
        if mode.startswith("f"):
            repo, filename = parse_ref(ref)
            self.spec = _forced_spec(repo, filename)
            self.warn(self.t("forced_note", ref=self.spec.ref))
            return
        self._analyze_and_maybe_use(ref)

    def _analyze_and_maybe_use(self, ref: str) -> None:
        self.note(self.t("analyzing", ref=ref))
        a = analyze_model(ref, self.device.vram_mb, client=self.client)
        if a.verdict == "unknown" or a.spec is None:
            self.warn(self.t("analyze_fail"))
            return
        size = human_mb(a.file_mb)
        if a.verdict == "great":
            self.ok(self.t("verdict_great", size=size, ctx=f"{a.max_context:,}"))
            use = self.ask_yes("use_it", default=True)
        elif a.verdict == "tight":
            self.warn(self.t("verdict_tight", size=size, ctx=f"{a.max_context:,}"))
            use = self.ask_yes("use_it", default=True)
        elif a.verdict == "cpu_offload":
            self.warn(self.t("verdict_cpu", size=size))
            use = self.ask_yes("use_anyway", default=False)
        else:
            self.warn(self.t("verdict_big", size=size, vram=human_mb(self.device.vram_mb)))
            use = self.ask_yes("use_anyway", default=False)
        if use:
            self.spec = a.spec

    # --------------------------------------------------------------- engine
    def install_engine(self, total: int) -> None:
        self.step(3, total, "s_engine")
        server = self.llama_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if server.exists():
            self.ok(self.t("engine_have"))
            self.state["server"] = str(server)
            return
        self.note(self.t("engine_dl"))
        try:
            rel = self.client.get(LLAMA_RELEASES, timeout=30.0).json()
        except httpx.HTTPError:
            self.warn(self.t("engine_none"))
            self.state["server"] = ""
            return
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        chosen = resolve_llama_asset(list(assets), platform.system(), platform.machine(),
                                     prefer_gpu=self.device.vendor != "cpu")
        if chosen is None:
            self.warn(self.t("engine_none"))
            self.note("https://github.com/ggml-org/llama.cpp/releases")
            self.state["server"] = ""
            return
        self.note(chosen.reason)
        zip_path = self.llama_dir / chosen.filename
        download(assets[chosen.filename], zip_path, "engine")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.llama_dir)
        found = next((p for p in self.llama_dir.rglob("llama-server*")
                      if p.name.startswith("llama-server")), None)
        if found:
            server = found
        try:
            zip_path.unlink()
        except OSError:
            pass
        if server.exists():
            if os.name != "nt":
                os.chmod(server, 0o755)
            self.ok(self.t("engine_ok"))
            self.state["server"] = str(server)
        else:
            self.warn(self.t("engine_none"))
            self.state["server"] = ""

    def download_model(self, total: int) -> None:
        self.step(4, total, "s_model_dl")
        repo, _, filename = self.spec.ref.partition("::")
        dest = self.models_dir / (filename or f"{repo.split('/')[-1]}.gguf")
        if dest.exists() and dest.stat().st_size > 1_000_000:
            self.ok(self.t("model_have"))
        else:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            self.note(self.t("model_tea", size=human_mb(self.spec.file_mb)))
            download(url, dest, "model")
            self.ok(self.t("model_ok"))
        self.state["model_path"] = str(dest)

    def setup_wallet(self, total: int) -> None:
        self.step(5, total, "s_wallet")
        self.note(self.t("wallet_note"))
        wallet_path = self.home / "node.wallet.json"
        kp = load_or_create(wallet_path)
        self.ok(self.t("wallet_addr", addr=kp.address))
        self.note(self.t("wallet_keep"))
        self.state["wallet"] = str(wallet_path)
        self.state["address"] = kp.address

    def connect_network(self, total: int) -> None:
        self.step(6, total, "s_network")
        seed = self.seed or self.ask(self.t("seed_prompt"), "http://10.0.0.223:9100")
        genesis_path = self.home / "genesis.json"
        try:
            resp = self.client.get(f"{seed.rstrip('/')}/genesis", timeout=10.0)
            resp.raise_for_status()
            Genesis.from_dict(resp.json()).save(genesis_path)
            self.ok(self.t("connected", chain=resp.json()["params"]["chain_id"], seed=seed))
            self.state["seed"] = seed
            self.state["genesis"] = str(genesis_path)
        except httpx.HTTPError as exc:
            self.warn(self.t("no_network", err=exc))
            self.note(self.t("no_network2"))
            self.state["seed"] = seed
            self.state["genesis"] = ""

    def set_price(self, total: int) -> None:
        self.step(7, total, "s_price")
        watts = 130.0 if self.device.vendor != "cpu" else 90.0
        report = price_report(watts=watts, tokens_per_sec=30.0)
        self.note(self.t("price_note", usd=report.price_usd_per_million))
        rec = report.suggested_price_udvt
        if self.ask_yes("price_ask", default=True, rec=rec):
            price = rec
        else:
            try:
                price = int(self.ask(self.t("price_custom"), str(rec)))
            except ValueError:
                price = rec
        self.ok(self.t("price_set", price=price))
        self.state["price"] = price

    # -------------------------------------------------------------- launcher
    def write_launcher(self) -> Path:
        s = self.state
        model, server = s.get("model", ""), s.get("server", "")
        model_path = s.get("model_path", "")
        if os.name == "nt":
            script = self.home / "start-node.bat"
            body = (
                "@echo off\r\n"
                f'start "llama-server" "{server}" -m "{model_path}" '
                "--host 127.0.0.1 --port 8085 -ngl 99 -c 8192\r\n"
                "timeout /t 8 >nul\r\n"
                f'python -m deltav.cli node --genesis "{s.get("genesis","")}" '
                f'--wallet "{s.get("wallet","")}" --host 0.0.0.0 --port 9100 '
                f'--backend llamaserver --model "{model}" '
                f'--data-dir "{self.home / "data"}" --price {s.get("price",0)} '
                f'--peer {s.get("seed","")}\r\n'
            )
        else:
            script = self.home / "start-node.sh"
            body = (
                "#!/bin/sh\n"
                f'"{server}" -m "{model_path}" --host 127.0.0.1 --port 8085 '
                "-ngl 99 -c 8192 &\n"
                "sleep 8\n"
                f'python -m deltav.cli node --genesis "{s.get("genesis","")}" '
                f'--wallet "{s.get("wallet","")}" --host 0.0.0.0 --port 9100 '
                f'--backend llamaserver --model "{model}" '
                f'--data-dir "{self.home / "data"}" --price {s.get("price",0)} '
                f'--peer {s.get("seed","")}\n'
            )
        script.write_text(body, encoding="utf-8")
        if os.name != "nt":
            os.chmod(script, 0o755)
        return script

    def launch(self, total: int, auto_start: bool = True) -> None:
        self.step(8, total, "s_launch")
        script = self.write_launcher()
        s = self.state
        ready = s.get("server") and s.get("model_path") and s.get("genesis")
        if not ready:
            self.warn(self.t("not_ready"))
            self.note(self.t("script_saved", path=script))
            return
        if not auto_start:
            self.ok(self.t("ready_run"))
            say(f"    {script}")
            return
        self.note(self.t("starting"))
        llama = subprocess.Popen(
            [s["server"], "-m", s["model_path"], "--host", "127.0.0.1",
             "--port", "8085", "-ngl", "99", "-c", "8192"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not self._wait_health("http://127.0.0.1:8085/health", 180):
            self.warn(self.t("engine_slow"))
            llama.terminate()
            return
        self.ok(self.t("engine_up"))
        env = dict(os.environ, LLAMA_SERVER_URL="http://127.0.0.1:8085")
        subprocess.Popen(
            [sys.executable, "-m", "deltav.cli", "node",
             "--genesis", s["genesis"], "--wallet", s["wallet"],
             "--host", "0.0.0.0", "--port", "9100",
             "--backend", "llamaserver", "--model", s["model"],
             "--data-dir", str(self.home / "data"),
             "--price", str(s.get("price", 0)), "--peer", s.get("seed", "")],
            env=env)
        if not self._wait_health("http://127.0.0.1:9100/health", 60):
            self.warn(self.t("node_slow"))
        self._finish(script)

    def _wait_health(self, url: str, timeout: int) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.client.get(url, timeout=3.0).status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            time.sleep(2)
        return False

    def _finish(self, script: Path) -> None:
        say()
        say(f"{C_OK}╔══════════════════════════════════════════╗{C_RESET}")
        say(f"{C_OK}║   {self.t('done_title')}{C_RESET}")
        say(f"{C_OK}╚══════════════════════════════════════════╝{C_RESET}\n")
        self.ok(self.t("done_panel"))
        say("    http://<this-computer>:9100/explorer")
        self.ok(self.t("done_addr"))
        say(f"    {self.state.get('address','')}")
        say()
        self.note(self.t("done_next", script=script))
        self.note(self.t("done_stop"))

    # --------------------------------------------------------------- driver
    def run(self, auto_start: bool = True) -> dict:
        self._load_state()
        total = 8
        self.welcome()
        try:
            self.detect_hardware(total); self._save_state()
            self.pick_model(total); self._save_state()
            self.install_engine(total); self._save_state()
            self.download_model(total); self._save_state()
            self.setup_wallet(total); self._save_state()
            self.connect_network(total); self._save_state()
            self.set_price(total); self._save_state()
            self.launch(total, auto_start=auto_start)
        except KeyboardInterrupt:
            say()
            self.warn(self.t("interrupted"))
        finally:
            self._save_state()
        return self.state


def _forced_spec(repo: str, filename: str):
    from ..router.catalog import ModelSpec
    return ModelSpec(repo_id=repo, filename=filename or "", family="custom",
                     params_b=0.0, quant="?", file_mb=0, quality=0.5, max_ctx=32768)


def run_setup(home: str = "", seed: str = "", lang: str = "", auto_start: bool = True) -> int:
    SetupWizard(home=home or None, seed=seed, lang=lang).run(auto_start=auto_start)
    return 0
