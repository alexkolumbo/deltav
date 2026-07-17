"""Interactive node-setup wizard: bare machine -> live earning node.

Bilingual (EN/RU), every step explained, safe defaults, visible progress,
resumable. Lets the operator accept the recommended model or paste their
own HuggingFace repo — analyzed for fit, or forced as-is.
"""
from __future__ import annotations

import json
import os
import platform
import re
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


def _safe_under(base: Path, name: str) -> Path:
    """Resolve `name` strictly under `base`, refusing path-traversal. A
    malicious archive member or HuggingFace filename could otherwise carry
    `../` or an absolute path and write outside the install dir."""
    leaf = Path(name).name  # drop any directory components
    if not leaf or leaf in (".", ".."):
        raise ValueError(f"unsafe filename: {name!r}")
    dest = (base / leaf).resolve()
    if base.resolve() not in dest.parents and dest != base.resolve():
        raise ValueError(f"path escapes {base}: {name!r}")
    return dest


def _safe_extract(zf: zipfile.ZipFile, dest_dir: Path) -> None:
    """Zip-slip-safe extraction: every member must resolve under dest_dir."""
    base = dest_dir.resolve()
    for member in zf.namelist():
        target = (dest_dir / member).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"unsafe archive member: {member!r}")
    zf.extractall(dest_dir)

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
                 lang: str = "", client: httpx.Client | None = None,
                 relay: str = ""):
        self.home = Path(home) if home else DEFAULT_HOME
        self.seed = seed
        # Relay base the node tunnels through so it's reachable network-wide
        # (behind NAT/CGNAT). If the seed itself is a `…/via/<id>` URL we can
        # derive it — a remote operator then needs only the public seed URL.
        self.relay = relay
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
        # Rank from the unified registry (curated catalog + HF-discovered DB).
        from ..registry import ModelRegistry
        reg = ModelRegistry(catalog=catalog)
        ranked = reg.rank(self.device.vram_mb, kind="chat", top=12,
                          served=self._served_models())
        if not ranked:
            self.spec = catalog.specs[0]
            self.warn(self.t("light_model"))
        else:
            self.spec = catalog.by_ref(ranked[0]["ref"]) or reg.discovered[ranked[0]["ref"]].to_spec()
            self._present_ranked(ranked[0])
            # Always offer the alternatives — one take-it-or-leave-it model
            # is not a choice. #1 stays the default, so Enter keeps it.
            self._choose_from_registry(ranked, catalog, reg)
        self.state["model"] = self.spec.ref
        # The context this hardware can actually hold — the launcher must not
        # ask llama-server for more, or the engine fails to start on tight fits.
        from ..router.planner import max_context_for
        self.state["ctx"] = max_context_for(self.spec, self.device.vram_mb)
        self.state["vision"] = bool(getattr(self.spec, "vision", False))

    def _served_models(self) -> set[str]:
        """Model refs live nodes already serve (🔥 in the list) — serving a
        warm model means immediate routing demand for a new node."""
        if not self.seed:
            return set()
        try:
            resp = self.client.get(f"{self.seed.rstrip('/')}/chain/nodes", timeout=8.0)
            resp.raise_for_status()
            return {m for n in resp.json().get("nodes", [])
                    if n.get("active") for m in n.get("models", [])}
        except (httpx.HTTPError, ValueError):
            return set()

    def _present_ranked(self, best: dict) -> None:
        self.ok(self.t("recommend", name=best["ref"].split("/")[-1].split("::")[0]))
        self.note(self.t("model_specs", b=best["params_b"], ctx=f"{best['max_context']:,}"))
        self.note(self.t("download_once", size=human_mb(best["file_mb"])))

    def _choose_from_registry(self, ranked: list, catalog, reg) -> None:
        for i, r in enumerate(ranked[:12], 1):
            tag = self.t("recommended_tag") if i == 1 else ""
            icons = ("👁" if r["vision"] else "") + ("🔥" if r["served"] else "")
            name = r["ref"].split("/")[-1].split("::")[0]
            say(f"    {i:2d}. {name:<42.42s} {r['params_b']:>5}B"
                f" · ctx {r['max_context']:>7,} · {r['quant'] or '?':<7}"
                f" · {human_mb(r['file_mb'])} {icons}{tag}")
        self.note(self.t("model_legend"))
        choice = self.ask(self.t("pick_number"), "1").lower()
        if choice == "c":
            self._custom_model()
            return
        try:
            ref = ranked[int(choice) - 1]["ref"]
            self.spec = catalog.by_ref(ref) or reg.discovered[ref].to_spec()
        except (ValueError, IndexError, KeyError):
            pass

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
        exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
        # Search the whole tree (the zip extracts into a subfolder), matching
        # the EXACT executable name — not "llama-server*", which also matches
        # llama-server-impl.dll and would then fail to launch (WinError 193).
        found = next((p for p in self.llama_dir.rglob(exe_name)
                      if p.is_file() and p.name == exe_name), None)
        if found:
            self.ok(self.t("engine_have"))
            self.state["server"] = str(found)
            return
        server = self.llama_dir / exe_name
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
            _safe_extract(zf, self.llama_dir)
        found = next((p for p in self.llama_dir.rglob(exe_name)
                      if p.is_file() and p.name == exe_name), None)
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
        dest = _safe_under(self.models_dir, filename or f"{repo.split('/')[-1]}.gguf")
        if dest.exists() and dest.stat().st_size > 1_000_000:
            self.ok(self.t("model_have"))
        else:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            self.note(self.t("model_tea", size=human_mb(self.spec.file_mb)))
            download(url, dest, "model")
            self.ok(self.t("model_ok"))
        self.state["model_path"] = str(dest)
        if self.state.get("vision"):
            self._download_mmproj(repo)

    def _download_mmproj(self, repo: str) -> None:
        """A vision model needs its projector (mmproj) or images won't work —
        recommending a vision model and launching it text-only would be a lie."""
        existing = next(self.models_dir.glob("mmproj*"), None)
        if existing and existing.stat().st_size > 100_000:
            self.ok(self.t("vision_have"))
            self.state["mmproj_path"] = str(existing)
            return
        try:
            info = self.client.get(
                f"https://huggingface.co/api/models/{repo}", timeout=20.0).json()
            names = [s.get("rfilename", "") for s in info.get("siblings", [])]
            mm = next((n for n in names
                       if n.lower().startswith("mmproj") and n.endswith(".gguf")), "")
        except (httpx.HTTPError, ValueError):
            mm = ""
        if not mm:
            self.warn(self.t("vision_skip"))
            self.state["mmproj_path"] = ""
            return
        try:
            dest = _safe_under(self.models_dir, mm)
        except ValueError:
            self.warn(self.t("vision_skip"))
            self.state["mmproj_path"] = ""
            return
        self.note(self.t("vision_dl"))
        try:
            download(f"https://huggingface.co/{repo}/resolve/main/{mm}", dest, "mmproj")
        except httpx.HTTPError:
            self.warn(self.t("vision_skip"))
            self.state["mmproj_path"] = ""
            return
        self.ok(self.t("vision_ok"))
        self.state["mmproj_path"] = str(dest)

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
        # Default to the PUBLIC relay-published seed URL, not a LAN IP: a
        # remote operator must both fetch genesis/sync AND derive the relay
        # from it. A `…/via/<id>` seed does all three with zero extra config.
        default_seed = "http://5.78.65.237:9200/via/dv1cfb5013a0ff17f0977f01eb3630ce9beb25cf6f5"
        seed = self.seed or self.ask(self.t("seed_prompt"), default_seed)
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
    def _engine_ctx(self) -> int:
        """llama-server context: what the hardware fits (never more — the
        engine refuses to start), capped at a practical serving default."""
        ctx = int(self.state.get("ctx") or 0)
        if ctx <= 0:
            return 4096
        return max(2048, min(8192, (ctx // 1024) * 1024))

    def _engine_args(self) -> list[str]:
        args = ["-m", self.state.get("model_path", ""),
                "--host", "127.0.0.1", "--port", "8085",
                "-ngl", "99", "-c", str(self._engine_ctx())]
        if self.state.get("mmproj_path"):
            args += ["--mmproj", self.state["mmproj_path"]]
        return args

    def _relay_base(self) -> str:
        """The relay this node should tunnel out through. Explicit --relay wins;
        otherwise derive it from a `…/via/<id>` seed URL (a remote node joins
        through the same relay the seed is published on). Empty on a pure-LAN
        setup, where connect=auto stays local."""
        if self.relay:
            return self.relay.rstrip("/")
        m = re.match(r"^(https?://[^/]+)/via/", self.state.get("seed", "") or "")
        return m.group(1) if m else ""

    def _node_args(self) -> list[str]:
        # connect=auto: the node works out its own public address (direct if
        # reachable, else through a relay) — external access with zero config.
        # A remote/NAT'd node MUST be told a relay, or it stays LAN-only (the
        # "no relay available — reachable only on the local network" trap).
        s = self.state
        args = ["--genesis", s.get("genesis", ""), "--wallet", s.get("wallet", ""),
                "--host", "0.0.0.0", "--port", "9100",
                "--backend", "llamaserver", "--model", s.get("model", ""),
                "--data-dir", str(self.home / "data"),
                "--price", str(s.get("price", 0)),
                "--peer", s.get("seed", ""), "--connect", "auto"]
        relay = self._relay_base()
        if relay:
            args += ["--relay-via", relay]
        return args

    @staticmethod
    def _sh(args: list[str]) -> str:
        return " ".join(f'"{a}"' if (" " in a or "\\" in a) else a for a in args)

    @staticmethod
    def _ps_list(args: list[str]) -> str:
        """Format args as a PowerShell -ArgumentList: each a single-quoted
        element, so embedded spaces (paths) never split into extra args — the
        `import`/`logging` split bug that broke the earlier launcher."""
        return ",".join("'" + str(a).replace("'", "''") + "'" for a in args)

    def _py(self) -> str:
        # The SAME interpreter the wizard runs under, absolute — bare "python"
        # on Windows often resolves to the Microsoft Store stub.
        return sys.executable or "python"

    def write_launcher(self) -> Path:
        """A launcher that SURVIVES: self-cleans a previous run, launches the
        engine and node as fully detached processes (they outlive the window /
        the launching session, unlike `start` which gets reaped), waits on the
        engine's HTTP health (never a fixed sleep), and logs to files so a
        failure is diagnosable. This is what makes unattended/scheduled runs
        actually stay up."""
        server = self.state.get("server", "")
        py = self._py()
        home = str(self.home)
        if os.name == "nt":
            script = self.home / "start-node.bat"
            # -FilePath is the exe; -ArgumentList is ONLY the args (never the
            # exe again, or the program receives its own path as argv[1]).
            eng = self._ps_list(self._engine_args())
            nod = self._ps_list(["-m", "deltav.cli", "node"] + self._node_args())
            wait = ('powershell -NoProfile -Command "for($i=0;$i -lt 180;$i++)'
                    '{try{Invoke-WebRequest -UseBasicParsing '
                    'http://127.0.0.1:8085/health -TimeoutSec 2 | Out-Null;'
                    "Write-Host 'engine up';exit 0}catch{Start-Sleep 2}};exit 1\"")
            body = (
                "@echo off\r\n"
                "title DeltaV Node\r\n"
                "echo [DeltaV] Cleaning any previous run...\r\n"
                "taskkill /F /IM llama-server.exe >nul 2>&1\r\n"
                'for /f "tokens=5" %%p in (\'netstat -ano ^| findstr ":9100 " '
                "^| findstr LISTENING') do taskkill /F /PID %%p >nul 2>&1\r\n"
                "timeout /t 2 >nul\r\n"
                "echo [DeltaV] Launching engine...\r\n"
                f'powershell -NoProfile -Command "Start-Process -FilePath \'{server}\' '
                f"-ArgumentList {eng} -RedirectStandardOutput '{home}\\engine.log' "
                f"-RedirectStandardError '{home}\\engine.err' -WindowStyle Minimized\"\r\n"
                "echo [DeltaV] Waiting for engine (up to 6 min)...\r\n"
                f"{wait}\r\n"
                "if errorlevel 1 goto fail\r\n"
                "echo [DeltaV] Launching node...\r\n"
                f'powershell -NoProfile -Command "Start-Process -FilePath \'{py}\' '
                f"-ArgumentList {nod} -RedirectStandardOutput '{home}\\node.log' "
                f"-RedirectStandardError '{home}\\node.err' -WindowStyle Minimized\"\r\n"
                "echo [DeltaV] Node launched. Engine + node keep running after "
                "you close this window.\r\n"
                "timeout /t 8\r\n"
                "goto end\r\n"
                ":fail\r\n"
                f"echo [DeltaV] ENGINE FAILED TO START. See {home}\\engine.err\r\n"
                "pause\r\n"
                ":end\r\n"
            )
        else:
            script = self.home / "start-node.sh"
            engine = self._sh([server] + self._engine_args())
            node = self._sh([py, "-m", "deltav.cli", "node"] + self._node_args())
            wait = ('i=0; while [ $i -lt 180 ]; do '
                    'curl -sf http://127.0.0.1:8085/health >/dev/null 2>&1 && break; '
                    'sleep 2; i=$((i+1)); done')
            body = (
                "#!/bin/sh\n"
                "# self-clean a previous run so a restart binds cleanly\n"
                "pkill -f 'llama-server' 2>/dev/null; "
                "fuser -k 9100/tcp 2>/dev/null; sleep 2\n"
                f'nohup {engine} >"{home}/engine.log" 2>&1 &\n'
                f"{wait}\n"
                f'exec {node} >"{home}/node.log" 2>&1\n'
            )
        script.write_text(body, encoding="utf-8")
        if os.name != "nt":
            os.chmod(script, 0o755)
        return script

    def write_stopper(self) -> Path:
        """A stop script (window-close / manual stop) that kills the node and
        its local engine by listening port, never a blanket process kill."""
        from .schedule import render_windows_stopper
        if os.name == "nt":
            script = self.home / "stop-node.bat"
            script.write_text(render_windows_stopper(home=str(self.home)),
                              encoding="utf-8")
        else:
            script = self.home / "stop-node.sh"
            script.write_text(
                "#!/bin/sh\npkill -f 'deltav.cli node' 2>/dev/null; "
                "pkill -f 'llama-server' 2>/dev/null\n", encoding="utf-8")
            os.chmod(script, 0o755)
        return script

    def launch(self, total: int, auto_start: bool = True) -> None:
        self.step(8, total, "s_launch")
        script = self.write_launcher()
        stopper = self.write_stopper()
        s = self.state
        ready = s.get("server") and s.get("model_path") and s.get("genesis")
        if not ready:
            self.warn(self.t("not_ready"))
            self.note(self.t("script_saved", path=script))
            return
        if not auto_start:
            self.ok(self.t("ready_run"))
            say(f"    {script}")
            self._finish(script)
            return

        # Auto-start (persistence) + optional daily schedule — the turnkey
        # path: the node comes up on its own and survives logout/reboot, so
        # nobody has to babysit a terminal.
        if self.ask_yes("autostart_ask", default=True):
            from .schedule import Schedule
            schedule = Schedule.always()
            if self.ask_yes("schedule_ask", default=False):
                start = self.ask(self.t("schedule_start"), "09:00")
                end = self.ask(self.t("schedule_end"), "23:00")
                try:
                    schedule = Schedule.daily(start, end)
                except ValueError:
                    self.warn(self.t("schedule_bad"))
                    schedule = Schedule.always()
            self.state["schedule"] = (
                {"start": schedule.window.start, "end": schedule.window.end}
                if schedule.window else "always")
            self._save_state()
            if self._install_autostart(script, stopper, schedule):
                self.ok(self.t("autostart_ok"))
                if schedule.window is not None:
                    self.note(self.t("autostart_win", start=schedule.window.start,
                                     end=schedule.window.end))
                if s.get("server"):
                    self.note(self.t("engine_session"))
                self._finish(script)
                return
            self.warn(self.t("autostart_fail"))
            # fall through to a direct one-off start

        # Direct start (no persistence, or auto-start install failed).
        self.note(self.t("starting"))
        llama = subprocess.Popen(
            [s["server"]] + self._engine_args(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not self._wait_health("http://127.0.0.1:8085/health", 180):
            self.warn(self.t("engine_slow"))
            llama.terminate()
            return
        self.ok(self.t("engine_up"))
        env = dict(os.environ, LLAMA_SERVER_URL="http://127.0.0.1:8085")
        subprocess.Popen(
            [sys.executable, "-m", "deltav.cli", "node"] + self._node_args(),
            env=env)
        if not self._wait_health("http://127.0.0.1:9100/health", 60):
            self.warn(self.t("node_slow"))
        self._finish(script)

    def _install_autostart(self, launcher: Path, stopper: Path, schedule) -> bool:
        """Register the OS scheduler so the node auto-starts (and, for a
        window, stops at close). Windows: Task Scheduler (interactive logon —
        the GPU engine gets a desktop session, no stored password). Linux:
        systemd *user* units. Returns True if it installed cleanly."""
        from .schedule import (render_windows_ps, apply_windows,
                               render_systemd_units, write_linux_units)
        import datetime
        if os.name == "nt":
            # start now unless we're outside a configured window
            start_now = True
            if schedule.window is not None:
                now = datetime.datetime.now().strftime("%H:%M")
                w = schedule.window
                start_now = (w.start <= now < w.end) if not w.wraps_midnight \
                    else (now >= w.start or now < w.end)
            ps = render_windows_ps(python=self._py(), launcher=str(launcher),
                                   stopper=str(stopper), node_argline="",
                                   home=str(self.home), schedule=schedule,
                                   start_now=start_now)
            ok, _ = apply_windows(ps)
            return ok
        try:
            import getpass
            units = render_systemd_units(launcher=str(launcher), home=str(self.home),
                                         user=getpass.getuser(), schedule=schedule)
            unit_dir = Path.home() / ".config" / "systemd" / "user"
            write_linux_units(units, unit_dir)
            subprocess.run(["systemctl", "--user", "daemon-reload"], timeout=15, check=False)
            enable = (["deltav-node.service"] if schedule.window is None
                      else ["deltav-node-start.timer", "deltav-node-stop.timer"])
            subprocess.run(["systemctl", "--user", "enable", "--now", *enable],
                           timeout=20, check=False)
            return True
        except (OSError, subprocess.SubprocessError):
            return False

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


def run_setup(home: str = "", seed: str = "", lang: str = "", auto_start: bool = True,
              relay: str = "") -> int:
    SetupWizard(home=home or None, seed=seed, lang=lang, relay=relay).run(auto_start=auto_start)
    return 0
