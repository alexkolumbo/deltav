"""Interactive node-setup wizard: bare machine -> live earning node.

Design goals: every step is explained in plain language, has a safe
default, and shows visible progress. State is saved so re-running is
instant. The wizard prints a launch script at the end so future starts
are one command.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx

from ..compute import detect_device
from ..config import DVT, Genesis
from ..economics import price_report
from ..router import Catalog, launch_hint, max_context_for, plan
from ..wallet import KeyPair, load_or_create, save_wallet
from .assets import resolve_llama_asset

LLAMA_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
DEFAULT_HOME = Path.home() / "deltav-node"

# ------------------------------------------------------------ presentation

C_OK, C_WARN, C_DIM, C_ACCENT, C_RESET = (
    ("\033[32m", "\033[33m", "\033[90m", "\033[36m", "\033[0m")
    if sys.stdout.isatty() else ("", "", "", "", "")
)


def say(msg: str = "") -> None:
    print(msg)


def step(n: int, total: int, title: str) -> None:
    say(f"\n{C_ACCENT}[{n}/{total}] {title}{C_RESET}")


def note(msg: str) -> None:
    say(f"  {C_DIM}{msg}{C_RESET}")


def ok(msg: str) -> None:
    say(f"  {C_OK}✓{C_RESET} {msg}")


def warn(msg: str) -> None:
    say(f"  {C_WARN}!{C_RESET} {msg}")


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"  {prompt}{suffix}: ").strip()
    except EOFError:
        answer = ""
    return answer or default


def ask_yes(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    answer = ask(f"{prompt} ({d})").lower()
    if not answer:
        return default
    return answer.startswith(("y", "д"))


def human_mb(mb: int) -> str:
    return f"{mb/1024:.1f} ГБ" if mb >= 1024 else f"{mb} МБ"


def download(url: str, dest: Path, label: str) -> None:
    """Stream a download with a simple progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        last = 0.0
        with tmp.open("wb") as fh:
            for chunk in r.iter_bytes(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if total and (now - last > 0.2 or done == total):
                    pct = done / total
                    bar = "█" * int(pct * 28) + "·" * (28 - int(pct * 28))
                    print(f"\r  {label} [{bar}] {pct*100:4.0f}%  {done/1e6:6.0f} МБ",
                          end="", flush=True)
                    last = now
    tmp.replace(dest)
    print()


class SetupWizard:
    def __init__(self, home: Path | None = None, seed: str = "", client: httpx.Client | None = None):
        self.home = Path(home) if home else DEFAULT_HOME
        self.seed = seed
        self.client = client or httpx.Client(timeout=30.0)
        self.state_file = self.home / "setup.json"
        self.state: dict = {}
        self.device = None
        self.spec = None
        self.llama_dir = self.home / "llama"
        self.models_dir = self.home / "models"

    # --------------------------------------------------------------- state
    def _load_state(self) -> None:
        if self.state_file.exists():
            self.state = json.loads(self.state_file.read_text(encoding="utf-8"))

    def _save_state(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self.state, indent=2, ensure_ascii=False),
                                   encoding="utf-8")

    # --------------------------------------------------------------- steps
    def welcome(self) -> None:
        say(f"{C_ACCENT}╔══════════════════════════════════════════╗{C_RESET}")
        say(f"{C_ACCENT}║   ΔV   Delta V — установка ноды           ║{C_RESET}")
        say(f"{C_ACCENT}╚══════════════════════════════════════════╝{C_RESET}")
        say()
        say("Нода — это ваш компьютер, который отвечает на запросы к ИИ")
        say("и зарабатывает за это токены DVT. Я проведу вас по шагам:")
        say(f"  {C_DIM}железо → движок → модель → кошелёк → сеть → запуск{C_RESET}")
        say()
        note(f"Всё установится в: {self.home}")

    def detect_hardware(self, total: int) -> None:
        step(1, total, "Смотрю, какое у вас железо")
        self.device = detect_device()
        d = self.device
        if d.vendor in ("nvidia", "amd", "intel"):
            ok(f"Видеокарта: {d.name} — {human_mb(d.vram_mb)} видеопамяти")
            note("Отлично — ИИ будет работать быстро на видеокарте.")
        else:
            warn(f"Видеокарта не найдена, буду считать на процессоре ({human_mb(d.vram_mb)} ОЗУ)")
            note("Заведётся, но ответы будут медленнее. Это нормально для старта.")
        self.state["device"] = d.to_dict()

    def pick_model(self, total: int) -> None:
        step(2, total, "Подбираю модель ИИ под ваше железо")
        catalog = Catalog()
        options = plan(self.device.vram_mb, objective="balanced", catalog=catalog)
        if not options:
            self.spec = catalog.specs[0]
            warn("Железо скромное — беру самую лёгкую модель.")
        else:
            best = options[0]
            self.spec = catalog.by_ref(best.ref)
            ok(f"Рекомендую: {self.spec.repo_id.split('/')[-1]}")
            note(f"{self.spec.params_b}B параметров, влезает контекст ~{best.max_context:,} токенов")
            note(f"Скачать нужно ~{human_mb(self.spec.file_mb)} один раз.")
            if len(options) > 1 and ask_yes("Показать другие варианты?", default=False):
                for i, o in enumerate(options[:6], 1):
                    tag = " (рекомендую)" if i == 1 else ""
                    say(f"    {i}. {o.ref.split('/')[-1].split('::')[0]} "
                        f"· {o.params_b}B · ctx {o.max_context:,}{tag}")
                choice = ask("Номер модели", "1")
                try:
                    self.spec = catalog.by_ref(options[int(choice) - 1].ref)
                except (ValueError, IndexError):
                    pass
        self.state["model"] = self.spec.ref

    def install_engine(self, total: int) -> None:
        step(3, total, "Ставлю движок (llama.cpp)")
        server = self.llama_dir / ("llama-server.exe" if os.name == "nt" else "llama-server")
        if server.exists():
            ok("Движок уже установлен.")
            self.state["server"] = str(server)
            return
        note("Скачиваю готовый бинарник — компилировать ничего не нужно.")
        try:
            rel = self.client.get(LLAMA_RELEASES, timeout=30.0).json()
        except httpx.HTTPError as exc:
            warn(f"Не смог получить список релизов: {exc}")
            self.state["server"] = ""
            return
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        chosen = resolve_llama_asset(list(assets), platform.system(), platform.machine(),
                                     prefer_gpu=self.device.vendor != "cpu")
        if chosen is None:
            warn("Готового бинарника под вашу систему нет — установите llama.cpp вручную.")
            note("https://github.com/ggml-org/llama.cpp/releases")
            self.state["server"] = ""
            return
        note(f"Вариант: {chosen.reason}")
        zip_path = self.llama_dir / chosen.filename
        download(assets[chosen.filename], zip_path, "движок")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.llama_dir)
        found = next((p for p in self.llama_dir.rglob("llama-server*")
                      if p.name.startswith("llama-server")), None)
        if found and found != server:
            server = found
        try:
            zip_path.unlink()
        except OSError:
            pass
        if server.exists():
            if os.name != "nt":
                os.chmod(server, 0o755)
            ok("Движок установлен.")
            self.state["server"] = str(server)
        else:
            warn("Что-то пошло не так при распаковке движка.")
            self.state["server"] = ""

    def download_model(self, total: int) -> None:
        step(4, total, "Скачиваю модель")
        repo, _, filename = self.spec.ref.partition("::")
        dest = self.models_dir / (filename or f"{repo.split('/')[-1]}.gguf")
        if dest.exists() and dest.stat().st_size > 1_000_000:
            ok("Модель уже скачана.")
        else:
            url = f"https://huggingface.co/{repo}/resolve/main/{filename}"
            note(f"~{human_mb(self.spec.file_mb)} — можно заварить чай.")
            download(url, dest, "модель")
            ok("Модель скачана.")
        self.state["model_path"] = str(dest)

    def setup_wallet(self, total: int) -> None:
        step(5, total, "Кошелёк ноды")
        note("Кошелёк — это адрес, на который капают заработанные DVT.")
        wallet_path = self.home / "node.wallet.json"
        kp = load_or_create(wallet_path)
        ok(f"Адрес: {kp.address}")
        note("Файл кошелька хранится локально. Не удаляйте его — это ваши ключи.")
        self.state["wallet"] = str(wallet_path)
        self.state["address"] = kp.address

    def connect_network(self, total: int) -> None:
        step(6, total, "Подключаюсь к сети")
        seed = self.seed or ask("Адрес любой живой ноды сети (seed)",
                                 "http://10.0.0.223:9100")
        genesis_path = self.home / "genesis.json"
        try:
            resp = self.client.get(f"{seed.rstrip('/')}/genesis", timeout=10.0)
            resp.raise_for_status()
            Genesis.from_dict(resp.json()).save(genesis_path)
            ok(f"Подключился к сети «{resp.json()['params']['chain_id']}» через {seed}")
            self.state["seed"] = seed
            self.state["genesis"] = str(genesis_path)
        except httpx.HTTPError as exc:
            warn(f"Не достучался до сети ({exc}).")
            note("Проверьте адрес seed-ноды и что она включена, затем запустите снова.")
            self.state["seed"] = seed
            self.state["genesis"] = ""

    def set_price(self, total: int) -> None:
        step(7, total, "Цена за работу")
        watts = self.device.vram_mb and 130 or 90
        report = price_report(watts=float(watts), tokens_per_sec=30.0)
        note(f"По среднемировой цене электричества + 50% сервиса выходит "
             f"~${report.price_usd_per_million}/млн токенов.")
        rec = report.suggested_price_udvt
        if ask_yes(f"Поставить рекомендованную цену {rec} udvt/токен?", default=True):
            price = rec
        else:
            try:
                price = int(ask("Своя цена (udvt/токен)", str(rec)))
            except ValueError:
                price = rec
        ok(f"Цена: {price} udvt за токен.")
        self.state["price"] = price

    def write_launcher(self) -> Path:
        """Emit a start script so future launches are one command."""
        s = self.state
        model = s.get("model", "")
        server = s.get("server", "")
        model_path = s.get("model_path", "")
        port = 9100
        if os.name == "nt":
            script = self.home / "start-node.bat"
            body = (
                "@echo off\r\n"
                f'start \"llama-server\" \"{server}\" -m \"{model_path}\" '
                f"--host 127.0.0.1 --port 8085 -ngl 99 -c 8192\r\n"
                "timeout /t 8 >nul\r\n"
                f'python -m deltav.cli node --genesis \"{s.get("genesis","")}\" '
                f'--wallet \"{s.get("wallet","")}\" --host 0.0.0.0 --port {port} '
                f'--backend llamaserver --model \"{model}\" '
                f'--data-dir \"{self.home / "data"}\" --price {s.get("price",0)} '
                f'--peer {s.get("seed","")}\r\n'
            )
        else:
            script = self.home / "start-node.sh"
            body = (
                "#!/bin/sh\n"
                f'\"{server}\" -m \"{model_path}\" --host 127.0.0.1 --port 8085 '
                "-ngl 99 -c 8192 &\n"
                "sleep 8\n"
                f'python -m deltav.cli node --genesis \"{s.get("genesis","")}\" '
                f'--wallet \"{s.get("wallet","")}\" --host 0.0.0.0 --port {port} '
                f'--backend llamaserver --model \"{model}\" '
                f'--data-dir \"{self.home / "data"}\" --price {s.get("price",0)} '
                f'--peer {s.get("seed","")}\n'
            )
        script.write_text(body, encoding="utf-8")
        if os.name != "nt":
            os.chmod(script, 0o755)
        return script

    def launch(self, total: int, auto_start: bool = True) -> None:
        step(8, total, "Запускаю ноду")
        script = self.write_launcher()
        s = self.state
        ready = s.get("server") and s.get("model_path") and s.get("genesis")
        if not ready:
            warn("Не все шаги завершились — доделайте их и запустите снова.")
            note(f"Скрипт запуска сохранён: {script}")
            return
        if not auto_start:
            ok("Всё готово. Запустите ноду командой:")
            say(f"    {script}")
            return

        note("Поднимаю движок и ноду…")
        llama = subprocess.Popen(
            [s["server"], "-m", s["model_path"], "--host", "127.0.0.1",
             "--port", "8085", "-ngl", "99", "-c", "8192"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not self._wait_health("http://127.0.0.1:8085/health", 180):
            warn("Движок не поднялся вовремя. Запустите скрипт вручную и посмотрите вывод.")
            llama.terminate()
            return
        ok("Движок отвечает.")

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
            warn("Нода запускается дольше обычного — проверьте эксплорер через минуту.")
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
        addr = self.state.get("address", "")
        say()
        say(f"{C_OK}╔══════════════════════════════════════════╗{C_RESET}")
        say(f"{C_OK}║   Нода запущена и в сети! 🎉              ║{C_RESET}")
        say(f"{C_OK}╚══════════════════════════════════════════╝{C_RESET}")
        say()
        ok("Панель ноды (откройте в браузере):")
        say(f"    http://<этот-компьютер>:9100/explorer")
        ok("Ваш адрес для заработка:")
        say(f"    {addr}")
        say()
        note(f"В следующий раз просто запустите: {script}")
        note("Чтобы остановить — закройте окна движка и ноды.")

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
            warn("Прервано. Прогресс сохранён — запустите снова, чтобы продолжить.")
        finally:
            self._save_state()
        return self.state


def run_setup(home: str = "", seed: str = "", auto_start: bool = True) -> int:
    wizard = SetupWizard(home=home or None, seed=seed)
    wizard.run(auto_start=auto_start)
    return 0
