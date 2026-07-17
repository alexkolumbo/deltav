"""Persistence + time-window scheduling for a node — so a machine can run a
Delta V node unattended (survive logout/reboot) and, optionally, only be
online during set hours and auto-start into that schedule.

Two orthogonal things:

* **auto-start (persistence)** — the node comes up on its own after a reboot
  or re-login and stays up. Without it the operator has to babysit a terminal;
  with it the box "just serves".
* **schedule (time windows)** — the node is only expected to be online during
  given daily hours (a home GPU that games in the evening, an office box that
  sleeps at night). Outside the window the node is stopped; at window-open it
  auto-starts. The router already routes around a node that's simply down, so
  a stopped node is correct behaviour, not a fault.

We generate the OS scheduler artifacts (Windows Task Scheduler via
PowerShell, Linux systemd *user* units) as pure strings — deterministic and
unit-testable — and `apply_*` shells out to install them. The GPU **engine**
needs the user's interactive desktop session for the driver, so on Windows the
tasks run with an interactive logon (they fire while the user is logged in,
even away); a pure-python node with no local engine can run headless.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

TASK_NODE = "DeltaVNode"          # always-on / window-open launcher
TASK_STOP = "DeltaVNode-Stop"     # window-close stopper


def _valid_hhmm(s: str) -> str:
    """Normalise 'H:MM' / 'HH:MM' -> 'HH:MM', rejecting anything else so a
    typo can't silently produce a task that never fires (or fires always)."""
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", s.strip())
    if not m:
        raise ValueError(f"time must be HH:MM, got {s!r}")
    h, mm = int(m.group(1)), int(m.group(2))
    if h > 23 or mm > 59:
        raise ValueError(f"time out of range: {s!r}")
    return f"{h:02d}:{mm:02d}"


@dataclass
class Window:
    """A daily online window [start, end). end <= start means it wraps past
    midnight (e.g. 22:00->06:00) — the node runs overnight."""
    start: str
    end: str

    def __post_init__(self) -> None:
        self.start = _valid_hhmm(self.start)
        self.end = _valid_hhmm(self.end)

    @property
    def wraps_midnight(self) -> bool:
        return self.end <= self.start


@dataclass
class Schedule:
    """`window is None` -> always-on (24/7). Otherwise the node is online only
    during the window and stopped outside it."""
    window: Window | None = None
    days: str = "DAILY"           # schtasks /D value or 'DAILY'

    @classmethod
    def always(cls) -> "Schedule":
        return cls(window=None)

    @classmethod
    def daily(cls, start: str, end: str) -> "Schedule":
        return cls(window=Window(start, end))


# ------------------------------------------------------------------- Windows
def render_windows_ps(*, python: str, launcher: str, stopper: str,
                      node_argline: str, home: str,
                      schedule: Schedule, start_now: bool = True,
                      task_node: str = TASK_NODE,
                      task_stop: str = TASK_STOP) -> str:
    """PowerShell that (re)registers the scheduler entries for `schedule`.

    Always-on: one task that runs the node at logon and never times out, with
    crash-restart. Windowed: additionally a daily start trigger at window-open
    and a companion Stop task at window-close. Interactive logon type needs no
    stored password and gives the GPU engine a real desktop session.
    """
    triggers = ['$t_logon = New-ScheduledTaskTrigger -AtLogOn']
    start_refs = ['$t_logon']
    if schedule.window is not None:
        triggers.append(
            f'$t_open  = New-ScheduledTaskTrigger -Daily -At {schedule.window.start}')
        start_refs.append('$t_open')
    triggers_block = "\n".join(triggers)
    start_trigger_list = ", ".join(start_refs)

    stop_block = ""
    if schedule.window is not None:
        # The stop task kills the node + engine at window-close.
        stop_block = f'''
Unregister-ScheduledTask -TaskName "{task_stop}" -Confirm:$false -ErrorAction SilentlyContinue
$stopAction = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c "{stopper}"'
$stopTrig   = New-ScheduledTaskTrigger -Daily -At {schedule.window.end}
Register-ScheduledTask -TaskName "{task_stop}" -Action $stopAction -Trigger $stopTrig `
  -Principal $principal -Settings $settings | Out-Null
'''
    return f'''$ErrorActionPreference = "SilentlyContinue"
# --- Delta V node auto-start{" + schedule" if schedule.window else ""} ---
Unregister-ScheduledTask -TaskName "{task_node}" -Confirm:$false -ErrorAction SilentlyContinue
{triggers_block}
$action    = New-ScheduledTaskAction -Execute "cmd.exe" -Argument '/c "{launcher}"' -WorkingDirectory "{home}"
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
  -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) `
  -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "{task_node}" -Action $action -Trigger {start_trigger_list} `
  -Principal $principal -Settings $settings | Out-Null
{stop_block}{'Start-ScheduledTask -TaskName "' + task_node + '"' if start_now else ''}
Write-Output "installed:{task_node}"
'''


def render_windows_stopper(*, home: str, node_port: int = 9100,
                           engine_port: int = 8085) -> str:
    """A .bat that stops the node (and its local engine) — used at window-close
    and for a manual stop. Kills by listening port so it never touches an
    unrelated python/llama process."""
    return (
        "@echo off\r\n"
        "title DeltaV Stop\r\n"
        'echo [DeltaV] Stopping node + engine...\r\n'
        f'for /f "tokens=5" %%p in (\'netstat -ano ^| findstr ":{node_port} " ^| findstr LISTENING\') do taskkill /F /PID %%p >nul 2>&1\r\n'
        f'for /f "tokens=5" %%p in (\'netstat -ano ^| findstr ":{engine_port} " ^| findstr LISTENING\') do taskkill /F /PID %%p >nul 2>&1\r\n'
        "taskkill /F /IM llama-server.exe >nul 2>&1\r\n"
        "echo [DeltaV] Stopped.\r\n"
    )


# --------------------------------------------------------------------- Linux
def render_systemd_units(*, launcher: str, home: str, user: str,
                         schedule: Schedule,
                         unit: str = "deltav-node") -> dict[str, str]:
    """systemd *user* units. Always-on: a Service that Restarts=always. Windowed:
    the same Service plus a Timer that starts it at window-open and a companion
    stop timer at window-close (systemd has no native window, so two OnCalendar
    timers bracket it)."""
    service = f'''[Unit]
Description=Delta V inference node
After=network-online.target

[Service]
Type=simple
WorkingDirectory={home}
ExecStart=/bin/sh "{launcher}"
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
'''
    units = {f"{unit}.service": service}
    if schedule.window is not None:
        oc_start = _hhmm_to_oncalendar(schedule.window.start)
        oc_stop = _hhmm_to_oncalendar(schedule.window.end)
        units[f"{unit}-start.timer"] = _timer("start", unit, oc_start)
        units[f"{unit}-start.service"] = _oneshot(
            f"systemctl --user start {unit}.service", "start node")
        units[f"{unit}-stop.timer"] = _timer("stop", unit, oc_stop)
        units[f"{unit}-stop.service"] = _oneshot(
            f"systemctl --user stop {unit}.service", "stop node")
    return units


def _hhmm_to_oncalendar(hhmm: str) -> str:
    return f"*-*-* {hhmm}:00"


def _timer(kind: str, unit: str, oncalendar: str) -> str:
    return f'''[Unit]
Description=Delta V node {kind} timer

[Timer]
OnCalendar={oncalendar}
Persistent=true

[Install]
WantedBy=timers.target
'''


def _oneshot(cmd: str, desc: str) -> str:
    return f'''[Unit]
Description=Delta V {desc}

[Service]
Type=oneshot
ExecStart=/bin/sh -c "{cmd}"
'''


# ------------------------------------------------------------------- apply
def apply_windows(ps_script: str) -> tuple[bool, str]:
    """Run the generated PowerShell (base64 -EncodedCommand avoids quoting
    hell). Returns (ok, message)."""
    import base64
    b64 = base64.b64encode(ps_script.encode("utf-16-le")).decode()
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-EncodedCommand", b64],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    ok = "installed:" in (out.stdout or "")
    return ok, (out.stdout or "") + (out.stderr or "")


def write_linux_units(units: dict[str, str], unit_dir: Path) -> list[Path]:
    unit_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in units.items():
        p = unit_dir / name
        p.write_text(text, encoding="utf-8")
        written.append(p)
    return written
