"""Node persistence + time-window scheduling artifact generation."""
import pytest

from deltav.setup.schedule import (
    Schedule, Window, render_windows_ps, render_windows_stopper,
    render_systemd_units, TASK_NODE, TASK_STOP,
)


def test_window_validation_and_wrap():
    w = Window("9:5" if False else "09:05", "23:00")
    assert w.start == "09:05" and w.end == "23:00"
    assert not w.wraps_midnight
    assert Window("22:00", "06:00").wraps_midnight       # overnight
    for bad in ("24:00", "12:60", "noon", "9", "12:5:5"):
        with pytest.raises(ValueError):
            Window(bad, "10:00")


def test_windows_ps_always_on():
    ps = render_windows_ps(python="py.exe", launcher="C:\\n\\start.bat",
                           stopper="C:\\n\\stop.bat", node_argline="",
                           home="C:\\n", schedule=Schedule.always())
    assert f'"{TASK_NODE}"' in ps
    assert "-AtLogOn" in ps                     # comes up on login
    assert "ExecutionTimeLimit ([TimeSpan]::Zero)" in ps   # never auto-killed
    assert "RestartCount 5" in ps               # crash-restart
    assert "-Daily -At" not in ps               # no window trigger
    assert TASK_STOP not in ps                  # no stop task when 24/7
    assert "Start-ScheduledTask" in ps          # starts immediately


def test_windows_ps_windowed():
    ps = render_windows_ps(python="py.exe", launcher="C:\\n\\start.bat",
                           stopper="C:\\n\\stop.bat", node_argline="",
                           home="C:\\n", schedule=Schedule.daily("09:00", "23:00"))
    assert "-Daily -At 09:00" in ps             # auto-start at window open
    assert f'"{TASK_STOP}"' in ps               # companion stopper
    assert "-Daily -At 23:00" in ps             # stop at window close
    assert "stop.bat" in ps


def test_windows_ps_start_now_toggle():
    kw = dict(python="py", launcher="l", stopper="s", node_argline="",
              home="h", schedule=Schedule.always())
    assert "Start-ScheduledTask" in render_windows_ps(**kw, start_now=True)
    assert "Start-ScheduledTask" not in render_windows_ps(**kw, start_now=False)


def test_windows_stopper_targets_ports_not_all_python():
    bat = render_windows_stopper(home="C:\\n")
    assert ":9100" in bat and ":8085" in bat
    assert "taskkill /F /IM llama-server.exe" in bat
    # must NOT blanket-kill python (would take down unrelated processes)
    assert "/IM python.exe" not in bat


def test_systemd_units_always_on():
    units = render_systemd_units(launcher="/n/start.sh", home="/n", user="bob",
                                 schedule=Schedule.always())
    assert set(units) == {"deltav-node.service"}
    svc = units["deltav-node.service"]
    assert "Restart=always" in svc and "/n/start.sh" in svc


def test_systemd_units_windowed_has_timers():
    units = render_systemd_units(launcher="/n/start.sh", home="/n", user="bob",
                                 schedule=Schedule.daily("08:30", "22:15"))
    assert "deltav-node-start.timer" in units and "deltav-node-stop.timer" in units
    assert "OnCalendar=*-*-* 08:30:00" in units["deltav-node-start.timer"]
    assert "OnCalendar=*-*-* 22:15:00" in units["deltav-node-stop.timer"]
