"""Unit tests for flybrain.ops (gpu_guard, qwen, scripts/gpu_window.sh).

No GPU, no sudo, no network: pynvml, torch and subprocess are replaced by fakes.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from flybrain.ops import gpu_guard as gg
from flybrain.ops import qwen

REPO = Path(__file__).resolve().parents[1]
GiB, MiB = gg.GiB, gg.MiB
OWN = os.getpid()
VLLM = 3918  # the root-owned worker seen on node1 card 0


# ---------------------------------------------------------------- fakes

class FakeNvml:
    """Minimal pynvml stand-in: one or more cards, memory + process list mutable by the test."""

    class NVMLError(Exception):
        pass

    def __init__(self, free=2600 * MiB, total=24 * GiB, procs=None, uuids=("GPU-aaaa-1111",)):
        self.free, self.total = free, total
        self.procs = {VLLM: 21514 * MiB} if procs is None else dict(procs)
        self.uuids = list(uuids)
        self.inits = 0
        self.fail_memory = False
        self.report_own = True

    def nvmlInit(self):
        self.inits += 1

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self.uuids)

    def nvmlDeviceGetHandleByIndex(self, i):
        return ("handle", i)

    def nvmlDeviceGetUUID(self, h):
        return self.uuids[h[1]]

    def nvmlDeviceGetMemoryInfo(self, h):
        if self.fail_memory:
            raise self.NVMLError("nvml down")
        return SimpleNamespace(free=self.free, used=self.total - self.free, total=self.total)

    def nvmlDeviceGetComputeRunningProcesses(self, h):
        return [SimpleNamespace(pid=p, usedGpuMemory=(None if (p == OWN and not self.report_own) else b))
                for p, b in self.procs.items()]

    def nvmlDeviceGetGraphicsRunningProcesses(self, h):
        return []


def fake_torch(free=3 * GiB, total=24 * GiB, uuid=None, reserved=0, available=True):
    calls: list = []
    cuda = SimpleNamespace(
        is_available=lambda: available,
        is_initialized=lambda: False,
        mem_get_info=lambda d=0: calls.append(("mem_get_info", d)) or (free, total),
        set_per_process_memory_fraction=lambda f, d=0: calls.append(("fraction", f, d)),
        get_device_properties=lambda d=0: SimpleNamespace(uuid=uuid),
        memory_reserved=lambda d=0: reserved,
        empty_cache=lambda: calls.append(("empty_cache",)),
        synchronize=lambda d=0: None,
    )
    return SimpleNamespace(cuda=cuda, calls=calls)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


ME = os.getuid() if hasattr(os, "getuid") else 1000
TEAMMATE = 40716  # same account, someone else's benchmark (seen on node1 with 7.2 GB)


def fake_uid(pid):
    return {VLLM: 0, 4000: 1001}.get(pid, ME)


def make_guard(nvml, exits, **kw):
    kw.setdefault("nvml_index", 0)
    kw.setdefault("pid_uid", fake_uid)
    return gg.GpuGuard(nvml=nvml, exit_fn=lambda c: exits.append((c, time.monotonic())), **kw)


# ---------------------------------------------------------------- gpu_guard: preflight

def test_preflight_refuses_on_nvml_before_touching_torch():
    nv = FakeNvml(free=1 * GiB)
    t = fake_torch()
    with pytest.raises(gg.GpuGuardError, match="no CUDA context"):
        gg.preflight(t, nvml=nv)
    assert t.calls == []  # torch.cuda.mem_get_info (context creation) never called


def test_preflight_refuses_when_torch_says_low():
    t = fake_torch(free=1 * GiB)
    with pytest.raises(gg.GpuGuardError, match="need 2.20"):
        gg.preflight(t, nvml=FakeNvml(free=3 * GiB))
    assert not [c for c in t.calls if c[0] == "fraction"]


def test_preflight_sets_fraction_and_alloc_conf(monkeypatch):
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    t = fake_torch(free=3 * GiB, total=24 * GiB)
    info = gg.preflight(t, nvml=FakeNvml(free=3 * GiB), fraction_cap_gib=1.4)
    frac = [c for c in t.calls if c[0] == "fraction"][0]
    assert frac[1] == pytest.approx(1.4 / 24.0) and frac[2] == 0
    assert info["fraction"] == pytest.approx(1.4 / 24.0)
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == gg.ALLOC_CONF_DEFAULT


def test_preflight_keeps_existing_alloc_conf(monkeypatch):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    gg.preflight(fake_torch(), nvml=FakeNvml(free=3 * GiB))
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_guard_gpu_pins_device_and_starts_thread(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    exits: list = []
    g = gg.guard_gpu(torch_mod=fake_torch(), nvml=FakeNvml(free=3 * GiB), poll_s=0.01,
                     pid_uid=fake_uid, exit_fn=lambda c: exits.append(c))
    try:
        assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        assert g._thread is not None and g._thread.is_alive()
        time.sleep(0.1)
        assert g.polls >= 2 and exits == []
        assert g.others_at_start == frozenset({VLLM})
    finally:
        g.close()
    assert g._thread is None


def test_guard_gpu_refuses_and_starts_nothing():
    with pytest.raises(gg.GpuGuardError):
        gg.guard_gpu(torch_mod=fake_torch(), nvml=FakeNvml(free=2 * GiB), visible_devices=None)


# ---------------------------------------------------------------- gpu_guard: index resolution

def test_nvml_index_by_uuid_then_visible_devices(monkeypatch):
    nv = FakeNvml(uuids=("GPU-aaaa", "GPU-bbbb"))
    assert gg.resolve_nvml_index(nv, 0, fake_torch(uuid="bbbb")) == 1
    assert gg.resolve_nvml_index(nv, 0, fake_torch(uuid="GPU-AAAA")) == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert gg.resolve_nvml_index(nv, 0, fake_torch(uuid=None)) == 1  # no uuid match -> env
    assert gg.resolve_nvml_index(nv, 0, None) == 1
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-bbbb")
    assert gg.resolve_nvml_index(nv, 0, None) == 0  # non-numeric env -> plain index


# ---------------------------------------------------------------- gpu_guard: rules

def test_own_cap_fires_exit_3():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, OWN: 1200 * MiB})
    exits: list = []
    g = make_guard(nv, exits, max_own_gib=1.6)
    assert g.check_once() is None
    nv.procs[OWN] = int(1.7 * GiB)
    assert g.check_once() == "own_cap"
    assert [c for c, _ in exits] == [gg.EXIT_LIMIT]


def test_card_free_floor_fires_exit_3():
    nv = FakeNvml(free=2600 * MiB)
    exits: list = []
    g = make_guard(nv, exits, min_card_free_mib=500)
    nv.free = 400 * MiB
    assert g.check_once() == "card_free"
    assert [c for c, _ in exits] == [gg.EXIT_LIMIT]


def test_vanish_checkpoints_then_exits_4():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, OWN: 800 * MiB})
    exits: list = []
    order: list = []
    g = make_guard(nv, exits, checkpoint_cb=lambda: order.append("ckpt"))
    assert g.others_at_start == frozenset({VLLM})
    assert g.check_once() is None
    del nv.procs[VLLM]
    nv.free = 23 * GiB
    assert g.check_once() == "vanished"
    assert order == ["ckpt"] and [c for c, _ in exits] == [gg.EXIT_RELEASED]


def test_vanish_requires_all_others_gone():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, 4000: 100 * MiB})
    exits: list = []
    g = make_guard(nv, exits)
    del nv.procs[VLLM]
    assert g.check_once() is None and exits == []
    del nv.procs[4000]
    assert g.check_once() == "vanished"


def test_vanish_ignores_same_account_processes():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, TEAMMATE: 7248 * MiB, OWN: 300 * MiB})
    exits: list = []
    g = make_guard(nv, exits)
    assert g.others_at_start == frozenset({VLLM})  # teammate's job is ours by uid, not tracked
    del nv.procs[TEAMMATE]
    assert g.check_once() is None and exits == []  # a teammate finishing must not release us
    del nv.procs[VLLM]
    assert g.check_once() == "vanished"


def test_vanish_rule_disabled_when_only_same_account_present():
    nv = FakeNvml(free=15 * GiB, procs={TEAMMATE: 7248 * MiB, OWN: 300 * MiB})
    exits: list = []
    g = make_guard(nv, exits)
    assert g.others_at_start == frozenset()
    nv.procs = {OWN: 300 * MiB}
    assert g.check_once() is None and exits == []


def test_vanish_rule_disabled_when_alone_at_start():
    nv = FakeNvml(free=23 * GiB, procs={OWN: 100 * MiB})
    exits: list = []
    g = make_guard(nv, exits)
    assert g.others_at_start == frozenset()
    nv.procs = {}
    assert g.check_once() is None and exits == []


def test_vanish_exit_deadline_beats_slow_checkpoint():
    nv = FakeNvml(procs={VLLM: 21514 * MiB})
    exits: list = []
    g = make_guard(nv, exits, checkpoint_cb=lambda: time.sleep(1.0), vanish_grace_s=0.2)
    nv.procs = {}
    t0 = time.monotonic()
    g.check_once()
    assert exits and exits[0][0] == gg.EXIT_RELEASED
    assert exits[0][1] - t0 < 0.7  # the deadline timer fired, not the callback's return


def test_checkpoint_exception_does_not_block_exit():
    nv = FakeNvml()
    exits: list = []

    def bad():
        raise RuntimeError("disk full")

    g = make_guard(nv, exits, checkpoint_cb=bad)
    nv.procs = {}
    assert g.check_once() == "vanished" and [c for c, _ in exits] == [gg.EXIT_RELEASED]


def test_blind_nvml_eventually_exits():
    nv = FakeNvml()
    exits: list = []
    g = make_guard(nv, exits, max_blind_polls=3)
    nv.fail_memory = True
    assert g.check_once() is None and g.check_once() is None
    assert g.check_once() == "blind" and [c for c, _ in exits] == [gg.EXIT_LIMIT]


def test_unreported_own_memory_falls_back_to_torch_reserved():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, OWN: 0})
    nv.report_own = False
    exits: list = []
    t = fake_torch(reserved=int(1.7 * GiB))
    g = make_guard(nv, exits, torch_mod=t)
    snap = g.snapshot()
    assert snap.own_reported is False and snap.own_b == int(1.7 * GiB)
    assert g.check_once() == "own_cap"


def test_idle_release_empties_cache_and_optionally_exits():
    nv = FakeNvml()
    clock = FakeClock()
    exits: list = []
    t = fake_torch()
    g = make_guard(nv, exits, torch_mod=t, clock=clock)
    g.release_when_idle(idle_s=10.0)
    clock.t = 5.0
    assert g.check_once() is None
    clock.t = 11.0
    assert g.check_once() == "idle"
    assert ("empty_cache",) in t.calls and exits == []
    ckpt: list = []
    g.checkpoint_cb = lambda: ckpt.append(1)
    g.release_when_idle(idle_s=10.0, exit_after=True)
    clock.t = 30.0
    assert g.check_once() == "idle" and ckpt == [1] and [c for c, _ in exits] == [gg.EXIT_IDLE]


def test_thread_polls_and_stops_cleanly():
    nv = FakeNvml()
    exits: list = []
    g = make_guard(nv, exits, poll_s=0.01).start()
    time.sleep(0.15)
    g.stop()
    n = g.polls
    assert n >= 3 and exits == []
    time.sleep(0.05)
    assert g.polls == n  # no polls after stop
    assert threading.active_count() >= 1


# ---------------------------------------------------------------- gpu_guard: standalone watchdog + status

def test_watch_pid_kills_on_cap_breach():
    pid = 777
    nv = FakeNvml(procs={VLLM: 21514 * MiB, pid: int(2 * GiB)})
    kills: list = []
    rc = gg.watch_pid(pid, nvml=nv, kill=lambda p, s: kills.append((p, s)), alive=lambda p: True,
                      sleep=lambda s: None, pid_uid=fake_uid)
    assert rc == gg.EXIT_LIMIT and kills == [(pid, signal.SIGKILL)]


def test_watch_pid_term_then_kill_when_tenants_vanish():
    pid = 777
    nv = FakeNvml(procs={VLLM: 21514 * MiB, pid: 100 * MiB})
    kills: list = []
    state = {"n": 0}

    def alive(p):
        return True

    def sleep(s):
        state["n"] += 1
        if state["n"] == 1:  # after the first poll the vLLM worker is gone
            del nv.procs[VLLM]

    rc = gg.watch_pid(pid, nvml=nv, kill=lambda p, s: kills.append((p, s)), alive=alive, sleep=sleep,
                      grace_s=0.05, pid_uid=fake_uid)
    assert rc == gg.EXIT_RELEASED
    assert kills == [(pid, signal.SIGTERM), (pid, signal.SIGKILL)]


def test_watch_pid_returns_0_when_process_exits():
    nv = FakeNvml(procs={VLLM: 21514 * MiB, 777: 100 * MiB})
    seen = {"n": 0}

    def alive(p):
        seen["n"] += 1
        return seen["n"] < 3

    rc = gg.watch_pid(777, nvml=nv, kill=lambda p, s: pytest.fail("no kill"), alive=alive, sleep=lambda s: None,
                      pid_uid=fake_uid)
    assert rc == 0


def test_status_reports_launchability():
    s = gg.status(0, nvml=FakeNvml(free=2 * GiB))
    assert s["launch_ok"] is False and s["processes"][0]["pid"] == VLLM
    assert gg.status(0, nvml=FakeNvml(free=3 * GiB))["launch_ok"] is True


# ---------------------------------------------------------------- qwen: fakes

class FakeShell:
    """Replaces qwen._run. Records every command; answers nvidia-smi/systemctl/sudo."""

    def __init__(self, sudo_ok=False, listing=None, free_seq=(2600,), total=24576, apps=""):
        self.calls: list[list[str]] = []
        self.sudo_ok = sudo_ok
        self.listing = MASTER_LISTING if listing is None else listing
        self.free_seq, self.total, self.apps = list(free_seq), total, apps
        self.sudo_rc: dict[tuple[str, ...], int] = {}

    def __call__(self, cmd, timeout=30.0):
        cmd = list(cmd)
        self.calls.append(cmd)
        rc, out = 0, ""
        if cmd[:3] == ["sudo", "-n", "true"]:
            rc = 0 if self.sudo_ok else 1
        elif cmd == ["sudo", "-n", "-l"]:
            rc, out = 0, self.listing
        elif cmd[:3] == ["sudo", "-n", "-l"]:
            rc, out = 0, cmd[4] + "\n"  # measured on master: exit 0 even when a password is needed
        elif cmd[:2] == ["sudo", "-n"]:
            rc = self.sudo_rc.get(tuple(cmd), 0)
        elif cmd[0] == "nvidia-smi" and cmd[1].startswith("--query-gpu"):
            free = self.free_seq.pop(0) if len(self.free_seq) > 1 else self.free_seq[0]
            out = f"{free}, {self.total - free}, {self.total}\n"
        elif cmd[0] == "nvidia-smi" and cmd[1].startswith("--query-compute-apps"):
            out = self.apps
        elif cmd[0] == qwen.SYSTEMCTL and cmd[1] == "is-active":
            out = "active\n"
        else:
            rc, out = 127, ""
        return CompletedProcess(cmd, rc, out, "" if rc == 0 else "sudo: a password is required\n")

    def privileged(self):
        return [c for c in self.calls if c[:2] == ["sudo", "-n"] and c[2] == qwen.SYSTEMCTL]


@pytest.fixture
def shell(monkeypatch):
    sh = FakeShell()
    monkeypatch.setattr(qwen, "_run", sh)
    clock = FakeClock()
    monkeypatch.setattr(qwen, "_clock", clock)
    monkeypatch.setattr(qwen, "_sleep", clock.sleep)
    monkeypatch.setattr(qwen, "_http_get", lambda url, timeout=3.0: (None, ""))
    sh.clock = clock
    return sh


def http_seq(monkeypatch, codes):
    seq = list(codes)

    def get(url, timeout=3.0):
        assert url == qwen.METRICS_URL
        code = seq.pop(0) if len(seq) > 1 else seq[0]
        return code, (METRICS_TEXT if code == 200 else "")

    monkeypatch.setattr(qwen, "_http_get", get)


# Verbatim shape of `sudo -n -l` on master (user has sudo WITH password + one unrelated NOPASSWD).
MASTER_LISTING = """Matching Defaults entries for unnati on master:
    env_reset, mail_badpass, secure_path=/usr/local/sbin\\:/usr/local/bin

User unnati may run the following commands on master:
    (ALL : ALL) ALL
    (root) NOPASSWD: /usr/bin/python3 /home/unnati/edge-tunnel-server.py *
"""
WHITELIST_LISTING = """User unnati may run the following commands on node1:
    (root) NOPASSWD: /bin/systemctl stop vllm-qwen.service, /bin/systemctl start vllm-qwen.service,
        /bin/systemctl stop agent-watchdog.timer, /bin/systemctl start agent-watchdog.timer
"""

METRICS_TEXT = """# HELP vllm:num_requests_running Number of requests in model execution batches.
vllm:num_requests_running{engine="0",model_name="qwen-27b-uncensored"} 2.0
vllm:num_requests_waiting{engine="0",model_name="qwen-27b-uncensored"} 1.0
vllm:num_requests_waiting_by_reason{engine="0",model_name="qwen-27b-uncensored",reason="capacity"} 0.0
vllm:gpu_cache_usage_perc{engine="0",model_name="qwen-27b-uncensored"} 0.25
"""


# ---------------------------------------------------------------- qwen: tests

def test_whitelist_is_exactly_the_four_commands():
    assert qwen.WHITELIST == frozenset({
        ("sudo", "-n", "/bin/systemctl", "stop", "vllm-qwen.service"),
        ("sudo", "-n", "/bin/systemctl", "start", "vllm-qwen.service"),
        ("sudo", "-n", "/bin/systemctl", "stop", "agent-watchdog.timer"),
        ("sudo", "-n", "/bin/systemctl", "start", "agent-watchdog.timer"),
    })
    assert qwen.METRICS_URL == "http://127.0.0.1:8000/metrics"


def test_sudo_helper_rejects_anything_else(shell):
    with pytest.raises(ValueError):
        qwen._sudo(("sudo", "-n", "/bin/systemctl", "restart", qwen.SERVICE))
    with pytest.raises(ValueError):
        qwen._sudo(("sudo", "/bin/systemctl", "stop", qwen.SERVICE))  # missing -n
    assert shell.calls == []


def test_can_sudo_probe_then_nopasswd_listing(shell):
    # master's real listing: sudo-with-password + an unrelated NOPASSWD entry -> refused
    assert qwen.can_sudo(qwen.CMD_STOP_SERVICE) is False
    assert ["sudo", "-n", "-l"] in shell.calls
    assert not [c for c in shell.calls if c[:4] == ["sudo", "-n", "-l", "--"]]  # never the per-command probe
    shell.listing = WHITELIST_LISTING
    assert qwen.can_sudo(qwen.CMD_STOP_SERVICE) is True
    assert qwen.can_sudo(qwen.CMD_START_TIMER) is True
    assert qwen.can_sudo(("sudo", "-n", "/bin/systemctl", "restart", qwen.SERVICE)) is False
    shell.sudo_ok = True
    assert qwen.can_sudo() is True
    assert all("-n" in c for c in shell.calls if c[0] == "sudo")


def test_nopasswd_listed_parser():
    cmd = qwen.CMD_STOP_SERVICE
    assert qwen.nopasswd_listed(MASTER_LISTING, cmd) is False          # (ALL : ALL) ALL needs a password
    assert qwen.nopasswd_listed(WHITELIST_LISTING, cmd) is True
    assert qwen.nopasswd_listed("    (root) NOPASSWD: ALL\n", cmd) is True
    assert qwen.nopasswd_listed("    (root) NOPASSWD: /bin/systemctl\n", cmd) is True
    assert qwen.nopasswd_listed("    (root) NOPASSWD: SETENV: /bin/systemctl stop vllm-qwen.service\n", cmd) is True
    assert qwen.nopasswd_listed("    (root) PASSWD: /bin/systemctl stop vllm-qwen.service\n", cmd) is False
    assert qwen.nopasswd_listed("    (root) NOPASSWD: /bin/systemctl stop other.service\n", cmd) is False
    assert qwen.nopasswd_listed("", cmd) is False


def test_stop_refuses_cleanly_without_sudo(shell):
    res = qwen.stop()
    assert res["ok"] is False and res["refused"] is True
    assert shell.privileged() == []
    assert "nothing was run" in res["reason"]


def test_stop_dry_run_runs_no_sudo_at_all(shell):
    res = qwen.stop(dry_run=True)
    assert res["ok"] and res["dry_run"]
    assert res["commands"] == ["sudo -n /bin/systemctl stop agent-watchdog.timer",
                               "sudo -n /bin/systemctl stop vllm-qwen.service"]
    assert not [c for c in shell.calls if c[0] == "sudo"]


def test_stop_order_and_memory_verification(shell):
    shell.sudo_ok = True
    shell.free_seq = [2600, 2600, 2600, 24000]
    res = qwen.stop(wait_s=60, poll_s=2)
    assert res["ok"] is True
    assert res["ran"] == ["sudo -n /bin/systemctl stop agent-watchdog.timer",
                          "sudo -n /bin/systemctl stop vllm-qwen.service"]
    assert res["free_mib_before"] == 2600 and res["free_mib_after"] == 24000
    assert res["metrics_after"]["up"] is False
    assert [c[3:] for c in shell.privileged()] == [["stop", qwen.TIMER], ["stop", qwen.SERVICE]]


def test_stop_reports_when_memory_not_freed(shell):
    shell.sudo_ok = True
    res = qwen.stop(wait_s=10, poll_s=2)
    assert res["ok"] is False and res["refused"] is False
    assert "not freed" in res["reason"] and shell.clock.t >= 10


def test_stop_service_failure_rearms_timer(shell):
    shell.sudo_ok = True
    shell.sudo_rc[qwen.CMD_STOP_SERVICE] = 1
    res = qwen.stop()
    assert res["ok"] is False
    assert res["ran"] == ["sudo -n /bin/systemctl stop agent-watchdog.timer",
                          "sudo -n /bin/systemctl stop vllm-qwen.service",
                          "sudo -n /bin/systemctl start agent-watchdog.timer"]


def test_stop_timer_failure_does_not_stop_service(shell):
    shell.sudo_ok = True
    shell.sudo_rc[qwen.CMD_STOP_TIMER] = 1
    res = qwen.stop()
    assert res["ok"] is False and res["ran"] == ["sudo -n /bin/systemctl stop agent-watchdog.timer"]


def test_start_waits_for_metrics_then_arms_timer(shell, monkeypatch):
    shell.sudo_ok = True
    http_seq(monkeypatch, [None, 503, 200])
    res = qwen.start(wait_s=300, poll_s=5)
    assert res["ok"] is True and res["healthy"] is True and res["timer_ok"] is True
    assert res["ran"] == ["sudo -n /bin/systemctl start vllm-qwen.service",
                          "sudo -n /bin/systemctl start agent-watchdog.timer"]
    assert shell.clock.t == pytest.approx(10.0)


def test_start_rearms_timer_even_if_unhealthy(shell, monkeypatch):
    shell.sudo_ok = True
    http_seq(monkeypatch, [None])
    res = qwen.start(wait_s=300, poll_s=5)
    assert res["ok"] is False and res["healthy"] is False
    assert res["ran"][-1] == "sudo -n /bin/systemctl start agent-watchdog.timer"
    assert shell.clock.t >= 300 and "not healthy" in res["reason"]


def test_start_refuses_without_sudo_and_dry_run(shell):
    assert qwen.start()["refused"] is True
    assert qwen.start(dry_run=True)["ok"] is True
    assert shell.privileged() == []


def test_parse_metrics_and_status_readonly(shell, monkeypatch):
    m = qwen.parse_metrics(METRICS_TEXT)
    assert m == {"num_requests_running": 2.0, "num_requests_waiting": 1.0,
                 "gpu_cache_usage_perc": 0.25, "model": "qwen-27b-uncensored"}
    http_seq(monkeypatch, [200])
    shell.apps = f"{VLLM}, 21514, VLLM::Worker_TP0, GPU-efd8\n{OWN}, 900, python, GPU-efd8\n"
    monkeypatch.setattr(qwen, "_pid_uid", lambda pid: 0 if pid == VLLM else os.getuid() if hasattr(os, "getuid") else 1000)
    s = qwen.status(probe_sudo=False)
    assert s["service"] == "active" and s["timer"] == "active"
    assert s["gpu"]["free_mib"] == 2600 and s["metrics"]["up"] and s["metrics"]["model"] == "qwen-27b-uncensored"
    assert s["own_gpu_pids"] == [OWN]
    assert not [c for c in shell.calls if c[0] == "sudo"]


def test_own_gpu_pids_by_proc_owner(monkeypatch):
    apps = [{"pid": 1, "used_mib": 1}, {"pid": 2, "used_mib": 1}, {"pid": 3, "used_mib": 1}]
    monkeypatch.setattr(qwen, "_pid_uid", lambda pid: {1: 0, 2: 1000, 3: None}[pid])
    assert qwen.own_gpu_pids(apps, uid=1000) == [2]


def test_compute_apps_parses_nvidia_smi(shell):
    shell.apps = "3918, 21514, VLLM::Worker_TP0, GPU-efd8\nbad line\n"
    apps = qwen.compute_apps()
    assert apps == [{"pid": 3918, "used_mib": 21514, "name": "VLLM::Worker_TP0", "gpu_uuid": "GPU-efd8"}]


def test_nvidia_smi_missing_is_none(monkeypatch):
    monkeypatch.setattr(qwen, "_run", lambda cmd, timeout=30.0: CompletedProcess(list(cmd), 127, "", "not found"))
    assert qwen.nvidia_smi_query() is None and qwen.compute_apps() == []
    assert qwen.unit_active(qwen.SERVICE) == "not found"


def test_real_run_never_prompts_and_maps_missing_binary():
    r = qwen._run(("definitely-not-a-binary-xyz",))
    assert r.returncode == 127


def test_cli_stop_defaults_to_dry_run_and_exit_codes(shell, capsys):
    assert qwen.main(["stop"]) == qwen.EXIT_OK
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "sudo -n /bin/systemctl stop vllm-qwen.service" in out
    assert not [c for c in shell.calls if c[0] == "sudo"]
    assert qwen.main(["stop", "--yes"]) == qwen.EXIT_REFUSED
    assert "REFUSED" in capsys.readouterr().out
    assert shell.privileged() == []
    assert qwen.main(["start", "--yes", "--json"]) == qwen.EXIT_REFUSED
    assert qwen.main(["status"]) == qwen.EXIT_OK
    assert "vllm-qwen.service: active" in capsys.readouterr().out
    assert qwen.main(["own-pids"]) == qwen.EXIT_OK


# ---------------------------------------------------------------- scripts/gpu_window.sh

BASH = shutil.which("bash")
SCRIPT = REPO / "scripts" / "gpu_window.sh"
needs_bash = pytest.mark.skipif(BASH is None or sys.platform.startswith("win"), reason="needs bash on a POSIX host")

FAKE_QWEN = r"""#!/bin/sh
# stands in for "python -m flybrain.ops.qwen <sub> ..." (argv: -m flybrain.ops.qwen <sub> ...)
sub="$3"
case "$sub" in
  status) echo "fake status"; exit 0 ;;
  stop) touch "$STATE_DIR/stopped"; echo "fake stop $4"; exit "${FAKE_STOP_RC:-0}" ;;
  start) touch "$STATE_DIR/started"; echo "fake start $4"; exit "${FAKE_START_RC:-0}" ;;
  own-pids)
    # list a sleeper only while it is really alive (a killed child of pytest lingers as a zombie)
    for p in $PRE_PID $NEW_PID; do
      if [ "$p" = "$NEW_PID" ] && [ ! -e "$STATE_DIR/stopped" ]; then continue; fi
      st="$(ps -o stat= -p "$p" 2>/dev/null)"
      case "$st" in ""|Z*) continue ;; esac
      echo "$p"
    done; exit 0 ;;
  *) echo "fake: unknown $sub" >&2; exit 9 ;;
esac
"""


@pytest.fixture
def fake_bin(tmp_path):
    """PATH dir with an instantly-failing nvidia-smi (master's real one takes 5 s to fail)."""
    b = tmp_path / "bin"
    b.mkdir()
    smi = b / "nvidia-smi"
    smi.write_text("#!/bin/sh\nexit 9\n")
    smi.chmod(0o755)
    return b


def run_script(*args, env_extra=None, timeout=60, path_prefix=None):
    env = dict(os.environ, PYTHON=sys.executable, PYTHONPATH=str(REPO), GPU_WINDOW_DRY_RUN="1")
    env.pop("FLYBRAIN_OUT", None)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}{os.pathsep}{env.get('PATH', '')}"
    env.update(env_extra or {})
    return subprocess.run([BASH, str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=timeout)


@needs_bash
def test_script_syntax():
    assert subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True).returncode == 0


@needs_bash
def test_script_dry_run_runs_command_then_restores(tmp_path, fake_bin):
    r = run_script("--dry-run", "--", "echo", "window-ok", env_extra={"FLYBRAIN_OUT": str(tmp_path)},
                   path_prefix=fake_bin)
    assert r.returncode == 0, r.stderr
    assert "window-ok" in r.stdout
    i_stop = r.stderr.index("closing window")
    i_cmd = r.stderr.index("command exited with status 0")
    i_start = r.stderr.index("restoring service")
    assert i_stop < i_cmd < i_start
    assert "--dry-run" in r.stderr and "--yes" not in r.stderr
    ledger = (tmp_path / "gpu_window.log").read_text()
    assert "window_open" in ledger and "restored" in ledger


@needs_bash
def test_script_failing_command_still_restores(fake_bin):
    r = run_script("--dry-run", "--", "sh", "-c", "exit 7", path_prefix=fake_bin)
    assert r.returncode == 7
    assert "restoring service" in r.stderr and "command exited with status 7" in r.stderr


@needs_bash
def test_script_usage_errors(fake_bin):
    assert run_script("--dry-run", path_prefix=fake_bin).returncode == 2
    assert run_script("--bogus", "--", "true", path_prefix=fake_bin).returncode == 2


@needs_bash
def test_script_live_mode_without_sudo_refuses_and_does_not_run_command(tmp_path, fake_bin):
    marker = tmp_path / "ran"
    r = run_script("--", "touch", str(marker), env_extra={"GPU_WINDOW_DRY_RUN": "0"}, path_prefix=fake_bin)
    # nvidia-smi present (fake), no whitelist: qwen stop --yes exits 3 -> script exits 2,
    # command never runs, no restore attempted.
    assert r.returncode == 2, r.stderr
    assert not marker.exists()
    assert "restoring service" not in r.stderr and "sudo -n refused" in r.stderr


def _spawn_sleeper():
    return subprocess.Popen(["sleep", "60"])


@pytest.fixture
def fake_qwen(tmp_path, fake_bin):
    """A fake `python` whose only job is to answer `-m flybrain.ops.qwen ...`; plus two sleepers:
    PRE (on the card before the window) and NEW (appears once the window is open)."""
    py = fake_bin / "python"
    py.write_text(FAKE_QWEN)
    py.chmod(0o755)
    state = tmp_path / "state"
    state.mkdir()
    pre, new = _spawn_sleeper(), _spawn_sleeper()
    env = {"PYTHON": str(py), "STATE_DIR": str(state), "PRE_PID": str(pre.pid), "NEW_PID": str(new.pid),
           "GPU_WINDOW_DRY_RUN": "0"}
    yield SimpleNamespace(env=env, state=state, pre=pre, new=new, bin=fake_bin)
    for p in (pre, new):
        if p.poll() is None:
            p.kill()
        p.wait()


@needs_bash
def test_script_live_kills_new_gpu_procs_spares_preexisting(fake_qwen):
    r = run_script("--", "sh", "-c", "exit 0", env_extra=fake_qwen.env, path_prefix=fake_qwen.bin)
    assert r.returncode == 0, r.stderr
    assert (fake_qwen.state / "stopped").exists() and (fake_qwen.state / "started").exists()
    time.sleep(0.2)
    assert fake_qwen.new.poll() is not None, "process that appeared during the window must be killed"
    assert fake_qwen.pre.poll() is None, "pre-existing same-account process must be spared"
    assert "leaving our pre-existing GPU processes alone" in r.stderr
    assert "--yes" in r.stderr


@needs_bash
def test_script_live_kill_all_env_kills_preexisting_too(fake_qwen):
    r = run_script("--", "true", env_extra=dict(fake_qwen.env, GPU_WINDOW_KILL_ALL="1"), path_prefix=fake_qwen.bin)
    assert r.returncode == 0, r.stderr
    time.sleep(0.2)
    assert fake_qwen.new.poll() is not None and fake_qwen.pre.poll() is not None


@needs_bash
def test_script_dry_run_never_signals(fake_qwen):
    r = run_script("--dry-run", "--", "true", env_extra=dict(fake_qwen.env, GPU_WINDOW_DRY_RUN="1"),
                   path_prefix=fake_qwen.bin)
    assert r.returncode == 0, r.stderr
    time.sleep(0.2)
    assert fake_qwen.new.poll() is None and fake_qwen.pre.poll() is None
    assert "would send TERM" in r.stderr


@needs_bash
def test_script_restore_failure_overrides_zero_status(fake_qwen):
    r = run_script("--", "true", env_extra=dict(fake_qwen.env, FAKE_START_RC="1"), path_prefix=fake_qwen.bin)
    assert r.returncode == 1 and "restart reported failure" in r.stderr
    r = run_script("--", "sh", "-c", "exit 5", env_extra=dict(fake_qwen.env, FAKE_START_RC="1"),
                   path_prefix=fake_qwen.bin)
    assert r.returncode == 5  # the command's own failure is kept


@needs_bash
def test_script_stop_outcomes(fake_qwen, tmp_path):
    marker = tmp_path / "ran"
    r = run_script("--", "touch", str(marker), env_extra=dict(fake_qwen.env, FAKE_STOP_RC="3"), path_prefix=fake_qwen.bin)
    assert r.returncode == 2 and not marker.exists() and not (fake_qwen.state / "started").exists()
    r = run_script("--", "touch", str(marker), env_extra=dict(fake_qwen.env, FAKE_STOP_RC="1"), path_prefix=fake_qwen.bin)
    assert r.returncode == 2 and not marker.exists() and (fake_qwen.state / "started").exists()


@needs_bash
def test_script_interrupt_kills_command_and_restores(fake_qwen):
    p = subprocess.Popen([BASH, str(SCRIPT), "--", "sleep", "30"],
                         env=dict(os.environ, PYTHONPATH=str(REPO), PATH=f"{fake_qwen.bin}{os.pathsep}{os.environ['PATH']}",
                                  **fake_qwen.env),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (fake_qwen.state / "stopped").exists():
        time.sleep(0.05)
    time.sleep(0.3)
    p.send_signal(signal.SIGTERM)
    _, err = p.communicate(timeout=30)
    assert p.returncode == 130, err
    assert "interrupted" in err and "restoring service" in err and (fake_qwen.state / "started").exists()
    assert not subprocess.run(["pgrep", "-f", "^sleep 30$"], capture_output=True).stdout.strip(), "sleep 30 must be gone"
