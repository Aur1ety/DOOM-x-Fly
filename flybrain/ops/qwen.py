"""Stop/start/status for the root-owned vLLM service on node1 (a public "GPU window").

Only these exact commands are ever run under sudo (the ones an admin may whitelist in
sudoers with NOPASSWD)::

    sudo -n /bin/systemctl stop  vllm-qwen.service
    sudo -n /bin/systemctl start vllm-qwen.service
    sudo -n /bin/systemctl stop  agent-watchdog.timer
    sudo -n /bin/systemctl start agent-watchdog.timer

Every action first probes `sudo -n true` and, failing that, parses `sudo -n -l` for a
`NOPASSWD:` entry naming the exact command; it refuses cleanly if neither holds. No password
prompt can ever appear (`-n` everywhere, stdin is /dev/null). Measured on master: `sudo -l
<cmd>` exits 0 when the command is merely *permitted with a password*, so that probe is not
used. stop() verifies with nvidia-smi that memory was actually freed; start() waits for
http://127.0.0.1:8000/metrics to answer 200 (up to 5 min) and re-arms the timer no matter
what. status() is read-only (nvidia-smi, unprivileged `systemctl is-active`, one GET of
/metrics).

CLI::

    python -m flybrain.ops.qwen status [--json]
    python -m flybrain.ops.qwen stop|start [--dry-run | --yes]     # default is --dry-run
    python -m flybrain.ops.qwen own-pids                            # our PIDs on the GPUs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Iterable

SERVICE = "vllm-qwen.service"
TIMER = "agent-watchdog.timer"
SYSTEMCTL = "/bin/systemctl"
METRICS_URL = "http://127.0.0.1:8000/metrics"

CMD_STOP_SERVICE = ("sudo", "-n", SYSTEMCTL, "stop", SERVICE)
CMD_START_SERVICE = ("sudo", "-n", SYSTEMCTL, "start", SERVICE)
CMD_STOP_TIMER = ("sudo", "-n", SYSTEMCTL, "stop", TIMER)
CMD_START_TIMER = ("sudo", "-n", SYSTEMCTL, "start", TIMER)
CMD_SUDO_PROBE = ("sudo", "-n", "true")
WHITELIST = frozenset({CMD_STOP_SERVICE, CMD_START_SERVICE, CMD_STOP_TIMER, CMD_START_TIMER})

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 3  # nothing was run (sudo not whitelisted); no cleanup needed


# -- seams (monkeypatched in tests) ------------------------------------------------------

def _run(cmd: Iterable[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    """subprocess.run with stdin=/dev/null; never raises, maps missing binary to rc 127."""
    cmd = list(cmd)
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as e:
        return subprocess.CompletedProcess(cmd, 127, "", f"{cmd[0]}: not found ({e})")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"{cmd[0]}: timeout after {timeout}s")


def _http_get(url: str, timeout: float = 3.0) -> tuple[int | None, str]:
    """(status, body). status None when the endpoint is unreachable."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed localhost URL
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, OSError, ValueError):
        return None, ""


_sleep = time.sleep
_clock = time.monotonic


# -- read-only probes ------------------------------------------------------------------------

def _sudo(cmd: tuple[str, ...]) -> subprocess.CompletedProcess:
    if tuple(cmd) not in WHITELIST:
        raise ValueError(f"not a whitelisted command: {cmd}")
    return _run(cmd, timeout=120.0)


_NOPASSWD_RE = re.compile(r"NOPASSWD:\s*(.*)$")
_TAG_RE = re.compile(r"^(?:[A-Z]+:\s*)+")  # strips further tags like "SETENV: "


def nopasswd_listed(listing: str, cmd: tuple[str, ...]) -> bool:
    """True if `sudo -l` output lists `cmd` (exact, or its binary alone, or ALL) on a
    NOPASSWD line. Entries on lines without NOPASSWD need a password and do not count."""
    want = " ".join(cmd[2:])
    # sudo wraps long command lists onto indented continuation lines that do not start with "(".
    lines: list[str] = []
    for raw in listing.splitlines():
        if lines and raw[:1].isspace() and not raw.strip().startswith("("):
            lines[-1] += " " + raw.strip()
        else:
            lines.append(raw)
    for line in lines:
        m = _NOPASSWD_RE.search(line)
        if not m:
            continue
        for entry in m.group(1).split(","):
            e = _TAG_RE.sub("", entry.strip())
            if e in (want, cmd[2], "ALL"):
                return True
    return False


def can_sudo(cmd: tuple[str, ...] | None = None) -> bool:
    """True if `sudo -n true` works (blanket passwordless sudo) or, when `cmd` is given,
    `sudo -n -l` lists it under NOPASSWD. Neither probe executes anything."""
    if _run(CMD_SUDO_PROBE, timeout=10.0).returncode == 0:
        return True
    if cmd is None:
        return False
    r = _run(("sudo", "-n", "-l"), timeout=10.0)
    return r.returncode == 0 and nopasswd_listed(r.stdout, cmd)


def unit_active(unit: str) -> str:
    """`systemctl is-active <unit>` as an unprivileged user (no sudo)."""
    r = _run((SYSTEMCTL, "is-active", unit), timeout=10.0)
    return (r.stdout.strip() or r.stderr.strip() or f"rc={r.returncode}").splitlines()[0]


def nvidia_smi_query(gpu: int = 0) -> dict[str, Any] | None:
    """{'free_mib','used_mib','total_mib'} for one card, or None if nvidia-smi is unusable."""
    r = _run(("nvidia-smi", "--query-gpu=memory.free,memory.used,memory.total",
              "--format=csv,noheader,nounits", "-i", str(gpu)), timeout=15.0)
    if r.returncode != 0:
        return None
    try:
        free, used, total = (int(x.strip()) for x in r.stdout.strip().splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    return {"gpu": gpu, "free_mib": free, "used_mib": used, "total_mib": total}


def compute_apps() -> list[dict[str, Any]]:
    """All processes holding GPU memory (any user, all cards) from nvidia-smi; [] if unusable."""
    r = _run(("nvidia-smi", "--query-compute-apps=pid,used_memory,process_name,gpu_uuid",
              "--format=csv,noheader,nounits"), timeout=15.0)
    out: list[dict[str, Any]] = []
    if r.returncode != 0:
        return out
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        used = int(parts[1]) if parts[1].isdigit() else None
        out.append({"pid": int(parts[0]), "used_mib": used,
                    "name": parts[2] if len(parts) > 2 else "", "gpu_uuid": parts[3] if len(parts) > 3 else ""})
    return out


def _pid_uid(pid: int) -> int | None:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def own_gpu_pids(apps: list[dict[str, Any]] | None = None, uid: int | None = None) -> list[int]:
    """PIDs on the GPUs that belong to this user (by /proc owner)."""
    uid = os.getuid() if uid is None else uid  # type: ignore[attr-defined]
    apps = compute_apps() if apps is None else apps
    return sorted({a["pid"] for a in apps if _pid_uid(a["pid"]) == uid})


_GAUGE_RE = re.compile(r"^(vllm:num_requests_running|vllm:num_requests_waiting|vllm:gpu_cache_usage_perc)"
                       r"\{([^}]*)\}\s+([0-9.eE+-]+)\s*$", re.M)
_MODEL_RE = re.compile(r'model_name="([^"]*)"')


def parse_metrics(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, labels, value in _GAUGE_RE.findall(text):
        key = name.split(":", 1)[1]
        out[key] = out.get(key, 0.0) + float(value)
        m = _MODEL_RE.search(labels)
        if m and "model" not in out:
            out["model"] = m.group(1)
    return out


def metrics(timeout: float = 3.0) -> dict[str, Any]:
    """One read-only GET of the vLLM /metrics endpoint."""
    code, body = _http_get(METRICS_URL, timeout)
    d: dict[str, Any] = {"up": code == 200, "http": code}
    if code == 200:
        d.update(parse_metrics(body))
    return d


def status(gpu: int = 0, probe_sudo: bool = False) -> dict[str, Any]:
    """Read-only picture of the service. `probe_sudo` is off by default because every
    refused `sudo -n` is logged to the admin's auth.log."""
    d: dict[str, Any] = {
        "service": unit_active(SERVICE),
        "timer": unit_active(TIMER),
        "gpu": nvidia_smi_query(gpu),
        "metrics": metrics(),
        "own_gpu_pids": own_gpu_pids(),
    }
    if probe_sudo:
        d["sudo_ok"] = can_sudo(CMD_STOP_SERVICE)
    return d


# -- actions -----------------------------------------------------------------------------------

def _fmt(cmds: Iterable[tuple[str, ...]]) -> list[str]:
    return [" ".join(c) for c in cmds]


def stop(dry_run: bool = False, wait_s: float = 120.0, min_freed_gib: float = 8.0,
         gpu: int = 0, poll_s: float = 2.0) -> dict[str, Any]:
    """Stop the watchdog timer, then the service; wait until nvidia-smi shows >= min_freed_gib
    more free memory on `gpu` (or the card nearly empty). Refuses (ok=False, refused=True)
    without running anything if sudo -n is not available."""
    plan = [CMD_STOP_TIMER, CMD_STOP_SERVICE]
    before = nvidia_smi_query(gpu)
    res: dict[str, Any] = {"ok": False, "dry_run": dry_run, "refused": False,
                           "commands": _fmt(plan), "ran": [], "free_mib_before": before and before["free_mib"]}
    if dry_run:
        res["ok"] = True
        res["note"] = "dry run: nothing executed"
        return res
    if not can_sudo(CMD_STOP_SERVICE):
        res.update(refused=True, reason="sudo -n is not permitted for this user; nothing was run")
        return res
    r = _sudo(CMD_STOP_TIMER)
    res["ran"].append(" ".join(CMD_STOP_TIMER))
    if r.returncode != 0:
        res["reason"] = f"stop timer failed rc={r.returncode}: {r.stderr.strip()}"
        return res
    r = _sudo(CMD_STOP_SERVICE)
    res["ran"].append(" ".join(CMD_STOP_SERVICE))
    if r.returncode != 0:
        res["reason"] = f"stop service failed rc={r.returncode}: {r.stderr.strip()}"
        _sudo(CMD_START_TIMER)  # best effort: give the admin's watchdog back
        res["ran"].append(" ".join(CMD_START_TIMER))
        return res
    t0 = _clock()
    freed = False
    after = before
    while True:
        after = nvidia_smi_query(gpu)
        if after is not None:
            base = before["free_mib"] if before else 0
            nearly_empty = after["free_mib"] >= after["total_mib"] - 1024
            freed = (after["free_mib"] - base >= min_freed_gib * 1024) or nearly_empty
        if freed or _clock() - t0 >= wait_s:
            break
        _sleep(poll_s)
    res.update(ok=freed, free_mib_after=after and after["free_mib"], elapsed_s=round(_clock() - t0, 1),
               metrics_after=metrics(timeout=2.0))
    if not freed:
        res["reason"] = ("nvidia-smi unusable; cannot verify" if after is None else
                         f"memory not freed within {wait_s:.0f} s "
                         f"(free {after['free_mib']} MiB, was {res['free_mib_before']} MiB)")
    return res


def wait_healthy(wait_s: float = 300.0, poll_s: float = 5.0) -> bool:
    """Poll /metrics until it returns HTTP 200 or wait_s elapses."""
    t0 = _clock()
    while True:
        if _http_get(METRICS_URL, timeout=3.0)[0] == 200:
            return True
        if _clock() - t0 >= wait_s:
            return False
        _sleep(poll_s)


def start(dry_run: bool = False, wait_s: float = 300.0, poll_s: float = 5.0) -> dict[str, Any]:
    """Start the service, wait for /metrics (HTTP 200, up to wait_s), then start the timer.
    The timer is re-armed even if the service does not come up - the admin's watchdog is
    what keeps their service alive."""
    plan = [CMD_START_SERVICE, CMD_START_TIMER]
    res: dict[str, Any] = {"ok": False, "dry_run": dry_run, "refused": False,
                           "commands": _fmt(plan), "ran": [], "healthy": None}
    if dry_run:
        res["ok"] = True
        res["note"] = "dry run: nothing executed"
        return res
    if not can_sudo(CMD_START_SERVICE):
        res.update(refused=True, reason="sudo -n is not permitted for this user; nothing was run")
        return res
    t0 = _clock()
    r = _sudo(CMD_START_SERVICE)
    res["ran"].append(" ".join(CMD_START_SERVICE))
    healthy = False
    try:
        if r.returncode != 0:
            res["reason"] = f"start service failed rc={r.returncode}: {r.stderr.strip()}"
        else:
            healthy = wait_healthy(wait_s, poll_s)
            if not healthy:
                res["reason"] = f"/metrics not healthy within {wait_s:.0f} s"
    finally:
        rt = _sudo(CMD_START_TIMER)
        res["ran"].append(" ".join(CMD_START_TIMER))
        res["timer_ok"] = rt.returncode == 0
    res.update(healthy=healthy, ok=(r.returncode == 0 and healthy and res["timer_ok"]),
               elapsed_s=round(_clock() - t0, 1))
    return res


# -- CLI ------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status", help="read-only summary")
    p.add_argument("--json", action="store_true")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--probe-sudo", action="store_true", help="also probe sudo -n (logged in auth.log)")
    for name in ("stop", "start"):
        p = sub.add_parser(name)
        g = p.add_mutually_exclusive_group()
        g.add_argument("--dry-run", action="store_true", help="print the commands, run nothing (default)")
        g.add_argument("--yes", action="store_true", help="actually run the whitelisted sudo commands")
        p.add_argument("--wait", type=float, default=None, help="seconds to wait for verification")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--json", action="store_true")
    sub.add_parser("own-pids", help="print our PIDs holding GPU memory, one per line")
    p = sub.add_parser("wait-healthy", help="exit 0 once /metrics answers 200")
    p.add_argument("--wait", type=float, default=300.0)
    args = ap.parse_args(argv)

    if args.cmd == "status":
        s = status(args.gpu, probe_sudo=args.probe_sudo)
        if args.json:
            print(json.dumps(s, indent=1))
        else:
            g, m = s["gpu"], s["metrics"]
            print(f"{SERVICE}: {s['service']}   {TIMER}: {s['timer']}")
            print("gpu: " + ("nvidia-smi unavailable" if g is None else
                             f"free {g['free_mib']} MiB / {g['total_mib']} MiB"))
            print(f"metrics: http={m.get('http')} up={m['up']} model={m.get('model', '?')} "
                  f"running={m.get('num_requests_running', '?')} waiting={m.get('num_requests_waiting', '?')}")
            print(f"own gpu pids: {s['own_gpu_pids'] or 'none'}" + (f"   sudo_ok={s['sudo_ok']}" if "sudo_ok" in s else ""))
        return EXIT_OK
    if args.cmd == "own-pids":
        for pid in own_gpu_pids():
            print(pid)
        return EXIT_OK
    if args.cmd == "wait-healthy":
        ok = wait_healthy(args.wait)
        print("healthy" if ok else "not healthy")
        return EXIT_OK if ok else EXIT_FAILED

    dry = not args.yes
    if args.cmd == "stop":
        res = stop(dry_run=dry, **({"wait_s": args.wait} if args.wait is not None else {}), gpu=args.gpu)
    else:
        res = start(dry_run=dry, **({"wait_s": args.wait} if args.wait is not None else {}))
    if args.json:
        print(json.dumps(res, indent=1))
    else:
        tag = "DRY RUN" if dry else "RUN"
        print(f"[{tag}] {args.cmd}: " + " ; ".join(res["commands"]))
        for k in ("ran", "free_mib_before", "free_mib_after", "healthy", "timer_ok", "elapsed_s", "reason", "note"):
            if res.get(k) not in (None, [], ""):
                print(f"  {k}: {res[k]}")
        print("  ok" if res["ok"] else ("  REFUSED (nothing run)" if res["refused"] else "  FAILED"))
    if res["ok"]:
        return EXIT_OK
    return EXIT_REFUSED if res["refused"] else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
