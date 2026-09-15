"""GPU guard for a card we share with a root-owned vLLM service (node1, 2x A30).

Self-imposed limits (no root, no MIG): refuse to launch when the card is nearly full, cap
torch's caching allocator, and run an independent NVML watchdog thread that kills this
process (os._exit) the moment we exceed our share or the card runs out of memory. If the
other tenants' processes all vanish (vLLM restarting), we checkpoint and exit within a few
seconds so their restart finds the memory free (vllm-project/vllm#20305).

Usage inside a trainer, before any CUDA allocation::

    from flybrain.ops.gpu_guard import guard_gpu
    guard = guard_gpu(checkpoint_cb=save_state)   # returns a running GpuGuard
    ...
    guard.touch()          # once per step, enables the idle-release helper

Exit codes: 3 = own memory over cap or card free below floor, 4 = other tenants vanished
(we released the card), 5 = idle exit. Standalone watchdog for any PID::

    python -m flybrain.ops.gpu_guard --watch <pid> [--gpu 0]
    python -m flybrain.ops.gpu_guard --status [--gpu 0]     # read-only, exit 2 if not launchable
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

GiB = 1 << 30
MiB = 1 << 20

EXIT_LIMIT = 3
EXIT_RELEASED = 4
EXIT_IDLE = 5

DEFAULT_MAX_OWN_GIB = 1.6
DEFAULT_MIN_CARD_FREE_MIB = 500
DEFAULT_REQUIRE_FREE_GIB = 2.2
DEFAULT_FRACTION_CAP_GIB = 1.4
DEFAULT_POLL_S = 0.5
DEFAULT_VANISH_GRACE_S = 5.0
ALLOC_CONF_DEFAULT = "expandable_segments:True,garbage_collection_threshold:0.6"

log = logging.getLogger("flybrain.gpu_guard")


class GpuGuardError(RuntimeError):
    """Raised when the guard refuses to let the process start."""


@dataclass
class Snapshot:
    """One NVML poll of the card: memory and per-process usage (bytes)."""

    free_b: int
    total_b: int
    own_b: int
    others: dict[int, int] = field(default_factory=dict)  # pid -> bytes (0 if unreported)
    own_reported: bool = True


def _pid_uid(pid: int) -> int | None:
    """Owner uid of a live process via /proc (None if gone or not Linux)."""
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def _normalize_uuid(x: Any) -> str:
    s = x.decode() if isinstance(x, (bytes, bytearray)) else str(x)
    s = s.strip().lower()
    if s.startswith("gpu-"):
        s = s[4:]
    return s.replace("-", "")


def _visible_device_index(device_index: int) -> int | None:
    """Map a torch device index to an NVML index via CUDA_VISIBLE_DEVICES when it is numeric."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        return None
    parts = [p.strip() for p in vis.split(",")]
    if device_index < len(parts) and parts[device_index].isdigit():
        return int(parts[device_index])
    return None


def resolve_nvml_index(nvml: Any, device_index: int = 0, torch_mod: Any = None) -> int:
    """NVML index of torch device `device_index`: UUID match first, then CUDA_VISIBLE_DEVICES."""
    if torch_mod is not None:
        try:
            want = _normalize_uuid(torch_mod.cuda.get_device_properties(device_index).uuid)
            for i in range(nvml.nvmlDeviceGetCount()):
                got = _normalize_uuid(nvml.nvmlDeviceGetUUID(nvml.nvmlDeviceGetHandleByIndex(i)))
                if got == want:
                    return i
        except Exception as e:  # noqa: BLE001 - uuid lookup is best effort
            log.debug("uuid match failed (%s); falling back to index", e)
    vis = _visible_device_index(device_index)
    return device_index if vis is None else vis


class GpuGuard:
    """NVML watchdog thread with three rules: own cap, card-free floor, other-tenants vanished.

    `nvml`, `torch_mod`, `exit_fn`, `clock` are injectable for tests. `check_once()` is the
    whole rule set and can be driven by hand.
    """

    def __init__(
        self,
        *,
        max_own_gib: float = DEFAULT_MAX_OWN_GIB,
        min_card_free_mib: float = DEFAULT_MIN_CARD_FREE_MIB,
        checkpoint_cb: Callable[[], None] | None = None,
        poll_s: float = DEFAULT_POLL_S,
        vanish_grace_s: float = DEFAULT_VANISH_GRACE_S,
        idle_s: float | None = None,
        idle_exit: bool = False,
        max_blind_polls: int = 20,
        device_index: int = 0,
        nvml_index: int | None = None,
        nvml: Any = None,
        torch_mod: Any = None,
        own_pid: int | None = None,
        pid_uid: Callable[[int], int | None] | None = None,
        exit_fn: Callable[[int], Any] = os._exit,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        pid_uid = _pid_uid if pid_uid is None else pid_uid
        self.max_own_b = int(max_own_gib * GiB)
        self.min_card_free_b = int(min_card_free_mib * MiB)
        self.checkpoint_cb = checkpoint_cb
        self.poll_s = poll_s
        self.vanish_grace_s = vanish_grace_s
        self.idle_s = idle_s
        self.idle_exit = idle_exit
        self.max_blind_polls = max_blind_polls
        self.device_index = device_index
        self.torch = torch_mod
        self.own_pid = os.getpid() if own_pid is None else own_pid
        self._exit = exit_fn
        self._clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_touch = clock()
        self._blind = 0
        self.polls = 0
        self.fired: str | None = None
        self._warned_unreported = False

        if nvml is None:
            import pynvml as nvml  # type: ignore[no-redef]
        self.nvml = nvml
        nvml.nvmlInit()
        self.nvml_index = (
            resolve_nvml_index(nvml, device_index, torch_mod) if nvml_index is None else nvml_index
        )
        self.handle = nvml.nvmlDeviceGetHandleByIndex(self.nvml_index)
        first = self.snapshot()
        # The vanish rule tracks only processes owned by ANOTHER user (the root-owned vLLM
        # workers). Jobs under our own account (a teammate's benchmark) come and go and must
        # not trigger a release. Measured on node1: same-account GPU jobs do occur.
        me = os.getuid() if hasattr(os, "getuid") else -1
        foreign = {p for p in first.others if (pid_uid(p) not in (None, me))}
        self.others_at_start: frozenset[int] = frozenset(foreign)
        self.total_b = first.total_b
        log.info(
            "gpu_guard: nvml index %d, free %.0f MiB / %.0f MiB, own %.0f MiB, other pids %s "
            "(tracked, other users: %s), cap %.2f GiB, floor %.0f MiB",
            self.nvml_index, first.free_b / MiB, first.total_b / MiB, first.own_b / MiB,
            sorted(first.others) or "none", sorted(self.others_at_start) or "none",
            max_own_gib, min_card_free_mib,
        )
        if not self.others_at_start:
            log.warning("gpu_guard: no other users' processes on the card at start; vanish rule disabled")

    # -- polling ------------------------------------------------------------------------

    def _processes(self) -> list[Any]:
        procs: list[Any] = []
        for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            fn = getattr(self.nvml, getter, None)
            if fn is None:
                continue
            try:
                procs.extend(fn(self.handle))
            except self.nvml.NVMLError as e:
                log.debug("%s failed: %s", getter, e)
        return procs

    def snapshot(self) -> Snapshot:
        mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        own_b, others, own_reported = 0, {}, True
        for p in self._processes():
            used = p.usedGpuMemory
            if used is None or (isinstance(used, int) and used < 0):
                used, reported = 0, False
            else:
                reported = True
            if p.pid == self.own_pid:
                own_b += int(used)
                own_reported = own_reported and reported
            else:
                others[int(p.pid)] = others.get(int(p.pid), 0) + int(used)
        if not own_reported and self.torch is not None:
            # NVML does not report per-process memory here: fall back to the allocator's
            # reserved bytes (under-estimates by the CUDA context, ~0.4-0.6 GB).
            try:
                own_b = max(own_b, int(self.torch.cuda.memory_reserved(self.device_index)))
            except Exception:  # noqa: BLE001
                pass
        return Snapshot(int(mem.free), int(mem.total), own_b, others, own_reported)

    def touch(self) -> None:
        """Mark the GPU as in use now (for the idle-release helper)."""
        self._last_touch = self._clock()

    def release_when_idle(self, idle_s: float | None = 300.0, exit_after: bool = False) -> None:
        """Enable idle release: after `idle_s` without touch(), empty torch's cache and,
        with `exit_after`, checkpoint and exit (code 5) so the context itself is freed."""
        self.idle_s, self.idle_exit = idle_s, exit_after
        self.touch()

    def release_cached(self) -> int:
        """gc + torch.cuda.empty_cache(); returns bytes of card memory freed (NVML, may be 0)."""
        before = self.snapshot().free_b
        gc.collect()
        if self.torch is not None:
            try:
                self.torch.cuda.synchronize(self.device_index)
                self.torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                log.warning("empty_cache failed: %s", e)
        return max(0, self.snapshot().free_b - before)

    # -- rules --------------------------------------------------------------------------

    def check_once(self, snap: Snapshot | None = None) -> str | None:
        """Apply the rules to one snapshot. Returns the rule that fired ('own_cap',
        'card_free', 'vanished', 'idle', 'blind') or None. Calls exit_fn when a rule fires."""
        self.polls += 1
        if snap is None:
            try:
                snap = self.snapshot()
                self._blind = 0
            except Exception as e:  # noqa: BLE001 - NVML hiccup
                self._blind += 1
                log.warning("gpu_guard: NVML poll failed (%d in a row): %s", self._blind, e)
                if self._blind >= self.max_blind_polls:
                    return self._fire("blind", EXIT_LIMIT,
                                      f"NVML unreadable for {self._blind} polls; cannot enforce cap")
                return None
        if not snap.own_reported and not self._warned_unreported:
            self._warned_unreported = True
            log.warning("gpu_guard: NVML does not report our per-process memory; own-cap rule "
                        "uses torch reserved bytes (excludes the CUDA context)")
        if snap.own_b > self.max_own_b:
            return self._fire("own_cap", EXIT_LIMIT,
                              f"own GPU memory {snap.own_b / MiB:.0f} MiB > cap {self.max_own_b / MiB:.0f} MiB")
        if snap.free_b < self.min_card_free_b:
            return self._fire("card_free", EXIT_LIMIT,
                              f"card free {snap.free_b / MiB:.0f} MiB < floor {self.min_card_free_b / MiB:.0f} MiB")
        if self.others_at_start and not (self.others_at_start & set(snap.others)):
            return self._fire("vanished", EXIT_RELEASED,
                              f"other tenants {sorted(self.others_at_start)} all gone; releasing the card",
                              checkpoint=True)
        if self.idle_s is not None and self._clock() - self._last_touch > self.idle_s:
            self._last_touch = self._clock()
            freed = self.release_cached()
            log.info("gpu_guard: idle > %.0f s, emptied cache (%.0f MiB freed)", self.idle_s, freed / MiB)
            if self.idle_exit:
                return self._fire("idle", EXIT_IDLE, "idle exit requested", checkpoint=True)
            return "idle"
        return None

    def _fire(self, rule: str, code: int, why: str, checkpoint: bool = False) -> str:
        self.fired = rule
        log.error("gpu_guard: %s -> exit %d (%s)", rule, code, why)
        if checkpoint:
            # Hard deadline: whatever the checkpoint does, we are gone within vanish_grace_s.
            timer = threading.Timer(self.vanish_grace_s, self._exit, args=(code,))
            timer.daemon = True
            timer.start()
            if self.checkpoint_cb is not None:
                try:
                    self.checkpoint_cb()
                except Exception as e:  # noqa: BLE001
                    log.error("gpu_guard: checkpoint_cb failed: %s", e)
            timer.cancel()
        self._exit(code)
        return rule

    # -- thread -------------------------------------------------------------------------

    def start(self) -> "GpuGuard":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gpu_guard", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.poll_s):
            try:
                self.check_once()
            except Exception as e:  # noqa: BLE001 - never let the watchdog die silently
                log.error("gpu_guard: unexpected error in watchdog: %s", e)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 4 * self.poll_s))
            self._thread = None

    def close(self) -> None:
        self.stop()
        try:
            self.nvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


def preflight(
    torch_mod: Any,
    *,
    device_index: int = 0,
    require_free_gib: float = DEFAULT_REQUIRE_FREE_GIB,
    fraction_cap_gib: float = DEFAULT_FRACTION_CAP_GIB,
    nvml: Any = None,
) -> dict[str, float]:
    """Refuse (GpuGuardError) unless the card has >= require_free_gib free, then cap the
    caching allocator at fraction_cap_gib / total. Sets PYTORCH_CUDA_ALLOC_CONF if unset.

    The NVML check runs first because torch.cuda.mem_get_info itself creates the CUDA
    context (~0.4-0.6 GB) - we do not want to create it on a card that is already full.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF_DEFAULT)
    need = int(require_free_gib * GiB)
    if nvml is not None:
        nvml.nvmlInit()
        idx = resolve_nvml_index(nvml, device_index, None)
        mem = nvml.nvmlDeviceGetMemoryInfo(nvml.nvmlDeviceGetHandleByIndex(idx))
        if int(mem.free) < need:
            raise GpuGuardError(
                f"refusing to launch: NVML says {mem.free / GiB:.2f} GiB free on card {idx}, "
                f"need {require_free_gib:.2f} GiB (no CUDA context was created)")
    if not torch_mod.cuda.is_available():
        raise GpuGuardError("refusing to launch: torch.cuda.is_available() is False")
    free_b, total_b = torch_mod.cuda.mem_get_info(device_index)
    if free_b < need:
        raise GpuGuardError(
            f"refusing to launch: {free_b / GiB:.2f} GiB free on device {device_index}, "
            f"need {require_free_gib:.2f} GiB")
    fraction = fraction_cap_gib * GiB / total_b  # == fraction_cap_gib / 24 on a 24 GiB A30
    torch_mod.cuda.set_per_process_memory_fraction(fraction, device_index)
    log.info("gpu_guard: preflight ok, free %.2f GiB of %.2f GiB, allocator fraction %.4f (%.2f GiB)",
             free_b / GiB, total_b / GiB, fraction, fraction_cap_gib)
    return {"free_gib": free_b / GiB, "total_gib": total_b / GiB, "fraction": fraction}


def guard_gpu(
    max_own_gib: float = DEFAULT_MAX_OWN_GIB,
    min_card_free_mib: float = DEFAULT_MIN_CARD_FREE_MIB,
    require_free_gib: float = DEFAULT_REQUIRE_FREE_GIB,
    fraction_cap_gib: float = DEFAULT_FRACTION_CAP_GIB,
    checkpoint_cb: Callable[[], None] | None = None,
    *,
    device_index: int = 0,
    visible_devices: str | None = "0",
    poll_s: float = DEFAULT_POLL_S,
    idle_s: float | None = None,
    idle_exit: bool = False,
    torch_mod: Any = None,
    nvml: Any = None,
    pid_uid: Callable[[int], int | None] | None = None,
    exit_fn: Callable[[int], Any] = os._exit,
) -> GpuGuard:
    """Call before any CUDA allocation. Pins CUDA_VISIBLE_DEVICES (if unset), runs the
    preflight (refuse / allocator cap / alloc conf) and starts the watchdog thread.

    `checkpoint_cb` must be fast and atomic (write tmp + rename): the process is gone
    5 s after the other tenants' PIDs vanish, whether or not the callback finished.
    """
    if visible_devices is not None:
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            if "torch" in sys.modules and getattr(sys.modules["torch"].cuda, "is_initialized", lambda: False)():
                log.warning("gpu_guard: CUDA already initialised; CUDA_VISIBLE_DEVICES=%s not applied",
                            visible_devices)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", ALLOC_CONF_DEFAULT)
    if torch_mod is None:
        import torch as torch_mod  # type: ignore[no-redef]
    if nvml is None:
        import pynvml as nvml  # type: ignore[no-redef]
    preflight(torch_mod, device_index=device_index, require_free_gib=require_free_gib,
              fraction_cap_gib=fraction_cap_gib, nvml=nvml)
    guard = GpuGuard(
        max_own_gib=max_own_gib, min_card_free_mib=min_card_free_mib, checkpoint_cb=checkpoint_cb,
        poll_s=poll_s, idle_s=idle_s, idle_exit=idle_exit, device_index=device_index,
        nvml=nvml, torch_mod=torch_mod, pid_uid=pid_uid, exit_fn=exit_fn,
    )
    return guard.start()


# -- standalone watchdog ------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def watch_pid(
    pid: int,
    *,
    gpu: int = 0,
    max_own_gib: float = DEFAULT_MAX_OWN_GIB,
    min_card_free_mib: float = DEFAULT_MIN_CARD_FREE_MIB,
    poll_s: float = DEFAULT_POLL_S,
    grace_s: float = DEFAULT_VANISH_GRACE_S,
    nvml: Any = None,
    kill: Callable[[int, int], Any] = os.kill,
    alive: Callable[[int], bool] = _pid_alive,
    sleep: Callable[[float], Any] = time.sleep,
    max_polls: int | None = None,
    pid_uid: Callable[[int], int | None] | None = None,
) -> int:
    """Watch another process from outside: SIGKILL it on cap breach (returns 3); on the
    other tenants vanishing send SIGTERM, wait grace_s, SIGKILL (returns 4). Returns 0 when
    the watched process exits on its own."""
    def fake_exit(code: int) -> None:  # the guard's exit_fn: we act on the child, not ourselves
        raise _Fired(code)

    guard = GpuGuard(max_own_gib=max_own_gib, min_card_free_mib=min_card_free_mib, poll_s=poll_s,
                     vanish_grace_s=grace_s, nvml_index=gpu, nvml=nvml, own_pid=pid,
                     pid_uid=pid_uid, exit_fn=fake_exit)
    polls = 0
    while alive(pid):
        try:
            guard.check_once()
        except _Fired as f:
            if f.code == EXIT_RELEASED:
                kill(pid, signal.SIGTERM)
                deadline = time.monotonic() + grace_s
                while alive(pid) and time.monotonic() < deadline:
                    sleep(min(0.1, poll_s))
            if alive(pid):
                kill(pid, signal.SIGKILL)
            log.error("gpu_guard --watch: killed pid %d (rule %s, code %d)", pid, guard.fired, f.code)
            return f.code
        polls += 1
        if max_polls is not None and polls >= max_polls:
            break
        sleep(poll_s)
    return 0


class _Fired(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


def status(gpu: int = 0, require_free_gib: float = DEFAULT_REQUIRE_FREE_GIB, nvml: Any = None) -> dict[str, Any]:
    """Read-only card summary: free/total MiB, processes, and whether a launch is allowed."""
    if nvml is None:
        import pynvml as nvml  # type: ignore[no-redef]
    nvml.nvmlInit()
    h = nvml.nvmlDeviceGetHandleByIndex(gpu)
    mem = nvml.nvmlDeviceGetMemoryInfo(h)
    procs = []
    try:
        procs = [{"pid": p.pid, "used_mib": None if p.usedGpuMemory is None else p.usedGpuMemory / MiB}
                 for p in nvml.nvmlDeviceGetComputeRunningProcesses(h)]
    except nvml.NVMLError as e:
        procs = [{"error": str(e)}]
    return {"gpu": gpu, "free_mib": mem.free / MiB, "total_mib": mem.total / MiB,
            "launch_ok": mem.free >= require_free_gib * GiB, "processes": procs}


def selftest(seconds: float = 3.0, alloc_mib: int = 64) -> int:
    """Start the guard on the real card (under all the rules), allocate `alloc_mib`, poll for
    `seconds`, print what NVML sees for this process. Measures the CUDA context size."""
    guard = guard_gpu(poll_s=0.5)
    import torch

    t0 = guard.snapshot()
    x = torch.empty(alloc_mib * MiB, dtype=torch.uint8, device=f"cuda:{guard.device_index}")
    torch.cuda.synchronize()
    t1 = guard.snapshot()
    time.sleep(seconds)
    t2 = guard.snapshot()
    print(f"selftest: nvml index {guard.nvml_index}, others at start {sorted(guard.others_at_start)}, "
          f"polls {guard.polls}, fired {guard.fired}")
    print(f"selftest: own after context {t0.own_b / MiB:.0f} MiB (reported={t0.own_reported}); "
          f"after +{alloc_mib} MiB tensor {t1.own_b / MiB:.0f} MiB; after {seconds:.0f}s {t2.own_b / MiB:.0f} MiB; "
          f"torch reserved {torch.cuda.memory_reserved(guard.device_index) / MiB:.0f} MiB")
    print(f"selftest: card free {t0.free_b / MiB:.0f} -> {t2.free_b / MiB:.0f} MiB; others now {sorted(t2.others)}")
    del x
    freed = guard.release_cached()
    print(f"selftest: release_cached freed {freed / MiB:.0f} MiB; own now {guard.snapshot().own_b / MiB:.0f} MiB")
    guard.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, metavar="PID", help="watch this PID from outside and kill it on breach")
    ap.add_argument("--status", action="store_true", help="print the card summary (read-only)")
    ap.add_argument("--selftest", action="store_true", help="start the guard on the real card, allocate 64 MB, exit")
    ap.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    ap.add_argument("--max-own-gib", type=float, default=DEFAULT_MAX_OWN_GIB)
    ap.add_argument("--min-card-free-mib", type=float, default=DEFAULT_MIN_CARD_FREE_MIB)
    ap.add_argument("--require-free-gib", type=float, default=DEFAULT_REQUIRE_FREE_GIB)
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL_S)
    ap.add_argument("--grace", type=float, default=DEFAULT_VANISH_GRACE_S)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    if args.status:
        s = status(args.gpu, args.require_free_gib)
        print(f"gpu {s['gpu']}: free {s['free_mib']:.0f} MiB / {s['total_mib']:.0f} MiB, "
              f"launch_ok={s['launch_ok']}, processes={s['processes']}")
        return 0 if s["launch_ok"] else 2
    if args.watch is not None:
        return watch_pid(args.watch, gpu=args.gpu, max_own_gib=args.max_own_gib,
                         min_card_free_mib=args.min_card_free_mib, poll_s=args.poll, grace_s=args.grace)
    if args.selftest:
        try:
            return selftest()
        except GpuGuardError as e:
            print(f"selftest: {e}", file=sys.stderr)
            return 2
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
