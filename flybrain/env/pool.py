"""Multiprocess ViZDoom pool: N worker processes, frames in shared-memory ring buffers,
actions/rewards/infos over pipes. Workers auto-reset (Gymnasium vector semantics: the step
that ends an episode returns the first obs of the next one and `info["final_info"]`).

    pool = EnvPool(8, "take_cover")
    obs, infos = pool.reset()                     # obs["gray"]: uint8 view [N, H, W]
    obs, r, term, trunc, infos = pool.step(actions)
    pool.close()

Returned obs arrays are VIEWS into the shared ring (valid for `ring_len` more steps); copy if
you keep them. `envs_per_worker > 1` lets one process step several envs in sequence, which
amortises the ~1 ms pipe round-trip per step (measured) when envs outnumber cores.
`python -m flybrain.env.pool --bench --workers 8 20` measures agent-steps/s and per-process RSS
(python worker + its vizdoom engine children).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import os
import sys
import time
from typing import Any

import numpy as np

from flybrain.env.doom import HEIGHT, N_ACTIONS, TRAIN_SEED_HI, TRAIN_SEED_LO, WIDTH, DoomEnv


class SeedStream:
    """Per-env seed sequence cycling inside [lo, hi]: env e gets lo + (e + k*n) % span."""

    def __init__(self, lo: int, hi: int, env: int, n_envs: int):
        self.lo, self.span, self.e, self.n, self.k = lo, hi - lo + 1, env, n_envs, 0

    def next(self) -> int:
        s = self.lo + (self.e + self.k * self.n) % self.span
        self.k += 1
        return s


def _shapes(n: int, ring: int, frameskip: int, all_frames: bool, rgb: bool) -> dict[str, tuple]:
    f = (frameskip,) if all_frames else ()
    shp = {"gray": (n, ring) + f + (HEIGHT, WIDTH)}
    if rgb:
        shp["rgb"] = (n, ring) + f + (HEIGHT, WIDTH, 3)
    return shp


def _worker(conn, wid: int, k_envs: int, scenario: str, env_kwargs: dict, raw: dict, shapes: dict,
            ring: int, seed_lo: int, seed_hi: int, n_envs: int, core: int | None) -> None:
    """Holds envs wid*k .. wid*k+k-1, steps them in sequence, writes frames to the shared ring."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    if core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
    e0 = wid * k_envs
    bufs = {k: np.frombuffer(raw[k], dtype=np.uint8).reshape(shapes[k])[e0:e0 + k_envs] for k in raw}
    envs = [DoomEnv(scenario, **env_kwargs) for _ in range(k_envs)]
    seeds = [SeedStream(seed_lo, seed_hi, e0 + j, n_envs) for j in range(k_envs)]
    t = 0

    def put(j, obs):
        slot = t % ring
        for k, b in bufs.items():
            b[j, slot] = obs[k]
        return slot

    try:
        while True:
            try:
                cmd, arg = conn.recv()
            except EOFError:  # parent went away
                break
            if cmd == "reset":
                out = []
                for j, env in enumerate(envs):
                    obs, info = env.reset(seed=seeds[j].next() if arg is None else int(arg[j]))
                    out.append((put(j, obs), info))
                t += 1
                conn.send(out)
            elif cmd == "step":
                out = []
                for j, env in enumerate(envs):
                    obs, r, term, trunc, info = env.step(int(arg[j]))
                    if term or trunc:
                        final = info
                        obs, info = env.reset(seed=seeds[j].next())
                        info["final_info"] = final
                    out.append((put(j, obs), r, term, trunc, info))
                t += 1
                conn.send(out)
            elif cmd == "attr":
                conn.send(getattr(envs[0], arg))
            elif cmd == "close":
                break
    finally:
        for env in envs:
            env.close()
        conn.close()


class EnvPool:
    """`n_workers` processes x `envs_per_worker` envs = `n` envs, indexed worker-major."""

    def __init__(self, n_workers: int, scenario: str = "take_cover", ring_len: int = 4,
                 envs_per_worker: int = 1, seed_range: tuple[int, int] = (TRAIN_SEED_LO, TRAIN_SEED_HI),
                 ctx: str = "fork", pin_cores: list[int] | None = None, **env_kwargs):
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        self.n_workers, self.k = int(n_workers), int(envs_per_worker)
        self.n = self.n_workers * self.k
        self.ring = int(ring_len)
        self.scenario = scenario
        self.env_kwargs = dict(env_kwargs)
        frameskip = env_kwargs.get("frameskip", 4)
        all_frames = env_kwargs.get("all_frames", False)
        rgb = env_kwargs.get("rgb", True)
        self.shapes = _shapes(self.n, self.ring, frameskip, all_frames, rgb)
        mpctx = mp.get_context(ctx if sys.platform != "win32" else "spawn")
        self._raw = {k: mpctx.RawArray(ctypes.c_uint8, int(np.prod(s))) for k, s in self.shapes.items()}
        self.bufs = {k: np.frombuffer(self._raw[k], dtype=np.uint8).reshape(self.shapes[k]) for k in self._raw}
        self.conns, self.procs = [], []
        for w in range(self.n_workers):
            parent, child = mpctx.Pipe()
            core = pin_cores[w % len(pin_cores)] if pin_cores else None
            p = mpctx.Process(target=_worker, daemon=False,
                              args=(child, w, self.k, scenario, self.env_kwargs, self._raw, self.shapes,
                                    self.ring, seed_range[0], seed_range[1], self.n, core))
            p.start()
            child.close()
            self.conns.append(parent)
            self.procs.append(p)
        self._t = 0
        self._pending = False
        self.n_actions = N_ACTIONS

    # -- protocol ---------------------------------------------------------------------------
    def _obs(self, slot: int) -> dict[str, np.ndarray]:
        return {k: b[:, slot] for k, b in self.bufs.items()}

    def reset(self, seeds: list[int] | None = None) -> tuple[dict[str, np.ndarray], list[dict]]:
        for w, c in enumerate(self.conns):
            c.send(("reset", None if seeds is None else [int(s) for s in seeds[w * self.k:(w + 1) * self.k]]))
        replies = [r for c in self.conns for r in c.recv()]  # flattened, env order
        slot = replies[0][0]
        self._t += 1
        return self._obs(slot), [r[1] for r in replies]

    def send(self, actions) -> None:
        """Dispatch actions without waiting (pair with `recv`)."""
        assert not self._pending, "recv() before sending again"
        actions = np.asarray(actions).reshape(-1)
        assert actions.shape[0] == self.n
        for w, c in enumerate(self.conns):
            c.send(("step", actions[w * self.k:(w + 1) * self.k].tolist()))
        self._pending = True

    def recv(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        assert self._pending
        replies = [r for c in self.conns for r in c.recv()]
        self._pending = False
        self._t += 1
        slot = replies[0][0]
        rew = np.array([r[1] for r in replies], dtype=np.float32)
        term = np.array([r[2] for r in replies], dtype=bool)
        trunc = np.array([r[3] for r in replies], dtype=bool)
        return self._obs(slot), rew, term, trunc, [r[4] for r in replies]

    def step(self, actions):
        self.send(actions)
        return self.recv()

    def env_attr(self, name: str) -> Any:
        """Read an attribute of env 0's DoomEnv (e.g. `noop_actions`, `button_names`)."""
        self.conns[0].send(("attr", name))
        return self.conns[0].recv()

    def pids(self) -> list[int]:
        return [p.pid for p in self.procs]

    def close(self) -> None:
        for c in self.conns:
            try:
                c.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        for c in self.conns:
            c.close()
        self.conns, self.procs = [], []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            if self.procs:
                self.close()
        except Exception:
            pass


# -- measurement --------------------------------------------------------------------------------
def rss_report(pool: EnvPool) -> dict[str, float]:
    """RSS (MiB) of the python workers and their vizdoom engine children, via psutil."""
    import psutil
    py, eng = [], []
    for pid in pool.pids():
        try:
            p = psutil.Process(pid)
            py.append(p.memory_info().rss / 2**20)
            eng.append(sum(k.memory_info().rss for k in p.children(recursive=True)) / 2**20)
        except psutil.Error:
            pass
    return {"rss_python_worker_mb_mean": float(np.mean(py)) if py else float("nan"),
            "rss_engine_children_mb_mean": float(np.mean(eng)) if eng else float("nan"),
            "rss_total_workers_mb": float(np.sum(py) + np.sum(eng)),
            "rss_parent_mb": psutil.Process().memory_info().rss / 2**20}


def bench(n_workers: int, scenario: str = "take_cover", seconds: float = 20.0,
          envs_per_worker: int = 1, **env_kwargs) -> dict[str, Any]:
    """Random-action throughput of the pool: agent-steps/s (= env-decisions/s) and RSS."""
    rng = np.random.default_rng(0)
    with EnvPool(n_workers, scenario, envs_per_worker=envs_per_worker, **env_kwargs) as pool:
        pool.reset()
        for _ in range(5):  # warm up
            pool.step(rng.integers(0, N_ACTIONS, pool.n))
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            pool.step(rng.integers(0, N_ACTIONS, pool.n))
            n += 1
        dt = time.perf_counter() - t0
        out = {"scenario": scenario, "workers": n_workers, "envs_per_worker": envs_per_worker,
               "envs": pool.n, "seconds": dt, "pool_steps": n,
               "agent_steps_per_s": n * pool.n / dt, "pool_steps_per_s": n / dt}
        out.update(rss_report(pool))
        out["loadavg_1min"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
        out.update({k: v for k, v in env_kwargs.items() if k in ("all_frames", "frameskip", "rgb")})
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ViZDoom env pool benchmark")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--workers", type=int, nargs="+", default=[8, 20])
    p.add_argument("--envs-per-worker", type=int, default=1)
    p.add_argument("--scenario", default="take_cover")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--out", default=None, help="JSON lines file to append results to")
    a = p.parse_args(argv)
    if not a.bench:
        p.print_help()
        return
    for w in a.workers:
        r = bench(w, a.scenario, a.seconds, a.envs_per_worker, all_frames=a.all_frames)
        print(json.dumps(r))
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
