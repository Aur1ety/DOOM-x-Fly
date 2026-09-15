"""Scripted floor policies and the evaluation protocol.

Policies act on the 7-action space of `flybrain.env.doom` and see the same obs/info an agent
sees. Evaluation: `n_episodes` episodes per policy per scenario on eval seeds (>= 10000),
reporting mean, IQM (mean of the middle 50%), median and a stratified-bootstrap CI.

    python -m flybrain.env.floors --scenarios take_cover defend_the_line \
        --policies random spin_fire constant_forward --episodes 20 --workers 6
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import multiprocessing.util
import os
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from flybrain import OUT_DIR
from flybrain.env.doom import ACTIONS, EVAL_SEED_LO, N_ACTIONS, DoomEnv

A = {name: i for i, name in enumerate(ACTIONS)}


class Policy:
    """Base: `reset(env, rng)` at episode start, `act(obs, info) -> int`."""

    name = "base"

    def reset(self, env: DoomEnv, rng: np.random.Generator) -> None:
        self.env, self.rng = env, rng

    def act(self, obs: dict, info: dict) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    """Uniform over all 7 actions (no-op actions included: this is the agent's action space)."""

    name = "random"

    def act(self, obs, info):
        return int(self.rng.integers(N_ACTIONS))


class ConstantPolicy(Policy):
    def __init__(self, action: str):
        self.action = A[action]
        self.name = f"constant_{action}"

    def act(self, obs, info):
        return self.action


class SpinFirePolicy(Policy):
    """Alternate turn and attack every decision (single-button action space, so no chord)."""

    name = "spin_fire"

    def __init__(self, turn: str = "turn_right"):
        self.turn = A[turn]
        self.t = 0

    def reset(self, env, rng):
        super().reset(env, rng)
        self.t = 0

    def act(self, obs, info):
        self.t += 1
        return self.turn if self.t % 2 else A["attack"]


class CentroidPolicy(Policy):
    """Labels-buffer heuristic on the nearest visible target's azimuth (pixel-centroid).

    Scenarios with ATTACK: turn toward the nearest enemy, fire when within `aim_deg`; scan
    right when nothing is visible. Scenarios without ATTACK (take_cover): strafe away from the
    nearest projectile, else away from the nearest enemy's side.
    """

    name = "centroid"

    def __init__(self, aim_deg: float = 6.0):
        self.aim_deg = aim_deg

    def act(self, obs, info):
        lab = info.get("labels", {})
        if "attack" not in self.env.noop_actions:
            if lab.get("enemy_visible", 0.0):
                az = lab["enemy_azimuth"]
                if abs(az) <= self.aim_deg:
                    return A["attack"]
                return A["turn_right"] if az > 0 else A["turn_left"]
            return A["turn_right"]
        for kind in ("projectile", "enemy"):
            if lab.get(f"{kind}_visible", 0.0):
                az = lab[f"{kind}_azimuth"]
                return A["strafe_left"] if az > 0 else A["strafe_right"]
        return A["strafe_left"] if self.rng.random() < 0.5 else A["strafe_right"]


POLICIES: dict[str, Callable[[], Policy]] = {
    "random": RandomPolicy,
    "spin_fire": SpinFirePolicy,
    "constant_forward": lambda: ConstantPolicy("move_forward"),
    "centroid": CentroidPolicy,
}
for _a in ACTIONS:
    POLICIES[f"constant_{_a}"] = (lambda a=_a: ConstantPolicy(a))
CHEAP = ("random", "spin_fire", "constant_forward")


def run_episode(env: DoomEnv, policy: Policy, seed: int, max_steps: int = 10_000) -> dict[str, Any]:
    """One episode; returns the wrapper's episode stats plus the seed."""
    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    policy.reset(env, rng)
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(policy.act(obs, info))
        if term or trunc:
            return dict(info["episode"])
    raise RuntimeError(f"episode did not end within {max_steps} steps")


# -- worker-side globals for multiprocess evaluation ---------------------------------------------
_ENV: DoomEnv | None = None


def _init(scenario: str, env_kwargs: dict) -> None:
    global _ENV
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _ENV = DoomEnv(scenario, **env_kwargs)
    # pool workers leave via os._exit: close the engine through multiprocessing's finalizers
    mp.util.Finalize(None, _ENV.close, exitpriority=10)


def _job(args: tuple[str, int]) -> dict[str, Any]:
    policy_name, seed = args
    return run_episode(_ENV, POLICIES[policy_name](), seed)


def summarize(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """mean / std / median / IQM with 95% bootstrap CI on mean and IQM."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    def iqm(v):
        v = np.sort(v, axis=-1)
        lo, hi = int(np.floor(0.25 * v.shape[-1])), int(np.ceil(0.75 * v.shape[-1]))
        return v[..., lo:hi].mean(axis=-1)

    rng = np.random.default_rng(seed)
    boot = x[rng.integers(0, n, size=(n_boot, n))]
    bm, bi = boot.mean(axis=1), iqm(boot)
    return {"n": n, "mean": float(x.mean()), "std": float(x.std(ddof=1)) if n > 1 else 0.0,
            "median": float(np.median(x)), "iqm": float(iqm(x)), "min": float(x.min()), "max": float(x.max()),
            "mean_ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))],
            "iqm_ci95": [float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5))]}


def evaluate(policy_name: str, scenario: str, n_episodes: int = 100, seed_base: int = EVAL_SEED_LO,
             workers: int = 1, env_kwargs: dict | None = None) -> dict[str, Any]:
    """Run `n_episodes` on seeds seed_base..seed_base+n-1; returns per-episode arrays + summaries."""
    env_kwargs = dict(env_kwargs or {})
    seeds = [seed_base + i for i in range(n_episodes)]
    t0 = time.perf_counter()
    if workers <= 1:
        with DoomEnv(scenario, **env_kwargs) as env:
            eps = [run_episode(env, POLICIES[policy_name](), s) for s in seeds]
    else:
        ctx = mp.get_context("fork")
        pool = ctx.Pool(workers, initializer=_init, initargs=(scenario, env_kwargs))
        try:
            eps = pool.map(_job, [(policy_name, s) for s in seeds], chunksize=1)
        finally:
            pool.close()
            pool.join()
    out = {"policy": policy_name, "scenario": scenario, "n_episodes": n_episodes, "seeds": seeds,
           "seconds": time.perf_counter() - t0, "env_kwargs": env_kwargs,
           "reward": [e["reward"] for e in eps], "kills": [e["kills"] for e in eps],
           "survival_tics": [e["survival_tics"] for e in eps], "steps": [e["steps"] for e in eps],
           "action_hist": np.sum([e["action_hist"] for e in eps], axis=0).tolist()}
    for k in ("reward", "kills", "survival_tics"):
        out[f"{k}_summary"] = summarize(np.array(out[k]))
    return out


def table(results: list[dict]) -> str:
    rows = ["| scenario | policy | n | reward mean | reward IQM | IQM 95% CI | survival tics mean | kills mean |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        s, k, t = r["reward_summary"], r["kills_summary"], r["survival_tics_summary"]
        rows.append(f"| {r['scenario']} | {r['policy']} | {s['n']} | {s['mean']:.1f} | {s['iqm']:.1f} | "
                    f"[{s['iqm_ci95'][0]:.1f}, {s['iqm_ci95'][1]:.1f}] | {t['mean']:.0f} | {k['mean']:.2f} |")
    return "\n".join(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="scripted floors")
    p.add_argument("--scenarios", nargs="+", default=["take_cover", "defend_the_line"])
    p.add_argument("--policies", nargs="+", default=list(CHEAP), choices=sorted(POLICIES))
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=EVAL_SEED_LO)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--sticky", type=float, default=0.0)
    p.add_argument("--out", default=str(OUT_DIR / "floors" / "floors.json"))
    a = p.parse_args(argv)
    results = []
    for scn in a.scenarios:
        for pol in a.policies:
            r = evaluate(pol, scn, a.episodes, a.seed_base, a.workers, {"sticky_prob": a.sticky})
            print(f"{scn:16s} {pol:22s} n={r['n_episodes']} reward mean={r['reward_summary']['mean']:.1f} "
                  f"iqm={r['reward_summary']['iqm']:.1f} survival={r['survival_tics_summary']['mean']:.0f} "
                  f"kills={r['kills_summary']['mean']:.2f}  ({r['seconds']:.0f}s)", flush=True)
            results.append(r)
    print(table(results))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
