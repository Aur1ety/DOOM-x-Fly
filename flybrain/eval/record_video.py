"""Record ViZDoom episodes to MP4 (the engine is headless on the server; this is how we watch it play).

Left panel: the game view (160x120 upscaled, nearest-neighbour). Right panel (optional): an
activity raster supplied by the policy via `policy.panel()` -> float array in [0, 1], e.g. the
readout / descending population rates of a connectome agent. Scripted policies have no panel.

    python -m flybrain.eval.record_video --scenario take_cover --policy centroid --episodes 3 \
        --out $FLYBRAIN_OUT/videos/take_cover_centroid.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from flybrain import OUT_DIR
from flybrain.env.doom import ACTIONS, EVAL_SEED_LO, DoomEnv


def _upscale(img: np.ndarray, k: int) -> np.ndarray:
    return np.repeat(np.repeat(img, k, axis=0), k, axis=1)


def _raster(panel: np.ndarray | None, height: int, width: int) -> np.ndarray:
    """Activity values in [0,1] -> a [height, width, 3] heat strip (rows = units, time scrolls left)."""
    out = np.zeros((height, width, 3), np.uint8)
    if panel is None:
        return out
    p = np.clip(np.asarray(panel, np.float32).ravel(), 0.0, 1.0)
    rows = np.linspace(0, len(p), height + 1).astype(int)
    v = np.array([p[a:b].mean() if b > a else 0.0 for a, b in zip(rows[:-1], rows[1:])])
    col = (255 * v).astype(np.uint8)
    out[:, :, 0] = col[:, None]; out[:, :, 1] = (col // 2)[:, None]; out[:, :, 2] = (255 - col)[:, None] // 3
    return out


def record(scenario: str, policy_name: str, episodes: int, out: Path, scale: int = 4, fps: int = 35 // 4,
           seed_base: int = EVAL_SEED_LO, policy=None, panel_width: int = 240) -> dict:
    import imageio.v2 as imageio
    from flybrain.env.floors import POLICIES

    env = DoomEnv(scenario, frameskip=4, all_frames=False, rgb=True, labels=True)
    pol = policy if policy is not None else POLICIES[policy_name]()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=7, macro_block_size=None)
    stats = []
    H = 120
    strip = np.zeros((H * scale, panel_width, 3), np.uint8)
    for e in range(episodes):
        seed = seed_base + e
        obs, info = env.reset(seed=seed)
        pol.reset(env, np.random.default_rng(seed))
        done = False
        while not done:
            a = pol.act(obs, info)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            frame = _upscale(obs["rgb"], scale)
            panel = pol.panel() if hasattr(pol, "panel") else None
            if panel is not None:
                strip = np.roll(strip, -1, axis=1)
                strip[:, -1] = _raster(panel, H * scale, 1)[:, 0]
            writer.append_data(np.concatenate([frame, strip], axis=1))
        stats.append(info["episode"])
    writer.close(); env.close()
    return {"out": str(out), "episodes": stats, "actions": list(ACTIONS)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="take_cover")
    ap.add_argument("--policy", default="centroid")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    out = a.out or OUT_DIR / "videos" / f"{a.scenario}_{a.policy}.mp4"
    res = record(a.scenario, a.policy, a.episodes, out, a.scale)
    for s in res["episodes"]:
        print({k: s[k] for k in ("seed", "reward", "kills", "survival_tics")})
    print("wrote", res["out"])


if __name__ == "__main__":
    main()
