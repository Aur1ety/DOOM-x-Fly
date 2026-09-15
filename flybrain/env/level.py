"""Real Doom levels (e.g. shareware E1M1 "Hangar") for the fly agent, plus a privileged scripted navigator.

The agent sees exactly what the take_cover agent saw: 160x120 grey + RGB frames (area-downsampled
from a 640x480 render, weapon visible, no HUD) at frameskip 4. The action set adds USE (doors,
switches). The navigator uses map geometry + the player's position (flybrain.env.wadmap) and the
labels buffer; it is the imitation teacher and a floor/ceiling reference, never the agent.

Reward shaping (for PPO fine-tuning and logging): progress = decrease of the walking distance to the
exit / 100 per decision (clipped), +1 per kill, +50 on exiting, -10 on death.

    python -m flybrain.env.level --episodes 5 --video $FLYBRAIN_OUT/videos/e1m1_navigator.mp4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

try:
    import vizdoom as vd
except ImportError:  # laptop
    vd = None

from flybrain import DATA_DIR
from flybrain.env.doom import parse_labels, wrap_deg
from flybrain.env.wadmap import build_nav, read_map

ACTIONS = ("move_forward", "move_backward", "turn_left", "turn_right", "strafe_left", "strafe_right", "attack", "use")
N_ACTIONS = len(ACTIONS)
FULL_W, FULL_H = 640, 480
W, H = 160, 120
DEFAULT_WAD = DATA_DIR / "wads" / "doom1.wad"


def _down(img: np.ndarray, k: int) -> np.ndarray:
    """Area-average downsample by integer factor k (HxW or HxWxC uint8)."""
    h, w = img.shape[0] // k, img.shape[1] // k
    x = img[: h * k, : w * k].astype(np.float32)
    x = x.reshape(h, k, w, k, *img.shape[2:]).mean((1, 3))
    return x.astype(np.uint8)


class LevelEnv:
    def __init__(self, wad: str | os.PathLike = DEFAULT_WAD, map_name: str = "E1M1", skill: int = 1,
                 fov: float | None = 108.0, frameskip: int = 4, sticky_prob: float = 0.0,
                 timeout_s: float = 180.0, keep_full: bool = False, nav_cell: float = 16.0,
                 start_wiggle: int = 0, wiggle_moves: bool = False, keep_tics: bool = False):
        if vd is None:
            raise ImportError("vizdoom is not installed (this module runs on the server)")
        self.map_name, self.frameskip, self.sticky_prob = map_name, frameskip, sticky_prob
        self.keep_full, self.start_wiggle, self.fov, self.wiggle_moves = keep_full, start_wiggle, fov, wiggle_moves
        self.keep_tics = keep_tics             # video only: also return every rendered tic (same game logic as make_action)
        g = vd.DoomGame()
        g.set_doom_game_path(str(wad)); g.set_doom_map(map_name); g.set_doom_skill(skill)
        g.set_window_visible(False); g.set_sound_enabled(False); g.set_console_enabled(False)
        g.set_mode(vd.Mode.PLAYER)
        g.set_screen_resolution(vd.ScreenResolution.RES_640X480); g.set_screen_format(vd.ScreenFormat.RGB24)
        g.set_render_hud(False); g.set_render_weapon(True); g.set_render_crosshair(False)
        g.set_labels_buffer_enabled(True); g.set_objects_info_enabled(True)
        for b in (vd.Button.MOVE_FORWARD, vd.Button.MOVE_BACKWARD, vd.Button.TURN_LEFT, vd.Button.TURN_RIGHT,
                  vd.Button.MOVE_LEFT, vd.Button.MOVE_RIGHT, vd.Button.ATTACK, vd.Button.USE):
            g.add_available_button(b)
        for v in (vd.GameVariable.POSITION_X, vd.GameVariable.POSITION_Y, vd.GameVariable.ANGLE, vd.GameVariable.VELOCITY_X,
                  vd.GameVariable.VELOCITY_Y, vd.GameVariable.HEALTH, vd.GameVariable.KILLCOUNT, vd.GameVariable.AMMO2,
                  vd.GameVariable.ARMOR):
            g.add_available_game_variable(v)
        g.set_episode_timeout(int(timeout_s * 35))
        g.init()
        self.game = g
        self.buttons = [[int(i == j) for i in range(N_ACTIONS)] for j in range(N_ACTIONS)]
        m = read_map(Path(wad), map_name)
        self.map = m
        self.nav = build_nav(m, nav_cell)
        self.d_start = self.nav.distance(*m.player_start[:2])
        self._rng = np.random.default_rng(0)

    # -- helpers -------------------------------------------------------------------------------
    def _vars(self) -> dict:
        g = self.game; gv = vd.GameVariable
        return {"x": g.get_game_variable(gv.POSITION_X), "y": g.get_game_variable(gv.POSITION_Y),
                "angle": g.get_game_variable(gv.ANGLE), "vx": g.get_game_variable(gv.VELOCITY_X),
                "vy": g.get_game_variable(gv.VELOCITY_Y), "health": g.get_game_variable(gv.HEALTH),
                "kills": g.get_game_variable(gv.KILLCOUNT), "ammo": g.get_game_variable(gv.AMMO2),
                "armor": g.get_game_variable(gv.ARMOR)}

    def _obs(self, state) -> dict:
        full = np.asarray(state.screen_buffer)
        gray_full = (0.299 * full[..., 0] + 0.587 * full[..., 1] + 0.114 * full[..., 2]).astype(np.uint8)
        obs = {"gray": _down(gray_full, 4), "rgb": _down(full, 4)}
        if self.keep_full:
            obs["full"] = full.copy()
        return obs

    def _info(self, state) -> dict:
        v = self._vars()
        info = dict(v)
        info["distance"] = self.nav.distance(v["x"], v["y"])
        info["progress"] = 1.0 - min(info["distance"], self.d_start) / self.d_start if math.isfinite(info["distance"]) else 0.0
        if state is not None:
            info["labels"] = parse_labels(state.labels, (v["x"], v["y"], v["angle"]), (v["vx"], v["vy"]))
        return info

    # -- API -----------------------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        seed = int(self._rng.integers(1 << 30)) if seed is None else int(seed)
        self.seed = seed
        self._rng = np.random.default_rng([seed, 0xE1])
        g = self.game
        g.set_seed(seed); g.new_episode()
        if self.fov is not None:
            g.send_game_command(f"fov {self.fov:g}"); g.advance_action(3, True)
        n_wiggle = int(self._rng.integers(0, self.start_wiggle + 1)) if self.start_wiggle else 0
        choices = [0, 1, 2, 3, 4, 5] if self.wiggle_moves else [2, 3, 4, 5]     # moves: also forward / backward
        for _ in range(n_wiggle):
            g.make_action(self.buttons[int(self._rng.choice(choices))], self.frameskip)
        state = g.get_state()
        self._prev_action = None
        info = self._info(state)
        self._last_obs = self._obs(state)
        self._ep = {"reward": 0.0, "shaped": 0.0, "steps": 0, "best_progress": info["progress"], "seed": seed,
                    "action_hist": np.zeros(N_ACTIONS, np.int64), "d_prev": info["distance"], "kills": 0.0, "wiggle": n_wiggle}
        return self._last_obs, info

    def step(self, action: int):
        g = self.game
        a = int(action)
        if self._prev_action is not None and self.sticky_prob > 0 and self._rng.random() < self.sticky_prob:
            a = self._prev_action
        self._prev_action = a
        tics = []
        if self.keep_tics:                     # set_action + 1-tic advances == make_action(a, frameskip), but keeps each frame
            g.set_action(self.buttons[a]); r = 0.0
            for _ in range(self.frameskip):
                g.advance_action(1, True); r += float(g.get_last_reward())
                if g.is_episode_finished():
                    break
                tics.append(np.asarray(g.get_state().screen_buffer).copy())
        else:
            r = float(g.make_action(self.buttons[a], self.frameskip))
        finished = g.is_episode_finished()
        dead = bool(g.is_player_dead())
        timeout = bool(g.is_episode_timeout_reached()) if finished else False
        exited = finished and not dead and not timeout
        state = None if finished else g.get_state()
        info = self._info(state) if state is not None else {**self._vars(), "distance": 0.0 if exited else self._ep["d_prev"], "labels": {},
                                                             "progress": 1.0 if exited else self._ep["best_progress"]}
        if state is not None:
            self._last_obs = self._obs(state)
        if self.keep_tics:
            self._last_obs["tics"] = tics
        d = info["distance"]
        prog = 0.0
        if math.isfinite(d) and math.isfinite(self._ep["d_prev"]):
            prog = float(np.clip((self._ep["d_prev"] - d) / 100.0, -1.0, 1.0))
            self._ep["d_prev"] = d
        kills = info.get("kills", self._ep["kills"])
        shaped = prog + 1.0 * (kills - self._ep["kills"]) + (50.0 if exited else 0.0) - (10.0 if dead else 0.0)
        self._ep["kills"] = kills
        self._ep["reward"] += r; self._ep["shaped"] += shaped; self._ep["steps"] += 1
        self._ep["action_hist"][a] += 1
        if "progress" in info:
            self._ep["best_progress"] = max(self._ep["best_progress"], info["progress"])
        if exited:
            self._ep["best_progress"] = 1.0
        info.update(action_taken=a, shaped=shaped, exited=exited, dead=dead, timeout=timeout)
        if finished:
            info["episode"] = {"seed": self._ep["seed"], "exited": exited, "dead": dead, "timeout": timeout,
                               "steps": self._ep["steps"], "tics": self._ep["steps"] * self.frameskip,
                               "best_progress": float(self._ep["best_progress"]), "shaped": float(self._ep["shaped"]),
                               "kills": float(kills), "action_hist": self._ep["action_hist"].tolist(), "wiggle": self._ep["wiggle"]}
        return self._last_obs, shaped, finished and (dead or exited), finished and timeout, info

    def close(self):
        if getattr(self, "game", None) is not None:
            self.game.close(); self.game = None


# ------------------------------------------------------------------------------------------------ navigator

A = {n: i for i, n in enumerate(ACTIONS)}


class Navigator:
    """Privileged scripted player: follows the distance field, fights visible enemies, opens doors, presses the exit."""

    name = "navigator"

    def __init__(self, aim_deg: float = 6.0, turn_deg: float = 14.0, engage_deg: float = 25.0, engage_dist: float = 900.0):
        self.aim_deg, self.turn_deg, self.engage_deg, self.engage_dist = aim_deg, turn_deg, engage_deg, engage_dist

    MODES = ("navigate", "fight", "exit", "use_burst", "unstick", "stall_use", "lost")

    def reset(self, env: LevelEnv, rng):
        self.env, self.rng = env, rng
        self.hist_d: list[float] = []; self.unstick = 0; self.use_burst = 0; self.exit_presses = 0
        self.last: int | None = None; self.mode = "navigate"

    def executed(self, a: int) -> None:
        """Tell the navigator which action was actually executed. If someone else drove (student, random
        burst), the lack of progress is not the navigator's stall: clear the stall history and any wiggle."""
        if self.last is not None and int(a) != self.last:
            self.hist_d.clear(); self.unstick = 0

    def act(self, obs, info) -> int:
        self.last = self._act(obs, info)
        return self.last

    def _act(self, obs, info) -> int:
        nav = self.env.nav
        x, y, ang, d = info["x"], info["y"], info["angle"], info["distance"]
        lab = info.get("labels", {}) or {}
        # 1. fight: a visible enemy close to the view centre
        if lab.get("enemy_visible", 0.0) and lab.get("enemy_distance", 1e9) < self.engage_dist and abs(lab["enemy_azimuth"]) < self.engage_deg:
            az = lab["enemy_azimuth"]
            self.mode = "fight"
            if abs(az) <= self.aim_deg:
                return A["attack"]
            return A["turn_right"] if az > 0 else A["turn_left"]
        # 2. at the exit: face the switch, walk up to it (USE range is 64 units from the line), press use
        if d < 128:
            self.mode = "exit"
            xy = self.env.map.line_xy(nav.exit_line); p0, p1 = xy[:2], xy[2:]
            seg = p1 - p0; t = np.clip(np.dot(np.array([x, y]) - p0, seg) / np.dot(seg, seg), 0.0, 1.0)
            near = p0 + t * seg; to_line = float(np.hypot(near[0] - x, near[1] - y))
            mid = (p0 + p1) / 2
            want = math.degrees(math.atan2(mid[1] - y, mid[0] - x))
            err = wrap_deg(want - ang)
            if abs(err) > self.turn_deg * 0.6:
                return A["turn_left"] if err > 0 else A["turn_right"]
            if to_line > 40:
                return A["move_forward"]
            self.exit_presses += 1
            return A["use"] if self.exit_presses % 3 else A["move_forward"]
        # 3. stuck handling: no progress over ~2 s -> press use (doors), then wiggle
        self.hist_d.append(d)
        if self.use_burst > 0:
            self.mode = "use_burst"; self.use_burst -= 1; return A["use"]
        if self.unstick > 0:
            self.mode = "unstick"; self.unstick -= 1; return int(self.rng.choice([A["strafe_left"], A["strafe_right"], A["move_backward"]]))
        if len(self.hist_d) > 20 and not (self.hist_d[-20] - d >= 24):     # NaN (inf - inf) counts as no progress
            self.hist_d.clear()
            self.use_burst = 2; self.unstick = int(self.rng.integers(2, 6))
            self.mode = "stall_use"; return A["use"]
        if not math.isfinite(d):
            self.mode = "lost"                   # off the distance field: the heading label is meaningless
        else:
            self.mode = "navigate"
        # 4. navigate
        wx, wy = nav.next_waypoint(x, y, lookahead=6)
        want = math.degrees(math.atan2(wy - y, wx - x))
        err = wrap_deg(want - ang)            # Doom angles grow counter-clockwise: err > 0 -> turn left
        if abs(err) > self.turn_deg:
            return A["turn_left"] if err > 0 else A["turn_right"]
        return A["move_forward"]


def run_episode(env: LevelEnv, policy, seed: int, writer=None, max_steps: int = 20_000) -> dict:
    obs, info = env.reset(seed)
    policy.reset(env, np.random.default_rng(seed))
    for _ in range(max_steps):
        a = policy.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        if writer is not None and "full" in obs:
            writer(obs["full"], info, a)
        if term or trunc:
            return info["episode"]
    raise RuntimeError("episode did not end")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", default=str(DEFAULT_WAD)); ap.add_argument("--map", default="E1M1")
    ap.add_argument("--skill", type=int, default=1); ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=10_000); ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--measure", action="store_true", help="measure turn / move per decision and exit")
    a = ap.parse_args(argv)
    env = LevelEnv(a.wad, a.map, a.skill, keep_full=a.video is not None)
    if a.measure:
        obs, info = env.reset(1)
        for name in ("turn_left", "turn_right", "move_forward", "strafe_left"):
            obs, info = env.reset(1); a0, x0, y0 = info["angle"], info["x"], info["y"]
            for _ in range(5):
                obs, r, t, tr, info = env.step(A[name])
            print(f"{name:12s} 5 decisions: dAngle {wrap_deg(info['angle'] - a0):7.2f} deg  dPos {math.hypot(info['x'] - x0, info['y'] - y0):7.1f} units")
        env.close(); return
    writer = None
    if a.video is not None:
        import imageio.v2 as iio
        a.video.parent.mkdir(parents=True, exist_ok=True)
        vw = iio.get_writer(str(a.video), fps=35 // 4, codec="libx264", quality=7, macro_block_size=None)
        writer = lambda frame, info, act: vw.append_data(frame)
    nav = Navigator()
    res = []
    t0 = time.time()
    for e in range(a.episodes):
        ep = run_episode(env, nav, a.seed_base + e, writer)
        res.append(ep)
        print(json.dumps({k: ep[k] for k in ("seed", "exited", "dead", "timeout", "tics", "best_progress", "kills")}), flush=True)
    if writer is not None:
        vw.close()
    ex = np.array([r["exited"] for r in res]); prog = np.array([r["best_progress"] for r in res])
    print(f"navigator on {a.map} skill {a.skill}: exited {ex.sum()}/{len(ex)}, mean best progress {prog.mean():.2f}, "
          f"mean tics {np.mean([r['tics'] for r in res]):.0f}, {time.time() - t0:.0f}s")
    env.close()


if __name__ == "__main__":
    main()
