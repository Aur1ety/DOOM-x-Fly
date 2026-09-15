"""Gymnasium-style wrapper over ViZDoom 1.3.0 for the LOBULA LOOP harness.

Fixed harness choices:
- headless, RES_160X120, one RGB24 render per frame; GRAY8 is derived with BT.709 weights
  (measured: identical to the engine's own GRAY8 to within +/-1 grey level).
- horizontal FOV 108 deg: the engine has no persistent FOV setting, so `fov 108` is sent as a
  console command after every `new_episode` and 3 no-op warm-up tics are advanced while the
  engine interpolates 90 -> 108 at 7 deg/tic (measured). Warm-up reward is discarded.
- frameskip 4; `all_frames=True` renders every intermediate tic (about 4x slower) so the
  front end can cross-fade.
- 7 discrete actions mapped onto each scenario's own button set; actions whose button the
  scenario does not expose are executed as no-ops (see `DoomEnv.noop_actions`).
- seeds: train 0..999, eval >= 10000. `check_seed_variation` measures that seeds move spawns.
- scenarios with no episode timeout in their cfg (take_cover, defend_the_line) get 2100 tics.
  `survival_tics` is the engine's episode_time, which includes episode_start_time (and the 3
  warm-up tics); the timeout is counted from episode_start_time (measured: basic ends at 314).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import vizdoom as vd
except ImportError:  # laptop never runs this; keeps the module importable for docs/tests
    vd = None

WIDTH, HEIGHT = 160, 120
ACTIONS = ("turn_left", "turn_right", "move_forward", "move_backward",
           "strafe_left", "strafe_right", "attack")
N_ACTIONS = len(ACTIONS)
SCENARIOS = ("take_cover", "defend_the_line", "defend_the_center", "basic")
TRAIN_SEED_LO, TRAIN_SEED_HI = 0, 999
EVAL_SEED_LO = 10000
DEFAULT_TIMEOUT = 2100  # tics; applied only where the scenario cfg leaves it unbounded
TICRATE = 35

# Label object names seen in the four scenarios (measured) plus the rest of the Doom bestiary.
ENEMY_NAMES = frozenset({
    "DoomImp", "Demon", "Spectre", "MarineChainsawVzd", "Cacodemon", "Zombieman", "ShotgunGuy",
    "ChaingunGuy", "HellKnight", "BaronOfHell", "LostSoul", "Revenant", "Arachnotron", "Fatso",
    "PainElemental", "Archvile", "Cyberdemon", "SpiderMastermind", "WolfensteinSS",
})
PROJECTILE_NAMES = frozenset({
    "DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "PlasmaBall", "RevenantTracer",
    "ArachnotronPlasma", "FatShot", "BFGBall",
})
IGNORE_NAMES = frozenset({"DoomPlayer", "Blood", "BulletPuff", "TeleportFog", "ItemFog",
                          "BloodSplatter", "Clip", "Shell", "Medikit", "Stimpack"})
GRAY_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)  # BT.709, sums to 1


def _button_map() -> dict[str, Any]:
    b = vd.Button
    return {"turn_left": b.TURN_LEFT, "turn_right": b.TURN_RIGHT, "move_forward": b.MOVE_FORWARD,
            "move_backward": b.MOVE_BACKWARD, "strafe_left": b.MOVE_LEFT,
            "strafe_right": b.MOVE_RIGHT, "attack": b.ATTACK}


def scenario_cfg(scenario: str, scenarios_dir: str | os.PathLike | None = None) -> Path:
    """Path of `<scenario>.cfg` (default: the vizdoom package's scenarios directory)."""
    root = Path(scenarios_dir) if scenarios_dir else Path(vd.scenarios_path)
    cfg = root / f"{scenario}.cfg"
    if not cfg.exists():
        raise FileNotFoundError(cfg)
    return cfg


def stage_to_local(dst: str | os.PathLike = "/tmp/flybrain_vizdoom") -> Path:
    """Copy the scenario cfg/wad files to a node-local dir (64 workers must not read
    WADs off NFS). Returns the dir to pass as `scenarios_dir`."""
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(vd.scenarios_path)
    for s in SCENARIOS:
        for ext in (".cfg", ".wad"):
            f = src / f"{s}{ext}"
            if f.exists() and not (dst / f.name).exists():
                shutil.copy2(f, dst / f.name)
    return dst


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """RGB24 [..., 3] uint8 -> GRAY8 [...] uint8 with BT.709 weights (within +/-1 of the engine's
    own GRAY8). float32 dot + 0.5 truncation measured fastest (35 us/frame) of six variants."""
    return (rgb.astype(np.float32) @ GRAY_WEIGHTS + 0.5).astype(np.uint8)


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def target_geometry(px: float, py: float, pang: float, pvx: float, pvy: float,
                    ox: float, oy: float, ovx: float, ovy: float) -> tuple[float, float, float]:
    """(azimuth_deg, distance, ttc_tics) of an object relative to the player.

    azimuth > 0 means the object is to the RIGHT of the view direction (screen-right; Doom
    angles grow counter-clockwise). ttc = distance / closing speed, inf when not closing.
    Velocities are per tic (Doom units), as the engine reports them.
    """
    dx, dy = ox - px, oy - py
    dist = math.hypot(dx, dy)
    bearing = math.degrees(math.atan2(dy, dx))
    az = wrap_deg(pang - bearing)
    rvx, rvy = ovx - pvx, ovy - pvy
    closing = -(dx * rvx + dy * rvy) / dist if dist > 1e-6 else 0.0
    ttc = dist / closing if closing > 1e-6 else math.inf
    return az, dist, ttc


def parse_labels(labels, pose: tuple[float, float, float], vel: tuple[float, float],
                 unknown: set | None = None) -> dict[str, float]:
    """Nearest visible enemy / projectile from the labels buffer.

    `labels`: iterable with vizdoom.Label attributes (object_name, object_position_x/y,
    object_velocity_x/y, x, width). Returns a flat dict; azimuths are in degrees (+ = right),
    distances in map units, ttc in tics (inf if not closing), `*_x` = bbox centre column (px).
    """
    px, py, pang = pose
    pvx, pvy = vel
    best = {"enemy": None, "projectile": None}
    count = {"enemy": 0, "projectile": 0}
    for lab in labels:
        name = lab.object_name
        if name in ENEMY_NAMES:
            kind = "enemy"
        elif name in PROJECTILE_NAMES:
            kind = "projectile"
        else:
            if name not in IGNORE_NAMES and unknown is not None:
                unknown.add(name)
            continue
        count[kind] += 1
        az, dist, ttc = target_geometry(px, py, pang, pvx, pvy, lab.object_position_x,
                                        lab.object_position_y, lab.object_velocity_x,
                                        lab.object_velocity_y)
        if best[kind] is None or dist < best[kind][1]:
            best[kind] = (az, dist, ttc, lab.x + lab.width / 2.0)
    out: dict[str, float] = {}
    for kind in ("enemy", "projectile"):
        b = best[kind]
        out[f"{kind}_visible"] = float(b is not None)
        out[f"{kind}_n"] = float(count[kind])
        out[f"{kind}_azimuth"] = b[0] if b else math.nan
        out[f"{kind}_distance"] = b[1] if b else math.nan
        out[f"{kind}_ttc"] = b[2] if b else math.nan
        out[f"{kind}_x"] = b[3] if b else math.nan
    return out


class DoomEnv:
    """One ViZDoom instance. obs = {"gray": uint8 [H,W] (or [frameskip,H,W] if all_frames),
    "rgb": uint8 [H,W,3] (or [frameskip,H,W,3])}. step -> (obs, reward, terminated, truncated,
    info). At episode end `info["episode"]` holds reward / kills / survival_tics / steps."""

    def __init__(self, scenario: str = "take_cover", frameskip: int = 4, all_frames: bool = False,
                 sticky_prob: float = 0.0, fov: float | None = 108.0, fov_warmup_tics: int = 3,
                 episode_timeout: int | None = None, rgb: bool = True, labels: bool = True,
                 seed: int | None = None, doom_skill: int | None = None,
                 scenarios_dir: str | os.PathLike | None = None):
        if vd is None:
            raise ImportError("vizdoom is not installed (this module runs on the server)")
        if scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}, got {scenario!r}")
        self.scenario = scenario
        self.frameskip = int(frameskip)
        self.all_frames = bool(all_frames)
        self.sticky_prob = float(sticky_prob)
        self.fov = fov
        self.fov_warmup_tics = int(fov_warmup_tics) if fov is not None else 0
        self.want_rgb = bool(rgb)
        self.want_labels = bool(labels)
        self.unknown_label_names: set[str] = set()

        g = vd.DoomGame()
        g.load_config(str(scenario_cfg(scenario, scenarios_dir)))
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_console_enabled(False)
        g.set_mode(vd.Mode.PLAYER)
        g.set_screen_resolution(vd.ScreenResolution.RES_160X120)
        g.set_screen_format(vd.ScreenFormat.RGB24)
        g.set_render_hud(False)
        g.set_labels_buffer_enabled(self.want_labels)
        g.set_objects_info_enabled(self.want_labels)
        if doom_skill is not None:
            g.set_doom_skill(int(doom_skill))
        self.cfg_timeout = g.get_episode_timeout()
        if episode_timeout is not None:
            g.set_episode_timeout(int(episode_timeout))
        elif self.cfg_timeout == 0:
            g.set_episode_timeout(DEFAULT_TIMEOUT)
        self.episode_timeout = g.get_episode_timeout()
        self.episode_start_time = g.get_episode_start_time()
        gv = vd.GameVariable
        wanted = [gv.HEALTH, gv.KILLCOUNT, gv.POSITION_X, gv.POSITION_Y, gv.ANGLE, gv.VELOCITY_X,
                  gv.VELOCITY_Y, gv.CAMERA_FOV, gv.DEAD, gv.AMMO2]
        have = list(g.get_available_game_variables())
        g.set_available_game_variables(have + [v for v in wanted if v not in have])
        self.game = g

        self.buttons = list(g.get_available_buttons())
        self.button_names = tuple(b.name for b in self.buttons)
        bmap = _button_map()
        self._button_vectors = []
        for a in ACTIONS:
            vec = [0] * len(self.buttons)
            if bmap[a] in self.buttons:
                vec[self.buttons.index(bmap[a])] = 1
            self._button_vectors.append(vec)
        self.noop_actions = tuple(a for a, v in zip(ACTIONS, self._button_vectors) if not any(v))
        self.available_actions = tuple(a for a in ACTIONS if a not in self.noop_actions)
        # index of an action that presses nothing in this scenario (None if all 7 are live)
        self.noop_action = ACTIONS.index(self.noop_actions[0]) if self.noop_actions else None
        self._noop = [0] * len(self.buttons)

        self._seed_counter = int(seed) if seed is not None else TRAIN_SEED_LO
        self._rng = np.random.default_rng(self._seed_counter)
        self._prev_action: int | None = None
        self._last_obs = None
        self.last_state = None
        self._ep = {}
        g.init()
        self.episode_seed: int | None = None

    # -- spaces (gymnasium optional) -------------------------------------------------------
    @property
    def observation_shape(self) -> dict[str, tuple[int, ...]]:
        f = (self.frameskip,) if self.all_frames else ()
        shp = {"gray": f + (HEIGHT, WIDTH)}
        if self.want_rgb:
            shp["rgb"] = f + (HEIGHT, WIDTH, 3)
        return shp

    @property
    def action_space(self):
        from gymnasium.spaces import Discrete
        return Discrete(N_ACTIONS)

    @property
    def observation_space(self):
        from gymnasium.spaces import Box, Dict
        return Dict({k: Box(0, 255, shape=s, dtype=np.uint8) for k, s in self.observation_shape.items()})

    # -- helpers ---------------------------------------------------------------------------
    def button_vector(self, action: int) -> list[int]:
        return list(self._button_vectors[int(action)])

    def _var(self, name: str) -> float:
        return float(self.game.get_game_variable(getattr(vd.GameVariable, name)))

    def _pose(self) -> tuple[tuple[float, float, float], tuple[float, float]]:
        return ((self._var("POSITION_X"), self._var("POSITION_Y"), self._var("ANGLE")),
                (self._var("VELOCITY_X"), self._var("VELOCITY_Y")))

    def _grab(self, state) -> tuple[np.ndarray, np.ndarray | None]:
        rgb = np.asarray(state.screen_buffer)
        gray = rgb_to_gray(rgb)
        return gray, (rgb.copy() if self.want_rgb else None)

    def _pack(self, grays: list[np.ndarray], rgbs: list) -> dict[str, np.ndarray]:
        if self.all_frames:
            while len(grays) < self.frameskip:  # episode ended mid-skip: repeat last frame
                grays.append(grays[-1])
                rgbs.append(rgbs[-1])
            obs = {"gray": np.stack(grays)}
            if self.want_rgb:
                obs["rgb"] = np.stack(rgbs)
        else:
            obs = {"gray": grays[-1]}
            if self.want_rgb:
                obs["rgb"] = rgbs[-1]
        return obs

    def _info(self, state) -> dict[str, Any]:
        info: dict[str, Any] = {
            "tic": int(self.game.get_episode_time()), "health": self._var("HEALTH"),
            "kills": self._var("KILLCOUNT"), "ammo": self._var("AMMO2"), "fov": self._var("CAMERA_FOV"),
        }
        pose, vel = self._pose()
        info["pos"] = pose[:2]
        info["angle"] = pose[2]
        if self.want_labels:
            labels = state.labels if state is not None else []
            info["labels"] = parse_labels(labels, pose, vel, self.unknown_label_names)
        return info

    # -- API -------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if seed is None:
            seed = self._seed_counter
            self._seed_counter += 1
        self.episode_seed = int(seed)
        self._rng = np.random.default_rng([int(seed), 0xD00])
        self._prev_action = None
        g = self.game
        g.set_seed(int(seed))
        g.new_episode()
        warm = 0.0
        if self.fov is not None:
            g.send_game_command(f"fov {self.fov:g}")
            if self.fov_warmup_tics > 0:
                g.advance_action(self.fov_warmup_tics, True)
                warm = float(g.get_total_reward())
        state = g.get_state()
        if state is None:
            raise RuntimeError("episode finished during FOV warm-up; lower fov_warmup_tics")
        self.last_state = state
        gray, rgb = self._grab(state)
        obs = self._pack([gray], [rgb])
        self._last_obs = obs
        self._ep = {"reward": 0.0, "steps": 0, "warmup_reward": warm, "seed": self.episode_seed,
                    "action_hist": np.zeros(N_ACTIONS, dtype=np.int64)}
        info = self._info(state)
        info["seed"] = self.episode_seed
        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        g = self.game
        action = int(action)
        sticky = False
        if self._prev_action is not None and self.sticky_prob > 0 and self._rng.random() < self.sticky_prob:
            action, sticky = self._prev_action, True
        self._prev_action = action
        buttons = self._button_vectors[action]
        reward = 0.0
        grays, rgbs = [], []
        state = None
        if self.all_frames:
            for _ in range(self.frameskip):
                reward += float(g.make_action(buttons, 1))
                if g.is_episode_finished():
                    break
                state = g.get_state()
                gr, rg = self._grab(state)
                grays.append(gr)
                rgbs.append(rg)
        else:
            reward += float(g.make_action(buttons, self.frameskip))
            if not g.is_episode_finished():
                state = g.get_state()
                gr, rg = self._grab(state)
                grays.append(gr)
                rgbs.append(rg)
        finished = g.is_episode_finished()
        if grays:
            obs = self._pack(grays, rgbs)
            self._last_obs = obs
            self.last_state = state
        else:  # no frame rendered after the terminal tic: repeat the last observation
            obs = self._last_obs
        timeout = bool(g.is_episode_timeout_reached())
        dead = bool(g.is_player_dead())
        terminated = finished and (dead or not timeout)
        truncated = finished and timeout and not dead
        self._ep["reward"] += reward
        self._ep["steps"] += 1
        self._ep["action_hist"][action] += 1
        info = self._info(state)
        info["action_taken"] = action
        info["sticky"] = sticky
        if finished:
            info["episode"] = {
                "reward": float(self._ep["reward"]), "kills": self._var("KILLCOUNT"),
                "survival_tics": int(g.get_episode_time()), "steps": int(self._ep["steps"]),
                "dead": dead, "timeout": timeout, "seed": self._ep["seed"],
                "warmup_reward": self._ep["warmup_reward"],
                "action_hist": self._ep["action_hist"].tolist(),
            }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if getattr(self, "game", None) is not None:
            self.game.close()
            self.game = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# -- seed hygiene ----------------------------------------------------------------------------
def spawn_snapshot(env: DoomEnv, seed: int, steps: int = 15) -> tuple:
    """Positions of all non-player objects after `steps` no-op decisions from `seed`."""
    env.reset(seed=seed)
    idle = env.noop_action if env.noop_action is not None else ACTIONS.index("move_backward")
    for _ in range(steps):
        _, _, term, trunc, _ = env.step(idle)
        if term or trunc:
            break
    st = env.last_state
    objs = tuple(sorted((o.name, round(o.position_x, 1), round(o.position_y, 1))
                        for o in st.objects if o.name != "DoomPlayer")) if st is not None else ()
    return (round(env._var("POSITION_X"), 1), round(env._var("POSITION_Y"), 1), objs)


def check_seed_variation(scenario: str, seeds: list[int] | None = None, steps: int = 15,
                         **env_kwargs) -> dict[str, Any]:
    """Measure that seeds change spawn/monster positions and that a seed reproduces exactly."""
    seeds = list(seeds) if seeds is not None else list(range(EVAL_SEED_LO, EVAL_SEED_LO + 10))
    with DoomEnv(scenario, **env_kwargs) as env:
        snaps = [spawn_snapshot(env, s, steps) for s in seeds]
        rep = [spawn_snapshot(env, seeds[0], steps) for _ in range(3)]
    return {"scenario": scenario, "n_seeds": len(seeds), "n_unique": len(set(map(repr, snaps))),
            "reproducible": all(r == snaps[0] for r in rep), "steps": steps}


def bench_single(scenario: str, seconds: float = 10.0, **env_kwargs) -> dict[str, float]:
    """Agent-steps/s of one env with random actions, plus RSS of this process and its engine."""
    import psutil
    env = DoomEnv(scenario, **env_kwargs)
    rng = np.random.default_rng(0)
    env.reset(seed=0)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        _, _, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
        n += 1
        if term or trunc:
            env.reset()
    dt = time.perf_counter() - t0
    me = psutil.Process()
    kids = me.children(recursive=True)
    out = {"scenario": scenario, "steps_per_s": n / dt, "steps": n, "seconds": dt,
           "rss_python_mb": me.memory_info().rss / 2**20,
           "rss_engine_mb": sum(k.memory_info().rss for k in kids) / 2**20}
    env.close()
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="ViZDoom wrapper checks")
    p.add_argument("--scenario", default="take_cover", choices=SCENARIOS)
    p.add_argument("--check-seeds", action="store_true", help="seed -> spawn variation report")
    p.add_argument("--bench", action="store_true", help="single-env steps/s and RSS")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--all-frames", action="store_true")
    p.add_argument("--labels", action="store_true", help="print label names seen in one episode")
    a = p.parse_args(argv)
    if a.check_seeds:
        print(check_seed_variation(a.scenario))
    if a.bench:
        print(bench_single(a.scenario, a.seconds, all_frames=a.all_frames))
    if a.labels:
        with DoomEnv(a.scenario) as env:
            env.reset(seed=EVAL_SEED_LO)
            names: dict[str, int] = {}
            rng = np.random.default_rng(0)
            while True:
                _, _, term, trunc, _ = env.step(int(rng.integers(N_ACTIONS)))
                for lab in (env.last_state.labels if env.last_state else []):
                    names[lab.object_name] = names.get(lab.object_name, 0) + 1
                if term or trunc:
                    break
            print({"scenario": a.scenario, "label_names": names, "buttons": env.button_names,
                   "noop_actions": env.noop_actions, "timeout": env.episode_timeout})
    if not (a.check_seeds or a.bench or a.labels):
        p.print_help()


if __name__ == "__main__":
    main()
