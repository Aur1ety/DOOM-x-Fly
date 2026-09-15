"""Tests for flybrain.env (doom wrapper, pool, floors) and the baseline PPO loop. Server only."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

vd = pytest.importorskip("vizdoom")

from flybrain.env import floors  # noqa: E402
from flybrain.env.doom import (ACTIONS, EVAL_SEED_LO, HEIGHT, N_ACTIONS, WIDTH, DoomEnv,  # noqa: E402
                               check_seed_variation, parse_labels, rgb_to_gray, target_geometry)
from flybrain.env.pool import EnvPool, SeedStream  # noqa: E402

EXPECTED_BUTTONS = {
    "take_cover": ("MOVE_LEFT", "MOVE_RIGHT"),
    "defend_the_line": ("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    "defend_the_center": ("TURN_LEFT", "TURN_RIGHT", "ATTACK"),
    "basic": ("MOVE_LEFT", "MOVE_RIGHT", "ATTACK"),
}
EXPECTED_LIVE = {
    "take_cover": ("strafe_left", "strafe_right"),
    "defend_the_line": ("turn_left", "turn_right", "attack"),
    "defend_the_center": ("turn_left", "turn_right", "attack"),
    "basic": ("strafe_left", "strafe_right", "attack"),
}


@pytest.fixture(scope="module")
def basic_env():
    env = DoomEnv("basic")
    yield env
    env.close()


# -- wrapper ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("scenario", list(EXPECTED_BUTTONS))
def test_action_mapping(scenario):
    with DoomEnv(scenario) as env:
        assert env.button_names == EXPECTED_BUTTONS[scenario]
        assert env.available_actions == EXPECTED_LIVE[scenario]
        for i, a in enumerate(ACTIONS):
            vec = env.button_vector(i)
            assert len(vec) == len(env.button_names)
            if a in env.noop_actions:
                assert sum(vec) == 0
            else:
                assert sum(vec) == 1
        assert env.episode_timeout > 0  # unbounded cfgs get 2100


def test_reset_step_shapes(basic_env):
    obs, info = basic_env.reset(seed=EVAL_SEED_LO)
    assert obs["gray"].shape == (HEIGHT, WIDTH) and obs["gray"].dtype == np.uint8
    assert obs["rgb"].shape == (HEIGHT, WIDTH, 3) and obs["rgb"].dtype == np.uint8
    assert info["fov"] == pytest.approx(108.0)  # FOV command applied during warm-up
    assert "labels" in info and info["seed"] == EVAL_SEED_LO
    obs, r, term, trunc, info = basic_env.step(ACTIONS.index("strafe_left"))
    assert obs["gray"].shape == (HEIGHT, WIDTH) and isinstance(r, float)
    assert not term and not trunc
    assert info["action_taken"] == ACTIONS.index("strafe_left") and info["sticky"] is False
    assert np.abs(rgb_to_gray(obs["rgb"]).astype(int) - obs["gray"].astype(int)).max() == 0


def test_all_frames_shapes():
    with DoomEnv("basic", all_frames=True, frameskip=4) as env:
        obs, _ = env.reset(seed=1)
        assert obs["gray"].shape == (4, HEIGHT, WIDTH)
        obs, *_ = env.step(ACTIONS.index("strafe_left"))
        assert obs["gray"].shape == (4, HEIGHT, WIDTH) and obs["rgb"].shape == (4, HEIGHT, WIDTH, 3)
        # consecutive intermediate frames are not all identical (the engine really rendered 4)
        assert any(not np.array_equal(obs["gray"][i], obs["gray"][i + 1]) for i in range(3))


def test_episode_ends_with_stats(basic_env):
    rng = np.random.default_rng(0)
    basic_env.reset(seed=5)
    for _ in range(200):
        _, _, term, trunc, info = basic_env.step(int(rng.integers(N_ACTIONS)))
        if term or trunc:
            break
    assert term or trunc
    ep = info["episode"]
    for k in ("reward", "kills", "survival_tics", "steps", "dead", "timeout", "seed", "action_hist"):
        assert k in ep
    # basic: 300-tic timeout counted from episode_start_time (14), +frameskip slack
    assert ep["steps"] <= 80 and ep["survival_tics"] <= 300 + 14 + 4
    assert sum(ep["action_hist"]) == ep["steps"]


def test_seed_reproducibility():
    def run(seed, env):
        env.reset(seed=seed)
        frames, labs = [], []
        for t in range(12):
            obs, _, term, trunc, info = env.step(t % 3)
            frames.append(obs["gray"].copy())
            labs.append(info["labels"])
            if term or trunc:
                break
        return frames, labs
    with DoomEnv("defend_the_line") as env:
        f1, l1 = run(EVAL_SEED_LO, env)
        f2, l2 = run(EVAL_SEED_LO, env)
        assert all(np.array_equal(a, b) for a, b in zip(f1, f2)) and l1 == l2
    r = check_seed_variation("defend_the_line", list(range(EVAL_SEED_LO, EVAL_SEED_LO + 6)))
    assert r["reproducible"] and r["n_unique"] >= 4, r


def test_sticky_actions():
    with DoomEnv("basic", sticky_prob=1.0) as env:
        env.reset(seed=3)
        _, _, _, _, info = env.step(ACTIONS.index("strafe_left"))
        assert info["sticky"] is False  # nothing to repeat yet
        for _ in range(5):
            _, _, _, _, info = env.step(ACTIONS.index("strafe_right"))
            assert info["sticky"] is True and info["action_taken"] == ACTIONS.index("strafe_left")


# -- labels ---------------------------------------------------------------------------------------
def test_target_geometry():
    # player at origin facing east (0 deg); object north-east -> bearing 45 -> azimuth -45 (left)
    az, dist, ttc = target_geometry(0, 0, 0, 0, 0, 100, 100, 0, 0)
    assert az == pytest.approx(-45) and dist == pytest.approx(math.hypot(100, 100)) and ttc == math.inf
    # object straight ahead at 200 units approaching at 4 units/tic -> ttc 50 tics
    az, dist, ttc = target_geometry(0, 0, 0, 0, 0, 200, 0, -4, 0)
    assert az == pytest.approx(0) and ttc == pytest.approx(50)
    # facing north (90): object east -> right of view -> positive azimuth
    az, *_ = target_geometry(0, 0, 90, 0, 0, 100, 0, 0, 0)
    assert az == pytest.approx(90)


def test_parse_labels_synthetic():
    def lab(name, x, y, vx=0.0, vy=0.0, sx=70, w=20):
        return SimpleNamespace(object_name=name, object_position_x=x, object_position_y=y,
                               object_velocity_x=vx, object_velocity_y=vy, x=sx, width=w)
    unknown = set()
    out = parse_labels([lab("DoomImp", 300, 0), lab("DoomImp", 100, 50), lab("DoomImpBall", 50, 0, -5, 0),
                        lab("DoomPlayer", 0, 0), lab("Mystery", 1, 1)], (0, 0, 0), (0, 0), unknown)
    assert out["enemy_n"] == 2 and out["enemy_distance"] == pytest.approx(math.hypot(100, 50))
    assert out["enemy_azimuth"] < 0 and out["enemy_x"] == 80
    assert out["projectile_visible"] == 1 and out["projectile_ttc"] == pytest.approx(10)
    assert unknown == {"Mystery"}
    empty = parse_labels([], (0, 0, 0), (0, 0))
    assert empty["enemy_visible"] == 0 and math.isnan(empty["enemy_azimuth"])


def test_labels_live_sign_matches_bbox():
    agree = total = 0
    with DoomEnv("defend_the_line") as env:
        env.reset(seed=EVAL_SEED_LO + 3)
        for _ in range(40):
            _, _, term, trunc, info = env.step(ACTIONS.index("turn_left"))
            lab = info["labels"]
            if lab["enemy_visible"]:
                total += 1
                agree += (lab["enemy_azimuth"] > 0) == (lab["enemy_x"] > WIDTH / 2)
            if term or trunc:
                break
    assert total >= 10 and agree / total >= 0.9


# -- pool -----------------------------------------------------------------------------------------
def test_seed_stream():
    s = SeedStream(0, 9, env=1, n_envs=4)
    assert [s.next() for _ in range(4)] == [1, 5, 9, 3]


def test_pool_roundtrip_matches_single_env():
    n = 2
    seeds = [7, 8]
    actions = np.array([[0, 1, 2, 6, 4, 5, 3, 4] * 2, [4, 5, 4, 5, 6, 6, 0, 1] * 2])
    with EnvPool(n, "basic", ring_len=3) as pool:
        obs, infos = pool.reset(seeds)
        assert obs["gray"].shape == (n, HEIGHT, WIDTH) and obs["rgb"].shape == (n, HEIGHT, WIDTH, 3)
        assert [i["seed"] for i in infos] == seeds
        first = obs["gray"].copy()
        pooled = []
        for t in range(actions.shape[1]):
            obs, r, term, trunc, infos = pool.step(actions[:, t])
            pooled.append((obs["gray"].copy(), r.copy(), np.logical_or(term, trunc)))
        assert pool.env_attr("noop_actions") == ("turn_left", "turn_right", "move_forward", "move_backward")
    # a single env with the same seed and actions must reproduce the worker's frames and rewards
    # up to the episode end (after which the worker has auto-reset to a fresh episode)
    with DoomEnv("basic") as env:
        for w in range(n):
            o, _ = env.reset(seed=seeds[w])
            assert np.array_equal(o["gray"], first[w])
            compared = 0
            for t in range(actions.shape[1]):
                o, r, term, trunc, _ = env.step(actions[w, t])
                assert r == pooled[t][1][w] and (term or trunc) == pooled[t][2][w]
                if term or trunc:
                    break
                assert np.array_equal(o["gray"], pooled[t][0][w])
                compared += 1
            assert compared >= 3


def test_pool_envs_per_worker_matches_flat():
    """2 workers x 2 envs must produce the same frames as 4 workers x 1 env (same seeds/actions)."""
    seeds = [11, 12, 13, 14]
    acts = np.random.default_rng(1).integers(0, N_ACTIONS, size=(6, 4))
    outs = []
    for nw, k in ((4, 1), (2, 2)):
        with EnvPool(nw, "basic", envs_per_worker=k, ring_len=2) as pool:
            obs, infos = pool.reset(seeds)
            assert pool.n == 4 and [i["seed"] for i in infos] == seeds
            frames = [obs["gray"].copy()]
            for t in range(acts.shape[0]):
                obs, r, term, trunc, infos = pool.step(acts[t])
                frames.append(obs["gray"].copy())
            outs.append((frames, r.copy()))
    assert all(np.array_equal(a, b) for a, b in zip(outs[0][0], outs[1][0]))
    assert np.array_equal(outs[0][1], outs[1][1])


def test_pool_autoreset():
    with EnvPool(2, "basic", ring_len=2) as pool:
        pool.reset([1, 2])
        finals = 0
        for _ in range(90):  # basic times out at 300 tics = 75 decisions
            obs, r, term, trunc, infos = pool.step(np.array([3, 3]))
            for i, d in zip(infos, np.logical_or(term, trunc)):
                if d:
                    finals += 1
                    assert "final_info" in i and "episode" in i["final_info"]
                    assert i["tic"] < i["final_info"]["tic"]  # a fresh episode came back
        assert finals >= 2


# -- floors ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("policy", ["random", "spin_fire", "constant_forward", "centroid", "constant_attack"])
def test_floors_two_episodes(policy):
    r = floors.evaluate(policy, "basic", n_episodes=2, seed_base=EVAL_SEED_LO)
    assert r["n_episodes"] == 2 and len(r["reward"]) == 2 and r["seeds"] == [EVAL_SEED_LO, EVAL_SEED_LO + 1]
    s = r["reward_summary"]
    assert s["n"] == 2 and math.isfinite(s["iqm"]) and len(s["iqm_ci95"]) == 2
    assert sum(r["action_hist"]) == sum(r["steps"])


def test_floors_parallel_matches_serial():
    a = floors.evaluate("random", "basic", n_episodes=2, workers=1)
    b = floors.evaluate("random", "basic", n_episodes=2, workers=2)
    assert a["reward"] == b["reward"] and a["survival_tics"] == b["survival_tics"]


def test_centroid_on_take_cover_and_summarize():
    with DoomEnv("take_cover") as env:
        ep = floors.run_episode(env, floors.CentroidPolicy(), EVAL_SEED_LO)
    assert ep["steps"] > 0 and ep["survival_tics"] <= 2100 + 4
    s = floors.summarize(np.arange(8.0))
    assert s["iqm"] == pytest.approx(np.mean([2, 3, 4, 5])) and s["median"] == 3.5


# -- baseline loop (tiny; never a result) ------------------------------------------------------
def test_baseline_loop_executes(tmp_path, monkeypatch):
    from flybrain.train import baseline_cnn_gru as b
    monkeypatch.setattr(b, "OUT_DIR", tmp_path)
    cfg = b.Config(scenario="basic", envs=2, rollout=6, updates=2, minibatches=1, threads=1,
                   time_limit=120, run="test", ckpt_every=1000)
    res = b.Trainer(cfg).train()
    assert res["updates_done"] == 2 and res["steps_done"] == 24 and res["loss_finite"]
    assert (tmp_path / "baseline" / "test" / "ckpt.pt").exists()
    pol = b.load_policy(tmp_path / "baseline" / "test" / "ckpt.pt")
    with DoomEnv("basic") as env:
        ep = floors.run_episode(env, pol, EVAL_SEED_LO)
    assert ep["steps"] > 0
