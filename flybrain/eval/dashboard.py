"""Dashboard demo video: the frozen MaleCNS wiring playing Doom E1M1, with the simulated brain on screen.

Layout (1920x1080, 35 fps = every game tic; the agent decides every 4th tic):
  header     title + one line on what is simulated, what is hand-built and what is trained
  left       the real game view (640x480 render, x2)
  top right  the simulated neurons' cell bodies at their scanned MaleCNS positions (front view, fixed);
             brightness = each neuron's simulated rate above its typical level, where the typical level and
             spread are measured once on a separate calibration run (no running average, no reset flash);
             cells driven directly by the eye model are tinted blue-grey
  bottom     E1M1 map with the walked path | the decision layer's probability for each of the 8 buttons
  footer     run, game time, route progress, kills, health, outcome; right: how the shown runs were chosen

Everything drawn is measured from the running simulation; nothing is decorative.

    python -m flybrain.eval.dashboard --ckpt $FLYBRAIN_OUT/e1m1/v2_r4/student.pt --seeds 40000 40001 --out $FLYBRAIN_OUT/videos/e1m1_dashboard.mp4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import DATA_DIR
from flybrain.env.level import ACTIONS, _down

W, H = 1920, 1080
HEAD, FOOT = 72, 56
GAME = (0, HEAD, 1280, 960)                     # x, y, w, h
BRAIN = (1280, HEAD, 640, 450)
MAP = (1280, HEAD + 450, 640, 262)
DECIDE = (1280, HEAD + 712, 640, 248)
PRETTY = {"move_forward": "forward", "move_backward": "back", "turn_left": "turn left", "turn_right": "turn right",
          "strafe_left": "strafe left", "strafe_right": "strafe right", "attack": "shoot", "use": "use / open"}
BG = (8, 10, 14)
DIM = (120, 130, 145)


class FFmpegWriter:
    """Raw RGB frames -> H.264 MP4 through the ffmpeg binary bundled with imageio-ffmpeg. Avoids imageio's exe probe,
    which writes to /dev/null (not writable for users on node1)."""

    def __init__(self, path: Path, w: int, h: int, fps: int, crf: int = 18):
        import glob
        import subprocess
        import imageio_ffmpeg
        exe = sorted(glob.glob(str(Path(imageio_ffmpeg.__file__).parent / "binaries" / "ffmpeg-*")))[-1]
        self.log = open(str(path) + ".ffmpeg.log", "wb")
        self.p = subprocess.Popen([exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
                                   "-i", "pipe:0", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                                   "-threads", "4", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE, stdout=self.log, stderr=self.log)

    def append_data(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close(); rc = self.p.wait(); self.log.close()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}; see {self.log.name}")


def _font(size: int):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def soma_positions(sub: dict, annotations: Path) -> tuple[np.ndarray, np.ndarray]:
    """(node indices with a known soma, their xyz) for the subgraph's nodes (MaleCNS voxel coordinates)."""
    import pyarrow.feather as feather
    t = feather.read_table(annotations, columns=["bodyId", "somaLocation"]).to_pandas().drop_duplicates("bodyId").set_index("bodyId")
    v = t["somaLocation"].reindex(np.asarray(sub["bodyId"]))
    ok = np.array([x is not None and not isinstance(x, float) and len(x) == 3 for x in v])
    return np.flatnonzero(ok), np.array([x for x in v[ok]], float)


class BrainPanel:
    """Front view of the simulated neurons' cell bodies at their scanned positions.

    Brightness of a dot = the neuron's simulated rate above its typical level, where "typical" (mean and
    spread per neuron) is measured once on a separate calibration run, so nothing resets or drifts during
    the video. Cells driven directly by the eye model (the clamped optic-lobe cells) are tinted blue-grey.
    """

    def __init__(self, nodes: np.ndarray, xyz: np.ndarray, is_clamped: np.ndarray, w: int, h: int):
        self.w, self.h = w, h
        lo, hi = np.percentile(xyz, 0.5, 0), np.percentile(xyz, 99.5, 0)
        keep = np.all((xyz >= lo) & (xyz <= hi), 1)                 # a few far outliers (neck) would shrink the view
        self.nodes, xyz = nodes[keep], xyz[keep]
        self.eye = is_clamped[self.nodes]
        c = (lo + hi) / 2
        p = (xyz - c) / (hi - lo)[[0, 1]].max()                      # normalised; scan x = left-right, y = dorsal-ventral
        s = 0.80 * self.w
        i = np.clip((p[:, 1] * s + self.h / 2 - 20).astype(int), 0, self.h - 1)
        j = np.clip((p[:, 0] * s + self.w / 2).astype(int), 0, self.w - 1)
        self.flat = i * self.w + j
        n = self.w * self.h
        self.base_dyn = np.bincount(self.flat[~self.eye], minlength=n).reshape(self.h, self.w).astype(np.float32)
        self.base_eye = np.bincount(self.flat[self.eye], minlength=n).reshape(self.h, self.w).astype(np.float32)
        self.mean = None; self.sd = None
        self.n_shown = len(self.nodes)

    def calibrate(self, rate_rows: np.ndarray) -> None:
        """rate_rows: [T, n_nodes_total] rates from a calibration run -> per-neuron typical level and spread."""
        r = rate_rows[:, self.nodes]
        self.mean = r.mean(0); self.sd = r.std(0) + 1e-3

    def activity(self, rates: np.ndarray) -> np.ndarray:
        """0..1 per shown neuron: rate above typical, in units of 3 spreads (0 = at or below typical)."""
        return np.clip((rates[self.nodes] - self.mean) / (3.0 * self.sd), 0.0, 1.0)

    def render(self, act: np.ndarray) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        hot = np.bincount(self.flat, weights=act, minlength=self.w * self.h).reshape(self.h, self.w)
        img = np.zeros((self.h, self.w, 3), np.float32)
        img[..., 0] = self.base_dyn * 0.09 + self.base_eye * 0.06 + hot * 1.0
        img[..., 1] = self.base_dyn * 0.09 + self.base_eye * 0.08 + hot * 0.85
        img[..., 2] = self.base_dyn * 0.09 + self.base_eye * 0.16 + hot * 0.35
        img = gaussian_filter(img, sigma=(0.9, 0.9, 0)) * 1.6           # light blur so single cell bodies are visible
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def render(ckpt: Path, seeds: list[int], out: Path, sample: bool, wiggle: int, timeout: float, device: str,
           annotations: Path, title_note: str, stats_line: str, scan: bool = False, temperature: float = 1.0,
           calib_seed: int = 39_990, calib_decisions: int = 400) -> list[dict]:
    import torch
    from PIL import Image, ImageDraw
    from flybrain.env.level import LevelEnv
    from flybrain.model.core import load_subgraph
    from flybrain.train.bc import load_student
    from flybrain.train.level_bc import SUBGRAPH, MapPanel

    st = load_student(ckpt, SUBGRAPH, device)
    sub = load_subgraph(SUBGRAPH)
    is_cl = np.asarray(sub["is_clamped"], bool)
    nodes, xyz = soma_positions(sub, annotations)
    brain = BrainPanel(nodes, xyz, is_cl, BRAIN[2], BRAIN[3])
    env = LevelEnv(keep_full=True, keep_tics=True, start_wiggle=wiggle, timeout_s=timeout)
    mp = MapPanel(env.map, env.nav, MAP[2], MAP[3])
    f_title, f_sub, f_txt, f_small = _font(30), _font(19), _font(23), _font(16)
    n_sim = int(sub["bodyId"].shape[0]); n_syn = int(np.asarray(sub["indices_pre"]).shape[0])
    n_eye = int(is_cl.sum())

    if not scan:
        # calibration: one separate run (never shown) gives each neuron's typical rate and spread during play
        torch.manual_seed(calib_seed)
        obs, info = env.reset(calib_seed); state = st.init_state(1); rows = []
        with torch.no_grad():
            for _ in range(calib_decisions):
                rows.append(st.core.rates(state["core"])[:, 0].float().cpu().numpy())
                lg = st.decide(obs["gray"][None], _down(obs["rgb"], 2)[None], state)[0]
                act = int(torch.distributions.Categorical(logits=lg / temperature).sample()) if sample else int(lg.argmax())
                obs, r, term, trunc, info = env.step(act)
                if term or trunc:
                    break
        brain.calibrate(np.stack(rows))
        print(json.dumps({"calibration_seed": calib_seed, "decisions": len(rows)}), flush=True)

    base = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(base)
    d.text((24, 10), "Doom (1993), level E1M1, played by a fruit fly's brain scan (MaleCNS v1.0)", fill=(235, 235, 235), font=f_title)
    d.text((24, 46), f"{n_sim:,} neurons and {n_syn:,} synapses simulated from the scan, unchanged.   Eye: hand-built model driving "
                     f"{n_eye:,} optic-lobe cells.   Decision layer: trained, reads {st.nodes.numel():,} neurons.   {title_note}", fill=DIM, font=f_sub)
    d.text((BRAIN[0] + 12, BRAIN[1] + 6), "Simulated neurons (rate model, not spikes)", fill=DIM, font=f_small)
    for k_, line in enumerate((f"{brain.n_shown:,} cell bodies at their scanned positions, front view",
                               "(fly cell bodies lie on the brain surface, so the view is a shell).",
                               "Brighter = rate above the neuron's typical level, measured on a separate run.",
                               "Blue-grey = cells driven directly by the eye model.")):
        d.text((BRAIN[0] + 12, BRAIN[1] + BRAIN[3] - 82 + 19 * k_), line, fill=DIM, font=f_small)
    d.text((DECIDE[0] + 12, DECIDE[1] + 6), "Decision layer: probability of each button; one is sampled every 4 game tics", fill=DIM, font=f_small)
    for x0, y0, w, h in (GAME, BRAIN, MAP, DECIDE):
        d.rectangle([x0, y0, x0 + w - 1, y0 + h - 1], outline=(46, 48, 54))
    base_np = np.asarray(base).copy()

    out.parent.mkdir(parents=True, exist_ok=True)
    vw = None if scan else FFmpegWriter(out, W, H, 35)
    stats, tic = [], 0
    t0 = time.time()
    for k, seed in enumerate(seeds):
        torch.manual_seed(seed)                      # per-run sampling seed: a scanned seed renders the same run
        obs, info = env.reset(seed); state = st.init_state(1)
        path = [(info["x"], info["y"])]; probs = np.full(len(ACTIONS), 1 / len(ACTIONS)); act = 0; done = False
        frames = [obs["full"]]
        ep_tic = 0
        while True:
            # draw the frames produced by the previous decision (or the first frame)
            if not scan:
                rates = st.core.rates(state["core"])[:, 0].float().cpu().numpy()
                activity = brain.activity(rates)
                brain_img = brain.render(activity)
            for fr in ([] if scan else frames):
                img = base_np.copy()
                g = np.repeat(np.repeat(fr, 2, 0), 2, 1)
                img[GAME[1]:GAME[1] + GAME[3], GAME[0]:GAME[0] + GAME[2]] = g
                img[BRAIN[1] + 26:BRAIN[1] + BRAIN[3] - 86, BRAIN[0] + 1:BRAIN[0] + BRAIN[2] - 1] = brain_img[26:-86, 1:-1]
                mimg = mp.render(path, info.get("x", path[-1][0]), info.get("y", path[-1][1]), info.get("angle", 0.0))
                img[MAP[1] + 1:MAP[1] + MAP[3] - 1, MAP[0] + 1:MAP[0] + MAP[2] - 1] = mimg[1:-1, 1:-1]
                pil = Image.fromarray(img); dr = ImageDraw.Draw(pil)
                for a_i, name in enumerate(ACTIONS):
                    col, row = a_i % 2, a_i // 2
                    y = DECIDE[1] + 38 + row * 50; x = DECIDE[0] + 16 + col * 316
                    chosen = a_i == act
                    dr.text((x, y), PRETTY[name], fill=(235, 235, 235) if chosen else DIM, font=f_txt)
                    bx = x + 150; bw = int(140 * probs[a_i])
                    dr.rectangle([bx, y + 6, bx + 140, y + 26], outline=(60, 62, 70))
                    dr.rectangle([bx, y + 6, bx + bw, y + 26], fill=(225, 225, 225) if chosen else (110, 112, 120))
                secs = ep_tic / 35
                outcome = ""
                if done:
                    outcome = "   exit reached" if info.get("exited") else "   died" if info.get("dead") else "   time up"
                foot = (f"run {k + 1}/{len(seeds)}   time {int(secs // 60)}:{int(secs % 60):02d}   route {100 * info.get('progress', 0.0):3.0f}%   "
                        f"kills {int(info.get('kills', 0))}   health {int(info.get('health', 0))}{outcome}")
                dr.text((24, H - FOOT + 12), foot, fill=(235, 235, 235), font=f_txt)
                dr.text((W - 24 - dr.textlength(stats_line, font=f_small), H - FOOT + 18), stats_line, fill=DIM, font=f_small)
                vw.append_data(np.asarray(pil))
                tic += 1; ep_tic += 1
            if done:
                for _ in range(0 if scan else 70):                   # hold the last frame 2 s
                    vw.append_data(np.asarray(pil))
                break
            gray, rgb = obs["gray"][None], _down(obs["rgb"], 2)[None]
            with torch.no_grad():
                lg = st.decide(gray, rgb, state)[0]
                probs = torch.softmax(lg, 0).cpu().numpy()
                act = int(torch.distributions.Categorical(logits=lg / temperature).sample()) if sample else int(lg.argmax())
            obs, r, term, trunc, info = env.step(act)
            done = term or trunc
            frames = obs["tics"] if obs["tics"] else [obs["full"]]
            if "x" in info:
                path.append((info["x"], info["y"]))
        ep = info["episode"]; stats.append(ep)
        print(json.dumps({k_: ep[k_] for k_ in ("seed", "exited", "dead", "timeout", "tics", "best_progress", "kills")} | {"render_s": round(time.time() - t0)}), flush=True)
    if vw is not None:
        vw.close()
    env.close()
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[40000, 40001, 40002])
    ap.add_argument("--greedy", action="store_true", help="argmax actions (default: sample, as in the evaluation)")
    ap.add_argument("--wiggle", type=int, default=6); ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--annotations", type=Path, default=DATA_DIR / "malecns_v1" / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    ap.add_argument("--title-note", default="")
    ap.add_argument("--stats-line", default="", help="e.g. 'exits the level in 31% of 100 test runs'")
    ap.add_argument("--scan", action="store_true", help="play the seeds without drawing (find which runs exit)")
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args(argv)
    stats = render(a.ckpt, a.seeds, a.out, not a.greedy, a.wiggle, a.timeout, a.device, a.annotations, a.title_note, a.stats_line, a.scan, a.temperature)
    print(json.dumps({"out": str(a.out), "exited": [s["exited"] for s in stats]}))


if __name__ == "__main__":
    main()
