"""Stimulus battery and physiology metrics for the frozen front end (M3 gate / K3 in PLAN 2).

Stimuli are functions of visual angle (azimuth, elevation, time). Two ways to present them:

* column path: evaluate at every column centre and feed ``FrontEnd.step_columns`` at the substep
  rate (dt = 19 ms). Isolates the EMD/adaptation physiology from pixels and frame rate.
* pixel path: render 160x120 frames at the 114 ms decision rate and run ``FrontEnd.process``
  (sampler, fill, cross-fade included) -- what the agent actually sees.

Direction convention for stimuli: ``direction_deg`` is the motion direction in the IMAGE,
0 = rightward (+azimuth), 90 = upward (+elevation). The expected preferred direction of the
T4/T5 subtypes therefore depends on the eye: a (front-to-back) = 0 deg in the RIGHT eye and
180 deg in the LEFT eye; b the reverse; c = 90 deg and d = 270 deg in both.

Metrics: direction-selectivity index per unit ``DSI = (R_pd - R_nd) / (R_pd + R_nd)`` at the
expected preferred direction (and whether argmax over 8 directions equals it); an LPLC2-like loom
proxy (Klapoetke 2017 Nature 551:237: sum of outward-motion EMD channels in the four quadrants
around the disc centre) checked for monotonicity in expansion rate; ON/OFF polarity to flashes.

CLI: ``python -m flybrain.vision.stimuli [--synthetic R | --geometry path] [--pixel] [--json out]``
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from flybrain import OUT_DIR
from flybrain.vision.frontend import FrontEnd
from flybrain.vision.geometry import CLAMPED_TYPES, Geometry, DEFAULT_PATH, pixel_directions

DIRECTIONS_DEG = (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0)
TEMPORAL_FREQS_HZ = (0.5, 1.0, 2.0, 4.0, 8.0, 12.0)
WAVELENGTH_DEG = 30.0
LOOM_RATES_DEG_S = (5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
EMD_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def expected_pd(type_name: str, side: str) -> float:
    """Expected preferred direction (image degrees) of a T4/T5 subtype in the given eye."""
    sub = type_name[-1]
    if sub == "a":
        return 0.0 if side == "R" else 180.0
    if sub == "b":
        return 180.0 if side == "R" else 0.0
    return 90.0 if sub == "c" else 270.0


# ---- stimulus functions (angles in degrees, time in seconds) --------------------------------

def grating(az, el, t, direction_deg, tf_hz, wavelength_deg=WAVELENGTH_DEG, contrast=1.0, mean=0.5):
    th = np.radians(direction_deg)
    phase = 2.0 * np.pi * ((az * np.cos(th) + el * np.sin(th)) / wavelength_deg - tf_hz * t)
    return mean * (1.0 + contrast * np.sin(phase))


def disc(az, el, centre, radius_deg, lum_in=0.1, lum_out=0.5, edge_deg=1.0):
    r = np.hypot(az - centre[0], el - centre[1])
    inside = np.clip((radius_deg - r) / edge_deg + 0.5, 0.0, 1.0)
    return lum_out + (lum_in - lum_out) * inside


# ---- presentation ----------------------------------------------------------------------------

def run_columns(fe: FrontEnd, lum_at, duration_s: float, B: int) -> torch.Tensor:
    """Column path. ``lum_at(t) -> [B, C] float`` over all geometry columns. Returns rates [n_steps, B, M]."""
    n = int(round(duration_s * 1000.0 / fe.dt_ms))
    state = fe.init_state(B)
    out = []
    for i in range(n):
        lum = torch.as_tensor(np.asarray(lum_at(i * fe.dt_ms / 1000.0), np.float32), device=fe.device)
        rates, state = fe.step_columns(fe.columns_from_full(lum), state)
        out.append(rates)
    return torch.stack(out, 0)


def pixel_angles(geom: Geometry):
    d, _ = pixel_directions(geom.meta["hfov_deg"], *geom.frame_hw[::-1])
    return np.degrees(np.arctan2(d[:, 0], d[:, 2])), np.degrees(np.arcsin(np.clip(d[:, 1], -1, 1)))


def run_frames(fe: FrontEnd, lum_at, duration_s: float, B: int, decision_ms: float = 114.0) -> torch.Tensor:
    """Pixel path. ``lum_at(t) -> [B, H*W] float in [0,1]``; frames every decision step. Returns rates [n_frames*K, B, M]."""
    H, W = fe.geom.frame_hw
    n = int(round(duration_s * 1000.0 / decision_ms))
    state = fe.init_state(B)
    out = []
    for i in range(n):
        img = np.clip(np.asarray(lum_at(i * decision_ms / 1000.0)) * 255.0, 0, 255).astype(np.uint8).reshape(B, H, W)
        rates, state = fe.process(img, None, state)
        out.append(rates)
    return torch.cat(out, 0)


# ---- gratings and DSI ------------------------------------------------------------------------

def gratings_battery(fe: FrontEnd, pixel: bool = False, directions=DIRECTIONS_DEG, tfs=TEMPORAL_FREQS_HZ,
                     settle_s: float = 0.5, wavelength_deg=WAVELENGTH_DEG) -> np.ndarray:
    """Mean rate per unit over >= 2 grating periods after settling: [n_dir, n_tf, M]."""
    geom = fe.geom
    if pixel:
        az, el = pixel_angles(geom)
    else:
        az, el = geom.col_az.astype(np.float64), geom.col_el.astype(np.float64)
    combos = [(d, f) for d in directions for f in tfs]
    B = len(combos)
    resp = np.zeros((len(directions), len(tfs), fe.n_clamped), np.float32)

    def lum_at(t):
        return np.stack([grating(az, el, t, d, f, wavelength_deg) for d, f in combos], 0)

    # one run long enough for the slowest TF: settle + 2 periods (window = integer periods per TF)
    dur = settle_s + 2.0 / min(tfs)
    rates = (run_frames if pixel else run_columns)(fe, lum_at, dur, B)      # [n_steps, B, M]
    dt = fe.dt_ms / 1000.0
    for bi, (d, f) in enumerate(combos):
        n_per = max(int(round((2.0 / f) / dt)), 1)
        i0 = int(round(settle_s / dt))
        seg = rates[i0:i0 + n_per, bi]
        resp[directions.index(d), tfs.index(f)] = seg.mean(0).cpu().numpy()
    return resp


def dsi_table(resp: np.ndarray, fe: FrontEnd, directions=DIRECTIONS_DEG, tfs=TEMPORAL_FREQS_HZ) -> list[dict]:
    """Per (type, side, tf): median DSI over responding driven units, fraction whose vector-average
    preferred direction lies within 45 deg of the expected PD (and, for reference, whose argmax
    equals it), mean PD/ND rates, and the responding fraction."""
    idx = fe.index
    rows = []
    for ty in EMD_TYPES:
        for side in ("L", "R"):
            m = (idx.type == ty) & (idx.side == side) & idx.driven
            if not m.any():
                continue
            pd = expected_pd(ty, side)
            ip, inn = directions.index(pd), directions.index((pd + 180.0) % 360.0)
            for fi, f in enumerate(tfs):
                r = resp[:, fi, m]                                   # [n_dir, n_units]
                rp, rn = r[ip], r[inn]
                dsi = (rp - rn) / (rp + rn + 1e-9)
                active = r.max(0) > 1e-6
                # preferred direction as the vector average of the tuning curve (the standard
                # physiology definition); a single-axis correlator is broad (~+-45 deg), so a
                # plain argmax over 8 directions flips on a few-percent margin and is reported
                # separately. Units with no response (lattice-edge columns lacking the neighbour
                # of that axis) have no direction and are excluded via `frac_active`.
                th = np.radians(np.asarray(directions, np.float64))[:, None]
                vx, vy = (r * np.cos(th)).sum(0), (r * np.sin(th)).sum(0)
                pd_vec = np.degrees(np.arctan2(vy, vx)) % 360.0
                err = np.abs((pd_vec - pd + 180.0) % 360.0 - 180.0)
                rows.append({"type": ty, "side": side, "tf_hz": f, "n_units": int(m.sum()),
                             "median_dsi": float(np.median(dsi[active])) if active.any() else float("nan"),
                             "frac_correct_pd": float((err[active] <= 45.0).mean()) if active.any() else 0.0,
                             "frac_argmax_pd": float((r.argmax(0) == ip)[active].mean()) if active.any() else 0.0,
                             "mean_pd_rate": float(rp.mean()), "mean_nd_rate": float(rn.mean()),
                             "frac_active": float(active.mean())})
    return rows


# ---- looming / receding / translating discs -------------------------------------------------

def loom_proxy(rates: torch.Tensor, fe: FrontEnd, centre, side: str, family: str = "T5") -> np.ndarray:
    """LPLC2-like outward-motion template around `centre`, summed over driven units of one eye: [n_steps]."""
    idx = fe.index
    az0, el0 = centre
    total = None
    for sub, cond in (("a", idx.az > az0), ("b", idx.az < az0), ("c", idx.el > el0), ("d", idx.el < el0)):
        ty = family + sub
        if side == "L" and sub in "ab":          # progressive/regressive flip in the left eye
            cond = (idx.az < az0) if sub == "a" else (idx.az > az0)
        m = (idx.type == ty) & (idx.side == side) & idx.driven & cond
        part = rates[:, torch.as_tensor(np.nonzero(m)[0], device=rates.device)].sum(1)
        total = part if total is None else total + part
    return total.cpu().numpy()


def disc_battery(fe: FrontEnd, centre=(25.0, 0.0), side="R", rates_deg_s=LOOM_RATES_DEG_S, r0=2.0, r1=40.0,
                 lum_in=0.1, lum_out=0.5, pre_s: float = 0.6) -> dict:
    """Expanding, receding and translating dark discs; returns the loom proxy time courses and summaries."""
    geom = fe.geom
    az, el = geom.col_az.astype(np.float64), geom.col_el.astype(np.float64)
    kinds = [("expand", v) for v in rates_deg_s] + [("recede", v) for v in rates_deg_s] + [("translate", v) for v in rates_deg_s]
    B = len(kinds)
    t_move = (r1 - r0) / np.array([v for _, v in kinds])          # seconds of motion per stimulus
    dur = pre_s + float(t_move.max()) + 0.3

    def lum_at(t):
        out = []
        for (kind, v), tm in zip(kinds, t_move):
            tt = np.clip(t - pre_s, 0.0, tm)
            if kind == "expand":
                out.append(disc(az, el, centre, r0 + v * tt, lum_in, lum_out))
            elif kind == "recede":
                out.append(disc(az, el, centre, r1 - v * tt, lum_in, lum_out))
            else:   # 10 deg disc translating rightward through the centre at edge speed v
                c = (centre[0] - 0.5 * (r1 - r0) + v * tt, centre[1])
                out.append(disc(az, el, c, 10.0, lum_in, lum_out))
        return np.stack(out, 0)

    rates = run_columns(fe, lum_at, dur, B)
    dt = fe.dt_ms / 1000.0
    res = {"centre": centre, "side": side, "rates_deg_s": list(rates_deg_s), "curves": {}}
    for bi, ((kind, v), tm) in enumerate(zip(kinds, t_move)):
        i0, i1 = int(round(pre_s / dt)), int(round((pre_s + tm) / dt)) + 1
        for fam in ("T4", "T5"):
            p = loom_proxy(rates[:, bi], fe, centre, side, fam)
            res["curves"].setdefault(kind, {}).setdefault(fam, {})[str(v)] = {
                "mean_during_motion": float(p[i0:i1].mean()), "peak": float(p.max()), "n_steps_motion": int(i1 - i0)}
    return res


# ---- flashes ---------------------------------------------------------------------------------

def flash_battery(fe: FrontEnd, base=0.3, high=0.6, t_up=0.5, t_down=1.5, dur=2.5) -> dict:
    """Whole-field luminance step up then down; per-type mean rate in the 150 ms after each step."""
    C = fe.geom.n_columns

    def lum_at(t):
        v = high if t_up <= t < t_down else base
        return np.full((1, C), v, np.float32)

    rates = run_columns(fe, lum_at, dur, 1)[:, 0]        # [n_steps, M]
    dt = fe.dt_ms / 1000.0
    win = max(int(round(0.15 / dt)), 1)
    iu, idn = int(round(t_up / dt)), int(round(t_down / dt))
    out = {}
    for ty in CLAMPED_TYPES:
        m = torch.as_tensor(np.nonzero((fe.index.type == ty) & fe.index.driven)[0], device=rates.device)
        if len(m) == 0:
            continue
        out[ty] = {"up": float(rates[iu:iu + win, m].mean()), "down": float(rates[idn:idn + win, m].mean()),
                   "baseline": float(rates[iu - win:iu, m].mean())}
    return out


# ---- CLI -------------------------------------------------------------------------------------

def _fmt_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Front-end stimulus battery")
    ap.add_argument("--geometry", default=str(DEFAULT_PATH))
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--pixel", action="store_true", help="also run the gratings through the pixel path")
    ap.add_argument("--json", default="")
    ap.add_argument("--fill", default="mean", choices=["mean", "black"])
    args = ap.parse_args(argv)
    geom = Geometry.synthetic(args.synthetic) if args.synthetic else Geometry.load(args.geometry)
    fe = FrontEnd(geom, fill=args.fill)
    t0 = time.time()
    resp = gratings_battery(fe)
    rows = dsi_table(resp, fe)
    print(f"# gratings, column path ({time.time() - t0:.1f}s)")
    print(_fmt_table(rows, ["type", "side", "tf_hz", "n_units", "median_dsi", "frac_correct_pd", "mean_pd_rate", "mean_nd_rate"]))
    result = {"gratings_column": rows}
    if args.pixel:
        t0 = time.time()
        rows_px = dsi_table(gratings_battery(fe, pixel=True), fe)
        print(f"# gratings, pixel path at 114 ms frames ({time.time() - t0:.1f}s)")
        print(_fmt_table(rows_px, ["type", "side", "tf_hz", "n_units", "median_dsi", "frac_correct_pd", "mean_pd_rate", "mean_nd_rate"]))
        result["gratings_pixel"] = rows_px
    t0 = time.time()
    discs = disc_battery(fe)
    print(f"# discs, loom proxy mean during motion ({time.time() - t0:.1f}s)")
    for kind, fams in discs["curves"].items():
        for fam, curve in fams.items():
            print(f"{kind:9s} {fam}: " + "  ".join(f"{v}:{c['mean_during_motion']:.3f}" for v, c in curve.items()))
    result["discs"] = discs
    flashes = flash_battery(fe)
    print("# flashes (mean rate 150 ms after step): type up down baseline")
    for ty, v in flashes.items():
        print(f"{ty:6s} {v['up']:.4f} {v['down']:.4f} {v['baseline']:.4f}")
    result["flashes"] = flashes
    if args.json:
        p = __import__("pathlib").Path(args.json); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=1))
        print("wrote", p)


if __name__ == "__main__":
    main()
