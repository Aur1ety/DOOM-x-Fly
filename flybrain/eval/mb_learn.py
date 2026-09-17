"""Step 2/3 of the learning experiment: does the frozen MaleCNS brain, with dopamine plasticity on its
KC -> MBON synapses only, form a real associative memory when a visual pattern is paired with the
punishment dopaminergic neurons?

Protocol (mirrors Hige et al. 2015, Neuron, which we score against):
  - build the gain-matched core + frozen eye model + the mushroom-body plasticity module,
  - present each visual pattern through the eye and read the MBON *rate* it evokes (BEFORE),
  - pair pattern A with the punishment DANs for a few trials (forward timing),
  - present each pattern again and read MBON rates (AFTER),
  - report the response drop for the paired vs the unpaired pattern.

Ground-truth targets (Hige et al. 2015): paired-pattern MBON response drops ~80 %; unpaired pattern
unchanged; the change persists. A shuffled-wiring control
answers whether the fly's specific wiring is doing the work.

    python -m flybrain.eval.mb_learn --pairings 8 --punish PPL101 --out $FLYBRAIN_OUT/mb/learn_ppl101.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR

# ------------------------------------------------------------------------------------------ stimuli

def make_patterns(kind: str = "bars") -> list[dict]:
    """A few clearly distinct static visual patterns as (gray [120,160], rgb [60,80,3]) uint8.

    'bars' = bright bar on a dark field at 4 orientations; the fly's form/contrast pathway sees each
    differently, driving different visual Kenyon cells. Index 0 is the pattern we punish."""
    H, W = 120, 160
    pats = []
    specs = [("vertical", (slice(20, 100), slice(70, 90))),
             ("horizontal", (slice(52, 68), slice(30, 130))),
             ("left-diag", None), ("right-diag", None)]
    for name, box in specs:
        g = np.full((H, W), 30, np.uint8)                      # dark grey field
        if box is not None:
            g[box] = 220
        else:
            yy, xx = np.mgrid[0:H, 0:W]
            band = (np.abs((yy - 60) + (1 if name == "left-diag" else -1) * (xx - 80)) < 10)
            g[band] = 220
        rgb = np.repeat(_down(g, 2)[:, :, None], 3, axis=2)
        pats.append({"name": name, "gray": g, "rgb": rgb})
    return pats


def _down(img: np.ndarray, k: int) -> np.ndarray:
    h, w = img.shape[0] // k, img.shape[1] // k
    return img[: h * k, : w * k].reshape(h, k, w, k).mean((1, 3)).astype(np.uint8)


def make_moving(n_frames: int = 12) -> list[dict]:
    """Drifting square-wave gratings at 4 directions (right/left/up/down) - the classic optomotor stimulus
    that maximally drives the fly's T4/T5 motion detectors. Each is a SEQUENCE of frames (the eye needs
    motion). Returns [{name, frames:[(gray,rgb)...]}]. Index 0 is the pattern we punish."""
    H, W = 120, 160
    period, speed = 40, 6
    yy, xx = np.mgrid[0:H, 0:W]
    dirs = [("right", (0, 1)), ("left", (0, -1)), ("up", (-1, 0)), ("down", (1, 0))]
    out = []
    for name, (dy, dx) in dirs:
        frames = []
        for t in range(n_frames):
            phase = speed * t
            coord = (xx * dx + yy * dy)
            g = np.where(((coord + phase) % period) < period // 2, 220, 30).astype(np.uint8)
            frames.append((g, np.repeat(_down(g, 2)[:, :, None], 3, axis=2)))
        out.append({"name": name, "frames": frames})
    return out


# ------------------------------------------------------------------------------------------ experiment

def build(subgraph: Path, geometry: Path, neurons: Path, device: str, op_point: str, lr: float,
          punish_types, reward_types, shuffle_seed: int | None):
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph
    from flybrain.model.gain import match_gain
    from flybrain.model.mb_plasticity import MushroomBodyPlasticity, node_types
    from flybrain.vision.frontend import FrontEnd
    from flybrain.vision.geometry import Geometry

    sub = load_subgraph(subgraph)
    if shuffle_seed is not None:                               # negative control: shuffle KC->MBON wiring
        sub = _shuffle_kc_mbon(sub, neurons, shuffle_seed)
    core = ConnectomeCore(sub, CoreConfig(param="per_edge"), backend="spmm", device=device)
    for p in core.parameters():
        p.requires_grad_(False)
    gm = match_gain(core, g_star=0.95, op_point=op_point, seed=0)
    geom = Geometry.load(geometry); fe = FrontEnd(geom, K=core.cfg.K, device=device)
    body = np.asarray(sub["bodyId"]); is_cl = np.asarray(sub["is_clamped"], bool)
    perm = torch.as_tensor(fe.index.permutation_for(body[is_cl]), device=device)
    types = node_types(sub, neurons)
    mb = MushroomBodyPlasticity(core, types, lr=lr, device=device, punish_types=punish_types, reward_types=reward_types)
    return core, fe, perm, mb, gm["w0"]


def _shuffle_kc_mbon(sub: dict, neurons: Path, seed: int) -> dict:
    """Degree-preserving shuffle of the KC->MBON presynaptic partners only (rest of the wiring intact)."""
    from flybrain.model.mb_plasticity import node_types
    sub = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sub.items()}
    types = node_types(sub, neurons)
    is_kc = np.char.startswith(types.astype(str), "KC")
    post = np.repeat(np.arange(len(sub["indptr_post"]) - 1), np.diff(sub["indptr_post"]))
    is_mbon = np.char.startswith(types.astype(str), "MBON")
    pre = sub["indices_pre"]
    sel = is_kc[pre] & is_mbon[post]                            # plastic edges
    kc_pre = pre[sel]
    rng = np.random.default_rng(seed)
    sub["indices_pre"] = pre.copy(); sub["indices_pre"][sel] = rng.permutation(kc_pre)   # same KC multiset, shuffled targets
    return sub


@torch.no_grad()
def clamp(fe, perm, pat: dict, device: str, settle: int = 6) -> torch.Tensor:
    """Run a static pattern through the eye model until it settles; return clamped rates [n_clamped]."""
    g = torch.as_tensor(pat["gray"][None], device=device); r = torch.as_tensor(pat["rgb"][None], device=device)
    st = fe.init_state(1); rates = None
    for _ in range(settle):
        rates, st = fe.process(g, r, st)                       # [K, 1, n_units]
    return rates[-1, 0].index_select(0, perm)                  # last substep, subgraph clamped order


@torch.no_grad()
def measure(mb, clamped: torch.Tensor, decisions: int) -> torch.Tensor:
    return mb.mbon_rates(mb.present(clamped, decisions))


@torch.no_grad()
def present_moving(core, fe, perm, frames: list, device: str) -> torch.Tensor:
    """Run a moving stimulus (sequence of frames) through eye + core, one core decision per frame, and
    return the per-node rates averaged over the second half of the sequence (steady motion response).
    This sustains the motion the eye needs, unlike a single held frame."""
    from flybrain.model.core import CoreState  # noqa: F401
    fst = fe.init_state(1); state = core.init_state(1); acc = None; n = 0
    for i, (g, r) in enumerate(frames):
        rates, fst = fe.process(torch.as_tensor(g[None], device=device), torch.as_tensor(r[None], device=device), fst)
        xc = rates.index_select(2, perm)                       # [K, 1, n_clamped]
        state = core.step(state, xc)
        if i >= len(frames) // 2:
            rr = core.rates(state)[:, 0]
            acc = rr if acc is None else acc + rr; n += 1
    return acc / max(n, 1)


def run(a) -> dict:
    device = a.device
    if device.startswith("cuda"):
        from flybrain.train.level_bc import _guarded_device
        device = _guarded_device(device)
    t0 = time.time()
    _core, fe, perm, mb, w0 = build(a.subgraph, a.geometry, a.neurons, device, a.op_point, a.lr,
                                   tuple(a.punish), tuple(a.reward), a.shuffle_seed)
    pats = make_patterns()
    clamps = [clamp(fe, perm, p, device) for p in pats]
    names = [p["name"] for p in pats]

    before = torch.stack([measure(mb, c, a.settle_decisions) for c in clamps])     # [P, n_mbon]

    # which MBONs are the "readout" for the paired pattern A: they get punishment dopamine AND respond to A
    da = mb.delta_punish[mb.mbon_idx] > 0
    respond_A = before[0] > 0.05 * float(before[0].max())
    focus = (da & respond_A).cpu().numpy()

    # forward pairing of pattern A (index 0) with punishment
    def pair(sign_order: str, trials: int):
        mb.reset()
        for _ in range(trials):
            r = mb.present(clamps[0], a.settle_decisions)      # KC rates during A
            if sign_order == "dan_alone":                      # dopamine with silent KCs: zero by the rule's algebra, not a timing test
                mb.reinforce(torch.zeros_like(r), us="punish", strength=a.strength)
            else:
                mb.reinforce(r, us="punish", strength=a.strength)

    pair("forward", a.pairings)
    after = torch.stack([measure(mb, c, a.settle_decisions) for c in clamps])

    def drop(bef, aft):
        m = torch.as_tensor(focus, device=device)
        b = bef[m].clamp_min(1e-3); f = aft[m]
        return float((1 - (f / b)).mean())

    res = {"w0": w0, "shuffle_seed": a.shuffle_seed, "punish": list(a.punish), "pairings": a.pairings,
           "n_focus_mbon": int(focus.sum()), "focus_mbon_local_idx": np.flatnonzero(focus).tolist()[:20],
           "drop_paired_A": drop(before[0], after[0]),
           "drop_unpaired": {names[i]: drop(before[i], after[i]) for i in range(1, len(pats))},
           "W_after": {"min": float(mb.W.min()), "mean": float(mb.W.mean()), "max": float(mb.W.max())}}

    # dopamine with no KC activity: identically 0 by the rule (k = 0, W = 1). Reported as such; this model has no time axis.
    pair("dan_alone", a.pairings)
    after_bwd = torch.stack([measure(mb, c, a.settle_decisions) for c in clamps])
    res["drop_paired_A_dan_alone_no_KC"] = drop(before[0], after_bwd[0])

    # persistence: after forward pairing, relax with the fitted forgetting tau and re-measure
    if a.forget_tau:
        pair("forward", a.pairings); mb.forget_tau = a.forget_tau
        mb.relax(decisions=a.relax_decisions)
        after_relax = torch.stack([measure(mb, c, a.settle_decisions) for c in clamps])
        res["drop_paired_A_after_forget"] = drop(before[0], after_relax[0])

    res["elapsed_s"] = time.time() - t0
    res["ground_truth"] = "Hige et al. 2015: paired MBON response drops ~80%; unpaired ~0"
    return res


@torch.no_grad()
def diagnose(a) -> dict:
    """No learning. Ask why depression was not stimulus-specific: are the patterns distinct at the optic
    lobe, at the Kenyon-cell code, and specifically at the visual Kenyon cells? Is the KC code sparse?"""
    device = a.device
    if device.startswith("cuda"):
        from flybrain.train.level_bc import _guarded_device
        device = _guarded_device(device)
    core, fe, perm, mb, w0 = build(a.subgraph, a.geometry, a.neurons, device, a.op_point, a.lr,
                                   tuple(a.punish), tuple(a.reward), a.shuffle_seed)
    types = mb.types
    kc_idx = torch.as_tensor(mb.kc_idx, device=device)
    kcgd = torch.as_tensor(np.flatnonzero(np.char.startswith(types, "KCg-d")), device=device)   # visual KCs
    kc_other = torch.as_tensor(np.flatnonzero(np.char.startswith(types, "KC") & ~np.char.startswith(types, "KCg-d")), device=device)
    if a.moving:
        mov = make_moving(a.moving_frames); names = [p["name"] for p in mov]
        rates = torch.stack([present_moving(core, fe, perm, p["frames"], device) for p in mov])   # [P, M]
        clamps = None
    else:
        pats = make_patterns(); names = [p["name"] for p in pats]
        clamps = torch.stack([clamp(fe, perm, p, device) for p in pats])
        rates = torch.stack([mb.present(clamps[i], a.settle_decisions) for i in range(len(pats))])
    P = len(names)

    def cos(mat):                                                                    # [P,P] cosine similarity
        x = mat / (mat.norm(dim=1, keepdim=True) + 1e-9)
        return (x @ x.T).cpu().numpy().round(3).tolist()

    kc = rates.index_select(1, kc_idx); vis = rates.index_select(1, kcgd)
    thr = 0.1

    def offdiag_mean(mat):
        m = np.asarray(cos(mat)); return round(float(m[~np.eye(len(m), dtype=bool)].mean()), 4)

    # localise where visual pattern information dies along eye -> VPN -> KC -> MBON. For each population,
    # measure the perturbation from a blank screen (pattern - blank): norm (does it respond at all?) and
    # cross-pattern cosine (is the response pattern-specific?). We know VPN/DN respond (the Doom agent read them).
    import pandas as pd

    from flybrain.model.core import load_subgraph
    body = np.asarray(load_subgraph(a.subgraph)["bodyId"])
    scv = pd.read_parquet(a.neurons, columns=["bodyId", "superclass"])
    scmap = dict(zip(scv["bodyId"].to_numpy().tolist(), scv["superclass"].fillna("").astype(str).to_numpy().tolist()))
    scarr = np.array([scmap.get(int(b), "") for b in body])
    pops = {"VPN": np.flatnonzero(scarr == "visual_projection"), "DN": np.flatnonzero(scarr == "descending_neuron"),
            "KCg-d(visual)": kcgd.cpu().numpy(), "KC(all)": kc_idx, "MBON": mb.mbon_idx}
    blank_gray = np.full((120, 160), 125, np.uint8); blank_rgb = np.repeat(_down(blank_gray, 2)[:, :, None], 3, 2)
    if a.moving:
        base = present_moving(core, fe, perm, [(blank_gray, blank_rgb)] * a.moving_frames, device)   # static field = no motion
    else:
        base = mb.present(clamp(fe, perm, {"gray": blank_gray, "rgb": blank_rgb}, device), a.settle_decisions)
    D = (rates - base)
    diff = {}
    for name, idxs in pops.items():
        it = torch.as_tensor(idxs, device=device, dtype=torch.long)
        dp = D.index_select(1, it)                                 # [P, n_pop]
        norms = [round(float(dp[i].norm()), 5) for i in range(P)]
        cosv = None
        if float(dp.norm()) > 1e-9:
            cm = np.array(cos(dp)); cosv = round(float(cm[~np.eye(P, dtype=bool)].mean()), 4)
        diff[name] = {"n": len(idxs), "resp_norm": norms, "pattern_cos": cosv}

    # settle sweep: does pattern-specific KC activity survive, and for how many decisions? (static only)
    sweep = {}
    for dsteps in (() if a.moving else (1, 2, 3, 5, 8, 12, 20)):
        rs = torch.stack([mb.present(clamps[i], dsteps) for i in range(P)])
        kcs = rs.index_select(1, kc_idx); viss = rs.index_select(1, kcgd)
        sweep[dsteps] = {"kc_cos_offdiag": offdiag_mean(kcs), "kc_visual_cos_offdiag": offdiag_mean(viss),
                         "kc_frac_active": round(float((kcs > thr).float().mean()), 3),
                         "kc_visual_frac_active": round(float((viss > thr).float().mean()), 3)}
    return {"w0": w0, "patterns": names,
            "moving": bool(a.moving), "clamp_cos_offdiag": (offdiag_mean(clamps) if clamps is not None else None),
            "kc_all_cos_offdiag(settle20)": offdiag_mean(kc), "kc_visual_cos_offdiag(settle20)": offdiag_mean(vis),
            "perturbation_from_blank": diff,
            "settle_sweep": sweep,
            "n_KCg_d": int(kcgd.numel()), "n_KC_other": int(kc_other.numel()),
            "note": "kc_cos_offdiag near 1 = codes overlap (bad); lower = patterns distinguishable at the KCs"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--geometry", type=Path, default=OUT_DIR / "vision" / "geometry_v1.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--op-point", default="rest")
    ap.add_argument("--pairings", type=int, default=8); ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--settle-decisions", type=int, default=20)
    ap.add_argument("--punish", nargs="+", default=["PPL1"], help="DAN type prefixes/names for punishment (e.g. PPL101)")
    ap.add_argument("--reward", nargs="+", default=["PAM"])
    ap.add_argument("--shuffle-seed", type=int, default=None, help="if set, shuffle KC->MBON wiring (negative control)")
    ap.add_argument("--forget-tau", type=float, default=None); ap.add_argument("--relax-decisions", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--diagnose", action="store_true", help="report pattern / KC-code distinctness and exit (no learning)")
    ap.add_argument("--moving", action="store_true", help="use drifting gratings (motion) instead of static patterns")
    ap.add_argument("--moving-frames", type=int, default=12)
    a = ap.parse_args(argv)
    res = diagnose(a) if a.diagnose else run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
