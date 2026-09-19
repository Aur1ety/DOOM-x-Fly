"""Teach the fly in time: a training episode with a clock, so the memory forms from experience.

The other memory scripts (mb_olfactory, mb_behaviour) have no time axis: they call the plasticity rule once
per "pairing block" and the depression size is set by the learning rate. That is not how a fly learns. A fly
learns because the odour and the dopamine overlap IN TIME, and the timing decides everything (Hige et al.
2015: dopamine 0.2 s AFTER the odour writes the memory; dopamine before it does nothing).

Here the odour switches its Kenyon cells on for a few seconds, dopamine (PPL1 -> MBON) arrives as discrete
pulses at a set delay, and a KC->MBON synapse depresses only where recent Kenyon activity (a decaying
eligibility trace) and dopamine coincide at that instant. The memory is not written by hand; it forms from
the overlap. One plasticity constant `eta` is anchored once so the standard protocol gives ~Hige's drop.

What is and is not a prediction here. The rule is a trace-gated MULTIPLICATIVE depression,
dW = -eta * dt * DA(t) * delta_j * e_i(t) * W_ij on KC->MBON only. It is NOT the Gkanias 2022 rule that
sections 7 to 9 use (there is no silent-Kenyon-cell recovery term), so the same-compartment coexistence
failure of section 7.4 does not arise in this module, and the two are not interchangeable. With a binary
Kenyon code every active KC->MBON11 synapse sees the same trace and the same dopamine, so the MBON11 drop for
any pulse schedule is a closed form of (eta, tau_elig, dt, schedule) once eta is anchored: the timing sweep and
the dose curve contain NO connectome information (the closed form is reported next to every measured value and
coincides). They show the trace mechanism, exactly as section 7.3's 1 - (1 - lr*delta)^p shows the rule. What
depends on the wiring is the unpaired-odour drop (Kenyon overlap at MBON11), the compartment pattern outside
MBON11, and the behavioural readout through the published valence map.

    python -m flybrain.eval.mb_online --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --out $FLYBRAIN_OUT/mb/online.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR
from flybrain.eval.mb_olfactory import MushroomBody, json_safe


def da_pulse_train(n_pulses: int, onset: float, freq: float, width: float, anchor: str = "start") -> np.ndarray:
    """Pulse start times (s) of `n_pulses` at `freq` Hz. anchor='start': the FIRST pulse starts at `onset` (s after
    odour onset). anchor='end': the LAST pulse ENDS at `onset`, so a negative onset puts the whole train before the
    odour (a genuine backward pairing) instead of letting a 1.5 s train straddle odour onset."""
    if n_pulses <= 0:
        return np.array([])
    starts = onset + np.arange(n_pulses) / freq
    if anchor == "end":
        starts = starts - ((n_pulses - 1) / freq + width)
    return starts


def _grid(dt, odour_on, odour_off, da_starts, da_width, tau_elig):
    """Common time grid: from the earliest event (odour or dopamine) to 3 tau_elig after the last."""
    t_start = float(min(0.0, odour_on, da_starts.min() if len(da_starts) else 0.0))
    t_end = max(odour_off, (da_starts.max() + da_width if len(da_starts) else 0.0)) + 3.0 * tau_elig
    n = round((t_end - t_start) / dt)
    da_on = np.zeros(n, bool)
    for s in da_starts:
        lo = max(0, round((s - t_start) / dt)); hi = max(0, round((s + da_width - t_start) / dt))
        da_on[lo:hi] = True
    return t_start, n, da_on


@torch.no_grad()
def train_episode(mb: MushroomBody, code: torch.Tensor, us: str, *, dt: float, odour_on: float, odour_off: float,
                  da_starts: np.ndarray, da_width: float, tau_elig: float, eta: float, tau_forget: float) -> None:
    """Run one timed episode, evolving the plastic KC->MBON weights mb.W in place.

    Eligibility trace e_kc(t): rises while a Kenyon cell is active, decays with tau_elig (the slow trace the
    dopamine acts on). At each instant with dopamine present, every KC->MBON synapse depresses by
    dW = -eta * DA(t) * delta_MBON * e_kc(t) * W * dt (multiplicative: monotonic, bounded), plus slow passive
    recovery toward rest. Dopamine that overlaps or shortly follows Kenyon activity finds e_kc > 0 and writes;
    dopamine that ends before the odour starts finds e_kc = 0 and writes nothing, by construction of a
    forward-only trace. NOTE: Cohn, Morantte & Ruta 2015 show the full rule is bidirectional (dopamine BEFORE the
    odour potentiates); adding that arm cleanly needs onset-gating and is a documented next refinement."""
    delta = {"punish": mb.delta_punish, "reward": mb.delta_reward}[us]
    de = delta[mb.e_mbon]
    elig = torch.zeros(mb.n_kc)
    t_start, n, da_on = _grid(dt, odour_on, odour_off, da_starts, da_width, tau_elig)
    for step in range(n):
        t = t_start + step * dt
        k = code if (odour_on <= t < odour_off) else torch.zeros_like(code)
        elig = elig * np.exp(-dt / tau_elig) + k * (dt / tau_elig)          # recent KC activity (0..~1)
        if da_on[step]:                                                     # dopamine now, KC active recently -> weaken
            mb.W = torch.clamp(mb.W - eta * dt * de * elig[mb.e_kc] * mb.W, 0.0, mb.w_max)
        if tau_forget < 1e6:
            mb.W = mb.W + (1.0 - mb.W) * (dt / tau_forget)                  # slow passive recovery (forgetting)


def closed_form_drop(eta: float, delta11: float, *, dt: float, odour_on: float, odour_off: float, da_starts: np.ndarray,
                     da_width: float, tau_elig: float, tau_forget: float) -> float:
    """The MBON11 drop train_episode gives for a BINARY code, on a scalar: every active KC->MBON11 synapse sees the
    same trace and the same dopamine, so the drop is 1 - prod over dopamine-on steps of (1 - eta*dt*delta11*e(t))
    with the same passive recovery. No wiring enters; this is what the anchored rule yields for a schedule."""
    t_start, n, da_on = _grid(dt, odour_on, odour_off, da_starts, da_width, tau_elig)
    e, w = 0.0, 1.0
    for step in range(n):
        t = t_start + step * dt
        e = e * np.exp(-dt / tau_elig) + (1.0 if odour_on <= t < odour_off else 0.0) * (dt / tau_elig)
        if da_on[step]:
            w = max(w - eta * dt * delta11 * e * w, 0.0)
        if tau_forget < 1e6:
            w = w + (1.0 - w) * (dt / tau_forget)
    return 1.0 - w


def drop_at(mb, before, code, mask):
    b = float(before[mask].sum())
    return round(1 - float(mb.mbon_response(code)[mask].sum()) / b, 4) if b > 0 else None


def calibrate_eta(mb, code, mask, target, us, base_kw) -> float:
    """One anchor: set eta so the STANDARD protocol gives `target` drop. Everything else stays a prediction."""
    lo, hi = 0.0, 200.0
    b0 = mb.mbon_response(code)
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        mb.reset(); train_episode(mb, code, us, eta=mid, **base_kw)
        d = 1 - float(mb.mbon_response(code)[mask].sum()) / float(b0[mask].sum())
        lo, hi = (mid, hi) if d < target else (lo, mid)
    return 0.5 * (lo + hi)


def run(a) -> dict:
    t0 = time.time()
    wiring = a.wiring if (a.wiring and a.wiring.exists()) else None
    mb = MushroomBody(a.subgraph, a.neurons, sparsity=0.05, punish=("PPL101",), reward=("PAM",),
                      binary=True, wiring=wiring, modality="olfactory")
    rng = np.random.default_rng(a.seed)
    stim = [rng.choice(len(mb.glom_names), size=6, replace=False) for _ in range(4)]
    codes = [mb.kc_code(mb.odour(s)) for s in stim]
    m11 = mb.type_mask("MBON11")
    before_all = [mb.mbon_response(c) for c in codes]; before = before_all[0]   # per-odour fresh-W baselines

    da = da_pulse_train(a.pulses, a.da_onset, a.da_freq, a.da_width)
    std = {"dt": a.dt, "odour_on": 0.0, "odour_off": a.odour_dur, "da_starts": da, "da_width": a.da_width,
           "tau_elig": a.tau_elig, "tau_forget": a.tau_forget}
    eta = calibrate_eta(mb, codes[0], m11, a.target, "punish", std)
    d11 = float(mb.delta_punish[m11].max())                                  # dopamine share reaching MBON11 (1.0 in the real wiring)

    def episode(code, us="punish", **over):
        kw = {**std, **over}
        mb.reset(); train_episode(mb, code, us, eta=eta, **kw)

    def train_for(n_pulses, onset):
        """Pulse train for a sweep point: negative onsets anchor the END of the train at `onset` so the whole train
        precedes the odour; non-negative onsets anchor the start."""
        return da_pulse_train(n_pulses, onset, a.da_freq, a.da_width, anchor="end" if onset < 0 else "start")

    def window(starts):
        if not len(starts):
            return {"train_s": None, "pulses_in_odour": 0, "relation_to_odour": "no dopamine"}
        t0_, t1_ = float(starts.min()), float(starts.max() + a.da_width)
        in_od = int(((starts + a.da_width > 0.0) & (starts < a.odour_dur)).sum())
        rel = ("before odour" if t1_ <= 0.0 else "after odour" if t0_ >= a.odour_dur else
               "during odour" if (t0_ >= 0.0 and t1_ <= a.odour_dur) else "straddles odour edge")
        return {"train_s": [round(t0_, 2), round(t1_, 2)], "pulses_in_odour": in_od, "relation_to_odour": rel}

    def point(starts):
        episode(codes[0], da_starts=starts)
        return {"drop_MBON11": drop_at(mb, before, codes[0], m11),
                "closed_form_MBON11": round(closed_form_drop(eta, d11, **{**std, "da_starts": starts}), 4), **window(starts)}

    # 1. standard forward episode: paired A vs unpaired B (each odour probed against its OWN baseline)
    episode(codes[0]); paired = drop_at(mb, before_all[0], codes[0], m11); unpaired = drop_at(mb, before_all[1], codes[1], m11)

    # 2. TIMING: the dopamine train moved relative to the odour window. NOT a wiring prediction (see the module
    #    docstring): closed_form_MBON11 is the anchored rule's value for the same schedule and coincides with the
    #    measurement. It shows the trace mechanism: a train that ends before the odour starts writes nothing (zero by
    #    construction), trains during the odour write, and trains after odour offset write in proportion to the
    #    decaying trace.
    timing = {str(onset): point(train_for(a.pulses, onset)) for onset in a.timing_onsets}

    # 3. TRAINING AMOUNT: more dopamine pulses -> more memory, saturating. Same status: the closed form of the
    #    anchored rule (the online analogue of section 7.3's 1 - (1 - lr*delta)^p), not a wiring prediction.
    dose = {str(p): point(train_for(p, a.da_onset)) for p in a.dose_pulses}

    # 4. controls that must stay near 0 by the mechanism (no fit): dopamine with no odour; odour with no dopamine
    episode(codes[0], odour_off=0.0); da_alone = drop_at(mb, before, codes[0], m11)
    episode(codes[0], da_starts=np.array([])); odour_alone = drop_at(mb, before, codes[0], m11)

    # 5. does the timed TEACHING change BEHAVIOUR? read the choice through the published valence map.
    from flybrain.eval.mb_behaviour import build_valence
    val, _ = build_valence(mb, a.neurons)
    base_resp = torch.stack([mb.mbon_response(c) for c in codes])
    norm = float(base_resp.abs().sum(1).mean())

    def score(R):
        return (R * val).sum(1) / norm

    def p_choose(sx, sy):
        return float(torch.sigmoid(torch.tensor(a.beta) * (sx - sy)))

    def score_after(idx, onset, us="punish"):
        episode(codes[idx], da_starts=train_for(a.pulses, onset), us=us)
        return score(torch.stack([mb.mbon_response(c) for c in codes]))

    def pi_at(onset):                                                      # reciprocal T-maze index (Tully & Quinn)
        sA = score_after(0, onset); h1 = p_choose(sA[1], sA[0]) - p_choose(sA[0], sA[1])
        sB = score_after(1, onset); h2 = p_choose(sB[0], sB[1]) - p_choose(sB[1], sB[0])
        return round(0.5 * (h1 + h2), 4)
    s0 = score(base_resp); back = -abs(a.da_onset) - 2.0                     # train ends 2.2 s BEFORE odour onset
    sf, sb = score_after(0, a.da_onset), score_after(0, back)
    behaviour = {"beta": a.beta,
                 "P_avoid_A": {"untrained": round(p_choose(s0[1], s0[0]), 4),
                               "forward_taught": round(p_choose(sf[1], sf[0]), 4),
                               "backward_taught": round(p_choose(sb[1], sb[0]), 4)},
                 "T_maze_PI": {"forward_taught": pi_at(a.da_onset), "backward_taught": pi_at(back)},
                 "backward_train_s": [round(float(train_for(a.pulses, back).min()), 2), round(float(train_for(a.pulses, back).max() + a.da_width), 2)],
                 "note": "forward teaching (dopamine during the odour) yields avoidance through the valence map; a train that "
                         "ends before the odour starts yields none, which is zero by construction of the trace, not a "
                         "measured timing curve. wild-type performance index 0.44-0.53."}
    # behavioural learning curve: avoidance rises with the amount of teaching and saturates once MBON11 is fully
    # depressed (it may dip slightly past that point as late pulses act only on other compartments' small shares)
    pi_dose = {}
    for p in a.dose_pulses:
        episode(codes[0], da_starts=train_for(p, a.da_onset))
        sA = score(torch.stack([mb.mbon_response(c) for c in codes]))
        pi_dose[str(p)] = round(p_choose(sA[1], sA[0]), 4)
    behaviour["P_avoid_A_vs_pulses"] = pi_dose

    res = {"protocol": {"odour_dur_s": a.odour_dur, "pulses": a.pulses, "da_onset_s": a.da_onset, "da_freq_hz": a.da_freq,
                        "da_width_s": a.da_width, "tau_elig_s": a.tau_elig, "tau_forget_s": a.tau_forget, "dt_s": a.dt,
                        "eta_anchored": round(eta, 4), "anchor_target": a.target, "delta_MBON11": round(d11, 4),
                        "rule": "trace-gated multiplicative depression dW = -eta*dt*DA(t)*delta_j*e_i(t)*W_ij on KC->MBON; "
                                "NOT the Gkanias 2022 rule of sections 7-9 (no silent-KC recovery term)"},
           "standard_forward": {"paired_A_MBON11": paired, "unpaired_B_MBON11": unpaired,
                                "note": "unpaired is the wiring-dependent number (Kenyon overlap at MBON11); paired is anchored"},
           "timing_MBON11": timing,
           "training_amount_MBON11": dose,
           "mechanism_controls": {"dopamine_alone_no_odour": da_alone, "odour_alone_no_dopamine": odour_alone,
                                  "note": "both are zero by the mechanism (no trace without odour; no depression without dopamine)"},
           "behaviour_from_timed_teaching": behaviour,
           "note": "eta is anchored once (standard protocol -> anchor_target at MBON11). With a binary code the MBON11 timing "
                   "and dose values are closed forms of (eta, tau_elig, dt, schedule) and carry no connectome information; "
                   "closed_form_MBON11 is reported next to each and coincides. Wiring-dependent: the unpaired-odour drop, "
                   "the compartment pattern outside MBON11, and the behavioural readout through the published valence map.",
           "ground_truth": "Hige 2015: forward pairing depresses, backward does not, more training more memory. Tully & Quinn: T-maze PI 0.44-0.53.",
           "elapsed_s": round(time.time() - t0, 1)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--dt", type=float, default=0.01); ap.add_argument("--odour-dur", type=float, default=5.0)
    ap.add_argument("--pulses", type=int, default=4); ap.add_argument("--da-onset", type=float, default=0.2)
    ap.add_argument("--da-freq", type=float, default=2.0); ap.add_argument("--da-width", type=float, default=0.05)
    ap.add_argument("--tau-elig", type=float, default=0.8); ap.add_argument("--tau-forget", type=float, default=1e9)
    ap.add_argument("--target", type=float, default=0.9)
    # negative = the train ENDS that many seconds before odour onset; positive = the train STARTS that long after it
    ap.add_argument("--timing-onsets", type=float, nargs="+", default=[-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0, 5.2, 6.0, 7.0])
    ap.add_argument("--dose-pulses", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = json_safe(run(a)); text = json.dumps(res, indent=1, allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
