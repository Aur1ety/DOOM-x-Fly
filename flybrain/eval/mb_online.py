"""Teach the fly in time: a training episode with a clock, so the memory forms from experience.

The other memory scripts (mb_olfactory, mb_behaviour) have no time axis: they call the plasticity rule once
per "pairing block" and the depression size is set by the learning rate. That is not how a fly learns. A fly
learns because the odour and the dopamine overlap IN TIME, and the timing decides everything (Hige et al.
2015: dopamine 0.2 s AFTER the odour writes the memory; dopamine before it does nothing).

Here the odour switches its Kenyon cells on for a few seconds, dopamine (PPL1 -> MBON) arrives as discrete
pulses at a set delay, and a KC->MBON synapse depresses only where recent Kenyon activity (a decaying
eligibility trace) and dopamine coincide at that instant. The memory is not written by hand; it emerges from
the overlap. One plasticity constant `eta` is anchored once so the standard protocol gives ~Hige's drop; the
TIMING law and the dependence on the amount of training then come out on their own, not fitted.

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
from flybrain.eval.mb_olfactory import MushroomBody


def da_pulse_train(n_pulses: int, onset: float, freq: float, width: float) -> np.ndarray:
    """Pulse start times (s): `n_pulses` at `freq` Hz starting `onset` s after odour onset."""
    return onset + np.arange(n_pulses) / freq if n_pulses > 0 else np.array([])


@torch.no_grad()
def train_episode(mb: MushroomBody, code: torch.Tensor, us: str, *, dt: float, odour_on: float, odour_off: float,
                  da_starts: np.ndarray, da_width: float, tau_elig: float, eta: float, tau_forget: float,
                  recover: float) -> None:
    """Run one timed episode, evolving the plastic KC->MBON weights mb.W in place.

    Eligibility trace e_kc(t): rises while a Kenyon cell is active, decays with tau_elig (the slow trace the
    dopamine acts on). At each instant with dopamine present, every KC->MBON synapse depresses by
    dW = -eta * DA(t) * delta_MBON * e_kc(t) * W * dt (multiplicative: monotonic, bounded), plus slow passive
    recovery toward rest. Dopamine that arrives AFTER the odour finds e_kc high and writes the memory; dopamine
    before the odour finds e_kc = 0 and does nothing, so the forward/backward timing law emerges (matches Hige
    2015). NOTE: Cohn, Morantte & Ruta 2015 show the full rule is bidirectional (dopamine BEFORE the odour
    potentiates); adding that arm cleanly needs onset-gating and is a documented next refinement."""
    delta = {"punish": mb.delta_punish, "reward": mb.delta_reward}[us]
    de = delta[mb.e_mbon]
    elig = torch.zeros(mb.n_kc)
    t_start = float(min(0.0, odour_on, da_starts.min() if len(da_starts) else 0.0))   # include dopamine before the odour
    t_end = max(odour_off, (da_starts.max() + da_width if len(da_starts) else 0.0)) + 3.0 * tau_elig
    n = round((t_end - t_start) / dt)
    da_on = np.zeros(n, bool)
    for s in da_starts:
        lo = max(0, round((s - t_start) / dt)); hi = max(0, round((s + da_width - t_start) / dt))
        da_on[lo:hi] = True
    for step in range(n):
        t = t_start + step * dt
        k = code if (odour_on <= t < odour_off) else torch.zeros_like(code)
        elig = elig * np.exp(-dt / tau_elig) + k * (dt / tau_elig)          # recent KC activity (0..~1)
        if da_on[step]:                                                     # dopamine now, KC active recently -> weaken
            mb.W = torch.clamp(mb.W - eta * dt * de * elig[mb.e_kc] * mb.W, 0.0, mb.w_max)
        if tau_forget < 1e6:
            mb.W = mb.W + (1.0 - mb.W) * (dt / tau_forget)                  # slow passive recovery (forgetting)


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
    before = mb.mbon_response(codes[0])

    da = da_pulse_train(a.pulses, a.da_onset, a.da_freq, a.da_width)
    std = {"dt": a.dt, "odour_on": 0.0, "odour_off": a.odour_dur, "da_starts": da, "da_width": a.da_width,
           "tau_elig": a.tau_elig, "tau_forget": a.tau_forget, "recover": a.recover}
    eta = calibrate_eta(mb, codes[0], m11, a.target, "punish", std)

    def episode(code, us="punish", **over):
        kw = {**std, **over}
        mb.reset(); train_episode(mb, code, us, eta=eta, **kw)

    # 1. standard forward episode: paired A vs unpaired B
    episode(codes[0]); paired = drop_at(mb, before, codes[0], m11); unpaired = drop_at(mb, before, codes[1], m11)

    # 2. TIMING LAW (emergent): dopamine onset swept relative to the odour. forward should write, backward should not.
    timing = {}
    for onset in a.timing_onsets:
        episode(codes[0], da_starts=da_pulse_train(a.pulses, onset, a.da_freq, a.da_width))
        timing[str(onset)] = drop_at(mb, before, codes[0], m11)

    # 3. TRAINING AMOUNT (emergent): more dopamine pulses -> more memory, saturating
    dose = {}
    for p in a.dose_pulses:
        episode(codes[0], da_starts=da_pulse_train(p, a.da_onset, a.da_freq, a.da_width))
        dose[str(p)] = drop_at(mb, before, codes[0], m11)

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
        episode(codes[idx], da_starts=da_pulse_train(a.pulses, onset, a.da_freq, a.da_width), us=us)
        return score(torch.stack([mb.mbon_response(c) for c in codes]))

    def pi_at(onset):                                                      # reciprocal T-maze index (Tully & Quinn)
        sA = score_after(0, onset); h1 = p_choose(sA[1], sA[0]) - p_choose(sA[0], sA[1])
        sB = score_after(1, onset); h2 = p_choose(sB[0], sB[1]) - p_choose(sB[1], sB[0])
        return round(0.5 * (h1 + h2), 4)
    s0 = score(base_resp); back = -abs(a.da_onset) - 2.0
    sf, sb = score_after(0, a.da_onset), score_after(0, back)
    behaviour = {"beta": a.beta,
                 "P_avoid_A": {"untrained": round(p_choose(s0[1], s0[0]), 4),
                               "forward_taught": round(p_choose(sf[1], sf[0]), 4),
                               "backward_taught": round(p_choose(sb[1], sb[0]), 4)},
                 "T_maze_PI": {"forward_taught": pi_at(a.da_onset), "backward_taught": pi_at(back)},
                 "note": "the TIMING of teaching decides the behaviour: forward teaching yields avoidance, backward yields none. "
                         "wild-type performance index 0.44-0.53."}
    # behavioural learning curve: avoidance grows with the amount of teaching
    pi_dose = {}
    for p in a.dose_pulses:
        episode(codes[0], da_starts=da_pulse_train(p, a.da_onset, a.da_freq, a.da_width))
        sA = score(torch.stack([mb.mbon_response(c) for c in codes]))
        pi_dose[str(p)] = round(p_choose(sA[1], sA[0]), 4)
    behaviour["P_avoid_A_vs_pulses"] = pi_dose

    res = {"protocol": {"odour_dur_s": a.odour_dur, "pulses": a.pulses, "da_onset_s": a.da_onset, "da_freq_hz": a.da_freq,
                        "tau_elig_s": a.tau_elig, "dt_s": a.dt, "eta_anchored": round(eta, 4), "anchor_target": a.target},
           "standard_forward": {"paired_A_MBON11": paired, "unpaired_B_MBON11": unpaired},
           "timing_law_MBON11": timing,
           "training_amount_MBON11": dose,
           "mechanism_controls": {"dopamine_alone_no_odour": da_alone, "odour_alone_no_dopamine": odour_alone},
           "behaviour_from_timed_teaching": behaviour,
           "note": "eta is anchored once to the standard forward protocol; the timing law, the dose curve, the controls and "
                   "the behavioural readout are consequences of the eligibility-trace mechanism plus the published valence map, not fitted.",
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
    ap.add_argument("--recover", type=float, default=1.0); ap.add_argument("--target", type=float, default=0.9)
    ap.add_argument("--timing-onsets", type=float, nargs="+", default=[-2.0, -1.0, -0.5, 0.0, 0.2, 0.5, 1.0, 2.0])
    ap.add_argument("--dose-pulses", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
