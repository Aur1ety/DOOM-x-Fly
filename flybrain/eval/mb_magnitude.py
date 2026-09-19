"""Is the size of the memory a prediction, or only a calibration? Score the rule against Hige's own pulse arms.

Sections 7 to 9 set one number by hand: the learning rate, chosen so that ONE pairing block reproduces Hige et
al. 2015's ~90 % drop of the paired odour's drive at MBON-gamma1pedc. That is a calibration and the write-up
says so. But Hige's pairing block is not one event: it is FOUR 1 ms dopamine pulses at 2 Hz starting 0.2 s
after the onset of a 1 s odour. How those pulses combine is a property of the rule rather than of the fit, so
it can be scored. Three candidate rules all hit 0.90 at the four-pulse arm and disagree elsewhere:

  block is the unit   sections 7 to 9: one rule application per pairing block, no pulse axis at all, so one
                      pulse and four pulses give the same 0.90. Ratio 1.
  independent pulses  each pulse multiplies the synapse by the same factor: drop(p) = 1 - (1 - x)^p with
                      x = 1 - 0.1^(1/4) = 0.438. Pulse count matters, pulse TIMING does not. Ratio 0.49.
  eligibility trace   the rule in mb_online.py: a pulse acts on the trace of recent Kenyon activity, which is
                      still charging early in the odour. Both count and timing matter.

Hige ran the discriminating experiment, though not quite the one that would separate them best. His single-
pulse arm gives 20 +/- 6.3 % depression of the UNPAIRED odour against 27 +/- 7.1 % for four pulses, a ratio of
0.74 +/- 0.30. That ratio is comparable to ours because under this rule a probe odour's drop is exactly (its
Kenyon-cell overlap with the trained odour) x (the paired drop), and the overlap cancels in a ratio.

The two arms differ in TWO ways, and both have to be reproduced: four pulses from +0.2 s versus one pulse at
+0.8 s. The delay is the point. By 0.8 s the eligibility trace has charged, so a trace rule expects one late
pulse to do nearly as much as four early ones, while a rule without timing expects it to do much less. Scoring
each arm at its own protocol is therefore not a detail; scoring the single pulse at +0.2 s instead (as an
earlier version of this module did) reverses which rule looks better.

What the measurement cannot do is choose: one ratio with that error bar leaves all three candidates standing.
The module reports the scores, says so, and states the experiment that WOULD separate them.

    python -m flybrain.eval.mb_magnitude --wiring $FLYBRAIN_OUT/mb/mb_wiring.npz --out $FLYBRAIN_OUT/mb/magnitude.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from flybrain import OUT_DIR
from flybrain.eval.mb_olfactory import MushroomBody, json_safe
from flybrain.eval.mb_online import closed_form_drop, da_pulse_train, train_episode

# The two arms of Hige et al. 2015 that this module scores, each with the protocol that produced it. They differ
# in pulse count AND in when the pulse falls inside the 1 s odour, so each is simulated on its own schedule.
ARMS = {
    "four_pulse": {"n_pulses": 4, "onset": 0.2, "freq": 2.0, "is_anchor": True,
                   "value": 0.27, "sem": 0.071, "n_cells": 7,
                   "what": "depression of the UNPAIRED odour (CS-) spike response at MBON-gamma1pedc",
                   "protocol": "1 s odour; four 1 ms CsChrimson pulses at 2 Hz starting 0.2 s after odour onset",
                   "source": "Hige, Aso, Modi, Rubin & Turner 2015, Neuron 88:985-998, Figure 1",
                   "quote": "(20 +/- 6.3 % ... versus 27 +/- 7.1 %; Figure 1)"},
    "single_pulse": {"n_pulses": 1, "onset": 0.8, "freq": 2.0, "is_anchor": False,
                     "value": 0.20, "sem": 0.063, "n_cells": None,
                     "what": "the same quantity after ONE pulse, delivered LATER in the odour",
                     "protocol": "1 s odour; a single 1 ms CsChrimson pulse 0.8 s after odour onset",
                     "source": "Hige et al. 2015, Figures S6F-S6I",
                     "quote": "we used 1-s odor pulses with just a single pulse of 1-ms light for pairing ... the "
                              "effects on CS- responses were slightly less than the four-pulse protocol we used "
                              "previously (20 +/- 6.3 %, mean +/- SEM ... versus 27 +/- 7.1 %)"},
}
# Direct measurements of the same synapse, for context on the calibrated magnitude. Not anchors: different
# induction (exogenous dopamine, direct Kenyon activation) and a different readout (EPSC amplitude).
CONTEXT = {
    "yamada2024_gamma_KC": {"value": 0.547, "sem": 0.085, "n_cells": 6,
                            "what": "LTD of gamma-KC -> MBON-gamma1pedc EPSC amplitude",
                            "source": "Yamada, Davidson & Hige 2024, J Physiol 602(9):2019-2045, Figs 2C-2D"},
    "yamada2024_alphabeta_KC": {"value": 0.818, "sem": 0.038, "n_cells": 6,
                                "what": "the same induction at alpha/beta-KC -> the same MBON; decays within ~10 min",
                                "source": "Yamada, Davidson & Hige 2024, Figs 8B-8C"},
}
NOTE_120 = ("Hige et al. also report that a 1 min odour with 120 pulses gave suppression 'similar' to four pulses. "
            "That separates nothing here: anchored at 0.90, every candidate is above 0.97 by eight pulses and at "
            "1.00 by 120, which on the CS- channel is a move from 0.27 to at most 0.30, inside its own 0.071 SEM. "
            "It is consistent with all three rules and is recorded only as a consistency check.")


def _bisect(f, lo: float, hi: float, iters: int = 60) -> float:
    """Smallest root of an increasing f on [lo, hi] (f(lo) < 0 < f(hi)); returns hi if it never crosses."""
    if f(hi) < 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _kw(tau: float, a) -> dict:
    return {"dt": a.dt, "odour_on": 0.0, "odour_off": a.odour_dur, "da_width": a.da_width,
            "tau_elig": tau, "tau_forget": a.tau_forget}


def arm_drop(eta: float, tau: float, arm: dict, a, onset: float | None = None, n: int | None = None) -> float:
    """Depression the trace rule gives for one arm, simulated on THAT arm's own schedule."""
    train = da_pulse_train(n if n is not None else arm["n_pulses"],
                           arm["onset"] if onset is None else onset, arm["freq"], a.da_width)
    return float(closed_form_drop(eta, 1.0, da_starts=train, **_kw(tau, a)))


def anchor_eta(tau: float, a, target: float) -> float:
    """Set the plasticity constant so the ANCHOR arm (Hige's four-pulse protocol) gives `target` at this tau."""
    return _bisect(lambda e: arm_drop(e, tau, ARMS["four_pulse"], a) - target, 0.0, a.eta_max)


def run(a) -> dict:
    t0 = time.time()
    four, single = ARMS["four_pulse"], ARMS["single_pulse"]
    counts = list(a.dose_pulses)

    # the timing-free null: pulse count only, anchored at the same four-pulse value
    x_ind = 1 - (1 - a.target) ** (1.0 / four["n_pulses"])

    # --- the measured ratio, and the three candidates scored on it
    r_meas = single["value"] / four["value"]
    r_sem = r_meas * float(np.hypot(single["sem"] / single["value"], four["sem"] / four["value"]))

    def z(r):
        return round((r - r_meas) / r_sem, 2)

    eta0 = anchor_eta(a.tau_elig, a, a.target)
    r_trace = arm_drop(eta0, a.tau_elig, single, a) / a.target
    trace_by_tau = {f"{tau:g}": round(arm_drop(anchor_eta(tau, a, a.target), tau, single, a) / a.target, 3)
                    for tau in a.taus}
    candidates = {
        "eligibility_trace": {"ratio": round(r_trace, 3), "z_vs_measured": z(r_trace),
                              "across_tau_sweep": trace_by_tau,
                              "what": "mb_online's rule: both pulse count and pulse timing matter"},
        "independent_pulses": {"ratio": round(x_ind / a.target, 3), "z_vs_measured": z(x_ind / a.target),
                               "what": "each pulse multiplies the synapse by the same factor; timing does not enter"},
        "block_is_the_unit": {"ratio": 1.0, "z_vs_measured": z(1.0),
                              "what": "sections 7 to 9: one pairing block is one rule application, so pulse count "
                                      "does not enter either"},
    }
    best = min(candidates, key=lambda k: abs(candidates[k]["z_vs_measured"]))
    test = {"quantity": "depression after the single-pulse arm divided by that after the four-pulse arm, at MBON-gamma1pedc",
            "why_comparable": "a probe odour's drop is (its Kenyon-cell overlap with the trained odour) x (the paired "
                              "drop), and the overlap does not depend on the dopamine schedule, so it cancels in the ratio",
            "arms_differ_in": "pulse COUNT (4 versus 1) and pulse TIMING (+0.2 s versus +0.8 s into a 1 s odour); each "
                              "arm is simulated on its own schedule, because under a trace rule the delay is the mechanism",
            "measured": {"ratio": round(r_meas, 3), "sem_propagated": round(r_sem, 3),
                         "from": f"{single['value']} +/- {single['sem']} (1 pulse at +{single['onset']} s) over "
                                 f"{four['value']} +/- {four['sem']} (4 pulses from +{four['onset']} s)"},
            "candidates": candidates, "closest_to_measurement": best,
            "does_it_separate_them": False,
            "reading": ("the measured ratio sits between the candidates and separates none of them: at this error bar "
                        f"the trace rule is closest (z {candidates['eligibility_trace']['z_vs_measured']}), independent "
                        f"pulses and the block rule are about equally far on either side "
                        f"({candidates['independent_pulses']['z_vs_measured']} and "
                        f"{candidates['block_is_the_unit']['z_vs_measured']}). One ratio of two noisy means cannot "
                        "choose among them. What it does show is that a rule in which the delay matters has no trouble "
                        "with the observation that ONE late pulse does nearly as much as four early ones, which is the "
                        "part a timing-free rule has to call a coincidence."),
            "caveats_worst_first": [
                ("the measured numbers are the CS- (unpaired) channel; the CS+ magnitude for the single-pulse arm "
                 "exists only in a supplementary plot and is not stated in readable text anywhere"),
                ("the two arms come from different experiment sets and the single-pulse arm's odour pair could not be "
                 "confirmed, while the same paper shows CS- depression scales with Kenyon-cell overlap (Pearson "
                 "r = 0.90), so odour identity is an unexcluded alternative explanation for 20 % versus 27 %"),
                "n for the single-pulse arm is not stated in the readable text; the four-pulse arm is n = 7",
                ("the propagated SEM is a first-order delta-method value on a ratio of two noisy means; with relative "
                 "errors near 0.3 the ratio's distribution is skewed, so the z values are indicative, not exact"),
                "spike counts, not the EPSC charge transfer that the 0.90 calibration uses"],
            "note_120_pulses": NOTE_120}

    # --- the experiment that WOULD separate the rules: hold the timing fixed and vary only the count
    fixed = {}
    for tau in a.taus:
        eta = anchor_eta(tau, a, a.target)
        d = [round(arm_drop(eta, tau, four, a, onset=a.da_onset, n=p), 4) for p in counts]
        inc = [round(d[i] - (d[i - 1] if i else 0.0), 4) for i in range(len(d))]
        i1, i2 = counts.index(1), counts.index(2)
        rates = {f"{f:g}": bool(_incs_rise(eta if f == four["freq"] else anchor_eta(tau, a, a.target), tau, a, f))
                 for f in a.da_freqs} if a.rate_sweep else None
        fixed[f"{tau:g}"] = {"eta_anchored": round(eta, 3), "drop_by_pulses": dict(zip(map(str, counts), d)),
                             "increment_per_pulse": dict(zip(map(str, counts), inc)),
                             "second_pulse_adds_more_than_first": bool(inc[i2] > inc[i1]),
                             "second_adds_more_by_pulse_rate_hz": rates,
                             "eta_hit_ceiling": bool(eta >= a.eta_max - 1e-6)}
    accel = [k for k, v in fixed.items() if v["second_pulse_adds_more_than_first"]]
    eta_fixed = anchor_eta(a.tau_elig, a, a.target)
    one_at_anchor_onset = round(arm_drop(eta_fixed, a.tau_elig, four, a, onset=a.da_onset, n=1), 4)
    discriminating = {
        "protocol": f"all pulses from +{a.da_onset} s (the four-pulse arm's onset), varying ONLY the count. This is "
                    f"NOT Hige's single-pulse arm, which delivered its pulse at +{single['onset']} s.",
        "one_pulse_trace_rule": one_at_anchor_onset, "one_pulse_independent": round(x_ind, 4),
        "separation": round(x_ind / one_at_anchor_onset, 2) if one_at_anchor_onset else None,
        "by_tau": fixed,
        "sign_test": {"what": "does the 2nd pulse add more than the 1st?",
                      "trace_rule_taus_where_true": accel, "independent_pulses": False, "block_rule": None,
                      "why_it_helps": "it compares two increments, so it needs no absolute magnitude and no anchor "
                                      "value; it does still depend on the trace constant being comparable to or "
                                      "longer than the pulse interval, and on the pulses falling inside the odour",
                      "not_unique_to_a_trace": "any mechanism that grows within a train (dopamine facilitation, "
                                               "receptor sensitisation, a threshold) can also give a rising "
                                               "increment; the test rules OUT timing-free pulse independence, it "
                                               "does not rule IN an eligibility trace"}}

    # --- the inverse: what a single-pulse measurement at the anchor onset would say about the trace constant.
    # Scanned rather than bisected: the relation need not be monotone in tau at every onset.
    scan_taus = list(np.geomspace(min(a.taus), max(a.taus), a.tau_scan))
    scan = [(t, arm_drop(anchor_eta(t, a, a.target), t, four, a, onset=a.da_onset, n=1)) for t in scan_taus]
    vals = [v for _, v in scan]
    monotone = bool(all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))
    band = [round(min(vals), 4), round(max(vals), 4)]
    implied = {}
    for m in a.hypothetical_measurements:
        roots = [round(float(scan[i][0] + (scan[i + 1][0] - scan[i][0]) *
                             (m - vals[i]) / (vals[i + 1] - vals[i])), 3)
                 for i in range(len(vals) - 1) if (vals[i] - m) * (vals[i + 1] - m) <= 0 and vals[i] != vals[i + 1]]
        implied[str(m)] = roots or None
    inverse = {"measured_single_pulse_at_anchor_onset_implies_tau_elig_s": implied,
               "reachable_band": band, "monotone_in_tau": monotone, "scan_points": a.tau_scan,
               "note": "roots are listed because the relation is not guaranteed monotone; None means no swept trace "
                       "constant reproduces that value, which would itself rule the trace form out"}

    # --- the closed form is exact only for a binary code; check it against the full circuit once
    check = None
    if a.wiring and Path(a.wiring).exists():
        mb = MushroomBody(a.subgraph, a.neurons, sparsity=0.05, punish=("PPL101",), reward=("PAM",),
                          binary=True, wiring=a.wiring, modality="olfactory")
        rng = np.random.default_rng(a.seed)
        code = mb.kc_code(mb.odour(rng.choice(len(mb.glom_names), size=6, replace=False)))
        m11 = mb.type_mask("MBON11"); before = mb.mbon_response(code)
        meas = {}
        for name, arm in ARMS.items():
            mb.reset()
            train_episode(mb, code, "punish", eta=eta0,
                          da_starts=da_pulse_train(arm["n_pulses"], arm["onset"], arm["freq"], a.da_width),
                          **_kw(a.tau_elig, a))
            meas[name] = round(1 - float(mb.mbon_response(code)[m11].sum() / before[m11].sum()), 4)
        cf = {name: round(arm_drop(eta0, a.tau_elig, arm, a), 4) for name, arm in ARMS.items()}
        check = {"full_circuit": meas, "closed_form": cf, "anchor_target": a.target,
                 "delta_MBON11": round(float(mb.delta_punish[m11].max()), 4),
                 "agrees": bool(all(abs(meas[k] - cf[k]) < 2e-3 for k in ARMS)
                                and abs(meas["four_pulse"] - a.target) < 2e-3)}

    verdict = (
        f"The magnitude itself stays CALIBRATED: no published number fixes the size of a single dopamine-driven "
        f"depression at this synapse, so one number has to be set, and it is the learning rate. What is NOT calibrated "
        f"is how the pulses inside a block combine, and Hige et al. 2015 ran arms that bear on it: one pulse at "
        f"+{single['onset']} s gives {single['value']} +/- {single['sem']} against {four['value']} +/- {four['sem']} for "
        f"four pulses from +{four['onset']} s, a ratio of {round(r_meas, 2)} +/- {round(r_sem, 2)}. Scoring each arm on "
        f"its OWN schedule (the arms differ in timing as well as count, and under a trace rule the delay is the whole "
        f"mechanism) gives: eligibility trace {candidates['eligibility_trace']['ratio']} "
        f"(z {candidates['eligibility_trace']['z_vs_measured']}), independent pulses "
        f"{candidates['independent_pulses']['ratio']} (z {candidates['independent_pulses']['z_vs_measured']}), block-is-"
        f"the-unit {candidates['block_is_the_unit']['ratio']} (z {candidates['block_is_the_unit']['z_vs_measured']}). "
        f"The trace rule is closest, but ONE ratio with this error bar separates none of them and the honest statement "
        f"is that the measurement is under-powered. The qualitative 120-pulse report adds nothing either, since every "
        f"candidate predicts 'similar' there. What the trace rule does buy is an explanation rather than a coincidence: "
        f"a single pulse delivered late in the odour, when the trace has charged, should do nearly as much as four "
        f"early ones, and that is what was seen. The experiment that WOULD separate the rules holds the timing fixed "
        f"and varies only the count: at the four-pulse arm's own onset the trace rule predicts {one_at_anchor_onset} "
        f"for one pulse against {round(x_ind, 4)} for independent pulses and {a.target} for the block rule. The "
        f"magnitude-free version of that is the curvature: with a trace comparable to or slower than the pulse interval "
        f"the SECOND pulse adds more than the first (tau = {', '.join(accel) or 'none in this sweep'} s), which pulse "
        f"independence can never give, though it is not unique to a trace. Read backwards, that experiment measures the "
        f"trace constant, which is hand-set here and which the literature sweep found unmeasured.")

    res = {"anchor": {"arm": "Hige et al. 2015 four-pulse protocol",
                      "protocol": f"{four['n_pulses']} pulses at {four['freq']} Hz from +{four['onset']} s into a "
                                  f"{a.odour_dur} s odour, {a.da_width * 1000:g} ms wide, dt {a.dt * 1000:g} ms",
                      "target_drop_MBON11": a.target, "eta_at_default_tau": round(eta0, 3),
                      "tau_elig_default": a.tau_elig,
                      "note": "pulses 3 and 4 fall after the 1 s odour ends, on a decaying trace; that is Hige's "
                              "protocol, not a modelling choice"},
           "measured_test_pulse_arms": test,
           "discriminating_experiment_timing_held_fixed": discriminating,
           "inverse_implied_trace_constant": inverse,
           "same_synapse_magnitudes_for_context": {**CONTEXT,
               "note": "direct EPSC measurements at the same synapse, induced with exogenous dopamine and direct "
                       "Kenyon activation rather than odour and PPL1, so not substitute anchors. They bracket the "
                       "calibrated 0.90 and show the magnitude is Kenyon-class dependent (0.55 gamma, 0.82 "
                       "alpha/beta), which this model does not represent: its rule is identical for every Kenyon cell."},
           "closed_form_vs_full_circuit": check,
           "ground_truth": ("Hige et al. 2015 (Neuron): four 1 ms PPL1-gamma1pedc pulses at 2 Hz from +0.2 s into a 1 s "
                            "odour depress the PAIRED odour's MBON-gamma1pedc response by 80 +/- 5.7 % (spikes) / "
                            "90 +/- 3.7 % (EPSC charge). No number is given for the PAIRED odour at any other pulse "
                            "count, which is what leaves the single-pulse paired value free to be a prediction. A "
                            "coarse series DOES exist in the UNPAIRED (CS-) channel, 20 +/- 6.3 % at one pulse against "
                            "27 +/- 7.1 % at four, and that series is what this module scores the rules against."),
           "verdict": verdict, "elapsed_s": round(time.time() - t0, 1)}
    return res


def _incs_rise(eta: float, tau: float, a, freq: float) -> bool:
    """Does the 2nd pulse add more than the 1st at this pulse rate, with the constant re-anchored at this rate?"""
    arm = {**ARMS["four_pulse"], "freq": freq}
    e = _bisect(lambda x: arm_drop(x, tau, arm, a) - a.target, 0.0, a.eta_max)
    d1 = arm_drop(e, tau, arm, a, onset=a.da_onset, n=1)
    d2 = arm_drop(e, tau, arm, a, onset=a.da_onset, n=2)
    return (d2 - d1) > d1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz")
    ap.add_argument("--target", type=float, default=0.9, help="the measured drop the anchor arm must reproduce")
    # Hige's protocol: 1 s odour, 1 ms pulses. dt must be <= the pulse width or a pulse spans zero steps.
    ap.add_argument("--odour-dur", type=float, default=1.0); ap.add_argument("--da-width", type=float, default=0.001)
    ap.add_argument("--dt", type=float, default=0.001); ap.add_argument("--tau-forget", type=float, default=1e9)
    ap.add_argument("--da-onset", type=float, default=0.2, help="onset of the anchor arm, reused by the fixed-timing sweep")
    ap.add_argument("--tau-elig", type=float, default=0.8, help="the hand-set trace constant the prediction depends on")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    ap.add_argument("--da-freqs", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--rate-sweep", action="store_true", default=True)
    ap.add_argument("--no-rate-sweep", dest="rate_sweep", action="store_false")
    ap.add_argument("--dose-pulses", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 16])
    ap.add_argument("--hypothetical-measurements", type=float, nargs="+", default=[0.15, 0.2, 0.3, 0.44, 0.6])
    ap.add_argument("--tau-scan", type=int, default=60); ap.add_argument("--eta-max", type=float, default=1e7)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    res = json_safe(run(a)); text = json.dumps(res, indent=1, allow_nan=False)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
