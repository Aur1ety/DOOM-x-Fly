"""Teach the fly's own mushroom body and watch its choices change. A text walkthrough, behaviour first.

Everything here runs the circuit built from the MaleCNS connectome and the published dopamine plasticity rule
(sections 7 to 9 of docs/RESULTS.md). Nothing is scripted: the odour drives the real projection-neuron ->
Kenyon-cell -> MBON wiring, the dopamine lands where the connectome says it lands, and the choice is read
through the published transmitter-to-valence map. The only fitted number is one motor gain (beta); the size of
the learned drop is calibrated once to Hige et al. 2015, and everything else follows from the wiring.

    python -m flybrain.eval.mb_demo                         # uses the wiring cache from mb_build.py
    python -m flybrain.eval.mb_demo --modality visual       # the same, for a visual object
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR
from flybrain.eval.mb_behaviour import build_valence
from flybrain.eval.mb_olfactory import MushroomBody


def bar(p: float, width: int = 20) -> str:
    p = max(0.0, min(1.0, p)); n = round(p * width)
    return "[" + "#" * n + "-" * (width - n) + f"] {round(100 * p):3d}%"


def rule(ch: str = "-", n: int = 60) -> str:
    return ch * n


def run(a) -> None:
    wiring = a.wiring if (a.wiring and a.wiring.exists()) else None
    mb = MushroomBody(a.subgraph, a.neurons, sparsity=0.05, punish=("PPL101",), reward=("PAM",),
                      binary=True, wiring=wiring, modality=a.modality)
    val, counts = build_valence(mb, a.neurons)
    m11 = mb.type_mask("MBON11")
    thing = "visual object" if a.modality == "visual" else "odour"

    rng = np.random.default_rng(a.seed)
    stim = [rng.choice(len(mb.glom_names), size=6, replace=False) for _ in range(4)]
    codes = [mb.kc_code(mb.odour(s)) for s in stim]                         # A, B, C, D
    mb.calibrate_lr(codes[0], mb.compartment_mask("punish"), 0.9, "punish")  # one pairing = Hige's 90% at MBON11

    def resp():
        return torch.stack([mb.mbon_response(c) for c in codes])
    base = resp(); norm = float(base.abs().sum(1).mean())

    def score(R):
        return (R * val).sum(1) / norm

    def p_choose(sx, sy):
        return float(torch.sigmoid(a.beta * (sx - sy)))

    def train(idx, us):
        mb.reset()
        for _ in range(a.pairings):
            mb.reinforce(codes[idx], us)

    def m11_of(R, i):
        return float(R[i][m11].sum())

    print(rule("="))
    print("  Teaching the fly's own mushroom body")
    print("  connectome wiring + the published dopamine rule, nothing scripted")
    print(rule("="))
    print(f"\nThe fly can tell several {thing}s apart. Its output cells carry a value:")
    print(f"  {counts['approach']} MBON types say approach (GABA / acetylcholine),"
          f" {counts['avoid']} say avoid (glutamate).")
    s0 = score(base); pa = p_choose(s0[0], s0[1])
    lead = "it has no built-in preference" if abs(pa - 0.5) < 0.1 else "its starting preference comes from the wiring alone"
    print(f"\nBefore any training, {lead} between {thing} A and {thing} B:")
    print(f"  choose A  {bar(pa)}")
    print(f"  choose B  {bar(1 - pa)}")

    print(f"\n{rule()}\n-- pairing {thing} A with PUNISHMENT (PPL1 dopamine), one block --\n{rule()}")
    train(0, "punish"); aft = resp()
    b_a, a_a = m11_of(base, 0), m11_of(aft, 0)
    b_b, a_b = m11_of(base, 1), m11_of(aft, 1)
    print(f"the punished {thing} A weakens its own output cell (MBON-gamma1pedc):")
    print(f"  before  {bar(1.0)}")
    print(f"  after   {bar(a_a / b_a)}   (Hige 2015: ~90% drop; this size is calibrated)")
    print(f"the UNPAIRED {thing} B is left alone (the memory is specific):")
    print(f"  before  {bar(1.0)}")
    print(f"  after   {bar(a_b / b_b)}")
    sA = score(aft)
    print("\nnow the fly avoids A:")
    print(f"  choose A  {bar(p_choose(sA[0], sA[1]))}")
    print(f"  choose B  {bar(p_choose(sA[1], sA[0]))}")

    # reciprocal T-maze performance index (Tully & Quinn): train A test A vs B, train B test B vs A
    train(0, "punish"); s = score(resp()); h1 = p_choose(s[1], s[0]) - p_choose(s[0], s[1])
    train(1, "punish"); s = score(resp()); h2 = p_choose(s[0], s[1]) - p_choose(s[1], s[0])
    pi = 0.5 * (h1 + h2)
    print(f"\n  performance index {pi:+.2f}   (wild-type flies, one training cycle: 0.44 to 0.53)")

    print(f"\n{rule()}\n-- several memories, one circuit: A punished, C rewarded, D untouched --\n{rule()}")
    mb.reset()
    for _ in range(a.pairings):
        mb.reinforce(codes[0], "punish"); mb.reinforce(codes[2], "reward")
    s = score(resp())
    pairs = [("C (rewarded)", "A (punished)", 2, 0), ("D (untouched)", "A (punished)", 3, 0),
             ("C (rewarded)", "D (untouched)", 2, 3)]
    print("the choices come out in the right order, with no extra fitting:")
    for nx, ny, ix, iy in pairs:
        before = p_choose(s0[ix], s0[iy]); after = p_choose(s[ix], s[iy])
        print(f"  P(choose {nx:<14} over {ny:<14}) {before:.2f} -> {after:.2f}")

    print(f"\n{rule()}")
    print("what is the fly's wiring, and what did we set:")
    print("  - which cell learns, the compartment, and the SIGN of the behaviour: from the")
    print("    connectome (shuffle the dopamine-to-MBON map and every choice above inverts).")
    print("  - the SIZE of the drop: calibrated once to Hige's 90%, not predicted.")
    print("  - one motor-gain knob (beta), fitted so the index lands in the real range.")
    print("ground truth: Hige 2015 (depression), Owald 2015 (reward), Aso 2014 (valence),")
    print("              Tully & Quinn (T-maze). Details in docs/RESULTS.md sections 7 to 9.")
    print(rule())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiring", type=Path, default=OUT_DIR / "mb" / "mb_wiring.npz",
                    help="full-connectome MB cache (mb_build.py); falls back to --subgraph if absent")
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--modality", default="olfactory", choices=["olfactory", "visual"])
    ap.add_argument("--pairings", type=int, default=1); ap.add_argument("--beta", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
