"""A dashboard video of the fly's own mushroom body learning, behaviour first.

Left: the simulated neurons' cell bodies at their scanned MaleCNS positions (front view), with the Kenyon
cells that fire for the current odour lit up. Right: the output cell's response (MBON-gamma1pedc) for the
trained odour and an unpaired one, and the fly's two-arm choice. Nothing is scripted: the odour drives the
real projection-neuron -> Kenyon-cell -> MBON wiring, the dopamine lands where the connectome says, the
choice is read through the published transmitter-to-valence map. The size of the drop is calibrated once to
Hige et al. 2015; everything else follows from the wiring.

    python -m flybrain.eval.mb_video --out $FLYBRAIN_OUT/videos/mb_memory.mp4
    python -m flybrain.eval.mb_video --frames 30 120 300 --png-dir /tmp/mbframes   # dump stills, no video
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from flybrain import OUT_DIR
from flybrain.eval.dashboard import BrainPanel, FFmpegWriter, _font, soma_positions
from flybrain.eval.mb_behaviour import build_valence
from flybrain.eval.mb_olfactory import MushroomBody

W, H, FPS = 1280, 720, 20
BG = (12, 14, 18)
DIM = (150, 156, 168)
FG = (232, 236, 242)
CYAN = (90, 210, 235)
RED = (235, 95, 90)
GREEN = (120, 210, 130)
CLOUD = (16, 78, W // 2 + 150, H - 40)                     # x0, y0, x1, y1
RX = CLOUD[2] + 40                                          # right column left edge
RW = W - RX - 30


def bar(d, x, y, w, h, frac, color, label, val_text):
    d.rectangle([x, y, x + w, y + h], outline=(60, 66, 76), width=2)
    fw = int(max(0.0, min(1.0, frac)) * (w - 4))
    d.rectangle([x + 2, y + 2, x + 2 + fw, y + h - 2], fill=color)
    d.text((x, y - 22), label, fill=DIM, font=_font(17))
    d.text((x + w - 62, y + h // 2 - 10), val_text, fill=FG, font=_font(18))


def build_states(mb, val, beta, seed):
    """Endpoints of ONE pairing block (Hige 2015 = one block ~= 90% at MBON-gamma1pedc)."""
    rng = np.random.default_rng(seed)
    stim = [rng.choice(len(mb.glom_names), size=6, replace=False) for _ in range(4)]
    codes = [mb.kc_code(mb.odour(s)) for s in stim]                     # A, B, C, D
    mb.calibrate_lr(codes[0], mb.compartment_mask("punish"), 0.9, "punish")
    m11 = mb.type_mask("MBON11")

    def resp():
        return torch.stack([mb.mbon_response(c) for c in codes])
    base = resp(); norm = float(base.abs().sum(1).mean())
    b_a = float(base[0][m11].sum()); b_b = float(base[1][m11].sum())

    def score(R):
        return (R * val).sum(1) / norm

    def pc(sx, sy):
        return float(torch.sigmoid(torch.tensor(beta) * (sx - sy)))
    s0 = score(base)
    mb.reset(); mb.reinforce(codes[0], "punish"); R = resp(); s = score(R)     # one aversive block on A
    a1 = float(R[0][m11].sum()) / b_a; b1 = float(R[1][m11].sum()) / b_b
    mb.reset(); mb.reinforce(codes[0], "punish"); mb.reinforce(codes[2], "reward"); sc = score(resp())  # + reward on C
    return {"codes": codes,
            "a1": a1, "b1": b1,
            "avoid0": pc(s0[1], s0[0]), "avoid1": pc(s[1], s[0]),
            "preferC0": pc(s0[2], s0[0]), "preferC1": pc(sc[2], sc[0])}


def frame(mb, brain, kc_node_of, st, phase, cap, active_code, mbonA, mbonB, choice, choice_label,
          dopamine, dop_text, pairing_txt):
    act = np.zeros(len(brain.nodes), np.float32)
    if active_code is not None:
        node_on = np.zeros(mb._n_nodes, bool)
        node_on[mb.kc[active_code.numpy() > 0]] = True
        act = node_on[brain.nodes].astype(np.float32)
    cloud = brain.render(act)                                            # [h,w,3]
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    d.text((24, 12), "A fruit fly's own memory circuit (mushroom body), from the MaleCNS scan", fill=FG, font=_font(28))
    d.text((24, 46), "the odour drives the real wiring; dopamine lands where the connectome says; the choice is read through the published valence map", fill=DIM, font=_font(16))
    img.paste(Image.fromarray(cloud), (CLOUD[0], CLOUD[1]))
    d.text((CLOUD[0] + 8, CLOUD[1] + 4), "simulated cell bodies, scanned positions; bright = Kenyon cells firing for this odour", fill=DIM, font=_font(15))
    if dopamine:
        dc = GREEN if "reward" in dop_text else RED
        d.rectangle([CLOUD[0], CLOUD[1], CLOUD[2], CLOUD[3]], outline=dc, width=5)
        d.text((CLOUD[0] + 14, CLOUD[3] - 34), dop_text, fill=dc, font=_font(22))
    # right column: MBON response bars, choice, caption (labels sit above each bar, so leave headroom)
    y = 96
    d.text((RX, y), "output cell response  (MBON-gamma1pedc)", fill=FG, font=_font(19)); y += 52
    bar(d, RX, y, RW, 34, mbonA, RED if mbonA < 0.6 else CYAN, "trained odour A", f"{round(mbonA * 100)}%"); y += 80
    bar(d, RX, y, RW, 34, mbonB, CYAN, "unpaired odour B (specific)", f"{round(mbonB * 100)}%"); y += 104
    d.text((RX, y), choice_label, fill=FG, font=_font(19)); y += 52
    bar(d, RX, y, RW, 34, choice, GREEN, "P(the fly's choice)", f"{round(choice * 100)}%"); y += 52
    if pairing_txt:
        d.text((RX, y), pairing_txt, fill=DIM, font=_font(18)); y += 30
    d.rectangle([RX, H - 168, W - 30, H - 40], outline=(50, 56, 66), width=2)
    d.text((RX + 10, H - 160), phase, fill=CYAN, font=_font(20))
    for i, line in enumerate(cap):
        d.text((RX + 10, H - 128 + i * 24), line, fill=FG, font=_font(17))
    d.text((24, H - 26), "size of the drop calibrated once to Hige 2015; specificity, compartment and the sign of the choice come from the wiring", fill=DIM, font=_font(15))
    return np.asarray(img)


def render(a) -> None:
    from flybrain.model.core import load_subgraph
    sub = load_subgraph(a.subgraph); is_cl = np.asarray(sub["is_clamped"], bool)
    mb = MushroomBody(a.subgraph, a.neurons, sparsity=0.05, punish=("PPL101",), reward=("PAM",), binary=True)
    mb._n_nodes = len(sub["bodyId"])
    val, _ = build_valence(mb, a.neurons)
    nodes, xyz = soma_positions(sub, a.annotations)
    brain = BrainPanel(nodes, xyz, is_cl, CLOUD[2] - CLOUD[0], CLOUD[3] - CLOUD[1])
    S = build_states(mb, val, a.beta, a.seed)
    codes = S["codes"]
    AV = "two-arm choice: avoid odour A"; PC = "two-arm choice: prefer odour C over A"

    seq = []
    def hold(sec, **kw):
        for _ in range(int(sec * FPS)):
            seq.append(kw)
    def ramp(sec, v0, v1, **kw):                                         # ease numeric fields v0 -> v1
        n = int(sec * FPS)
        for t in range(n):
            f = 0.5 - 0.5 * np.cos(np.pi * (t + 1) / n)
            seq.append({**kw, **{k: v0[k] + (v1[k] - v0[k]) * f for k in v0}})

    hold(1.5, active_code=None, mbonA=1.0, mbonB=1.0, choice=S["avoid0"], choice_label=AV, dopamine=False, dop_text="", pairing_txt="",
         phase="1. the resting circuit", cap=["the fly can smell several odours;", "at rest it has no preference."])
    hold(2.0, active_code=codes[0], mbonA=1.0, mbonB=1.0, choice=S["avoid0"], choice_label=AV, dopamine=False, dop_text="", pairing_txt="",
         phase="2. present odour A", cap=["a specific sparse set of Kenyon", "cells fires for odour A."])
    hold(1.6, active_code=codes[1], mbonA=1.0, mbonB=1.0, choice=S["avoid0"], choice_label=AV, dopamine=False, dop_text="", pairing_txt="",
         phase="2. present odour B", cap=["a different set fires for odour B;", "the code is odour-specific."])
    ramp(2.6, {"mbonA": 1.0, "choice": S["avoid0"]}, {"mbonA": S["a1"], "choice": S["avoid1"]},
         active_code=codes[0], mbonB=1.0, choice_label=AV, dopamine=True, dop_text="DOPAMINE  PPL1  (punishment)",
         pairing_txt="one pairing block (Hige 2015)", phase="3. pair odour A with punishment",
         cap=["dopamine (PPL1) arrives with odour A;", "the KC->MBON synapses weaken."])
    hold(2.2, active_code=codes[0], mbonA=S["a1"], mbonB=S["b1"], choice=S["avoid1"], choice_label=AV, dopamine=False, dop_text="", pairing_txt="",
         phase="4. test: the memory is specific", cap=["odour A barely drives its cell now;", "the unpaired odour B is untouched."])
    hold(2.4, active_code=None, mbonA=S["a1"], mbonB=S["b1"], choice=S["avoid1"], choice_label=AV, dopamine=False, dop_text="", pairing_txt="",
         phase="5. the choice has changed", cap=[f"the fly now avoids A: {round(S['avoid1'] * 100)}%.", "wild-type flies: index 0.44-0.53."])
    ramp(2.4, {"choice": S["preferC0"]}, {"choice": S["preferC1"]},
         active_code=codes[2], mbonA=S["a1"], mbonB=S["b1"], choice_label=PC, dopamine=True, dop_text="DOPAMINE  PAM  (reward)",
         pairing_txt="one pairing block", phase="6. reward works too",
         cap=["pairing odour C with reward (PAM),", f"the fly comes to prefer C: {round(S['preferC1'] * 100)}%."])
    hold(2.0, active_code=None, mbonA=S["a1"], mbonB=S["b1"], choice=S["preferC1"], choice_label=PC, dopamine=False, dop_text="", pairing_txt="",
         phase="7. one circuit, several memories", cap=["specific, in the right compartment,", "and it changes what the fly does."])

    kc_node_of = None
    if a.png_dir:
        Path(a.png_dir).mkdir(parents=True, exist_ok=True)
        for fi in a.frames:
            kw = seq[min(fi, len(seq) - 1)]
            Image.fromarray(frame(mb, brain, kc_node_of, None, **kw)).save(Path(a.png_dir) / f"frame_{fi:04d}.png")
        print(f"wrote {len(a.frames)} stills to {a.png_dir}; total timeline {len(seq)} frames ({len(seq)/FPS:.1f}s)")
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    vw = FFmpegWriter(a.out, W, H, FPS)
    for kw in seq:
        vw.append_data(frame(mb, brain, kc_node_of, None, **kw))
    vw.close()
    print(f"wrote {a.out}  ({len(seq)} frames, {len(seq)/FPS:.1f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--annotations", type=Path, default=None, help="MaleCNS body-annotations feather (for soma positions)")
    ap.add_argument("--beta", type=float, default=11.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "videos" / "mb_memory.mp4")
    ap.add_argument("--png-dir", type=Path, default=None); ap.add_argument("--frames", type=int, nargs="+", default=[30, 120, 300])
    a = ap.parse_args(argv)
    if a.annotations is None:
        from flybrain import MALECNS_DIR
        a.annotations = MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
    render(a)


if __name__ == "__main__":
    main()
