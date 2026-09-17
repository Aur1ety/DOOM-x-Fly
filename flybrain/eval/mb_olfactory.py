"""Olfactory learning in the fly's own mushroom body, run as the circuit actually works.

Structure from the MaleCNS connectome (subgraph v5): excitatory uniglomerular olfactory projection neurons (PN)
-> Kenyon cells (KC) -> mushroom body output neurons (MBON), plus the dopaminergic PPL1 (punishment) and PAM
(reward) -> MBON wiring that says which MBON compartment each reinforcement reaches.

Physiology (disclosed, grounded in measured KC properties): Kenyon cells are feedforward coincidence detectors
with a high threshold, and APL feedback keeps only the most strongly driven ~5 % active (k-winners-take-all).
The uniform recurrent rate model used for Doom erases this code (mb_sparse.py); the connectome's PN->KC wiring
alone produces it.

Learning rule (Gkanias, McCurdy, Nitabach & Webb, eLife 2022) on KC->MBON synapses only:
    dW_ij = -lr * delta_j * (k_i + W_ij - 1),   W in [0, w_max]
delta_j >= 0 is the dopamine MBON j receives (from the connectome's DAN -> MBON synapse counts), k_i the KC
activity. For an active KC this depresses the synapse; for a silent KC it relaxes a depressed synapse back
toward 1 ("dopamine alone restores"). Reward is also depression, in the PAM compartments (Owald et al. 2015).

What is a prediction and what is not. With a binary KC code the paired-odour drop at an MBON after p pairings
is exactly 1 - (1 - lr*delta)^p: its size is set by lr, not by the wiring. So lr is calibrated ONCE so that
one pairing gives Hige et al. 2015's ~90 % (synaptic charge) at MBON11 (the gamma1pedc compartment), and that
number is reported as calibrated, not as a result. Everything else follows from the wiring and the rule: which
other MBON types change (compartment map), how much unpaired odours change (code overlap), how the change
generalises to similar odours, and whether several memories can coexist.

Controls: dan_mbon shuffle (permute which MBON type each reinforcement reaches: the compartment endpoint must
fail), dense code (sparsity 1: the specificity endpoint must fail), and degree-preserving pn_kc / kc_mbon
shuffles (expected to match; they do not test the result). Reinforcing with no KC activity is zero by the
rule's algebra and is reported as that, not as a timing test: this model has no time axis.

    python -m flybrain.eval.mb_olfactory --binary-code --out $FLYBRAIN_OUT/mb/olf3_binary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from flybrain import OUT_DIR

PN_SUFFIX = {"adPN", "lPN", "ilPN", "l2PN", "il2PN"}   # excitatory uniglomerular olfactory PN classes (MaleCNS names)
W_REST = 1.0
HIGE_CHARGE_DROP = 0.9


def is_olfactory_pn(t: str) -> bool:
    """<glomerulus>_<class>PN: excitatory, uniglomerular, olfactory. Multiglomerular types carry '+', the VP
    glomeruli are thermo/hygrosensory, and vPNs are GABAergic; all excluded."""
    if "_" not in t:
        return False
    g, s = t.rsplit("_", 1)
    return s in PN_SUFFIX and "+" not in g and not g.startswith("VP")


class MushroomBody:
    """Feedforward PN -> KC(kWTA) -> MBON circuit from the connectome, with plastic KC->MBON synapses."""

    def __init__(self, subgraph: Path | None = None, neurons: Path | None = None, sparsity: float = 0.05, punish=("PPL101",),
                 reward=("PAM",), lr: float = 0.9, w_max: float = 2.0, shuffle: str | None = None, seed: int = 0,
                 binary: bool = False, recover_rate: float = 1.0, wiring: Path | None = None, modality: str = "olfactory"):
        self.binary, self.sparsity, self.lr, self.w_max, self.recover_rate = binary, sparsity, lr, w_max, recover_rate
        self.modality = modality
        rng = np.random.default_rng(seed)
        if wiring is not None:                                               # full-connectome cache (mb_build.py)
            self._load_from_wiring(wiring, modality)
        else:                                                               # weight-thresholded Doom subgraph
            self._load_from_subgraph(subgraph, neurons)
        self.glom_names = self.channel_names; self.glom = self.channel      # backward-compatible aliases
        n_kc = self._Wpk.shape[0]
        self._tiebreak = rng.random(n_kc)                                    # fixed random order for kWTA ties
        W_pk = self._Wpk
        if shuffle == "pn_kc":                                               # degree-preserving: permute input identity
            W_pk = W_pk[:, rng.permutation(W_pk.shape[1])]
        self.W_pk = W_pk.tocsr()
        W_km = self._Wkm
        if shuffle == "kc_mbon":                                             # degree-preserving: permute KC identity
            W_km = sparse.coo_matrix((W_km.data, (W_km.row, rng.permutation(W_km.col))), shape=W_km.shape)
        self.e_mbon = torch.as_tensor(W_km.row, dtype=torch.long); self.e_kc = torch.as_tensor(W_km.col, dtype=torch.long)
        self.e_base = torch.as_tensor(W_km.data, dtype=torch.float32)       # synapse counts (KCs are excitatory)
        self.n_edges = len(self.e_base)
        self.W = torch.ones(self.n_edges)
        self.delta_punish = self._dan_template(punish)
        self.delta_reward = self._dan_template(reward)
        self.n_input = self.W_pk.shape[1]; self.n_kc = self.W_pk.shape[0]; self.n_mbon = len(self.mbon_type)
        self.n_drivable_kc = int((np.asarray((self.W_pk != 0).sum(axis=1)).ravel() > 0).sum())   # KCs this pathway can drive
        self.n_punish_dan = int(sum(any(tp.startswith(p) for p in punish) for tp in self._dan_type))
        self.n_reward_dan = int(sum(any(tp.startswith(p) for p in reward) for tp in self._dan_type))
        if shuffle == "dan_mbon":                                            # informative control: scramble the compartment map
            self.delta_punish = self._permute_types(self.delta_punish, rng)
            self.delta_reward = self._permute_types(self.delta_reward, rng)

    def _load_from_subgraph(self, subgraph, neurons):
        import pandas as pd

        from flybrain.model.core import load_subgraph
        sub = load_subgraph(subgraph); body = np.asarray(sub["bodyId"]); m = len(body)
        n = pd.read_parquet(neurons, columns=["bodyId", "type"]).drop_duplicates("bodyId").set_index("bodyId")
        t = n["type"].reindex(body).fillna("").astype(str).to_list()
        A = sparse.csr_matrix((np.asarray(sub["weight"], np.float64), np.asarray(sub["indices_pre"], np.int64),
                               np.asarray(sub["indptr_post"], np.int64)), shape=(m, m))
        sel = lambda f: np.flatnonzero(np.array([f(x) for x in t]))
        pn = sel(is_olfactory_pn); kc = sel(lambda x: x.startswith("KC")); mbon = sel(lambda x: x.startswith("MBON"))
        dan = sel(lambda x: x.startswith(("PPL1", "PAM")))
        self.pn, self.kc, self.mbon = pn, kc, mbon                           # subgraph-body indices (behaviour readout uses these)
        self.mbon_type = [t[i] for i in mbon]; self._dan_type = [t[i] for i in dan]
        ch = {}
        for j, i in enumerate(pn):
            ch.setdefault(t[i].split("_")[0], []).append(j)
        self.channel_names = sorted(ch); self.channel = ch
        self._Wpk = A[kc][:, pn].tocsr(); self._Wkm = A[mbon][:, kc].tocoo()
        self._dan_dense = np.asarray(A[mbon][:, dan].todense())

    def _load_from_wiring(self, wiring, modality):
        z = np.load(wiring, allow_pickle=False)
        self.mbon_type = [str(x) for x in z["mbon_type"]]; self._dan_type = [str(x) for x in z["dan_type"]]
        pre = "vk" if modality == "visual" else "pk"
        self._Wpk = sparse.csr_matrix((z[f"{pre}_data"], z[f"{pre}_indices"], z[f"{pre}_indptr"]), shape=tuple(z[f"{pre}_shape"]))
        self._Wkm = sparse.coo_matrix((z["km_data"], (z["km_row"], z["km_col"])), shape=tuple(z["km_shape"]))
        self._dan_dense = z["dan_mbon"]
        labels = [str(x) for x in (z["vpn_type"] if modality == "visual" else z["opn_glom"])]
        ch = {}
        for j, name in enumerate(labels):
            ch.setdefault(name, []).append(j)
        self.channel_names = sorted(ch); self.channel = ch

    def _dan_template(self, prefixes) -> torch.Tensor:
        """Per-MBON reinforcement strength (0..1) from DAN -> MBON synapse counts of the DAN types matching
        `prefixes`, pooled over the hemispheric copies of each MBON type (so reconstruction asymmetry cannot give
        one copy of a compartment a different dose), then normalised to max 1."""
        cols = [k for k, tp in enumerate(self._dan_type) if any(tp.startswith(p) for p in prefixes)]
        d = np.zeros(len(self.mbon_type))
        if cols:
            raw = self._dan_dense[:, cols].sum(axis=1)
            for tp in set(self.mbon_type):
                idx = [i for i, x in enumerate(self.mbon_type) if x == tp]
                d[idx] = raw[idx].mean()
        return torch.as_tensor(d / d.max() if d.max() > 0 else d, dtype=torch.float32)

    def _permute_types(self, delta: torch.Tensor, rng) -> torch.Tensor:
        tps = sorted(set(self.mbon_type)); val = {tp: float(delta[self.mbon_type.index(tp)]) for tp in tps}
        perm = dict(zip(tps, [val[x] for x in np.array(tps)[rng.permutation(len(tps))]]))
        return torch.as_tensor([perm[tp] for tp in self.mbon_type], dtype=torch.float32)

    def type_mask(self, name: str) -> torch.Tensor:
        return torch.as_tensor([tp == name for tp in self.mbon_type])

    def compartment_mask(self, us: str = "punish", thr: float = 0.5) -> torch.Tensor:
        """MBONs whose type receives at least `thr` of the maximal DAN dose: the compartment(s) the chosen DANs
        really innervate (PPL101 -> MBON11 in the real wiring). Falls back to MBON11 if nothing qualifies."""
        d = {"punish": self.delta_punish, "reward": self.delta_reward}[us]
        m = d >= thr
        return m if bool(m.any()) else self.type_mask("MBON11")

    # -- circuit --------------------------------------------------------------------------------
    def odour(self, gloms, strength: float = 1.0) -> np.ndarray:
        v = np.zeros(self.W_pk.shape[1])
        for g in gloms:
            v[self.glom[self.glom_names[g]]] = strength
        return v

    def kc_code(self, pn_vec: np.ndarray) -> torch.Tensor:
        """Feedforward drive through the real PN->KC wiring, then APL-style k-winners-take-all: exactly the top
        `sparsity` fraction of driven Kenyon cells fire (ties broken in a fixed random order); graded rates are
        normalised to max 1, or read as fire / not fire with binary=True."""
        drive = np.asarray(self.W_pk @ pn_vec).ravel()
        k = max(1, round(self.sparsity * self.n_drivable_kc))                # top `sparsity` of the KCs this pathway can drive
        order = np.lexsort((self._tiebreak, -drive))                         # strongest drive first
        win = order[:k]; win = win[drive[win] > 0]
        code = np.zeros_like(drive); code[win] = drive[win]
        if self.binary:
            code = (code > 0).astype(np.float64)
        elif code.max() > 0:
            code = code / code.max()
        return torch.as_tensor(code, dtype=torch.float32)

    def _sum_to_mbon(self, per_edge: torch.Tensor) -> torch.Tensor:
        return torch.zeros(self.n_mbon).index_add_(0, self.e_mbon, per_edge)

    def mbon_response(self, kc: torch.Tensor) -> torch.Tensor:
        """Summed plastic synaptic drive from the active KCs onto each MBON (Hige's EPSC-charge analogue)."""
        return self._sum_to_mbon(self.e_base * self.W * kc[self.e_kc])

    # -- learning -------------------------------------------------------------------------------
    def reinforce(self, kc: torch.Tensor, us: str, strength: float = 1.0) -> None:
        delta = {"punish": self.delta_punish, "reward": self.delta_reward}[us][self.e_mbon] * strength
        k = kc[self.e_kc]
        dW = -self.lr * delta * (k + self.recover_rate * (self.W - W_REST))   # recover_rate 1 = the published rule
        self.W = torch.clamp(self.W + dW, 0.0, self.w_max)

    def reset(self) -> None:
        self.W = torch.ones(self.n_edges)

    # -- closed forms (exact for recover_rate 1, lr*delta <= 1) ---------------------------------
    def calibrate_lr(self, kc: torch.Tensor, mask: torch.Tensor, target: float = HIGE_CHARGE_DROP,
                     us: str = "punish", strength: float = 1.0) -> float:
        """Set lr so that ONE pairing of `kc` drops the (response-weighted) drive of the MBONs in `mask` by
        `target`. After one pairing from W = 1 the drop at MBON j is lr*delta_j*S2_j/S1_j with
        S1 = sum e*k, S2 = sum e*k^2 (S2/S1 = 1 for a binary code)."""
        delta = {"punish": self.delta_punish, "reward": self.delta_reward}[us] * strength
        k = kc[self.e_kc]; ek = self.e_base * k
        S1, S2 = self._sum_to_mbon(ek), self._sum_to_mbon(ek * k)
        denom = float((delta[mask] * S2[mask]).sum()); dmax = float(delta[mask].max()) if bool(mask.any()) else 0.0
        if denom == 0.0 or dmax == 0.0:
            raise ValueError("calibrate_lr: no reinforcement reaches the masked compartment (delta[mask] all zero); "
                             "check the punish/reward DAN selection and the mask")
        lr = target * float(S1[mask].sum()) / denom
        cap = 1.0 / dmax                                  # lr*delta > 1 makes the rule overshoot and oscillate: never exceed it
        self.lr_capped = lr > cap
        self.lr = min(lr, cap)
        return self.lr

    def analytic_drop(self, kc_train: torch.Tensor, kc_probe: torch.Tensor, mask: torch.Tensor, p: int,
                      us: str = "punish", strength: float = 1.0) -> float:
        """Published rule, p pairings of kc_train, then probe with kc_probe: drop over `mask` =
        sum_j S12_j (1 - (1 - lr delta_j)^p) / sum_j S1_j, S12 = sum e*k_probe*k_train."""
        delta = {"punish": self.delta_punish, "reward": self.delta_reward}[us] * strength
        kt, kp = kc_train[self.e_kc], kc_probe[self.e_kc]
        S1, S12 = self._sum_to_mbon(self.e_base * kp), self._sum_to_mbon(self.e_base * kp * kt)
        f = 1 - (1 - self.lr * delta).clamp(0.0, 1.0) ** p
        return float((S12[mask] * f[mask]).sum() / S1[mask].sum())

    def overlap(self, kc_train: torch.Tensor, kc_probe: torch.Tensor, mask: torch.Tensor) -> float:
        """Fraction of the probe odour's drive onto `mask` that passes through synapses of KCs active for the
        trained odour (the wiring quantity that sets cross-odour generalisation)."""
        kt, kp = kc_train[self.e_kc], kc_probe[self.e_kc]
        S1, S12 = self._sum_to_mbon(self.e_base * kp), self._sum_to_mbon(self.e_base * kp * kt)
        return float(S12[mask].sum() / S1[mask].sum())


def drop(before: torch.Tensor, after: torch.Tensor, mask: torch.Tensor) -> float:
    """Response-weighted fractional drop over the MBONs in `mask` (silent MBONs carry no weight)."""
    sb = float(before[mask].sum())
    return float(1 - after[mask].sum() / sb) if sb > 0 else float("nan")


def per_type(before: torch.Tensor, after: torch.Tensor, mb: MushroomBody, top: int = 8) -> dict:
    """Drop per MBON type plus each type's share of the total lost drive."""
    lost = (before - after); total = float(lost.sum())
    rows = []
    for tp in sorted(set(mb.mbon_type)):
        mk = mb.type_mask(tp); b = float(before[mk].sum())
        if b > 0:
            rows.append((tp, round(1 - float(after[mk].sum()) / b, 4), round(float(lost[mk].sum()) / total, 4) if total else None))
    rows.sort(key=lambda r: -(r[2] or 0))
    return {tp: {"drop": d, "share_of_total_depression": s} for tp, d, s in rows[:top]}


def run(a) -> dict:
    t0 = time.time()
    lr0 = 0.9 if a.lr == "auto" else float(a.lr)
    mb = MushroomBody(a.subgraph, a.neurons, a.sparsity, tuple(a.punish), tuple(a.reward), lr0, a.w_max, a.shuffle,
                      a.seed, a.binary_code, a.recover_rate, wiring=a.wiring, modality=a.modality)
    rng = np.random.default_rng(a.seed)
    odours = [rng.choice(len(mb.glom_names), size=a.glom_per_odour, replace=False) for _ in range(a.odours)]
    codes = [mb.kc_code(mb.odour(o)) for o in odours]
    X = torch.stack(codes); Xn = X / (X.norm(dim=1, keepdim=True) + 1e-9); C = (Xn @ Xn.T).numpy()
    off = C[~np.eye(len(odours), dtype=bool)]
    m11 = mb.type_mask("MBON11")
    comp, comp_r = mb.compartment_mask("punish"), mb.compartment_mask("reward")
    if a.lr == "auto":                                                       # one pairing = target drop in the punished compartment (MBON11 for real wiring)
        mb.calibrate_lr(codes[0], comp, a.target_drop, "punish", a.strength)
    P = a.pairings
    res = {"circuit": {"modality": a.modality, "n_input": mb.n_input, "n_channels": len(mb.glom_names), "n_KC": mb.n_kc, "n_MBON": mb.n_mbon,
                       "n_plastic_KC_MBON_edges": mb.n_edges, "n_punish_DAN": mb.n_punish_dan, "n_reward_DAN": mb.n_reward_dan,
                       "punish_compartment_types": sorted({mb.mbon_type[i] for i in np.flatnonzero(comp.numpy())}),
                       "reward_compartment_types": sorted({mb.mbon_type[i] for i in np.flatnonzero(comp_r.numpy())}),
                       "delta_punish_MBON11": round(float(mb.delta_punish[m11].mean()), 4),
                       "delta_reward_MBON11": round(float(mb.delta_reward[m11].mean()), 4)},
           "kc_code": {"sparsity": a.sparsity, "binary": a.binary_code,
                       "active_frac": [round(float((c > 0).float().mean()), 4) for c in codes[:4]],
                       "cross_odour_cos_mean": round(float(off.mean()), 4), "cross_odour_cos_max": round(float(off.max()), 4)},
           "rule": {"lr": round(mb.lr, 4), "lr_calibrated": a.lr == "auto", "target_drop_in_punish_compartment_p1": a.target_drop if a.lr == "auto" else None,
                    "lr_capped_at_stability_limit": bool(getattr(mb, "lr_capped", False)),
                    "recover_rate": a.recover_rate, "pairings": P, "strength": a.strength,
                    "note": "paired drop at p pairings = (S2/S1) * (1 - (1 - lr*delta)^p) by the rule; with lr calibrated it is "
                            "not a prediction. Wiring-dependent endpoints: compartment map, unpaired/generalisation, coexistence."},
           "shuffle": a.shuffle, "punish": list(a.punish), "reward": list(a.reward)}

    def resp():
        return torch.stack([mb.mbon_response(c) for c in codes])              # [O, n_mbon]

    before = resp()

    # --- one memory: odour A + punishment, p pairings. Measured next to the closed form.
    curve = {}
    mb.reset()
    for p in range(1, max(P, a.curve_max) + 1):
        mb.reinforce(codes[0], "punish", a.strength)
        after = resp()
        curve[p] = {"paired_A_MBON11": round(drop(before[0], after[0], m11), 4),
                    "closed_form_MBON11": round(mb.analytic_drop(codes[0], codes[0], m11, p, "punish", a.strength), 4),
                    "unpaired_MBON11_mean": round(float(np.mean([drop(before[i], after[i], m11) for i in range(1, len(codes))])), 4),
                    "unpaired_closed_form": round(float(np.mean([mb.analytic_drop(codes[0], codes[i], m11, p, "punish", a.strength) for i in range(1, len(codes))])), 4)}
        if p == P:
            one = after
    res["drop_vs_pairings_MBON11"] = curve
    unp = [drop(before[i], one[i], m11) for i in range(1, len(codes))]
    ov = [mb.overlap(codes[0], codes[i], m11) for i in range(1, len(codes))]
    res["one_memory"] = {"pairings": P,
                         "paired_A_MBON11": round(drop(before[0], one[0], m11), 4),
                         "paired_A_compartment": round(drop(before[0], one[0], comp), 4),
                         "unpaired_MBON11_mean": round(float(np.mean(unp)), 4), "unpaired_MBON11_each": [round(x, 4) for x in unp],
                         "unpaired_over_paired": round(float(np.mean(unp)) / max(drop(before[0], one[0], m11), 1e-9), 4),
                         "overlap_at_MBON11_mean": round(float(np.mean(ov)), 4),
                         "per_type_drop_A": per_type(before[0], one[0], mb),
                         "share_of_depression_in_MBON11": round(float((before[0] - one[0])[m11].sum() / (before[0] - one[0]).sum()), 4),
                         "W_stats": {"min": round(float(mb.W.min()), 4), "mean": round(float(mb.W.mean()), 4), "frac_depressed": round(float((mb.W < 0.99).float().mean()), 4)}}

    # --- generalisation: odours sharing s of A's glomeruli, after the same training
    gen = {}
    a_set = list(odours[0]); pool = [g for g in range(len(mb.glom_names)) if g not in a_set]
    for s in sorted({a.glom_per_odour - 1, a.glom_per_odour - 2, a.glom_per_odour // 2, 1, 0}, reverse=True):
        ds, cs, ovs = [], [], []
        for r in range(a.gen_reps):
            keep = list(rng.choice(a_set, size=s, replace=False)); new = list(rng.choice(pool, size=a.glom_per_odour - s, replace=False))
            code = mb.kc_code(mb.odour(keep + new))
            ds.append(drop(mb._sum_to_mbon(mb.e_base * code[mb.e_kc]), mb.mbon_response(code), m11))
            cs.append(float((code / (code.norm() + 1e-9)) @ (codes[0] / (codes[0].norm() + 1e-9)))); ovs.append(mb.overlap(codes[0], code, m11))
        gen[f"shared_{s}_of_{a.glom_per_odour}"] = {"drop_MBON11": round(float(np.mean(ds)), 4), "kc_cosine_to_A": round(float(np.mean(cs)), 4),
                                                    "overlap_at_MBON11": round(float(np.mean(ovs)), 4)}
    res["generalisation"] = gen

    # --- DAN with no KC activity: zero by the rule's algebra (k = 0, W = 1). Not a timing test.
    mb.reset()
    for _ in range(P):
        mb.reinforce(torch.zeros_like(codes[0]), "punish", a.strength)
    res["dan_alone_no_KC_drop_A"] = {"value": round(drop(before[0], mb.mbon_response(codes[0]), m11), 4), "note": "identically 0 by the rule; no time axis in this model"}

    # --- reward: odour C + PAM. Same rule, its own compartments (Owald 2015: reward = depression of approach-avoiding MBONs).
    mb.reset()
    for _ in range(P):
        mb.reinforce(codes[2], "reward", a.strength)
    rew = resp()
    res["reward_memory"] = {"paired_C_reward_compartments": round(drop(before[2], rew[2], comp_r), 4),
                            "paired_C_MBON11": round(drop(before[2], rew[2], m11), 4),
                            "per_type_drop_C": per_type(before[2], rew[2], mb),
                            "unpaired_reward_compartments_mean": round(float(np.mean([drop(before[i], rew[i], comp_r) for i in range(len(codes)) if i != 2])), 4)}

    # --- coexistence. Same compartment: A then B (blocked) and A/B interleaved. Different compartments: A punished, C rewarded.
    def block(seq):
        mb.reset()
        for idx, us in seq:
            for _ in range(P):
                mb.reinforce(codes[idx], us, a.strength)
        return resp()
    mb.reset()
    for _ in range(P):
        mb.reinforce(codes[0], "punish", a.strength)
    a_alone = drop(before[0], mb.mbon_response(codes[0]), m11)
    ab = block([(0, "punish"), (1, "punish")])
    a_after_b = drop(before[0], ab[0], m11)
    mb.reset()
    for _ in range(P):
        mb.reinforce(codes[0], "punish", a.strength); mb.reinforce(codes[1], "punish", a.strength)
    abi = resp()
    ac = block([(0, "punish"), (2, "reward")])
    d11 = float(mb.delta_punish[m11].mean()) * a.strength
    res["coexistence"] = {
        "same_compartment_blocked_A_then_B": {"A_alone": round(a_alone, 4), "A_after_B": round(a_after_b, 4), "B": round(drop(before[1], ab[1], m11), 4),
                                              "A_retained_fraction": round(a_after_b / max(a_alone, 1e-9), 4),
                                              "closed_form_retained_nonshared_KCs": round(float(max(1 - mb.lr * d11 * mb.recover_rate, 0.0) ** P), 4)},
        "same_compartment_interleaved": {"A": round(drop(before[0], abi[0], m11), 4), "B": round(drop(before[1], abi[1], m11), 4)},
        "different_compartments_A_punish_C_reward": {"A_MBON11": round(drop(before[0], ac[0], m11), 4), "C_reward_compartments": round(drop(before[2], ac[2], comp_r), 4),
                                                     "C_MBON11": round(drop(before[2], ac[2], m11), 4), "A_reward_compartments": round(drop(before[0], ac[0], comp_r), 4),
                                                     "D_untouched_MBON11": round(drop(before[3], ac[3], m11), 4)},
        "note": "the rule's silent-KC term relaxes depressed synapses toward 1 whenever dopamine arrives, so a second memory in the same "
                "compartment erodes the first by (1 - lr*delta*recover_rate)^p on non-shared KCs; recover_rate 1 is the published rule"}
    res["elapsed_s"] = round(time.time() - t0, 1)
    res["ground_truth"] = "Hige 2015: one pairing block ~80% (spikes) / ~90% (charge) at MBON-gamma1pedc; unpaired odour unchanged. Owald 2015: reward = depression in PAM compartments."
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--wiring", type=Path, default=None, help="full-connectome MB cache from mb_build.py; overrides --subgraph")
    ap.add_argument("--modality", default="olfactory", choices=["olfactory", "visual"])
    ap.add_argument("--sparsity", type=float, default=0.05); ap.add_argument("--odours", type=int, default=8)
    ap.add_argument("--glom-per-odour", type=int, default=6); ap.add_argument("--pairings", type=int, default=1)
    ap.add_argument("--curve-max", type=int, default=8); ap.add_argument("--gen-reps", type=int, default=20)
    ap.add_argument("--lr", default="auto", help="'auto' = calibrate once so one pairing gives --target-drop at MBON11; or a number")
    ap.add_argument("--target-drop", type=float, default=HIGE_CHARGE_DROP); ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--w-max", type=float, default=2.0); ap.add_argument("--recover-rate", type=float, default=1.0)
    ap.add_argument("--punish", nargs="+", default=["PPL101"]); ap.add_argument("--reward", nargs="+", default=["PAM"])
    ap.add_argument("--shuffle", default=None, choices=[None, "pn_kc", "kc_mbon", "dan_mbon"])
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--binary-code", action="store_true", help="KC fires or not (spike-like), instead of graded rates")
    a = ap.parse_args(argv)
    res = run(a)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
