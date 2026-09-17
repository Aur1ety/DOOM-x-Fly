"""Mushroom-body plasticity for the frozen MaleCNS core: dopamine-gated change of KC -> MBON synapses only.

Everything in the connectome stays frozen except the Kenyon-cell -> MBON output synapses (the ones the
literature says learn). A reinforcement signal delta_j is delivered to each MBON j by firing the real
dopaminergic neurons that innervate its compartment (PPL1 = punishment, PAM = reward), read straight from
the connectome's DAN -> MBON wiring, not assigned by hand. The weight change follows the published
dopaminergic plasticity rule (Gkanias, McCurdy, Nitabach & Webb, eLife 2022, "An incentive circuit for
memory dynamics in the mushroom body"), verified against that paper:

    dW_ij = delta_j * ( k_i + W_ij - w_rest ),   w_rest = 1, W >= 0

  regimes (w_rest = 1):
    delta < 0, KC active   -> depression      (aversive pairing weakens KC->MBON)
    delta > 0, KC active   -> potentiation   (not used: reward is depression in PAM compartments, Owald 2015)
    delta < 0, KC silent   -> recovery toward w_rest
    delta > 0, KC silent   -> saturation away from w_rest

The effective synapse fed to the core is base_value * W (via ConnectomeCore.set_edge_gain), so W = 1 is the
untouched frozen wiring. Ground-truth targets we score this against (Hige et al. 2015, Neuron): a paired
stimulus's MBON response drops ~80 % (spikes) / ~90 % (charge); the unpaired stimulus is unchanged; forward
the change persists tens of minutes. Dopamine with silent KCs changes nothing by the rule's algebra.

    python -m flybrain.model.mb_plasticity --subgraph $FLYBRAIN_OUT/graph/subgraph_v5.npz --smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flybrain import OUT_DIR

PUNISH_PREFIX = "PPL1"          # PPL1 cluster = aversive/punishment dopaminergic neurons
REWARD_PREFIX = "PAM"           # PAM cluster = appetitive/reward dopaminergic neurons
KC_PREFIX = "KC"
MBON_PREFIX = "MBON"
W_REST = 1.0


def node_types(sub: dict, neurons: Path) -> np.ndarray:
    """type string per subgraph node, in node order (empty string where unknown)."""
    import pandas as pd
    n = pd.read_parquet(neurons, columns=["bodyId", "type"])
    t = dict(zip(n["bodyId"].to_numpy().tolist(), n["type"].fillna("").astype(str).to_numpy().tolist()))
    return np.array([t.get(int(b), "") for b in np.asarray(sub["bodyId"])])


class MushroomBodyPlasticity:
    """Holds the plastic KC->MBON weights and the DAN-driven reinforcement, and drives a frozen core.

    core       : a ConnectomeCore (any param; its edge_values are scaled by our per-edge gain)
    types      : type string per node (from node_types)
    lr         : learning-rate scale on the published rule (folds in the per-trial dopamine magnitude)
    w_max      : clamp on the plastic weight (potentiation ceiling)
    forget_tau : passive recovery of W toward w_rest per decision when no dopamine (fitted, not measured)
    """

    def __init__(self, core, types: np.ndarray, lr: float = 0.2, w_max: float = 2.0,
                 forget_tau: float | None = None, device: str = "cpu",
                 punish_types=(PUNISH_PREFIX,), reward_types=(REWARD_PREFIX,)):
        self.core, self.lr, self.w_max, self.forget_tau, self.device = core, lr, w_max, forget_tau, device
        self.types = types.astype(str)
        pre = core.pre_of_edge.cpu().numpy(); post = core.post_of_edge.cpu().numpy()

        def match(specs):
            """union of nodes whose type starts with any of the given prefixes / exact names."""
            m = np.zeros(len(self.types), bool)
            for s in specs:
                m |= np.char.startswith(self.types, s)
            return m
        is_kc, is_mbon = match((KC_PREFIX,)), match((MBON_PREFIX,))
        self.kc_idx = np.flatnonzero(is_kc); self.mbon_idx = np.flatnonzero(is_mbon)
        self.mbon_idx_t = torch.as_tensor(self.mbon_idx, device=device, dtype=torch.long)
        self.punish_idx = np.flatnonzero(match(punish_types)); self.reward_idx = np.flatnonzero(match(reward_types))

        # plastic edges = KC (pre) -> MBON (post)
        plastic = is_kc[pre] & is_mbon[post]
        self.plastic_edge = torch.as_tensor(np.flatnonzero(plastic), device=device, dtype=torch.long)
        self.pre_of_plastic = torch.as_tensor(pre[plastic], device=device, dtype=torch.long)     # KC node per plastic edge
        self.post_of_plastic = torch.as_tensor(post[plastic], device=device, dtype=torch.long)   # MBON node per plastic edge
        self.n_plastic = int(self.plastic_edge.numel())

        # per-MBON reinforcement template from the real DAN -> MBON wiring, normalised to max drive 1
        base = core.edge_values().detach()
        self.delta_punish = self._dan_template(base.cpu().numpy(), pre, post, self.punish_idx)   # >=0 magnitude per node
        self.delta_reward = self._dan_template(base.cpu().numpy(), pre, post, self.reward_idx)

        self.n_edges = int(core.n_edges)
        self._gain = torch.ones(self.n_edges, device=device)                                     # persistent; only plastic slots change
        self.W = torch.ones(self.n_plastic, device=device)

    def _dan_template(self, values: np.ndarray, pre: np.ndarray, post: np.ndarray, dan_nodes: np.ndarray) -> torch.Tensor:
        """Per-node vector: how strongly each MBON is innervated by the given DAN population (0..1)."""
        drive = np.zeros(self.core.n_nodes)
        if len(dan_nodes):
            is_dan = np.zeros(self.core.n_nodes, bool); is_dan[dan_nodes] = True
            sel = is_dan[pre]
            np.add.at(drive, post[sel], np.abs(values[sel]))
        mb = drive[self.mbon_idx]
        if mb.max() > 0:
            drive[self.mbon_idx] = mb / mb.max()
        return torch.as_tensor(drive, device=self.device, dtype=torch.float32)

    # -- API -------------------------------------------------------------------------------------
    def reset(self) -> None:
        self.W = torch.ones(self.n_plastic, device=self.device)
        self.apply_to_core()

    def apply_to_core(self) -> None:
        """Push the current plastic weights into the core, updating only the plastic slots in place."""
        self._gain[self.plastic_edge] = self.W
        self.core.set_edge_gain(self._gain)

    @torch.no_grad()
    def reinforce(self, rates: torch.Tensor, us: str = "punish", strength: float = 1.0) -> dict:
        """One pairing update. `rates` = per-node rates [M] (or [M,B] -> uses mean over batch) from the
        stimulus that is being paired. `us` selects the dopamine population; `strength` scales delta.

        dW_e = lr * strength * delta_{post(e)} * ( k_{pre(e)} + W_e - w_rest ), then clamp W to [0, w_max].
        """
        if rates.dim() == 2:
            rates = rates.mean(1)
        rates = rates.to(self.device)
        template = {"punish": -self.delta_punish, "reward": -self.delta_reward, "none": torch.zeros_like(self.delta_punish)}[us]   # both depress, own compartments (Owald 2015)
        delta_e = template[self.post_of_plastic] * strength                 # signed reinforcement per plastic edge
        k_e = rates[self.pre_of_plastic]                                     # presynaptic KC rate per plastic edge
        dW = self.lr * delta_e * (k_e + self.W - W_REST)
        self.W = torch.clamp(self.W + dW, 0.0, self.w_max)
        self.apply_to_core()
        moved = (dW.abs() > 1e-6)
        return {"n_plastic": self.n_plastic, "n_moved": int(moved.sum()), "mean_dW": float(dW.mean()),
                "W_min": float(self.W.min()), "W_mean": float(self.W.mean()), "W_max": float(self.W.max())}

    @torch.no_grad()
    def relax(self, decisions: int = 1) -> None:
        """Passive recovery of W toward w_rest between trials (forgetting); no-op if forget_tau is None."""
        if self.forget_tau:
            a = 1.0 - np.exp(-decisions / self.forget_tau)
            self.W = self.W + a * (W_REST - self.W)
            self.apply_to_core()

    @torch.no_grad()
    def present(self, clamped: torch.Tensor, decisions: int = 20) -> torch.Tensor:
        """Run the core from rest with a fixed stimulus (clamped optic-lobe rates) and return the settled
        per-node rates [M]. This is the stimulus-response measurement the eval uses: read MBON *rate* to a
        specific pattern, before vs after pairing (Hige et al. measured spike rate, so we must too, not a
        weight proxy). `clamped` = [n_clamped] or [n_clamped, 1]."""
        from flybrain.model.core import (
            CoreState,  # noqa: F401  (kept for clarity that we drive CoreState)
        )
        c = torch.as_tensor(clamped, dtype=self.core.dtype, device=self.device).reshape(-1)
        if c.numel() != self.core.n_clamped:
            raise ValueError(f"clamped must be [{self.core.n_clamped}], got {tuple(c.shape)}")
        cb = c[None, :]                                        # [B=1, n_clamped], held across substeps
        state = self.core.init_state(1)
        for _ in range(decisions):
            state = self.core.step(state, cb)
        return self.core.rates(state)[:, 0]                    # [M]

    def mbon_rates(self, rates: torch.Tensor) -> torch.Tensor:
        """MBON rates [n_MBON] from a full per-node rate vector [M]."""
        return rates.index_select(0, self.mbon_idx_t)

    def mbon_weight_totals(self) -> torch.Tensor:
        """MONITOR ONLY (not the eval metric): summed plastic KC->MBON weight onto each MBON node. This is
        stimulus-independent and so washes out the stimulus-specificity of depression; use present() +
        mbon_rates() for anything scored against the ground-truth 80 % drop."""
        out = torch.zeros(self.core.n_nodes, device=self.device)
        out.index_add_(0, self.post_of_plastic, self.W)
        return out[self.mbon_idx]

    def summary(self) -> dict:
        return {"n_plastic_edges": self.n_plastic, "n_KC": len(self.kc_idx), "n_MBON": len(self.mbon_idx),
                "n_punish_DAN": len(self.punish_idx), "n_reward_DAN": len(self.reward_idx),
                "mbon_with_punish_drive": int((self.delta_punish[self.mbon_idx] > 0).sum()),
                "mbon_with_reward_drive": int((self.delta_reward[self.mbon_idx] > 0).sum()), "lr": self.lr, "w_max": self.w_max}


def load_types(sub: dict, neurons: Path) -> np.ndarray:
    """convenience alias so callers can import one function."""
    return node_types(sub, neurons)


# ------------------------------------------------------------------------------------------ smoke test

def smoke(subgraph: Path, neurons: Path, device: str) -> dict:
    """No training: build the module, verify the plastic-edge count and that a fake pairing depresses only
    the co-active KC->MBON edges (paired stimulus) and leaves everything else at w_rest."""
    from flybrain.model.core import ConnectomeCore, CoreConfig, load_subgraph

    sub = load_subgraph(subgraph)
    types = node_types(sub, neurons)
    core = ConnectomeCore(sub, CoreConfig(param="type_tied"), backend="spmm", device=device)
    mb = MushroomBodyPlasticity(core, types, lr=0.3, device=device)
    out = {"summary": mb.summary()}

    # a fake "stimulus": half the KCs active (a sparse code), the rest silent
    rng = np.random.default_rng(0)
    rates = torch.zeros(core.n_nodes, device=device)
    active_kc = mb.kc_idx[rng.random(len(mb.kc_idx)) < 0.5]
    rates[active_kc] = 1.0
    active_set = set(active_kc.tolist())

    W0 = mb.W.clone()
    step = mb.reinforce(rates, us="punish", strength=1.0)
    out["pairing"] = step

    # checks: edges that moved must be (KC active) AND (MBON has punishment drive); W must stay >= 0
    pre = mb.pre_of_plastic.cpu().numpy(); post = mb.post_of_plastic.cpu().numpy()
    moved = (mb.W - W0).abs().cpu().numpy() > 1e-6
    kc_active_edge = np.array([p in active_set for p in pre])
    mbon_has_da = (mb.delta_punish[post].cpu().numpy() > 0)
    should_move = kc_active_edge & mbon_has_da
    out["checks"] = {
        "plastic_edges": mb.n_plastic,
        "expected_plastic_edges_note": "compare to the offline count (~33,496 in v5)",
        "moved_only_where_expected": bool(np.array_equal(moved, should_move)),
        "all_moved_are_depression": bool(((mb.W - W0).cpu().numpy()[moved] < 0).all()) if moved.any() else True,
        "W_nonneg": bool((mb.W >= 0).all()),
        "core_gain_set": core.edge_gain is not None,
        "frozen_edges_unit_gain": bool(torch.allclose(core.edge_gain[torch.isin(torch.arange(core.n_edges, device=device), mb.plastic_edge, invert=True)], torch.ones(1, device=device))),
    }
    # recovery: silent-KC + punishment should push a depressed weight back toward 1
    mb2 = MushroomBodyPlasticity(core, types, lr=0.3, device=device)
    mb2.W = torch.full((mb2.n_plastic,), 0.4, device=device)
    silent = torch.zeros(core.n_nodes, device=device)
    rec = mb2.reinforce(silent, us="punish", strength=1.0)
    out["recovery"] = {"W_mean_before": 0.4, "W_mean_after": rec["W_mean"], "recovered_upward": rec["W_mean"] > 0.4}

    # measurement primitive: present a random static stimulus and read MBON rates through the dynamics
    mb3 = MushroomBodyPlasticity(core, types, lr=0.3, device=device)
    stim = torch.as_tensor(rng.random(core.n_clamped).astype(np.float32), device=device)
    r = mb3.present(stim, decisions=15)
    mb_r = mb3.mbon_rates(r)
    out["present"] = {"rates_shape": list(r.shape), "n_mbon_read": int(mb_r.numel()),
                      "finite": bool(torch.isfinite(mb_r).all()), "mbon_rate_mean": float(mb_r.mean())}

    # the specific published pair, selectable: PPL1-gamma1pedc = PPL101 -> MBON11
    mb4 = MushroomBodyPlasticity(core, types, lr=0.3, device=device, punish_types=("PPL101",))
    out["specific_pair"] = {"n_punish_DAN": len(mb4.punish_idx),
                            "mbon_with_punish_drive": int((mb4.delta_punish[mb4.mbon_idx] > 0).sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subgraph", type=Path, default=OUT_DIR / "graph" / "subgraph_v5.npz")
    ap.add_argument("--neurons", type=Path, default=OUT_DIR / "graph" / "neurons.parquet")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true", help="run the build + fake-pairing self-check and exit")
    a = ap.parse_args(argv)
    if a.smoke:
        print(json.dumps(smoke(a.subgraph, a.neurons, a.device), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
