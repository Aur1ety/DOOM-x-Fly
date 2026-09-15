"""Rate RNN over a MaleCNS subgraph (PLAN.md section 3, "Neuron model" / "Trainable parameters").

    h <- h + (dt / tau_i) * ( -(h - v_rest_i) + (W r)_i )
    r  = min(softplus(h), r_max)
    W_ij = sign[pre_j] * magnitude_ij          CSR rows = POST, cols = PRE, y = W @ r

Parametrisations
    per_edge   magnitude = softplus(theta[E]);  theta init = softplus^-1(w0 * log1p(count))
    type_tied  magnitude = w0 * log1p(count) * exp(log_alpha[type_pre] + log_beta[type_post]
                                                    + pair_scale[pair])   (pair_scale only for
               the top-P (pre_type, post_type) pairs covering `pair_cover` of synaptic weight)
    frozen     magnitude fixed at the per_edge init (readout-only arm)

Clamped nodes (front-end driven) have their rates overwritten every substep; dynamic nodes
evolve. K substeps per decision, dt = decision_ms / K. State is a `CoreState` named tuple.
The sparse product goes through flybrain.model.spmm when importable, otherwise through a
chunked torch reference with the same API (`backend="reference"`).

Inhibitory centring at init (Cornford et al. 2021 style): inhibitory magnitudes are rescaled
by a fixed, non-trainable factor so that at a uniform presynaptic rate the summed inhibitory
input equals the summed excitatory input -- globally (`centre="global"`, default) or per
postsynaptic neuron with the factor clipped to [1/centre_clip, centre_clip] (`centre="post"`).
The factor is folded into the init (per_edge / frozen) or into the fixed edge_scale
(type_tied) and is identical in recipe for real and null graphs.

CLI:  python -m flybrain.model.core --synthetic --param type_tied
      python -m flybrain.model.core --subgraph $FLYBRAIN_DATA/subgraph_core.npz --param per_edge
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:  # the kernel is built in parallel; fall back to the reference path if it is absent
    from flybrain.model import spmm as _kernel
except Exception:  # pragma: no cover - depends on what is synced
    _kernel = None

SUBGRAPH_KEYS = (
    "is_clamped", "is_output", "type_id", "sign", "indptr_post", "indices_pre", "weight",
    "indptr_pre", "indices_post", "perm_csr_to_csc",
)
REF_CHUNK = 1_000_000  # edges per chunk in the reference spmm (never an [E, B] tensor beyond this)


# ----------------------------------------------------------------------------- subgraph I/O


def load_subgraph(path) -> dict:
    """Load a subgraph_*.npz / null_*.npz into a dict of numpy arrays (CONTRACTS.md keys)."""
    with np.load(path, allow_pickle=False) as z:
        sub = {k: z[k] for k in z.files}
    missing = [k for k in SUBGRAPH_KEYS if k not in sub]
    if missing:
        raise KeyError(f"{path}: missing subgraph keys {missing}")
    return sub


def build_csc(indptr_post: np.ndarray, indices_pre: np.ndarray):
    """CSC (indptr_pre, indices_post, perm_csr_to_csc) from a CSR indexed by post."""
    n = len(indptr_post) - 1
    post = np.repeat(np.arange(n, dtype=np.int64), np.diff(indptr_post))
    pre = indices_pre.astype(np.int64)
    perm = np.lexsort((post, pre))  # sort by pre, then post
    indptr_pre = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr_pre[1:])
    return indptr_pre, post[perm].astype(np.int32), perm.astype(np.int64)


def synthetic_subgraph(n_nodes: int = 80, n_clamped: int = 16, n_output: int = 8, n_types: int = 10,
                       mean_in_degree: float = 6.0, frac_untyped: float = 0.05, seed: int = 0) -> dict:
    """Random subgraph-format fixture: heavy-tailed synapse counts, type-consistent signs,
    no edges into clamped nodes, every dynamic node has >= 1 input."""
    rng = np.random.default_rng(seed)
    m = n_nodes
    is_clamped = np.zeros(m, dtype=bool)
    is_clamped[:n_clamped] = True
    is_output = np.zeros(m, dtype=bool)
    is_output[m - n_output:] = True
    type_id = rng.integers(0, n_types, m).astype(np.int32)
    type_id[rng.random(m) < frac_untyped] = -1
    type_sign = rng.choice(np.array([1, -1], dtype=np.int8), size=n_types, p=[0.7, 0.3])
    sign = np.where(type_id >= 0, type_sign[np.clip(type_id, 0, None)], 1).astype(np.int8)
    posts, pres = [], []
    for j in np.flatnonzero(~is_clamped):
        k = min(m, 1 + rng.poisson(mean_in_degree))
        src = rng.choice(m, size=k, replace=False)
        posts.append(np.full(k, j)); pres.append(src)
    post = np.concatenate(posts); pre = np.concatenate(pres)
    order = np.lexsort((pre, post))
    post, pre = post[order], pre[order]
    n = m
    indptr_post = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post, minlength=n), out=indptr_post[1:])
    weight = (1 + rng.geometric(0.15, size=len(pre))).astype(np.float32)
    indptr_pre, indices_post, perm = build_csc(indptr_post, pre.astype(np.int32))
    return {
        "node_idx": np.arange(m, dtype=np.int64), "bodyId": 1000 + np.arange(m, dtype=np.int64),
        "is_clamped": is_clamped, "is_output": is_output, "is_dynamic": ~is_clamped,
        "type_id": type_id, "sign": sign,
        "indptr_post": indptr_post, "indices_pre": pre.astype(np.int32), "weight": weight,
        "indptr_pre": indptr_pre, "indices_post": indices_post, "perm_csr_to_csc": perm,
        "w_min": np.float64(1.0), "report": np.str_(json.dumps({"synthetic": True, "seed": seed})),
    }


# ----------------------------------------------------------------------------- reference kernel


class RefGraph:
    """Reference twin of flybrain.model.spmm.SparseGraph (same fields, torch tensors)."""

    def __init__(self, indptr_post, indices_pre, indptr_pre, indices_post, perm_csr_to_csc, device="cpu"):
        dev = torch.device(device)
        self.indptr_post = torch.as_tensor(np.asarray(indptr_post, dtype=np.int64), device=dev)
        self.indices_pre = torch.as_tensor(np.asarray(indices_pre, dtype=np.int32), device=dev)
        self.indptr_pre = torch.as_tensor(np.asarray(indptr_pre, dtype=np.int64), device=dev)
        self.indices_post = torch.as_tensor(np.asarray(indices_post, dtype=np.int32), device=dev)
        self.perm_csr_to_csc = torch.as_tensor(np.asarray(perm_csr_to_csc, dtype=np.int64), device=dev)
        self.n = int(self.indptr_post.numel() - 1)
        self.n_edges = int(self.indices_pre.numel())
        counts = torch.diff(self.indptr_post)
        self.row_of_edge = torch.repeat_interleave(torch.arange(self.n, device=dev), counts).to(torch.int32)
        self.device = dev

    @classmethod
    def from_arrays(cls, indptr_post, indices_pre, indptr_pre, indices_post, perm_csr_to_csc, device="cpu"):
        return cls(indptr_post, indices_pre, indptr_pre, indices_post, perm_csr_to_csc, device)

    @classmethod
    def from_npz(cls, path, device="cpu"):
        s = load_subgraph(path)
        return cls(s["indptr_post"], s["indices_pre"], s["indptr_pre"], s["indices_post"],
                   s["perm_csr_to_csc"], device)

    def to(self, device):
        return RefGraph(self.indptr_post, self.indices_pre, self.indptr_pre, self.indices_post,
                        self.perm_csr_to_csc, device)


def ref_spmm(values: Tensor, x: Tensor, g) -> Tensor:
    """y[post] = sum_e values[e] * x[pre_e]; chunked over edges; plain torch autograd."""
    y = x.new_zeros(g.n, x.shape[1])
    for s in range(0, g.n_edges, REF_CHUNK):
        e = slice(s, min(s + REF_CHUNK, g.n_edges))
        contrib = values[e, None] * x[g.indices_pre[e].long()]
        y = y.index_add(0, g.row_of_edge[e].long(), contrib)
    return y


def ref_spmm_binned(bin_values: Tensor, bin_index: Tensor, x: Tensor, g, edge_scale: Optional[Tensor] = None) -> Tensor:
    values = bin_values[bin_index]
    if edge_scale is not None:
        values = values * edge_scale
    return ref_spmm(values, x, g)


def available_backends() -> list[str]:
    return ["reference"] + (["spmm"] if _kernel is not None else [])


def _resolve_backend(name: str):
    if name == "auto":
        name = "spmm" if _kernel is not None else "reference"
    if name == "spmm":
        if _kernel is None:
            raise ImportError("flybrain.model.spmm not importable; use backend='reference'")
        return name, _kernel.SparseGraph, _kernel.spmm, _kernel.spmm_binned
    if name == "reference":
        return name, RefGraph, ref_spmm, ref_spmm_binned
    raise ValueError(f"unknown backend {name!r}")


# ----------------------------------------------------------------------------- model


def inv_softplus(y: Tensor) -> Tensor:
    """Inverse of softplus for y > 0 (stable form)."""
    return y + torch.log(-torch.expm1(-y))


@dataclass
class CoreConfig:
    param: str = "type_tied"        # per_edge | type_tied | frozen
    K: int = 6                      # substeps per decision
    decision_ms: float = 114.0      # frameskip 4 at 35 Hz
    r_max: float = 5.0
    tau_ms: float = 20.0            # membrane time constant at init
    tau_max_ms: float = 500.0       # tau clamped to [dt, tau_max]
    w0: float = 1.0                 # global magnitude scale; set per graph by gain.match_gain
    weight_transform: str = "log1p"  # log1p | linear   (base magnitude from synapse count)
    centre: str = "global"          # none | global | post   (inhibitory centring at init)
    centre_clip: float = 10.0
    pair_cover: float = 0.8         # type pairs covering this fraction of weight get pair_scale
    pair_cap: int = 8000
    bias: bool = False              # extra tonic current parameter (degenerate with v_rest)

    @property
    def dt_ms(self) -> float:
        return self.decision_ms / self.K


class CoreState(NamedTuple):
    h: Tensor            # [M, B] membrane state (clamped rows unused)
    x_clamped: Tensor    # [n_clamped, B] last clamped rates seen
    r_out_prev: Tensor   # [n_output, B] output rates at the previous decision


class ConnectomeCore(nn.Module):
    """Sign-constrained sparse rate RNN over a subgraph. State layout is [M, B]."""

    def __init__(self, sub: dict, cfg: Optional[CoreConfig] = None, backend: str = "auto",
                 device="cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.cfg = cfg = cfg or CoreConfig()
        if cfg.param not in ("per_edge", "type_tied", "frozen"):
            raise ValueError(f"unknown param {cfg.param!r}")
        self.backend, GraphCls, self._spmm, self._spmm_binned = _resolve_backend(backend)
        self.dtype = dtype
        dev = torch.device(device)
        self.graph = GraphCls.from_arrays(sub["indptr_post"], sub["indices_pre"], sub["indptr_pre"],
                                          sub["indices_post"], sub["perm_csr_to_csc"], dev)
        m, e = self.graph.n, self.graph.n_edges
        self.n_nodes, self.n_edges = m, e

        is_clamped = np.asarray(sub["is_clamped"], dtype=bool)
        is_output = np.asarray(sub["is_output"], dtype=bool)
        indptr = np.asarray(sub["indptr_post"], dtype=np.int64)
        if np.any(np.diff(indptr)[is_clamped] != 0):
            raise ValueError("subgraph stores edges INTO clamped nodes; CONTRACTS.md forbids that")
        if np.any(is_output & is_clamped):
            raise ValueError("output nodes must be dynamic")
        post = np.repeat(np.arange(m), np.diff(indptr))
        pre = np.asarray(sub["indices_pre"], dtype=np.int64)
        sign = np.asarray(sub["sign"]).astype(np.float64)
        if not np.all(np.isin(sign, (1.0, -1.0))):
            raise ValueError("sign must be +1/-1")
        count = np.asarray(sub["weight"], dtype=np.float64)
        if count.min() <= 0:
            raise ValueError("synapse counts must be > 0")
        # dense type remap; all null types (-1) share one bucket
        raw_types = np.asarray(sub["type_id"], dtype=np.int64)
        uniq, type_local = np.unique(raw_types, return_inverse=True)
        self.n_types = int(len(uniq))
        self.n_untyped_bucket = int(np.sum(uniq < 0))

        lt = torch.long
        self.register_buffer("clamped_idx", torch.as_tensor(np.flatnonzero(is_clamped), dtype=lt))
        self.register_buffer("dyn_idx", torch.as_tensor(np.flatnonzero(~is_clamped), dtype=lt))
        self.register_buffer("out_idx", torch.as_tensor(np.flatnonzero(is_output), dtype=lt))
        self.register_buffer("type_id", torch.as_tensor(type_local, dtype=lt))
        self.register_buffer("sign", torch.as_tensor(sign, dtype=dtype))
        self.register_buffer("count", torch.as_tensor(count, dtype=dtype))
        self.register_buffer("pre_of_edge", torch.as_tensor(pre, dtype=lt))
        self.register_buffer("post_of_edge", torch.as_tensor(post, dtype=lt))
        self.register_buffer("edge_sign", torch.as_tensor(sign[pre], dtype=dtype))
        self.n_clamped, self.n_dynamic, self.n_output = len(self.clamped_idx), len(self.dyn_idx), len(self.out_idx)

        # base magnitude at w0 = 1 (weight transform x inhibitory centring), fixed
        wt = np.log1p(count) if cfg.weight_transform == "log1p" else count.copy()
        if cfg.weight_transform not in ("log1p", "linear"):
            raise ValueError(f"unknown weight_transform {cfg.weight_transform!r}")
        self.centre_factor_stats = self._centre(wt, sign[pre], post, m)
        self.register_buffer("base_unit", torch.as_tensor(wt * self._cfac, dtype=dtype))

        # membrane parameters: per neuron (per_edge / frozen) or per type (type_tied)
        dt = cfg.dt_ms
        if not (0 < cfg.tau_ms and dt <= cfg.tau_max_ms):
            raise ValueError("need tau_ms > 0 and dt <= tau_max_ms")
        if cfg.param == "type_tied":
            n_units = self.n_types
            self.register_buffer("unit_of_dyn", self.type_id[self.dyn_idx].clone())
        else:
            n_units = self.n_dynamic
            self.register_buffer("unit_of_dyn", torch.arange(self.n_dynamic, dtype=lt))
        # tau = clamp(softplus(tau_raw), dt, tau_max); a tau_ms below dt is clamped to dt (coef = 1)
        tau_raw = torch.full((n_units,), float(inv_softplus(torch.tensor(cfg.tau_ms, dtype=torch.float64))), dtype=dtype)
        v_rest = torch.zeros(n_units, dtype=dtype)
        bias = torch.zeros(n_units, dtype=dtype)
        trainable = cfg.param != "frozen"
        self.tau_raw = nn.Parameter(tau_raw, requires_grad=trainable)
        self.v_rest = nn.Parameter(v_rest, requires_grad=trainable)
        self.bias = nn.Parameter(bias, requires_grad=trainable) if cfg.bias else None

        # synaptic parameters
        if cfg.param == "type_tied":
            self._build_pairs(type_local, pre, post, count)
            self.log_alpha = nn.Parameter(torch.zeros(self.n_types, dtype=dtype))
            self.log_beta = nn.Parameter(torch.zeros(self.n_types, dtype=dtype))
            self.pair_scale = nn.Parameter(torch.zeros(self.n_pairs_scaled, dtype=dtype))
            self.register_buffer("edge_scale", torch.zeros(e, dtype=dtype))
        elif cfg.param == "per_edge":
            self.theta = nn.Parameter(torch.zeros(e, dtype=dtype))
            self.register_buffer("theta0", torch.zeros(e, dtype=dtype))
        else:
            self.register_buffer("values_frozen", torch.zeros(e, dtype=dtype))
        self.set_w0(cfg.w0)
        self.to(dev)

    # ---- construction helpers

    def _centre(self, wt: np.ndarray, edge_sign: np.ndarray, post: np.ndarray, m: int) -> dict:
        """Cornford-2021-style balance: rescale inhibitory magnitudes so summed I input = summed E input."""
        cfg = self.cfg
        inh = edge_sign < 0
        fac = np.ones_like(wt)
        stats = {"mode": cfg.centre, "n_inh_edges": int(inh.sum()), "n_exc_edges": int((~inh).sum())}
        if cfg.centre == "none" or not inh.any() or inh.all():
            stats["factor"] = 1.0
        elif cfg.centre == "global":
            c = float(wt[~inh].sum() / wt[inh].sum())
            fac[inh] = c
            stats["factor"] = c
        elif cfg.centre == "post":
            exc_in = np.bincount(post, weights=wt * (~inh), minlength=m)
            inh_in = np.bincount(post, weights=wt * inh, minlength=m)
            ratio = np.where(inh_in > 0, exc_in / np.maximum(inh_in, 1e-12), 1.0)
            ratio = np.clip(ratio, 1.0 / cfg.centre_clip, cfg.centre_clip)
            fac[inh] = ratio[post[inh]]
            stats.update(factor_min=float(ratio.min()), factor_max=float(ratio.max()),
                         factor_median=float(np.median(ratio[inh_in > 0])))
        else:
            raise ValueError(f"unknown centre {cfg.centre!r}")
        self._cfac = fac
        return stats

    def _build_pairs(self, type_local: np.ndarray, pre: np.ndarray, post: np.ndarray, count: np.ndarray):
        """Unique (pre_type, post_type) pairs as bins; the top-P by synaptic weight get pair_scale."""
        tp, tq = type_local[pre], type_local[post]
        key = tp.astype(np.int64) * self.n_types + tq
        uniq, inv = np.unique(key, return_inverse=True)
        wsum = np.bincount(inv, weights=count, minlength=len(uniq))
        order = np.argsort(-wsum, kind="stable")
        cum = np.cumsum(wsum[order]) / wsum.sum()
        n_cover = int(np.searchsorted(cum, self.cfg.pair_cover, side="left") + 1) if self.cfg.pair_cover > 0 else 0
        n_scaled = int(min(n_cover, self.cfg.pair_cap, len(uniq)))
        rank = np.empty_like(order); rank[order] = np.arange(len(order))  # new bin id = rank by weight
        self.n_pairs, self.n_pairs_scaled = int(len(uniq)), n_scaled
        self.pair_weight_covered = float(cum[n_scaled - 1]) if n_scaled > 0 else 0.0
        self.register_buffer("bin_index", torch.as_tensor(rank[inv], dtype=torch.long))
        self.register_buffer("pair_pre", torch.as_tensor((uniq // self.n_types)[order], dtype=torch.long))
        self.register_buffer("pair_post", torch.as_tensor((uniq % self.n_types)[order], dtype=torch.long))

    @torch.no_grad()
    def set_w0(self, w0: float) -> None:
        """Re-initialise magnitudes at global scale w0 (tied scales are left untouched)."""
        self.cfg.w0 = float(w0)
        base = self.cfg.w0 * self.base_unit
        if self.cfg.param == "per_edge":
            self.theta.copy_(inv_softplus(base))
            self.theta0.copy_(self.theta)
        elif self.cfg.param == "type_tied":
            self.edge_scale.copy_(self.edge_sign * base)
        else:
            self.values_frozen.copy_(self.edge_sign * base)

    # ---- parameters

    def tau(self) -> Tensor:
        """Membrane time constants (ms) per unit, in [dt, tau_max]."""
        return torch.clamp(F.softplus(self.tau_raw), min=self.cfg.dt_ms, max=self.cfg.tau_max_ms)

    def bin_values(self) -> Tensor:
        """type_tied: exp(log_alpha[pre] + log_beta[post] + pair_scale) per unique type pair."""
        ls = self.log_alpha[self.pair_pre] + self.log_beta[self.pair_post]
        if self.n_pairs_scaled > 0:
            ls = ls.index_add(0, torch.arange(self.n_pairs_scaled, device=ls.device), self.pair_scale)
        return torch.exp(ls)

    def edge_values(self) -> Tensor:
        """Signed synaptic values W_e in CSR order (materialised; for monitoring / per_edge step)."""
        if self.cfg.param == "per_edge":
            return self.edge_sign * F.softplus(self.theta)
        if self.cfg.param == "type_tied":
            return self.edge_scale * self.bin_values()[self.bin_index]
        return self.values_frozen

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def dale_ok(self) -> bool:
        """True iff every edge's sign equals its presynaptic sign (zero magnitude tolerated)."""
        v = self.edge_values().detach()
        return bool(torch.all(v * self.edge_sign >= 0))

    def drift_stats(self) -> dict:
        """A-edge anti-drift instrumentation: Spearman rho(|W|, synapse count), fraction moved > 2x."""
        if self.cfg.param != "per_edge":
            return {}
        from scipy.stats import spearmanr
        mag = F.softplus(self.theta.detach()).cpu().double().numpy()
        mag0 = F.softplus(self.theta0).cpu().double().numpy()
        moved = np.abs(np.log(np.maximum(mag, 1e-30) / np.maximum(mag0, 1e-30))) > math.log(2.0)
        rho = float(spearmanr(mag, self.count.cpu().numpy()).statistic) if len(mag) > 2 else float("nan")
        return {"spearman_rho": rho, "frac_moved_2x": float(moved.mean())}

    def drift_penalty(self) -> Tensor:
        """L2 of theta toward its log1p(count) init (per_edge); zero otherwise."""
        if self.cfg.param != "per_edge":
            return torch.zeros((), dtype=self.dtype, device=self.sign.device)
        return ((self.theta - self.theta0) ** 2).sum()

    # ---- dynamics

    def rate(self, h: Tensor) -> Tensor:
        return torch.clamp(F.softplus(h), max=self.cfg.r_max)

    def rate_slope(self, h: Tensor) -> Tensor:
        """d rate / d h = sigmoid(h) below r_max, 0 once clipped."""
        return torch.sigmoid(h) * (F.softplus(h) < self.cfg.r_max).to(h.dtype)

    def apply_W(self, x: Tensor, values: Optional[Tensor] = None) -> Tensor:
        """y = W @ x for x [M, B] (uses the binned path for type_tied unless values are given)."""
        if values is None and self.cfg.param == "type_tied":
            return self._spmm_binned(self.bin_values(), self.bin_index, x, self.graph, self.edge_scale)
        return self._spmm(self.edge_values() if values is None else values, x, self.graph)

    def node_coefs(self):
        """Per-node [M, 1] tensors: dt/tau (0 for clamped rows), v_rest, bias."""
        m = self.n_nodes
        zeros = self.sign.new_zeros(m, 1)
        coef = zeros.index_copy(0, self.dyn_idx, (self.cfg.dt_ms / self.tau())[self.unit_of_dyn, None])
        vrest = zeros.index_copy(0, self.dyn_idx, self.v_rest[self.unit_of_dyn, None])
        bias = zeros.index_copy(0, self.dyn_idx, self.bias[self.unit_of_dyn, None]) if self.bias is not None else None
        return coef, vrest, bias

    def init_state(self, batch: int) -> CoreState:
        _, vrest, _ = self.node_coefs()
        h = vrest.detach().expand(self.n_nodes, batch).clone()
        xc = h.new_zeros(self.n_clamped, batch)
        return CoreState(h=h, x_clamped=xc, r_out_prev=self.rate(h[self.out_idx]))

    def step(self, state: CoreState, clamped_rates: Tensor) -> CoreState:
        """One decision = K substeps. clamped_rates: [B, n_clamped] (held) or [K, B, n_clamped]."""
        xc = torch.as_tensor(clamped_rates, dtype=self.dtype, device=state.h.device)
        if xc.dim() == 2:
            xc = xc.unsqueeze(0).expand(self.cfg.K, -1, -1)
        if xc.shape != (self.cfg.K, state.h.shape[1], self.n_clamped):
            raise ValueError(f"clamped_rates shape {tuple(xc.shape)} != (K={self.cfg.K}, B, n_clamped={self.n_clamped})")
        h = state.h
        r_out_prev = self.rate(h[self.out_idx])
        coef, vrest, bias = self.node_coefs()
        values = self.edge_values() if self.cfg.param != "type_tied" else None
        for k in range(self.cfg.K):
            r = self.rate(h).index_copy(0, self.clamped_idx, xc[k].T)
            drive = -(h - vrest) + self.apply_W(r, values)
            if bias is not None:
                drive = drive + bias
            h = h + coef * drive
        return CoreState(h=h, x_clamped=xc[-1].T, r_out_prev=r_out_prev)

    def rates(self, state: CoreState) -> Tensor:
        """Rates [M, B]; clamped rows carry the last clamped input exactly."""
        return self.rate(state.h).index_copy(0, self.clamped_idx, state.x_clamped)

    def readout_features(self, state: CoreState) -> Tensor:
        """[B, 2 * n_output]: output rates and their first differences since the last decision."""
        r_out = self.rate(state.h[self.out_idx])
        return torch.cat([r_out, r_out - state.r_out_prev], 0).T

    def summary(self) -> dict:
        d = {"param": self.cfg.param, "backend": self.backend, "n_nodes": self.n_nodes, "n_edges": self.n_edges,
             "n_clamped": self.n_clamped, "n_dynamic": self.n_dynamic, "n_output": self.n_output,
             "n_types": self.n_types, "n_trainable": self.n_trainable(), "dt_ms": self.cfg.dt_ms,
             "w0": self.cfg.w0, "centre": self.centre_factor_stats}
        if self.cfg.param == "type_tied":
            d.update(n_pairs=self.n_pairs, n_pairs_scaled=self.n_pairs_scaled,
                     pair_weight_covered=self.pair_weight_covered)
        return d


# ----------------------------------------------------------------------------- CLI


def _rollout_report(model: ConnectomeCore, batch: int, n_decisions: int, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    state = model.init_state(batch)
    t0 = time.perf_counter()
    for _ in range(n_decisions):
        xc = torch.rand(model.cfg.K, batch, model.n_clamped, generator=gen) * 2.0
        state = model.step(state, xc.to(state.h.device))
    elapsed = time.perf_counter() - t0
    r = model.rates(state)[model.dyn_idx]
    return {"decisions": n_decisions, "batch": batch, "ms_per_decision": 1e3 * elapsed / n_decisions,
            "finite": bool(torch.isfinite(state.h).all()), "mean_rate_dynamic": float(r.mean()),
            "frac_saturated": float((r >= model.cfg.r_max).float().mean()),
            "features_shape": list(model.readout_features(state).shape)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="ConnectomeCore smoke run (forward only)")
    ap.add_argument("--subgraph", help="subgraph_*.npz / null_*.npz")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-nodes", type=int, default=80)
    ap.add_argument("--n-clamped", type=int, default=16)
    ap.add_argument("--mean-in-degree", type=float, default=6.0)
    ap.add_argument("--param", default="type_tied", choices=["per_edge", "type_tied", "frozen"])
    ap.add_argument("--backend", default="auto", choices=["auto", "reference", "spmm"])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--w0", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--decisions", type=int, default=20)
    ap.add_argument("--backward", action="store_true", help="also time one backward pass")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.synthetic:
        sub = synthetic_subgraph(a.n_nodes, a.n_clamped, max(1, a.n_nodes // 10), mean_in_degree=a.mean_in_degree, seed=a.seed)
    elif a.subgraph:
        sub = load_subgraph(a.subgraph)
    else:
        ap.error("give --subgraph or --synthetic")
    model = ConnectomeCore(sub, CoreConfig(param=a.param, K=a.K, w0=a.w0), backend=a.backend)
    out = {"model": model.summary(), "rollout": _rollout_report(model, a.batch, a.decisions, a.seed)}
    if a.backward and model.n_trainable() > 0:
        state = model.init_state(a.batch)
        xc = torch.rand(a.K, a.batch, model.n_clamped) * 2.0
        t0 = time.perf_counter()
        for _ in range(4):
            state = model.step(state, xc)
        model.readout_features(state).sum().backward()
        out["fwd_bwd_4_decisions_ms"] = 1e3 * (time.perf_counter() - t0)
        out["dale_ok_after_grad"] = model.dale_ok()
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
