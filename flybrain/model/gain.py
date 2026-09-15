"""Gain matching and activity monitoring for ConnectomeCore (PLAN.md section 3).

`match_gain` sets the global magnitude scale w0 by bisection (in log w0) so that the
linearised operating-point gain -- the spectral radius of D * W restricted to dynamic nodes,
D = diag(d rate / d h) at the operating point -- equals g_star (0.95). The radius is estimated
by power iteration from a seeded start vector, so the recipe is deterministic given
(graph, seed) and is applied identically to real and null graphs.

Operating point (`op_point`):
    "drive"        (default) one-shot mean drive: h = v_rest + W r_bar, with r_bar = the mean
                   front-end level on clamped nodes and the rest rate softplus(v_rest) on dynamic
                   nodes. No recurrent self-consistency, so g(w0) is smooth and (up to saturation)
                   monotone in w0; bisection is well posed.
    "rest"         D at h = v_rest (D = 0.5 everywhere); g is exactly linear in w0.
    "fixed_point"  the fully relaxed state under constant mean drive. Physically the most honest,
                   but the nonlinear fixed point bifurcates into partially saturated states as w0
                   grows, so g(w0) is rugged and its maximum can sit below g_star; the search
                   then raises instead of returning a wrong w0.
In every mode the relaxed fixed point at the chosen w0 is reported (mean rate, saturated and
silent fractions, self-consistent gain) so "silent/saturated at init" is visible immediately.

Also: `spectral_radius_monitor` (same estimator at a training batch's own state) and
`activity_band_penalty` (per-region mean-rate band regulariser).

CLI:  python -m flybrain.model.gain --synthetic --param per_edge
      python -m flybrain.model.gain --subgraph $FLYBRAIN_DATA/subgraph_core.npz --param type_tied
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Optional, Union

import torch
from torch import Tensor

from flybrain.model.core import ConnectomeCore, CoreConfig, CoreState, load_subgraph, synthetic_subgraph

OP_POINTS = ("drive", "rest", "fixed_point")
SILENT_RATE = 0.05


def _clamped(model: ConnectomeCore, level: Union[float, Tensor], batch: int) -> Tensor:
    """[B, n_clamped] constant clamped rates from a scalar or per-cell level."""
    xc = torch.as_tensor(level, dtype=model.dtype, device=model.sign.device)
    return xc.expand(batch, model.n_clamped) if xc.dim() <= 1 else xc


@torch.no_grad()
def operating_point(model: ConnectomeCore, clamped_level: Union[float, Tensor] = 1.0, batch: int = 1,
                    n_decisions: int = 50) -> tuple[CoreState, float]:
    """Relax the core under constant clamped rates; returns (state, max |dh| over the last decision)."""
    xc = _clamped(model, clamped_level, batch)
    state = model.init_state(batch)
    delta = float("inf")
    for _ in range(n_decisions):
        new = model.step(state, xc)
        delta = float((new.h - state.h).abs().max())
        state = new
    return state, delta


@torch.no_grad()
def mean_drive_state(model: ConnectomeCore, clamped_level: Union[float, Tensor] = 1.0) -> CoreState:
    """One-shot operating point h = v_rest + bias + W r_bar (see module docstring)."""
    _, vrest, bias = model.node_coefs()
    xc = _clamped(model, clamped_level, 1).T                       # [n_clamped, 1]
    r = model.rate(vrest).index_copy(0, model.clamped_idx, xc)
    h = vrest + model.apply_W(r)
    if bias is not None:
        h = h + bias
    h = h.index_copy(0, model.clamped_idx, vrest[model.clamped_idx])
    return CoreState(h=h, x_clamped=xc, r_out_prev=model.rate(h[model.out_idx]))


@torch.no_grad()
def operating_state(model: ConnectomeCore, op_point: str = "drive", clamped_level: Union[float, Tensor] = 1.0,
                    n_decisions: int = 50) -> tuple[CoreState, float]:
    """State at which the gain is linearised, plus the settle residual (0 for non-relaxed modes)."""
    if op_point == "drive":
        return mean_drive_state(model, clamped_level), 0.0
    if op_point == "rest":
        st = model.init_state(1)
        return CoreState(st.h, _clamped(model, clamped_level, 1).T, st.r_out_prev), 0.0
    if op_point == "fixed_point":
        return operating_point(model, clamped_level, 1, n_decisions)
    raise ValueError(f"op_point must be one of {OP_POINTS}, got {op_point!r}")


@torch.no_grad()
def linearised_gain(model: ConnectomeCore, state: CoreState, n_iter: int = 200, seed: int = 0) -> float:
    """Spectral radius of D W on dynamic nodes (D = batch-mean rate slope at `state`) by power iteration.

    Estimate = geometric-mean growth per step over the second half of the iterations, which
    converges to the radius even when the dominant eigenvalue is a complex pair.
    """
    dev = model.sign.device
    d = model.rate_slope(state.h).mean(1, keepdim=True).index_fill(0, model.clamped_idx, 0.0)
    values = model.edge_values()
    gen = torch.Generator(device=dev).manual_seed(seed)
    v = torch.randn(model.n_nodes, 1, generator=gen, dtype=model.dtype, device=dev)
    v = v.index_fill(0, model.clamped_idx, 0.0)
    v = v / v.norm()
    n_avg = max(1, n_iter // 2)
    log_growth = 0.0
    for i in range(n_iter):
        v = d * model.apply_W(v, values)
        nrm = float(v.norm())
        if nrm == 0.0 or not math.isfinite(nrm):
            return 0.0 if nrm == 0.0 else float("inf")
        if i >= n_iter - n_avg:
            log_growth += math.log(nrm)
        v = v / nrm
    return math.exp(log_growth / n_avg)


def spectral_radius_monitor(model: ConnectomeCore, state: CoreState, n_iter: int = 100, seed: int = 0) -> float:
    """Training-time monitor: gain at the batch's current (actual) operating point; no grad."""
    st = CoreState(state.h.detach(), state.x_clamped.detach(), state.r_out_prev.detach())
    return linearised_gain(model, st, n_iter=n_iter, seed=seed)


@torch.no_grad()
def relaxed_report(model: ConnectomeCore, clamped_level: Union[float, Tensor], n_decisions: int,
                   n_iter: int, seed: int) -> dict:
    """What the network actually does at the current w0 under constant mean drive."""
    st, delta = operating_point(model, clamped_level, 1, n_decisions)
    r = model.rates(st)[model.dyn_idx]
    return {"mean_rate_dynamic": float(r.mean()), "frac_saturated": float((r >= model.cfg.r_max).float().mean()),
            "frac_silent": float((r < SILENT_RATE).float().mean()), "settle_max_dh": delta,
            "gain_at_fixed_point": linearised_gain(model, st, n_iter, seed)}


def match_gain(model: ConnectomeCore, g_star: float = 0.95, clamped_level: Union[float, Tensor] = 1.0,
               op_point: str = "drive", tol: float = 1e-3, max_bisect: int = 40, n_iter: int = 200,
               n_decisions: int = 50, seed: int = 0) -> dict:
    """Set model w0 so the linearised gain at the operating point equals g_star; leaves the model there.

    Bracket search doubles w0 upward from the linear-regime guess and stops at the first
    interval where g crosses g_star; if g starts decreasing before crossing (saturation), a
    RuntimeError reports the peak gain reached. Returns a JSON-able dict (w0, gain, trace,
    relaxed-state report).
    """
    if op_point not in OP_POINTS:
        raise ValueError(f"op_point must be one of {OP_POINTS}, got {op_point!r}")
    trace: list[tuple[float, float]] = []

    def g_of(w0: float) -> float:
        model.set_w0(w0)
        st, _ = operating_state(model, op_point, clamped_level, n_decisions)
        g = linearised_gain(model, st, n_iter, seed)
        trace.append((w0, g))
        return g

    eps = 1e-3
    g_eps = g_of(eps)
    if not (g_eps > 0.0 and math.isfinite(g_eps)):
        raise RuntimeError(f"gain at w0={eps} is {g_eps}: no recurrent loop among dynamic nodes?")
    lo = 0.5 * g_star * eps / g_eps                     # gain ~ linear in w0 near zero
    g_lo = g_of(lo)
    for _ in range(60):                                 # make sure we start below the target
        if g_lo < g_star:
            break
        lo /= 2.0
        g_lo = g_of(lo)
    else:
        raise RuntimeError(f"gain never drops below g*={g_star} (g({lo})={g_lo})")
    hi, g_hi = lo, g_lo
    peak = (lo, g_lo)
    for _ in range(60):                                 # double upward until g crosses g*
        hi, g_prev = hi * 2.0, g_hi
        g_hi = g_of(hi)
        if g_hi >= g_star:
            break
        if g_hi < g_prev:
            raise RuntimeError(f"gain peaks at {peak[1]:.4f} (w0={peak[0]:.4g}) below g*={g_star} "
                               f"with op_point={op_point!r}: the operating point saturates before the target gain")
        lo, g_lo, peak = hi, g_hi, (hi, g_hi)
    else:
        raise RuntimeError(f"could not bracket g*={g_star}: g({hi})={g_hi}")

    best = (hi, g_hi)
    for _ in range(max_bisect):
        mid = math.sqrt(lo * hi)
        g_mid = g_of(mid)
        if abs(g_mid - g_star) < abs(best[1] - g_star):
            best = (mid, g_mid)
        if abs(g_mid - g_star) < tol:
            break
        if g_mid < g_star:
            lo = mid
        else:
            hi = mid
    w0, g = best
    model.set_w0(w0)
    res = {"w0": w0, "gain": g, "g_star": g_star, "op_point": op_point, "n_evals": len(trace), "seed": seed,
           "clamped_level": float(clamped_level) if not isinstance(clamped_level, Tensor) else "tensor",
           "trace": trace}
    res.update(relaxed_report(model, clamped_level, n_decisions, n_iter, seed))
    return res


def activity_band_penalty(rates: Tensor, lo: float, hi: float, region_id: Optional[Tensor] = None) -> Tensor:
    """Squared distance of per-region mean rate from [lo, hi], averaged over regions.

    rates: [M, B] (model layout). region_id: LongTensor[M] (e.g. type or neuropil id);
    None = one region. Differentiable in `rates`.
    """
    per_node = rates.mean(1)
    if region_id is None:
        m = per_node.mean().unsqueeze(0)
    else:
        n_reg = int(region_id.max()) + 1
        sums = per_node.new_zeros(n_reg).index_add(0, region_id, per_node)
        cnt = per_node.new_zeros(n_reg).index_add(0, region_id, torch.ones_like(per_node))
        m = sums / cnt.clamp(min=1.0)
    return (torch.relu(lo - m) ** 2 + torch.relu(m - hi) ** 2).mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gain-match a ConnectomeCore (forward only, no training)")
    ap.add_argument("--subgraph")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n-nodes", type=int, default=80)
    ap.add_argument("--param", default="type_tied", choices=["per_edge", "type_tied", "frozen"])
    ap.add_argument("--backend", default="auto", choices=["auto", "reference", "spmm"])
    ap.add_argument("--centre", default="global", choices=["none", "global", "post"])
    ap.add_argument("--op-point", default="drive", choices=list(OP_POINTS))
    ap.add_argument("--g-star", type=float, default=0.95)
    ap.add_argument("--clamped-level", type=float, default=1.0)
    ap.add_argument("--n-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="write the result JSON here")
    a = ap.parse_args(argv)
    if a.synthetic:
        sub = synthetic_subgraph(a.n_nodes, seed=a.seed)
    elif a.subgraph:
        sub = load_subgraph(a.subgraph)
    else:
        ap.error("give --subgraph or --synthetic")
    model = ConnectomeCore(sub, CoreConfig(param=a.param, centre=a.centre), backend=a.backend)
    res = match_gain(model, g_star=a.g_star, clamped_level=a.clamped_level, op_point=a.op_point,
                     n_iter=a.n_iter, seed=a.seed)
    res["model"] = model.summary()
    txt = json.dumps(res, indent=1, default=str)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)


if __name__ == "__main__":
    main()
