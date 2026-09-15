"""Tests for flybrain.model.core and flybrain.model.gain on synthetic subgraph fixtures (CPU).

Parametrised over the sparse backends that import: the in-module reference path always,
flybrain.model.spmm when present.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from flybrain.model import core as C
from flybrain.model import gain as G

BACKENDS = C.available_backends()
torch.set_num_threads(1)


def dense_W(model: C.ConnectomeCore) -> np.ndarray:
    """Dense [M, M] matrix (rows = post, cols = pre) from the model's CSR values."""
    W = np.zeros((model.n_nodes, model.n_nodes), dtype=np.float64)
    v = model.edge_values().detach().double().numpy()
    np.add.at(W, (model.post_of_edge.numpy(), model.pre_of_edge.numpy()), v)
    return W


def dense_gain(model: C.ConnectomeCore, state: C.CoreState) -> float:
    """Exact spectral radius of D W on dynamic nodes via numpy eigvals."""
    d = model.rate_slope(state.h.detach()).mean(1).double().numpy()
    d[model.clamped_idx.numpy()] = 0.0
    A = d[:, None] * dense_W(model)
    return float(np.abs(np.linalg.eigvals(A)).max())


@pytest.fixture(scope="module")
def sub():
    return C.synthetic_subgraph(n_nodes=80, n_clamped=16, n_output=8, n_types=10, seed=0)


def make(sub, backend="reference", **kw):
    return C.ConnectomeCore(sub, C.CoreConfig(**kw), backend=backend)


# ----------------------------------------------------------------------------- fixture sanity


def test_fixture_contract(sub):
    m = len(sub["sign"])
    assert sub["indptr_post"][-1] == len(sub["indices_pre"]) == len(sub["weight"])
    assert np.all(np.diff(sub["indptr_post"])[sub["is_clamped"]] == 0)          # no edges into clamped
    assert np.all(np.diff(sub["indptr_post"])[~sub["is_clamped"]] >= 1)         # every dynamic node fed
    assert np.array_equal(sub["weight"][sub["perm_csr_to_csc"]], sub["weight"][sub["perm_csr_to_csc"]])
    post = np.repeat(np.arange(m), np.diff(sub["indptr_post"]))
    assert np.array_equal(sub["indices_post"], post[sub["perm_csr_to_csc"]])
    assert np.all(np.diff(sub["indptr_pre"]) == np.bincount(sub["indices_pre"], minlength=m))
    assert (sub["type_id"] == -1).any()


def test_npz_roundtrip(sub, tmp_path):
    p = tmp_path / "subgraph_test.npz"
    np.savez_compressed(p, **sub)
    loaded = C.load_subgraph(p)
    a, b = make(sub, param="per_edge"), make(loaded, param="per_edge")
    xc = torch.rand(6, 3, a.n_clamped)
    fa, fb = a.readout_features(a.step(a.init_state(3), xc)), b.readout_features(b.step(b.init_state(3), xc))
    assert torch.equal(fa, fb)


# ----------------------------------------------------------------------------- reference kernel


def test_reference_spmm_matches_dense_and_gradchecks(sub):
    g = C.RefGraph.from_arrays(sub["indptr_post"], sub["indices_pre"], sub["indptr_pre"],
                               sub["indices_post"], sub["perm_csr_to_csc"], "cpu")
    E, M, B = g.n_edges, g.n, 3
    values = torch.randn(E, dtype=torch.float64, requires_grad=True)
    x = torch.randn(M, B, dtype=torch.float64, requires_grad=True)
    y = C.ref_spmm(values, x, g)
    W = torch.zeros(M, M, dtype=torch.float64).index_put((g.row_of_edge.long(), g.indices_pre.long()), values, accumulate=True)
    assert torch.allclose(y, W @ x)
    assert torch.autograd.gradcheck(lambda v, xx: C.ref_spmm(v, xx, g), (values, x))
    bins = torch.randint(0, 5, (E,))
    bv = torch.randn(5, dtype=torch.float64, requires_grad=True)
    scale = torch.randn(E, dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda b, xx: C.ref_spmm_binned(b, bins, xx, g, scale), (bv, x))


@pytest.mark.skipif(len(BACKENDS) < 2, reason="flybrain.model.spmm not importable")
@pytest.mark.parametrize("param", ["per_edge", "type_tied"])
def test_backends_agree(sub, param):
    a, b = make(sub, "reference", param=param), make(sub, "spmm", param=param)
    xc = torch.rand(6, 4, a.n_clamped)
    sa, sb = a.step(a.init_state(4), xc), b.step(b.init_state(4), xc)
    assert torch.allclose(sa.h, sb.h, atol=2e-3, rtol=1e-4)      # float32 summation order differs
    a.readout_features(sa).sum().backward(); b.readout_features(sb).sum().backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and torch.allclose(pa.grad, pb.grad, atol=1e-4, rtol=1e-3), na


# ----------------------------------------------------------------------------- parameter counts


def expected_pair_count(sub, cover=0.8, cap=8000):
    t = sub["type_id"].astype(np.int64).copy()
    t[t < 0] = t.max() + 1
    post = np.repeat(np.arange(len(t)), np.diff(sub["indptr_post"]))
    key = t[sub["indices_pre"]] * (t.max() + 1) + t[post]
    _, inv = np.unique(key, return_inverse=True)
    w = np.sort(np.bincount(inv, weights=sub["weight"]))[::-1]
    cum = np.cumsum(w) / w.sum()
    return int(min(cap, np.searchsorted(cum, cover) + 1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_parameter_counts(sub, backend):
    E, T = len(sub["weight"]), len(np.unique(sub["type_id"]))
    n_dyn = int((~sub["is_clamped"]).sum())
    pe = make(sub, backend, param="per_edge")
    assert pe.n_trainable() == E + 2 * n_dyn                      # theta + per-neuron tau, v_rest
    assert dict(pe.named_parameters())["theta"].shape == (E,)
    tt = make(sub, backend, param="type_tied")
    P = expected_pair_count(sub)
    assert tt.n_pairs_scaled == P and 0 < P < tt.n_pairs
    assert tt.pair_weight_covered >= 0.8
    assert tt.n_trainable() == 4 * T + P                           # alpha, beta, tau, v_rest per type + pairs
    assert make(sub, backend, param="type_tied", pair_cap=3).n_pairs_scaled == 3
    assert make(sub, backend, param="type_tied", bias=True).n_trainable() == 5 * T + P
    fr = make(sub, backend, param="frozen")
    assert fr.n_trainable() == 0
    fr.step(fr.init_state(2), torch.rand(6, 2, fr.n_clamped))


# ----------------------------------------------------------------------------- dynamics


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", ["per_edge", "type_tied", "frozen"])
def test_rollout_finite_bounded(sub, backend, param):
    model = make(sub, backend, param=param, w0=2.0)
    B = 5
    state = model.init_state(B)
    gen = torch.Generator().manual_seed(1)
    for _ in range(30):
        xc = torch.rand(model.cfg.K, B, model.n_clamped, generator=gen) * model.cfg.r_max
        state = model.step(state, xc)
        r = model.rates(state)
        assert torch.isfinite(state.h).all()
        assert (r >= 0).all() and (r <= model.cfg.r_max).all()
    f = model.readout_features(state)
    assert f.shape == (B, 2 * model.n_output) and torch.isfinite(f).all()
    assert torch.allclose(f[:, model.n_output:], (model.rate(state.h[model.out_idx]) - state.r_out_prev).T)


@pytest.mark.parametrize("backend", BACKENDS)
def test_clamped_rows_exactly_overwritten(sub, backend):
    model = make(sub, backend, param="per_edge")
    B = 3
    state = model.init_state(B)
    h0 = state.h.clone()
    xc = torch.rand(model.cfg.K, B, model.n_clamped) * 4
    state = model.step(state, xc)
    r = model.rates(state)
    assert torch.equal(r[model.clamped_idx], xc[-1].T)                       # exact, last substep
    assert torch.equal(state.h[model.clamped_idx], h0[model.clamped_idx])     # clamped h never evolves
    assert not torch.equal(state.h[model.dyn_idx], h0[model.dyn_idx])
    # held input (2-D) equals the same input repeated over K
    s2 = model.step(model.init_state(B), xc[0])
    s3 = model.step(model.init_state(B), xc[0:1].expand(model.cfg.K, -1, -1))
    assert torch.equal(s2.h, s3.h)
    with pytest.raises(ValueError):
        model.step(model.init_state(B), torch.zeros(model.cfg.K + 1, B, model.n_clamped))


def test_substep_matches_manual_euler(sub):
    """One decision of the model equals K hand-written Euler steps against the dense matrix."""
    model = make(sub, param="per_edge", K=3)
    B = 2
    W = torch.tensor(dense_W(model), dtype=torch.float32)
    xc = torch.rand(3, B, model.n_clamped)
    state = model.step(model.init_state(B), xc)
    h = model.init_state(B).h.clone()
    coef = torch.zeros(model.n_nodes, 1)
    coef[model.dyn_idx] = (model.cfg.dt_ms / model.tau().detach())[:, None]
    for k in range(3):
        r = torch.clamp(F.softplus(h), max=5.0)
        r[model.clamped_idx] = xc[k].T
        h = h + coef * (-h + W @ r)
    assert torch.allclose(h, state.h, atol=1e-5)
    assert abs(model.cfg.dt_ms - 38.0) < 1e-9


def test_tau_clamped_to_band(sub):
    model = make(sub, param="type_tied")
    with torch.no_grad():
        model.tau_raw[:] = torch.linspace(-50, 50, model.n_types)
    tau = model.tau()
    assert (tau >= model.cfg.dt_ms).all() and (tau <= model.cfg.tau_max_ms).all()
    assert abs(float(make(sub, param="per_edge").tau().mean()) - 20.0) < 1e-4


# ----------------------------------------------------------------------------- Dale / signs


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", ["per_edge", "type_tied"])
def test_dale_signs_survive_optimizer_steps(sub, backend, param):
    torch.manual_seed(0)
    model = make(sub, backend, param=param)
    opt = torch.optim.Adam(model.parameters(), lr=0.3)  # deliberately large steps
    B = 4
    for it in range(15):
        state = model.init_state(B)
        for _ in range(3):
            state = model.step(state, torch.rand(model.cfg.K, B, model.n_clamped))
        loss = (model.readout_features(state) * torch.randn(B, 2 * model.n_output)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        v = model.edge_values().detach()
        assert torch.isfinite(v).all() and torch.isfinite(state.h).all()
        assert model.dale_ok()
        assert torch.equal(torch.sign(v)[v != 0], model.edge_sign[v != 0])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    if param == "per_edge":
        st = model.drift_stats()
        assert 0 <= st["frac_moved_2x"] <= 1 and np.isfinite(st["spearman_rho"])
        assert model.drift_penalty() > 0


# ----------------------------------------------------------------------------- type_tied algebra


def test_type_tied_P0_reduces_to_alpha_beta(sub):
    model = make(sub, param="type_tied", pair_cover=0.0)
    assert model.n_pairs_scaled == 0 and model.pair_scale.numel() == 0
    with torch.no_grad():
        gen = torch.Generator().manual_seed(0)
        model.log_alpha.normal_(generator=gen); model.log_beta.normal_(generator=gen)
    tpre, tpost = model.type_id[model.pre_of_edge], model.type_id[model.post_of_edge]
    expect = model.edge_sign * model.cfg.w0 * model.base_unit * torch.exp(model.log_alpha[tpre] + model.log_beta[tpost])
    assert torch.allclose(model.edge_values(), expect, rtol=1e-5, atol=1e-6)
    # a per_edge model with theta = softplus^-1(|expect|) produces the same dynamics
    pe = make(sub, param="per_edge")
    with torch.no_grad():
        pe.theta.copy_(C.inv_softplus(expect.abs()))
    xc = torch.rand(6, 2, model.n_clamped)
    ha, hb = model.step(model.init_state(2), xc).h, pe.step(pe.init_state(2), xc).h
    assert torch.allclose(ha, hb, rtol=1e-5, atol=1e-3)   # |h| reaches ~1e3 here; float32 round-off


def test_type_tied_pair_scale_only_top_pairs(sub):
    model = make(sub, param="type_tied", pair_cap=4)
    with torch.no_grad():
        model.pair_scale.fill_(1.0)
    v = model.edge_values().detach().abs()
    base = (model.cfg.w0 * model.base_unit)
    scaled = model.bin_index < 4
    assert torch.allclose(v[scaled], base[scaled] * np.e, rtol=1e-5)
    assert torch.allclose(v[~scaled], base[~scaled], rtol=1e-5)
    # at init (all zeros) type_tied and per_edge have identical values
    a, b = make(sub, param="type_tied").edge_values(), make(sub, param="per_edge").edge_values()
    assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


def test_centring_balances_inhibition(sub):
    g = make(sub, param="per_edge", centre="global")
    v = g.edge_values().detach()
    assert abs(float(v[v > 0].sum() + v[v < 0].sum())) < 1e-3 * float(v.abs().sum())
    p = make(sub, param="per_edge", centre="post")
    vp = p.edge_values().detach()
    post = p.post_of_edge
    exc = torch.zeros(p.n_nodes).index_add(0, post, vp.clamp(min=0))
    inh = torch.zeros(p.n_nodes).index_add(0, post, vp.clamp(max=0))
    both = (exc > 0) & (inh < 0)
    ratio = (exc / -inh)[both]
    assert ((ratio - 1).abs() < 1e-4).all() or ((ratio <= 10.0 + 1e-4) & (ratio >= 0.1 - 1e-4)).all()
    n = make(sub, param="per_edge", centre="none")
    assert torch.allclose(n.edge_values().abs(), n.cfg.w0 * torch.log1p(n.count))


# ----------------------------------------------------------------------------- gain matching


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("param", ["per_edge", "type_tied"])
@pytest.mark.parametrize("op_point", ["drive", "rest"])
def test_match_gain_hits_target(sub, backend, param, op_point):
    model = make(sub, backend, param=param)
    res = G.match_gain(model, g_star=0.95, op_point=op_point, n_iter=400, seed=0)
    assert abs(res["gain"] - 0.95) <= 0.01
    assert model.cfg.w0 == res["w0"] and res["w0"] > 0
    # verify against the exact eigenvalue at the same operating point
    state, _ = G.operating_state(model, op_point, 1.0)
    assert abs(dense_gain(model, state) - 0.95) <= 0.01
    # the relaxed network at this w0 is neither silent nor fully saturated
    assert res["settle_max_dh"] < 1e-3
    assert 0.0 < res["mean_rate_dynamic"] < model.cfg.r_max and res["frac_saturated"] < 1.0
    assert np.isfinite(res["gain_at_fixed_point"])


def test_match_gain_fixed_point_mode_is_honest(sub):
    """The self-consistent fixed point may saturate before g*; then match_gain must raise, not lie."""
    model = make(sub, param="per_edge")
    try:
        res = G.match_gain(model, g_star=0.95, op_point="fixed_point", n_iter=400, seed=0)
    except RuntimeError as e:
        assert "peak" in str(e)
    else:
        assert abs(res["gain"] - 0.95) <= 0.01
        state, delta = G.operating_point(model, 1.0, 1, 50)
        assert delta < 1e-3 and abs(dense_gain(model, state) - 0.95) <= 0.01


def test_match_gain_deterministic_and_graph_specific(sub):
    a = G.match_gain(make(sub, param="per_edge"), seed=3)
    b = G.match_gain(make(sub, param="per_edge"), seed=3)
    assert a["w0"] == b["w0"] and a["gain"] == b["gain"]
    other = C.synthetic_subgraph(n_nodes=80, n_clamped=16, n_output=8, n_types=10, seed=7)  # a "null" graph
    c = G.match_gain(make(other, param="per_edge"), seed=3)
    assert abs(c["gain"] - 0.95) <= 0.01 and c["w0"] != a["w0"]
    # per_edge and type_tied share the init, so they must land on the same w0
    d = G.match_gain(make(sub, param="type_tied"), seed=3)
    assert abs(d["w0"] - a["w0"]) <= 1e-6 * a["w0"]


def test_rest_mode_is_linear_in_w0(sub):
    model = make(sub, param="per_edge")
    res = G.match_gain(model, g_star=0.95, op_point="rest", n_iter=400)
    model.set_w0(2 * res["w0"])
    st, _ = G.operating_state(model, "rest", 1.0)
    assert abs(G.linearised_gain(model, st, 400) - 1.9) < 0.02


def test_spectral_radius_monitor_and_power_iteration(sub):
    torch.manual_seed(0)
    model = make(sub, param="per_edge", w0=0.3)
    state = model.init_state(2)
    for _ in range(5):
        state = model.step(state, torch.rand(6, 2, model.n_clamped))
    est = G.spectral_radius_monitor(model, state, n_iter=400, seed=0)
    assert abs(est - dense_gain(model, state)) < 0.01
    assert G.spectral_radius_monitor(model, state, n_iter=400, seed=0) == est
    assert state.h.requires_grad  # monitor must not detach the caller's graph


# ----------------------------------------------------------------------------- activity band


def test_activity_band_penalty():
    r = torch.full((10, 4), 1.0, requires_grad=True)
    assert float(G.activity_band_penalty(r, 0.5, 2.0)) == 0.0
    pen = G.activity_band_penalty(r, 2.0, 3.0)
    assert abs(float(pen) - 1.0) < 1e-6
    pen.backward()
    assert r.grad is not None and (r.grad < 0).all()
    region = torch.tensor([0] * 5 + [1] * 5)
    rr = torch.cat([torch.zeros(5, 4), torch.full((5, 4), 3.0)])
    assert abs(float(G.activity_band_penalty(rr, 1.0, 2.0, region)) - 1.0) < 1e-6   # (1^2 + 1^2) / 2
