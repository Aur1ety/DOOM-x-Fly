"""Tests for the frozen front end: geometry, EMD physiology, fill, loom, batching (run on the server)."""
import numpy as np
import pytest
import torch

from flybrain.vision import stimuli
from flybrain.vision.frontend import FrontEnd
from flybrain.vision.geometry import CLAMPED_TYPES, DEFAULT_PATH, Geometry, NBR_NAMES

torch.manual_seed(0)


@pytest.fixture(scope="module")
def geom():
    return Geometry.synthetic(12)


@pytest.fixture(scope="module")
def fe(geom):
    return FrontEnd(geom)


def _row_sums(indptr, weights):
    n = len(indptr) - 1
    return np.bincount(np.repeat(np.arange(n), np.diff(indptr)), weights=weights, minlength=n)


def _units(fe, ty, side=None, driven=True):
    idx = fe.index
    m = idx.type == ty
    if side is not None:
        m &= idx.side == side
    if driven:
        m &= idx.driven
    return np.nonzero(m)[0]


# ---- geometry ---------------------------------------------------------------------------------

def test_lattice_neighbours_and_angles(geom):
    F, Bk, UF, UB, DF, DB = range(6)
    assert list(geom.meta["nbr_names"]) == list(NBR_NAMES)
    nbr = geom.nbr
    for i in range(geom.n_columns):
        f = nbr[i, F]
        if f >= 0:
            assert nbr[f, Bk] == i                                       # front/back are inverses
            # the front neighbour lies toward the midline: larger az in the left eye, smaller in the right
            # (|az| is not monotone once the frontmost columns cross into the binocular overlap band)
            if geom.col_side[i] == 0:
                assert geom.col_az[f] > geom.col_az[i]
            else:
                assert geom.col_az[f] < geom.col_az[i]
        for k in (UF, UB):
            if nbr[i, k] >= 0:
                assert geom.col_el[nbr[i, k]] > geom.col_el[i]
        for k in (DF, DB):
            if nbr[i, k] >= 0:
                assert geom.col_el[nbr[i, k]] < geom.col_el[i]
    # sides: left eye azimuths <= +overlap/2, right eye >= -overlap/2
    ov = geom.meta["overlap_deg"] / 2
    assert geom.col_az[geom.col_side == 0].max() <= ov + 1e-4 and geom.col_az[geom.col_side == 1].min() >= -ov - 1e-4


def test_sampler_coverage(geom):
    row_sum = _row_sums(geom.pix_indptr, geom.pix_weights)[np.diff(geom.pix_indptr) > 0]
    assert np.all(geom.pix_weights >= 0)
    assert row_sum.max() <= 1.0 + 1e-5
    assert np.allclose(row_sum, geom.coverage[np.diff(geom.pix_indptr) > 0], atol=1e-5)
    assert 0 <= geom.coverage.min() and geom.coverage.max() <= 1.0 + 1e-6
    n_l, n_r = (geom.driven & (geom.col_side == 0)).sum(), (geom.driven & (geom.col_side == 1)).sum()
    assert n_l > 50 and n_r > 50 and n_l == n_r                        # symmetric synthetic eyes
    # interior columns are fully covered (pixel quadrature error < 2%); "interior" excludes the
    # binocular overlap band, where the FOV split deliberately ramps each eye's weight to 0
    ov = geom.meta["overlap_deg"] / 2
    own_half = np.where(geom.col_side == 0, geom.col_az <= -ov, geom.col_az >= ov)
    inside = (np.abs(geom.col_az) < 40) & (np.abs(geom.col_el) < 30) & own_half
    assert geom.coverage[inside].min() > 0.98
    assert geom.pix_indices.max() < geom.frame_hw[0] * geom.frame_hw[1]


def test_unit_order_and_index(fe, geom):
    idx = fe.index
    assert len(idx) == geom.n_cells == fe.n_clamped
    assert np.all(np.diff(idx.type_id) >= 0)                              # sorted by type
    assert set(idx.type) == set(CLAMPED_TYPES)
    perm = idx.permutation_for(idx.bodyId[::-1][:100])
    assert np.array_equal(idx.bodyId[perm], idx.bodyId[::-1][:100])
    with pytest.raises(KeyError):
        idx.permutation_for(np.array([-1]))


# ---- physiology (column path, substep rate) ---------------------------------------------------

def test_dsi_and_preferred_direction(fe):
    tfs = (2.0,)
    resp = stimuli.gratings_battery(fe, tfs=tfs, settle_s=0.4)
    rows = stimuli.dsi_table(resp, fe, tfs=tfs)
    assert len(rows) == 16                                                # 8 EMD types x 2 sides
    for r in rows:
        assert r["median_dsi"] >= 0.5, r
        assert r["frac_correct_pd"] >= 0.8, r
        assert r["mean_pd_rate"] > 0.05, r


def test_on_off_polarity(fe):
    f = stimuli.flash_battery(fe)
    for ty in ("Mi1", "Mi4"):
        assert f[ty]["up"] > 0.05 and f[ty]["up"] > 10 * f[ty]["down"]
    for ty in ("Mi9", "Tm1", "Tm2", "Tm4", "Tm9"):
        assert f[ty]["down"] > 0.05 and f[ty]["down"] > 10 * f[ty]["up"]
    for ty in stimuli.EMD_TYPES:                                          # no motion in a uniform flash
        assert f[ty]["up"] == 0.0 and f[ty]["down"] == 0.0
    for ty in ("Tm20", "Tm5a", "Tm5b", "Tm5c", "TmY5a"):                   # grey has no colour opponency
        assert f[ty]["up"] == 0.0 and f[ty]["down"] == 0.0


def test_loom_monotone(fe):
    rates = (5.0, 10.0, 20.0, 40.0, 80.0)
    d = stimuli.disc_battery(fe, rates_deg_s=rates)
    loom = [d["curves"]["expand"]["T5"][str(v)]["mean_during_motion"] for v in rates]
    assert all(b > a for a, b in zip(loom, loom[1:])), loom
    recede = [d["curves"]["recede"]["T5"][str(v)]["mean_during_motion"] for v in rates]
    assert max(recede) < 0.05 * max(loom)
    trans = [d["curves"]["translate"]["T5"][str(v)]["mean_during_motion"] for v in rates]
    assert all(t < 0.5 * l for t, l in zip(trans, loom))


# ---- pixel path: fill, cross-fade, batching ---------------------------------------------------

def _uniform(geom, v, B=1):
    H, W = geom.frame_hw
    return np.full((B, H, W), int(round(v * 255)), np.uint8)


def _run_frames(fe, frames):
    st = fe.init_state(frames[0].shape[0])
    outs = []
    for f in frames:
        r, st = fe.process(f, None, st)
        outs.append(r)
    return torch.cat(outs, 0)                                             # [n_frames*K, B, M]


def test_static_scene_is_silent(geom):
    rng = np.random.default_rng(1)
    H, W = geom.frame_hw
    tex = rng.integers(0, 256, size=(1, H, W), dtype=np.uint8)
    for fill in ("mean", "black"):
        fe = FrontEnd(geom, fill=fill)
        out = _run_frames(fe, [tex] * 25)                                  # 2.85 s static
        last = out[-1, 0]
        assert last.max() < 1e-3, (fill, float(last.max()))


def test_mean_fill_no_edge_black_fill_has_one(geom):
    frames = [_uniform(geom, 0.3)] * 10 + [_uniform(geom, 0.6)] * 3
    resp = {}
    for fill in ("mean", "black"):
        fe = FrontEnd(geom, fill=fill)
        out = _run_frames(fe, frames)                                      # [78, 1, M]
        step = out[60:66, 0].mean(0)                                       # the frame after the step
        mi1 = _units(fe, "Mi1", driven=True)
        idx = fe.index
        outside = np.nonzero((idx.type == "Mi1") & (idx.col >= 0) & (geom.coverage[np.maximum(idx.col, 0)] == 0))[0]
        assert len(mi1) > 50 and len(outside) > 50
        resp[fill] = (step[mi1], step[outside], step[[i for ty in stimuli.EMD_TYPES for i in _units(fe, ty)]])
    r_in, r_out, r_emd = resp["mean"]
    assert r_in.mean() > 0.1
    assert torch.allclose(r_out, r_in.mean().expand_as(r_out), rtol=0.05, atol=1e-3)   # outside == inside
    assert (r_in.std() / r_in.mean()) < 0.05                                          # uniform inside
    assert r_emd.max() < 0.01 * r_in.mean()                                           # no motion artefact
    b_in, b_out, _ = resp["black"]
    assert b_in.mean() > 0.1 and b_out.max() == 0.0                                   # black fill: silent outside


def test_cross_fade_spreads_the_step(geom):
    fe = FrontEnd(geom, K=6)
    frames = [_uniform(geom, 0.3)] * 6 + [_uniform(geom, 0.6)]
    out = _run_frames(fe, frames)[-6:, 0]                                  # the 6 substeps of the step frame
    mi1 = _units(fe, "Mi1")
    m = out[:, mi1].mean(1)
    assert m[0] < m[2] < m[5]                                              # ramps up across substeps


def test_batch_consistency_and_reset(geom):
    fe = FrontEnd(geom)
    rng = np.random.default_rng(2)
    H, W = geom.frame_hw
    a = rng.integers(0, 256, size=(H, W), dtype=np.uint8); b = rng.integers(0, 256, size=(H, W), dtype=np.uint8)
    seq4 = [np.stack([a, b, a, b]), np.stack([b, a, b, a]), np.stack([a, a, a, a])]
    out4 = _run_frames(fe, seq4)
    assert out4.shape == (18, 4, fe.n_clamped) and out4.dtype == torch.float32
    assert torch.isfinite(out4).all() and out4.min() >= 0
    assert torch.equal(out4[:, 0], out4[:, 2]) and torch.equal(out4[:, 1], out4[:, 3])
    out1 = _run_frames(fe, [f[:1] for f in seq4])
    assert torch.allclose(out1[:, 0], out4[:, 0], atol=1e-5)
    assert not torch.allclose(out4[:, 0], out4[:, 1])
    # reset: env 1 re-initialised on the next frame behaves like a fresh state
    st = fe.init_state(2)
    for f in seq4:
        _, st = fe.process(f[:2], None, st)
    fe.reset(st, np.array([False, True]))
    r, st = fe.process(np.stack([b, b]), None, st)
    fresh, _ = fe.process(np.stack([b]), None, fe.init_state(1))
    assert torch.allclose(r[:, 1], fresh[:, 0], atol=1e-6) and not torch.allclose(r[:, 0], fresh[:, 0], atol=1e-6)


def test_rgb_path_drives_chromatic_types(geom):
    fe = FrontEnd(geom)
    H, W = geom.frame_hw
    gray = _uniform(geom, 0.5)
    red = np.zeros((1, H, W, 3), np.uint8); red[..., 0] = 200; red[..., 1] = 50; red[..., 2] = 50
    st = fe.init_state(1)
    for _ in range(5):
        r, st = fe.process(gray, red, st)
    r = r[-1, 0]
    assert r[_units(fe, "TmY5a")].mean() > 0.1 and r[_units(fe, "Tm20")].max() == 0.0     # R-G positive
    assert r[_units(fe, "Tm5b")].mean() > 0.1 and r[_units(fe, "Tm5a")].max() == 0.0      # B-Y negative
    half = fe.rgb_hw
    small = red[:, ::fe.rgb_ds, ::fe.rgb_ds]
    r2, _ = fe.process(gray, small, fe.init_state(1))
    assert small.shape[1:3] == half and torch.isfinite(r2).all()


# ---- real geometry (skipped unless built on the server) --------------------------------------

@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="geometry_v1.npz not built")
def test_real_geometry_file():
    g = Geometry.load(DEFAULT_PATH)
    s = g.summary()
    assert s["n_columns"] == 1771 and s["columns_L"] == 879 and s["columns_R"] == 892
    assert 100 < s["driven_L"] < 400 and 100 < s["driven_R"] < 400
    assert 0.9 < s["side_gain_L"] < 1.1 and 0.9 < s["side_gain_R"] < 1.1
    assert s["assign_counts"]["unassigned"] < 0.01 * g.n_cells
    for side in ("L", "R"):
        assert not g.meta["frame_evidence"][side]["diagnostics"]["fallback"]
    fe = FrontEnd(g)
    assert fe.n_clamped == g.n_cells and len(np.unique(g.cell_bodyId)) == g.n_cells
    nz = np.diff(g.pix_indptr) > 0
    assert np.allclose(_row_sums(g.pix_indptr, g.pix_weights)[nz], g.coverage[nz], atol=1e-5)
    neurons = DEFAULT_PATH.parent.parent / "graph" / "neurons.parquet"
    if neurons.exists():
        import pandas as pd
        n = pd.read_parquet(neurons, columns=["bodyId", "type"])
        n = n[n["type"].isin(CLAMPED_TYPES)]
        assert set(n["bodyId"].tolist()) == set(g.cell_bodyId.tolist())         # aligns with the data module
        perm = fe.index.permutation_for(n["bodyId"].to_numpy())
        assert np.array_equal(fe.index.type[perm], n["type"].to_numpy())
