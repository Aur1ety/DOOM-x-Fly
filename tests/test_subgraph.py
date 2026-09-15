"""Tests for flybrain.data.subgraph on a synthetic graph_full.npz + neurons.parquet fixture."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from flybrain.data import subgraph as sg

# (type, somaSide, superclass, consensus_nt). Index = position.
NODES = [
    ("T4a", "L", "optic", "acetylcholine"),          # 0 clamped
    ("T4a", "R", "optic", "acetylcholine"),          # 1 clamped
    ("Tm1", "L", "optic", "acetylcholine"),          # 2 clamped; only a weak (w=3) output
    ("Li1", "L", "optic", "gaba"),                   # 3 core, fwd 1
    ("LC4", "L", "visual_projection", "acetylcholine"),  # 4 forced + core
    ("n1", "R", "central", "acetylcholine"),         # 5 core (fwd 1, bwd 4), autapse, w=5 input
    ("n2", "R", "central", "glutamate"),             # 6 core
    ("n3", "R", "central", "acetylcholine"),         # 7 core
    ("n4", "R", "central", "acetylcholine"),         # 8 core (fwd 4, bwd 1)
    ("DNxx99", "R", "descending_neuron", "acetylcholine"),  # 9 DN at fwd 5: excluded (not forced)
    ("DNa02", "L", "descending_neuron", "acetylcholine"),   # 10 forced + core + readout
    ("DNa02", "R", "descending_neuron", "acetylcholine"),   # 11 forced, unreachable (w=2 input)
    ("weak", "L", "central", "acetylcholine"),       # 12 core only when w_min <= 3
    ("dead", "L", "central", "acetylcholine"),       # 13 fwd 1 but never reaches a DN
    ("orphan", "L", "central", "acetylcholine"),     # 14 reaches a DN, unreachable from I
    ("MDN", "L", "descending_neuron", "acetylcholine"),     # 15 forced, isolated, readout
    ("MDN", "R", "descending_neuron", "acetylcholine"),     # 16 forced, isolated, readout
    ("LPi34-12", "L", "optic", "glutamate"),         # 17 forced via LPi*, unreachable from I
    ("HSE", "R", "optic", "acetylcholine"),          # 18 forced via HS*, isolated
    (None, "R", "central", "unclear"),               # 19 core, null type
    ("AOTU019", "L", "central", "acetylcholine"),    # 20 core + readout (attack_midline)
    ("pIP10", "R", "central", "acetylcholine"),      # 21 forced, isolated, readout
    ("DNp01", "R", "descending_neuron", "acetylcholine"),   # 22 forced, isolated, readout
    ("vncX", "L", "vnc_intrinsic", "acetylcholine"),        # 23 VNC: excluded by default
]
EDGES = [  # (pre, post, weight)
    (0, 3, 10), (3, 4, 10), (4, 10, 10),                                   # I -> DNa02 L in 3 hops
    (1, 5, 5), (5, 6, 6), (6, 7, 7), (7, 8, 8), (8, 9, 9), (5, 5, 6),       # 5-hop chain + autapse
    (2, 12, 3), (12, 10, 10),                                              # weak chain
    (0, 13, 10),                                                           # dead end
    (14, 10, 10),                                                          # orphan
    (4, 11, 2),                                                            # weak into DNa02 R
    (3, 0, 10), (0, 2, 10),                                                # into clamped: dropped
    (17, 4, 10),                                                           # LPi -> LC4
    (0, 19, 10), (19, 10, 10),                                             # via null-type node
    (4, 20, 10), (20, 10, 10),                                             # AOTU019 path
    (10, 23, 10), (23, 10, 10),                                            # DN <-> VNC loop
]
N = len(NODES)
SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "unclear": 1}
BODY = 1000 + 7 * np.arange(N)

# Expectations at w_min=5, hops=4 (worked out by hand from the lists above).
CORE_5 = {3, 4, 5, 6, 7, 8, 10, 19, 20}
FORCED = {4, 10, 11, 15, 16, 17, 18, 20, 21, 22}
CLAMPED = {0, 1, 2}
NODES_5 = sorted(CORE_5 | FORCED | CLAMPED)              # 19 nodes
EDGES_5 = {(0, 3), (3, 4), (4, 10), (1, 5), (5, 6), (6, 7), (7, 8), (5, 5),
           (17, 4), (0, 19), (19, 10), (4, 20), (20, 10)}  # 13 stored edges


def dense_w():
    w = np.zeros((N, N), dtype=np.float32)   # rows = post, cols = pre
    for pre, post, wt in EDGES:
        w[post, pre] = wt
    return w


def write_fixture(d):
    """Write graph_full.npz + neurons.parquet with the CONTRACTS.md keys into dir d."""
    pre = np.array([e[0] for e in EDGES]); post = np.array([e[1] for e in EDGES])
    wt = np.array([e[2] for e in EDGES], dtype=np.float32)
    a = sp.coo_matrix((wt, (post, pre)), shape=(N, N)).tocsr()
    a.sort_indices()
    indptr_post = a.indptr.astype(np.int64); indices_pre = a.indices.astype(np.int32)
    weight = a.data.astype(np.float32); e = weight.shape[0]
    post_e = np.repeat(np.arange(N), np.diff(indptr_post))
    perm = np.lexsort((post_e, indices_pre)).astype(np.int64)
    indptr_pre = np.zeros(N + 1, np.int64); np.cumsum(np.bincount(indices_pre, minlength=N), out=indptr_pre[1:])
    sign = np.array([SIGN[n[3]] for n in NODES], dtype=np.int8)
    np.savez_compressed(d / "graph_full.npz", n_neurons=N, n_edges=e, bodyId=BODY.astype(np.int64),
                        sign=sign, indptr_post=indptr_post, indices_pre=indices_pre, weight=weight,
                        indptr_pre=indptr_pre, indices_post=post_e[perm].astype(np.int32),
                        perm_csr_to_csc=perm, ledger=json.dumps({"n_neurons": N, "n_edges": e}))
    types = [n[0] for n in NODES]
    uniq = sorted({t for t in types if t is not None})
    tid = np.array([uniq.index(t) if t is not None else -1 for t in types], dtype=np.int32)
    tbl = pa.table({
        "idx": pa.array(np.arange(N, dtype=np.int64)), "bodyId": pa.array(BODY.astype(np.int64)),
        "type": pa.array(types, type=pa.string()),
        "instance": pa.array([f"{t}_{s}" if t else None for t, s, _, _ in NODES], type=pa.string()),
        "superclass": pa.array([n[2] for n in NODES], type=pa.string()),
        "class": pa.array([None] * N, type=pa.string()), "subclass": pa.array([None] * N, type=pa.string()),
        "somaSide": pa.array([n[1] for n in NODES], type=pa.string()),
        "rootSide": pa.array([n[1] for n in NODES], type=pa.string()),
        "assignedOlHex1": pa.array([None] * N, type=pa.int32()), "assignedOlHex2": pa.array([None] * N, type=pa.int32()),
        "status": pa.array(["Traced"] * N, type=pa.string()), "statusLabel": pa.array(["Traced"] * N, type=pa.string()),
        "consensus_nt": pa.array([n[3] for n in NODES], type=pa.string()),
        "sign": pa.array(sign), "type_id": pa.array(tid),
    })
    pq.write_table(tbl, d / "neurons.parquet")
    return d


@pytest.fixture(scope="module")
def gdir(tmp_path_factory):
    return write_fixture(tmp_path_factory.mktemp("graph"))


def full_graph(gdir):
    """FullGraph with the default (VNC) exclusion applied, as build_subgraph does."""
    g = sg.FullGraph(gdir)
    g.set_excluded(sg.superclass_prefix_mask(g.cells["superclass"], sg.EXCLUDE_SUPERCLASS_PREFIX))
    return g


@pytest.fixture(scope="module")
def built(gdir):
    sub, report = sg.build_subgraph(gdir, name="t", w_min=5.0, verbose=False)
    return sub, report


# ----------------------------------------------------------------------------- type sets

def test_type_matching(gdir):
    g = sg.FullGraph(gdir)
    m = sg.match_types(g.types, sg.FORCED_PATTERNS)
    assert m["LPi*"] == ["LPi34-12"] and m["HS*"] == ["HSE"] and m["LC4"] == ["LC4"]
    assert m["LC6"] == [] and m["VS*"] == []
    assert set(np.flatnonzero(sg.type_mask(g.types, sg.FORCED_PATTERNS))) == FORCED
    assert set(np.flatnonzero(sg.type_mask(g.types, sg.CLAMPED_TYPES))) == CLAMPED
    assert sg.norm_side("L") == "L" and sg.norm_side("right") == "R" and sg.norm_side(None) == "other"


# ----------------------------------------------------------------------------- reachability

def test_hop_distances_and_core(gdir):
    g = full_graph(gdir)
    clamped = sg.type_mask(g.types, sg.CLAMPED_TYPES)
    output = np.array([n[2] == "descending_neuron" for n in NODES])
    forced = sg.type_mask(g.types, sg.FORCED_PATTERNS)
    sel = sg.select_nodes(g, 5.0, 4, clamped, output, forced)
    fwd, bwd = sel["fwd"], sel["bwd"]
    assert fwd[0] == 0 and fwd[3] == 1 and fwd[4] == 2 and fwd[20] == 3 and fwd[8] == 4
    assert fwd[9] == -1 and fwd[12] == -1 and fwd[14] == -1      # 5 hops / weak edge / orphan
    assert bwd[10] == 0 and bwd[8] == 1 and bwd[3] == 2 and bwd[6] == 3 and bwd[5] == 4
    assert bwd[1] == -1 and bwd[13] == -1 and bwd[2] == -1       # 5 hops / dead end / weak
    assert set(np.flatnonzero(sel["core"])) == CORE_5
    assert set(np.flatnonzero(sel["dynamic"])) == CORE_5 | FORCED
    # backward BFS must not route through a clamped cell (edge 3->0 is dropped): 0 reaches only via 0->19
    assert bwd[0] == 2


def test_hop_limit_is_sharp(gdir):
    g = full_graph(gdir)
    clamped = sg.type_mask(g.types, sg.CLAMPED_TYPES)
    output = np.array([n[2] == "descending_neuron" for n in NODES])
    forced = sg.type_mask(g.types, sg.FORCED_PATTERNS)
    core5 = sg.select_nodes(g, 5.0, 5, clamped, output, forced)["core"]
    assert core5[9] and set(np.flatnonzero(core5)) == CORE_5 | {9}
    core3 = sg.select_nodes(g, 5.0, 3, clamped, output, forced)["core"]
    assert not core3[5] and not core3[8] and set(np.flatnonzero(core3)) == CORE_5 - {5, 8}


def test_threshold_changes_core(gdir):
    g = full_graph(gdir)
    clamped = sg.type_mask(g.types, sg.CLAMPED_TYPES)
    output = np.array([n[2] == "descending_neuron" for n in NODES])
    forced = sg.type_mask(g.types, sg.FORCED_PATTERNS)
    s3 = sg.select_nodes(g, 3.0, 4, clamped, output, forced)
    assert set(np.flatnonzero(s3["core"])) == CORE_5 | {12} and s3["edge_mask"].sum() == 15
    s6 = sg.select_nodes(g, 6.0, 4, clamped, output, forced)
    assert set(np.flatnonzero(s6["core"])) == {3, 4, 10, 19, 20} and s6["edge_mask"].sum() == 8


# ----------------------------------------------------------------------------- budget clamp

def test_budget_clamp_raises_w_min(gdir):
    _, rep = sg.build_subgraph(gdir, w_min=5.0, max_dynamic=13, verbose=False)
    assert rep["w_min"] == 6.0 and [t["w_min"] for t in rep["clamp_trajectory"]] == [5.0, 6.0]
    assert rep["counts"]["n_dynamic"] == 12 and rep["counts"]["budget_ok"]
    _, rep = sg.build_subgraph(gdir, w_min=5.0, max_edges=10, verbose=False)
    assert rep["w_min"] == 6.0 and rep["counts"]["n_edges"] == 8
    _, rep = sg.build_subgraph(gdir, w_min=5.0, verbose=False)
    assert rep["w_min"] == 5.0 and len(rep["clamp_trajectory"]) == 1


def test_vnc_exclusion(gdir):
    sub, rep = sg.build_subgraph(gdir, verbose=False)
    assert 23 not in sub["node_idx"] and rep["counts"]["n_excluded_cells"] == 1
    assert rep["counts"]["n_excluded_edges"] == 2 and rep["rule"]["exclude_superclass_prefix"] == "vnc_"
    sub, rep = sg.build_subgraph(gdir, verbose=False, exclude_prefix="")
    assert 23 in sub["node_idx"] and rep["counts"]["n_excluded_cells"] == 0
    assert rep["counts"]["n_core"] == 10 and rep["counts"]["n_edges"] == 15   # 10->23->10 loop enters
    # prefix list; named cells (forced AOTU019 = 20, pIP10 = 21 in 'central') are immune
    sub, rep = sg.build_subgraph(gdir, verbose=False, exclude_prefix="central,vnc_")
    assert rep["counts"]["n_excluded_cells"] == 9 and rep["counts"]["n_excluded_named_immune"] == 2
    assert set(sub["node_idx"][sub["is_dynamic"]]) == {3, 4, 10, 20} | FORCED
    assert rep["counts"]["n_core"] == 4 and 19 not in sub["node_idx"]


def test_output_set_readout(gdir):
    # backward BFS seeded from readout cells only: DNxx99 (9) is no longer a seed, so the chain
    # 5-6-7-8 that only reaches it drops out of the core
    sub, rep = sg.build_subgraph(gdir, verbose=False, output_set="readout")
    assert rep["rule"]["output_set"] == "readout" and rep["counts"]["n_core"] == 5
    assert set(sub["node_idx"][~sub["is_clamped"]]) == {3, 4, 10, 19, 20} | FORCED
    assert rep["dn_reachability"]["full_graph_4hop"]["n_dn"] == 6      # DN reports still cover all DNs
    with pytest.raises(ValueError):
        sg.build_subgraph(gdir, verbose=False, output_set="bogus")


def test_budget_clamp_fails_loudly_when_forced_set_exceeds_budget(gdir):
    with pytest.raises(RuntimeError, match="budget clamp failed"):
        sg.build_subgraph(gdir, w_min=5.0, max_dynamic=5, verbose=False)


# ----------------------------------------------------------------------------- local CSR/CSC

def test_local_graph_matches_dense_rule(built):
    sub, _ = built
    node_idx = sub["node_idx"]
    assert node_idx.tolist() == NODES_5
    m = node_idx.shape[0]
    a = sp.csr_matrix((sub["weight"], sub["indices_pre"], sub["indptr_post"]), shape=(m, m)).toarray()
    w = dense_w()
    w[w < 5.0] = 0
    w[np.array(sorted(CLAMPED))] = 0                       # rows = post: nothing into clamped
    np.testing.assert_array_equal(a, w[np.ix_(node_idx, node_idx)])
    stored = {(int(node_idx[c]), int(node_idx[r])) for r, c in zip(*np.nonzero(a))}
    assert stored == EDGES_5


def test_csr_csc_invariants(built):
    sub, _ = built
    m, e = sub["node_idx"].shape[0], sub["weight"].shape[0]
    for k in ("indptr_post", "indptr_pre"):
        ip = sub[k]
        assert ip.dtype == np.int64 and ip[0] == 0 and ip[-1] == e and (np.diff(ip) >= 0).all()
    assert sub["indices_pre"].dtype == np.int32 and sub["indices_post"].dtype == np.int32
    assert sub["weight"].dtype == np.float32 and sub["sign"].dtype == np.int8 and sub["type_id"].dtype == np.int32
    assert (sub["indices_pre"] >= 0).all() and (sub["indices_pre"] < m).all()
    assert (sub["weight"] >= sub["w_min"]).all() and float(sub["w_min"]) == 5.0
    # CSC == scipy's transpose of the CSR, and weight_csc = weight[perm]
    csr = sp.csr_matrix((sub["weight"], sub["indices_pre"], sub["indptr_post"]), shape=(m, m))
    csc = csr.tocsc(); csc.sort_indices()
    np.testing.assert_array_equal(csc.indptr, sub["indptr_pre"])
    np.testing.assert_array_equal(csc.indices, sub["indices_post"])
    np.testing.assert_array_equal(csc.data, sub["weight"][sub["perm_csr_to_csc"]])
    assert sorted(sub["perm_csr_to_csc"].tolist()) == list(range(e))
    # no edges into clamped nodes; rows of clamped are empty
    assert (np.diff(sub["indptr_post"])[sub["is_clamped"]] == 0).all()
    assert np.array_equal(sub["is_dynamic"], ~sub["is_clamped"])
    assert not (sub["is_output"] & sub["is_clamped"]).any()
    # sorted within rows (canonical form), no duplicates
    for r in range(m):
        cols = sub["indices_pre"][sub["indptr_post"][r]:sub["indptr_post"][r + 1]]
        assert (np.diff(cols) > 0).all()


def test_node_arrays_match_neurons_table(built, gdir):
    sub, _ = built
    cells = sg.load_cells(gdir / "neurons.parquet")
    ni = sub["node_idx"]
    assert (np.diff(ni) > 0).all()
    np.testing.assert_array_equal(sub["bodyId"], cells["bodyId"][ni])
    np.testing.assert_array_equal(sub["sign"], cells["sign"][ni])
    np.testing.assert_array_equal(sub["type_id"], cells["type_id"][ni])
    assert sub["type_id"][list(ni).index(19)] == -1                 # null type keeps -1
    assert set(ni[sub["is_clamped"]]) == CLAMPED
    assert set(ni[sub["is_output"]]) == {10, 11, 15, 16, 20, 21, 22}


# ----------------------------------------------------------------------------- report

def test_report_counts_and_reachability(built):
    _, rep = built
    c = rep["counts"]
    assert c["n_clamped"] == 3 and c["n_clamped_with_stored_out_edges"] == 2
    assert c["n_core"] == 9 and c["n_dynamic"] == 16 and c["n_nodes"] == 19
    assert c["n_edges"] == 13 and c["n_edges_clamped_to_dynamic"] == 3 and c["n_edges_dynamic_to_dynamic"] == 10
    assert c["n_autapses"] == 1 and c["n_output_cells"] == 7 and c["n_forced_not_in_core"] == 7
    assert c["n_output_superclass"] == 6 and c["n_dn_in_subgraph"] == 5
    w = rep["weight"]
    assert abs(w["stored"] - 112.0) < 1e-6                                 # sum of EDGES_5 weights
    assert abs(w["within_nodes_unthresholded"] - 114.0) < 1e-6             # + (4,11,w=2)
    assert abs(w["frac_within_nodes"] - 112 / 114) < 1e-6
    assert w["frac_of_full"] < 1.0 and 0 < w["frac_of_dynamic_inputs"] <= 1.0
    d = rep["dn_reachability"]
    assert d["full_graph_4hop"] == {"n_dn": 6, "n_reached": 1, "hop_hist": [0, 1, 0, 0], "frac": 1 / 6}
    assert d["subgraph"]["n_dn_in_subgraph"] == 5 and d["subgraph"]["n_reached"] == 1
    r = d["readout"]
    assert r["n_cells"] == 7 and r["n_in_subgraph"] == 7 and r["n_reached"] == 2
    hops = {(x["type"], x["side"]): x["hops"] for x in r["per_cell"]}
    assert hops[("DNa02", "L")] == 2 and hops[("AOTU019", "L")] == 3 and hops[("DNa02", "R")] == -1
    f = rep["forced"]
    assert "LC6" in f["missing_patterns"] and f["hits"]["LPi*"] == {"types": ["LPi34-12"], "n_cells": 1, "n_in_core": 0}
    assert f["hits"]["LC4"]["n_in_core"] == 1


def test_lr_audit(built):
    _, rep = built
    a = rep["lr_audit"]
    flagged = {f["type"] for f in a["flagged"]}
    assert "Tm1" in flagged and "T4a" not in flagged and "DNa02" not in flagged and "MDN" not in flagged
    assert a["per_type"]["<null>"] == {"L": 0, "R": 1, "other": 0}
    assert a["per_type"]["T4a"] == {"L": 1, "R": 1, "other": 0}
    assert a["n_flagged_ge5"] == 0


# ----------------------------------------------------------------------------- readout

def test_readout_resolution(gdir):
    cells, missing = sg.resolve_readout(sg.load_cells(gdir / "neurons.parquet"))
    assert [(c["type"], c["side"]) for c in cells] == [
        ("DNa02", "L"), ("DNa02", "R"), ("MDN", "L"), ("MDN", "R"), ("DNp01", "R"),
        ("AOTU019", "L"), ("pIP10", "R")]
    assert cells[0]["bodyId"] == int(BODY[10]) and cells[0]["role"] == "turn"
    assert {c["role"] for c in cells} == {"turn", "backward", "dodge", "attack_midline", "attack_pursuit"}
    assert missing == [t for t in sg.READOUT_ROLES if t not in {"DNa02", "MDN", "DNp01", "AOTU019", "pIP10"}]
    assert set(sg.READOUT_ROLES.values()) <= {"turn", "forward", "backward", "dodge",
                                              "attack_midline", "attack_pursuit", "unassigned"}


def test_readout_hash_stability(gdir, tmp_path):
    cells, missing = sg.resolve_readout(sg.load_cells(gdir / "neurons.parquet"))
    h = sg.cells_sha256(cells)
    assert h == sg.cells_sha256(json.loads(json.dumps(cells)))               # JSON round trip
    assert h == sg.cells_sha256([dict(reversed(list(c.items()))) for c in cells])  # key order
    assert h != sg.cells_sha256(cells[::-1])                                  # list order matters
    changed = json.loads(json.dumps(cells)); changed[0]["bodyId"] += 1
    assert h != sg.cells_sha256(changed)
    p = tmp_path / "readout.json"
    doc = sg.write_readout(cells, p, missing, "fixture")
    assert doc["sha256"] == h and sg.load_readout(p)["sha256"] == h
    with pytest.raises(FileExistsError):
        sg.write_readout(cells, p, missing, "fixture")
    sg.write_readout(cells, p, missing, "fixture", force=True)
    p.write_text(json.dumps({**doc, "sha256": "0" * 64}))
    with pytest.raises(ValueError):
        sg.load_readout(p)


# ----------------------------------------------------------------------------- CLI / files

def test_cli_end_to_end(gdir, tmp_path):
    rpath = tmp_path / "readout_v1.json"
    assert sg.main(["--graph-dir", str(gdir), "--readout", "--readout-path", str(rpath)]) == 0
    doc = sg.load_readout(rpath)
    assert doc["version"] == 1 and len(doc["cells"]) == 7
    assert sg.main(["--graph-dir", str(gdir), "--name", "t", "--readout-path", str(rpath),
                    "--json-out", str(tmp_path / "rep.json")]) == 0
    sub = sg.load_subgraph(gdir / "subgraph_t.npz")
    keys = {"node_idx", "bodyId", "is_clamped", "is_output", "is_dynamic", "type_id", "sign",
            "indptr_post", "indices_pre", "weight", "indptr_pre", "indices_post", "perm_csr_to_csc",
            "w_min", "report"}
    assert keys <= set(sub)
    assert sub["report"]["readout_sha256"] == doc["sha256"] and sub["is_output"].sum() == 7
    assert json.loads((tmp_path / "rep.json").read_text())["w_min"] == 5.0
    # vocab check runs on the synthetic table too
    assert sg.main(["--vocab", str(gdir / "neurons.parquet"), "--json-out", str(tmp_path / "v.json")]) == 0
    v = json.loads((tmp_path / "v.json").read_text())
    assert v["clamped_types"]["T4a"] == {"n": 2, "L": 1, "R": 1, "other": 0}
    assert "LC6" in v["forced_missing"] and "DNa01" in v["readout_missing"] and v["descending"]["n"] == 6


def test_build_is_deterministic(gdir):
    a, ra = sg.build_subgraph(gdir, verbose=False)
    b, rb = sg.build_subgraph(gdir, verbose=False)
    for k in a:
        if k != "report":
            np.testing.assert_array_equal(a[k], b[k])
    assert ra == rb
