"""Tests for flybrain.data.nulls on synthetic subgraphs (run on the server)."""

import json
import time

import numpy as np
import pytest

from flybrain.data import nulls
from flybrain.data.nulls import (KINDS, NODE_KEYS, NULL_KEYS, REQUIRED_INVARIANTS, check_null,
                                 check_subgraph, edge_list, failed_invariants, load_graph,
                                 make_null, save_graph, synthetic_subgraph)


def _edges(g):
    pre, post, _ = edge_list(g)
    return set(zip(pre.tolist(), post.tolist()))


@pytest.fixture(scope="module")
def g100k():
    return synthetic_subgraph(n_nodes=4000, n_edges=100_000, seed=1)


@pytest.fixture(scope="module")
def g10k():
    return synthetic_subgraph(n_nodes=800, n_edges=10_000, seed=2)


def test_synthetic_is_a_valid_subgraph(g100k):
    check_subgraph(g100k)
    checks = check_null(g100k, g100k)
    assert failed_invariants("N8", checks) == []
    assert checks["frac_edges_changed"] == 0.0


def test_hash_set_matches_python_set():
    """Insert/find/remove with backward-shift deletion against a reference set."""
    rng = np.random.default_rng(0)
    table, mask = nulls._new_table(64)  # 128 slots, keys collide a lot
    ref = set()
    for _ in range(5000):
        key = np.int64(rng.integers(0, 200))
        op = rng.random()
        present = nulls._find(table, key, mask) >= 0
        assert present == (int(key) in ref)
        if op < 0.5 and not present and len(ref) < 60:
            nulls._insert(table, key, mask)
            ref.add(int(key))
        elif op >= 0.5 and present:
            assert nulls._remove(table, key, mask)
            ref.remove(int(key))
    for k in range(200):
        assert (nulls._find(table, np.int64(k), mask) >= 0) == (k in ref)
    assert int((table != nulls._EMPTY).sum()) == len(ref)


def test_n1_invariants_100k(g100k):
    null = make_null(g100k, "N1", seed=0)
    checks = check_null(g100k, null)
    assert failed_invariants("N1", checks) == [], failed_invariants("N1", checks)
    assert null["null_kind"] == KINDS["N1"]
    assert null["attempted_swaps"] == 20 * 100_000
    assert 0 < null["achieved_swaps"] <= null["attempted_swaps"]
    assert null["frac_edges_changed"] > 0.9
    assert checks["frac_edges_changed"] == pytest.approx(null["frac_edges_changed"])
    assert checks["n_self_loops_null"] == 0
    # weights follow the presynaptic endpoint: per-node out-strength is preserved too
    p0, _, w0 = edge_list(g100k)
    p1, _, w1 = edge_list(null)
    n = len(g100k["sign"])
    assert np.allclose(np.bincount(p0, w0, n), np.bincount(p1, w1, n))
    for k in NODE_KEYS:
        assert null[k].dtype == g100k[k].dtype
    for k in ("indices_pre", "indices_post", "perm_csr_to_csc", "weight", "indptr_post"):
        assert null[k].dtype == g100k[k].dtype


def test_n1_weight_carriers(g10k):
    n = len(g10k["sign"])
    _, q0, w0 = edge_list(g10k)
    post_null = make_null(g10k, "N1", seed=3, weight_carrier="post")
    _, q1, w1 = edge_list(post_null)
    assert failed_invariants("N1", check_null(g10k, post_null)) == []
    assert np.allclose(np.bincount(q0, w0, n), np.bincount(q1, w1, n))  # in-strength kept
    shuf = make_null(g10k, "N1", seed=3, weight_carrier="shuffle")
    assert failed_invariants("N1", check_null(g10k, shuf)) == []
    assert json.loads(shuf["null_params"])["weight_carrier"] == "shuffle"


def test_n1_deterministic_and_seed_sensitive(g10k):
    a = make_null(g10k, "N1", seed=7)
    b = make_null(g10k, "N1", seed=7)
    c = make_null(g10k, "N1", seed=8)
    for k in ("indices_pre", "weight", "indices_post", "perm_csr_to_csc", "indptr_post"):
        assert np.array_equal(a[k], b[k])
    assert a["achieved_swaps"] == b["achieved_swaps"]
    assert _edges(a) != _edges(c)


def test_n1_freezes_existing_autapses():
    g = synthetic_subgraph(n_nodes=600, n_edges=12_000, seed=5, n_self_loops=25)
    check_subgraph(g)
    null = make_null(g, "N1", seed=0)
    checks = check_null(g, null)
    assert failed_invariants("N1", checks) == []
    assert checks["n_self_loops_orig"] == 25 and checks["n_self_loops_null"] == 25
    pre, post, _ = edge_list(g)
    loops = set(pre[pre == post].tolist())
    p1, q1, _ = edge_list(null)
    assert set(p1[p1 == q1].tolist()) == loops


def test_n8_invariants(g100k):
    null = make_null(g100k, "N8", seed=0)
    checks = check_null(g100k, null)
    assert failed_invariants("N8", checks) == []
    assert null["frac_edges_changed"] == 0.0
    assert null["achieved_swaps"] == 0 and null["attempted_swaps"] == 0
    assert np.array_equal(null["indices_pre"], g100k["indices_pre"])
    assert not np.array_equal(null["weight"], g100k["weight"])
    assert np.array_equal(np.sort(null["weight"]), np.sort(g100k["weight"]))


def test_n2a_invariants(g100k):
    null = make_null(g100k, "N2a", seed=0)
    checks = check_null(g100k, null)
    assert failed_invariants("N2a", checks) == []
    assert null["frac_edges_changed"] > 0.5
    assert not checks["in_degree_preserved"] or not checks["out_degree_preserved"]
    # per-block weight multisets are kept, not only the global one
    t = nulls._dense_types(g100k["type_id"])
    nt = t.max() + 1
    for g in (g100k, null):
        pre, post, w = edge_list(g)
        key = t[pre] * nt + t[post]
        g["_blockw"] = np.lexsort((w, key))
        g["_blockw"] = (key[g["_blockw"]], w[g["_blockw"]])
    assert np.array_equal(g100k["_blockw"][0], null["_blockw"][0])
    assert np.array_equal(g100k["_blockw"][1], null["_blockw"][1])
    del g100k["_blockw"], null["_blockw"]
    # untyped nodes are singleton blocks: their mutual edges cannot move
    pre, post, _ = edge_list(g100k)
    untyped = g100k["type_id"] < 0
    both = untyped[pre] & untyped[post]
    assert set(zip(pre[both].tolist(), post[both].tolist())) <= _edges(null)


def test_n4_invariants(g100k):
    null = make_null(g100k, "N4", seed=0)
    checks = check_null(g100k, null)
    assert failed_invariants("N4", checks) == []
    assert null["frac_edges_changed"] > 0.9
    assert json.loads(null["null_params"])["n_type_blocks"] == 1


def test_no_self_loops_created_when_none_exist(g10k):
    for kind in KINDS:
        null = make_null(g10k, kind, seed=1)
        assert check_null(g10k, null)["n_self_loops_null"] == 0


def test_rejects_bad_subgraph(g10k):
    bad = dict(g10k)
    bad["indices_pre"] = g10k["indices_pre"].copy()
    bad["indices_pre"][1] = bad["indices_pre"][0]  # duplicate edge within a row
    with pytest.raises(ValueError):
        check_subgraph(bad)
    with pytest.raises(ValueError):
        make_null(g10k, "N9", seed=0)


def test_npz_roundtrip_and_cli(tmp_path, g10k):
    src = tmp_path / "subgraph_test.npz"
    save_graph(src, g10k)
    loaded = load_graph(src)
    assert set(loaded) == set(g10k)
    assert loaded["report"] == g10k["report"] and loaded["w_min"] == g10k["w_min"]
    out = tmp_path / "nulls"
    rc = nulls.main(["--graph", str(src), "--kind", "N1", "--seed", "0", "1", "--out", str(out)])
    assert rc == 0
    for seed in (0, 1):
        null = load_graph(out / f"null_N1_{seed}.npz")
        for k in list(g10k) + list(NULL_KEYS):
            assert k in null, k
        assert null["seed"] == seed and null["null_kind"] == KINDS["N1"]
        assert failed_invariants("N1", check_null(loaded, null)) == []
        assert null["elapsed_s"] > 0 and null["peak_rss_mb"] > 0
    # a null file is itself a valid input (keys are replaced, not duplicated)
    again = make_null(load_graph(out / "null_N1_0.npz"), "N8", seed=0)
    assert again["null_kind"] == KINDS["N8"] and set(again) == set(null)


def test_swap_rate_1m_edges(capsys):
    """Measured N1 throughput on a 1M-edge synthetic graph (numbers printed with -s)."""
    g = synthetic_subgraph(n_nodes=25_000, n_edges=1_000_000, seed=11)
    t0 = time.perf_counter()
    null = make_null(g, "N1", seed=0)
    total = time.perf_counter() - t0
    p = json.loads(null["null_params"])
    rate = {"n_edges": 1_000_000, "attempted_swaps": null["attempted_swaps"],
            "achieved_swaps": null["achieved_swaps"], "swap_kernel_s": p["swap_kernel_s"],
            "attempts_per_s": null["attempted_swaps"] / p["swap_kernel_s"],
            "swaps_per_s": null["achieved_swaps"] / p["swap_kernel_s"],
            "make_null_s": total, "frac_edges_changed": null["frac_edges_changed"],
            "peak_rss_mb": null["peak_rss_mb"]}
    with capsys.disabled():
        print("\nSWAP_RATE " + json.dumps(rate))
    checks = check_null(g, null)
    assert failed_invariants("N1", checks) == []
    assert null["frac_edges_changed"] > 0.9
    assert rate["attempts_per_s"] > 1e5
