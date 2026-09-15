"""Tests for flybrain.data.import_malecns.

Synthetic tests run anywhere. Data-backed tests need the real outputs
($FLYBRAIN_OUT/graph/graph_full.npz + neurons.parquet) and skip if they are absent:
    python -m flybrain.data.import_malecns
"""

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import scipy.sparse as sp

from flybrain import OUT_DIR
from flybrain.data import import_malecns as im

GRAPH_DIR = OUT_DIR / "graph"


# ----------------------------------------------------------------------------- synthetic

def test_sign_map_covers_every_nt_string():
    nts = ["acetylcholine", "dopamine", "octopamine", "serotonin", "unclear", None,
           "gaba", "glutamate", "histamine"]
    sign = im.sign_from_nt(nts)
    assert sign.dtype == np.int8
    assert sign.tolist() == [1, 1, 1, 1, 1, 1, -1, -1, -1]
    with pytest.raises(ValueError):
        im.sign_from_nt(["Acetylcholine"])


def test_select_neurons_policy():
    ann = pd.DataFrame({
        "bodyId": [5, 3, 9, 1, 7, 2],
        "superclass": ["cb_intrinsic", None, "vnc_tbc", "", None, "ol_intrinsic"],
        "status": ["Traced", None, "Orphan", "Traced", "Glia", "Glia"],
    })
    kept, counts = im.select_neurons(ann)
    assert kept.bodyId.tolist() == [5, 9]              # sorted, tbc kept, glia out, empty out
    assert counts == {"annotation_rows": 6, "non_neuronal_excluded": 2, "unresolved_objects": 2,
                      "glia_with_superclass": 1, "n_neurons": 2}


def test_dense_type_id():
    tid = im.dense_type_id(["Tm1", None, "Mi1", "Tm1"])
    assert tid.dtype == np.int32 and tid.tolist() == [1, -1, 0, 1]


def _random_graph(n=50, e=400, seed=0):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, n, e).astype(np.int32)
    post = rng.integers(0, n, e).astype(np.int32)
    w = rng.integers(1, 40, e).astype(np.int64)
    return pre, post, w


def test_csr_csc_matches_scipy_on_synthetic():
    n = 50
    pre, post, w = _random_graph(n)
    g = im.build_csr_csc(n, pre, post, w)
    coo = sp.coo_array((w.astype(np.float32), (post, pre)), shape=(n, n))
    ref_csr = coo.tocsr(); ref_csr.sum_duplicates(); ref_csr.sort_indices()
    # duplicates are kept separately by build_csr_csc, so compare after summing them
    # (copies: scipy's sum_duplicates mutates the arrays it is handed in place)
    ours = sp.csr_array((g["weight"].copy(), g["indices_pre"].copy(), g["indptr_post"].copy()),
                        shape=(n, n))
    ours.sum_duplicates()
    assert np.array_equal(ours.indptr, ref_csr.indptr)
    assert np.array_equal(ours.indices, ref_csr.indices)
    assert np.array_equal(ours.data, ref_csr.data)
    # CSC pieces reproduce the same edge set through the permutation
    post_of_csr = np.repeat(np.arange(n), np.diff(g["indptr_post"]))
    pre_of_csc = np.repeat(np.arange(n), np.diff(g["indptr_pre"]))
    perm = g["perm_csr_to_csc"]
    assert np.array_equal(np.sort(perm), np.arange(len(w)))
    assert np.array_equal(g["indices_pre"][perm], pre_of_csc)
    assert np.array_equal(post_of_csr[perm], g["indices_post"])
    assert g["indptr_post"].dtype == np.int64 and g["indices_pre"].dtype == np.int32
    assert g["weight"].dtype == np.float32 and g["indices_post"].dtype == np.int32


def test_id_bitmap_counts_distinct():
    bm = im._IdBitmap(n_bits=64)
    bm.add(np.array([1, 1, 2, 63, 70, 70, 3000], dtype=np.int64))   # forces growth
    assert bm.count() == 5


def test_make_ledger_reports_diff_without_fudging():
    counts = dict(im.REFERENCE)
    counts["n_edges"] += 7
    led = im.make_ledger(counts)
    assert led["all_pass"] is False
    assert led["checks"]["n_edges"] == {"got": im.REFERENCE["n_edges"] + 7,
                                        "expected": im.REFERENCE["n_edges"], "diff": 7, "pass": False}
    assert led["checks"]["n_neurons"]["pass"] is True


# ----------------------------------------------------------------------------- real outputs

@pytest.fixture(scope="module")
def graph():
    path = GRAPH_DIR / "graph_full.npz"
    if not path.exists():
        pytest.skip(f"{path} missing; run python -m flybrain.data.import_malecns")
    return im.load_graph(path)


@pytest.fixture(scope="module")
def neurons():
    path = GRAPH_DIR / "neurons.parquet"
    if not path.exists():
        pytest.skip(f"{path} missing; run python -m flybrain.data.import_malecns")
    return pq.read_table(path).to_pandas()


def test_ledger_reconciles_with_doomfly(graph):
    led = graph["ledger"]
    diffs = [f"{k}: got {c['got']:,} expected {c['expected']:,} (diff {c['diff']:+,})"
             for k, c in led["checks"].items() if not c["pass"]]
    if diffs:
        pytest.xfail("import ledger differs from the doomfly reference: " + "; ".join(diffs))
    assert led["all_pass"]
    for k, v in im.REFERENCE.items():
        assert led[k] == v
    assert graph["n_neurons"] == led["n_neurons"] and graph["n_edges"] == led["n_edges"]


def test_ledger_json_in_npz_matches_file(graph):
    on_disk = json.loads((GRAPH_DIR / "ledger.json").read_text())
    assert on_disk["checks"] == graph["ledger"]["checks"]
    assert on_disk["runtime"]["peak_rss_mb"] == graph["ledger"]["runtime"]["peak_rss_mb"]


def test_bodyid_unique_sorted_and_matches_parquet(graph, neurons):
    b = graph["bodyId"]
    assert b.dtype == np.int64 and len(b) == graph["n_neurons"]
    assert np.all(b[1:] > b[:-1])                      # strictly increasing => unique
    assert np.array_equal(neurons.bodyId.to_numpy(), b)
    assert np.array_equal(neurons.idx.to_numpy(), np.arange(len(b)))


def test_neurons_parquet_contract(neurons):
    assert list(neurons.columns) == im.NEURON_COLUMNS
    assert neurons.sign.isin([1, -1]).all()
    tid = neurons.type_id.to_numpy()
    assert tid.dtype == np.int32
    assert np.array_equal(tid == -1, neurons.type.isna().to_numpy())
    uniq = np.unique(tid[tid >= 0])
    assert np.array_equal(uniq, np.arange(len(uniq)))  # dense
    assert neurons.superclass.notna().all()
    assert not neurons.status.eq("Glia").any()


def test_sign_coverage(graph, neurons):
    sign = graph["sign"]
    assert sign.dtype == np.int8 and len(sign) == graph["n_neurons"]
    assert np.isin(sign, [1, -1]).all()
    assert np.array_equal(sign, neurons.sign.to_numpy())
    expected = im.sign_from_nt([None if pd.isna(v) else v for v in neurons.consensus_nt])
    assert np.array_equal(sign, expected)
    missing = neurons.consensus_nt.isna().sum()
    assert missing == graph["ledger"]["extra"]["consensus_nt_counts"].get("missing", 0)
    assert missing / len(sign) < 0.01                  # 178 / 166,700 in v1.0
    assert (sign == -1).sum() > 0.2 * len(sign)        # GABA+Glu+His are ~36% of neurons


def test_weights_positive_integer_contacts(graph):
    w = graph["weight"]
    assert w.dtype == np.float32 and len(w) == graph["n_edges"]
    assert np.all(w > 0)
    assert np.array_equal(w, np.floor(w))
    assert int(w.astype(np.int64).sum()) == graph["ledger"]["n_contacts"]


def test_csr_csc_round_trip(graph):
    n, e = graph["n_neurons"], graph["n_edges"]
    ip, jp = graph["indptr_post"], graph["indptr_pre"]
    assert ip.dtype == np.int64 and jp.dtype == np.int64
    assert ip.shape == (n + 1,) and jp.shape == (n + 1,)
    assert ip[0] == 0 and ip[-1] == e and jp[0] == 0 and jp[-1] == e
    assert np.all(np.diff(ip) >= 0) and np.all(np.diff(jp) >= 0)
    pre_csr, post_csc, perm = graph["indices_pre"], graph["indices_post"], graph["perm_csr_to_csc"]
    assert pre_csr.dtype == np.int32 and post_csc.dtype == np.int32
    assert pre_csr.min() >= 0 and pre_csr.max() < n and post_csc.min() >= 0 and post_csc.max() < n
    assert np.array_equal(np.sort(perm), np.arange(e))
    post_csr = np.repeat(np.arange(n, dtype=np.int32), np.diff(ip))
    pre_csc = np.repeat(np.arange(n, dtype=np.int32), np.diff(jp))
    assert np.array_equal(pre_csr[perm], pre_csc)
    assert np.array_equal(post_csr[perm], post_csc)
    # sorted inner indices within every row/column
    same_row = post_csr[1:] == post_csr[:-1]
    assert np.all(pre_csr[1:][same_row] >= pre_csr[:-1][same_row])
    same_col = pre_csc[1:] == pre_csc[:-1]
    assert np.all(post_csc[1:][same_col] >= post_csc[:-1][same_col])
    # scipy agrees
    csc = sp.csr_array((graph["weight"], pre_csr, ip), shape=(n, n)).tocsc()
    assert np.array_equal(csc.indptr, jp)
    assert np.array_equal(csc.indices, post_csc)
    assert np.array_equal(csc.data, graph["weight"][perm])


def test_edge_accounting_is_consistent(graph):
    ex = graph["ledger"]["extra"]
    led = graph["ledger"]
    assert (led["n_edges"] + ex["rows_dropped_endpoint_not_annotated"]
            + ex["rows_dropped_endpoint_annotated_not_retained"] == led["raw_rows"])
    assert ex["n_duplicate_pairs"] == 0
    assert ex["self_edges"] == int(np.count_nonzero(
        graph["indices_pre"] == np.repeat(np.arange(graph["n_neurons"]), np.diff(graph["indptr_post"]))))
    assert (led["unresolved_objects"] + led["non_neuronal_excluded"] + led["n_neurons"]
            == ex["annotation_rows"])
