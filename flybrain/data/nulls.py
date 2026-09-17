"""Null-graph generators (CONTRACTS.md "Null graphs").

Reads any subgraph_*.npz and writes null_<kind>_<seed>.npz with the same keys plus
null_kind, seed, achieved_swaps, attempted_swaps, frac_edges_changed, elapsed_s, peak_rss_mb.

  N1   degree+weight-preserving directed double-edge swap (a->b, c->d) -> (a->d, c->b).
       Every node keeps its in/out degree, the weight multiset is exact (each weight travels
       with its presynaptic endpoint by default), no self-edge or duplicate edge is ever
       created, existing autapses are frozen. Achieved swaps are logged, not the attempt cap.
  N8   mask-only: identical edges, magnitudes permuted.
  N2a  type-block shuffle: type->type edge counts (and per-block weight multisets) kept,
       individual (pre, post) pairings resampled uniformly within each block.
  N4   Erdos-Renyi: same N, E and weight multiset; pairs uniform over (all pre) x (dynamic post).

Sign lives on the presynaptic node, so every null keeps it trivially. CSR rows = post,
cols = pre; per-edge arrays follow CSR order. Everything is deterministic given the seed.

CLI:  python -m flybrain.data.nulls --graph <npz> --kind N1 --seed 0 [1 2 ...] --out <dir>
      python -m flybrain.data.nulls --bench 1000000          # swap rate on a synthetic graph
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit

from flybrain import OUT_DIR

KINDS = {
    "N1": "N1_degree_weight_rewire",
    "N8": "N8_mask_only",
    "N2a": "N2a_type_block",
    "N4": "N4_erdos_renyi",
}
NODE_KEYS = ("node_idx", "bodyId", "is_clamped", "is_output", "is_dynamic", "type_id", "sign")
CSR_KEYS = ("indptr_post", "indices_pre", "weight", "indptr_pre", "indices_post", "perm_csr_to_csc")
NULL_KEYS = ("null_kind", "seed", "achieved_swaps", "attempted_swaps", "frac_edges_changed",
             "elapsed_s", "peak_rss_mb", "null_params")
WEIGHT_CARRIERS = ("pre", "post", "shuffle")

_COMMON = ("same_nodes", "sign_identical", "n_edges_equal", "weight_multiset_preserved",
           "no_edges_into_clamped", "no_new_self_loops", "no_duplicate_edges", "csc_consistent")
_DEGREE = ("in_degree_preserved", "out_degree_preserved", "clamped_out_degree_preserved",
           "output_in_degree_preserved")
REQUIRED_INVARIANTS = {
    "N1": _COMMON + _DEGREE,
    "N8": _COMMON + _DEGREE + ("edges_identical",),
    "N2a": _COMMON + ("type_block_counts_preserved",),
    "N4": _COMMON,
}


# ----------------------------------------------------------------------------- hash set
# Open-addressing (linear probing) set of int64 edge keys `pre * n + post`, with
# backward-shift deletion so long runs of swaps never accumulate tombstones.

_EMPTY = np.int64(-1)
_C1 = np.uint64(0xBF58476D1CE4E5B9)
_C2 = np.uint64(0x94D049BB133111EB)


@njit(cache=True, inline="always")
def _slot(key, mask):
    """splitmix64 finaliser of a non-negative int64 key, masked to a table slot."""
    z = np.uint64(key)
    z ^= z >> np.uint64(30)
    z *= _C1
    z ^= z >> np.uint64(27)
    z *= _C2
    z ^= z >> np.uint64(31)
    return np.int64(z & np.uint64(mask))


@njit(cache=True)
def _find(table, key, mask):
    """Slot holding `key`, or -1."""
    h = _slot(key, mask)
    while True:
        k = table[h]
        if k == key:
            return h
        if k == _EMPTY:
            return -1
        h = (h + 1) & mask


@njit(cache=True)
def _insert(table, key, mask):
    """Insert a key known to be absent."""
    h = _slot(key, mask)
    while table[h] != _EMPTY:
        h = (h + 1) & mask
    table[h] = key


@njit(cache=True)
def _remove(table, key, mask):
    """Remove a key (backward-shift deletion). Returns False if it was absent."""
    i = _find(table, key, mask)
    if i < 0:
        return False
    table[i] = _EMPTY
    j = i
    while True:
        j = (j + 1) & mask
        k = table[j]
        if k == _EMPTY:
            return True
        h = _slot(k, mask)
        if i <= j:
            stays = i < h <= j
        else:
            stays = h > i or h <= j
        if stays:
            continue
        table[i] = k
        table[j] = _EMPTY
        i = j


@njit(cache=True)
def _fill_table(table, src, dst, n, mask):
    """Insert every edge; returns the index of the first duplicate edge, or -1."""
    for e in range(src.shape[0]):
        key = src[e] * n + dst[e]
        if _find(table, key, mask) >= 0:
            return e
        _insert(table, key, mask)
    return -1


def _new_table(n_edges: int) -> tuple[np.ndarray, np.int64]:
    """Empty table with load factor <= 0.5 for n_edges keys; returns (table, mask)."""
    size = 1 << max(4, int(np.ceil(np.log2(max(2 * n_edges, 1)))))
    return np.full(size, _EMPTY, dtype=np.int64), np.int64(size - 1)


def _edge_table(src: np.ndarray, dst: np.ndarray, n: int) -> tuple[np.ndarray, np.int64]:
    table, mask = _new_table(len(src))
    dup = _fill_table(table, src, dst, np.int64(n), mask)
    if dup >= 0:
        raise ValueError(f"duplicate edge ({src[dup]} -> {dst[dup]}) at CSR position {dup}")
    return table, mask


# ----------------------------------------------------------------------------- kernels

@njit(cache=True)
def _swap_kernel(src, dst, w, n, table, mask, n_attempts, seed, carry_post):
    """Directed double-edge swaps in place. Returns the number of achieved swaps.

    Weights stay on their record (i.e. with the presynaptic endpoint); with carry_post
    they are exchanged too so each weight follows its postsynaptic endpoint instead.
    """
    np.random.seed(seed)
    m = src.shape[0]
    achieved = 0
    for _ in range(n_attempts):
        i = np.random.randint(0, m)
        j = np.random.randint(0, m)
        a = src[i]
        b = dst[i]
        c = src[j]
        d = dst[j]
        if a == c or b == d:  # same edge, same pre or same post: the swap is a no-op
            continue
        if a == b or c == d:  # existing autapses are frozen
            continue
        if a == d or c == b:  # would create an autapse
            continue
        k_ad = a * n + d
        k_cb = c * n + b
        if _find(table, k_ad, mask) >= 0 or _find(table, k_cb, mask) >= 0:
            continue
        _remove(table, a * n + b, mask)
        _remove(table, c * n + d, mask)
        _insert(table, k_ad, mask)
        _insert(table, k_cb, mask)
        dst[i] = d
        dst[j] = b
        if carry_post:
            t = w[i]
            w[i] = w[j]
            w[j] = t
        achieved += 1
    return achieved


@njit(cache=True)
def _block_sample_kernel(bpre, bpost, pre_members, pre_off, post_members, post_off,
                         n, allow_self, table, mask, seed):
    """For edge e draw a fresh distinct pair uniformly from block (bpre[e], bpost[e]).

    Block t's candidate pre nodes are pre_members[pre_off[t]:pre_off[t+1]], likewise post.
    Rejection sampling against `table` (must start empty); the caller guarantees every
    block has enough distinct pairs, so the loop terminates.
    """
    np.random.seed(seed)
    m = bpre.shape[0]
    new_pre = np.empty(m, np.int64)
    new_post = np.empty(m, np.int64)
    for e in range(m):
        p0 = pre_off[bpre[e]]
        pn = pre_off[bpre[e] + 1] - p0
        q0 = post_off[bpost[e]]
        qn = post_off[bpost[e] + 1] - q0
        while True:
            a = pre_members[p0 + np.random.randint(0, pn)]
            b = post_members[q0 + np.random.randint(0, qn)]
            if a == b and not allow_self:
                continue
            key = a * n + b
            if _find(table, key, mask) >= 0:
                continue
            _insert(table, key, mask)
            new_pre[e] = a
            new_post[e] = b
            break
    return new_pre, new_post


# ----------------------------------------------------------------------------- graph I/O

def load_graph(path: str | Path) -> dict:
    """Load a subgraph_*/null_*.npz into a dict (0-d arrays become Python scalars/str)."""
    with np.load(path, allow_pickle=False) as f:
        return {k: (f[k].item() if f[k].ndim == 0 else f[k]) for k in f.files}


def save_graph(path: str | Path, g: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **g)


def edge_list(g: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pre, post, weight) in CSR order, int64/int64/float32."""
    n = len(g["sign"])
    post = np.repeat(np.arange(n, dtype=np.int64), np.diff(g["indptr_post"]))
    return g["indices_pre"].astype(np.int64), post, g["weight"].astype(np.float32)


def build_csr_csc(n: int, pre: np.ndarray, post: np.ndarray, w: np.ndarray, like: dict) -> dict:
    """CSR (rows = post, sorted by (post, pre)) and CSC over an edge list; dtypes copied from `like`."""
    order = np.lexsort((pre, post))
    pre_s, post_s, w_s = pre[order], post[order], w[order]
    perm = np.lexsort((post_s, pre_s))
    idx_dtype = like["indices_pre"].dtype
    return {
        "indptr_post": _offsets(np.bincount(post_s, minlength=n)),
        "indices_pre": pre_s.astype(idx_dtype),
        "weight": w_s.astype(np.float32),
        "indptr_pre": _offsets(np.bincount(pre_s, minlength=n)),
        "indices_post": post_s[perm].astype(like["indices_post"].dtype),
        "perm_csr_to_csc": perm.astype(like["perm_csr_to_csc"].dtype),
    }


def _offsets(counts: np.ndarray) -> np.ndarray:
    out = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=out[1:])
    return out


def check_subgraph(g: dict) -> None:
    """Raise if `g` violates the subgraph contract this module relies on."""
    for k in NODE_KEYS + CSR_KEYS:
        if k not in g:
            raise KeyError(f"subgraph is missing key {k!r}")
    n = len(g["sign"])
    pre, post, w = edge_list(g)
    e = len(pre)
    if g["indptr_post"].shape != (n + 1,) or g["indptr_pre"].shape != (n + 1,):
        raise ValueError("indptr length != n_nodes + 1")
    if len(w) != e or len(g["indices_post"]) != e or len(g["perm_csr_to_csc"]) != e:
        raise ValueError("per-edge arrays disagree on the number of edges")
    if not np.array_equal(g["is_dynamic"], ~g["is_clamped"]):
        raise ValueError("is_dynamic != ~is_clamped")
    if not np.isin(g["sign"], (-1, 1)).all():
        raise ValueError("sign must be +1/-1")
    if e and (pre.min() < 0 or pre.max() >= n):
        raise ValueError("indices_pre out of range")
    if g["is_clamped"][post].any():
        raise ValueError("stored edges into clamped nodes (contract: they are dropped)")
    if not (w > 0).all():
        raise ValueError("weights must be > 0")
    if np.unique(post * n + pre).size != e:
        raise ValueError("duplicate (pre, post) edges")


# ----------------------------------------------------------------------------- nulls

def rewire_n1(g: dict, seed: int, swap_factor: float = 20.0, weight_carrier: str = "pre"):
    """N1: degree+weight-preserving double-edge swap. Returns (pre, post, w, stats)."""
    if weight_carrier not in WEIGHT_CARRIERS:
        raise ValueError(f"weight_carrier must be one of {WEIGHT_CARRIERS}")
    n = len(g["sign"])
    src, dst, w = edge_list(g)
    src, dst, w = src.copy(), dst.copy(), w.copy()
    table, mask = _edge_table(src, dst, n)
    attempts = int(round(swap_factor * len(src)))
    t0 = time.perf_counter()
    achieved = _swap_kernel(src, dst, w, np.int64(n), table, mask, attempts,
                            seed & 0xFFFFFFFF, weight_carrier == "post")
    kernel_s = time.perf_counter() - t0
    if weight_carrier == "shuffle":
        w = w[np.random.default_rng(seed).permutation(len(w))]
    stats = {"achieved_swaps": int(achieved), "attempted_swaps": int(attempts),
             "swap_kernel_s": kernel_s, "weight_carrier": weight_carrier,
             "swap_factor": swap_factor}
    return src, dst, w, stats


def shuffle_n8(g: dict, seed: int):
    """N8: same edges, magnitudes permuted."""
    pre, post, w = edge_list(g)
    return pre, post, w[np.random.default_rng(seed).permutation(len(w))]


def _dense_types(type_id: np.ndarray) -> np.ndarray:
    """Dense type labels; every untyped node (-1) becomes its own singleton type."""
    t = type_id.astype(np.int64)
    typed = t >= 0
    out = np.empty_like(t)
    n_typed = 0
    if typed.any():
        _, out[typed] = np.unique(t[typed], return_inverse=True)
        n_typed = int(out[typed].max()) + 1
    out[~typed] = n_typed + np.arange(int((~typed).sum()))
    return out


def resample_blocks(g: dict, seed: int, by_type: bool):
    """N2a (by_type) / N4 (single block): resample pairings, keep per-block weight multisets."""
    n = len(g["sign"])
    pre, post, w = edge_list(g)
    allow_self = bool((pre == post).any())
    dyn = np.flatnonzero(g["is_dynamic"]).astype(np.int64)
    t = _dense_types(g["type_id"]) if by_type else np.zeros(n, dtype=np.int64)
    n_types = int(t.max()) + 1 if n else 1
    pre_members = np.argsort(t, kind="stable").astype(np.int64)
    pre_off = _offsets(np.bincount(t, minlength=n_types))
    post_members = dyn[np.argsort(t[dyn], kind="stable")]
    post_off = _offsets(np.bincount(t[dyn], minlength=n_types))

    bpre, bpost = t[pre], t[post]
    bkey = bpre * n_types + bpost
    blocks, counts = np.unique(bkey, return_counts=True)
    tp, tq = blocks // n_types, blocks % n_types
    capacity = np.diff(pre_off)[tp] * np.diff(post_off)[tq]
    if not allow_self:
        capacity -= np.where(tp == tq, np.diff(post_off)[tq], 0)
    if (counts > capacity).any():
        raise ValueError("a type block has more edges than distinct candidate pairs")

    table, mask = _new_table(len(pre))
    new_pre, new_post = _block_sample_kernel(bpre, bpost, pre_members, pre_off, post_members,
                                            post_off, np.int64(n), allow_self, table, mask,
                                            seed & 0xFFFFFFFF)
    # weights: random permutation within each block (one block == global permutation)
    rng = np.random.default_rng(seed)
    grouped = np.argsort(bkey, kind="stable")
    shuffled = np.lexsort((rng.random(len(w)), bkey))
    new_w = np.empty_like(w)
    new_w[grouped] = w[shuffled]
    return new_pre, new_post, new_w, {"n_type_blocks": int(len(blocks)), "n_types": int(n_types)}


def frac_edges_changed(orig: dict, null: dict) -> float:
    """Fraction of the original (pre, post) pairs that no longer exist."""
    n = len(orig["sign"])
    p0, q0, _ = edge_list(orig)
    p1, q1, _ = edge_list(null)
    if len(p0) == 0:
        return 0.0
    kept = np.isin(q0 * n + p0, q1 * n + p1, assume_unique=True).mean()
    return float(1.0 - kept)


def peak_rss_mb() -> float:
    """Process peak resident set size in MB (Linux ru_maxrss; psutil fallback elsewhere)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        import psutil
        return psutil.Process().memory_info().peak_wset / 2**20


_warm = False


def _warmup() -> None:
    """Compile (or load cached) kernels on a toy graph so timings exclude JIT."""
    global _warm
    if _warm:
        return
    g = synthetic_subgraph(n_nodes=24, n_edges=60, seed=0)
    rewire_n1(g, 0, swap_factor=2.0)
    resample_blocks(g, 0, by_type=True)
    _warm = True


def make_null(g: dict, kind: str, seed: int, swap_factor: float = 20.0,
              weight_carrier: str = "pre") -> dict:
    """Build null graph `kind` from subgraph dict `g`; returns the null dict (contract keys)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {list(KINDS)}")
    check_subgraph(g)
    _warmup()
    n = len(g["sign"])
    t0 = time.perf_counter()
    stats: dict = {}
    if kind == "N1":
        pre, post, w, stats = rewire_n1(g, seed, swap_factor, weight_carrier)
    elif kind == "N8":
        pre, post, w = shuffle_n8(g, seed)
    elif kind == "N2a":
        pre, post, w, stats = resample_blocks(g, seed, by_type=True)
    else:
        pre, post, w, stats = resample_blocks(g, seed, by_type=False)
    null = {k: v for k, v in g.items() if k not in CSR_KEYS and k not in NULL_KEYS}
    null.update(build_csr_csc(n, pre, post, w, like=g))
    frac = frac_edges_changed(g, null)
    elapsed = time.perf_counter() - t0
    null.update({
        "null_kind": KINDS[kind],
        "seed": int(seed),
        "achieved_swaps": int(stats.get("achieved_swaps", 0)),
        "attempted_swaps": int(stats.get("attempted_swaps", 0)),
        "frac_edges_changed": float(frac),
        "elapsed_s": float(elapsed),
        "peak_rss_mb": float(peak_rss_mb()),
        "null_params": json.dumps({"kind": kind, "n_nodes": int(n), "n_edges": int(len(pre)),
                                   "n_self_loops": int((pre == post).sum()), **stats}),
    })
    return null


# ----------------------------------------------------------------------------- checks

def _csc_consistent(g: dict) -> bool:
    n = len(g["sign"])
    pre, post, _ = edge_list(g)
    perm = g["perm_csr_to_csc"].astype(np.int64)
    if len(perm) != len(pre) or (len(perm) and np.unique(perm).size != len(perm)):
        return False
    pre_c = pre[perm]
    return (np.array_equal(g["indptr_pre"], _offsets(np.bincount(pre_c, minlength=n)))
            and np.array_equal(pre_c, np.repeat(np.arange(n), np.diff(g["indptr_pre"])))
            and np.array_equal(g["indices_post"].astype(np.int64), post[perm]))


def check_null(orig: dict, null: dict) -> dict:
    """Every contract invariant as a dict of booleans/numbers (see REQUIRED_INVARIANTS)."""
    n = len(orig["sign"])
    p0, q0, w0 = edge_list(orig)
    p1, q1, w1 = edge_list(null)
    k0, k1 = q0 * n + p0, q1 * n + p1
    indeg0, indeg1 = np.bincount(q0, minlength=n), np.bincount(q1, minlength=n)
    outdeg0, outdeg1 = np.bincount(p0, minlength=n), np.bincount(p1, minlength=n)
    cl, op = orig["is_clamped"], orig["is_output"]
    t = _dense_types(orig["type_id"])
    nt = int(t.max()) + 1 if n else 1
    out = {
        "same_nodes": all(np.array_equal(orig[k], null[k]) for k in NODE_KEYS),
        "sign_identical": np.array_equal(orig["sign"], null["sign"]),
        "n_edges_equal": len(k0) == len(k1),
        "weight_multiset_preserved": len(w0) == len(w1) and np.array_equal(np.sort(w0), np.sort(w1)),
        "in_degree_preserved": np.array_equal(indeg0, indeg1),
        "out_degree_preserved": np.array_equal(outdeg0, outdeg1),
        "clamped_out_degree_preserved": np.array_equal(outdeg0[cl], outdeg1[cl]),
        "output_in_degree_preserved": np.array_equal(indeg0[op], indeg1[op]),
        "no_edges_into_clamped": not cl[q1].any(),
        "n_self_loops_orig": int((p0 == q0).sum()),
        "n_self_loops_null": int((p1 == q1).sum()),
        "no_duplicate_edges": np.unique(k1).size == len(k1),
        "csc_consistent": _csc_consistent(null),
        "edges_identical": np.array_equal(np.sort(k0), np.sort(k1)),
        "type_block_counts_preserved": np.array_equal(np.sort(t[p0] * nt + t[q0]),
                                                      np.sort(t[p1] * nt + t[q1])),
        "frac_edges_changed": frac_edges_changed(orig, null),
    }
    out["no_new_self_loops"] = out["n_self_loops_null"] <= out["n_self_loops_orig"]
    return {k: (bool(v) if isinstance(v, np.bool_) else v) for k, v in out.items()}


def failed_invariants(kind: str, checks: dict) -> list[str]:
    return [k for k in REQUIRED_INVARIANTS[kind] if not checks[k]]


# ----------------------------------------------------------------------------- synthetic

def synthetic_subgraph(n_nodes: int, n_edges: int, n_clamped: int | None = None,
                       n_output: int | None = None, n_types: int | None = None, seed: int = 0,
                       n_self_loops: int = 0, w_min: float = 5.0) -> dict:
    """Random subgraph in the contract format (pre: any node; post: dynamic nodes only)."""
    rng = np.random.default_rng(seed)
    n_clamped = n_nodes // 3 if n_clamped is None else n_clamped
    n_output = max(1, n_nodes // 50) if n_output is None else n_output
    n_types = max(1, n_nodes // 10) if n_types is None else n_types
    perm = rng.permutation(n_nodes)
    is_clamped = np.zeros(n_nodes, dtype=bool)
    is_clamped[perm[:n_clamped]] = True
    dyn = np.sort(perm[n_clamped:]).astype(np.int64)
    is_output = np.zeros(n_nodes, dtype=bool)
    is_output[rng.choice(dyn, size=min(n_output, len(dyn)), replace=False)] = True
    type_id = rng.integers(0, n_types, n_nodes).astype(np.int32)
    type_id[rng.random(n_nodes) < 0.05] = -1
    sign = np.where(rng.random(n_nodes) < 0.7, 1, -1).astype(np.int8)

    n_plain = n_edges - n_self_loops
    if n_plain > n_nodes * len(dyn) - len(dyn) or n_self_loops > len(dyn):
        raise ValueError("too many edges requested for this node count")
    z = np.zeros(n_plain, dtype=np.int64)
    table, mask = _new_table(n_plain)
    pre, post = _block_sample_kernel(z, z, np.arange(n_nodes, dtype=np.int64),
                                     np.array([0, n_nodes], dtype=np.int64), dyn,
                                     np.array([0, len(dyn)], dtype=np.int64), np.int64(n_nodes),
                                     False, table, mask, int(rng.integers(0, 2**31)))
    if n_self_loops:
        loops = rng.choice(dyn, size=n_self_loops, replace=False)
        pre, post = np.concatenate([pre, loops]), np.concatenate([post, loops])
    w = (w_min + np.floor(rng.lognormal(0.5, 1.0, n_edges))).astype(np.float32)
    g = {
        "node_idx": np.arange(n_nodes, dtype=np.int64),
        "bodyId": (10_000 + np.arange(n_nodes)).astype(np.int64),
        "is_clamped": is_clamped, "is_output": is_output, "is_dynamic": ~is_clamped,
        "type_id": type_id, "sign": sign, "w_min": float(w_min),
        "report": json.dumps({"synthetic": True, "n_nodes": n_nodes, "n_edges": n_edges}),
    }
    like = {"indices_pre": np.empty(0, np.int32), "indices_post": np.empty(0, np.int32),
            "perm_csr_to_csc": np.empty(0, np.int64)}
    g.update(build_csr_csc(n_nodes, pre, post, w, like))
    return g


# ----------------------------------------------------------------------------- CLI

def bench(n_edges: int, seed: int = 0, swap_factor: float = 20.0) -> dict:
    """Measured N1 swap rate and peak RSS on a synthetic graph (mean degree ~40)."""
    t0 = time.perf_counter()
    g = synthetic_subgraph(n_nodes=max(1000, n_edges // 40), n_edges=n_edges, seed=seed)
    gen_s = time.perf_counter() - t0
    _warmup()
    t0 = time.perf_counter()
    null = make_null(g, "N1", seed, swap_factor)
    total_s = time.perf_counter() - t0
    p = json.loads(null["null_params"])
    return {
        "n_nodes": len(g["sign"]), "n_edges": n_edges, "seed": seed,
        "attempted_swaps": null["attempted_swaps"], "achieved_swaps": null["achieved_swaps"],
        "accept_rate": null["achieved_swaps"] / max(1, null["attempted_swaps"]),
        "frac_edges_changed": null["frac_edges_changed"],
        "swap_kernel_s": p["swap_kernel_s"],
        "attempts_per_s": null["attempted_swaps"] / p["swap_kernel_s"],
        "swaps_per_s": null["achieved_swaps"] / p["swap_kernel_s"],
        "make_null_total_s": total_s, "synthetic_gen_s": gen_s,
        "peak_rss_mb": peak_rss_mb(),
    }


def _run_one(args: tuple) -> dict:
    graph, kind, seed, out_dir, swap_factor, weight_carrier, check = args
    g = load_graph(graph)
    null = make_null(g, kind, seed, swap_factor, weight_carrier)
    path = Path(out_dir) / f"null_{kind}_{seed}.npz"
    save_graph(path, null)
    summary = {k: null[k] for k in NULL_KEYS if k != "null_params"}
    summary["path"] = str(path)
    if check:
        checks = check_null(g, null)
        summary["failed_invariants"] = failed_invariants(kind, checks)
        summary["n_self_loops"] = checks["n_self_loops_null"]
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--graph", help="input subgraph_*.npz")
    p.add_argument("--kind", choices=list(KINDS), default="N1")
    p.add_argument("--seed", type=int, nargs="+", default=[0], help="one output file per seed")
    p.add_argument("--out", default=str(OUT_DIR / "nulls"), help="output directory")
    p.add_argument("--swap-factor", type=float, default=20.0, help="N1 attempts = factor * E")
    p.add_argument("--weight-carrier", choices=WEIGHT_CARRIERS, default="pre",
                   help="N1: weights follow the pre endpoint, the post endpoint, or are shuffled")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes over seeds")
    p.add_argument("--no-check", action="store_true", help="skip the invariant check")
    p.add_argument("--bench", type=int, metavar="N_EDGES", help="N1 swap-rate benchmark only")
    a = p.parse_args(argv)

    if a.bench:
        print(json.dumps(bench(a.bench, a.seed[0], a.swap_factor), indent=1))
        return 0
    if not a.graph:
        p.error("--graph is required unless --bench is given")
    jobs = [(a.graph, a.kind, s, a.out, a.swap_factor, a.weight_carrier, not a.no_check)
            for s in a.seed]
    if a.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            results = list(ex.map(_run_one, jobs))
    else:
        results = [_run_one(j) for j in jobs]
    bad = 0
    for r in results:
        print(json.dumps(r))
        bad += bool(r.get("failed_invariants"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
