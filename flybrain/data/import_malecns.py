"""Import the MaleCNS v1.0 flat connectome into graph_full.npz + neurons.parquet.

Node/edge policy reproduces doomfly's ``doom/connectome.py`` (nftechie/doomfly) so that the
import ledger can be reconciled to the row:

* nodes: every annotation row with a non-empty ``superclass`` (uncertain ``*_tbc`` classes
  included), minus rows whose ``status == "Glia"``; no restriction to Traced status.
  Rows with no superclass and not glia are doomfly's "unresolved objects".
* edges: every released weight row whose two endpoints are both retained; no weight threshold;
  autapses kept.  ``weight`` is the synapse (contact) count.
* sign: from the presynaptic neuron's ``consensus_nt``; ACh/DA/OA/5-HT/unclear/missing -> +1,
  GABA/Glu/histamine -> -1.  Never folded into ``weight``.

Layout follows CONTRACTS.md: CSR rows = POST (``indptr_post``, ``indices_pre``, ``weight``),
CSC rows = PRE (``indptr_pre``, ``indices_post``, ``perm_csr_to_csc``), neuron index = row of
``neurons.parquet`` = ``bodyId`` ascending.

Run on the server:  python -m flybrain.data.import_malecns [--verify-sha]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from flybrain import MALECNS_DIR, OUT_DIR

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# doomfly reference ledger (CONTRACTS.md). Matched to the row or the diff is reported.
REFERENCE = {
    "raw_rows": 151_856_684,
    "unresolved_objects": 33_013,
    "non_neuronal_excluded": 11_864,
    "n_neurons": 166_700,
    "n_edges": 25_582_938,
    "n_contacts": 124_177_617,
}

# consensus_nt string -> sign of the presynaptic neuron. Missing NT row -> +1 (default).
NT_SIGN = {
    "acetylcholine": 1, "dopamine": 1, "octopamine": 1, "serotonin": 1, "unclear": 1,
    "gaba": -1, "glutamate": -1, "histamine": -1,
}

NEURON_COLUMNS = ["idx", "bodyId", "type", "instance", "superclass", "class", "subclass",
                  "somaSide", "rootSide", "assignedOlHex1", "assignedOlHex2", "status",
                  "statusLabel", "consensus_nt", "sign", "type_id"]


# ----------------------------------------------------------------------------- nodes

def _clean_str(col: pd.Series) -> list:
    """Object list with None for nulls (also flattens pandas Categorical / string dtypes)."""
    col = col.astype(object)
    return [None if pd.isna(v) else str(v) for v in col.tolist()]


def select_neurons(ann: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the doomfly node policy. Returns (kept rows sorted by bodyId, node counts)."""
    if not ann.bodyId.is_unique:
        raise ValueError("duplicate bodyId in the annotation table")
    superclass = ann.superclass.astype(object)
    has_superclass = superclass.notna() & (superclass.astype(str) != "")
    glia = ann.status.astype(object).eq("Glia").fillna(False).astype(bool)
    keep = has_superclass & ~glia
    counts = {
        "annotation_rows": int(len(ann)),
        "non_neuronal_excluded": int(glia.sum()),
        "unresolved_objects": int((~has_superclass & ~glia).sum()),
        "glia_with_superclass": int((glia & has_superclass).sum()),
        "n_neurons": int(keep.sum()),
    }
    kept = ann.loc[keep].sort_values("bodyId", kind="stable").reset_index(drop=True)
    return kept, counts


def sign_from_nt(consensus_nt: list) -> np.ndarray:
    """int8 sign per neuron from consensus_nt strings (None -> +1)."""
    sign = np.empty(len(consensus_nt), dtype=np.int8)
    for k, nt in enumerate(consensus_nt):
        if nt is None:
            sign[k] = 1
        elif nt in NT_SIGN:
            sign[k] = NT_SIGN[nt]
        else:
            raise ValueError(f"unknown consensus_nt value {nt!r}")
    return sign


def join_nt(body_ids: np.ndarray, nt_path: Path) -> list:
    """consensus_nt string per bodyId (None where no NT row), via a searchsorted join."""
    table = feather.read_table(nt_path, columns=["body", "consensus_nt"])
    body = table.column("body").to_numpy()
    nt = np.asarray(table.column("consensus_nt").to_pylist(), dtype=object)
    order = np.argsort(body, kind="stable")
    body, nt = body[order], nt[order]
    if np.any(body[1:] == body[:-1]):
        raise ValueError("duplicate body in the neurotransmitter table")
    pos = np.searchsorted(body, body_ids)
    pos_c = np.minimum(pos, len(body) - 1)
    hit = body[pos_c] == body_ids
    return [nt[p] if h else None for p, h in zip(pos_c.tolist(), hit.tolist())]


def dense_type_id(types: list) -> np.ndarray:
    """int32 dense id over sorted unique non-null type strings; -1 for null."""
    uniq = sorted({t for t in types if t is not None})
    lookup = {t: i for i, t in enumerate(uniq)}
    return np.asarray([lookup.get(t, -1) for t in types], dtype=np.int32)


# ----------------------------------------------------------------------------- edges

def _member(sorted_ids: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(position, found) of each query in a sorted unique id array."""
    pos = np.searchsorted(sorted_ids, query)
    pos = np.minimum(pos, len(sorted_ids) - 1)
    return pos, sorted_ids[pos] == query


class _IdBitmap:
    """Set of non-negative int64 ids as a growable bitmap (cheap distinct-count over ~1e8 ids)."""

    def __init__(self, n_bits: int = 1 << 32):
        self.bits = np.zeros(n_bits >> 3, dtype=np.uint8)

    def add(self, ids: np.ndarray) -> None:
        if len(ids) == 0:
            return
        top = int(ids.max())
        if top < 0:
            raise ValueError("negative body id")
        while top >= len(self.bits) * 8:
            self.bits = np.concatenate([self.bits, np.zeros_like(self.bits)])
        # one pass per bit value: within a pass every write is the same value, so duplicate
        # byte indices in the fancy-index |= are harmless (no slow np.bitwise_or.at needed)
        low = ids & 7
        for bit in range(8):
            self.bits[ids[low == bit] >> 3] |= np.uint8(1 << bit)

    def count(self) -> int:
        # chunked popcount lookup: np.bincount would upcast the 512 MB bitmap to int64 (4 GB)
        popcount = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)
        step = 1 << 24
        return int(sum(int(popcount[self.bits[k:k + step]].sum())
                       for k in range(0, len(self.bits), step)))


def stream_edges(weights_path: Path, kept_ids: np.ndarray, annotated_ids: np.ndarray,
                 log_every: int = 500) -> dict:
    """One pass over the weights feather, batch by batch. Returns kept edges + accounting.

    ``kept_ids`` and ``annotated_ids`` must be sorted unique int64 arrays. Kept edges come back
    as int32 (pre_idx, post_idx) into ``kept_ids`` plus int64 weights, in file order.
    """
    stats = {"raw_rows": 0, "raw_contacts": 0, "n_edges": 0, "n_contacts": 0,
             "rows_dropped_endpoint_not_annotated": 0,
             "rows_dropped_endpoint_annotated_not_retained": 0,
             "self_edges": 0, "weight_one_edges": 0, "max_weight": 0}
    pre_chunks, post_chunks, w_chunks = [], [], []
    unannotated = _IdBitmap()
    annotated_not_retained_seen = np.zeros(len(annotated_ids), dtype=bool)
    t0 = time.time()
    with pa.OSFile(str(weights_path), "rb") as handle:
        reader = ipc.open_file(handle)
        cols = [reader.schema.get_field_index(c) for c in ("body_pre", "body_post", "weight")]
        n_batches = reader.num_record_batches
        for b in range(n_batches):
            batch = reader.get_batch(b)
            pre, post, w = (batch.column(c).to_numpy() for c in cols)
            if w.dtype.kind not in "iu":
                raise ValueError(f"weights are {w.dtype}, expected integer")
            if len(w) and w.min() < 1:
                raise ValueError("non-positive synapse count in the weights file")
            i, ok_i = _member(kept_ids, pre)
            j, ok_j = _member(kept_ids, post)
            keep = ok_i & ok_j
            a_i, ann_i = _member(annotated_ids, pre)
            a_j, ann_j = _member(annotated_ids, post)
            not_annotated = ~ann_i | ~ann_j
            stats["raw_rows"] += len(w)
            stats["raw_contacts"] += int(w.sum())
            stats["n_edges"] += int(keep.sum())
            stats["n_contacts"] += int(w[keep].sum())
            stats["rows_dropped_endpoint_not_annotated"] += int(not_annotated.sum())
            stats["rows_dropped_endpoint_annotated_not_retained"] += int((~keep & ~not_annotated).sum())
            stats["self_edges"] += int(np.count_nonzero(keep & (pre == post)))
            stats["weight_one_edges"] += int(np.count_nonzero(keep & (w == 1)))
            stats["max_weight"] = max(stats["max_weight"], int(w[keep].max()) if keep.any() else 0)
            unannotated.add(np.concatenate([pre[~ann_i], post[~ann_j]]))
            annotated_not_retained_seen[a_i[ann_i & ~ok_i]] = True
            annotated_not_retained_seen[a_j[ann_j & ~ok_j]] = True
            pre_chunks.append(i[keep].astype(np.int32))
            post_chunks.append(j[keep].astype(np.int32))
            w_chunks.append(w[keep])
            if log_every and (b + 1) % log_every == 0:
                print(f"  batch {b + 1}/{n_batches}  rows {stats['raw_rows']:,}  kept "
                      f"{stats['n_edges']:,}  {time.time() - t0:.0f}s", flush=True)
    stats["distinct_edge_bodies_not_annotated"] = unannotated.count()
    stats["distinct_edge_bodies_annotated_not_retained"] = int(annotated_not_retained_seen.sum())
    stats["stream_wall_s"] = round(time.time() - t0, 1)
    return {"pre": np.concatenate(pre_chunks), "post": np.concatenate(post_chunks),
            "weight": np.concatenate(w_chunks), "stats": stats}


def build_csr_csc(n: int, pre: np.ndarray, post: np.ndarray, weight: np.ndarray) -> dict:
    """CSR by POST (rows=post, cols=pre) and CSC by PRE, both with sorted inner indices.

    ``perm_csr_to_csc`` maps CSC position -> CSR position: ``weight[perm]`` is the CSC weight.
    Duplicate (pre, post) pairs are kept as separate entries and counted in ``n_duplicate_pairs``.
    """
    order = np.lexsort((pre, post))                       # primary post, secondary pre
    post_c, pre_c, w_c = post[order], pre[order], weight[order]
    indptr_post = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post_c, minlength=n), out=indptr_post[1:])
    perm = np.lexsort((post_c, pre_c)).astype(np.int64)  # CSC order over CSR positions
    indptr_pre = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre_c, minlength=n), out=indptr_pre[1:])
    dup = int(np.count_nonzero((post_c[1:] == post_c[:-1]) & (pre_c[1:] == pre_c[:-1])))
    if w_c.max() >= 2 ** 24:
        raise ValueError("synapse count not exactly representable in float32")
    return {"indptr_post": indptr_post, "indices_pre": pre_c.astype(np.int32),
            "weight": w_c.astype(np.float32), "indptr_pre": indptr_pre,
            "indices_post": post_c[perm].astype(np.int32), "perm_csr_to_csc": perm,
            "n_duplicate_pairs": dup}


# ----------------------------------------------------------------------------- ledger / io

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256sums(data_dir: Path) -> dict:
    path = data_dir / "SHA256SUMS"
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].lstrip("*")] = parts[0]
    return out


def make_ledger(counts: dict) -> dict:
    """Flat ledger with the six contract numbers, the reference, and pass/fail per item."""
    ledger = {k: int(counts[k]) for k in REFERENCE}
    ledger["reference"] = dict(REFERENCE)
    ledger["checks"] = {k: {"got": int(counts[k]), "expected": v, "diff": int(counts[k]) - v,
                            "pass": int(counts[k]) == v} for k, v in REFERENCE.items()}
    ledger["all_pass"] = all(c["pass"] for c in ledger["checks"].values())
    return ledger


def load_graph(path: Path | None = None) -> dict:
    """Load graph_full.npz into a dict of arrays; ``ledger`` is parsed from JSON."""
    path = Path(path) if path else OUT_DIR / "graph" / "graph_full.npz"
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    out["n_neurons"], out["n_edges"] = int(out["n_neurons"]), int(out["n_edges"])
    out["ledger"] = json.loads(str(out["ledger"]))
    return out


def run(data_dir: Path = MALECNS_DIR, out_dir: Path | None = None, verify_sha: bool = False,
        log_every: int = 500) -> dict:
    """Full import. Writes <out_dir>/graph_full.npz, neurons.parquet, ledger.json; returns ledger."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir) if out_dir else OUT_DIR / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    proc = psutil.Process()

    # -- nodes
    ann = feather.read_table(data_dir / ANNOTATIONS).to_pandas()
    kept, node_counts = select_neurons(ann)
    body_ids = kept.bodyId.to_numpy().astype(np.int64)
    annotated_ids = np.sort(ann.bodyId.to_numpy().astype(np.int64))
    consensus_nt = join_nt(body_ids, data_dir / NEUROTRANSMITTERS)
    sign = sign_from_nt(consensus_nt)
    print(f"nodes: {node_counts}  ({time.time() - t_start:.0f}s)", flush=True)

    # -- edges
    edges = stream_edges(data_dir / WEIGHTS, body_ids, annotated_ids, log_every=log_every)
    n = len(body_ids)
    graph = build_csr_csc(n, edges["pre"], edges["post"], edges["weight"])
    del edges["pre"], edges["post"], edges["weight"]
    in_deg = np.diff(graph["indptr_post"])
    out_deg = np.diff(graph["indptr_pre"])
    print(f"edges: {edges['stats']}  ({time.time() - t_start:.0f}s)", flush=True)

    # -- ledger
    counts = {**node_counts, **edges["stats"]}
    ledger = make_ledger(counts)
    nt_counts = pd.Series([nt or "missing" for nt in consensus_nt]).value_counts().to_dict()
    ledger["extra"] = {
        **{k: v for k, v in counts.items() if k not in REFERENCE},
        "n_duplicate_pairs": graph["n_duplicate_pairs"],
        "isolated_neurons": int(np.count_nonzero((in_deg == 0) & (out_deg == 0))),
        "n_types": int(kept.type.nunique()), "type_null": int(kept.type.isna().sum()),
        "consensus_nt_counts": {k: int(v) for k, v in nt_counts.items()},
        "sign_counts": {"+1": int(np.count_nonzero(sign == 1)), "-1": int(np.count_nonzero(sign == -1))},
        "superclass_counts": {k: int(v) for k, v in kept.superclass.value_counts().items()},
    }
    sums = read_sha256sums(data_dir)
    source = {name: {"bytes": (data_dir / name).stat().st_size, "sha256_listed": sums.get(name)}
              for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS)}
    if verify_sha:
        for name, info in source.items():
            info["sha256_computed"] = sha256_file(data_dir / name)
            info["sha256_verified"] = info["sha256_computed"] == info["sha256_listed"]
    ledger["source"] = {"data_dir": str(data_dir), "files": source,
                        "policy_reference": "github.com/nftechie/doomfly doom/connectome.py",
                        "nt_sign_map": NT_SIGN}

    # -- neurons.parquet
    table = pa.table({
        "idx": pa.array(np.arange(n, dtype=np.int32)),
        "bodyId": pa.array(body_ids),
        **{c: pa.array(_clean_str(kept[c]), type=pa.string())
           for c in ("type", "instance", "superclass", "class", "subclass", "somaSide",
                     "rootSide", "status", "statusLabel")},
        "assignedOlHex1": pa.array(kept.assignedOlHex1.to_numpy(dtype=np.float64)),
        "assignedOlHex2": pa.array(kept.assignedOlHex2.to_numpy(dtype=np.float64)),
        "consensus_nt": pa.array(consensus_nt, type=pa.string()),
        "sign": pa.array(sign),
        "type_id": pa.array(dense_type_id(_clean_str(kept.type))),
    }).select(NEURON_COLUMNS)
    tmp = out_dir / "neurons.parquet.partial"
    pq.write_table(table, tmp)
    tmp.replace(out_dir / "neurons.parquet")

    # -- graph_full.npz
    ledger["runtime"] = {
        "wall_s": round(time.time() - t_start, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "rss_end_mb": round(proc.memory_info().rss / 2 ** 20, 1),
        "host": platform.node(), "numpy": np.__version__, "pyarrow": pa.__version__,
        "note": "peak_rss excludes the npz compression step, which runs after this is recorded",
    }
    tmp = out_dir / "graph_full.npz.partial.npz"
    np.savez_compressed(
        tmp, n_neurons=np.int64(n), n_edges=np.int64(len(graph["weight"])),
        bodyId=body_ids, sign=sign, indptr_post=graph["indptr_post"],
        indices_pre=graph["indices_pre"], weight=graph["weight"], indptr_pre=graph["indptr_pre"],
        indices_post=graph["indices_post"], perm_csr_to_csc=graph["perm_csr_to_csc"],
        ledger=json.dumps(ledger))
    tmp.replace(out_dir / "graph_full.npz")
    ledger["runtime"]["wall_s_incl_write"] = round(time.time() - t_start, 1)
    ledger["runtime"]["peak_rss_mb_incl_write"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    (out_dir / "ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    return ledger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=MALECNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=None, help="default $FLYBRAIN_OUT/graph")
    parser.add_argument("--verify-sha", action="store_true", help="recompute source SHA-256s")
    parser.add_argument("--log-every", type=int, default=500, help="progress every N batches")
    args = parser.parse_args()
    ledger = run(args.data_dir, args.out_dir, args.verify_sha, args.log_every)
    print(json.dumps({k: ledger[k] for k in ("checks", "all_pass", "runtime")}, indent=2))
    for k, c in ledger["checks"].items():
        print(f"{'PASS' if c['pass'] else 'FAIL'}  {k:<22} got {c['got']:>13,}  "
              f"expected {c['expected']:>13,}  diff {c['diff']:+,}")


if __name__ == "__main__":
    main()
