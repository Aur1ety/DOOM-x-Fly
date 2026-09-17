"""Subgraph rule and readout set (CONTRACTS.md).

Reads ``$FLYBRAIN_OUT/graph/graph_full.npz`` + ``neurons.parquet``, applies the rule below and
writes ``subgraph_<name>.npz`` next to them and ``configs/readout_v1.json`` in the repo.

Rule (deterministic, re-runnable):
  I  clamped inputs = cells whose ``type`` is in CLAMPED_TYPES. Rates are set by the front
                      end, so EVERY edge into a clamped cell is dropped (incl. clamped->clamped).
  D  outputs        = cells with ``superclass == "descending_neuron"`` (the original plan said "descending";
                      that string does not occur in MaleCNS v1.0, "descending_neuron" = 1,314 cells).
  C  core           = forward-reachable from I within ``hops`` AND backward-reachable from D
                      within ``hops``, over edges with weight >= w_min after the clamp drop.
  F  forced         = cells whose ``type`` matches FORCED_PATTERNS ('*' = glob, else exact),
                      kept regardless of reachability. Forcing applies to nodes only; their
                      edges still obey w_min.
  dynamic = (C | F) minus I.  Stored edges: weight >= w_min, post in dynamic, pre in dynamic|I.
  Budget clamp: w_min is raised in steps of 1 until |dynamic| <= max_dynamic and
  |stored edges| <= max_edges. Stored edges include clamped->dynamic (what the SpMM runs).
  Candidate pool: cells whose superclass starts with "vnc_" are excluded (brain only,
  no VNC); named cells (I, D, F) are never excluded. ``--exclude-superclass`` widens the list,
  ``--output-set readout`` seeds the backward BFS from the readout cells only (sensitivity).

Convention: CSR rows = POST, cols = PRE; sign belongs to the presynaptic cell.

CLI:
  python -m flybrain.data.subgraph                 # build subgraph_v1.npz
  python -m flybrain.data.subgraph --readout       # write configs/readout_v1.json (frozen)
  python -m flybrain.data.subgraph --vocab         # measured type vocabulary on the annotations
  python -m flybrain.data.subgraph --hops 3 --exclude-superclass vnc_,ol_intrinsic --dry-run
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather
import pyarrow.parquet as pq
import scipy.sparse as sp

from flybrain import MALECNS_DIR, OUT_DIR

CLAMPED_TYPES: tuple[str, ...] = (
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
    "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20", "Tm5a", "Tm5b", "Tm5c", "TmY5a",
)
OUTPUT_SUPERCLASS = "descending_neuron"   # measured: MaleCNS v1.0 has no "descending" value
FORCED_PATTERNS: tuple[str, ...] = (
    "LC4", "LC6", "LC9", "LC10a", "LC11", "LC12", "LC15", "LC16", "LC17", "LPLC1", "LPLC2",
    "LPi*", "HS*", "VS*", "AOTU019", "AOTU025", "PFL2", "PFL3", "EPG",
    "PEN_a(PEN1)", "PEN_b(PEN2)",          # measured MaleCNS strings for PEN1 / PEN2
    "pC1*", "P1*",                          # P1* matches nothing in v1.0 (P1 sits under pC1_*)
    "DNa01", "DNa02", "DNb05", "DNb06", "DNg13", "DNg100",
    "DNp01", "DNp02", "DNp03", "DNp04", "DNp09", "DNp10", "DNp11", "DNp103",
    "MDN", "pIP10", "DNpe050", "DNp67",
)
# Readout type -> role. Dict order is the canonical cell order.
# DNpe050 / DNp67 are pre-registered readout cells without a role in the plan's table: they
# get "unassigned" (in the feature vector, no biological-prior weight) rather than a guess.
READOUT_ROLES: dict[str, str] = {
    "DNa02": "turn", "DNa01": "turn", "DNb05": "turn", "DNb06": "turn", "DNg13": "turn",
    "DNp09": "forward", "DNg100": "forward",
    "MDN": "backward",
    "DNp01": "dodge", "DNp02": "dodge", "DNp04": "dodge", "DNp11": "dodge", "DNp103": "dodge",
    "AOTU019": "attack_midline",
    "pIP10": "attack_pursuit",
    "DNpe050": "unassigned", "DNp67": "unassigned",
}
EXCLUDE_SUPERCLASS_PREFIX = "vnc_"   # brain only, no VNC (vnc_intrinsic/sensory/motor/...)
DEFAULT_W_MIN = 5.0
MAX_DYNAMIC = 20_000
MAX_EDGES = 2_500_000
HOPS = 4
LR_THRESHOLD = 0.9

ANNOTATIONS = MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
GRAPH_DIR = OUT_DIR / "graph"
REPO_ROOT = Path(__file__).resolve().parents[2]
READOUT_PATH = REPO_ROOT / "configs" / "readout_v1.json"

_SIDE_ORDER = {"L": 0, "R": 1}


# ----------------------------------------------------------------------------- loading

def load_cells(path: Path, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read neurons.parquet or the raw annotations feather into numpy columns.

    String columns become object arrays (None for null); numeric columns stay numeric.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        tbl = pq.read_table(path, columns=columns)
    else:
        tbl = pyarrow.feather.read_table(path, columns=columns)
    out: dict[str, np.ndarray] = {}
    for name in tbl.column_names:
        col = tbl.column(name)
        t = col.type
        if pa.types.is_string(t) or pa.types.is_large_string(t) or pa.types.is_dictionary(t):
            out[name] = np.array(col.to_pylist(), dtype=object)
        else:
            out[name] = col.to_numpy(zero_copy_only=False)
    return out


class FullGraph:
    """graph_full.npz + neurons.parquet (CONTRACTS.md) as numpy, plus 0/1 adjacency shells.

    ``set_excluded(mask)`` removes cells from the candidate pool: every edge touching them is
    ignored by the BFS and they are never selected (see build_subgraph for the policy).
    """

    def __init__(self, graph_dir: Path):
        graph_dir = Path(graph_dir)
        z = np.load(graph_dir / "graph_full.npz")
        self.n = int(z["n_neurons"])
        self.e = int(z["n_edges"])
        self.bodyId = z["bodyId"].astype(np.int64)
        self.sign = z["sign"].astype(np.int8)
        self.indptr_post = z["indptr_post"].astype(np.int64)
        self.indices_pre = z["indices_pre"].astype(np.int32)
        self.weight = z["weight"].astype(np.float32)
        self.indptr_pre = z["indptr_pre"].astype(np.int64)
        self.indices_post = z["indices_post"].astype(np.int32)
        self.perm = z["perm_csr_to_csc"].astype(np.int64)
        assert self.indices_pre.shape[0] == self.e and self.indptr_post.shape[0] == self.n + 1
        self.post_of_edge = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.indptr_post))
        self.cells = load_cells(
            graph_dir / "neurons.parquet",
            ["idx", "bodyId", "type", "superclass", "somaSide", "sign", "type_id"],
        )
        assert self.cells["bodyId"].shape[0] == self.n, "neurons.parquet rows != n_neurons"
        assert np.array_equal(self.cells["bodyId"].astype(np.int64), self.bodyId), "bodyId order mismatch"
        self.set_excluded(np.zeros(self.n, dtype=bool))
        ones = np.ones(self.e, dtype=np.float32)
        # y = A_post @ x : input to each post from pre-set x.  y = A_pre @ x : pres feeding x.
        self.A_post = sp.csr_matrix((ones, self.indices_pre, self.indptr_post), shape=(self.n, self.n))
        self.A_pre = sp.csr_matrix((ones.copy(), self.indices_post, self.indptr_pre), shape=(self.n, self.n))

    def set_excluded(self, mask: np.ndarray) -> None:
        self.excluded = mask.astype(bool)
        self.edge_ok = ~self.excluded[self.post_of_edge] & ~self.excluded[self.indices_pre]

    @property
    def types(self) -> np.ndarray:
        return self.cells["type"]


def superclass_prefix_mask(superclass: np.ndarray, prefixes: str | None) -> np.ndarray:
    """bool[N]: superclass starts with any of the comma-separated ``prefixes`` ('' / None = none)."""
    pref = tuple(p.strip() for p in (prefixes or "").split(",") if p.strip())
    if not pref:
        return np.zeros(len(superclass), dtype=bool)
    return np.fromiter((isinstance(s, str) and s.startswith(pref) for s in superclass),
                       dtype=bool, count=len(superclass))


# ----------------------------------------------------------------------------- type sets

def match_types(types: np.ndarray, patterns) -> dict[str, list[str]]:
    """Pattern -> sorted unique type strings it matches ('*'/'?'/'[' = glob, else exact)."""
    uniq = sorted({t for t in types if isinstance(t, str)})
    out: dict[str, list[str]] = {}
    for p in patterns:
        if any(c in p for c in "*?["):
            out[p] = [t for t in uniq if fnmatch.fnmatchcase(t, p)]
        else:
            out[p] = [t for t in uniq if t == p]
    return out


def type_mask(types: np.ndarray, patterns) -> np.ndarray:
    """bool[N]: cell type matches any pattern."""
    hits = {t for ts in match_types(types, patterns).values() for t in ts}
    return np.fromiter((t in hits for t in types), dtype=bool, count=len(types))


def norm_side(s) -> str:
    """somaSide -> 'L' | 'R' | 'other'."""
    if isinstance(s, str):
        u = s.strip().upper()
        if u in ("L", "LEFT"):
            return "L"
        if u in ("R", "RIGHT"):
            return "R"
    return "other"


# ----------------------------------------------------------------------------- reachability

def hop_distances(mat: sp.csr_matrix, seeds: np.ndarray, hops: int) -> np.ndarray:
    """Hop distance from seeds (bool[N]) following y = mat @ x; -1 = not within ``hops``."""
    dist = np.full(mat.shape[0], -1, dtype=np.int16)
    dist[seeds] = 0
    frontier = seeds.astype(np.float32)
    for k in range(1, hops + 1):
        new = ((mat @ frontier) > 0) & (dist < 0)
        if not new.any():
            break
        dist[new] = k
        frontier = new.astype(np.float32)
    return dist


def select_nodes(g: FullGraph, w_min: float, hops: int, clamped: np.ndarray,
                 output: np.ndarray, forced: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the rule at one w_min. Returns fwd/bwd hop distances, core, dynamic, edge masks."""
    # threshold + drop edges into clamped + drop edges touching excluded (VNC) cells
    keep = (g.weight >= w_min) & ~clamped[g.post_of_edge] & g.edge_ok
    g.A_post.data = keep.astype(np.float32)
    g.A_pre.data = keep[g.perm].astype(np.float32)
    fwd = hop_distances(g.A_post, clamped & ~g.excluded, hops)
    bwd = hop_distances(g.A_pre, output & ~g.excluded, hops)
    core = (fwd >= 0) & (bwd >= 0) & ~clamped & ~g.excluded
    dynamic = (core | forced) & ~clamped & ~g.excluded
    edge_mask = keep & dynamic[g.post_of_edge] & (dynamic | clamped)[g.indices_pre]
    return {"keep": keep, "fwd": fwd, "bwd": bwd, "core": core, "dynamic": dynamic,
            "edge_mask": edge_mask}


def budget_clamp(g: FullGraph, w_min: float, hops: int, clamped, output, forced,
                 max_dynamic: int = MAX_DYNAMIC, max_edges: int = MAX_EDGES,
                 w_max: float | None = None, verbose: bool = True):
    """Raise w_min by 1 until the budget holds. Returns (w_min, selection, trajectory)."""
    if w_max is None:
        w_max = float(g.weight.max()) + 1.0
    traj = []
    while True:
        sel = select_nodes(g, w_min, hops, clamped, output, forced)
        n_dyn, n_edges = int(sel["dynamic"].sum()), int(sel["edge_mask"].sum())
        traj.append({"w_min": float(w_min), "n_core": int(sel["core"].sum()),
                     "n_dynamic": n_dyn, "n_edges": n_edges,
                     "core_by_superclass": by_superclass(g.cells["superclass"], sel["core"])})
        if verbose:
            print(f"  w_min={w_min:g}: core={traj[-1]['n_core']} dynamic={n_dyn} edges={n_edges}",
                  file=sys.stderr)
        if n_dyn <= max_dynamic and n_edges <= max_edges:
            return float(w_min), sel, traj
        if w_min >= w_max:
            raise RuntimeError(
                f"budget clamp failed: at w_min={w_min:g} still dynamic={n_dyn} (max {max_dynamic}), "
                f"edges={n_edges} (max {max_edges}); forced set alone = {int((forced & ~clamped).sum())}")
        w_min += 1.0


# ----------------------------------------------------------------------------- local graph

def build_local(g: FullGraph, sel: dict, clamped: np.ndarray, is_output_global: np.ndarray,
                w_min: float) -> dict[str, np.ndarray]:
    """Local CSR/CSC over the selected nodes (sorted by global index), per CONTRACTS.md."""
    node_idx = np.flatnonzero(sel["dynamic"] | clamped).astype(np.int64)
    m = node_idx.shape[0]
    local = np.full(g.n, -1, dtype=np.int64)
    local[node_idx] = np.arange(m)
    em = sel["edge_mask"]
    row = local[g.post_of_edge[em]]
    col = local[g.indices_pre[em]]
    w = g.weight[em]
    assert (row >= 0).all() and (col >= 0).all()
    order = np.lexsort((col, row))                      # CSR: by post, then pre
    row, col, w = row[order], col[order], w[order]
    indptr_post = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(row, minlength=m), out=indptr_post[1:])
    perm = np.lexsort((row, col)).astype(np.int64)      # CSC: by pre, then post
    indptr_pre = np.zeros(m + 1, dtype=np.int64)
    np.cumsum(np.bincount(col, minlength=m), out=indptr_pre[1:])
    return {
        "node_idx": node_idx,
        "bodyId": g.bodyId[node_idx],
        "is_clamped": clamped[node_idx],
        "is_output": is_output_global[node_idx],
        "is_dynamic": ~clamped[node_idx],
        "type_id": g.cells["type_id"][node_idx].astype(np.int32),
        "sign": g.sign[node_idx],
        "indptr_post": indptr_post,
        "indices_pre": col.astype(np.int32),
        "weight": w.astype(np.float32),
        "indptr_pre": indptr_pre,
        "indices_post": row[perm].astype(np.int32),
        "perm_csr_to_csc": perm,
        "w_min": np.float64(w_min),
    }


def local_adjacency(sub: dict) -> sp.csr_matrix:
    """0/1 local CSR (rows = post) for BFS on a subgraph dict."""
    m = sub["node_idx"].shape[0]
    ones = np.ones(sub["weight"].shape[0], dtype=np.float32)
    return sp.csr_matrix((ones, sub["indices_pre"], sub["indptr_post"]), shape=(m, m))


# ----------------------------------------------------------------------------- reports

def by_superclass(superclass: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    """Cell counts per superclass among ``mask``, largest first."""
    vals, n = np.unique(np.array([s if isinstance(s, str) else "<null>" for s in superclass[mask]]),
                        return_counts=True)
    return dict(sorted(zip(vals.tolist(), n.tolist()), key=lambda kv: -kv[1]))


def lr_audit(types: np.ndarray, sides: np.ndarray, mask: np.ndarray,
             threshold: float = LR_THRESHOLD) -> dict:
    """Per-type L/R counts among ``mask``; flag types with min(L,R)/max(L,R) < threshold."""
    per: dict[str, dict[str, int]] = {}
    for t, s in zip(types[mask], sides[mask]):
        d = per.setdefault(t if isinstance(t, str) else "<null>", {"L": 0, "R": 0, "other": 0})
        d[norm_side(s)] += 1
    flagged = []
    for t, d in sorted(per.items()):
        hi, lo = max(d["L"], d["R"]), min(d["L"], d["R"])
        if hi and lo / hi < threshold:
            flagged.append({"type": t, "L": d["L"], "R": d["R"], "ratio": round(lo / hi, 3)})
    big = [f for f in flagged if max(f["L"], f["R"]) >= 5]   # gaps in populous types, not singletons
    return {"threshold": threshold, "n_types": len(per), "n_flagged": len(flagged),
            "n_flagged_cells": int(sum(f["L"] + f["R"] for f in flagged)),
            "n_flagged_ge5": len(big), "flagged_ge5": big, "flagged": flagged, "per_type": per}


def resolve_readout(cells: dict[str, np.ndarray], roles: dict[str, str] = READOUT_ROLES):
    """Readout types -> [{type, side, bodyId, role}], sorted (role order, L<R<other, bodyId).

    Returns (cells_list, missing_types). Every cell of a listed type is taken, both sides.
    """
    by_type: dict[str, list[int]] = {}
    for i, t in enumerate(cells["type"]):
        if t in roles:
            by_type.setdefault(t, []).append(i)
    body, side = cells["bodyId"], cells["somaSide"]
    out, missing = [], []
    for t, role in roles.items():
        idx = by_type.get(t, [])
        if not idx:
            missing.append(t)
            continue
        rows = sorted(((norm_side(side[i]), int(body[i])) for i in idx),
                      key=lambda r: (_SIDE_ORDER.get(r[0], 2), r[1]))
        out += [{"type": t, "side": s, "bodyId": b, "role": role} for s, b in rows]
    return out, missing


def cells_sha256(cells: list[dict]) -> str:
    """sha256 of the canonical JSON of the cells list (sorted keys, no whitespace)."""
    blob = json.dumps(cells, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def write_readout(cells: list[dict], path: Path, missing: list[str], source: str,
                  force: bool = False) -> dict:
    """Write configs/readout_v1.json (refuses to overwrite unless force: the set is frozen)."""
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists and the readout set is frozen; use --force")
    doc = {"version": 1, "cells": cells, "sha256": cells_sha256(cells),
           "n_cells": len(cells), "missing_types": missing, "source": source}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def load_readout(path: Path) -> dict:
    """Load and verify a readout json (hash must match its cells list)."""
    doc = json.loads(Path(path).read_text())
    if cells_sha256(doc["cells"]) != doc["sha256"]:
        raise ValueError(f"{path}: sha256 does not match its cells list")
    return doc


def dn_report(g: FullGraph, sel: dict, sub: dict, output: np.ndarray, readout_cells: list[dict],
              hops: int) -> dict:
    """DN reachability: all DNs in the full thresholded graph, DNs and readout cells locally."""
    fwd = sel["fwd"]
    n_dn = int(output.sum())
    full = {"n_dn": n_dn, "n_reached": int(((fwd > 0) & output).sum()),
            "hop_hist": np.bincount(fwd[output & (fwd > 0)], minlength=hops + 1)[1:].tolist()}
    full["frac"] = full["n_reached"] / n_dn if n_dn else None
    ldist = hop_distances(local_adjacency(sub), sub["is_clamped"], hops)
    dn_local = output[sub["node_idx"]]
    loc = {"n_dn_in_subgraph": int(dn_local.sum()),
           "n_reached": int(((ldist > 0) & dn_local).sum()),
           "hop_hist": np.bincount(ldist[dn_local & (ldist > 0)], minlength=hops + 1)[1:].tolist()}
    loc["frac"] = loc["n_reached"] / loc["n_dn_in_subgraph"] if loc["n_dn_in_subgraph"] else None
    body_to_local = {int(b): i for i, b in enumerate(sub["bodyId"].tolist())}
    per_cell = []
    for c in readout_cells:
        i = body_to_local.get(int(c["bodyId"]))
        per_cell.append({**c, "in_subgraph": i is not None,
                         "hops": int(ldist[i]) if i is not None else -1})
    n_reached = sum(1 for c in per_cell if c["hops"] > 0)
    readout = {"n_cells": len(per_cell), "n_in_subgraph": sum(c["in_subgraph"] for c in per_cell),
               "n_reached": n_reached,
               "frac": n_reached / len(per_cell) if per_cell else None, "per_cell": per_cell}
    return {"full_graph_4hop": full, "subgraph": loc, "readout": readout}


def build_report(g: FullGraph, name: str, w_min0: float, w_min: float, hops: int, traj: list,
                 sel: dict, sub: dict, clamped, output, forced, forced_match: dict,
                 readout_cells: list[dict], readout_sha: str | None, max_dynamic: int,
                 max_edges: int) -> dict:
    """The JSON report stored in the npz (CONTRACTS.md subgraph_<name>.npz 'report')."""
    dyn, em = sel["dynamic"], sel["edge_mask"]
    node_mask = dyn | clamped
    pre_e, post_e = g.indices_pre, g.post_of_edge
    stored_w = float(g.weight[em].sum())
    within_nodes = (dyn[post_e] & node_mask[pre_e])          # same nodes, same clamp drop, no threshold
    dyn_inputs = dyn[post_e]                                  # every input edge to a dynamic cell
    out_deg_clamped = np.bincount(sub["indices_pre"], minlength=sub["node_idx"].shape[0])[sub["is_clamped"]]
    forced_only = forced & ~sel["core"] & ~clamped
    forced_rep = {}
    for p, ts in forced_match.items():
        if ts:
            m = type_mask(g.types, ts)
            forced_rep[p] = {"types": ts, "n_cells": int(m.sum()), "n_in_core": int((m & sel["core"]).sum())}
    return {
        "name": name,
        "rule": {"clamped_types": list(CLAMPED_TYPES), "output_superclass": OUTPUT_SUPERCLASS,
                 "forced_patterns": list(FORCED_PATTERNS), "hops": hops, "w_min_start": w_min0,
                 "max_dynamic": max_dynamic, "max_edges": max_edges,
                 "edges_into_clamped_dropped": True, "forced_edges_obey_w_min": True},
        "w_min": w_min,
        "clamp_trajectory": traj,
        "counts": {
            "n_total": g.n, "n_edges_total": g.e,
            "n_excluded_cells": int(g.excluded.sum()),
            "n_excluded_edges": int((~g.edge_ok).sum()),
            "n_forced_excluded": int((forced & g.excluded).sum()),
            "n_output_superclass_excluded": int((output & g.excluded).sum()),
            "n_clamped": int(clamped.sum()),
            "n_clamped_with_stored_out_edges": int((out_deg_clamped > 0).sum()),
            "n_output_superclass": int(output.sum()),
            "n_dn_in_subgraph": int((output & dyn).sum()),
            "n_core": int(sel["core"].sum()),
            "n_forced_cells": int((forced & ~clamped).sum()),
            "n_forced_not_in_core": int(forced_only.sum()),
            "n_dynamic": int(dyn.sum()),
            "n_nodes": int(node_mask.sum()),
            "n_edges": int(em.sum()),
            "n_edges_clamped_to_dynamic": int((em & clamped[pre_e]).sum()),
            "n_edges_dynamic_to_dynamic": int((em & dyn[pre_e]).sum()),
            "n_autapses": int((sub["indices_pre"] == np.repeat(np.arange(sub["node_idx"].shape[0]),
                                                                np.diff(sub["indptr_post"]))).sum()),
            "n_output_cells": int(sub["is_output"].sum()),
            "budget_ok": bool(dyn.sum() <= max_dynamic and em.sum() <= max_edges),
        },
        "weight": {
            "stored": stored_w,
            "full_total": float(g.weight.sum()),
            "frac_of_full": stored_w / float(g.weight.sum()),
            "within_nodes_unthresholded": float(g.weight[within_nodes].sum()),
            "frac_within_nodes": stored_w / max(float(g.weight[within_nodes].sum()), 1e-9),
            "dynamic_inputs_total": float(g.weight[dyn_inputs].sum()),
            "frac_of_dynamic_inputs": stored_w / max(float(g.weight[dyn_inputs].sum()), 1e-9),
        },
        "by_superclass": {"clamped": by_superclass(g.cells["superclass"], clamped),
                          "core": by_superclass(g.cells["superclass"], sel["core"]),
                          "dynamic": by_superclass(g.cells["superclass"], dyn)},
        "forced": {"hits": forced_rep,
                   "missing_patterns": [p for p, ts in forced_match.items() if not ts]},
        "dn_reachability": dn_report(g, sel, sub, output, readout_cells, hops),
        "lr_audit": lr_audit(g.types, g.cells["somaSide"], node_mask),
        "readout_sha256": readout_sha,
    }


# ----------------------------------------------------------------------------- top level

def build_subgraph(graph_dir: Path, name: str = "v1", w_min: float = DEFAULT_W_MIN,
                   hops: int = HOPS, max_dynamic: int = MAX_DYNAMIC, max_edges: int = MAX_EDGES,
                   readout: dict | None = None, verbose: bool = True,
                   exclude_prefix: str | None = EXCLUDE_SUPERCLASS_PREFIX,
                   output_set: str = "descending") -> tuple[dict, dict]:
    """Run the full rule. Returns (subgraph arrays incl. 'report' JSON string, report dict).

    exclude_prefix: comma-separated superclass prefixes removed from the candidate pool
    (default "vnc_" = brain only, no VNC). Named cells (clamped, forced, descending) are
    never excluded, so forced inclusion always wins.
    output_set: "descending" (pre-registered: D = every descending_neuron) or "readout"
    (sensitivity: D = the pre-registered readout cells only; DN reports still cover all DNs).
    """
    g = FullGraph(graph_dir)
    types = g.types
    clamped = type_mask(types, CLAMPED_TYPES)
    output = np.fromiter((s == OUTPUT_SUPERCLASS for s in g.cells["superclass"]), dtype=bool, count=g.n)
    forced_match = match_types(types, FORCED_PATTERNS)
    forced = type_mask(types, FORCED_PATTERNS)
    prefix_hit = superclass_prefix_mask(g.cells["superclass"], exclude_prefix)
    named = clamped | forced | output
    g.set_excluded(prefix_hit & ~named)
    n_immune = int((prefix_hit & named).sum())
    if readout is None:
        cells, missing = resolve_readout(g.cells)
        readout = {"cells": cells, "sha256": None, "missing_types": missing}
    body_to_idx = {int(b): i for i, b in enumerate(g.bodyId.tolist())}
    is_output = np.zeros(g.n, dtype=bool)
    for c in readout["cells"]:
        i = body_to_idx.get(int(c["bodyId"]))
        if i is None:
            print(f"warning: readout bodyId {c['bodyId']} ({c['type']}) not in graph", file=sys.stderr)
        else:
            is_output[i] = True
    if output_set == "readout":
        bfs_output = is_output.copy()
    elif output_set == "descending":
        bfs_output = output
    else:
        raise ValueError(f"output_set must be 'descending' or 'readout', got {output_set!r}")
    w_min0 = float(w_min)
    w_min, sel, traj = budget_clamp(g, w_min0, hops, clamped, bfs_output, forced,
                                    max_dynamic, max_edges, verbose=verbose)
    not_dyn = is_output & ~sel["dynamic"]
    if not_dyn.any():
        print(f"warning: {int(not_dyn.sum())} readout cells are not dynamic nodes", file=sys.stderr)
    sub = build_local(g, sel, clamped, is_output & sel["dynamic"], w_min)
    report = build_report(g, name, w_min0, w_min, hops, traj, sel, sub, clamped, output, forced,
                          forced_match, readout["cells"], readout.get("sha256"), max_dynamic, max_edges)
    report["rule"]["output_set"] = output_set
    report["rule"]["exclude_superclass_prefix"] = exclude_prefix or ""
    report["counts"]["n_excluded_named_immune"] = n_immune
    sub["report"] = np.array(json.dumps(report))
    return sub, report


def save_subgraph(sub: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **sub)


def load_subgraph(path: Path) -> dict:
    """subgraph_<name>.npz -> dict of arrays; 'report' is parsed to a dict."""
    z = np.load(path)
    out = {k: z[k] for k in z.files}
    out["w_min"] = float(out["w_min"])
    out["report"] = json.loads(str(out["report"]))
    return out


def vocab_check(path: Path = ANNOTATIONS) -> dict:
    """Measured type / superclass / somaSide vocabulary on a cell table (annotations feather)."""
    c = load_cells(path, ["bodyId", "type", "superclass", "somaSide"])
    types, sc, sides = c["type"], c["superclass"], c["somaSide"]
    uniq = sorted({t for t in types if isinstance(t, str)})
    lower = {t.lower(): t for t in uniq}

    def counts(mask):
        s = [norm_side(x) for x in sides[mask]]
        return {"n": int(mask.sum()), "L": s.count("L"), "R": s.count("R"), "other": s.count("other")}

    def vc(arr):
        vals, n = np.unique(np.array([x if isinstance(x, str) else "<null>" for x in arr]), return_counts=True)
        return dict(sorted(zip(vals.tolist(), n.tolist()), key=lambda kv: -kv[1]))

    def near(p):
        key = p.rstrip("*").lower()
        return [lower[t] for t in sorted(lower) if key in t][:12]

    clamped = {}
    for t in CLAMPED_TYPES:
        m = np.fromiter((x == t for x in types), dtype=bool, count=len(types))
        clamped[t] = counts(m) if m.any() else None
    forced = {}
    for p, ts in match_types(types, FORCED_PATTERNS).items():
        forced[p] = {t: counts(np.fromiter((x == t for x in types), dtype=bool, count=len(types)))
                     for t in ts} if ts else {"MISSING": True, "near": near(p)}
    readout = {}
    for t in READOUT_ROLES:
        m = np.fromiter((x == t for x in types), dtype=bool, count=len(types))
        readout[t] = ({**counts(m), "superclass": vc(sc[m])} if m.any()
                      else {"MISSING": True, "near": near(t)})
    dn = np.fromiter((s == OUTPUT_SUPERCLASS for s in sc), dtype=bool, count=len(sc))
    return {
        "source": str(path), "n_rows": int(len(types)), "n_unique_types": len(uniq),
        "n_null_type": int(sum(1 for t in types if not isinstance(t, str))),
        "superclass_counts": vc(sc), "somaSide_counts": vc(sides),
        "descending": counts(dn),
        "clamped_types": clamped,
        "clamped_missing": [t for t, v in clamped.items() if v is None],
        "clamped_total": int(sum(v["n"] for v in clamped.values() if v)),
        "forced_patterns": forced,
        "forced_missing": [p for p, v in forced.items() if "MISSING" in v],
        "readout_types": readout,
        "readout_missing": [t for t, v in readout.items() if "MISSING" in v],
        "readout_total_cells": int(sum(v["n"] for v in readout.values() if "n" in v)),
    }


def summary_lines(report: dict) -> list[str]:
    c, w, d = report["counts"], report["weight"], report["dn_reachability"]
    return [
        f"subgraph {report['name']}: w_min={report['w_min']:g} (start {report['rule']['w_min_start']:g}, "
        f"hops {report['rule']['hops']}, output_set {report['rule'].get('output_set', 'descending')})",
        f"  nodes {c['n_nodes']} = clamped {c['n_clamped']} + dynamic {c['n_dynamic']} "
        f"(core {c['n_core']}, forced-only {c['n_forced_not_in_core']}); DNs in subgraph {c['n_dn_in_subgraph']}"
        f"/{c['n_output_superclass']}; excluded superclass '{report['rule']['exclude_superclass_prefix']}' "
        f"cells {c['n_excluded_cells']} (named immune {c.get('n_excluded_named_immune', 0)})",
        f"  edges {c['n_edges']} (clamped->dyn {c['n_edges_clamped_to_dynamic']}, dyn->dyn "
        f"{c['n_edges_dynamic_to_dynamic']}, autapses {c['n_autapses']}); budget_ok={c['budget_ok']}",
        f"  weight retained: {w['frac_within_nodes']:.3f} within nodes, {w['frac_of_dynamic_inputs']:.3f} "
        f"of dynamic inputs, {w['frac_of_full']:.3f} of full graph",
        f"  DN reach <=4 hops: full {d['full_graph_4hop']['n_reached']}/{d['full_graph_4hop']['n_dn']}, "
        f"subgraph {d['subgraph']['n_reached']}/{d['subgraph']['n_dn_in_subgraph']}, readout "
        f"{d['readout']['n_reached']}/{d['readout']['n_cells']}",
        f"  forced missing: {report['forced']['missing_patterns']}",
        f"  L/R audit: {report['lr_audit']['n_flagged']}/{report['lr_audit']['n_types']} types flagged (<{report['lr_audit']['threshold']})",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graph-dir", type=Path, default=GRAPH_DIR, help="dir with graph_full.npz + neurons.parquet")
    ap.add_argument("--name", default="v1", help="writes subgraph_<name>.npz")
    ap.add_argument("--w-min", type=float, default=DEFAULT_W_MIN)
    ap.add_argument("--hops", type=int, default=HOPS)
    ap.add_argument("--max-dynamic", type=int, default=MAX_DYNAMIC)
    ap.add_argument("--max-edges", type=int, default=MAX_EDGES)
    ap.add_argument("--exclude-superclass", default=EXCLUDE_SUPERCLASS_PREFIX, metavar="PREFIXES",
                    help="comma-separated superclass prefixes removed from the candidate pool "
                         f"(default '{EXCLUDE_SUPERCLASS_PREFIX}'; named cells are never excluded)")
    ap.add_argument("--keep-vnc", action="store_true", help="shorthand for --exclude-superclass ''")
    ap.add_argument("--output-set", choices=["descending", "readout"], default="descending",
                    help="seed set for the backward BFS: all descending neurons (pre-registered) or the readout cells")
    ap.add_argument("--readout", action="store_true", help="resolve and write the readout json, then exit")
    ap.add_argument("--readout-path", type=Path, default=READOUT_PATH)
    ap.add_argument("--cells", type=Path, default=None,
                    help="cell table for --readout (default: neurons.parquet in --graph-dir, else the annotations feather)")
    ap.add_argument("--force", action="store_true", help="overwrite an existing readout json")
    ap.add_argument("--vocab", nargs="?", const=str(ANNOTATIONS), default=None, metavar="FEATHER",
                    help="measured type vocabulary check on a cell table, then exit")
    ap.add_argument("--json-out", type=Path, default=None, help="also write the report/vocab JSON here")
    ap.add_argument("--dry-run", action="store_true", help="build and report, do not write the npz")
    a = ap.parse_args(argv)

    if a.vocab is not None:
        rep = vocab_check(Path(a.vocab))
        txt = json.dumps(rep, indent=1)
        print(txt)
        if a.json_out:
            a.json_out.parent.mkdir(parents=True, exist_ok=True)
            a.json_out.write_text(txt + "\n")
        return 0

    if a.readout:
        src = a.cells or (a.graph_dir / "neurons.parquet")
        if not Path(src).exists():
            src = ANNOTATIONS
        cells, missing = resolve_readout(load_cells(src, ["bodyId", "type", "somaSide"]))
        doc = write_readout(cells, a.readout_path, missing, str(src), force=a.force)
        print(f"wrote {a.readout_path}: {len(cells)} cells, sha256 {doc['sha256']}, missing {missing}")
        return 0

    readout = load_readout(a.readout_path) if a.readout_path.exists() else None
    if readout is None:
        print(f"warning: {a.readout_path} not found; resolving readout cells on the fly (unfrozen)",
              file=sys.stderr)
    sub, report = build_subgraph(a.graph_dir, a.name, a.w_min, a.hops, a.max_dynamic, a.max_edges, readout,
                                 exclude_prefix="" if a.keep_vnc else a.exclude_superclass,
                                 output_set=a.output_set)
    print("\n".join(summary_lines(report)))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(report, indent=1) + "\n")
    if not a.dry_run:
        out = a.graph_dir / f"subgraph_{a.name}.npz"
        save_subgraph(sub, out)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
