"""Hex-column geometry of the MaleCNS optic lobes and the frozen pixel sampler.

Everything here is FROZEN (zero trainable parameters) and identical across every arm.

Approximations, stated once (the upgrade path is the published Zhao 2025 eye map):

* ``assignedOlHex1/2`` are axial coordinates on the medulla column lattice (Nern et al. 2025).
  We treat that lattice as PLANAR with a uniform inter-ommatidial angle ``SPACING_DEG`` = 5.7 deg
  (Drosophila mean; the real angle runs ~4.5 deg frontally to ~8 deg laterally).
* Lattice orientation (which lattice axis is horizontal) and the anterior/dorsal signs are
  ESTIMATED FROM THE WIRING, not assumed: along a T4 dendrite the Mi9 inputs sit at the leading
  tip and the Mi4/C3 inputs at the trailing base of the preferred-direction axis (Takemura 2017
  eLife 6:e24394; Shinomiya 2019 eLife 8:e40025; Arenz 2017 Curr Biol 27:929), and T4a/b/c/d
  prefer front-to-back / back-to-front / up / down motion (Maisak 2013 Nature 500:212). Hence
  centroid(Mi9) - centroid(Mi4+C3) over T4a cells points ANTERIOR and over T4c cells VENTRAL.
  T5 (Tm9 tip vs Tm1/Tm2/Tm4 centre) is computed as a cross-check only.
* Azimuth is measured from the midline (right positive), elevation from the eye equator, taken
  as the dorso-ventral centroid of the lattice. The frontmost column of each eye sits at
  -/+ ``OVERLAP_DEG``/2 (10 deg binocular overlap).
* Cells of clamped types without hex coordinates (all T4/T5, Tm5a-c, TmY5a, plus a few cells of
  hex-bearing types) are placed in the column of their strongest hex-bearing presynaptic partner
  (else strongest postsynaptic partner, else unassigned and silent).
* Pixel sampling: pinhole camera, ``HFOV_DEG`` horizontal, square pixels. Each column integrates
  the frame under a Gaussian acceptance function of FWHM ``FWHM_DEG`` (acceptance angle
  delta-rho ~5 deg, Gonzalez-Bellido 2011 PNAS 108:4224) truncated at 3 sigma. Acceptance mass
  that falls outside the frame, or outside the eye's half of the FOV split, is replaced at run
  time by that eye's mean luminance, never by black.

CLI: ``python -m flybrain.vision.geometry --build`` (runs on master, reads the feathers,
writes ``$FLYBRAIN_OUT/vision/geometry_v1.npz``); ``--report`` prints the summary of a saved file.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np

from flybrain import MALECNS_DIR, OUT_DIR

# The clamped input set, in the fixed channel order used by the front end.
CLAMPED_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
                 "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9",
                 "Tm20", "Tm5a", "Tm5b", "Tm5c", "TmY5a")
# Types that carry assignedOlHex1/2 in MaleCNS v1.0 (measured on the annotation feather:
# 23,720 cells, all superclass ol_intrinsic, 892 R + 879 L distinct columns).
HEX_TYPES = ("L1", "L2", "L3", "L5", "C2", "C3", "T1",
             "Mi1", "Mi4", "Mi9", "Tm1", "Tm2", "Tm4", "Tm9", "Tm20")
SIDES = ("L", "R")

SPACING_DEG = 5.7      # inter-ommatidial angle
FWHM_DEG = 5.7         # acceptance angle (FWHM of the Gaussian angular sensitivity)
HFOV_DEG = 108.0       # ViZDoom horizontal FOV
OVERLAP_DEG = 10.0     # binocular overlap band of the FOV split
FRAME_W, FRAME_H = 160, 120
TRUNC_SIGMA = 3.0
DEFAULT_PATH = OUT_DIR / "vision" / "geometry_v1.npz"

# Axial hex basis in the raw lattice plane (unit spacing): p = h1 * E1 + h2 * E2.
E1 = np.array([1.0, 0.0])
E2 = np.array([0.5, np.sqrt(3.0) / 2.0])
# Six lattice steps (dh1, dh2); the raw-plane vectors are the six nearest-neighbour directions.
LATTICE_STEPS = np.array([[1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]], dtype=np.int64)
# Neighbour slots of Geometry.nbr, defined relative to the eye's own front/up frame.
NBR_NAMES = ("front", "back", "up_front", "up_back", "down_front", "down_back")
ASSIGN_KINDS = ("own_hex", "pre_partner", "post_partner", "unassigned")


def hex_to_raw(h: np.ndarray) -> np.ndarray:
    """Axial (h1, h2) -> raw lattice plane coordinates in units of one column spacing."""
    h = np.asarray(h, dtype=np.float64)
    return h[:, :1] * E1 + h[:, 1:2] * E2


def hex_distance(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Lattice distance from the origin for the 60-degree axial basis."""
    return (np.abs(h1) + np.abs(h2) + np.abs(h1 + h2)) // 2


@dataclass
class Geometry:
    """Columns (both eyes), lattice neighbours, angles, pixel sampler, and the clamped-cell table.

    Column arrays are indexed 0..C-1 over both eyes; cell arrays 0..M-1 in the FIXED UNIT ORDER
    of the front end output (sorted by CLAMPED_TYPES order, side L then R, hex1, hex2, bodyId).
    """
    col_side: np.ndarray            # int8 [C], 0 = L, 1 = R
    col_hex: np.ndarray             # int16 [C, 2]
    col_az: np.ndarray              # float32 [C], deg from midline, right positive
    col_el: np.ndarray              # float32 [C], deg from the eye equator, up positive
    nbr: np.ndarray                 # int32 [C, 6], lattice neighbour per NBR_NAMES, -1 if absent
    coverage: np.ndarray            # float32 [C], acceptance mass inside the frame (0..1)
    driven: np.ndarray              # bool [C], coverage >= 0.5
    pix_indptr: np.ndarray          # int64 [C+1]      CSR sampler over pixels (row-major H*W)
    pix_indices: np.ndarray         # int32 [nnz]
    pix_weights: np.ndarray         # float32 [nnz], row sums == coverage
    side_gain: np.ndarray           # float32 [2], mean driven count / this side's driven count
    cell_bodyId: np.ndarray         # int64 [M]
    cell_type: np.ndarray           # unicode [M]
    cell_type_id: np.ndarray        # int16 [M], index into CLAMPED_TYPES
    cell_side: np.ndarray           # int8 [M]
    cell_col: np.ndarray            # int32 [M], column index or -1
    cell_assign: np.ndarray         # int8 [M], index into ASSIGN_KINDS
    meta: dict = field(default_factory=dict)

    # ---- derived conveniences -------------------------------------------------------------
    @property
    def n_columns(self) -> int:
        return int(self.col_side.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cell_bodyId.shape[0])

    @property
    def frame_hw(self) -> tuple[int, int]:
        return int(self.meta["frame_h"]), int(self.meta["frame_w"])

    def cell_hex(self) -> np.ndarray:
        """int16 [M, 2] hex of each cell's column (-1 where unassigned)."""
        out = np.full((self.n_cells, 2), -1, dtype=np.int16)
        ok = self.cell_col >= 0
        out[ok] = self.col_hex[self.cell_col[ok]]
        return out

    def summary(self) -> dict:
        s = {"n_columns": self.n_columns, "n_cells": self.n_cells}
        for si, sd in enumerate(SIDES):
            m = self.col_side == si
            s[f"columns_{sd}"] = int(m.sum())
            s[f"driven_{sd}"] = int((m & self.driven).sum())
            s[f"side_gain_{sd}"] = float(self.side_gain[si])
        s["assign_counts"] = {k: int((self.cell_assign == i).sum()) for i, k in enumerate(ASSIGN_KINDS)}
        per_type = {}
        for ti, ty in enumerate(CLAMPED_TYPES):
            m = self.cell_type_id == ti
            if m.any():
                cols = self.cell_col[m]; ok = cols >= 0
                per_type[ty] = {"n": int(m.sum()), "assigned": int(ok.sum()),
                                "driven": int(self.driven[cols[ok]].sum())}
        s["per_type"] = per_type
        s.update({k: v for k, v in self.meta.items() if not isinstance(v, (list, dict))})
        return s

    # ---- persistence ----------------------------------------------------------------------
    def save(self, path) -> None:
        path = __import__("pathlib").Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in self.__dict__.items() if k != "meta"}
        np.savez_compressed(path, meta=json.dumps(self.meta), **arrays)

    @classmethod
    def load(cls, path=DEFAULT_PATH) -> "Geometry":
        with np.load(path, allow_pickle=False) as z:
            kw = {k: z[k] for k in z.files if k != "meta"}
            meta = json.loads(str(z["meta"]))
        return cls(meta=meta, **kw)

    # ---- constructors ---------------------------------------------------------------------
    @classmethod
    def synthetic(cls, radius: int = 16, hfov_deg: float = HFOV_DEG, frame_w: int = FRAME_W,
                  frame_h: int = FRAME_H, types=CLAMPED_TYPES) -> "Geometry":
        """Hex disc of `radius` per eye, one cell of every clamped type per column (tests, benches)."""
        r = np.arange(-radius, radius + 1)
        h1, h2 = np.meshgrid(r, r, indexing="ij")
        keep = hex_distance(h1, h2) <= radius
        hexes = np.stack([h1[keep], h2[keep]], 1).astype(np.int16)
        side = np.concatenate([np.zeros(len(hexes), np.int8), np.ones(len(hexes), np.int8)])
        col_hex = np.concatenate([hexes, hexes])
        frames = {0: (-E1, np.array([0.0, 1.0])), 1: (-E1, np.array([0.0, 1.0]))}
        body, ctype_id, cside, ccol = [], [], [], []
        for ti, ty in enumerate(types):
            for si in (0, 1):
                idx = np.nonzero(side == si)[0]
                body.append(1_000_000 * (ti + 1) + 100_000 * si + np.arange(len(idx)))
                ctype_id.append(np.full(len(idx), ti, np.int16)); cside.append(np.full(len(idx), si, np.int8)); ccol.append(idx)
        body = np.concatenate(body); ctype_id = np.concatenate(ctype_id); cside = np.concatenate(cside); ccol = np.concatenate(ccol)
        meta = {"source": "synthetic", "radius": radius}
        return _assemble(side, col_hex, frames, body, ctype_id, cside, ccol,
                         np.zeros(len(body), np.int8), meta, hfov_deg, frame_w, frame_h)

    @classmethod
    def from_malecns(cls, annotations=None, weights=None, hfov_deg: float = HFOV_DEG,
                     frame_w: int = FRAME_W, frame_h: int = FRAME_H, verbose: bool = True) -> "Geometry":
        """Build from the MaleCNS feathers (master only: reads the 152M-row weights file)."""
        import pyarrow.feather as pf
        annotations = annotations or MALECNS_DIR / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        weights = weights or MALECNS_DIR / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        t0 = time.time()
        ann = pf.read_table(annotations, columns=["bodyId", "type", "somaSide", "assignedOlHex1", "assignedOlHex2"]).to_pandas()
        ann = ann[ann["type"].isin(set(CLAMPED_TYPES) | set(HEX_TYPES)) & ann["somaSide"].isin(SIDES)].copy()
        ann["has_hex"] = ann["assignedOlHex1"].notna() & ann["assignedOlHex2"].notna()
        ann["side_i"] = (ann["somaSide"] == "R").astype(np.int8)
        hexcells = ann[ann["has_hex"]]
        # columns = distinct (side, h1, h2)
        key = np.stack([hexcells["side_i"].to_numpy(), hexcells["assignedOlHex1"].to_numpy().astype(int),
                        hexcells["assignedOlHex2"].to_numpy().astype(int)], 1)
        ukey, inv = np.unique(key, axis=0, return_inverse=True)
        col_side = ukey[:, 0].astype(np.int8)
        col_hex = ukey[:, 1:].astype(np.int16)
        hex_body = hexcells["bodyId"].to_numpy()
        hex_col = inv.ravel().astype(np.int32)
        hex_type = hexcells["type"].to_numpy()
        if verbose:
            print(f"annotations: {len(ann)} cells of interest, {len(hexcells)} with hex, "
                  f"{(col_side == 0).sum()} L + {(col_side == 1).sum()} R columns ({time.time() - t0:.1f}s)")

        # edges touching the hex-bearing cells, from the weights feather (one pass, ~10 s)
        clamped = ann[ann["type"].isin(CLAMPED_TYPES)]
        need = clamped.loc[~clamped["has_hex"], "bodyId"].to_numpy()
        t0 = time.time()
        wt = pf.read_table(weights, columns=["body_pre", "body_post", "weight"])
        pre = wt.column("body_pre").to_numpy(); post = wt.column("body_post").to_numpy()
        w = wt.column("weight").to_numpy().astype(np.float32); del wt
        in_hex_pre = _isin(pre, hex_body); in_hex_post = _isin(post, hex_body)
        in_need_post = _isin(post, need); in_need_pre = _isin(pre, need)
        e_in = in_hex_pre & in_need_post        # hex cell -> hexless clamped cell
        e_out = in_need_pre & in_hex_post       # hexless clamped cell -> hex cell
        edges_in = (pre[e_in], post[e_in], w[e_in])
        edges_out = (pre[e_out], post[e_out], w[e_out])
        del pre, post, w, in_hex_pre, in_hex_post, in_need_post, in_need_pre, e_in, e_out
        if verbose:
            print(f"weights: {len(edges_in[0])} input edges, {len(edges_out[0])} output edges "
                  f"onto {len(need)} hex-less clamped cells ({time.time() - t0:.1f}s)")

        # lattice frame per side from T4 wiring (Mi9 tip vs Mi4/C3 base)
        body_to_col = dict(zip(hex_body.tolist(), hex_col.tolist()))
        body_to_type = dict(zip(hex_body.tolist(), hex_type.tolist()))
        t4 = clamped[clamped["type"].str.startswith("T4") | clamped["type"].str.startswith("T5")]
        frames, evidence = estimate_frames(edges_in, t4, body_to_col, body_to_type, col_side, col_hex, verbose)

        # column assignment for every clamped cell
        cbody = clamped["bodyId"].to_numpy(); ctype = clamped["type"].to_numpy()
        cside = clamped["side_i"].to_numpy().astype(np.int8)
        ccol = np.full(len(cbody), -1, np.int32); cassign = np.full(len(cbody), 3, np.int8)
        own = clamped["has_hex"].to_numpy()
        okey = np.stack([cside[own], clamped.loc[own, "assignedOlHex1"].to_numpy().astype(int),
                         clamped.loc[own, "assignedOlHex2"].to_numpy().astype(int)], 1)
        ccol[own] = _lookup_rows(ukey, okey); cassign[own] = 0
        by_pre = _strongest_partner(edges_in[1], edges_in[0], edges_in[2])     # post -> best pre
        by_post = _strongest_partner(edges_out[0], edges_out[1], edges_out[2])  # pre -> best post
        for i in np.nonzero(~own)[0]:
            b = int(cbody[i])
            if b in by_pre:
                ccol[i] = body_to_col[by_pre[b]]; cassign[i] = 1
            elif b in by_post:
                ccol[i] = body_to_col[by_post[b]]; cassign[i] = 2
        # a partner-derived column must be on the cell's own side; otherwise treat as unassigned
        bad = (ccol >= 0) & (col_side[np.maximum(ccol, 0)] != cside)
        ccol[bad] = -1; cassign[bad] = 3
        ctype_id = np.array([CLAMPED_TYPES.index(t) for t in ctype], np.int16)
        meta = {"source": "malecns_v1.0_minconf0.5", "frame_evidence": evidence,
                "n_cross_side_partner_rejected": int(bad.sum())}
        return _assemble(col_side, col_hex, frames, cbody, ctype_id, cside, ccol, cassign, meta,
                         hfov_deg, frame_w, frame_h)


# ---- building blocks ----------------------------------------------------------------------

def _isin(x: np.ndarray, ids: np.ndarray) -> np.ndarray:
    ids = np.unique(ids)
    span = int(ids.max() - ids.min()) + 1 if len(ids) else 0
    return np.isin(x, ids, kind="table" if 0 < span < 2**30 else None)


def _lookup_rows(table: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Row index in `table` (unique, sorted rows) of each row of `query`; -1 if absent."""
    d = {tuple(r): i for i, r in enumerate(table.tolist())}
    return np.array([d.get(tuple(q), -1) for q in query.tolist()], np.int32)


def _strongest_partner(key: np.ndarray, partner: np.ndarray, w: np.ndarray) -> dict:
    """For each key, the partner with the largest summed weight."""
    if len(key) == 0:
        return {}
    pair = np.stack([key, partner], 1)
    upair, inv = np.unique(pair, axis=0, return_inverse=True)
    tot = np.bincount(inv.ravel(), weights=w, minlength=len(upair))
    order = np.lexsort((-tot, upair[:, 0]))
    upair, tot = upair[order], tot[order]
    first = np.r_[True, upair[1:, 0] != upair[:-1, 0]]
    return dict(zip(upair[first, 0].tolist(), upair[first, 1].tolist()))


def estimate_frames(edges_in, t4t5, body_to_col: dict, body_to_type: dict, col_side, col_hex, verbose=True):
    """Per-side (front, up) unit vectors in the raw lattice plane from T4 input geometry.

    d_sub = centroid(Mi9 inputs) - centroid(Mi4+C3 inputs) per T4 cell, averaged per subtype.
    front = normalise(d_a - d_b) snapped to the nearest lattice axis; up = perp(front) with the
    sign of (d_d - d_c). Falls back to front = -E1, up = +y if the evidence is missing.
    """
    pre, post, w = edges_in
    pre_col = np.array([body_to_col.get(int(b), -1) for b in pre]); ok = pre_col >= 0
    pre_type = np.array([body_to_type.get(int(b), "") for b in pre])
    pre, post, w, pre_col, pre_type = pre[ok], post[ok], w[ok], pre_col[ok], pre_type[ok]
    raw = hex_to_raw(col_hex)
    cell_side = dict(zip(t4t5["bodyId"].tolist(), t4t5["side_i"].tolist()))
    cell_type = dict(zip(t4t5["bodyId"].tolist(), t4t5["type"].tolist()))
    post_side = np.array([cell_side.get(int(b), -1) for b in post])
    post_type = np.array([cell_type.get(int(b), "") for b in post])
    groups = {"T4": ("Mi9", ("Mi4", "C3")), "T5": ("Tm9", ("Tm1", "Tm2", "Tm4"))}
    frames, evidence = {}, {}
    for si in (0, 1):
        ev = {}
        for fam, (tip, base) in groups.items():
            for sub in "abcd":
                ty = fam + sub
                m = (post_side == si) & (post_type == ty)
                d = _mean_offset(post[m], pre_type[m], pre_col[m], w[m], raw, tip, base)
                ev[ty] = None if d is None else [float(d[0]), float(d[1])]
        frames[si], ev["diagnostics"] = _frame_from_offsets(ev)
        evidence[SIDES[si]] = ev
        if verbose:
            print(f"side {SIDES[si]}: front={np.round(frames[si][0], 3).tolist()} up={np.round(frames[si][1], 3).tolist()} "
                  f"diag={ev['diagnostics']}")
    return frames, evidence


def _mean_offset(post, pre_type, pre_col, w, raw, tip, base):
    """Mean over cells of weighted centroid(tip inputs) - weighted centroid(base inputs)."""
    out = []
    for b in np.unique(post):
        m = post == b
        mt = m & (pre_type == tip); mb = m & np.isin(pre_type, base)
        if mt.any() and mb.any():
            ct = (raw[pre_col[mt]] * w[mt, None]).sum(0) / w[mt].sum()
            cb = (raw[pre_col[mb]] * w[mb, None]).sum(0) / w[mb].sum()
            out.append(ct - cb)
    return None if not out else np.mean(out, 0)


def _frame_from_offsets(ev: dict):
    steps = hex_to_raw(LATTICE_STEPS)
    diag = {"fallback": False}
    if any(ev.get("T4" + s) is None for s in "abcd"):
        diag["fallback"] = True
        return (-E1.copy(), np.array([0.0, 1.0])), diag
    da, db, dc, dd = (np.array(ev["T4" + s]) for s in "abcd")
    f_raw = (da - db) / 2.0; u_raw = (dd - dc) / 2.0
    diag["front_raw_len"] = float(np.linalg.norm(f_raw)); diag["up_raw_len"] = float(np.linalg.norm(u_raw))
    fr = f_raw / (np.linalg.norm(f_raw) + 1e-12); ur = u_raw / (np.linalg.norm(u_raw) + 1e-12)
    diag["front_up_cos"] = float(fr @ ur)
    k = int(np.argmax(steps @ fr))
    front = steps[k] / np.linalg.norm(steps[k])
    diag["snap_deg"] = float(np.degrees(np.arccos(np.clip(front @ fr, -1, 1))))
    up = np.array([-front[1], front[0]])
    if up @ ur < 0:
        up = -up
    diag["up_snap_deg"] = float(np.degrees(np.arccos(np.clip(up @ ur, -1, 1))))
    if all(ev.get("T5" + s) is not None for s in "abcd"):
        f5 = (np.array(ev["T5a"]) - np.array(ev["T5b"])) / 2.0
        u5 = (np.array(ev["T5d"]) - np.array(ev["T5c"])) / 2.0
        diag["t5_front_cos"] = float(f5 @ front / (np.linalg.norm(f5) + 1e-12))
        diag["t5_up_cos"] = float(u5 @ up / (np.linalg.norm(u5) + 1e-12))
    if diag["snap_deg"] > 20.0 or abs(diag["front_up_cos"]) > 0.5:
        diag["fallback"] = True
        return (-E1.copy(), np.array([0.0, 1.0])), diag
    return (front, up), diag


def _assemble(col_side, col_hex, frames, cbody, ctype_id, cside, ccol, cassign, meta,
              hfov_deg, frame_w, frame_h) -> Geometry:
    C = len(col_side)
    raw = hex_to_raw(col_hex)
    col_az = np.zeros(C, np.float32); col_el = np.zeros(C, np.float32)
    nbr = np.full((C, 6), -1, np.int32)
    lut = {(int(s), int(a), int(b)): i for i, (s, (a, b)) in enumerate(zip(col_side.tolist(), col_hex.tolist()))}
    steps_raw = hex_to_raw(LATTICE_STEPS)
    for si in (0, 1):
        front, up = frames[si]
        m = col_side == si
        a = -(raw[m] @ front)          # posterior coordinate, columns
        e = raw[m] @ up                # dorsal coordinate, columns
        a = a - a.min(); e = e - e.mean()
        sgn = 1.0 if si == 1 else -1.0
        col_az[m] = sgn * (-OVERLAP_DEG / 2.0 + a * SPACING_DEG)
        col_el[m] = e * SPACING_DEG
        # neighbour slots: classify the six lattice steps in this eye's frame
        comp = np.stack([steps_raw @ front, steps_raw @ up], 1)
        slot = np.empty(6, np.int64)
        for k, (cf, cu) in enumerate(comp):
            if abs(cu) < 1e-6:
                slot[k] = 0 if cf > 0 else 1
            elif cu > 0:
                slot[k] = 2 if cf > 0 else 3
            else:
                slot[k] = 4 if cf > 0 else 5
        for i in np.nonzero(m)[0]:
            h1, h2 = int(col_hex[i, 0]), int(col_hex[i, 1])
            for k, (d1, d2) in enumerate(LATTICE_STEPS.tolist()):
                j = lut.get((si, h1 + d1, h2 + d2), -1)
                nbr[i, slot[k]] = j
    indptr, indices, weights, coverage = build_sampler(col_az, col_el, col_side, hfov_deg, frame_w, frame_h)
    driven = coverage >= 0.5
    n_driven = np.array([(driven & (col_side == s)).sum() for s in (0, 1)], np.float64)
    side_gain = (n_driven.mean() / np.maximum(n_driven, 1)).astype(np.float32)
    # fixed unit order: type, side, hex1, hex2, bodyId (unassigned cells sort after assigned ones)
    hx = np.where(ccol[:, None] >= 0, col_hex[np.maximum(ccol, 0)], np.int16(9999))
    order = np.lexsort((cbody, hx[:, 1], hx[:, 0], cside, ctype_id))
    cell_type = np.array(CLAMPED_TYPES, dtype="U8")[ctype_id[order]]
    meta = dict(meta, spacing_deg=SPACING_DEG, fwhm_deg=FWHM_DEG, hfov_deg=hfov_deg,
                vfov_deg=float(2 * np.degrees(np.arctan((frame_h / 2) / ((frame_w / 2) / np.tan(np.radians(hfov_deg / 2)))))),
                overlap_deg=OVERLAP_DEG, frame_w=frame_w, frame_h=frame_h, trunc_sigma=TRUNC_SIGMA,
                frames={SIDES[s]: {"front": frames[s][0].tolist(), "up": frames[s][1].tolist()} for s in (0, 1)},
                nbr_names=list(NBR_NAMES), assign_kinds=list(ASSIGN_KINDS), clamped_types=list(CLAMPED_TYPES))
    return Geometry(col_side=col_side.astype(np.int8), col_hex=col_hex.astype(np.int16), col_az=col_az, col_el=col_el,
                    nbr=nbr, coverage=coverage.astype(np.float32), driven=driven,
                    pix_indptr=indptr, pix_indices=indices, pix_weights=weights, side_gain=side_gain,
                    cell_bodyId=cbody[order].astype(np.int64), cell_type=cell_type, cell_type_id=ctype_id[order],
                    cell_side=cside[order].astype(np.int8), cell_col=ccol[order].astype(np.int32),
                    cell_assign=cassign[order].astype(np.int8), meta=meta)


def pixel_directions(hfov_deg: float, w: int, h: int):
    """Unit view direction [H*W, 3] (x right, y up, z forward) and solid angle [H*W] per pixel."""
    f = (w / 2.0) / np.tan(np.radians(hfov_deg / 2.0))
    px = (np.arange(w) + 0.5 - w / 2.0) / f
    py = -(np.arange(h) + 0.5 - h / 2.0) / f
    xn, yn = np.meshgrid(px, py, indexing="xy")          # [H, W]
    xn, yn = xn.ravel(), yn.ravel()
    r2 = 1.0 + xn ** 2 + yn ** 2
    d = np.stack([xn, yn, np.ones_like(xn)], 1) / np.sqrt(r2)[:, None]
    domega = 1.0 / (f ** 2 * r2 ** 1.5)
    return d, domega


def build_sampler(col_az, col_el, col_side, hfov_deg=HFOV_DEG, w=FRAME_W, h=FRAME_H,
                  fwhm_deg=FWHM_DEG, overlap_deg=OVERLAP_DEG, trunc_sigma=TRUNC_SIGMA):
    """CSR Gaussian pixel sampler per column plus per-column coverage in [0, 1].

    Weights are normalised by the analytic acceptance mass, so a fully visible column has row sum
    ~1 (clipped to 1); the missing mass (1 - coverage) is filled with the eye mean at run time.
    The FOV split is a linear ramp over the overlap band: left eye weight 1 at az <= -ov/2 ... 0
    at az >= +ov/2, mirrored for the right eye.
    """
    d, domega = pixel_directions(hfov_deg, w, h)
    sigma = np.radians(fwhm_deg) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    mass = 2.0 * np.pi * sigma ** 2 * (1.0 - np.exp(-trunc_sigma ** 2 / 2.0))
    az = np.radians(col_az.astype(np.float64)); el = np.radians(col_el.astype(np.float64))
    dc = np.stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], 1)
    half = np.radians(hfov_deg / 2.0) + trunc_sigma * sigma
    vhalf = np.arctan((h / 2.0) / ((w / 2.0) / np.tan(np.radians(hfov_deg / 2.0)))) + trunc_sigma * sigma
    cand = (np.abs(az) <= half) & (np.abs(el) <= vhalf)
    split = np.where(col_side == 0, np.clip((overlap_deg / 2.0 - col_az) / overlap_deg, 0, 1),
                     np.clip((overlap_deg / 2.0 + col_az) / overlap_deg, 0, 1)).astype(np.float64)
    C = len(col_az)
    lengths = np.zeros(C, np.int64); idx_list, w_list = [], []
    coverage = np.zeros(C, np.float64)
    for i in range(C):
        if not (cand[i] and split[i] > 0):
            continue
        ang = np.arccos(np.clip(d @ dc[i], -1.0, 1.0))
        sel = np.nonzero(ang < trunc_sigma * sigma)[0]
        if len(sel) == 0:
            continue
        wts = np.exp(-0.5 * (ang[sel] / sigma) ** 2) * domega[sel] / mass * split[i]
        cov = float(wts.sum())
        if cov > 1.0:
            wts /= cov; cov = 1.0
        coverage[i] = cov; lengths[i] = len(sel)
        idx_list.append(sel.astype(np.int32)); w_list.append(wts.astype(np.float32))
    indptr = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    indices = np.concatenate(idx_list) if idx_list else np.zeros(0, np.int32)
    weights = np.concatenate(w_list) if w_list else np.zeros(0, np.float32)
    return indptr, indices, weights, coverage


# ---- CLI ----------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build / inspect the frozen hex-column geometry")
    ap.add_argument("--build", action="store_true", help="build from the MaleCNS feathers (master)")
    ap.add_argument("--synthetic", type=int, default=0, help="build a synthetic disc of this radius instead")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--report", action="store_true", help="print the summary of --out")
    args = ap.parse_args(argv)
    if args.build or args.synthetic:
        t0 = time.time()
        g = Geometry.synthetic(args.synthetic) if args.synthetic else Geometry.from_malecns()
        g.save(args.out)
        print(f"saved {args.out} ({time.time() - t0:.1f}s)")
        print(json.dumps(g.summary(), indent=1))
    elif args.report:
        print(json.dumps(Geometry.load(args.out).summary(), indent=1))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
