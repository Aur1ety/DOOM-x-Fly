"""Doom map geometry from a WAD (vanilla format) and a geodesic distance-to-exit field.

Used for (1) reward shaping on real levels (progress = decrease in walking distance to the exit),
(2) the privileged scripted navigator that serves as the imitation teacher. The agent itself never
sees any of this; it only sees rendered frames.

    python -m flybrain.env.wadmap --wad $FLYBRAIN_DATA/wads/doom1.wad --map E1M1 --png out.png

Passability approximations (stated): the player is a 16-unit-radius, 56-unit-tall body that can
step up 24 units; a two-sided line is walkable if both sides have >= 56 units of floor-to-ceiling
gap after DOOR sectors (back side of a manual door line) are treated as open (ceiling = lowest
neighbouring ceiling - 4); one-sided lines and lines flagged "impassable" block. Damaging floors
(nukage) are walkable at a higher cost.
"""
from __future__ import annotations

import argparse
import heapq
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXIT_SPECIALS = {11, 51, 52, 124, 197, 198}           # S1 exit, S1 secret exit, W1 exit, W1 secret exit (+Boom)
DOOR_SPECIALS = {1, 26, 27, 28, 31, 32, 33, 34, 117, 118}   # manual (push) doors, incl. keyed
DAMAGE_SECTORS = {4, 5, 7, 11, 16}
PLAYER_H, STEP_H, RADIUS = 56, 24, 16


@dataclass
class DoomMap:
    name: str
    vertices: np.ndarray        # [V, 2] int
    lines: np.ndarray           # [L, 7] v1, v2, flags, special, tag, front_side, back_side (-1 = none)
    sides: np.ndarray           # [S] sector index per sidedef
    sectors: np.ndarray         # [Z, 4] floor, ceiling, special, tag
    things: np.ndarray          # [T, 5] x, y, angle, type, flags
    meta: dict = field(default_factory=dict)

    @property
    def player_start(self) -> tuple[float, float, float]:
        t = self.things[self.things[:, 3] == 1][0]
        return float(t[0]), float(t[1]), float(t[2])

    def line_xy(self, i: int) -> np.ndarray:
        v1, v2 = self.lines[i, 0], self.lines[i, 1]
        return np.array([*self.vertices[v1], *self.vertices[v2]], float)

    def exit_lines(self) -> list[int]:
        return [i for i in range(len(self.lines)) if int(self.lines[i, 3]) in EXIT_SPECIALS]

    def door_lines(self) -> list[int]:
        return [i for i in range(len(self.lines)) if int(self.lines[i, 3]) in DOOR_SPECIALS]


def read_map(wad: Path, name: str) -> DoomMap:
    data = Path(wad).read_bytes()
    ident, n_lumps, dir_ofs = struct.unpack_from("<4sii", data, 0)
    lumps = []
    for i in range(n_lumps):
        ofs, size, nm = struct.unpack_from("<ii8s", data, dir_ofs + 16 * i)
        lumps.append((nm.rstrip(b"\0").decode("ascii", "replace"), ofs, size))
    idx = next(i for i, l in enumerate(lumps) if l[0] == name)
    parts = {}
    for nm, ofs, size in lumps[idx + 1: idx + 11]:
        parts[nm] = data[ofs: ofs + size]

    def arr(lump, fmt, rec):
        raw = parts[lump]
        return np.array([struct.unpack_from(fmt, raw, o) for o in range(0, len(raw) - rec + 1, rec)])

    vertices = arr("VERTEXES", "<hh", 4)
    ld = arr("LINEDEFS", "<HHHHHHH", 14).astype(np.int64)
    ld[:, 5] = np.where(ld[:, 5] == 0xFFFF, -1, ld[:, 5]); ld[:, 6] = np.where(ld[:, 6] == 0xFFFF, -1, ld[:, 6])
    sd_raw = parts["SIDEDEFS"]
    sides = np.array([struct.unpack_from("<hh8s8s8sH", sd_raw, o)[5] for o in range(0, len(sd_raw), 30)])
    se_raw = parts["SECTORS"]
    sectors = np.array([(lambda r: (r[0], r[1], r[5], r[6]))(struct.unpack_from("<hh8s8shhh", se_raw, o)) for o in range(0, len(se_raw), 26)])
    things = arr("THINGS", "<hhhhh", 10)
    return DoomMap(name, vertices, ld, sides, sectors, things, {"wad": str(wad), "ident": ident.decode()})


def _open_heights(m: DoomMap) -> np.ndarray:
    """[Z, 2] floor, ceiling with manual-door sectors treated as open."""
    fc = m.sectors[:, :2].astype(float).copy()
    for i in m.door_lines():
        b = m.lines[i, 6]
        if b < 0:
            continue
        z = m.sides[b]
        neigh = []
        for j in range(len(m.lines)):
            f, bb = m.lines[j, 5], m.lines[j, 6]
            if f < 0 or bb < 0:
                continue
            zf, zb = m.sides[f], m.sides[bb]
            if zf == z and zb != z:
                neigh.append(m.sectors[zb, 1])
            elif zb == z and zf != z:
                neigh.append(m.sectors[zf, 1])
        if neigh:
            fc[z, 1] = max(fc[z, 1], min(neigh) - 4)
    return fc


def blocking_segments(m: DoomMap, with_ledges: bool = False):
    """[N, 4] segments the player cannot cross (walls, impassable lines, too-low gaps, and - unless
    `with_ledges` - too-high steps in either direction). With `with_ledges`, returns (walls [N, 4],
    ledges [L, 4], ledge_floors [L, 2] = floor on the front / back side): a ledge blocks stepping UP by
    more than STEP_H but the player can drop down it."""
    fc = _open_heights(m)
    segs, ledges, lf = [], [], []
    for i in range(len(m.lines)):
        v1, v2, flags, special, tag, f, b = m.lines[i]
        xy = [*m.vertices[v1], *m.vertices[v2]]
        if b < 0 or (flags & 1):
            segs.append(xy); continue
        zf, zb = m.sides[f], m.sides[b]
        floor = max(fc[zf, 0], fc[zb, 0]); ceil = min(fc[zf, 1], fc[zb, 1])
        step = abs(fc[zf, 0] - fc[zb, 0])
        if ceil - floor < PLAYER_H:
            segs.append(xy)
        elif step > STEP_H:
            if with_ledges:
                ledges.append(xy); lf.append((fc[zf, 0], fc[zb, 0]))
            else:
                segs.append(xy)
    if with_ledges:
        return np.asarray(segs, float), np.asarray(ledges, float).reshape(-1, 4), np.asarray(lf, float).reshape(-1, 2)
    return np.asarray(segs, float)


def _crosses(p: np.ndarray, q: np.ndarray, segs: np.ndarray) -> np.ndarray:
    """Vectorised: does segment p->q (arrays [E,2]) intersect any of segs [N,4]? -> bool [E]."""
    out = np.zeros(len(p), bool)
    a, b = segs[:, :2], segs[:, 2:]
    for s in range(0, len(p), 4096):
        P, Q = p[s:s + 4096, None, :], q[s:s + 4096, None, :]
        r = Q - P; sv = (b - a)[None]
        denom = r[..., 0] * sv[..., 1] - r[..., 1] * sv[..., 0]
        qp = a[None] - P
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[..., 0] * sv[..., 1] - qp[..., 1] * sv[..., 0]) / denom
            u = (qp[..., 0] * r[..., 1] - qp[..., 1] * r[..., 0]) / denom
        hit = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        out[s:s + 4096] = hit.any(1)
    return out


@dataclass
class NavField:
    origin: np.ndarray      # map coords of cell (0, 0) centre
    cell: float
    dist: np.ndarray        # [H, W] walking distance to the goal (inf = unreachable)
    goal: np.ndarray        # map coords of the goal point
    start: np.ndarray
    exit_line: int
    block_out: np.ndarray | None = None   # [H, W, 8] bool: cannot move from the cell to neighbour k (NBRS order)

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        c = np.floor((np.array([x, y]) - self.origin) / self.cell + 0.5).astype(int)
        return int(c[1]), int(c[0])

    def distance(self, x: float, y: float) -> float:
        r, c = self.cell_of(x, y)
        if 0 <= r < self.dist.shape[0] and 0 <= c < self.dist.shape[1]:
            return float(self.dist[r, c])
        return float("inf")

    def next_waypoint(self, x: float, y: float, lookahead: int = 4) -> np.ndarray:
        """Follow steepest descent over passable moves for `lookahead` cells; return that cell's map coordinates."""
        r, c = self.cell_of(x, y)
        H, W = self.dist.shape
        for _ in range(lookahead):
            best = (self.dist[r, c] if 0 <= r < H and 0 <= c < W else np.inf, r, c)
            for k, (dr, dc) in enumerate(NBRS):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W and 0 <= r < H and 0 <= c < W):
                    continue
                if self.block_out is not None and self.block_out[r, c, k]:
                    continue
                if self.dist[rr, cc] < best[0]:
                    best = (self.dist[rr, cc], rr, cc)
            if (best[1], best[2]) == (r, c):
                break
            r, c = best[1], best[2]
        return self.origin + np.array([c, r]) * self.cell


NBRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def _ledge_blocks(p: np.ndarray, q: np.ndarray, ledges: np.ndarray, lf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For moves p->q and q->p ([E, 2] each): blocked because the move crosses a ledge upwards."""
    pq = np.zeros(len(p), bool); qp = np.zeros(len(p), bool)
    for s in range(len(ledges)):
        seg = ledges[s:s + 1]
        hit = _crosses(p, q, seg)
        if not hit.any():
            continue
        a, d = seg[0, :2], seg[0, 2:] - seg[0, :2]
        cross = d[0] * (p[:, 1] - a[1]) - d[1] * (p[:, 0] - a[0])
        p_front = cross < 0                          # Doom front side = right of v1->v2 (y up)
        f_floor, b_floor = lf[s]
        start = np.where(p_front, f_floor, b_floor); end = np.where(p_front, b_floor, f_floor)
        pq |= hit & (end - start > STEP_H)
        qp |= hit & (start - end > STEP_H)
    return pq, qp


def build_nav(m: DoomMap, cell: float = 16.0, nukage_cost: float = 3.0) -> NavField:
    segs, ledges, lf = blocking_segments(m, with_ledges=True)
    lo = m.vertices.min(0) - cell; hi = m.vertices.max(0) + cell
    W = int(np.ceil((hi[0] - lo[0]) / cell)) + 1; H = int(np.ceil((hi[1] - lo[1]) / cell)) + 1
    xs = lo[0] + np.arange(W) * cell; ys = lo[1] + np.arange(H) * cell
    gx, gy = np.meshgrid(xs, ys)
    centres = np.stack([gx.ravel(), gy.ravel()], 1)
    # passable moves (8-neighbourhood); the body radius is approximated by also testing offset parallels.
    # walls block both directions; ledges block only the upward direction (the player can drop down)
    moves = [(0, 1), (1, 0), (1, 1), (1, -1)]
    blocked_pq, blocked_qp = {}, {}
    for dr, dc in moves:
        p = centres; q = centres + np.array([dc, dr]) * cell
        d = q - p; nrm = np.stack([-d[:, 1], d[:, 0]], 1); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        wall = np.zeros(len(p), bool); up_pq = np.zeros(len(p), bool); up_qp = np.zeros(len(p), bool)
        for off in (0.0, RADIUS * 0.75, -RADIUS * 0.75):
            P, Q = p + nrm * off, q + nrm * off
            wall |= _crosses(P, Q, segs)
            a_, b_ = _ledge_blocks(P, Q, ledges, lf)
            up_pq |= a_; up_qp |= b_
        blocked_pq[(dr, dc)] = (wall | up_pq).reshape(H, W)
        blocked_qp[(dr, dc)] = (wall | up_qp).reshape(H, W)
    block_out = np.ones((H, W, 8), bool)
    for k, (dr, dc) in enumerate(NBRS):
        if (dr, dc) in blocked_pq:                   # move (r,c)->(r+dr,c+dc) is a canonical p->q
            block_out[:, :, k] = blocked_pq[(dr, dc)]
        else:                                        # it is q->p of the canonical move from (r+dr,c+dc)
            src = blocked_qp[(-dr, -dc)]
            sh = np.ones((H, W), bool)
            r0, r1 = max(0, -dr), H - max(0, dr); c0, c1 = max(0, -dc), W - max(0, dc)
            sh[r0:r1, c0:c1] = src[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            block_out[:, :, k] = sh
    # damaging floors: point-in-sector is expensive; approximate with a per-cell cost from the nearest
    # damaging-sector line midpoints (cells within 64 units of a nukage boundary cost more)
    dmg_sectors = {z for z in range(len(m.sectors)) if int(m.sectors[z, 2]) in DAMAGE_SECTORS}
    cost = np.ones((H, W))
    if dmg_sectors:
        mids = []
        for i in range(len(m.lines)):
            f, b = m.lines[i, 5], m.lines[i, 6]
            zs = {m.sides[f]} | ({m.sides[b]} if b >= 0 else set())
            if zs & dmg_sectors:
                xy = m.line_xy(i); mids.append((xy[:2] + xy[2:]) / 2)
        mids = np.asarray(mids)
        if len(mids):
            dmin = np.min(np.linalg.norm(centres[:, None, :] - mids[None, :, :], axis=2), axis=1).reshape(H, W)
            cost = np.where(dmin < 64, nukage_cost, 1.0)
    # goal: in front of the exit switch
    ex = m.exit_lines()
    if not ex:
        raise ValueError(f"{m.name}: no exit line found")
    i = ex[0]; xy = m.line_xy(i)
    mid = (xy[:2] + xy[2:]) / 2; d = xy[2:] - xy[:2]
    front = np.array([d[1], -d[0]]) / np.linalg.norm(d)       # Doom: front side is to the right of v1->v2
    goal = mid + front * 32
    gr, gc = int(round((goal[1] - lo[1]) / cell)), int(round((goal[0] - lo[0]) / cell))
    dist = np.full((H, W), np.inf); dist[gr, gc] = 0.0
    pq = [(0.0, gr, gc)]
    kidx = {o: k for k, o in enumerate(NBRS)}
    while pq:
        dcur, r, c = heapq.heappop(pq)
        if dcur > dist[r, c]:
            continue
        for dr, dc in NBRS:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if block_out[rr, cc, kidx[(-dr, -dc)]]:  # the player would move (rr,cc) -> (r,c)
                continue
            step = cell * (1.4142 if dr and dc else 1.0) * cost[rr, cc]
            nd = dcur + step
            if nd < dist[rr, cc]:
                dist[rr, cc] = nd; heapq.heappush(pq, (nd, rr, cc))
    sx, sy, _ = m.player_start
    return NavField(origin=lo.astype(float), cell=cell, dist=dist, goal=goal, start=np.array([sx, sy]), exit_line=i, block_out=block_out)


def render_png(m: DoomMap, nav: NavField, path: Path, route: np.ndarray | None = None) -> None:
    import imageio.v2 as iio
    H, W = nav.dist.shape
    scale = 4
    img = np.zeros((H * scale, W * scale, 3), np.uint8)
    finite = np.isfinite(nav.dist)
    if finite.any():
        dn = np.zeros_like(nav.dist); dn[finite] = nav.dist[finite] / nav.dist[finite].max()
        base = np.zeros((H, W, 3), np.uint8)
        base[finite] = (np.stack([40 + 160 * (1 - dn[finite]), 40 + 60 * dn[finite], 60 + 120 * dn[finite]], 1)).astype(np.uint8)
        img = np.repeat(np.repeat(base, scale, 0), scale, 1)

    def to_px(x, y):
        return int((x - nav.origin[0]) / nav.cell * scale), int((y - nav.origin[1]) / nav.cell * scale)

    def draw_line(x0, y0, x1, y1, col):
        n = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in np.linspace(0, 1, n):
            px, py = int(x0 + (x1 - x0) * t), int(y0 + (y1 - y0) * t)
            if 0 <= py < img.shape[0] and 0 <= px < img.shape[1]:
                img[py, px] = col

    for s in blocking_segments(m):
        draw_line(*to_px(s[0], s[1]), *to_px(s[2], s[3]), (230, 230, 230))
    for i in m.door_lines():
        xy = m.line_xy(i); draw_line(*to_px(xy[0], xy[1]), *to_px(xy[2], xy[3]), (60, 200, 255))
    xy = m.line_xy(nav.exit_line); draw_line(*to_px(xy[0], xy[1]), *to_px(xy[2], xy[3]), (255, 60, 60))
    if route is not None and len(route) > 1:
        for a, b in zip(route[:-1], route[1:]):
            draw_line(*to_px(*a), *to_px(*b), (255, 230, 0))
    for (x, y), col in ((nav.start, (0, 255, 0)), (nav.goal, (255, 0, 0))):
        px, py = to_px(x, y)
        img[max(py - 4, 0):py + 5, max(px - 4, 0):px + 5] = col
    iio.imwrite(path, img[::-1])          # flip so +y is up


def trace_route(nav: NavField, max_steps: int = 2000) -> np.ndarray:
    p = nav.start.copy(); pts = [p.copy()]
    for _ in range(max_steps):
        w = nav.next_waypoint(*p, lookahead=1)
        if np.allclose(w, p) or nav.distance(*w) >= nav.distance(*p):
            break
        p = w; pts.append(p.copy())
        if nav.distance(*p) < nav.cell:
            break
    return np.asarray(pts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wad", type=Path, required=True)
    ap.add_argument("--map", default="E1M1")
    ap.add_argument("--cell", type=float, default=16.0)
    ap.add_argument("--png", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="save the nav field (.npz)")
    a = ap.parse_args(argv)
    m = read_map(a.wad, a.map)
    nav = build_nav(m, a.cell)
    route = trace_route(nav)
    sx, sy, sa = m.player_start
    info = {"map": a.map, "vertices": len(m.vertices), "lines": len(m.lines), "sectors": len(m.sectors), "things": len(m.things),
            "player_start": [sx, sy, sa], "exit_lines": m.exit_lines(), "exit_special": int(m.lines[nav.exit_line, 3]),
            "door_lines": len(m.door_lines()), "blocking_segments": int(len(blocking_segments(m))),
            "grid": list(nav.dist.shape), "reachable_cells": int(np.isfinite(nav.dist).sum()),
            "start_distance_to_exit": nav.distance(sx, sy), "goal": nav.goal.tolist(), "route_points": int(len(route)),
            "route_reaches_goal": bool(len(route) and nav.distance(*route[-1]) < 2 * nav.cell)}
    print(json.dumps(info, indent=1))
    if a.png:
        render_png(m, nav, a.png, route)
    if a.out:
        np.savez(a.out, origin=nav.origin, cell=nav.cell, dist=nav.dist, goal=nav.goal, start=nav.start, exit_line=nav.exit_line)


if __name__ == "__main__":
    main()
