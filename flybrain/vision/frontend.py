"""Frozen visual front end: EMD bank + form + chromatic channels (PLAN 3, items 4-7). ZERO trainable parameters.

Per decision step the caller hands over one frame batch (frameskip 4 = 114 ms) and gets the
clamped rates for all K substeps as ``[K, B, n_clamped]`` float32 (``rates[k]`` is the
``clamped_rates[B, n_clamped]`` of CONTRACTS.md for substep k), in the fixed unit order of
``FrontEnd.index`` (``ClampedIndex``; align to ``node_idx[is_clamped]`` by bodyId).

Pipeline (every time constant is an exact first-order exponential at dt = 114/K ms, so dt > tau is safe):

1. Column luminance through the geometry's Gaussian sampler at full frame resolution and R/G/B
   through the same sampler built at half resolution (rows sum to ``coverage``). The missing
   acceptance mass ``1 - coverage`` is filled with that eye's mean over its driven columns for the
   current frame (``fill='mean'``, default) or with 0 (``fill='black'``, the pre-registered
   ablation). Columns with no coverage share one "virtual" column per eye carrying exactly the fill.
2. Linear cross-fade of the column signals between the previous and current frame across the K
   substeps (PLAN 3.6; step-and-hold into a correlator is the defect it avoids).
3. Per substep: 15 ms photoreceptor low-pass ``lp``; 200 ms running per-column mean ``m``; contrast
   against the adapted mean. Default ``contrast='tanh_weber'``:
   ``c = tanh((lp - m) / (m + eps))`` -- a symmetric saturating (Weber) contrast whose operating
   point is the running mean, so absolute room brightness cancels (PLAN 3.5). The literal
   hyperbolic Naka-Rushton form ``c = (lp - m) / (lp + m + 2 eps)`` is kept as
   ``contrast='naka_rushton'``; it makes OFF responses ~3x larger than ON at full contrast, which
   nothing downstream has a reason to want. ON = relu(c), OFF = relu(-c).
4. Reichardt/EMD bank (Hassenstein-Reichardt, delay tau_d = 50 ms, full opponent, half-wave
   rectified): the delayed signal of the neighbour on the side the preferred motion ARRIVES from,
   times the direct signal at the column, minus the mirror term. Direction convention (Maisak
   2013 Nature 500:212), in each eye's own frame:
     a = front-to-back (progressive: rightward in the image for the RIGHT eye, leftward for the LEFT)
     b = back-to-front (regressive)        c = upward        d = downward
   T4x uses ON, T5x uses OFF. "Up"/"down" neighbours are the mean of the two oblique lattice
   neighbours above/below (vertical is not a lattice axis). A missing neighbour contributes 0.
5. Sustained/form channels: per-type first-order low-pass of ON or OFF with a fixed gain;
   chromatic channels from Doom RGB opponent contrasts (engineered, see ``TYPE_PARAMS``).
6. Gather per clamped cell: ``rate = gain_type * side_gain * channel[type][column]``.

CLI: ``python -m flybrain.vision.frontend --bench`` (runtime at B=64 on CPU).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch

from flybrain.vision.geometry import CLAMPED_TYPES, SIDES, Geometry, DEFAULT_PATH, build_sampler

# ---- per-type fixed parameters --------------------------------------------------------------
# tau_ms: approximate response time constants read from the cited papers (calcium/voltage
# kinetics; NOT fitted here). gain: unit-less normalisation (1.0 = contrast units); EMD channels
# get EMD_GAIN on top so a full-contrast 30 deg grating at 2 Hz lands on the form-channel scale.
# Sources: Behnia 2014 Nature 512:427 (Mi1/Tm1 ~30 ms, Tm2 faster than Tm1);
# Arenz 2017 Curr Biol 27:929 (Mi4, Mi9 slow/sustained ~150 ms; Mi9 depolarises to OFF);
# Strother 2017 Neuron 94:168 (Mi9 OFF polarity); Serbe 2016 Neuron 89:829 (Tm4 transient,
# Tm9 sustained); Maisak 2013 (T4 ON / T5 OFF, four cardinal directions).
# Chromatic types: Doom RGB has no correspondence to the fly's UV/blue/green receptor set, so
# these five channels are ENGINEERED opponent proxies (R-G and B-Y contrast, Y = (R+G)/2),
# chosen so the chromatic types are driven by colour rather than silent. Sign assignment loosely
# follows Schnaitmann 2018 Cell 172:318 / Karuppudurai 2014 Neuron 81:603 (Tm5a blue-ON,
# Tm5c short-wavelength-ON sustained, Tm5b/Tm20 long-wavelength) and is not a physiological claim.
SOURCES = ("t4a", "t4b", "t4c", "t4d", "t5a", "t5b", "t5c", "t5d",
           "on", "off", "rg_pos", "rg_neg", "by_pos", "by_neg")
TYPE_PARAMS: dict[str, tuple[str, float, float]] = {
    # type: (source channel, tau_ms, gain)
    "T4a": ("t4a", 30.0, 1.0), "T4b": ("t4b", 30.0, 1.0), "T4c": ("t4c", 30.0, 1.0), "T4d": ("t4d", 30.0, 1.0),
    "T5a": ("t5a", 30.0, 1.0), "T5b": ("t5b", 30.0, 1.0), "T5c": ("t5c", 30.0, 1.0), "T5d": ("t5d", 30.0, 1.0),
    "Mi1": ("on", 30.0, 1.0), "Mi4": ("on", 150.0, 1.0), "Mi9": ("off", 150.0, 1.0),
    "Tm1": ("off", 30.0, 1.0), "Tm2": ("off", 20.0, 1.0), "Tm4": ("off", 30.0, 1.0), "Tm9": ("off", 150.0, 1.0),
    "Tm20": ("rg_neg", 60.0, 1.0), "Tm5a": ("by_pos", 60.0, 1.0), "Tm5b": ("by_neg", 60.0, 1.0),
    "Tm5c": ("by_pos", 150.0, 1.0), "TmY5a": ("rg_pos", 60.0, 1.0),
}
EMD_GAIN = 4.0
assert tuple(TYPE_PARAMS) == CLAMPED_TYPES


@dataclass
class ClampedIndex:
    """Unit order of the front end output (one row per clamped cell)."""
    bodyId: np.ndarray      # int64 [M]
    type: np.ndarray        # unicode [M]
    type_id: np.ndarray     # int16 [M] into CLAMPED_TYPES
    side: np.ndarray        # unicode [M] 'L'/'R'
    hex: np.ndarray         # int16 [M, 2], -1 if no column
    col: np.ndarray         # int32 [M] geometry column or -1
    az: np.ndarray          # float32 [M] deg (nan if no column)
    el: np.ndarray          # float32 [M]
    driven: np.ndarray      # bool [M] column has coverage >= 0.5

    def __len__(self) -> int:
        return len(self.bodyId)

    def permutation_for(self, body_ids) -> np.ndarray:
        """Positions in the unit order for the given bodyIds (e.g. neurons.bodyId[node_idx[is_clamped]])."""
        order = np.argsort(self.bodyId)
        pos = np.clip(np.searchsorted(self.bodyId[order], body_ids), 0, len(order) - 1)
        found = self.bodyId[order[pos]] == np.asarray(body_ids)
        if not found.all():
            raise KeyError(f"{int((~found).sum())} bodyIds are not clamped units of this front end")
        return order[pos]


@dataclass
class FrontEndState:
    """Per-environment dynamical state over the reduced column set (R columns + 1 zero column)."""
    prev: torch.Tensor      # [B, R, 4] previous frame's (lum, R, G, B) columns
    lp: torch.Tensor        # [B, R, 4] photoreceptor low-pass
    mean: torch.Tensor      # [B, R] 200 ms running mean luminance
    d_on: torch.Tensor      # [B, R] delayed ON
    d_off: torch.Tensor     # [B, R] delayed OFF
    bank: torch.Tensor      # [B, T, R + 1] per-type channel outputs (gain applied); last column stays 0
    live: torch.Tensor      # [B] bool, False until the first frame after a reset


class FrontEnd:
    def __init__(self, geom: Geometry, K: int = 6, decision_ms: float = 114.0, device="cpu",
                 fill: str = "mean", contrast: str = "tanh_weber", tau_photo_ms: float = 15.0,
                 tau_mean_ms: float = 200.0, tau_delay_ms: float = 50.0, eps: float = 0.02,
                 side_norm: bool = True, rgb_downsample: int = 2):
        assert fill in ("mean", "black") and contrast in ("tanh_weber", "naka_rushton")
        self.geom, self.K, self.dt_ms = geom, K, decision_ms / K
        self.device, self.fill, self.contrast, self.eps = torch.device(device), fill, contrast, eps
        C = geom.n_columns
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), device=self.device, dtype=dt)
        # reduced column set: every column with pixel coverage, its lattice neighbours, + 2 virtual
        active = geom.coverage > 0
        nb = geom.nbr[active]; active[nb[nb >= 0]] = True
        self.act_idx = np.nonzero(active)[0]
        Ra = len(self.act_idx); self.R = Ra + 2
        col_map = np.full(C, -1, np.int64)
        col_map[self.act_idx] = np.arange(Ra)
        col_map[~active] = Ra + geom.col_side[~active].astype(np.int64)   # virtual column of that eye
        self.col_map = col_map
        r_side = np.concatenate([geom.col_side[self.act_idx], [0, 1]]).astype(np.int64)
        r_driven = np.concatenate([geom.driven[self.act_idx], [False, False]])
        nbr = np.full((self.R, 6), -1, np.int64)
        nbr[:Ra] = np.where(geom.nbr[self.act_idx] >= 0, col_map[np.maximum(geom.nbr[self.act_idx], 0)], -1)
        has = nbr >= 0
        self.nbr = t(np.where(has, nbr, np.arange(self.R)[:, None]))     # missing neighbour -> self (term vanishes)
        up_w = has[:, 2:4].astype(np.float32); dn_w = has[:, 4:6].astype(np.float32)
        self.up_w = t(up_w / np.maximum(up_w.sum(1, keepdims=True), 1.0))
        self.dn_w = t(dn_w / np.maximum(dn_w.sum(1, keepdims=True), 1.0))
        self.side_r, self.driven_r = t(r_side), t(r_driven)
        drv = np.zeros((2, self.R), np.float32)                              # per-eye mean over driven columns
        for s in (0, 1):
            m = r_driven & (r_side == s); drv[s, m] = 1.0 / max(int(m.sum()), 1)
        self.drv_mean = t(drv)
        # samplers: luminance at full resolution (from the geometry), RGB at reduced resolution
        H, W = geom.frame_hw
        self.HW = H * W
        self.S, self.cov = self._csr(geom.pix_indptr, geom.pix_indices, geom.pix_weights, geom.coverage, self.HW)
        self.rgb_ds = rgb_downsample
        self.rgb_hw = (H // rgb_downsample, W // rgb_downsample)
        ip, ix, wv, cov = build_sampler(geom.col_az, geom.col_el, geom.col_side, geom.meta["hfov_deg"],
                                        self.rgb_hw[1], self.rgb_hw[0], geom.meta["fwhm_deg"], geom.meta["overlap_deg"],
                                        geom.meta["trunc_sigma"])
        self.S_rgb, self.cov_rgb = self._csr(ip, ix, wv, cov, self.rgb_hw[0] * self.rgb_hw[1])
        # filters and per-type parameters
        e = lambda tau: float(np.exp(-self.dt_ms / tau))
        self.a_photo, self.a_mean, self.a_delay = e(tau_photo_ms), e(tau_mean_ms), e(tau_delay_ms)
        self.T = len(CLAMPED_TYPES)
        self.a_type = t([e(TYPE_PARAMS[ty][1]) for ty in CLAMPED_TYPES], torch.float32)[None, :, None]
        self.src_idx = t([SOURCES.index(TYPE_PARAMS[ty][0]) for ty in CLAMPED_TYPES])
        type_gain = np.array([TYPE_PARAMS[ty][2] * (EMD_GAIN if TYPE_PARAMS[ty][0][0] == "t" else 1.0)
                              for ty in CLAMPED_TYPES], np.float32)
        side_gain = geom.side_gain if side_norm else np.ones(2, np.float32)
        self.gain = t(type_gain[:, None] * side_gain[r_side][None, :], torch.float32)[None]      # [1, T, R]
        # units: flat index into bank[B, T, R + 1]; unassigned cells point at the zero column R
        ucol = np.where(geom.cell_col >= 0, col_map[np.maximum(geom.cell_col, 0)], self.R)
        self.unit_flat = t(geom.cell_type_id.astype(np.int64) * (self.R + 1) + ucol)
        self.n_clamped = geom.n_cells
        ok = geom.cell_col >= 0
        az = np.full(geom.n_cells, np.nan, np.float32); el = az.copy()
        az[ok] = geom.col_az[geom.cell_col[ok]]; el[ok] = geom.col_el[geom.cell_col[ok]]
        drv_u = np.zeros(geom.n_cells, bool); drv_u[ok] = geom.driven[geom.cell_col[ok]]
        self.index = ClampedIndex(bodyId=geom.cell_bodyId, type=geom.cell_type, type_id=geom.cell_type_id,
                                  side=np.array(SIDES)[geom.cell_side], hex=geom.cell_hex(), col=geom.cell_col,
                                  az=az, el=el, driven=drv_u)

    def _csr(self, indptr, indices, weights, coverage, n_pix):
        """Sparse CSR sampler restricted to the active columns, and their coverage [R] (virtual = 0)."""
        rows = [np.arange(indptr[i], indptr[i + 1]) for i in self.act_idx]
        sel = np.concatenate(rows) if rows else np.zeros(0, np.int64)
        lens = np.array([len(r) for r in rows], np.int64)
        t = lambda a, dt=None: torch.as_tensor(np.asarray(a), device=self.device, dtype=dt)
        S = torch.sparse_csr_tensor(t(np.r_[0, np.cumsum(lens)]), t(indices[sel].astype(np.int64)), t(weights[sel]),
                                    size=(len(self.act_idx), n_pix), device=self.device)
        cov = t(np.concatenate([coverage[self.act_idx], [0.0, 0.0]]), torch.float32)
        return S, cov

    # ---- state ------------------------------------------------------------------------------
    def init_state(self, B: int) -> FrontEndState:
        z = lambda *shape: torch.zeros(*shape, device=self.device, dtype=torch.float32)
        return FrontEndState(prev=z(B, self.R, 4), lp=z(B, self.R, 4), mean=z(B, self.R), d_on=z(B, self.R),
                             d_off=z(B, self.R), bank=z(B, self.T, self.R + 1),
                             live=torch.zeros(B, dtype=torch.bool, device=self.device))

    def reset(self, state: FrontEndState, mask) -> FrontEndState:
        """Mark environments in `mask` as reset; their dynamics re-initialise on the next frame."""
        state.live[torch.as_tensor(np.asarray(mask), device=self.device)] = False
        return state

    # ---- frame path -------------------------------------------------------------------------
    def sample(self, gray: torch.Tensor, rgb: torch.Tensor | None) -> torch.Tensor:
        """uint8 gray [B,H,W] (+ rgb [B,H,W,3] or [B,H/ds,W/ds,3]) -> column signals [B, R, 4] (lum, R, G, B)."""
        B = gray.shape[0]
        g = gray.reshape(B, self.HW).t().to(torch.float32).div_(255.0)                    # [HW, B]
        lum = self._fill(self.cov, torch.sparse.mm(self.S, g).t()[:, :, None])             # [B, R, 1]
        if rgb is None:
            return lum.expand(B, self.R, 4).contiguous()
        h, w = self.rgb_hw
        if rgb.shape[1] != h:                                       # pool full-resolution RGB down (cheap; the
            x = rgb.permute(0, 3, 1, 2).to(torch.float32)           # env should rather deliver RGB at rgb_hw)
            x = torch.nn.functional.avg_pool2d(x, self.rgb_ds)      # [B, 3, h, w]
        else:
            x = rgb.permute(0, 3, 1, 2).to(torch.float32)
        x = x.reshape(B * 3, h * w).t().div_(255.0)                                          # [hw, B*3]
        col = self._fill(self.cov_rgb, torch.sparse.mm(self.S_rgb, x).t().reshape(B, 3, -1).permute(0, 2, 1))
        return torch.cat([lum, col], 2)

    def _fill(self, cov: torch.Tensor, sampled: torch.Tensor) -> torch.Tensor:
        """sampled [B, Ra, k] -> [B, R, k] with the two virtual columns and the eye-mean fill."""
        B, _, k = sampled.shape
        cols = torch.cat([sampled, torch.zeros(B, 2, k, device=self.device)], 1)
        if self.fill == "mean":
            covered = cols / cov.clamp_min(1e-6)[None, :, None]
            eye_mean = torch.einsum("sr,brk->bsk", self.drv_mean, covered)                 # [B, 2, k]
            cols = cols + (1.0 - cov)[None, :, None] * eye_mean.index_select(1, self.side_r)
        return cols

    @torch.no_grad()
    def process(self, gray, rgb, state: FrontEndState) -> tuple[torch.Tensor, FrontEndState]:
        """One decision step: frames -> clamped rates [K, B, n_clamped] float32 and the new state."""
        gray = torch.as_tensor(gray, device=self.device)
        rgb = None if rgb is None else torch.as_tensor(rgb, device=self.device)
        cur = self.sample(gray, rgb)
        state = self._init_dead(state, cur)
        B = cur.shape[0]
        out = torch.empty(self.K, B, self.n_clamped, device=self.device, dtype=torch.float32)
        for k in range(1, self.K + 1):
            x = state.prev + (k / self.K) * (cur - state.prev)
            state = self.substep(x, state)
            torch.index_select(state.bank.reshape(B, -1), 1, self.unit_flat, out=out[k - 1])
        state.prev = cur
        return out, state

    # ---- column path (stimulus battery) -----------------------------------------------------
    def columns_from_full(self, lum_full, rgb_full=None) -> torch.Tensor:
        """Signals for ALL geometry columns [B, C] (+ [B, C, 3]) -> reduced set [B, R, 4].

        Virtual columns get the mean over the driven columns of their eye (the stimulus is defined
        in angle space, so there is no frustum and every column sees it directly).
        """
        lum_full = torch.as_tensor(np.asarray(lum_full), device=self.device, dtype=torch.float32)
        B, C = lum_full.shape
        if rgb_full is None:
            rgb_full = lum_full[:, :, None].expand(B, C, 3)
        else:
            rgb_full = torch.as_tensor(np.asarray(rgb_full), device=self.device, dtype=torch.float32)
        full = torch.cat([lum_full[:, :, None], rgb_full], 2)
        cols = torch.cat([full.index_select(1, torch.as_tensor(self.act_idx, device=self.device)),
                          torch.zeros(B, 2, 4, device=self.device)], 1)
        cols[:, -2:] = torch.einsum("sr,brk->bsk", self.drv_mean, cols)
        return cols

    @torch.no_grad()
    def step_columns(self, cols: torch.Tensor, state: FrontEndState) -> tuple[torch.Tensor, FrontEndState]:
        """One substep on reduced column signals [B, R, 4] -> rates [B, n_clamped]."""
        state = self._init_dead(state, cols)
        state = self.substep(cols, state)
        state.prev = cols
        return self.rates(state), state

    # ---- dynamics ---------------------------------------------------------------------------
    def _init_dead(self, state: FrontEndState, cur: torch.Tensor) -> FrontEndState:
        dead = ~state.live
        if bool(dead.any()):
            state.prev[dead] = cur[dead]; state.lp[dead] = cur[dead]; state.mean[dead] = cur[dead, :, 0]
            state.d_on[dead] = 0.0; state.d_off[dead] = 0.0; state.bank[dead] = 0.0
            state.live[dead] = True
        return state

    def substep(self, x: torch.Tensor, s: FrontEndState) -> FrontEndState:
        """x: [B, R, 4] column (lum, R, G, B) for this substep."""
        s.lp = self.a_photo * s.lp + (1.0 - self.a_photo) * x
        lum = s.lp[:, :, 0]
        if self.contrast == "tanh_weber":
            c = torch.tanh((lum - s.mean) / (s.mean + self.eps))
        else:
            c = (lum - s.mean) / (lum + s.mean + 2.0 * self.eps)
        s.mean = self.a_mean * s.mean + (1.0 - self.a_mean) * lum
        on, off = torch.relu(c), torch.relu(-c)
        s.d_on = self.a_delay * s.d_on + (1.0 - self.a_delay) * on
        s.d_off = self.a_delay * s.d_off + (1.0 - self.a_delay) * off
        r, g, b = s.lp[:, :, 1], s.lp[:, :, 2], s.lp[:, :, 3]
        rg = (r - g) / (r + g + 2.0 * self.eps)
        by = (b - 0.5 * (r + g)) / (b + 0.5 * (r + g) + 2.0 * self.eps)
        src = torch.cat([self._emd(on, s.d_on), self._emd(off, s.d_off),
                         torch.stack([on, off, torch.relu(rg), torch.relu(-rg), torch.relu(by), torch.relu(-by)], 1)], 1)
        drive = src.index_select(1, self.src_idx) * self.gain                              # [B, T, R]
        bank = s.bank[:, :, :self.R]
        bank.mul_(self.a_type).add_((1.0 - self.a_type) * drive)
        return s

    def _emd(self, y: torch.Tensor, yd: torch.Tensor) -> torch.Tensor:
        """Four opponent HRC outputs [B, 4, R] (a, b, c, d) from direct y and delayed yd, both [B, R]."""
        nb = self.nbr
        g = lambda z, k: z.index_select(1, nb[:, k])
        y_f, y_b, yd_f, yd_b = g(y, 0), g(y, 1), g(yd, 0), g(yd, 1)
        y_up = self.up_w[:, 0] * g(y, 2) + self.up_w[:, 1] * g(y, 3)
        y_dn = self.dn_w[:, 0] * g(y, 4) + self.dn_w[:, 1] * g(y, 5)
        yd_up = self.up_w[:, 0] * g(yd, 2) + self.up_w[:, 1] * g(yd, 3)
        yd_dn = self.dn_w[:, 0] * g(yd, 4) + self.dn_w[:, 1] * g(yd, 5)
        a = yd_f * y - y_f * yd        # motion arriving from the front  (front-to-back)
        b = yd_b * y - y_b * yd        # arriving from the back          (back-to-front)
        c = yd_dn * y - y_dn * yd      # arriving from below             (upward)
        d = yd_up * y - y_up * yd      # arriving from above             (downward)
        return torch.relu(torch.stack([a, b, c, d], 1))

    def rates(self, s: FrontEndState) -> torch.Tensor:
        """Gather the clamped rates [B, n_clamped] from the channel bank."""
        return s.bank.reshape(s.bank.shape[0], -1).index_select(1, self.unit_flat)

    def bank_full(self, s: FrontEndState) -> torch.Tensor:
        """Channel bank re-expanded to all geometry columns: [B, T, C] (diagnostics)."""
        return s.bank.index_select(2, torch.as_tensor(self.col_map, device=self.device))


# ---- CLI: runtime measurement ----------------------------------------------------------------

def bench(geom: Geometry, B: int = 64, frames: int = 50, K: int = 6, device="cpu", seed: int = 0) -> dict:
    fe = FrontEnd(geom, K=K, device=device)
    rng = np.random.default_rng(seed)
    H, W = geom.frame_hw
    gray = rng.integers(0, 256, size=(frames, B, H, W), dtype=np.uint8)
    rgb = rng.integers(0, 256, size=(frames, B, H, W, 3), dtype=np.uint8)
    state = fe.init_state(B)
    fe.process(gray[0], rgb[0], state)
    t_s = 0.0
    for i in range(frames):
        ts = time.perf_counter(); fe.sample(torch.as_tensor(gray[i]), torch.as_tensor(rgb[i])); t_s += time.perf_counter() - ts
    t0 = time.perf_counter()
    for i in range(frames):
        out, state = fe.process(gray[i], rgb[i], state)
    total = time.perf_counter() - t0
    per_batch_ms = 1000 * total / frames
    n_col = int(fe.driven_r.sum())
    return {"B": B, "K": K, "frames": frames, "device": str(device), "n_clamped": fe.n_clamped, "reduced_columns": fe.R,
            "driven_columns": n_col, "ms_per_batched_frame": per_batch_ms,
            "ms_per_env_frame": per_batch_ms / B, "us_per_column_env_frame": 1000 * per_batch_ms / B / n_col,
            "sampler_ms_per_batched_frame": 1000 * t_s / frames, "out_shape": tuple(out.shape)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Frozen front end: runtime benchmark")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--geometry", default=str(DEFAULT_PATH))
    ap.add_argument("--synthetic", type=int, default=0, help="use a synthetic hex disc of this radius")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    geom = Geometry.synthetic(args.synthetic) if args.synthetic else Geometry.load(args.geometry)
    if args.bench:
        for k, v in bench(geom, args.B, args.frames, args.K, args.device).items():
            print(f"{k}: {v}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
