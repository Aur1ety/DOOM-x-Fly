"""Sparse matmul kernel with autograd w.r.t. the per-edge values.

PyTorch CSR tensors give no gradient w.r.t. the matrix, so this module wraps the CSR SpMM
(cuSPARSE on CUDA, MKL/native on CPU) in a custom ``autograd.Function``:

    forward          y = W @ x,  W = CSR(indptr_post, indices_pre, values)            [M, B]
    grad wrt x       W^T @ grad_out, via the prebuilt CSC of the same edge set
    grad wrt values  grad_values[e] = sum_b grad_out[post(e), b] * x[pre(e), b]
                     by SDDMM (``torch.sparse.sampled_addmm``: cuSPARSE on CUDA, MKL on CPU) when
                     it is available, else in edge chunks (never an [E, B] tensor beyond one chunk).
                     ``USE_SDDMM = False`` forces the chunked reference path.

Convention (CONTRACTS.md): CSR rows = POST, cols = PRE. The sign lives in the presynaptic neuron
and is already folded into ``values`` by the caller. Every per-edge array is in CSR order.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

torch.sparse.check_sparse_tensor_invariants.disable()   # explicit opt-out (our CSR/CSC are validated once)

EDGE_CHUNK = 1_000_000   # hard upper bound on edges per backward chunk (contract)
CHUNK_ELEMS = 1 << 24    # also cap chunk * B, so one fp32 chunk tensor stays <= 64 MB
USE_SDDMM = True         # use torch.sparse.sampled_addmm (SDDMM) for grad wrt values where it works


def edge_chunk(n_batch: int, chunk: Optional[int] = None) -> int:
    """Edges per backward chunk: explicit request (capped at EDGE_CHUNK) or the memory-based default."""
    if chunk is not None:
        return max(1, min(int(chunk), EDGE_CHUNK))
    return max(1, min(EDGE_CHUNK, CHUNK_ELEMS // max(1, n_batch)))


# ----------------------------------------------------------------------------- numpy helpers

def build_csr_csc(pre, post, n: int, weight=None) -> dict:
    """CSR (by post) + CSC (by pre) + perm for a duplicate-free edge list, contract dtypes.

    Returns dict with indptr_post[int64], indices_pre[int32], indptr_pre[int64],
    indices_post[int32], perm_csr_to_csc[int64] and, if given, weight[float32] in CSR order.
    """
    pre = np.asarray(pre, dtype=np.int64)
    post = np.asarray(post, dtype=np.int64)
    if pre.shape != post.shape:
        raise ValueError("pre and post must have the same length")
    if pre.size and (pre.min() < 0 or post.min() < 0 or pre.max() >= n or post.max() >= n):
        raise ValueError("edge endpoints out of range")
    order = np.lexsort((pre, post))            # CSR order: by post, then pre
    pre_c, post_c = pre[order], post[order]
    indptr_post = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(post_c, minlength=n), out=indptr_post[1:])
    perm = np.lexsort((post_c, pre_c))         # CSR-ordered edges sorted by pre, then post
    indptr_pre = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(pre_c, minlength=n), out=indptr_pre[1:])
    out = dict(
        indptr_post=indptr_post,
        indices_pre=pre_c.astype(np.int32),
        indptr_pre=indptr_pre,
        indices_post=post_c[perm].astype(np.int32),
        perm_csr_to_csc=perm.astype(np.int64),
    )
    if weight is not None:
        out["weight"] = np.asarray(weight, dtype=np.float32)[order]
    return out


def random_edges(n: int, n_edges: int, seed: int = 0, autapses: bool = False):
    """Uniform random duplicate-free edge list (pre, post) with exactly n_edges edges."""
    if n_edges > n * n:
        raise ValueError("too many edges for n nodes")
    rng = np.random.default_rng(seed)
    keys = np.empty(0, dtype=np.int64)
    while keys.size < n_edges:
        need = n_edges - keys.size
        pre = rng.integers(0, n, size=int(need * 1.1) + 16, dtype=np.int64)
        post = rng.integers(0, n, size=pre.size, dtype=np.int64)
        if not autapses:
            keep = pre != post
            pre, post = pre[keep], post[keep]
        keys = np.unique(np.concatenate([keys, post * n + pre]))
    keys = rng.permutation(keys)[:n_edges]
    return keys % n, keys // n


# ----------------------------------------------------------------------------- graph structure

@dataclass
class SparseGraph:
    """Index structure of one graph on one device. rows = POST, cols = PRE, M nodes, E edges."""

    n: int
    n_edges: int
    indptr_post: torch.Tensor      # int64 [M+1]   CSR
    indices_pre: torch.Tensor      # int32 [E]
    indptr_pre: torch.Tensor       # int64 [M+1]   CSC of the same edges
    indices_post: torch.Tensor     # int32 [E]
    perm_csr_to_csc: torch.Tensor  # int64 [E]     values_csc = values[perm]
    device: torch.device
    # int32 copies of the indptrs: torch.sparse_csr_tensor wants crow and col in one dtype
    crow_post: torch.Tensor = field(default=None, repr=False)
    crow_pre: torch.Tensor = field(default=None, repr=False)
    # row_of_edge (int32 [E], post index of each CSR edge) is built on first access: only the
    # chunked gradient path needs it, and at whole-brain scale it is 100 MB of device memory.
    _row_of_edge: torch.Tensor = field(default=None, repr=False)

    @property
    def row_of_edge(self) -> torch.Tensor:
        if self._row_of_edge is None:
            counts = self.indptr_post[1:] - self.indptr_post[:-1]
            self._row_of_edge = torch.repeat_interleave(
                torch.arange(self.n, dtype=torch.int32, device=self.device), counts)
        return self._row_of_edge

    @classmethod
    def from_arrays(cls, indptr_post, indices_pre, indptr_pre, indices_post, perm_csr_to_csc,
                    device="cpu", check: bool = True) -> "SparseGraph":
        device = torch.device(device)
        ip_post = torch.as_tensor(np.asarray(indptr_post), dtype=torch.int64)
        ix_pre = torch.as_tensor(np.asarray(indices_pre), dtype=torch.int32)
        ip_pre = torch.as_tensor(np.asarray(indptr_pre), dtype=torch.int64)
        ix_post = torch.as_tensor(np.asarray(indices_post), dtype=torch.int32)
        perm = torch.as_tensor(np.asarray(perm_csr_to_csc), dtype=torch.int64)
        n = int(ip_post.numel()) - 1
        e = int(ix_pre.numel())
        if ip_pre.numel() != n + 1 or ix_post.numel() != e or perm.numel() != e:
            raise ValueError("CSR/CSC array sizes disagree")
        if int(ip_post[-1]) != e or int(ip_pre[-1]) != e or int(ip_post[0]) or int(ip_pre[0]):
            raise ValueError("indptr does not span the edge array")
        g = cls(n, e, ip_post, ix_pre, ip_pre, ix_post, perm, torch.device("cpu"))
        if check:
            g.validate()                      # on the host, so the device never holds row_of_edge
            g._row_of_edge = None
        return g.to(device)

    @classmethod
    def from_npz(cls, path, device="cpu", check: bool = True) -> "SparseGraph":
        """Load graph_full / subgraph_* / null_* .npz (CONTRACTS.md keys)."""
        z = np.load(path)
        return cls.from_arrays(z["indptr_post"], z["indices_pre"], z["indptr_pre"], z["indices_post"],
                               z["perm_csr_to_csc"], device=device, check=check)

    @classmethod
    def from_edges(cls, pre, post, n: int, device="cpu") -> "SparseGraph":
        """Build from a duplicate-free (pre, post) edge list (tests, benchmarks)."""
        a = build_csr_csc(pre, post, n)
        return cls.from_arrays(a["indptr_post"], a["indices_pre"], a["indptr_pre"], a["indices_post"],
                               a["perm_csr_to_csc"], device=device)

    def __post_init__(self):
        self.device = self.indptr_post.device      # concrete (indexed) device of the tensors
        if self.crow_post is None:
            self.crow_post = self.indptr_post.to(torch.int32)
        if self.crow_pre is None:
            self.crow_pre = self.indptr_pre.to(torch.int32)

    def to(self, device) -> "SparseGraph":
        device = torch.empty(0, device=device).device   # normalise "cuda" -> "cuda:0"
        if device == self.device:
            return self
        return SparseGraph(self.n, self.n_edges, self.indptr_post.to(device), self.indices_pre.to(device),
                           self.indptr_pre.to(device), self.indices_post.to(device),
                           self.perm_csr_to_csc.to(device), device)

    def validate(self) -> None:
        """Check that CSR, CSC and perm describe the same edge set (O(E), raises on failure)."""
        n, e = self.n, self.n_edges
        if e == 0:
            return
        if int(self.indices_pre.min()) < 0 or int(self.indices_pre.max()) >= n:
            raise ValueError("indices_pre out of range")
        if int(self.indices_post.min()) < 0 or int(self.indices_post.max()) >= n:
            raise ValueError("indices_post out of range")
        counts = self.indptr_pre[1:] - self.indptr_pre[:-1]
        col_of_csc = torch.repeat_interleave(torch.arange(n, dtype=torch.int32, device=self.device), counts)
        if not torch.equal(self.row_of_edge.index_select(0, self.perm_csr_to_csc), self.indices_post):
            raise ValueError("perm_csr_to_csc does not map CSR post indices onto indices_post")
        if not torch.equal(self.indices_pre.index_select(0, self.perm_csr_to_csc), col_of_csc):
            raise ValueError("perm_csr_to_csc does not map CSR pre indices onto the CSC row structure")

    def n_bytes(self) -> int:
        """Bytes held by the index arrays on the device (row_of_edge counted only once built)."""
        ts = [self.indptr_post, self.indices_pre, self.indptr_pre, self.indices_post,
              self.perm_csr_to_csc, self.crow_post, self.crow_pre]
        if self._row_of_edge is not None:
            ts.append(self._row_of_edge)
        return sum(t.numel() * t.element_size() for t in ts)


# ----------------------------------------------------------------------------- primitives

def _csr_mm(crow: torch.Tensor, col: torch.Tensor, values: torch.Tensor, x: torch.Tensor, n: int) -> torch.Tensor:
    """CSR(crow, col, values) @ x, dense result [n, B]."""
    if values.numel() == 0:
        return torch.zeros(n, x.shape[1], dtype=x.dtype, device=x.device)
    a = torch.sparse_csr_tensor(crow, col, values, size=(n, n), check_invariants=False)
    return torch.mm(a, x)


def _for_each_edge_chunk(grad_out: torch.Tensor, x: torch.Tensor, g: SparseGraph, chunk: int,
                         consume: Callable[[int, int, torch.Tensor], None]) -> None:
    """Per-edge gradient grad_out[post] . x[pre], in chunks; consume(start, end, grad_chunk)."""
    e_total = g.n_edges
    for s in range(0, e_total, chunk):
        e = min(e_total, s + chunk)
        ge = grad_out.index_select(0, g.row_of_edge[s:e])          # [c, B]
        ge.mul_(x.index_select(0, g.indices_pre[s:e]))              # in place: one [c, B] live
        consume(s, e, ge.sum(1))


_SDDMM_OK: dict = {}


def _sddmm_available(device: torch.device) -> bool:
    """One-time probe per device: does torch.sparse.sampled_addmm run here and agree with chunking?

    Measured (torch 2.11, 1M edges): SDDMM beats the chunked gather 2-4x on an A30 and 7-12x on
    CPU (MKL), and never materialises anything per (edge, batch).
    """
    key = str(device)
    if key not in _SDDMM_OK:
        ok = False
        if hasattr(torch.sparse, "sampled_addmm"):
            try:
                pre, post = random_edges(64, 400, seed=1)
                g = SparseGraph.from_edges(pre, post, 64, device=device)
                go = torch.randn(64, 5, device=device)
                x = torch.randn(64, 5, device=device)
                ref = torch.empty(g.n_edges, device=device)
                _for_each_edge_chunk(go, x, g, 7, lambda s, e, v: ref[s:e].copy_(v))
                ok = torch.allclose(_grad_values_sddmm(go, x, g), ref, atol=1e-5, rtol=1e-4)
            except Exception:
                ok = False
        _SDDMM_OK[key] = ok
    return _SDDMM_OK[key]


def _grad_values_sddmm(grad_out: torch.Tensor, x: torch.Tensor, g: SparseGraph) -> torch.Tensor:
    """cuSPARSE SDDMM: (grad_out @ x^T) sampled at the CSR pattern, values in CSR order."""
    zeros = torch.zeros(g.n_edges, dtype=grad_out.dtype, device=grad_out.device)
    a = torch.sparse_csr_tensor(g.crow_post, g.indices_pre, zeros, size=(g.n, g.n), check_invariants=False)
    out = torch.sparse.sampled_addmm(a, grad_out, x.t().contiguous(), beta=0.0, alpha=1.0)
    return out.values()


def _use_sddmm(t: torch.Tensor) -> bool:
    return USE_SDDMM and _sddmm_available(t.device)


def _grad_x(grad_out: torch.Tensor, values: torch.Tensor, g: SparseGraph) -> torch.Tensor:
    """W^T @ grad_out through the CSC arrays (a CSR with rows = pre)."""
    return _csr_mm(g.crow_pre, g.indices_post, values.index_select(0, g.perm_csr_to_csc), grad_out, g.n)


# ----------------------------------------------------------------------------- autograd functions

class _SpMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, x, g, chunk):
        ctx.g, ctx.chunk = g, chunk
        ctx.save_for_backward(values, x)
        return _csr_mm(g.crow_post, g.indices_pre, values, x, g.n)

    @staticmethod
    def backward(ctx, grad_out):
        values, x = ctx.saved_tensors
        g = ctx.g
        grad_out = grad_out.contiguous()
        grad_values = grad_x = None
        if ctx.needs_input_grad[1]:
            grad_x = _grad_x(grad_out, values, g)
        if ctx.needs_input_grad[0]:
            if _use_sddmm(grad_out):
                grad_values = _grad_values_sddmm(grad_out, x, g)
            else:
                grad_values = torch.empty_like(values)
                _for_each_edge_chunk(grad_out, x, g, edge_chunk(x.shape[1], ctx.chunk),
                                     lambda s, e, v: grad_values[s:e].copy_(v))
        return grad_values, grad_x, None, None


class _SpMMBinned(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bin_values, x, bin_index, edge_scale, g, chunk):
        ctx.g, ctx.chunk = g, chunk
        ctx.save_for_backward(bin_values, x, bin_index, edge_scale)
        values = bin_values.index_select(0, bin_index)
        if edge_scale is not None:
            values = values * edge_scale
        return _csr_mm(g.crow_post, g.indices_pre, values, x, g.n)

    @staticmethod
    def backward(ctx, grad_out):
        bin_values, x, bin_index, edge_scale = ctx.saved_tensors
        g = ctx.g
        grad_out = grad_out.contiguous()
        grad_bins = grad_x = None
        if ctx.needs_input_grad[1]:
            values = bin_values.index_select(0, bin_index)
            if edge_scale is not None:
                values = values * edge_scale
            grad_x = _grad_x(grad_out, values, g)
        if ctx.needs_input_grad[0]:
            grad_bins = torch.zeros_like(bin_values)

            def consume(s, e, v):
                if edge_scale is not None:
                    v = v * edge_scale[s:e]
                grad_bins.index_add_(0, bin_index[s:e], v)

            if _use_sddmm(grad_out):
                consume(0, g.n_edges, _grad_values_sddmm(grad_out, x, g))
            else:
                _for_each_edge_chunk(grad_out, x, g, edge_chunk(x.shape[1], ctx.chunk), consume)
        return grad_bins, grad_x, None, None, None, None


# ----------------------------------------------------------------------------- public API

def _check_inputs(values: torch.Tensor, x: torch.Tensor, g: SparseGraph) -> torch.Tensor:
    if x.dim() != 2 or x.shape[0] != g.n:
        raise ValueError(f"x must be [M={g.n}, B], got {tuple(x.shape)}")
    if values.dim() != 1 or values.numel() != g.n_edges:
        raise ValueError(f"values must be [E={g.n_edges}], got {tuple(values.shape)}")
    if values.dtype != x.dtype:
        raise TypeError(f"values ({values.dtype}) and x ({x.dtype}) must share a dtype")
    if values.device != g.device or x.device != g.device:
        raise ValueError("values, x and graph must be on the same device")
    return x.contiguous()


def spmm(values: torch.Tensor, x: torch.Tensor, g: SparseGraph, chunk: Optional[int] = None) -> torch.Tensor:
    """y[post, b] = sum over CSR edges e into post of values[e] * x[indices_pre[e], b].

    Differentiable w.r.t. values (per edge) and x. ``chunk`` forces the edges-per-chunk of the
    per-edge gradient (tests); default is memory based, never above EDGE_CHUNK.
    """
    x = _check_inputs(values, x, g)
    return _SpMM.apply(values, x, g, chunk)


def spmm_binned(bin_values: torch.Tensor, bin_index: torch.Tensor, x: torch.Tensor, g: SparseGraph,
                edge_scale: Optional[torch.Tensor] = None, chunk: Optional[int] = None) -> torch.Tensor:
    """Type-tied spmm: values[e] = bin_values[bin_index[e]] * (edge_scale[e] or 1).

    Gradient flows to bin_values (segment-reduced per chunk) and x; edge_scale is a fixed buffer.
    """
    if bin_index.dim() != 1 or bin_index.numel() != g.n_edges:
        raise ValueError(f"bin_index must be [E={g.n_edges}]")
    if bin_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("bin_index must be int32 or int64")
    if edge_scale is not None:
        if edge_scale.shape != (g.n_edges,) or edge_scale.dtype != bin_values.dtype:
            raise ValueError("edge_scale must be [E] with the dtype of bin_values")
        if edge_scale.requires_grad:
            raise ValueError("edge_scale is a fixed buffer; no gradient is produced for it")
    if x.dim() != 2 or x.shape[0] != g.n or x.dtype != bin_values.dtype:
        raise ValueError("x must be [M, B] with the dtype of bin_values")
    if bin_values.dim() != 1:
        raise ValueError("bin_values must be [P]")
    return _SpMMBinned.apply(bin_values, x.contiguous(), bin_index, edge_scale, g, chunk)


def dense_matrix(values: torch.Tensor, g: SparseGraph) -> torch.Tensor:
    """Dense [M, M] copy, D[post, pre] = values (reference only; small graphs)."""
    d = torch.zeros(g.n, g.n, dtype=values.dtype, device=values.device)
    d[g.row_of_edge.long(), g.indices_pre.long()] = values
    return d


# ----------------------------------------------------------------------------- CLI

def _self_check(device: str) -> None:
    torch.manual_seed(0)
    pre, post = random_edges(40, 300, seed=3)
    g = SparseGraph.from_edges(pre, post, 40, device=device)
    v = torch.randn(g.n_edges, dtype=torch.float64, device=device, requires_grad=True)
    x = torch.randn(g.n, 3, dtype=torch.float64, device=device, requires_grad=True)
    y = spmm(v, x, g)
    y_ref = dense_matrix(v.detach(), g) @ x.detach()
    print(f"forward max|diff| vs dense: {(y - y_ref).abs().max().item():.3e}")
    ok = torch.autograd.gradcheck(lambda a, b: spmm(a, b, g, chunk=7), (v, x), eps=1e-6, atol=1e-5)
    print(f"gradcheck (float64, chunk=7): {'PASS' if ok else 'FAIL'}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="spmm kernel: self-check or a quick look at a graph .npz")
    p.add_argument("--device", default="cpu")
    p.add_argument("--npz", help="graph_full / subgraph / null .npz to load and time one forward on")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args(argv)
    if a.npz is None:
        _self_check(a.device)
        return
    t0 = time.time()
    g = SparseGraph.from_npz(a.npz, device=a.device)
    print(f"{a.npz}: M={g.n:,} E={g.n_edges:,} index bytes={g.n_bytes()/2**20:.1f} MiB load {time.time()-t0:.1f}s")
    v = torch.rand(g.n_edges, device=g.device)
    x = torch.rand(g.n, a.batch, device=g.device)
    spmm(v, x, g)
    if g.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        y = spmm(v, x, g)
    if g.device.type == "cuda":
        torch.cuda.synchronize()
    print(f"forward B={a.batch}: {(time.time()-t0)/5*1e3:.2f} ms/step, mean |y| {y.abs().mean().item():.3f}")


if __name__ == "__main__":
    main()
