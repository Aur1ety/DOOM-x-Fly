"""Tests for flybrain.model.spmm. Run on master (CPU) and node1 (CUDA, under the GPU guard)."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from flybrain.model import spmm as K

HAS_CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA on this host")


@pytest.fixture(scope="session")
def cuda():
    """CUDA device under the shared-card guard (free-memory check, allocator cap, NVML watchdog)."""
    if not HAS_CUDA:
        pytest.skip("no CUDA on this host")
    from flybrain.model.bench import gpu_guard
    gpu_guard()
    return torch.device("cuda")


@pytest.fixture(autouse=True)
def _sddmm_default():
    K.USE_SDDMM = True
    yield
    K.USE_SDDMM = True


@pytest.fixture(params=[False, True], ids=["chunked", "sddmm"])
def sddmm(request):
    """Run a gradient test on both per-edge-gradient paths (chunked reference and SDDMM)."""
    K.USE_SDDMM = request.param
    if request.param and not K._sddmm_available(torch.device("cpu")):
        pytest.skip("sampled_addmm not available on CPU in this torch")
    return request.param


def small_edges(seed: int, n: int = 24, e: int = 90):
    """Duplicate-free edges over n nodes with empty rows (posts 0,1) and isolated nodes (last 3)."""
    pre, post = K.random_edges(n - 3, e + 20, seed=seed)
    keep = post >= 2
    return pre[keep][:e], post[keep][:e]


def small_graph(seed: int, device="cpu", n: int = 24, e: int = 90) -> K.SparseGraph:
    pre, post = small_edges(seed, n, e)
    return K.SparseGraph.from_edges(pre, post, n, device=device)


def rand_inputs(g: K.SparseGraph, B: int, dtype, seed: int = 0, grads=(True, True)):
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(g.n_edges, dtype=dtype, generator=gen).to(g.device).requires_grad_(grads[0])
    x = torch.randn(g.n, B, dtype=dtype, generator=gen).to(g.device).requires_grad_(grads[1])
    return v, x


# ----------------------------------------------------------------------------- structure

def test_build_csr_csc_matches_scipy():
    pre, post = small_edges(1)
    a = K.build_csr_csc(pre, post, 24, weight=np.arange(len(pre)) + 1.0)
    csr = sp.csr_matrix((np.arange(len(pre)) + 1.0, (post, pre)), shape=(24, 24))
    csr.sort_indices()
    assert np.array_equal(a["indptr_post"], csr.indptr)
    assert np.array_equal(a["indices_pre"], csr.indices)
    assert np.array_equal(a["weight"], csr.data)
    csc = csr.tocsc()
    csc.sort_indices()
    assert np.array_equal(a["indptr_pre"], csc.indptr)
    assert np.array_equal(a["indices_post"], csc.indices)
    assert np.array_equal(a["weight"][a["perm_csr_to_csc"]], csc.data)
    assert a["indptr_post"].dtype == np.int64 and a["indices_pre"].dtype == np.int32
    assert a["perm_csr_to_csc"].dtype == np.int64


def test_graph_has_empty_rows_and_isolated_nodes():
    g = small_graph(2)
    deg_in = g.indptr_post[1:] - g.indptr_post[:-1]
    deg_out = g.indptr_pre[1:] - g.indptr_pre[:-1]
    assert (deg_in[:2] == 0).all()
    assert (deg_in[-3:] == 0).all() and (deg_out[-3:] == 0).all()
    assert g.row_of_edge.dtype == torch.int32 and g.perm_csr_to_csc.dtype == torch.int64
    assert torch.equal(g.row_of_edge[g.perm_csr_to_csc], g.indices_post)


def test_validate_rejects_bad_perm():
    g = small_graph(3)
    bad = g.perm_csr_to_csc.clone()
    bad[0], bad[1] = g.perm_csr_to_csc[1], g.perm_csr_to_csc[0]
    with pytest.raises(ValueError):
        K.SparseGraph.from_arrays(g.indptr_post, g.indices_pre, g.indptr_pre, g.indices_post, bad)


def test_from_npz_roundtrip(tmp_path):
    pre, post = small_edges(4)
    a = K.build_csr_csc(pre, post, 24, weight=np.ones(len(pre)))
    path = tmp_path / "subgraph_test.npz"
    np.savez_compressed(path, n_neurons=24, n_edges=len(pre), **a)
    g = K.SparseGraph.from_npz(path)
    g2 = K.SparseGraph.from_edges(pre, post, 24)
    for k in ("indptr_post", "indices_pre", "indptr_pre", "indices_post", "perm_csr_to_csc", "row_of_edge"):
        assert torch.equal(getattr(g, k), getattr(g2, k)), k
    assert g.to("cpu") is g


# ----------------------------------------------------------------------------- forward

@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_matches_dense(B, dtype):
    g = small_graph(5)
    v, x = rand_inputs(g, B, dtype, grads=(False, False))
    y = K.spmm(v, x, g)
    y_ref = K.dense_matrix(v, g) @ x
    assert y.shape == (g.n, B)
    torch.testing.assert_close(y, y_ref, rtol=1e-6 if dtype == torch.float64 else 1e-5, atol=1e-6)
    assert (y[:2] == 0).all()        # empty rows get zero input


def test_empty_graph():
    g = K.SparseGraph.from_edges(np.zeros(0, np.int64), np.zeros(0, np.int64), 5)
    v = torch.zeros(0, requires_grad=True)
    x = torch.randn(5, 2, requires_grad=True)
    y = K.spmm(v, x, g)
    assert y.shape == (5, 2) and (y == 0).all()
    y.sum().backward()
    assert x.grad is not None and (x.grad == 0).all()


def test_input_validation():
    g = small_graph(6)
    with pytest.raises(ValueError):
        K.spmm(torch.zeros(g.n_edges + 1), torch.zeros(g.n, 2), g)
    with pytest.raises(ValueError):
        K.spmm(torch.zeros(g.n_edges), torch.zeros(g.n + 1, 2), g)
    with pytest.raises(TypeError):
        K.spmm(torch.zeros(g.n_edges, dtype=torch.float64), torch.zeros(g.n, 2), g)


# ----------------------------------------------------------------------------- gradients (CPU)

@pytest.mark.parametrize("B", [1, 3])
@pytest.mark.parametrize("chunk", [None, 7])
def test_gradcheck_per_edge(B, chunk, sddmm):
    g = small_graph(7)
    v, x = rand_inputs(g, B, torch.float64)
    assert torch.autograd.gradcheck(lambda a, b: K.spmm(a, b, g, chunk=chunk), (v, x), eps=1e-6, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("B", [1, 3])
def test_backward_matches_dense(B, sddmm):
    g = small_graph(8)
    v, x = rand_inputs(g, B, torch.float64)
    go = torch.randn(g.n, B, dtype=torch.float64)
    K.spmm(v, x, g).backward(go)
    d = K.dense_matrix(v.detach(), g).requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    (d @ x2).backward(go)
    torch.testing.assert_close(x.grad, x2.grad)
    torch.testing.assert_close(v.grad, d.grad[g.row_of_edge.long(), g.indices_pre.long()])


def test_chunk_boundaries_identical():
    K.USE_SDDMM = False                      # the chunked path is what is under test here
    g = small_graph(9, n=40, e=200)          # 200 edges: not a multiple of 7, several chunks
    go = torch.randn(g.n, 3)
    grads = []
    for chunk in (None, 7, 1, 200, 10 ** 9):
        v, x = rand_inputs(g, 3, torch.float32)
        K.spmm(v, x, g, chunk=chunk).backward(go)
        grads.append((v.grad.clone(), x.grad.clone()))
    for gv, gx in grads[1:]:
        torch.testing.assert_close(gv, grads[0][0])
        torch.testing.assert_close(gx, grads[0][1])
    assert K.edge_chunk(64) == min(K.EDGE_CHUNK, K.CHUNK_ELEMS // 64)
    assert K.edge_chunk(64, 10 ** 9) == K.EDGE_CHUNK


def test_sddmm_matches_chunked_cpu():
    if not K._sddmm_available(torch.device("cpu")):
        pytest.skip("sampled_addmm not available on CPU in this torch")
    n, e = 3000, 80_000
    pre, post = K.random_edges(n, e, seed=24)
    g = K.SparseGraph.from_edges(pre, post, n)
    go = torch.randn(n, 16)
    grads = []
    for use in (False, True):
        K.USE_SDDMM = use
        v, x = rand_inputs(g, 16, torch.float32, seed=2)
        K.spmm(v, x, g, chunk=9999).backward(go)
        grads.append(v.grad.clone())
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-5)


def test_grad_only_where_needed():
    g = small_graph(10)
    v, x = rand_inputs(g, 2, torch.float32, grads=(True, False))
    K.spmm(v, x, g).sum().backward()
    assert v.grad is not None and x.grad is None
    v, x = rand_inputs(g, 2, torch.float32, grads=(False, True))
    K.spmm(v, x, g).sum().backward()
    assert v.grad is None and x.grad is not None


# ----------------------------------------------------------------------------- binned path

def binned_setup(g: K.SparseGraph, P: int, B: int, dtype, with_scale: bool, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    bin_index = torch.randint(0, P, (g.n_edges,), generator=gen).to(g.device)
    bin_values = torch.randn(P, dtype=dtype, generator=gen).to(g.device).requires_grad_(True)
    scale = torch.rand(g.n_edges, dtype=dtype, generator=gen).to(g.device) + 0.5 if with_scale else None
    x = torch.randn(g.n, B, dtype=dtype, generator=gen).to(g.device).requires_grad_(True)
    return bin_values, bin_index, scale, x


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("B", [1, 3])
def test_gradcheck_binned(with_scale, B, sddmm):
    g = small_graph(11)
    bv, bi, sc, x = binned_setup(g, 5, B, torch.float64, with_scale)
    fn = lambda a, b: K.spmm_binned(a, bi, b, g, edge_scale=sc, chunk=7)
    assert torch.autograd.gradcheck(fn, (bv, x), eps=1e-6, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("with_scale", [False, True])
@pytest.mark.parametrize("chunk", [None, 7])
def test_binned_equals_unbinned(with_scale, chunk, sddmm):
    g = small_graph(12, n=40, e=200)
    bv, bi, sc, x = binned_setup(g, 6, 3, torch.float64, with_scale)
    go = torch.randn(g.n, 3, dtype=torch.float64)
    y = K.spmm_binned(bv, bi, x, g, edge_scale=sc, chunk=chunk)
    y.backward(go)
    values = bv.detach()[bi] * (sc if sc is not None else 1.0)
    v = values.clone().requires_grad_(True)
    x2 = x.detach().clone().requires_grad_(True)
    y2 = K.spmm(v, x2, g, chunk=chunk)
    y2.backward(go)
    torch.testing.assert_close(y, y2)
    torch.testing.assert_close(x.grad, x2.grad)
    per_edge = v.grad * (sc if sc is not None else 1.0)
    expected = torch.zeros_like(bv).index_add_(0, bi, per_edge)
    torch.testing.assert_close(bv.grad, expected)
    assert bv.grad.shape == (6,)


def test_binned_rejects_trainable_scale():
    g = small_graph(13)
    bv, bi, sc, x = binned_setup(g, 4, 2, torch.float32, True)
    with pytest.raises(ValueError):
        K.spmm_binned(bv, bi, x, g, edge_scale=sc.requires_grad_(True))


# ----------------------------------------------------------------------------- CUDA

@cuda_only
@pytest.mark.parametrize("sddmm", [False, True])
@pytest.mark.parametrize("B", [1, 3, 64])
def test_cuda_matches_cpu(cuda, sddmm, B):
    K.USE_SDDMM = sddmm
    n, e = 2000, 50_000                       # several chunks at chunk=7000
    pre, post = K.random_edges(n, e, seed=20)
    g_cpu = K.SparseGraph.from_edges(pre, post, n)
    g_gpu = g_cpu.to(cuda)
    assert g_gpu.device.type == "cuda" and g_gpu.n_edges == e
    v, x = rand_inputs(g_cpu, B, torch.float32, seed=1)
    go = torch.randn(n, B)
    y = K.spmm(v, x, g_cpu, chunk=7000)
    y.backward(go)
    v2 = v.detach().to(cuda).requires_grad_(True)
    x2 = x.detach().to(cuda).requires_grad_(True)
    y2 = K.spmm(v2, x2, g_gpu, chunk=7000)
    y2.backward(go.to(cuda))
    torch.testing.assert_close(y2.cpu(), y, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x2.grad.cpu(), x.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v2.grad.cpu(), v.grad, rtol=1e-5, atol=1e-5)


@cuda_only
def test_cuda_sddmm_matches_chunked(cuda):
    n, e = 3000, 80_000
    pre, post = K.random_edges(n, e, seed=21)
    g = K.SparseGraph.from_edges(pre, post, n, device=cuda)
    go = torch.randn(n, 16, device=cuda)
    grads = []
    for sddmm in (False, True):
        K.USE_SDDMM = sddmm
        v, x = rand_inputs(g, 16, torch.float32, seed=2)
        K.spmm(v, x, g, chunk=9999).backward(go)
        grads.append(v.grad.clone())
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-5, atol=1e-5)


@cuda_only
@pytest.mark.parametrize("sddmm", [False, True])
def test_cuda_gradcheck(cuda, sddmm):
    K.USE_SDDMM = sddmm
    g = small_graph(14, device=cuda)
    v, x = rand_inputs(g, 3, torch.float64)
    assert torch.autograd.gradcheck(lambda a, b: K.spmm(a, b, g, chunk=7), (v, x), eps=1e-6, atol=1e-4, rtol=1e-4)
    bv, bi, sc, x = binned_setup(g, 5, 3, torch.float64, True)
    fn = lambda a, b: K.spmm_binned(a, bi, b, g, edge_scale=sc, chunk=7)
    assert torch.autograd.gradcheck(fn, (bv, x), eps=1e-6, atol=1e-4, rtol=1e-4)


@cuda_only
@pytest.mark.parametrize("eager_first", [False, True], ids=["graphed-first", "eager-first"])
def test_cuda_graph_capture_matches_eager(cuda, eager_first):
    """Stretch: a T-substep loop with forward+backward captured in a CUDA graph equals eager."""
    n, e, B, T = 2000, 50_000, 16, 4
    pre, post = K.random_edges(n, e, seed=23)
    g = K.SparseGraph.from_edges(pre, post, n, device=cuda)
    v, h0 = rand_inputs(g, B, torch.float32, seed=3, grads=(True, False))
    v.data.mul_(0.1)

    def loop(values, h):
        for _ in range(T):
            h = torch.tanh(0.9 * h + K.spmm(values, h, g))
        return h

    def eager():
        v.grad = None
        y = loop(v, h0)
        y.sum().backward()
        return y.detach().clone(), v.grad.clone()

    if eager_first:
        y_ref, g_ref = eager()
    torch.cuda.synchronize()
    graphed = torch.cuda.make_graphed_callables(loop, (v, h0), num_warmup_iters=2)
    outs = []
    for _ in range(2):                       # replay twice: second replay must give the same answer
        v.grad = None
        y2 = graphed(v, h0)
        y2.sum().backward()
        outs.append((y2.detach().clone(), v.grad.clone()))
    if not eager_first:
        y_ref, g_ref = eager()
    for y2, g2 in outs:
        torch.testing.assert_close(y2, y_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(g2, g_ref, rtol=1e-4, atol=1e-5)


@cuda_only
def test_cuda_binned_matches_cpu(cuda):
    n, e = 2000, 50_000
    pre, post = K.random_edges(n, e, seed=22)
    g_cpu = K.SparseGraph.from_edges(pre, post, n)
    g_gpu = g_cpu.to(cuda)
    bv, bi, sc, x = binned_setup(g_cpu, 50, 8, torch.float32, True)
    go = torch.randn(n, 8)
    K.spmm_binned(bv, bi, x, g_cpu, edge_scale=sc).backward(go)
    bv2 = bv.detach().to(cuda).requires_grad_(True)
    x2 = x.detach().to(cuda).requires_grad_(True)
    K.spmm_binned(bv2, bi.to(cuda), x2, g_gpu, edge_scale=sc.to(cuda)).backward(go.to(cuda))
    torch.testing.assert_close(bv2.grad.cpu(), bv.grad, rtol=1e-4, atol=1e-4)   # atomics: order differs
    torch.testing.assert_close(x2.grad.cpu(), x.grad, rtol=1e-5, atol=1e-5)
