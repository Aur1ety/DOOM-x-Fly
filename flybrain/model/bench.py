"""Micro-benchmark of the spmm kernel: substeps/s and peak memory on synthetic graphs.

One substep = h <- tanh(0.9 h + W @ h) over a random graph. "infer" is forward only under
no_grad; "train" is forward through T substeps plus backward into the per-edge values and h.
All numbers are measured on the host it runs on; env-decisions/s at K=6 is derived (steps/s * B / 6).

GPU guard (node1's A30s are held by a root-owned vLLM service): refuse to start unless >= 2.2 GiB
free, cap the caching allocator at 1.4/24 of the card, and run an NVML watchdog thread (0.5 s)
that hard-exits if card-free < 500 MiB or this process's used memory > 1.6 GiB.

    python -m flybrain.model.bench --device cuda            # node1
    python -m flybrain.model.bench --device cpu --threads 8 # master
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # one card, before torch is imported

import numpy as np
import torch

from flybrain.model import spmm as K

MIB = 2 ** 20
GIB = 2 ** 30


# ----------------------------------------------------------------------------- GPU guard

def gpu_guard(min_free_gib: float = 2.2, fraction: float = 1.4 / 24.0, poll_s: float = 0.5,
              free_floor_mib: int = 500, own_cap_gib: float = 1.6) -> dict:
    """Enforce the shared-card rules for CUDA device 0 of this process. Returns a status dict.

    Exits the process (code 2) if the card has too little free memory to start, and starts a
    daemon thread that os._exit(3)s if free memory or our own usage crosses the limits, or (4) if
    the other compute processes that were on the card at start (the vLLM service) disappear.
    """
    import pynvml

    if not torch.cuda.is_available():
        raise SystemExit("gpu_guard: CUDA not available")
    free, total = torch.cuda.mem_get_info(0)
    if free < min_free_gib * GIB:
        print(f"gpu_guard: only {free/GIB:.2f} GiB free on the card (< {min_free_gib}); refusing to start",
              file=sys.stderr)
        raise SystemExit(2)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)

    phys = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(phys)
    me = os.getpid()

    def own_and_others():
        own, others = 0, set()
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if p.pid == me:
                own += p.usedGpuMemory or 0
            else:
                others.add(p.pid)
        return own, others

    _, others0 = own_and_others()
    missing = 0

    def watch():
        nonlocal missing
        while True:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                own, others = own_and_others()
            except pynvml.NVMLError as err:   # transient NVML failure: keep going, do not kill
                print(f"gpu_guard: NVML error {err}", file=sys.stderr)
                time.sleep(poll_s)
                continue
            if mem.free < free_floor_mib * MIB:
                print(f"gpu_guard: card free {mem.free/MIB:.0f} MiB < {free_floor_mib}; exiting", file=sys.stderr)
                os._exit(3)
            if own > own_cap_gib * GIB:
                print(f"gpu_guard: own usage {own/GIB:.2f} GiB > {own_cap_gib}; exiting", file=sys.stderr)
                os._exit(3)
            # NVML's process list can come back empty for a single poll; require 3 in a row (1.5 s)
            missing = missing + 1 if (others0 and not others) else 0
            if missing >= 3:
                print("gpu_guard: the other GPU processes (vLLM) vanished; releasing the card", file=sys.stderr)
                os._exit(4)
            time.sleep(poll_s)

    threading.Thread(target=watch, daemon=True, name="gpu_guard").start()
    return dict(physical_index=phys, free_at_start_mib=free // MIB, total_mib=total // MIB,
                fraction=fraction, other_pids=sorted(others0), own_query=lambda: own_and_others()[0])


# ----------------------------------------------------------------------------- benchmark

def make_graph(n: int, n_edges: int, device, seed: int = 0, int32_perm: bool = False) -> K.SparseGraph:
    pre, post = K.random_edges(n, n_edges, seed=seed)
    g = K.SparseGraph.from_edges(pre, post, n, device=device)
    if int32_perm:
        # Proposed contract change (CONTRACTS.md says int64): E < 2^31 always holds here and the
        # kernel only ever uses perm through index_select, which accepts int32. Saves 4 B/edge.
        g.perm_csr_to_csc = g.perm_csr_to_csc.to(torch.int32)
    return g


def substeps(values: torch.Tensor, h: torch.Tensor, g: K.SparseGraph, T: int) -> torch.Tensor:
    for _ in range(T):
        h = torch.tanh(0.9 * h + K.spmm(values, h, g))
    return h


def estimate_bytes(g: K.SparseGraph, B: int, T: int, train: bool) -> int:
    """Rough peak allocator estimate used to skip configs that cannot fit under the guard.

    Calibrated against measured peaks (node1, 2026-09-14): under-estimates by ~5-10 % without the
    fixed cuSPARSE slack, hence the 64 MiB term.
    """
    e, m = g.n_edges, g.n
    est = g.n_bytes() + 4 * e + 4 * m * B * 3            # index arrays, values, h/x/tmp
    if train:
        est += 4 * e                                       # grad_values
        est += 4 * m * B * T                               # saved h per substep (shared by spmm/tanh)
        if K.USE_SDDMM and K._sddmm_available(g.device):
            est += 4 * e * 2                               # SDDMM zeros + output (transient)
        else:
            est += 4 * e + 4 * K.edge_chunk(B) * B * 2     # values[perm] + two chunk tensors
    return int(est * 1.05) + 64 * MIB


def bench_one(name: str, g: K.SparseGraph, B: int, T: int, train: bool, reps: int, device,
              cuda_graph: bool = False) -> dict:
    is_cuda = device.type == "cuda"
    torch.manual_seed(0)
    scale = 1.0 / np.sqrt(max(1.0, g.n_edges / g.n))
    values = (torch.randn(g.n_edges, device=device) * scale).requires_grad_(train)
    h0 = torch.randn(g.n, B, device=device)
    loop = lambda v, h: substeps(v, h, g, T)
    if cuda_graph:
        # Stretch: capture the whole T-substep loop (forward, and backward if training) in one
        # CUDA graph. Static shapes only; the eager warm-up runs first so cuSPARSE plans/SDDMM probe
        # happen outside capture.
        if train:
            loop = torch.cuda.make_graphed_callables(loop, (values, h0), num_warmup_iters=3)
        else:
            with torch.no_grad():
                loop(values, h0)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_out = substeps(values, h0, g, T)
            loop = lambda v, h: (graph.replay(), static_out)[1]

    def step():
        if train:
            loss = loop(values, h0).sum()
            loss.backward()
            values.grad = None
        else:
            with torch.no_grad():
                loop(values, h0)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    step()                                                 # warm-up (cuSPARSE plans, SDDMM probe)
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    if is_cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    steps_s = reps * T / dt
    out = dict(graph=name, mode=("train" if train else "infer") + (" (cuda-graph)" if cuda_graph else ""),
               B=B, T=T, steps_s=steps_s,
               ms_step=1e3 / steps_s, env_steps_s=steps_s * B, decisions_s_k6=steps_s * B / 6)
    if is_cuda:
        out["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / MIB
        out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
    del values, h0
    if is_cuda:
        torch.cuda.empty_cache()
    return out


def profile_train_step(g: K.SparseGraph, B: int, T: int, device) -> None:
    """torch.profiler breakdown of one forward+backward pass through T substeps (diagnostic)."""
    from torch.profiler import ProfilerActivity, profile

    values = (torch.randn(g.n_edges, device=device) * 0.03).requires_grad_(True)
    h0 = torch.randn(g.n, B, device=device)
    for _ in range(2):                                     # warm-up
        substeps(values, h0, g, T).sum().backward()
        values.grad = None
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with profile(activities=acts) as prof:
        substeps(values, h0, g, T).sum().backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
    key = "cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=key, row_limit=14))


def configs(which: str):
    small = ("10k/1M", 10_000, 1_000_000)
    brain = ("166.7k/25.5M", 166_700, 25_500_000)
    if which in ("small", "all"):
        for B in (16, 64):
            yield small, B, 32, False
        for B in (16, 64):
            yield small, B, 32, True
    if which in ("brain", "all"):
        for B in (1, 16, 64):
            yield brain, B, 16, False
        yield brain, 8, 16, True


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="spmm kernel micro-benchmark")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--which", default="all", choices=["small", "brain", "all"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=None, help="CPU threads (default: leave torch's)")
    p.add_argument("--no-sddmm", action="store_true", help="force the chunked per-edge gradient on CUDA")
    p.add_argument("--chunk", type=int, default=None, help="force edges per backward chunk")
    p.add_argument("--budget-mib", type=int, default=1144, help="skip configs whose estimate exceeds this (1.2 GB rule)")
    p.add_argument("--force", action="store_true", help="run even if the estimate exceeds the budget")
    p.add_argument("--cuda-graph", action="store_true", help="stretch: capture the substep loop in a CUDA graph")
    p.add_argument("--profile", action="store_true", help="profile one train step of the first config and exit")
    p.add_argument("--int32-perm", action="store_true", help="hold perm_csr_to_csc as int32 on the device (-4 B/edge)")
    a = p.parse_args(argv)

    device = torch.device(a.device)
    host = socket.gethostname().split(".")[0]
    guard = None
    if device.type == "cuda":
        guard = gpu_guard()
        torch.cuda.init()
        torch.ones(1, device=device)
        torch.cuda.synchronize()
        ctx_mib = guard["own_query"]() / MIB
        print(f"host {host} | cuda phys {guard['physical_index']} | free at start {guard['free_at_start_mib']} MiB"
              f" | other pids {guard['other_pids']} | CUDA context (NVML, no allocations) {ctx_mib:.0f} MiB")
    else:
        if a.threads:
            torch.set_num_threads(a.threads)
        print(f"host {host} | cpu threads {torch.get_num_threads()}")
    if a.no_sddmm:
        K.USE_SDDMM = False
    if a.chunk:
        K.EDGE_CHUNK = min(K.EDGE_CHUNK, a.chunk)
        K.CHUNK_ELEMS = a.chunk * 64
    print(f"torch {torch.__version__} | sddmm {'on' if K.USE_SDDMM else 'off'} | reps {a.reps}"
          f" | perm {'int32' if a.int32_perm else 'int64'}")

    if a.profile:
        (name, n, e), B, T, _ = next(configs(a.which))
        profile_train_step(make_graph(n, e, device), B, T, device)
        return

    rows, cache = [], {}
    for (name, n, e), B, T, train in configs(a.which):
        if name not in cache:
            t0 = time.perf_counter()
            cache.clear()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            cache[name] = make_graph(n, e, device, int32_perm=a.int32_perm)
            print(f"built {name} in {time.perf_counter()-t0:.1f}s, index {cache[name].n_bytes()/MIB:.0f} MiB", flush=True)
        g = cache[name]
        est = estimate_bytes(g, B, T, train) / MIB
        if device.type == "cuda" and est > a.budget_mib and not a.force:
            rows.append(dict(graph=name, mode="train" if train else "infer", B=B, T=T, skipped=f"est {est:.0f} MiB"))
            print(f"skip {name} {'train' if train else 'infer'} B={B}: estimated {est:.0f} MiB > budget", flush=True)
            continue
        r = bench_one(name, g, B, T, train, a.reps, device, cuda_graph=a.cuda_graph and device.type == "cuda")
        if guard is not None:
            r["nvml_own_mib"] = guard["own_query"]() / MIB
        rows.append(r)
        print(f"{name:>13} {r['mode']} B={B:<3} T={T:<3} {r['steps_s']:8.1f} steps/s"
              + (f"  peak alloc {r['peak_alloc_mib']:.0f} MiB  nvml {r['nvml_own_mib']:.0f} MiB" if guard else ""),
              flush=True)

    print()
    print(f"### spmm bench: {host} {a.device}" + (f" threads={torch.get_num_threads()}" if a.device == "cpu" else ""))
    print()
    cols = "| graph | mode | B | T | steps/s | ms/step | env-steps/s (B x steps/s) | decisions/s @K=6 (derived) | peak alloc MiB | peak reserved MiB | NVML own MiB |"
    print(cols)
    print("|" + "---|" * (cols.count("|") - 1))
    for r in rows:
        if "skipped" in r:
            print(f"| {r['graph']} | {r['mode']} | {r['B']} | {r['T']} | skipped: {r['skipped']} | | | | | | |")
            continue
        mem = (f"{r['peak_alloc_mib']:.0f} | {r['peak_reserved_mib']:.0f} | {r['nvml_own_mib']:.0f}"
               if "peak_alloc_mib" in r else "n/a | n/a | n/a")
        print(f"| {r['graph']} | {r['mode']} | {r['B']} | {r['T']} | {r['steps_s']:.1f} | {r['ms_step']:.2f} | "
              f"{r['env_steps_s']:.0f} | {r['decisions_s_k6']:.0f} | {mem} |")


if __name__ == "__main__":
    main()
