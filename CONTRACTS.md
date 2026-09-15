# Data contracts between modules

All arrays are numpy; files are `.npz` (compressed) or `.parquet`. Paths are relative to
`$FLYBRAIN_DATA` (server: `~/flybrain-doom/data`). Outputs go to `$FLYBRAIN_OUT`
(server: `~/flybrain-doom/outputs`).

## Graph convention (everywhere)

- A neuron's **index** is its row in `neurons.parquet` (0..N-1). `bodyId` is the MaleCNS id.
- Edge `(pre, post, w)`: `w` = synapse count (float32, >0). Sign belongs to the **presynaptic**
  neuron: `sign[pre] in {+1, -1}` from `consensus_nt` (ACh/DA/OA/5-HT/unclear -> +1;
  GABA/Glu/histamine -> -1). Signs are never stored inside `w`.
- **CSR is indexed by POST** (rows = post, cols = pre) so that `W @ r` gives each neuron's input:
  `indptr_post[N+1]`, `indices_pre[E]`, `weight[E]` (int64, int32, float32).
- **CSC is indexed by PRE** (rows = pre, cols = post): `indptr_pre[N+1]`, `indices_post[E]`,
  `perm_csr_to_csc[E]` such that `weight_csc = weight[perm_csr_to_csc]`.
- Edge order in `weight` is the CSR order. Any per-edge parameter (theta) follows CSR order.

## `graph_full.npz` (from `flybrain.data.import_malecns`)

Keys: `n_neurons`, `n_edges`, `bodyId[N]`, `sign[N]` (int8), `indptr_post`, `indices_pre`,
`weight`, `indptr_pre`, `indices_post`, `perm_csr_to_csc`, plus `ledger` (JSON string with:
raw_rows, unresolved_objects, non_neuronal_excluded, n_neurons, n_edges, n_contacts, and the
doomfly reference numbers and pass/fail per item).

Reference ledger (doomfly, must match to the row or the diff is reported, never fudged):
raw weight rows 151,856,684; unresolved objects 33,013; non-neuronal excluded 11,864;
neurons 166,700; edges 25,582,938; contacts 124,177,617. Node policy: every annotated entry
with a superclass, including uncertain classes; exclude explicit glia; no restriction to Traced
status. Edge policy: all released edges between retained entries, no weight threshold, keep
autapses.

## `neurons.parquet`

Columns: `idx`, `bodyId`, `type`, `instance`, `superclass`, `class`, `subclass`, `somaSide`,
`rootSide`, `assignedOlHex1`, `assignedOlHex2`, `status`, `statusLabel`, `consensus_nt`,
`sign`, `type_id` (int32, dense id over unique `type`; -1 for null type).

## `subgraph_<name>.npz` (from `flybrain.data.subgraph`)

Keys: `node_idx[M]` (index into `neurons.parquet`), `bodyId[M]`, `is_clamped[M]` (bool: rates set
by the front end; incoming edges dropped), `is_output[M]` (bool: pre-registered readout DNs),
`is_dynamic[M]` (= ~is_clamped), `type_id[M]`, `sign[M]`, local CSR/CSC (`indptr_post`,
`indices_pre`, `weight`, `indptr_pre`, `indices_post`, `perm_csr_to_csc`) over the M local
indices, `w_min` (float), `report` (JSON string: counts per set, achieved w_min, fraction of
total synaptic weight retained, DN reachability within 4 hops, left/right completeness audit,
forced-inclusion hits/misses).

Edges INTO clamped nodes are not stored (their rates are inputs). Edges FROM clamped nodes to
dynamic nodes are stored.

## Null graphs `null_<kind>_<seed>.npz` (from `flybrain.data.nulls`)

Same keys as a subgraph plus `null_kind` in {`N1_degree_weight_rewire`, `N8_mask_only`,
`N2a_type_block`, `N4_erdos_renyi`}, `seed`, `achieved_swaps`, `attempted_swaps`,
`frac_edges_changed`, `elapsed_s`, `peak_rss_mb`. Invariants (tested): in/out degree of every
node preserved (N1, N8), weight multiset preserved (N1), sign array identical (all), clamped-node
out-degree and output-node in-degree preserved (N1, N8), no self-edges created if none existed.

## Readout set `configs/readout_v1.json`

`{"version": 1, "cells": [{"type": "DNa02", "side": "R", "bodyId": ..., "role": "turn"} ...],
 "sha256": "<hash of the cells list>"}`. Written once by `flybrain.data.subgraph --readout`,
then frozen. Roles: turn, forward, backward, dodge, attack_midline, attack_pursuit.

## Model state and front-end interface

- Front end produces `clamped_rates[B, n_clamped]` float32 per substep, in the order of
  `node_idx[is_clamped]`.
- Core exposes `step(h, clamped_rates) -> h` and `readout_features(h) -> [B, F]` with
  `F = 2 * n_output` (rates and first differences of the output cells).
- Parametrisations: `per_edge` (theta[E]), `type_tied` (alpha[T], beta[T], pair_scale[P] for the
  top-P type pairs by weight, plus tau[T], v_rest[T], bias[T]).
- Gain matching: `flybrain.model.gain.match_gain(graph, params, g_star=0.95)` returns the global
  scale `w0` such that the linearised operating-point gain (power iteration on D·W at mean
  drive) equals `g_star`. Applied identically to real and null graphs, same RNG seed index.
