# M0/M1 report: setup, data, kernel, front end, harness (2026-09-14)

Every number below is **measured** on the Intel Unnati cluster (master: 24-core Xeon Silver 4310,
125 GB; node1: 56-core Xeon Gold 6330, 2× A30 24 GB) unless marked *estimated*. Commands are
given so each number can be re-run. Nothing here is a training result.

## 1. Import ledger vs the doomfly reference (`python -m flybrain.data.import_malecns --verify-sha`)

| item | ours | doomfly reference | pass |
|---|---|---|---|
| raw weight rows | 151,856,684 | 151,856,684 | pass |
| unresolved objects | 33,013 | 33,013 | pass |
| non-neuronal excluded | 11,864 | 11,864 | pass |
| neurons | 166,700 | 166,700 | pass |
| edges | 25,582,938 | 25,582,938 | pass |
| synaptic contacts | 124,177,617 | 124,177,617 | pass |

Source files SHA-256-verified against `data/malecns_v1/SHA256SUMS`; weights file 1,051,241,946 B (the
size doomfly quotes). Wall 122 s, peak RSS 2.4 GB on master. Node policy: annotated entry with a
superclass, `status != 'Glia'`, no Traced restriction; all released edges between retained
entries, no threshold, 101 autapses kept. Signs from `consensus_nt` (ACh/DA/OA/5-HT/unclear → +1;
GABA/Glu/histamine → −1): 107,438 excitatory / 59,262 inhibitory neurons; 2,999 `unclear`, 178
without an NT row (both default +1, a pre-registered sensitivity axis; the raw string is stored).

## 2. Subgraph (`python -m flybrain.data.subgraph`)

The plan's budget clamp (≤ 20,000 dynamic neurons, ≤ 2.5 M edges) forced `w_min` from 5 to **32**
and destroyed named pathways. With the caps lifted (affordable on full A30s) the picture is:

| version | w_min | dynamic neurons | clamped inputs | edges | weight kept (within nodes / of dynamic inputs / of whole brain) | forced cells missing | DNs reachable ≤ 4 hops |
|---|---|---|---|---|---|---|---|
| v1 (plan cap) | 32 | 19,757 | 30,906 | 180,470 | 0.388 / 0.247 / 0.095 | **1,717 / 3,484** (LC12, LC16, PFL2, PFL3, PEN2 entirely) | 819 / 1,314 |
| w_min 10 | 10 | 95,496 | 30,906 | 1,955,877 | 0.542 / 0.512 / 0.377 | 84 | 1,301 |
| w_min 8 | 8 | 101,752 | 30,906 | 2,641,759 | 0.595 / 0.570 / 0.427 | 16 | 1,306 |
| **v5 (proposed)** | **5** | **108,062** | **30,906** | **4,639,332** | **0.715 / 0.692 / 0.524** | **0** (only the `P1*` pattern has no cells under that name) | **1,312** |

Notes. 40 % of all MaleCNS edges are single-synapse; the plan's "≥ 80 % of weight retained" gate
is not reachable at any `w_min` ≥ 2 under the 4-hop rule and must be re-set (v5 keeps 69 % of the
dynamic nodes' inputs). VNC excluded (`vnc_*` superclasses, 20,429 cells). Readout set:
`configs/readout_v1.json`, 36 cells (17 types × 2 sides, MDN ×4), sha256 `650631…089ae`, all 36
reachable from the clamped inputs (DNa02 at 2 hops, DNa01 at 4). Left/right audit: R1-R6 have
2,265 R vs 1,112 L; 1,488 of 8,608 types in v5 are flagged < 0.9 L/R ratio (per-side
normalisation is on in the front end; the asymmetric-vs-subsampled comparison is a sensitivity arm).

## 3. Null graphs (`python -m flybrain.data.nulls`)

| graph | kind | attempted swaps | achieved | frac edges changed | wall | peak RSS | invariants |
|---|---|---|---|---|---|---|---|
| v1 (180k edges) | N1 seed 0 | 3,609,400 | 3,521,461 | 0.995 | 1.2 s | 172 MB | all pass |
| v1 | N1 seed 1 | 3,609,400 | 3,521,976 | 0.995 | 1.1 s | 172 MB | all pass |
| v1 | N8 seed 0 | - | - | 0 (magnitudes permuted) | 0.1 s | 196 MB | all pass |
| v5 (4.64 M edges) | N1 seed 0 | 92,786,640 | 91,966,915 | 0.997 | 39 s | 889 MB | all pass (26 autapses preserved) |
| v5 | N1 seed 1 | 92,786,640 | 91,967,400 | 0.997 | 38 s | 889 MB | all pass |
| v5 | N8 seed 0 | - | - | 0 | 2.7 s | 921 MB | all pass |

Invariants checked on the real file: in/out degree of every node, weight multiset, sign array,
clamped-node out-degree, output-node in-degree, autapse count. Open pre-registration choice: N1
keeps each weight with its **presynaptic** neuron by default (`--weight-carrier pre`); `post` and
`shuffle` are implemented.

## 4. Gain matching (`python -m flybrain.model.gain --subgraph … --op-point {drive,rest,fixed_point}`)

Brain piece v5 (139k nodes, 4.64 M edges), type-tied parametrisation (8,608 types → 42,432 trainable:
4 per type + 8,000 pair scales covering 50.8 % of pair weight; global inhibitory centring factor 1.86),
target linearised gain g* = 0.95, clamped inputs at level 1.0, seed 0:

| operating point | w0 | achieved gain | bisection evals | mean dynamic rate | saturated | silent | self-consistent gain at the fixed point |
|---|---|---|---|---|---|---|---|
| drive (plan default) | 0.00541 | 0.949 | 9 | 0.672 | 0.18 % | 0.02 % | 0.63 |
| rest | 0.00829 | 0.951 | 14 | 0.654 | 0.28 % | 0.18 % | 0.76 |
| fixed_point (self-consistent) | 0.01379 | 0.974 (tolerance not met in 45 evals) | 45 | 0.630 | 0.41 % | 0.89 % | 0.97 |
| per_edge, rest (4,855,456 trainable; same init as type-tied by construction) | 0.00829 | 0.951 | 14 | 0.654 | 0.28 % | 0.18 % | 0.76 |

On the real graph the saturation pathology seen on the 80-node synthetic fixture (41 % saturated at
the drive point) does **not** occur: both operating points leave > 99.7 % of units in the linear
range and the two scales differ by only 1.5×. Either is defensible; `rest` lands closer to the
target at the self-consistent fixed point (0.76 vs 0.63).

Finding from the synthetic fixture (37 tests): the self-consistent nonlinear gain is non-monotone
in the global scale `w0`; matching the *linearised* gain at the mean-drive operating point can leave
a large fraction of units saturated, matching at rest leaves them mostly unsaturated. Which
operating point is pre-registered is a decision for the plan owner; all three are deterministic
and identical in recipe for real and null graphs.

## 5. Kernel (`flybrain/model/spmm.py`): correctness and speed

Correctness: 72 kernel tests + 31 core tests pass on node1 (CUDA) and master (CPU): float64
gradcheck vs dense, CPU/CUDA agreement, binned (type-tied) vs per-edge equivalence, chunk
boundaries, CUDA-graph capture vs eager. The per-edge gradient uses cuSPARSE SDDMM on the GPU
(`sampled_addmm`, values in CSR order) with a chunked gather fallback; no `[E, B]` tensor is ever
materialised beyond one chunk.

Speed on synthetic graphs (custom backward, brain-steps/s = batch × substeps / s):

| graph | mode | 2 GB slice (Qwen up) | full A30 (Qwen down) | node1 CPU 48 thr | master CPU 20 thr |
|---|---|---|---|---|---|
| 10k nodes / 1M edges | inference | 252k (B=64) | 364k (B=1024) | 93k | 64k |
| 10k / 1M | train | 14.4k (B=64, 0.86 GB) | 26.5k (B≥256, plateau) | 587 | 293 |
| 20k / 2.5M | train | - | 10.4k (B≥256) | - | - |
| 166.7k / 25.5M (whole CNS) | inference | 6.6k (B=64) | 8.4k (B=256) | 1,247 | 385 |
| 166.7k / 25.5M | train | 100 (B=8, 1.65 GB) | 538 (B=64, 3.9 GB) | 50 | 10 |

Real v5 matrix (139k nodes, 4.64 M edges) on one A30 (`python -m flybrain.model.bench_real`):

| param | trainable | batch | mode | brain-steps/s | env-decisions/s | peak VRAM |
|---|---|---|---|---|---|---|
| type_tied | 42,432 | 64 | inference | 23,900 | 3,985 | 0.56 GB |
| type_tied | 42,432 | 64 | train (fwd+bwd, per-decision checkpoint) | **2,350** | **391** | 2.4 GB |
| type_tied | 42,432 | 256 | inference | 31,200 | 5,204 | 1.4 GB |
| type_tied | 42,432 | 256 | train | 1,875 | 313 | 8.4 GB |
| per_edge | 4,855,456 | 64 | inference | 25,900 | 4,318 | 0.54 GB |
| per_edge | 4,855,456 | 64 | train | **2,613** | **435** | 2.3 GB |
| per_edge | 4,855,456 | 256 | inference | 32,000 | 5,327 | 1.4 GB |
| per_edge | 4,855,456 | 256 | train | 1,916 | 319 | 8.3 GB |

Training is node-bound (139k rows), not edge-bound: batch 64 beats batch 256 once the K=6 substeps
are recomputed in backward. At ~400 decisions/s, one 5 M-frame take_cover seed (1.25 M decisions,
2 PPO epochs) is ≈ **2 h on one card** before env/Python overhead; the 250-run confirmatory tier
≈ 500 card-hours ≈ **10 days on both cards**. *Estimated*, not run: w_min 8 (102k / 2.64 M) would
be ~1.5× faster; CUDA-graph capture of the substep loop is untried on this graph.

## 6. Front end (`python -m flybrain.vision.stimuli --json …`), real MaleCNS geometry

Geometry: 1,771 hex columns (879 L, 892 R) from `assignedOlHex1/2`; lattice frame per eye
estimated from T4 dendrite geometry (Mi9 tip vs Mi4/C3 base), no fallback needed; 108° frustum
drives ~250 columns per eye; mean-luminance fill outside it; 30,906 clamped cells indexed.

Gratings (column path, 30° wavelength, 2 Hz; DSI = (PD−ND)/(PD+ND); PD = vector average, correct
if within 45° of the Maisak-2013 direction for that subtype and eye):

| type | side | units | median DSI | PD correct | argmax correct | responding |
|---|---|---|---|---|---|---|
| T4a | L / R | 158 / 156 | 0.985 / 0.985 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.94 |
| T4b | L / R | 162 / 159 | 0.985 / 0.985 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T4c | L / R | 170 / 161 | 0.989 / 0.990 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T4d | L / R | 168 / 172 | 0.989 / 0.989 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T5a | L / R | 157 / 156 | 0.987 / 0.987 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.95 |
| T5b | L / R | 162 / 155 | 0.987 / 0.987 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T5c | L / R | 171 / 164 | 0.991 / 0.991 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| T5d | L / R | 163 / 155 | 0.991 / 0.991 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

Temporal-frequency tuning (T5d L): DSI 1.00 / 1.00 / 0.99 / 0.91 / 0.66 / 0.42 at 0.5 / 1 / 2 / 4 /
8 / 12 Hz. Looming proxy (T5 OFF channels, dark expanding disc) rises with expansion rate 5 → 80
°/s (0.70 → 13.9) then saturates/declines (12.2 at 160, 8.2 at 320 °/s); receding discs drive the
T4 (ON) proxy instead; translation is 5–10× weaker. Flashes: ON types (Mi1, Mi4) respond to
brightening only, OFF types (Mi9, Tm1/2/4/9) to darkening only; chromatic types silent to grey.
**Caveat:** T4a R and T5a R have 6 % / 5 % non-responding units (columns whose front neighbour is
missing at the lattice edge); the synthetic-lattice tests showed a single-axis correlator is broad
(±45°), so argmax over 8 directions is not a valid PD metric; the vector-average metric is.

## 7. ViZDoom harness (`flybrain/env`)

Headless on Ubuntu 20.04 with no X server: works. Single env 4,075 steps/s per core (basic,
160×120). Pool on master (`python -m flybrain.env.pool --bench`):

| workers × envs | agent-steps/s | RSS per worker (python + engine) |
|---|---|---|
| 8 × 1 | 1,880–2,200 | 31 + 21 MB |
| 16 × 1 | 3,030 | 31 + 22 MB |
| 20 × 1 | 3,290–3,460 | 32 + 22 MB |
| 16 × 4 = 64 envs | 3,840 | 33 + 85 MB |

Scripted floors, 100 held-out episodes each (seeds ≥ 10,000; `python -m flybrain.env.floors`):

| scenario | policy | reward mean | IQM [95 % CI] | survival tics | kills |
|---|---|---|---|---|---|
| take_cover | random | 255.1 | 229 [205, 257] | 259 | 0 |
| take_cover | centroid (labels-buffer heuristic) | 272.5 | 267 [250, 285] | 276 | 0 |
| take_cover | constant strafe left / right | 183.0 / 170.8 | 176 / 162 | 187 / 175 | 0 |
| take_cover | stand still (attack / turn / forward are no-ops here) | 154.3 | 145 [135, 156] | 158 | 0 |
| defend_the_line | centroid (aims via labels) | 19.6 | 19.5 [18.6, 20.6] | 710 | 20.7 |
| defend_the_line | constant attack | 9.9 | 9.9 [9.4, 10.2] | 544 | 11.0 |
| defend_the_line | random | 7.4 | 7.0 [6.3, 7.8] | 547 | 8.4 |
| defend_the_line | spin + fire | 4.1 | 3.6 [2.9, 4.4] | 385 | 5.1 |
| defend_the_line | constant turn / strafe / forward | 3.2 | 2.9 [2.3, 3.4] | 410 | 4.2 |

Published anchors: take_cover ≈ 1,092 ± 556 for World Models' 1,088-parameter controller
("solved" ≥ 750); defend_the_line ≈ 27 kills early and ≈ 40 at 500 M frames for Sample Factory's
CNN+GRU. Headroom between the blind floor and competence exists on both scenarios.

Seed → spawn variation (`python -m flybrain.env.doom --check-seeds`, 10 seeds, 15 steps): take_cover 9 / 10 distinct
spawn snapshots, defend_the_line 10 / 10, both exactly reproducible per seed.

## 8. GPU sharing and ops (`flybrain/ops`)

The A30s are held by a root-owned vLLM (Qwen 27B) service; a passwordless sudo whitelist now
allows `unnati` to stop/start `vllm-qwen.service` and `agent-watchdog.timer` (nothing else).
`gpu_guard.py` (NVML watchdog, 1.6 GB cap under sharing, exits if the vLLM PIDs vanish) and
`qwen.py` / `scripts/gpu_window.sh` (stop → run → restart, with trap-based restore) have 54 mocked
tests; the window script was exercised 8× with `sleep 30` while Qwen was already down. vLLM
restart takes 2 min 50 s; it fails to start if we hold memory, so every GPU job is killed before
`start`.

## 9. Gate table (milestones M0 / M1)

| gate | result |
|---|---|
| M0 ledger reconciles to the row | **GO** (6/6) |
| M0 ViZDoom headless, no X, no sudo | **GO** |
| M0 budget clamp yields ≤ 20k dynamic / ≤ 2.5 M edges at w_min ≤ 10 keeping ≥ 80 % weight | **NO-GO as written**: at w_min ≤ 10 the 4-hop core is 95–108k neurons; ≥ 80 % weight is unreachable (max 71.5 % at w_min 5). Proposed re-scope: v5 (108k / 4.64 M / 69 % of inputs, all pathways present). |
| M0 ≥ 95 % of readout DNs reachable ≤ 4 hops | **GO** (36/36; 1,312/1,314 of all DNs in v5) |
| M0 per-type L/R audit done | **GO** (reported; 1,488 types flagged) |
| M0 env RSS / 64-env footprint measured | **GO** (64 envs ≈ 1.9 GB) |
| M1 gradcheck ≤ 1e-4 vs dense; CPU/GPU ≤ 1e-5 | **GO** |
| M1 peak VRAM ≤ 1.3 GB (2 GB-slice rule) | superseded: full cards available; v5 numbers below |
| M1 ≥ 6,400 env-decisions/s at B=64 with backward | **NO-GO as written** for v5 (391–435; the gate was set for a 20k-node core, v5 has 139k nodes). Consequence: ≈ 2 h per 5 M-frame seed, ≈ 10 days for the 250-run tier on two cards. Decision: accept, or w_min 8, or fewer seeds. |
| M3 T4/T5 DSI ≥ 0.5, correct polarity and preferred directions; LPLC2-proxy monotone | **GO** for DSI/PD/polarity; loom proxy monotone to 80 °/s then saturates (gate wording should say "monotone over 5–80 °/s") |

## 10. Issues, worst first

1. **Scope**: the plan's 20k-neuron cap is incompatible with the 4-hop rule on real MaleCNS; v5 (108k neurons) requires full A30s (state ≈ 10 GB at batch 256 with checkpointing), which means the shared GPU has to be free during training.
2. **Weight-retention gate** (≥ 80 %) must be re-set; 40 % of edges are single synapses.
3. **Gain-matching operating point** must be pre-registered (drive vs rest); on v5 both are healthy (< 0.3 % saturated), unlike the synthetic fixture, so this is a choice, not a problem.
4. **N1 weight carrier** (pre / post / shuffle) is a scientific choice not fixed by the plan.
5. take_cover's scripted floors mostly collapse to "stand still" (only strafe buttons exist); the strafe-only and random floors are the meaningful ones. Sticky actions were off (`--sticky 0.0`) for these floors.
6. The eye map is a planar hex approximation (5.7° spacing); the published Zhao-2025 map is a later upgrade.
7. `P1*` matches no MaleCNS type name (the arousal cells are named `pC1*`, which are present); the arousal-gating story needs the right type list.
8. No pixel-path grating run yet (`--pixel`); the column path isolates physiology from the sampler.

## 11. Gate 0: take_cover, brain piece v5, untrained, gain-matched at `rest` (`python -m flybrain.eval.probe`)

40,000 recorded decisions (626 episodes, random + labels-aiming scripted play, seeds 20,000+), one
frame per decision through the frozen front end and the frozen core (K = 6 substeps); ridge probes
from population rates at the last substep to the labels-buffer targets, lambda picked on validation
episodes, R² on 20 % held-out episodes (7,584 test rows for enemy azimuth). Five N1 rewires.

| population | dims | target | **real wiring (A)** | N1 rewires (5), mean [min–max] | front end only (random 2,048-d projection, no core) |
|---|---|---|---|---|---|
| descending neurons | 1,312 | enemy azimuth | **0.457** | **0.621** [0.611–0.638] | 0.604 |
| descending neurons | 1,312 | projectile azimuth | 0.452 | 0.568 [0.560–0.576] | 0.558 |
| descending neurons | 1,312 | enemy visible | 0.335 | 0.517 [0.507–0.527] | 0.599 |
| descending neurons | 1,312 | log time-to-collision | 0.420 | 0.742 [0.736–0.753] | 0.723 |
| visual projection neurons | 9,199 | enemy azimuth | 0.666 | 0.692 [0.690–0.695] | - |
| pre-registered readout cells | 36 | enemy azimuth | **0.221** | 0.140 [0.130–0.158] | - |
| pre-registered readout cells | 36 | projectile azimuth | 0.323 | 0.348 [0.317–0.367] | - |

Gain matching per graph (same recipe, same seed): real w0 = 0.00828 (mean rate 0.654, 0.28 % saturated);
rewires w0 = 0.0121–0.0127 (mean rate 0.700, 0 % saturated).

**Verdict: NO-GO on both pre-registered criteria** (DN R² 0.457 < 0.50; real − rewired mean = −0.16,
not ≥ +0.10). The confirmatory tier is cut to one scenario and one null and the effort
moves to the reactivity result.

Reading, worst first:
1. At the untrained, gain-matched stage the real wiring delivers **less** target information to the
   descending neurons than degree- and weight-preserving rewires, on every target. This is the
   reservoir result (random mixing is a good reservoir) and matches arXiv 2604.04033.
2. The 36 literature-chosen readout cells are the one place the real wiring wins (0.22 vs 0.14),
   i.e. biologically informed I/O placement carries some signal, which is exactly the confound the
   N-READOUT control exists for. R² 0.22 is weak in absolute terms.
3. **Vision works.** The front end alone decodes azimuth at 0.60 and the fly's visual projection
   neurons at 0.67 under the real wiring; this is the step where doomfly, fly-craftax and
   Closed-Loop Fly failed. The information loss is between VPN and DN (0.67 → 0.46) in the real
   graph, not in the eye.
4. Untested here: the `drive` operating point (sensitivity run), defend_the_line (running), and
   whether training re-routes the information (that is the M6 pilot, which the plan already scopes).

Follow-ups (same protocol):

| run | real wiring, DN azimuth R² | N1 rewires mean | front end only | readout-36: real vs rewires | verdict |
|---|---|---|---|---|---|
| take_cover, op-point `drive` | 0.417 | 0.618 | 0.604 | 0.197 vs 0.121–0.139 | NO-GO |
| defend_the_line, op-point `rest` (256 episodes) | 0.410 | 0.701 | 0.712 | 0.248 vs 0.164–0.285 | NO-GO |

The result is insensitive to the operating point and replicates on the second scenario; on
defend_the_line the readout-36 advantage vanishes. Untrained real wiring is a worse conduit of
target information to the descending neurons than degree/weight-preserving rewiring, everywhere.
