# FINAL PLAN — "LOBULA LOOP"

## 1. Verdict

**Chosen: a principled merge, ml-first (FLYLOOP) as the structural backbone, with bio-first's honesty discipline and hybrid's cheap vision gate + de-risking tools grafted on.** No single design wins: bio-first is the most honest and most feasible but the "will it play" judge rates its ~932-parameter frozen-reservoir policy at 3/10, and a project that never plays produces no comparison at all; ml-first is the only design with enough policy capacity to actually reach competence, but its 2.04M per-edge headline arm degrades the scientific claim to "the connectome is a useful sparsity mask." The merge fixes this by making a **type-tied ~15k-parameter arm the headline** (bio-first/hybrid discipline, ml-first's relegated A-lite promoted), keeping the per-edge arm as an explicitly weaker secondary claim, and running a pre-registered **capacity ladder** so the minimum capacity that plays is discovered in a pilot rather than assumed.

| Design | Feasibility | Honesty | Will-it-play | Total |
|---|---|---|---|---|
| bio-first (FROZEN FLY) | **7** | **8** | 3 | 18 |
| ml-first (FLYLOOP) | 6 | 6 | **8** | 20 |
| hybrid-staged (CHIASM) | 5 | 7 | 5 | 17 |
| **LOBULA LOOP (this plan)** | targets 7 | targets 8 | targets 7 | — |

Key structural decisions and why:

- **Frozen, non-trainable front end; the vision fork is orthogonal to the topology claim.** The topology experiment only requires that the front end and decoder be frozen and *identical across arms*. So the default is ml-first's hand-built EMD front end (low risk, three prior teams failed at simulating the optic lobe). The real-optic-lobe path (hybrid's flyvis transfer) is a **Phase-2 upgrade behind its own gate**, not a dependency. Consequence, stated up front: **we do not claim "fly vision."** The claim is scoped to lobula → optic glomeruli → central brain → descending neurons.
- **Headline arm is type-tied (~15k params), not per-edge.** Per-edge is run and reported, but under a different, weaker claim.
- **Brain only, no VNC.** Turning is stride-length asymmetry across six legs; reading motor neurons without a body model is meaningless (Yang & Wilson, PMC10614758). bio-first's VNC inclusion doubles cost for a passenger it admits may contribute nothing.
- **Single pre-registered primary scenario** (take_cover), with defend_the_line as a pre-registered replication. No "at least one of two scenarios" disjunction — that is a free multiplicity loophole both ml-first and hybrid left open.

---

## 2. Pre-registered success and kill criteria

Frozen at M5, registered on OSF with a third-party timestamp (not just a repo hash), analysis script frozen and committed before any confirmatory data exists, arm labels scrambled under a key held by a script until the analysis is run.

### Success criteria

| # | Claim | Requirement | Scenario |
|---|---|---|---|
| S0 | **It plays** | A-tied beats the best blind/scripted floor (random, spin+fire, constant-forward, constant-action, open-loop replay of own best sequence, pixel-centroid, linear-on-pixels, **linear-on-front-end-features**) by ≥25% relative, Welch p<0.01, 25 seeds | take_cover (primary) |
| S1 | **Reactive, not replay** | Live frames beat black AND mean-luminance AND time-scrambled by ≥20% relative, p<0.01 (R1); yoked video from a different episode collapses to blind floor (R2); mirror test flips the sign of turn-vs-azimuth correlation (R3); per-trial stimulus-locked turn/dodge within 3 tics with a reported latency **distribution**, not just a mean (R4); closed loop beats open-loop replay on held-out seeds (R5) | both |
| S2 | **Fly wiring matters** | A-tied > N1 (degree+weight-preserving rewire, 5 graphs) on ALL THREE: Holm-adjusted Welch p<0.05; rliable probability of improvement ≥0.70 with bootstrap CI lower bound >0.5; normalized gain ≥0.10 (0 = random policy, 1 = over-parameterized tuned GRU) | take_cover primary, defend_the_line must replicate and must not contradict |
| S3 | **Not decoration** | A-tied > N6 (core bypass) and > N-READOUT (null with best-of-20 readout search) by the same three-part standard | take_cover |
| S4 | **Measured weights matter (weaker, separate)** | A-tied > N8 (real mask, randomized magnitudes, sign+degree preserved) by the same standard | take_cover |
| S5 | **Pathway specificity** | Pre-registered lesion exceeds 95th percentile of 20 size-matched random lesions, monotone dose-response, AND the predicted double dissociation (LC4/LPLC2→DNp01/02/04/11 hurts take_cover not defend_the_line; LC10a→AOTU019→DNa02 the reverse) | both |
| S6 | **No difference** | TOST equivalence within ±0.10 normalized. A bare p>0.05 never counts. | — |

Unit of analysis is one training seed scored as its mean over 100 held-out episodes. 25 seeds per confirmatory arm, never below 20. Train seeds 0–999, eval seeds ≥10000, plus one texture-swapped and one layout-swapped held-out variant evaluated with frozen weights.

**Degenerate-normalization guard (missing from all three source designs):** if the tuned-GRU ceiling on a scenario lands within 2× the blind floor's CI width, the normalized-gain threshold is meaningless and that scenario is declared uninformative before unblinding, not reinterpreted after.

### Kill criteria

| # | Fires at | Trigger | Response |
|---|---|---|---|
| K0 | M0 | Import ledger does not reconcile with doomfly's counts to the row; or microbenchmark gives <50 batched decisions/s at B=64 with backward on a shared A30 and <20 on 20 CPU cores | Re-scope via the budget clamp (raise w_min) before building anything; if still short, drop to 1M edges and one scenario |
| K1 | M1 | Gradcheck fails to 1e-4 against dense, or CPU/GPU forward disagree beyond 1e-5 | Kernel is wrong; do not proceed. A silently wrong per-edge gradient still produces plausible learning curves — this is the most dangerous bug class in the plan |
| K2 | M2 | A standard CNN+GRU PPO agent cannot reach published competence on take_cover in our harness | The harness is broken (wrapper, frameskip, action set, termination, eval split), not the science. Fix before touching the fly |
| K3 | M3 | Front end cannot produce T4/T5 DSI ≥0.5 with correct ON/OFF polarity and correct cardinal preferred directions, or monotone LPLC2 loom response | Fix the EMD bank; the flyvis path is Phase-2 and not a rescue for this |
| K4 | M4 | Gate 0: held-out enemy/fireball azimuth R² <0.50 from the readout population, OR fails to beat the mean of 5 degree-preserving rewires by ≥0.10 R² | **Cheapest kill in the plan (~1 GPU-day).** This is the test doomfly, fly-craftax, Closed-Loop Fly and Fly Chess all failed downstream at far greater cost. Cut the confirmatory tier to one scenario and one null; reallocate to the reactivity result |
| K5 | M6 pilot | R1 blindfold fails: live frames do not beat black by ≥20% at p<0.01 | **No "plays Doom" claim of any kind.** This is doomfly's headline failure reproduced. Pivot to signal-flow diagnosis and publish the negative |
| K6 | M6 pilot | Action entropy collapses / policy converges to a near-constant or periodic action sequence in ≥3 of 5 seeds | Degenerate solution (doomfly: attack pressed 93% of tics, behaviour reproduced by one fixed command). Restart with entropy floor raised; if it recurs, the scenario is unusable |
| K7 | M7/M8 | A-tied not better than N1 under all three thresholds | **No "wiring matters" claim, ever.** Report TOST equivalence; this replicates arXiv 2604.04033 at larger scale on a genuinely visual task |
| K8 | M7 | N6 core bypass equivalent to A-tied (TOST ±0.10) | Front end + decoder do the work; the connectome is decoration. Say exactly that |
| K9 | M7 | N-READOUT (null with best-of-20 readout search) equivalent to A-tied | The effect was biologically informed I/O placement, not wiring. Say exactly that |
| K10 | M8 | A-edge wins while A-tied ties the nulls, OR Spearman ρ(trained \|W\|, synapse count) < 0.3 | Downgrade to "the connectome sparsity mask and sign pattern are a useful prior," not "the connectome's weights matter" |
| K11 | M9 | Lesions indistinguishable from size-matched random lesions | No specificity claim, no circuit story in the paper |
| K12 | any | Our job is implicated in a vLLM outage, or the NVML watchdog fires twice in one week | All GPU work stops permanently; project finishes CPU-only at reduced scope. Someone else's public service outranks this experiment |

**Forbidden claims regardless of outcome:** "the fly is playing Doom"; "fly vision" (our front end is hand-specified); "whole brain" (this is a ~20k-neuron subgraph of one male fly); "the fly" in general; natural motor meaning for the strafe or attack decoders; "learns like a fly"; "first" (post-mid-2026 work is thinly indexed, FlyDoom has published nothing, priority is uncheckable). No consciousness, experience, pain or wanting language anywhere.

---

## 3. Architecture

### Connectome scope

MaleCNS v1.0, minconf-0.5 flat connectome (CC BY, doi 10.1016/j.cell.2026.08.015). **Brain only, no VNC.**

Import is a correctness test before it is a dataset: reproduce doomfly's ledger exactly — 151,856,684 raw weight rows, 33,013 unresolved objects, 11,864 non-neuronal excluded, 166,700 neurons, 25,582,938 edges, 124,177,617 contacts, edges file 1,051,241,946 bytes, SHA-256 locked. A >0.5% difference is an import bug, not a discovery.

Subgraph rule (deterministic, re-runnable, pre-registered):

- **Clamped input set I** (rates set by the front end; their incoming edges are not simulated): T4a–d, T5a–d, plus the medulla/lobula feature layer Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9, Tm20, Tm5a–c, TmY5a. EST ~32.6k cells, exact count from the feathers at M0. Seeding at T4/T5 alone starves LC4/LC10a/LC11, which are fed by Tm/Ll — that is precisely the "signal dies before the descending neurons" failure seen in Fly Chess (0 descending spikes).
- **Output set D**: `superclass == "descending"`, EST ~1,300 cells.
- **Core C** = (forward-reachable from I within 4 hops) ∩ (backward-reachable from D within 4 hops), edges with synapse count ≥ w_min (default 5).
- **Forced inclusions** regardless of threshold: LC4, LC6, LC9, LC10a, LC11, LC12, LC15, LC16, LC17, LPLC1, LPLC2, LPi\*, HS(N/E/S), VS1–10, AOTU019, AOTU025, PFL2, PFL3, EPG, PEN1/2, pC1/P1, DNa01, DNa02, DNb05, DNb06, DNg13, DNg100, DNp01/02/03/04/09/10/11/103, MDN, pIP10, DNpe050, DNp67.
- **Budget clamp** (the only self-correcting scope knob in any of the three designs, grafted from ml-first): raise w_min until core ≤20,000 dynamic neurons and ≤2.5M edges. Report achieved w_min and fraction of total synaptic weight retained (target ≥80%).

**Laterality.** R1–R6 are 2,265 right vs 1,112 left — the left optic lobe is under-proofread, and every right-minus-left readout (DNa02 above all) is biased by construction. Because the front end is hand-built and clamps T4/T5/Tm directly, we do **not** need hybrid's fabricated mirror-lobe construction (its worst honesty flaw: the steering signal computed across a symmetry the authors imposed). Instead: per-side normalization by driven-column count, and a day-1 per-type completeness audit. If left-side core cell counts are <90% of right by type, the affected types are subsampled to the smaller side, symmetrically, in every arm. The asymmetric version is a sensitivity arm.

### Neuron model

Rate, not spiking. A LIF net needs dt=0.1 ms vs ~20 ms for a rate model — ~200× the steps and ~200× the BPTT memory, which alone puts training out of reach under a 2 GB cap. Photoreceptors, lamina and VPNs are largely graded anyway; doomfly's own review names uniform LIF as failure cause #1 (all 6,865 T4 and 6,720 T5 cells produced zero spikes).

```
h_i <- h_i + (dt/tau_i) * ( -(h_i - b_i) + sum_j W_ij r_j + I_i )
r_i = min(softplus(h_i), r_max = 5)
W_ij = s_j * softplus(theta_ij)
```

- `dt = 114 ms / K`, `K = 6` substeps per decision (frameskip 4), so dt ≈ 19 ms, matched to flyvis's 20 ms step. K=6 gives 6 synaptic hops, enough for T4/T5 → lobula → VPN → central → DN. K ∈ {4, 8} as sensitivity arms. **K is identical across every arm** or the comparison is confounded.
- Signs `s_j` from presynaptic `consensus_nt`, fixed, never trained: ACh/DA/OA/5-HT → +1; GABA, Glu, histamine → −1; `unclear` → +1 default. Glu→inhibitory is a GluCl *receptor assumption*, not a measurement; the release deliberately forces DA/OA/5-HT to `unclear` unless experimentally known; synister is 87% accurate per synapse (Eckstein 2024). **Both the Glu sign and the `unclear` default are pre-registered sensitivity axes, not constants.** If the headline result flips with the Glu sign, that fact is the headline.
- **Initialization (the single most important methodological detail).** `theta_ij = softplus⁻¹(w0 · log1p(synapse_count_ij))`, inhibitory units centred per Cornford 2021. `w0` is then set **per graph by bisection so the linearised operating-point gain (power iteration on D·W at mean input drive) equals g* = 0.95 for every arm, real or null**, with the same RNG recipe and same paired seed index. Skip this and the result is worthless before it is run: Dhiman 2026 saw a flyvis topology advantage of 0.514 vs 0.698 collapse to +0.0003 once initialization was shared and the null was degree-preserving.
- **No tonic drive anywhere.** bio-first reintroduces a constant P1/arousal current into the LC10a pursuit pathway; that is doomfly root cause #2 (remove their 12 mV lamina current and outputs went silent even with the image present; keep it and it drove the behaviour). We inject none. If the network is silent at init, that is a gain-matching bug in the initializer, not a current to hand-inject. The one concession: a single trainable arousal scalar is allowed in the A-tied arm, and the **zero-arousal condition is a gate, not an appendix ablation** — performance must survive it, or the arousal-driven fraction of behaviour is reported as a headline number.
- **Spectral radius is monitored during training, not only at init** (power iteration every 500 updates), plus an activity-band regularizer keeping per-region mean rate inside a fixed band. Bounded softplus prevents divergence but substitutes *saturation*, which destroys information just as thoroughly as silence — a flaw all three source designs missed.

### Input mapping (frozen, zero trainable parameters)

This is the core of the experimental design. With a trainable CNN encoder the CNN solves Doom and the connectome is decoration (K8).

1. ViZDoom 1.3.0 renders GRAY8 + downsampled RGB24 at 160×120, off-screen, horizontal FOV widened to 108°.
2. Column geometry from `assignedOlHex1`/`assignedOlHex2` converted to azimuth/elevation at ~5.7° inter-ommatidial spacing, via the published eye map (Zhao 2025). We do not invent a pixel→cell mapping.
3. FOV split: left half of the frustum → left-eye frontal columns, right half → right-eye, 10° overlap band. EST ~250 driven columns/eye of ~800.
4. **Mean-luminance fill outside the frustum, never black** (grafted from hybrid; the sharpest unforced-error catch in any design). Zero-fill creates a permanent artificial ON/OFF edge at the frustum boundary that T4/T5 will lock onto — an agent that looks "visual" while responding to a rendering artefact. Fill mode is a pre-registered ablation.
5. **Naka-Rushton contrast adaptation against a 200 ms running per-ommatidium mean**, after a 15 ms first-order low-pass (grafted from hybrid). Doom's absolute brightness swings wildly between rooms; a fixed transduction curve clips or floors the ON/OFF channels. This is one of the mechanisms behind doomfly's "black beats live vision."
6. **Inter-frame cross-fade across the K substeps** (grafted from bio-first). ml-first holds one frameskip-4 frame constant across all 6 substeps — a 114 ms step-and-hold into a Reichardt correlator, the exact defect it criticises in doomfly (285 substeps of constant input). We linearly interpolate the retinal image between previous and current frame across the substeps.
7. Reichardt/EMD bank: delay-and-correlate neighbouring columns along four cardinal axes → four ON channels driving T4a–d, four OFF channels driving T5a–d, subtype letter matched to preferred direction (Maisak 2013). Form/sustained channels → Mi/Tm; chromatic opponency → Tm20/Tm5a–c/TmY5a. Each clamped type gets one fixed gain and one fixed low-pass tau from published values.

Phase-2 upgrade (gated, not a dependency): replace the EMD bank with the real MaleCNS optic lobe initialized from flyvis by type name with per-type current recalibration. Only if M3 and M4 both pass first and schedule allows. hybrid's own `proven_vs_hoped` rates this HIGH risk (different type vocabulary, flyvis α/β in units of *its* connectome's synapse counts) and its fallback is a full 45k-neuron BPTT training run under a 1.9 GiB cap — a ~4× larger problem than the one flyvis solved.

### Output mapping (pre-registered, hash-locked before any run, identical across every arm)

~50 cells across 17 DN types, chosen from published function, written down before Gate 0. No post-hoc selection — that is how doomfly ended up with DNp20 (a neck/gaze DN) as "turn" and calls its assignments "engineered controller assignments, not established natural motor functions."

| Action | Cells | Evidence |
|---|---|---|
| Turn | DNa02, DNa01, DNb05, DNb06, DNg13 (right−left) | Strong. Rotational velocity ∝ right−left DNa02 rate; DNa02 transient / DNa01 sustained (Rayshubskiy, eLife 2025). Steering distributed over ≥5 DN types (Yang & Wilson) |
| Forward | DNp09, DNg100 | Strong (Bidaye 2020; Braun 2024) |
| Backward | MDN (4 cells) | Strong (Sen 2017) |
| Dodge/strafe | DNp01 (giant fibre), DNp02, DNp04, DNp11, DNp103 | Escape response strongly evidenced (von Reyn 2014; Ache 2019); MaleCNS itself shows LC4→DNp04 87.0, →DNp01 46.9, →DNp02 35.1, →DNp11 29.9; LPLC2→DNp01 24.4. **Calling it "strafe" instead of "takeoff" is ARBITRARY and labelled arbitrary in the paper** |
| Attack | conjunction: `w1·mean(AOTU019_L,R) − w2·|AOTU019_L−AOTU019_R| + w3·pIP10 + b` | **There is no fly shooting neuron and we say so.** AOTU019 prefers objects near the visual midline and sits on the measured LC10a→AOTU019→DNa02 pursuit chain (Collie 2026; MaleCNS LC10a→AOTU019 54.2, AOTU019→DNa02 289 synapses). The computation is real; the trigger is engineered |

Action head: linear map from the ~100-d DN feature vector (rates + first differences) to 7 logits + 1 value, initialized from the biological prior, ~808 trainable scalars, identical count in every arm. A **zero-trainable frozen-decoder arm** is run and reported even though it will score lower, because it is the strongest form of the claim.

The claim is always "fly wiring beats matched nulls **under a fixed decoder**," never "the fly decided to shoot."

### Trainable parameters

| Arm | What is trained | Count | Claim it supports |
|---|---|---|---|
| **A-tied (HEADLINE)** | Per-connected-type-pair scale (capped at 8,000 by a pre-registered rule: type-pair for the top-N pairs covering 80% of total synaptic weight, α_pre·β_post factorization for the tail); per-type τ, V_rest, b tied at hemilineage level where a type has <5 cells (~6,000); clamped input gains/taus (40); readout (812) | **~15,000** (hard cap 20,000) | "The fly's measured wiring matters." Cannot drift from measured synapse counts |
| A-edge (secondary) | Per-edge magnitude, sign fixed, no new edges (~2.5M); per-neuron b, τ (40,000); readout (812) | **~2.54M** | Weaker: "the connectome mask + sign pattern + init is a useful prior." Requires drift instrumentation |
| A-frozen (exploratory) | Readout only, core frozen at matched-gain init | ~812 | Reservoir control on A and on N1 |

Nothing about the graph mask, the signs, or the front end is ever trainable. **Zero trainable parameters in the visual path.**

Anti-drift instrumentation, mandatory for A-edge: L2 toward the log1p(synapse-count) init, plus a reported drift statistic — Spearman ρ between trained |W| and real synapse count, and the fraction of edges whose magnitude moved >2×.

### Algorithm

PPO with truncated BPTT through the recurrent core. 64 parallel envs, frameskip 4, 160×120, T = 32 decisions per segment, gradient checkpointing at decision boundaries (the K=6 substeps recomputed in backward). GAE λ=0.95, γ=0.99, clip 0.2, 2 epochs/segment, Adam 3e-4 linear decay, entropy 0.003→0.001 **with a hard floor**, value coef 0.5, grad clip 0.5.

- **Confirmatory runs are pure PPO from scratch.** A BC warm start from a teacher washes out exactly the inductive-bias difference being measured (ml-first is right; hybrid's privileged-teacher-BC-as-main-path is close to circular, since it trains the VPN layer to encode azimuth and then offers azimuth decodability as evidence about the circuit).
- **BC from a privileged labels-buffer teacher is used as a 2-day DE-RISKING PROBE only** (grafted from hybrid, repurposed): before funding 25-seed runs, check whether the architecture can behaviour-clone a scripted oracle at all. If a sparse RNN of this size cannot fit an oracle on take_cover, PPO from scratch certainly will not work, and you learn that in 2 days instead of 14. It never enters a confirmatory arm.
- **Custom `autograd.Function` is mandatory, not a preference.** PyTorch CSR tensors get no gradient w.r.t. the matrix; only COO `sparse.mm/addmm` does. Forward: cuSPARSE CSR SpMM. Backward-to-input: SpMM against a prebuilt CSC transpose. Backward-to-parameters: fused per-edge reduction of `g[post]·x[pre]` accumulated straight into type-pair bins (A-tied) or theta (A-edge), never materializing an edges×batch tensor (the naive version costs 102 MB × B per step and caps B×T at ~10). **The whole substep loop is captured in a CUDA graph** — at K=6 with ~8 kernels each and fwd+bwd, ~150 launches × 7 µs ≈ 1 ms per batched decision, comparable to the sparse compute itself. No source design modelled launch overhead.
- **Tuning budget: 20-trial random search per arm over LR, entropy coef, and init scale, spent on the connectome arm first, reported in full.** ml-first states "identical tuning budget" as a fairness rule but never quantifies it; an untuned novel sparse architecture against a well-understood GRU is rigged in the wrong direction. Per-arm LR is allowed to differ (graphs with different spectral structure have different loss-landscape curvature, so a shared LR is not the fair choice it appears to be) and every chosen value is published.
- **Anti-degeneracy monitoring during training** (missing from all three): per-episode action histogram, action entropy, and state-conditioned action variance logged every update, with an entropy floor and a degeneracy-triggered restart. The modal ViZDoom failure — and doomfly's actual outcome — is collapse onto a near-constant sequence.
- ES/CMA-ES is **not** the main optimiser. ES used 1B Atari frames vs A3C's 320M (Salimans 2017) and CMA-ES's covariance is O(n²), practical only to ~2k parameters. It is kept only on master's CPUs for the A-frozen and frozen-decoder tracks.

---

## 4. Controls, ablations, and proving "reactive not replay"

| Tier | Control | What it rules out |
|---|---|---|
| **Confirmatory** | **N1** degree+weight-preserving rewire: directed double-edge swap preserving every neuron's in/out degree, the exact weight multiset, and presynaptic NT sign (Dale intact); clamped-input out-degrees and DN in-degrees preserved; 5 independent graphs; same paired init seed; gain-matched to g*=0.95. **Log achieved swap count, not the attempt cap** (FlyDoom's swapper gives up at 20× attempts) | Degree + weight distribution explain it. THE decisive null |
| **Confirmatory** | **N6** core bypass: same frozen front end → parameter-matched feedforward → same readout, no recurrence | Front end + decoder do the work |
| **Confirmatory** | **N7** parameter-matched GRU (h=100 for A-tied ≈15k params; h=750 for A-edge), identical front-end features and readout dimensionality, plus an over-parameterized GRU as the normalization ceiling | "Any recurrent net of this size does it" |
| **Confirmatory** | **N8** mask-only: real connectome mask, magnitudes re-randomized with sign and degree preserved | Separates sparsity pattern from measured synapse counts |
| **Confirmatory** | **N-READOUT** (new; addresses the judges' strongest missing objection): each null graph gets a **best-of-20 readout search** — 20 candidate readout cell-sets picked by the same structural criteria (in-degree from core, graph position), each trained briefly, best taken. Connectome-with-fixed-biological-readout vs null-with-best-readout. 10 seeds | The effect was the experimenter's decades of fly literature choosing good I/O cells, not the graph. Without this a reviewer will say the null was handed meaningless indices |
| Exploratory | **N2a** cell-type-block shuffle (type→type counts preserved, individual pairings randomized) | "Type-level statistics matter, individual wiring does not" |
| Exploratory | **N2b** I/O relabel, degree-matched (promoted from bio-first's exploratory tier) | Routing vs topology (the larval-connectome finding) |
| Exploratory | **N3** sign shuffle at matched E/I fraction; **N4** Erdős–Rényi (reported but labelled a straw man — FlyGM's biggest margin, 8.29 vs 125.36°, is over ER and could be pure degree); **N5** frozen-core reservoir on both A and N1; **N7b** LSTM, MLP, trainable sparse RNN with matched N/edges/sign ratio | Assorted |
| Exploratory, first to cut | **N-FLYWIRE**: a size-matched FlyWire (female brain) subgraph built by the identical rule, run through the identical pipeline, 10 seeds | Separates "THIS fly's wiring matters" from "any real biological topology beats a rewire." Without it, a win supports the weaker statement and the title overclaims |
| Floors (no training) | Random policy; constant-action; spin+fire; constant-forward; open-loop replay of own best training sequence; pixel-centroid heuristic; linear-on-raw-pixels (ARS); **linear-on-frozen-front-end-features** | The last one is the true "the front end already solved it" test. If it scores near A-tied, the recurrent core — real or rewired — is decoration and no topology comparison is worth running |
| Sensitivity | Glu sign ±; `unclear` NT default ±; synapse confidence 0.5 / higher; w_min ∈ {3,5,10}; K ∈ {4,6,8}; frustum fill mean vs black; zero-arousal | Hidden hyperparameters |

### Proving reactive, not replay

All five must pass; R1–R3 are **hard gates at the M6 pilot, before the confirmatory arms are funded** (bio-first defers these to M7, after its 21-day confirmatory block — a number would exist for weeks before anyone knew the agent was reactive).

- **R1 blindfold** — black, mean-luminance, time-scrambled frames must drop to the blind floor. Doomfly: black input scored 3/3/2 kills vs live vision's 1/1/1.
- **R2 yoked video** — frames from a different episode at the same tic index must collapse performance.
- **R3 mirror** — left-right flipped frames must flip the sign of the turn-vs-enemy-azimuth correlation.
- **R4 stimulus-locked** — inject an enemy/fireball at a random azimuth via the labels buffer; turn-toward or dodge within 3 tics, reported as a **per-trial latency distribution**, not a population mean. Population aggregates are what let Closed-Loop Fly's silent-DNa02 artefact pass as a result.
- **R5 closed-loop beats open-loop replay** of its own best action sequence on held-out seeds.
- **Test-time wiring swap** — load an N1 graph into a trained connectome policy; performance must collapse.
- **Seed hygiene** (missing from all three): verify that held-out seeds actually vary spawn positions and enable sticky actions / action-repeat stochasticity. Without this, a memorized open-loop sequence can pass the held-out-seed test — the exact failure R5 exists to catch.
- **DN bottleneck re-probe after training** (missing from all three): re-run the Gate-0 azimuth probe on the trained model. If the ~100-d DN readout saturates, the ceiling is set by an information bottleneck, not by wiring, and every arm comparison is measuring bottleneck width.

**Statistics:** seed as unit, 100 held-out episodes per seed, 25 confirmatory / 10 exploratory seeds; rliable IQM with stratified bootstrap CIs and probability of improvement; Welch + Holm across all arm × scenario comparisons; TOST for equivalence; AdaStop sequential testing to stop dead arms early. **Blinded analysis**: arm labels scrambled under a key until the frozen analysis script is run.

---

## 5. Compute plan

### VRAM budget per training job (worst case = A-edge, 2.5M edges, B=64, T=32)

| Item | Size | Basis |
|---|---|---|
| CUDA context (outside the allocator) | 0.40–0.60 GB | Measured at M1; largest and least certain term. No cuDNN (zero conv layers anywhere), no NCCL, no DDP |
| CSR + CSC + edge permutation index | 50 MB | 2.5M × (4 B val + 4 B idx) × 2 + 10 MB perm |
| theta + grad + Adam m,v | 40 MB | 2.5M × 4 tensors × 4 B (A-tied: 0.3 MB) |
| Per-neuron b, τ + grads + Adam | 1.3 MB | 40k × 2 × 4 × 4 B |
| Decision-boundary state checkpoints | 82 MB | 20k dynamic × 64 × 32 × fp16 |
| Stored frames for front-end recompute | 39 MB | 160×120 uint8 × 64 × 32 |
| Backward working set (K=6 substep recompute, fp16) | 150 MB | 6 substeps × ~3 tensors × 52.6k × 64 × 2 B |
| cuSPARSE workspace + fragmentation | 150 MB | |
| **Typical total** | **≈1.06 GB** | |
| **Planning figure** | **1.2 GB** | |
| **Watchdog hard kill** | **1.6 GB** | |

Note: the input side of the SpMM is ~52.6k rows (20k dynamic + 32.6k clamped), not 20k — ml-first's ledger dimensioned the state at 20k throughout, which is small in absolute terms but shows the ledger was not walked end to end.

### GPU safety guard (self-imposed; no root, no MIG)

- `torch.cuda.mem_get_info` at startup; **refuse to launch if <2.4 GiB free** on the target card.
- `CUDA_VISIBLE_DEVICES` pins one card. One process, one CUDA context. No cuDNN, no NCCL, no DDP.
- Measure context after init, then `set_per_process_memory_fraction = (1.6 GiB − context)/24` ≈ 0.044. This caps only the caching allocator, not the context or library handles.
- `PYTORCH_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6`.
- **Independent NVML watchdog (nvidia-ml-py, no root) at 1 Hz: SIGKILL our trainer if our usage exceeds 1.6 GiB OR card-free memory drops below 400 MB.**
- **vLLM restart guard.** vLLM checks free memory *at startup* and fails with "Free memory on device … less than desired GPU memory utilization" (vllm-project/vllm#20305). If we hold 1.2 GB when their server restarts we could keep a public service DOWN without ever OOMing it. The watchdog polls for the vLLM PIDs; if they vanish we checkpoint and exit within 5 seconds to release the context, then wait 10 minutes before requeueing. Release the GPU whenever idle >5 min.
- Checkpoint every 200 updates to **node1-local `/tmp`**, rsynced to NFS hourly. On a shared card the watchdog and the PID guard will fire repeatedly; without frequent checkpoints each event costs a whole run.

### Core budget (node1: 56 cores — must sum to 56)

| Allocation | Cores | Notes |
|---|---|---|
| ViZDoom env workers | 40 | 64 envs, shared-memory ring buffers into a single learner process. WADs + iwad copied to node1-local `/tmp`, never read from NFS by 64 workers |
| Learner processes (2, one per card) | 6 | One persistent trainer **daemon per card** holding ONE CUDA context and consuming a run queue — with ~300 short runs, paying 0.3–0.6 GB of context and the CSR upload once per card instead of 300 times is free throughput |
| NVML watchdog + vLLM PID guard + logging | 2 | |
| **Reserved for the root-owned vLLM host side and the OS** | **8** | We never touch their processes |

`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` in every worker (doomfly measured their retina transform going 38 ms → 0.5 ms with this one change; numpy+ViZDoom workers oversubscribe badly otherwise). `taskset` pinning per worker.

We do **not** use Sample Factory's multi-process architecture: its inference workers each add a ~0.3–0.6 GB CUDA context, which alone breaks our cap. Hand-written single-process PPO loop. Sample Factory is used only as a published throughput reference, which also sidesteps its numpy<2 / gymnasium<1.0 pin conflict with vizdoom 1.3.0.

### master (24 cores, 125 GB RAM, no GPU, internet)

- Data ingest: download the flat-connectome feathers (~1.95 GB), SHA-256 verify, build CSR/CSC, reproduce the import ledger.
- Null generation: 5 degree+weight-preserving rewires over ~2.5M edges (~50M swap attempts each), numba/C++ swapper, parallel over 20 cores. **Measure and report the swap rate and peak RSS** — all three source designs asserted this cost without deriving it.
- **All evaluation, the entire reactivity battery, all lesion runs, all ablation and sensitivity forward sweeps.** These are inference-only and constitute ~60% of total runs. Structural win: the contended GPU is touched only by training, so the card is free whenever we score.
- The CMA-ES track for A-frozen and frozen-decoder (≤2k params, population 64), GPU-free.
- rliable/AdaStop statistics, figure generation.

Caveat: master is a login node. Whether heavy jobs are tolerated there is **unverified** and there is no sudo and no admin to ask. Day-1 check; if not, evaluation moves to node1's env-worker cores between training runs and the schedule stretches.

### Throughput (EST — must be replaced by measurement at M0/M1)

Per batched decision at B=64, A-edge: 2.5M edges × 6 substeps ≈ 960 MFMA forward, ~2.9 G with backward; memory traffic ≈ 456 MB forward, ~1.4 GB with backward. Against the A30's 933 GB/s HBM2 ceiling at a realistic 25% efficiency: ~6 ms per batched decision → ~160 batched decisions/s → ~10,000 env-decisions/s at the microbenchmark level. End-to-end with PPO overhead, Python, and vLLM contention, assume **10–30× worse: 200–600 env-decisions/s**, i.e. **~35–100 min per 5M-frame seed**.

**Throughput definition (fixing ml-first's 64× internal contradiction): all gates are stated in *env-decisions/s* = batched-iterations/s × B.** M1's floor is 6,400 env-decisions/s (100 batched at B=64); below 1,280 env-decisions/s we re-scope.

Confirmatory budget: 5 arms × 25 seeds × 5M frames (take_cover) ≈ 125 runs ≈ 200 GPU-h; replication on defend_the_line at 8M ≈ 300 GPU-h; exploratory tier ≈ 150 GPU-h. ~650 GPU-h over two cards at realistic 50% availability ≈ **27 days**, which is why AdaStop is in the plan and why the exploratory tier runs at 10 seeds.

---

## 6. Milestones

**Training-approval milestones are marked ⚠️ and require the user's explicit "start" before launch.** Everything up to M5 is setup, data prep, kernels, forward-only physiology, probe fitting, and pre-registration.

| M | Deliverable | Days | Go/no-go |
|---|---|---|---|
| **M0 — Setup and data prep only. NO TRAINING.** | Python 3.11 via uv in `$HOME`; **ViZDoom 1.3.0 headless verified on this box with no X server** (named no-sudo fallback: software GL, or a user-space Xvfb build) — this is assumed by all three source designs and verified by none; feather files downloaded to NFS, SHA-256 locked; CSR/CSC built; import ledger reconciled against doomfly's counts; **day-1 shared micro-benchmark sweeping measured A30 and CPU SpMM throughput and peak VRAM over (nnz, batch, dtype) on the real matrix**; scope selected by the budget clamp *from that measurement*; all null graphs generated with achieved swap counts logged; per-type left/right completeness audit; ViZDoom per-process RSS and 64-env memory footprint measured | 5 | GO if the ledger matches to the row; ViZDoom runs headless; the clamp yields ≤20k dynamic neurons / ≤2.5M edges at w_min ≤10 retaining ≥80% of synaptic weight; ≥95% of the pre-registered DN set reachable from the clamped inputs within 4 hops. **The benchmark sets the scope numerically — scope is not chosen first and benchmarked after** |
| **M1 — Kernel** | Custom `autograd.Function` (cuSPARSE CSR forward, CSC-transpose backward, fused per-edge/type-bin reduction), CUDA-graph capture of the substep loop, CPU reference path, gradcheck vs dense, NVML watchdog, vLLM PID guard, persistent per-card trainer daemon, measured throughput/VRAM table at B ∈ {1,16,64} | **8** (not 2 — ml-first budgets 2 days for a bespoke sparse autograd kernel; the dangerous failure is a silently wrong per-edge gradient that still produces plausible curves) | GO if gradcheck passes to 1e-4 vs dense, CPU/GPU forward agree to 1e-5, peak VRAM ≤1.3 GB, ≥6,400 env-decisions/s at B=64 with backward on a shared A30. Below 1,280 → shrink core to ≤1M edges and re-benchmark |
| **M2 ⚠️ — Harness smoke test (TRAINING: a standard CNN+GRU baseline, not the fly)** | A plain CNN+GRU PPO agent trained to published competence on take_cover through **the exact env wrapper, reward path, action set, frameskip, termination, eval protocol and held-out seed split** we will use. Sticky-action / spawn-randomization verification | 4 | GO if it reaches published take_cover competence. **This is the difference between "the connectome cannot play" and "our harness is broken."** No source design does this; all three would misread a broken wrapper as a scientific null |
| **M3 — Frozen front end (no training)** | EMD bank + Naka-Rushton adaptation + cross-fade + mean-luminance fill; hex-column geometry; FOV split; 64-env worker pool; all scripted floors measured including linear-on-front-end-features; full stimulus battery (gratings at 8 directions × 6 temporal frequencies, expanding/receding/translating discs) | 5 | GO if T4/T5 DSI ≥0.5 with correct ON/OFF polarity and correct cardinal preferred directions; LPLC2 monotone in expansion rate; ≥1,000 env-FPS with 40 workers; at least one primary scenario has a blind floor clearly below the GRU ceiling (headroom for vision to matter) |
| **M4 — Gate 0 (probe fitting, not agent training)** | Cores frozen at matched-gain init; 200k recorded gameplay frames; ridge probes from the DN/VPN population to nearest-enemy azimuth, log time-to-collision, enemy presence, for A vs N1×5 vs N2a vs front-end-only; frustum-fill ablation | 4 | **The cheapest kill gate in the plan (~1 GPU-day).** GO if held-out azimuth R² ≥0.50 AND ≥ +0.10 over the mean of 5 rewires (p<0.05). Fail → cut the confirmatory tier to one scenario and one null before spending three weeks |
| **M5 — Pre-registration freeze (no compute)** | OSF registration with third-party timestamp; frozen analysis script committed; blinding key generated; all thresholds, arms, seed counts, budgets, decoder body IDs, exclusion and outlier rules locked | 3 | GO only if every claim in the planned abstract maps to a gate in the document |
| **M6 ⚠️ — Capacity ladder + RL pilot (TRAINING)** | 2-day BC capability probe (can the architecture fit a labels-buffer oracle at all?); then PPO pilot at 2M frames: A-tied, A-edge, N1, GRU, 8 seeds each on take_cover. Full reactivity battery R1–R5 on pilot models. First drift statistic. Measured seed SD for the power calculation | 12 | **HARD GATE.** GO only if (a) A-tied or A-edge beats the best blind/scripted floor by ≥25% at p<0.01, AND (b) R1 passes: live beats black by ≥20% at p<0.01, AND (c) no action-entropy collapse in ≥6 of 8 seeds. R1 failure = doomfly's central failure reproduced → pivot to signal-flow diagnosis, do not spend 14 days on 25-seed arms. The ladder decides which arm is headline: if only A-edge plays, the claim is downgraded per K10 **before** the confirmatory runs |
| **M7 ⚠️ — Confirmatory primary (TRAINING)** | A-tied, N1×5, N6, N7-GRU, N8, N-READOUT × 25 seeds × 5M frames on take_cover, AdaStop sequential stopping; all evaluation on master's CPUs | 14 | GO to M8 regardless of sign — a clean null is the publishable outcome. Stop only if measured seed SD gives <60% power for the observed effect, in which case report the CI and declare it underpowered rather than adding seeds post hoc |
| **M8 ⚠️ — Replication + exploratory tier (TRAINING)** | Same arms × 25 seeds × 8M frames on defend_the_line; exploratory arms (N2a, N2b, N3, N4, N5, N7b, A-edge, A-frozen, frozen-decoder, N-FLYWIRE) × 10 seeds on take_cover | 12 | Unconditional. Report whichever kill criteria fired |
| **M9 — Mechanism and generalization (inference only, CPU)** | R1–R5 with per-trial latency distributions on all confirmatory seeds; DN bottleneck re-probe post-training; I1 lesions vs 20 size-matched random lesions with dose-response and the pre-registered double dissociation; test-time wiring swap; held-out textures and layout; all sensitivity arms (Glu sign, unclear default, w_min, K, fill mode, zero-arousal) | 7 | Mechanism claims allowed only if the double dissociation holds and each lesion exceeds the 95th percentile of its random-lesion distribution |
| **M10 — Write-up and release** | Unblind, run the frozen analysis script, LaTeX manuscript + .bib; pre-registration diff; every negative run and per-seed raw score released; full run ledger; licences (MaleCNS CC BY, ViZDoom MIT, Freedoom BSD-3, flyvis MIT, doomfly MIT for reused import/audit code) | 7 | Every abstract claim maps to a passed gate. Claims that cannot are deleted, not softened |

**Nominal total: 81 days. Realistic with a 1.6× multiplier on M1–M3 (novel CUDA kernel, novel front end, shared cluster, no admin rights): ~100 days.** All three source designs budget zero debugging tail. When the schedule slips, the pre-committed cut order is: (1) N-FLYWIRE, (2) the exploratory tier drops to 5 seeds, (3) defend_the_line replication drops to 15 seeds, (4) M8 replication cut entirely and the claim scoped to one scenario. **This order is written into the pre-registration at M5, before any result is visible.**

Phase-2 (real optic lobe via flyvis transfer) is explicitly **out of the 81 days** and is only attempted if M3 and M4 both pass comfortably and the confirmatory tier finishes early.

---

## 7. Caveats and risks (worst first)

- The most likely scientific outcome is a null: the connectome ties N1. The best-controlled published test saw a fly-connectome advantage collapse from 0.514-vs-0.698 to +0.0003 under shared init and degree-preserving rewiring (arXiv 2604.04033); brain-based reservoirs tie classical ESNs. I would put P(S2 passes) below 30%.
- A silently wrong per-edge or per-type-bin gradient in the custom sparse kernel produces plausible learning curves and invalidates everything downstream; M1's gradcheck is the only thing standing between us and weeks of fiction.
- Every throughput and VRAM number is bandwidth arithmetic; no A30 benchmark exists anywhere for a connectome-shaped sparse matmul, and the card's compute is shared with an unpredictable root-owned vLLM server whose bursts will swing our wall-clock 2–10×.
- Holding GPU memory when vLLM restarts could keep a stranger's public service down without ever OOMing it, and we have no admin access to coordinate — this is the one failure here that hurts someone other than us.
- Vision may never reach the descending neurons: doomfly (black beat live), fly-craftax ("vision contributes nothing"), Closed-Loop Fly (readout artefact) and Fly Chess (0 descending spikes) all failed at exactly this step, and our Gate 0 threshold (R² ≥0.50) is a genuine coin-flip.
- The hand-built EMD front end means we do **not** get to claim "fly vision"; even a clean topology win supports a narrower claim than the project goal states.
- Bounded softplus prevents runaway but substitutes saturation, which destroys information just as thoroughly — spectral-radius monitoring during training is a mitigation, not a proof.
- ViZDoom headless on Ubuntu 20.04 with no X server and no sudo is unverified; if the fallback also fails, the project stops at M0.
- master is a login node and whether heavy jobs are tolerated there is unverified; ~60% of our runs are scheduled onto it.
- The attack decoder is engineered (there is no fly shooting neuron), so defend_the_line rests on gradient descent shaping upstream wiring until a 2-cell midline-preference readout becomes a usable trigger — weaker than take_cover.
- Glutamate-as-inhibitory is a receptor assumption, not a measurement, and thousands of cells are `unclear` because DA/OA/5-HT are force-set that way; FlyGM and Shiu disagree on this sign, and the headline could flip with it.
- Doom's 108° frustum fills a minority of a ~270° eye with ~5° acuity and no fovea; nobody has tested whether that delivers enough drive.
- 20k type-tied parameters may be too few to clear the blind floor at all, in which case every arm sits at the floor and "A ties N1" carries zero information about wiring — a degenerate comparison, not an equivalence result.
- T=32 decisions of truncated BPTT may be insufficient credit assignment for take_cover's ~1–2 s fireball anticipation.
- Subgraph counts (~32.6k clamped, ~20k core, ~2.5M edges, connected type-pair count) are all estimates until the feathers are read; the budget clamp makes the plan survive being wrong, the numbers themselves do not.
- Nobody can verify "first" — post-mid-2026 work is thinly indexed and FlyDoom has published nothing. We make no priority claim.

---

## 8. Proven vs hoped

### Proven (read at the cited source; we re-ran none of it)

- MaleCNS v1.0 scale and schema: 166,700 neurons / 25,582,938 edges / 124,177,617 contacts; 151,856,684 raw rows, 33,013 unresolved, 11,864 non-neuronal excluded; edges file 1,051,241,946 bytes (github.com/nftechie/doomfly, `doom/connectome.py` + source.lock.json). Annotations camelCase (`bodyId`, `type`, `superclass`, `somaSide`, `assignedOlHex1/2`), weights snake_case (`body_pre`, `body_post`, `weight`), NT file `consensus_nt` (github.com/janelia-flyem/flyem-snapshot).
- NT prediction accuracy 87%/synapse, 94%/neuron, 91%/known cell type (Eckstein 2024, Cell). The pipeline force-sets DA/OA/5-HT to `unclear` unless experimentally known.
- The escape and pursuit pathways exist inside MaleCNS itself: LC4→DNp04 87.0, →DNp01 46.9, →DNp02 35.1, →DNp11 29.9; LPLC2→DNp01 24.4; LC10a→AOTU019 54.2; AOTU019→DNa02 289 synapses; PFL3→DNa02 380 synapses.
- DN function: DNa02 right−left ∝ rotational velocity, DNa02 transient / DNa01 sustained (eLife 102230); steering distributed over ≥5 DN types (PMC10614758); DNp09 forward (Bidaye 2020; Nature s41586-024-07523-9); MDN backward (Sen 2017); DNp01 giant-fibre escape (nn.3741); T4/T5 are the ON/OFF EMDs tuned to four cardinal directions (Nature 12320); AOTU019 midline preference in the pursuit chain (Collie 2026).
- flyvis: 45,669 neurons, 65 types, 1,513,231 synapses, **734 trained parameters**, and task performance correlates with realistic DSI at r=0.60 (Nature s41586-024-07939-3). This is the precedent for the type-tied discipline.
- Untrained fly wiring passes sensorimotor signal: 91% of 164 predictions matched with one free parameter (Shiu 2024, Nature s41586-024-07763-9).
- A connectome alone is often insufficient to constrain dynamics; unmeasured single-neuron parameters must be trainable (Beiran & Litwin-Kumar, bioRxiv 2024.02.22.581667).
- The decisive negative: topology advantage collapsing to +0.0003 under shared init + degree-preserving rewiring (arXiv 2604.04033). Brain-based reservoirs tie ESNs (PLoS CB 10.1371/journal.pcbi.1010639).
- Every doomfly failure number quoted here is from its own repo docs: black 3/3/2 vs live 1/1/1 kills; retina cut identical to black screen; 0 spikes in 6,865 T4 / 6,720 T5 / 4,064 KC; 9,468 of 166,700 neurons active on a 500 ms frame; membrane voltages to −224 mV; identical 93.79% suppression in paired and no-US conditions; held-out survival 3.657 s for both plastic and shuffled (exactly how long it takes to die standing still); attack pressed on 93% of tics.
- ViZDoom take_cover is solvable by a **1,088-parameter** controller at 1092±556 (worldmodels.github.io). Sample Factory gives throughput (322,907 sim FPS on 36 cores at 128×72, frameskip 4) and per-scenario curves (arXiv 2006.11751). vizdoom 1.3.0 ships a cp311 manylinux_2_28 wheel installable on glibc 2.31.
- PyTorch CSR tensors get **no** gradient w.r.t. the matrix; only COO `sparse.mm/addmm` does (PyTorch 2.14 sparse docs). `set_per_process_memory_fraction` caps only the caching allocator, not the CUDA context.
- vLLM fails at startup if free memory is below its target (vllm-project/vllm#20305).
- A30 HBM2 bandwidth 933 GB/s (NVIDIA A30 datasheet).
- No prior project has shown live pixels driving play that beats a blind-input control. FlyGM's degree-rewire win (8.29° vs 13.55°) is the only fly-scale topology positive and is self-reported, with unweighted nulls against a weighted real graph, a 32-d trainable per-neuron descriptor, and an unmatched MLP.

### Hoped / untested (these are the bets)

1. **That the connectome beats N1 at all.** The best-controlled evidence says probably not. Below 30% in my estimate. This is the central untested hypothesis and the whole plan is built so the null is a clean, pre-registered, publishable result.
2. **That a ~20k-neuron / ~2.5M-edge subgraph with ~15k type-tied parameters plays competently.** World Models at 1,088 parameters is encouraging, but that controller sat on a *trained* VAE+RNN world model, not a frozen connectome. No connectome-based agent has ever been shown to play any pixel-based game competently.
3. **That the hand-built EMD front end delivers usable signal through real lobula wiring to the DNs.** Plausible from first principles, untested at this scale, and three separate teams failed at the analogous step.
4. **Every throughput and VRAM number.** Bandwidth arithmetic on a contended card. The CUDA-context term (0.40–0.60 GB) is the largest and least certain component of the memory budget. M0/M1 replace these with measurements or the plan re-scopes.
5. **All subgraph counts** (~32.6k clamped, ~20k core, ~2.5M edges, connected type-pair count, ~1,300 DNs, ~50 readout cells). Derived from per-column arithmetic plus doomfly's published T4/T5 counts.
6. **That the type-pair capping rule (top-N pairs covering 80% of weight, α·β for the tail) is expressive enough.** Invented here, untested.
7. **That gain matching by bisection to g\*=0.95 is sufficient to neutralize the initialization confound.** It is the right idea and stronger than anything in the literature we found, but it matches a linearised operating point, not the full nonlinear dynamics.
8. **That K=6 substeps and T=32 decisions give enough credit assignment depth for take_cover.**
9. **That the best-of-20 readout search is a fair null.** It is a stronger test than anyone has run, but 20 structural candidates may still not be equivalent to decades of fly literature choosing DNa02.
10. **That ViZDoom runs headless here, that master tolerates heavy jobs, and that 64 env processes fit node1's memory and scheduler.** All day-1 checks, none verified.
11. **That 25 seeds gives ~80% power** — a calculation at d=1.0 with the true effect size unknown; M6's pilot replaces the assumed SD with a measured one.
12. **The day estimates.** One operator, no milestone overlap stated, a 1.6× debug multiplier applied to engineering milestones only. 81 days is the floor, ~100 the expectation, and that assumes nothing on the cluster changes underneath us.

---

## 9. Open questions for you

1. **Is the honesty-over-spectacle trade acceptable?** This plan makes the ~15k-parameter type-tied arm the headline, which maximizes the defensibility of "fly wiring matters" but materially lowers the chance of a visually impressive agent. The per-edge 2.5M-parameter arm is much more likely to play well and much less defensible. If you want the demo first and the science second, we invert the ladder — say so now, because it changes the pre-registration.

2. **How hard is the ~100-day realistic estimate?** If the real budget is closer to 6 weeks, the honest cut is: drop the defend_the_line replication and the whole exploratory tier, run one scenario with A-tied vs N1 vs N6 vs GRU at 20 seeds, and scope the claim accordingly. That is still stronger than anything published in this space, but it loses the double dissociation and the replication.

3. **Do we have any channel to the cluster admin at all?** Everything in the GPU safety section is a self-imposed soft cap because we cannot use MIG, cannot set a real memory limit, and cannot coordinate around a vLLM restart. Even a one-line heads-up to whoever owns that service would materially reduce the worst non-scientific risk in the plan.