# Results: a connectome-constrained agent that plays ViZDoom (2026-09-14/15)

**Status: E1M1 (the real first level of Doom) delivered: 57 % of runs reach the exit, reactive, not a
memorised route (§6). take_cover delivered earlier (§1–3, §5); defend_the_line not solved (§4).**

All numbers measured on the Intel Unnati cluster (node1: 2× A30). Scenario take_cover unless stated.
Scores are episode reward (= survival tics + small bonuses); 50 held-out episodes (seeds ≥ 10,000),
greedy actions. IQM = interquartile mean.

## 1. Floors, teacher, and the connectome agent

| agent | trained parameters | reward mean | IQM [95 % CI] |
|---|---|---|---|
| stand still (any no-op button) | 0 | 154 | 145 [135, 156] |
| constant strafe left / right | 0 | 183 / 171 | 176 / 162 |
| random | 0 | 255 | 229 [205, 257] |
| labels-buffer dodging script | 0 | 273 | 267 [250, 285] |
| CNN+GRU PPO teacher (`m2_tc_rs01`, 2 M decisions, reward scale 0.01) | ~0.4 M | 866 | 784 [673, 920] |
| **connectome agent v1**: frozen MaleCNS subgraph v5 (108,062 dynamic + 30,906 clamped neurons, 4,639,332 synapses, gain-matched at rest, w0 = 0.00828) + linear readout of its 1,312 descending neurons (rates + first differences, standardised) | **21,000** (readout only) | **565** | **509 [438, 603]** |
| connectome agent v1, **blindfolded** (black frames) | - | 179 | 168 [156, 183] |
| connectome agent v1, flat-grey frames | - | 186 | 174 [159, 192] |

The agent's score collapses to the stand-still floor without vision: the behaviour is driven by the
frames through the frozen wiring, not by internal dynamics. Three-episode video (game view + the
descending-neuron activity raster): `connectome_take_cover_frozen_v1.mp4` (episode rewards 604 mean;
kept on the server under `outputs/videos/`, not in the repo).

Training of the readout: behaviour cloning (KL to the teacher's action distribution) on 60,000
teacher decisions (340 episodes, 10 % random actions for coverage), 3 epochs, windows of 32
decisions, Adam 1e-2; held-out teacher-agreement 0.767 (majority-action baseline 0.547).

## 2. Why the first cloning attempt failed (and what fixed it)

Attempt 1 trained the 4.86 M per-edge magnitudes plus the readout jointly (Adam 3e-4, raw rates
into the readout). Result: KL fell 1.06 → 0.32 under teacher forcing, yet closed-loop the agent
pressed strafe-right 100 % of the time in every condition (171 = the strafe-right floor, identical
with black, grey and scrambled frames; a DAgger round changed nothing). Diagnosis
(`flybrain/eval/action_probe.py`): from the *frozen* brain a softmax readout already predicts the
teacher's action with held-out agreement 0.829 from the descending neurons (0.787 from the 36
pre-registered cells, 0.844 from the visual projection neurons, 0.925 from the eye alone; majority
0.547). The information was present; the optimisation failed (un-normalised inputs, readout LR too
small, the wiring free to fit the teacher's timing habits). Fix: standardise the readout inputs,
train the readout alone at LR 1e-2, leave the wiring frozen.

## 3. What this does and does not show

- Shows: the real MaleCNS wiring, unmodified, transmits enough of what the eye model computes to
  its descending neurons for a linear readout to dodge fireballs; the agent is reactive (R1 test).
- Does not show: that the fly wiring is better than random wiring. Gate 0 (`M0_report.md` §11)
  found degree- and weight-preserving rewires carry *more* target information to the descending
  neurons (R² 0.62 vs 0.46). A rewired control would very likely reach a similar score.
- Not claimed: "the fly plays Doom", "fly vision" (the eye is a hand-built EMD front end), natural
  motor meaning for the strafe/attack mapping, or anything about learning in the fly.

## 4. defend_the_line (in progress)

Teachers: PPO attempt 1 collapsed onto one action (18.3 kills); attempt 2 with a stronger entropy
bonus reached **19.5 kills** without collapsing (random 8.4, hold-fire 11.0, labels-aiming script
20.7). Frozen-wiring + linear readout cloned from teacher 2 (held-out agreement 0.727): **16.3
kills normal, 16.3 blindfolded**; it presses attack 95–99 % of the time; a copy of the teacher's
dominant habit, not a reactive policy. Cloning the labels-aiming script instead (60 k decisions:
69 % fire, 31 % turns; frozen wiring + linear readout, held-out agreement 0.667): **12.7 kills with
vision vs 10.4 blindfolded** (IQM 11.0 [10.0, 12.1] vs 9.0 [8.4, 9.8]), a real but small
reactivity gap; it turns 8 % of the time vs the script's 31 %. Enemy azimuth is less decodable
from the descending neurons on this scenario (Gate 0 R² 0.41 vs 0.46). The nonlinear-readout
variant collapsed onto "fire" (99 %): 16.2 kills with vision vs 16.5 blind, not reactive. A
class-balanced cloning run (turn decisions up-weighted 1/frequency) over-corrected: with vision it
turns 68 % of the time (mostly right) and scores 8.3 kills, blindfolded it fires and scores 15.9:
strongly input-dependent, but not a competent policy. **defend_the_line stays unsolved with the
frozen-wiring recipe**: enemy azimuth is only weakly linearly available in the descending neurons
(Gate 0 R² 0.41), and every readout either ignores it (fire-only) or over-reacts to it.

## 5. take_cover: pushing past the linear-readout plateau

DAgger round (`bc3b_frozen_lin_d1`: 40 k student-visited states labelled by the teacher, readout
retrained; held-out agreement 0.784): **505 mean / 465 IQM [380, 569]**, blindfolded 179, no
better than v1 (565 / 509 [438, 603]); the linear readout of the frozen wiring plateaus at
~500–560. Two variants: (a) unfreezing the 4.86 M synapse magnitudes at LR 1e-4 with a drift
penalty, warm-started from v1, destabilised within two epochs (gradient norms 470 → 970, KL rising
0.39 → 0.92, agreement falling) and was stopped; (b) **frozen wiring with a 256-unit nonlinear
readout** (`bc4b_frozen_mlp`, 4 epochs on teacher + DAgger data, held-out agreement 0.793):
**633 mean / 555 IQM [475, 632]**, blindfolded 171, the best agent so far, wiring still untouched.
Demo video (5 episodes, mean 656): `connectome_take_cover_frozen_mlp_v2.mp4` (server `outputs/videos/`).
A second DAgger round on this agent (`bc4c_frozen_mlp_d2`, 40 k more student-visited states,
held-out agreement 0.792) scored **488 / 435 IQM [364, 521]**, blindfolded 171, worse; DAgger does
not help here (the teacher's advice on the weaker student's trajectories is inconsistent with its
own play). Final take_cover agent: `bc4b_frozen_mlp` (633 / 555).

## 6. A real Doom level: E1M1 "Hangar" (2026-09-15)

Level: shareware DOOM1.WAD v1.9 (id Software's free release, md5 f0cefca49926d00903cf57551d901abe), map E1M1,
skill 1 ("I'm too young to die"). Untouched level geometry, monsters, doors, nukage and exit switch. The agent
renders at 640×480 without the status bar, field of view 108°, and decides every 4 game tics (≈9 decisions/s)
among 8 buttons: forward, back, turn left/right, strafe left/right, shoot, use. It sees a 160×120 copy of the
frame (grey + colour). Runs end at the exit switch, on death, or after 3 minutes of game time. Every run starts
from the level's own player start after 0–6 random turn/strafe decisions (0–20 incl. forward/back moves in
the "wide" condition), so a fixed button sequence cannot succeed.

**Agent** (`outputs/e1m1/v2_r3/student.pt`, temperature 1.25): the same frozen MaleCNS subgraph v5
(138,968 simulated neurons, 4.64 M synapses, wiring never trained; one global synaptic scale set once by
gain matching) behind the same hand-built eye model, read out by a 256-unit two-layer head on the rates and
rate changes of 10,511 neurons (1,312 descending + 9,199 visual projection neurons; 5.4 M trained parameters,
all in the head). Actions are *sampled* from the head's distribution at temperature 1.25.

Training: imitation of a privileged scripted navigator (map geometry + player position; 98 % exits):
200 demonstration runs (144,808 decisions, 5 % random-movement bursts for coverage), then 3 DAgger rounds
(the agent drives, the navigator labels; 48 runs each). Labels the navigator produced from hidden state
(random un-stick wiggles, off-grid positions) are masked from the loss. Round 3 was chosen on 32 validation
runs (seeds 10,000+); the table below is a separate, untouched block.

### 6.1 Final test: 100 runs each, seeds 20,000–20,099, never used before

| player | reached the exit [95 % CI] | mean best route progress [95 % CI] | median | died | timed out | kills |
|---|---|---|---|---|---|---|
| random buttons | 0 % | 13 % [12, 14] | 13 % | 0 | 100 | 0.0 |
| always forward | 0 % | 14 % [13, 15] | 13 % | 0 | 100 | 0.0 |
| open-loop replay of a recorded navigator run (blind) | 6 % [2, 11] | 48 % [42, 54] | 39 % | 49 | 45 | 0.4 |
| **connectome agent** | **57 % [47, 66]** | **92 % [88, 95]** | **100 %** | 37 | 6 | 1.6 |
| connectome agent, wide starts + 10 % sticky actions | 64 % [55, 74] | 92 % [88, 95] | 100 % | 28 | 8 | 1.7 |
| connectome agent, **its own frames in random order** | 0 % | 22 % [20, 24] | 19 % | 9 | 91 | 0.1 |
| connectome agent, black frames | 0 % | 8 % [7, 9] | 7 % | 0 | 100 | 0.0 |
| connectome agent, flat-grey frames | 0 % | 7 % [7, 8] | 6 % | 0 | 100 | 0.0 |
| connectome agent, greedy (argmax) actions | 0 % | 10 % [9, 10] | 9 % | 0 | 100 | 0.0 |
| navigator (privileged ceiling) | 98 % [95, 100] | 99 % [98, 100] | 100 % | 1 | 1 | 3.9 |
| navigator, wide starts + sticky | 95 % [90, 99] | 98 % [96, 100] | 100 % | 4 | 1 | - |
| replay, wide starts + sticky | 0 % | 35 % [31, 40] | 27 % | 42 | 58 | - |

Reading: the agent finishes the real level in 57 % of runs and on average gets 92 % of the way; the median
run reaches the exit. Widening the starts and forcing 10 % repeated actions does not hurt (64 %, within
noise), while the same perturbation drops the blind replay of a recorded run from 6 % to 0 % exits; the
route is not memorised. Feeding the agent its *own* frames in random order (same image statistics, wrong
content) drops it to 0 % exits and 22 % progress; black or flat-grey frames leave it below the always-forward
floor: the behaviour is driven by what it sees. Greedy (argmax) actions fail completely (0 %, 10 %): the
agent walks into a wall and keeps pressing forward; sampling is what lets it recover. Its failures are
mostly deaths (37 of 43): it navigates well and fights badly (1.6 kills per run vs the navigator's 3.9).

### 6.2 What decided the outcome (validation runs, 32 each, seeds 10,000+)

| change | exit rate | progress | note |
|---|---|---|---|
| readout = descending neurons only (round 0) | 0 % | 17 % | = always-forward floor; DN turn recall 0.44 |
| + visual projection neurons in the readout | 0 % | 10 % | greedy actions: walks into a wall and stays |
| + sampled actions (T = 1.0) | 16 % | 69 % | gets un-stuck; reactive (scrambled frames: 0 % / 23 %) |
| + 3 DAgger rounds (T = 1.0) | 28 % | 47 % | offline agreement 0.73 → 0.90, closed loop flat |
| + temperature 1.25 (round 3) | **69 %** | **95 %** | T 0.5: 0 %, T 0.75: 6 %, T 1.5: 59 % |
| class-balanced refit (turns up-weighted) | 0 % | 66 % | over-turns; dropped |

Offline held-out agreement with the navigator: 0.90 (round 3; majority action 0.76, copy-previous-action
0.79). A float16 feature cache silently broke round 0 (the closed-loop core is float32; DN rates vary by
~0.004, below float16 resolution): fixed by caching float32 and verified by replaying recorded frames
through the live path (99.8–100 % identical decisions).

### 6.3 Videos

`videos/e1m1_fly_dashboard_runs_40000-40002.mp4`, the first three of twelve consecutive runs (seeds
40,000–40,011, not selected; 7 of the 12 reached the exit): run 1 dies at 72 % of the route, runs 2 and 3
reach the exit. `videos/e1m1_fly_dashboard_best_40004.mp4`, the fastest of those twelve (71 s of game
time). Layout: the game at 1280×960; the simulated neurons' cell bodies at their scanned MaleCNS positions
(119,524 of the 138,968 have a recorded soma; front view, fixed), each dot's brightness = its simulated rate
above its own typical level, where the typical level and spread per neuron are measured once on a separate
calibration run (seed 39,990, 400 decisions) so nothing resets or drifts during the video; cells driven
directly by the eye model are tinted blue-grey; the level map with the walked path; the decision layer's
probability for each button. The footer states how the shown runs were chosen and the final-test result.
Nothing drawn is decorative. (`flybrain/eval/dashboard.py`)

### 6.4 Caveats (worst first)

1. The readout uses 9,199 visual projection neurons as well as the 1,312 descending neurons: much of the
   steering signal is taken from the visual system's output, not from the brain's own motor commands
   (descending neurons alone reach the always-forward floor). The wiring between eye and readout is real
   and untouched, but the deeper central brain contributes less than in the take_cover agent (§1).
2. Success needs stochastic actions at temperature 1.25; the greedy policy walks into walls. Some of the
   route progress is "noise finds the way out"; the scrambled-frames and replay controls bound how much
   (0 % and 6 % exits respectively).
3. The head has 5.4 M parameters (a 256-unit MLP on 21,022 inputs), a large trained component, though it
   sees only the connectome's rates and never the frames.
4. The eye is a hand-built motion/contrast model, not fly vision; "fly plays Doom" is not claimed.
5. Whether fly wiring beats random wiring is untested on E1M1 (Gate 0, `M0_report.md`, argues against it
   on take_cover, and that measurement used a float16 feature cache, so it should be redone).
6. Skill 1 only; one level; dying is the dominant failure.


## 7. Memory in the fly's own mushroom body (2026-09-16/17)

Goal: not a memory bolted onto the model, but the fly's own memory circuit doing what it does in the animal:
learn that an odour predicts punishment or reward, keep several such memories, and let them change a choice.
Ground truth is published fly physiology and behaviour, fixed before the runs: Hige et al. 2015 (Neuron), one
pairing block of an odour with the PPL1-gamma1pedc dopamine neuron depresses that odour's drive onto
MBON-gamma1pedc>alpha/beta (MBON11) by about 80% in spikes and 90% in synaptic charge, the unpaired odour is
unchanged; Owald et al. 2015, reward also works by depression, in the PAM compartments; Aso et al. 2014, MBON
transmitter predicts valence; Tully and Quinn T-maze, wild-type single-cycle performance index 0.44 to 0.53.
Learning rule: Gkanias, McCurdy, Nitabach and Webb 2022 (eLife) on KC->MBON synapses only,
dW = -lr * delta_j * (k_i + W_ij - 1). Wiring comes from the full MaleCNS connectome (minconf 0.5, no weight
threshold), cached once by `flybrain/eval/mb_build.py`. Code: `flybrain/model/mb_plasticity.py`,
`flybrain/eval/mb_sparse.py`, `flybrain/eval/mb_olfactory.py`, `flybrain/eval/mb_behaviour.py`; results in
`outputs/mb/olf5_*.json`. Everything runs on the CPU in seconds. Sections 7 to 9 were regenerated after an
adversarial code review that found two real bugs (a Kenyon-cell code that drifted between the recurrent model's
substeps, and a k-winners-take-all threshold sized to the whole Kenyon-cell pool rather than the driven
subset); both are fixed here.

### 7.1 Negative result: the Doom model's Kenyon cells cannot hold an odour

In the uniform, gain-matched recurrent rate model used for Doom, the 4,064 Kenyon cells respond to every input
the same way: with synthetic odours driving the projection neurons, every cell is active and no odour can be
told from another. This is shown properly in section 9.1 (the fix for the drift bug, plus a decodability test,
does not change it). The feedforward PN->KC drive alone is odour specific; the recurrent dynamics erase it. So
the mushroom body is run as the circuit actually works: feedforward, with the real Kenyon-cell physiology
applied and disclosed.

### 7.2 The circuit, taken from the connectome

Projection neurons -> Kenyon cells -> MBONs, plus the dopamine -> MBON wiring, all from the full connectome:

| element | count |
|---|---|
| olfactory projection neurons (excitatory, uniglomerular; thermo/hygro VP glomeruli and GABAergic vPNs excluded) | 220 cells, 50 glomeruli |
| Kenyon cells | 4,064 (3,755 receive olfactory input) |
| MBONs | 97 cells, 37 types |
| plastic KC->MBON connections | 61,210 |
| PPL101 (punishment) -> MBON, pooled per type | lands on MBON11; 97% of all learning lands there |
| PAM (313 cells, reward) -> MBON | MBON03, 05, 06 lead |

Kenyon-cell physiology (disclosed): feedforward drive through the real PN->KC wiring, then k-winners-take-all
keeping the top 5% of the Kenyon cells this pathway can drive (APL feedback). Result: 4.6% active per odour,
cross-odour cosine 0.11, from the wiring alone. A "binary" reading (a cell fires or not, the spike-count
analogue) is the main condition; odours are synthetic glomerulus sets; one rule call is one pairing block.

### 7.3 What is calibrated and what is predicted

With a binary code the paired drop at an MBON after p pairings is exactly (1 - (1 - lr * delta)^p), so its size
is set by lr, not by the wiring. lr is set once so one pairing gives Hige's 90% at MBON11, and that number is a
calibration, not a result. Everything else follows from the wiring and the rule and can fail:

| endpoint (one pairing, odour A + PPL101, read at MBON11) | value | note |
|---|---|---|
| paired odour drop | 0.90 | calibrated to Hige |
| unpaired odours, mean | 0.13 | = fraction of the odour's MBON11 drive through KCs shared with A; Hige: unchanged |
| share of all lost drive landing on MBON11 | 0.97 | Hige: compartment specific |
| reward (odour C + PAM), drop in PAM compartments / at MBON11 | 0.69 / 0.06 | Owald: reward depresses in PAM compartments |
| A punished and C rewarded together: A / C | 0.85 / 0.69 | two memories in different compartments coexist |

Generalisation follows glomerulus overlap:

| test odour shares with A | 5/6 | 4/6 | 3/6 | 1/6 | 0/6 |
|---|---|---|---|---|---|
| drop at MBON11 | 0.65 | 0.45 | 0.31 | 0.12 | 0.06 |

### 7.4 Controls that can fail

| condition | paired | unpaired | share on MBON11 | reading |
|---|---|---|---|---|
| real wiring, binary code (main) | 0.90 | 0.13 | 0.97 | specific |
| dopamine -> MBON map shuffled | 0.00 (lands on MBON10) | 0.00 | 0.00 | compartment is wiring-set |
| dense code (no k-winners-take-all) | 0.90 | 0.43 | 0.97 | sparse code gives specificity |
| second seed | 0.90 | 0.11 | 0.97 | robust |
| degree-preserving PN->KC / KC->MBON shuffles | 0.90 | ~0.13 | ~0.97 | match; not a test, expected |

The clear negative: at the calibrated strength the published rule cannot hold two memories in the same
compartment, because every dopamine pulse also relaxes the synapses of silent Kenyon cells back toward rest.
Real flies do hold several. Scaling that recovery term down (a rule change, disclosed) retains the first memory.

### 7.5 Does the memory change what the fly does?

Choice through the published valence map (71 approach MBONs, GABA or ACh; 26 avoidance, glutamate; MBON11 is
GABAergic, so depressing it removes approach). P(choose X over Y) = sigmoid(beta (s(X) - s(Y))). Reciprocal
T-maze performance index by beta: 0.05, 0.10, 0.20, 0.38, 0.62 at beta 1, 2, 4, 8, 16 (untrained 0 at every
beta); the wild-type 0.44 to 0.53 is met near beta 10. Beta is fitted; the sign and ordering are not. Several
memories change choices in the right order without fitting: A punished, C rewarded, D untouched give
P(C over A) 0.48 -> 0.82, P(D over A) 0.50 -> 0.70; the shuffled dopamine map inverts this.

### 7.6 Caveats (worst first)

1. The mushroom body is run as a feedforward circuit lifted out of the recurrent model (7.1); section 9 puts it
   back inside the recurrent brain and reports what survives.
2. The 0.90 paired drop is calibrated, not predicted. The predictions are specificity, compartment,
   generalisation, reward compartments, coexistence, and choice ordering.
3. The published rule fails same-compartment coexistence at this strength; the fix shown is a rule change.
4. Odours are synthetic; the antennal lobe is bypassed; there is no time axis, so timing is not tested.
5. Binary Kenyon-cell reading is a choice (graded ceiling ~55%).

## 8. Visual memory: the same circuit, a second sense (2026-09-17)

The plan was smell first, then vision. The same mushroom-body model, switched to the fly's visual input
pathway, tests whether it can also learn that a visual object predicts punishment or reward. All wiring is from
the full connectome cache; the section 7 olfactory result reproduces on it. Code: `mb_olfactory.py
--modality visual`, `mb_behaviour.py --modality visual`; results in `outputs/mb/vis5_*.json`, `beh5_vis.json`.

### 8.1 The visual pathway is real but small

Visual projection neurons reach the visual Kenyon cells (type KCg-d) through the ventral accessory calyx. This
pathway is far smaller and weaker than the olfactory one, matching the biology:

| pathway | connections | inputs reaching KCs | KCs reached | median synapse count |
|---|---|---|---|---|
| smell: olfactory PN to KC | 20,300 | 215 | 3,755 | 17 |
| vision: VPN to KCg-d | 1,088 | 200 of 9,201 | 203 of 206 | 3 |

A visual object is a random set of six of the 101 anatomical visual-projection types (lobula and medulla
feature detectors), the visual analogue of glomeruli.

### 8.2 The visual memory is specific, about as sharp as smell

Kenyon-cell competition is applied to the driven visual subpopulation (the review found an earlier version
sized the winner count to the whole 4,064-cell pool, which disabled competition among the ~200 visual cells and
made vision look artificially coarse; fixed here). With the fix the visual code is sparse and distinct:

| endpoint (one pairing, object A + PPL101, read at MBON11) | smell | vision |
|---|---|---|
| Kenyon cells active per stimulus | 4.6% (~188) | 0.4% (~17) |
| cross-object code similarity | 0.11 | 0.10 |
| paired drop at MBON11 | 0.90 (calibrated) | 0.90 (calibrated) |
| unpaired objects, mean | 0.13 | 0.02 |
| share of all learning landing on MBON11 | 0.97 | 0.95 |
| generalisation, 5 of 6 channels shared | 0.65 | 0.79 |
| reward (PAM) in reward compartments / at MBON11 | 0.69 / 0.06 | 0.65 / 0.06 |
| A punished and C rewarded together: A / C | 0.85 / 0.69 | 0.85 / 0.65 |

Reading: the same circuit learns visual punishment and reward, lands them at the correct output cells and
dopamine compartments, holds a punishment and a reward memory at once, and does so with specificity at least as
sharp as smell (unpaired leakage 0.02, lower than smell's 0.13, because the visual code is sparser). Controls:
shuffling the dopamine-to-MBON map destroys it (paired 0.0004, lands on MBON10 not MBON11); a second seed
agrees (unpaired 0.08); the dense-code control (no competition) raises leakage to 0.19, so the sparse code is
what gives the specificity, as for smell.

### 8.3 Visual memory changes choice, moderately

Through the same valence map (one pairing): the reciprocal T-maze index rises to a peak of about 0.27 near
gain 8 to 16 (untrained 0 at every gain), versus smell's 0.62. So visual memory does shift choice in the right
direction, less strongly than smell. The learned change also reaches the descending neurons more here than for
the pure feedforward olfactory readout (two-hop relative change 0.019 for the trained object versus 0.001 for
an untouched one), because the visual output cells sit on shorter paths to steering, though this is still a
small signal. Visual behaviour is real but weaker than olfactory, consistent with visual conditioning being
harder in flies.

## 9. Putting the memory back inside the recurrent brain (2026-09-17)

Sections 7 and 8 run the mushroom body feedforward. This section tests whether the memory can live inside the
full recurrent brain that plays Doom (138,968 neurons, 4.64 M connections). Code: `flybrain/eval/mb_embed.py`;
results in `outputs/mb/embed2.json`, `outputs/mb/kcsparse2_*.json`.

### 9.1 Why it cannot be done the naive way

The recurrent gain-matched model cannot compute a sparse, stimulus-specific Kenyon-cell code. Driving the
projection neurons (re-imposed every substep, so the odour is not overwritten between integration steps) and
sweeping the Kenyon-cell threshold gives either every cell active or every cell silent, and in neither case can
odour identity be recovered. This is tested three ways so it is not a metric artifact: raw cosine, cosine after
removing the per-cell common-mode pedestal, and nearest-neighbour decoding of odour identity from the code.

| Kenyon-cell threshold | fraction active | centered cross-odour cosine | nearest-neighbour odour decoding (chance 0.17) |
|---|---|---|---|
| default | 1.00 | -0.20 | 0.00 |
| raised a little | 0.85 | -0.20 | 0.00 |
| raised more | 0.00 | degenerate | 0.00 |

Decoding is at or below chance everywhere: the recurrent code carries no recoverable odour identity. This is a
property of the uniform rate model, not the fly; real Kenyon cells are feedforward coincidence detectors.

### 9.2 The faithful embedding, and what it shows

So the Kenyon-cell code is computed the way the cell works (real PN->KC wiring plus k-winners-take-all) and
injected at the Kenyon-cell layer of the full recurrent core, pinned every substep; the learned KC->MBON
changes are applied to the core's own edges; and the memory is read at the output cells and descending neurons
through the full recurrent brain. The memory lives on the connectome's real synapses; only Kenyon-cell activity
is computed feedforward, which is disclosed.

| endpoint, inside the full recurrent brain (one pairing, odour A + PPL101) | value |
|---|---|
| MBON11 fraction of input from Kenyon cells (wiring) | 0.88 |
| unpaired odours, mean drop at MBON11 (wiring-derived) | 0.10 |
| paired odour A, drop at MBON11 (calibrated, capped) | 0.68 |
| paired-over-unpaired specificity ratio | 7 to 1 |
| descending-neuron drive, whole-brain relative change (trained) | 0.0003 |
| descending-neuron drive, largest single-neuron change (on a scale of 5) | 0.006 |
| descending neurons changing by more than 1% of the peak rate | 0 |

Reading. The memory does express inside the full recurrent brain, and specifically: the trained odour drops
about seven times more than unpaired odours at MBON11, on the connectome's real synapses. The specificity is
the wiring-derived result; the paired magnitude (0.68) is a calibrated quantity, and it caps at 0.68 even at
the maximum learning rate, because once the KC->MBON11 synapses are fully depressed the recurrent loop still
restores part of the output. The honest negative: the change does not reach the descending neurons. The
whole-brain relative change is 0.0003, the single most-affected steering neuron moves 0.006 on a scale of 5,
and not one descending neuron changes by even 1% of the peak firing rate. The memory-to-action loop is not
closed inside this model; the behaviour results of sections 7 and 8 go through the published valence map.

A methodology note, in the spirit of reporting what broke: the adversarial review of this code found that an
initial version pinned the Kenyon-cell code only once per decision, but the core runs six substeps per
decision, so the injected code drifted and the memory looked completely absent. Pinning every substep fixed it.
The same drift bug was found and fixed in the sparse-code test of 9.1. The injection has to hold at the
integration timescale, not the decision timescale.

### 9.3 Caveats (worst first)

1. The memory expresses at the output cell but not at the descending neurons (largest change 0.006 of 5): it
   does not yet change what the modelled fly does. Closing that loop is unsolved.
2. Kenyon-cell activity is injected feedforward, not computed by the recurrent model, because the recurrent
   model cannot produce the code (9.1). This is faithful to Kenyon-cell physiology but is a modelling choice.
3. The paired magnitude is calibrated and caps near 0.68; only the specificity ratio and the descending-neuron
   readout are wiring-derived claims.
4. The descending-neuron readout runs on the weight-thresholded Doom subgraph, and the operating point drives
   many cells near their rate ceiling, both of which can only reduce the apparent reach of the memory.
