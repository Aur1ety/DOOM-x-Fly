# Results: a connectome-constrained agent that plays ViZDoom (2026-09-14/15)

**Status: E1M1 (the real first level of Doom) delivered: 57 % of runs reach the exit, reactive, not a
memorised route (§6). take_cover delivered earlier (§1–3, §5); defend_the_line not solved (§4).**

All numbers measured on a shared cluster (gpu-host: 2× A30). Scenario take_cover unless stated.
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
threshold), cached once by `flybrain/eval/mb_build.py` (the cache records its build time and the builder's
source hash). Code: `flybrain/model/mb_plasticity.py`, `flybrain/eval/mb_olfactory.py`,
`flybrain/eval/mb_behaviour.py`; results in `outputs/mb/olf5_*.json`, with the spread over ten independent odour
draws in `outputs/mb/mb_seeds.json` (`flybrain/eval/mb_seeds.py`). The feedforward modules of sections 7, 8 and
10 run on the CPU in seconds; the two section-9 modules need a GPU. `tests/test_mb.py` guards the defects listed
below on a synthetic wiring cache, without the data.

Sections 7, 8 and 10 were regenerated after two adversarial code reviews (the two section-9 GPU outputs,
`kcsparse2_*.json` and `embed2.json`, are the pre-review files; their reruns are pending). The first review
found a Kenyon-cell code that
drifted between the recurrent model's substeps and a k-winners-take-all threshold sized to the whole Kenyon-cell
pool rather than the driven subset. The second (2026-09-19, eight reviewers, every finding independently
re-verified) found that the generalisation tables had been read after eight pairings while labelled as one
(7.3, 8.2), that the section 9.1 decodability test could only ever return zero, that the visual pathway was
described as reaching one Kenyon-cell type when the cache reaches three (8.1), that two behavioural comparisons
between smell and vision said the opposite of what the outputs contain (8.3), that one table cell had no source
(9.2), and that several numbers rested on a single odour draw. All are fixed here and each correction is stated
where it applies.

### 7.1 Negative result: the Doom model's Kenyon cells cannot hold an odour

In the uniform, gain-matched recurrent rate model used for Doom, the 4,064 Kenyon cells respond to every input
the same way: with synthetic odours driving the projection neurons, every cell is active and no odour can be
told from another. This is shown properly in section 9.1 (the fix for the drift bug does not change it). The
feedforward PN->KC drive alone is odour specific; the recurrent dynamics erase it. So
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
| PAM (316 cells, reward) -> MBON | MBON03, 05, 06 lead |

Kenyon-cell physiology (disclosed): feedforward drive through the real PN->KC wiring, then k-winners-take-all
keeping the top 5% of the Kenyon cells this pathway can drive (a stand-in for the APL's feedback inhibition;
section 10.5 computes the code from the real APL loop instead and finds the APL sets only the sparseness
level). Result: 188 cells active per odour (5% of the 3,755 driven cells, 4.6% of all 4,064; the JSON field
`active_frac` uses the 4,064 denominator), cross-odour cosine 0.10 +- 0.01 over ten
odour draws (0.11 for the seed-0 draw the tables below quote), from the wiring alone. A "binary" reading (a
cell fires or not, the spike-count analogue) is the main condition; odours are synthetic glomerulus sets; one
rule call is one pairing block. Unless a row says otherwise, a table cell is the seed-0 draw and the spread in
brackets is mean +- SD over ten draws (`outputs/mb/mb_seeds.json`).

### 7.3 What is calibrated and what is predicted

With a binary code the paired drop at an MBON after p pairings is exactly (1 - (1 - lr * delta)^p), so its size
is set by lr, not by the wiring. lr is set once so one pairing gives Hige's 90% at MBON11, and that number is a
calibration, not a result. Everything else follows from the wiring and the rule and can fail:

| endpoint (one pairing, odour A + PPL101, read at MBON11) | value | note |
|---|---|---|
| paired odour drop | 0.90 | calibrated to Hige |
| unpaired odours, mean | 0.13 (ten draws 0.09 +- 0.03) | = 0.9 x the fraction of the odour's MBON11 drive through KCs shared with A (that share is 0.145 here); Hige: unchanged |
| unpaired odours, the single worst odour of a draw | 0.30 (ten draws 0.21 +- 0.06) | one odour in eight can share enough Kenyon cells with A to lose a fifth of its drive; "unchanged" holds on average, not for every odour |
| share of all lost drive landing on MBON11 | 0.97 (0.96 +- 0.01) | Hige: compartment specific |
| reward (odour C + PAM), drop in PAM compartments / at MBON11 | 0.69 / 0.06 | Owald: reward depresses in PAM compartments |
| A punished and C rewarded together: A / C | 0.85 / 0.69 | two memories in different compartments coexist |

Generalisation follows glomerulus overlap (one pairing; with a binary code the drop is exactly 0.9 times the
share of the test odour's MBON11 drive that passes through A's Kenyon cells):

| test odour shares with A | 5/6 | 4/6 | 3/6 | 1/6 | 0/6 |
|---|---|---|---|---|---|
| drop at MBON11, seed 0 | 0.59 | 0.40 | 0.28 | 0.11 | 0.06 |
| ten draws | 0.60 +- 0.04 | 0.41 +- 0.04 | 0.29 +- 0.02 | 0.12 +- 0.02 | 0.05 +- 0.01 |

Correction (2026-09-19): an earlier version of this table read 0.65 / 0.45 / 0.31 / 0.12 / 0.06. The review
found those values were measured after the eight pairings of the pairing curve above, not after one, and are
the glomerulus overlap itself (the saturated limit). The code now reads the generalisation from the one-pairing
memory, and `tests/test_mb.py` checks it.

### 7.4 Controls that can fail

| condition | paired | unpaired | share on MBON11 | reading |
|---|---|---|---|---|
| real wiring, binary code (main) | 0.90 | 0.13 | 0.97 | specific |
| dopamine -> MBON map shuffled | 0.00 (lands on MBON10) | 0.00 | 0.01 | compartment is wiring-set |
| dense code (no k-winners-take-all) | 0.90 | 0.43 | 0.97 | sparse code gives specificity |
| ten independent odour draws (`mb_seeds.py`) | 0.90 in every draw | 0.09 +- 0.03 (0.05 to 0.13) | 0.96 +- 0.01 | robust |
| degree-preserving PN->KC / KC->MBON shuffles | 0.90 | ~0.13 | ~0.97 | match; not a test, expected |

The clear negative: at the calibrated strength the published rule cannot hold two memories in the same
compartment, because every dopamine pulse also relaxes the synapses of silent Kenyon cells back toward rest
(after odour B is trained, A keeps 0.19 +- 0.06 of its memory, ten draws). Real flies do hold several. Scaling
that recovery term down (a rule change, disclosed) retains the first memory.

### 7.5 Does the memory change what the fly does?

Choice through the published valence map (71 approach MBONs, GABA or ACh; 26 avoidance, glutamate; MBON11 is
GABAergic, so depressing it removes approach). The transmitter of every MBON type is the connectome's consensus
call, unanimous across the cells of all 37 types, and agrees with Aso et al. 2014 for the types they assigned.
P(choose X over Y) = sigmoid(beta (s(X) - s(Y))). Reciprocal T-maze performance index by beta, one pairing:

| beta | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| index, seed-0 odour pair | 0.05 | 0.09 | 0.18 | 0.35 | 0.63 | 0.90 |
| index, ten independently drawn pairs | 0.04 +- 0.00 | 0.09 +- 0.01 | 0.17 +- 0.02 | 0.34 +- 0.03 | 0.59 +- 0.05 | 0.85 +- 0.06 |

The wild-type 0.44 to 0.53 is met near beta 10 to 11 (the walkthrough and video use 11). Beta is fitted; the
sign and ordering are not. The untrained index is not a control: under the reciprocal design it is 0 by
construction for any circuit. What matters is the untrained circuit's innate preference between the two
odours, which for smell is small (|bias| 0.05 +- 0.04 at beta 8) and does not limit the index. The size of what
one pairing writes, free of beta and of the pair drawn, is the drop in the punished odour's approach score:
0.10 +- 0.01 over the ten pairs. Correction (2026-09-19): an earlier version quoted 0.05 / 0.10 / 0.20 / 0.38 /
0.62 at beta 1 / 2 / 4 / 8 / 16; the output (`beh5_olf.json`) reads 0.05 / 0.09 / 0.18 / 0.35 / 0.63, and the
earlier "untrained 0 at every beta" was the tautology above.

Several memories change choices in the right order without fitting: A punished, C rewarded, D untouched give
P(C over A) 0.48 -> 0.82, P(D over A) 0.50 -> 0.70, P(C over D) 0.49 -> 0.67. With the dopamine-to-MBON map
shuffled the same training gives P(C over A) 0.48 -> 0.34, P(D over A) 0.50 -> 0.48, P(C over D) 0.49 -> 0.36,
and the index at beta 8 falls from 0.35 to 0.02 (`beh5_olf_shufdan.json`). The two effects have different
causes. Under this shuffle the punishment lands on MBON10 (`olf5_shufdan.json`), whose output cells carry the
same approach sign as MBON11 but weigh far less in A's score: the punished odour's score shift collapses from
0.10 to 0.005 while keeping its sign (the index stays positive, just tiny). The reward moves from MBON03/05/06
(glutamate, avoidance) to MBON12/16/25-like, where most of the depression falls on approach-sign cells, so
rewarding C lowers C's approach score and C now loses to A and to D; D against A barely moves. Correction
(2026-09-19): an earlier version said the shuffled map "inverts" all of these choices; only the C-over-A and
C-over-D choices invert, and the earlier sentence had no output behind it.

### 7.6 Caveats (worst first)

1. The mushroom body is run as a feedforward circuit lifted out of the recurrent model (7.1); section 9 puts it
   back inside the recurrent brain and reports what survives.
2. The 0.90 paired drop is calibrated, not predicted. The predictions are specificity, compartment,
   generalisation, reward compartments, coexistence, and choice ordering.
3. The published rule fails same-compartment coexistence at this strength; the fix shown is a rule change.
4. Odours are synthetic; the antennal lobe is bypassed; this section has no time axis, so timing is not tested
   here (section 10.1 adds a clock and an eligibility trace).
5. Binary Kenyon-cell reading is a choice (graded ceiling ~55%).

## 8. Visual memory: the same circuit, a second sense (2026-09-17)

The plan was smell first, then vision. The same mushroom-body model, switched to the fly's visual input
pathway, tests whether it can also learn that a visual object predicts punishment or reward. All wiring is from
the full connectome cache; the section 7 olfactory result reproduces on it. Code: `mb_olfactory.py
--modality visual`, `mb_behaviour.py --modality visual`; results in `outputs/mb/vis5_*.json`, `beh5_vis.json`.

### 8.1 The visual pathway is real but small

Visual projection neurons reach the Kenyon cells that carry visual input: mostly KCg-d (through the ventral
accessory calyx) and KCab-p (dorsal accessory calyx), plus a handful of cells of other types. The cache keeps
every Kenyon cell with any visual input, and that whole set is the pool the model competes over (an earlier
version of this table described the pathway as KCg-d only, with 200 inputs and 203 cells; the numbers below are
what the cache actually contains). The pathway is far smaller and weaker than the olfactory one, matching the
biology:

| pathway | connections | inputs reaching KCs | KCs reached | median synapse count |
|---|---|---|---|---|
| smell: olfactory PN to KC | 20,300 | 215 of 220 | 3,755 of 4,064 | 17 |
| vision: VPN to KC | 1,616 | 252 of 9,201 | 332 of 4,064 (203 KCg-d, 108 KCab-p, 21 other) | 3 |

A visual object is a random set of six of the 101 anatomical visual-projection types (lobula and medulla
feature detectors), the visual analogue of glomeruli.

### 8.2 The visual memory is specific, but coarser than smell over ten draws

Kenyon-cell competition is applied to the driven visual subpopulation, the 332 cells with visual input (the
first review found an earlier version sized the winner count to the whole 4,064-cell pool, which disabled
competition among the visual cells and made vision look artificially coarse; fixed). With the fix the visual
code is sparse and distinct:

| endpoint (one pairing, object A + PPL101, read at MBON11) | smell | vision |
|---|---|---|
| Kenyon cells active per stimulus | 188 (5% of the 3,755 driven; 4.6% of all 4,064) | 17 (5% of the 332 driven; 0.4% of all 4,064) |
| cross-object code similarity | 0.11 (0.10 +- 0.01) | 0.10 (0.11 +- 0.04) |
| paired drop at MBON11 | 0.90 (calibrated) | 0.90 (calibrated) |
| unpaired objects, mean, seed 0 | 0.13 | 0.02 |
| unpaired objects, mean, ten draws | 0.09 +- 0.03 | 0.18 +- 0.12 (0.02 to 0.39) |
| unpaired, single worst object of a draw, ten draws | 0.21 +- 0.06 (up to 0.30) | 0.42 +- 0.24 (up to 0.87) |
| share of all learning landing on MBON11 | 0.97 | 0.95 |
| generalisation, 5 of 6 channels shared (one pairing) | 0.59 (0.60 +- 0.04) | 0.71 (0.71 +- 0.07) |
| reward (PAM) in reward compartments / at MBON11 | 0.69 / 0.06 | 0.65 / 0.06 |
| A punished and C rewarded together: A / C | 0.85 / 0.69 | 0.85 / 0.65 |

Reading: the same circuit learns visual punishment and reward, lands them at the correct output cells and
dopamine compartments, and holds a punishment and a reward memory at once. Its specificity is NOT as sharp as
smell's. The seed-0 draw in the table is an unusually clean one; over ten draws the unpaired objects lose
0.18 +- 0.12 of their MBON11 drive (smell: 0.09 +- 0.03), and the worst object of a draw loses up to 0.87. The
reason is the size of the pool: the visual code is 17 winners out of 332 cells, so two random objects can share
most of their Kenyon cells, whereas two odours pick 188 cells out of 3,755. Correction (2026-09-19): an earlier
version of this section claimed visual specificity "at least as sharp as smell" from the single 0.02 draw; the
ten-draw numbers above replace it. Controls: shuffling the dopamine-to-MBON map destroys the memory at MBON11
(0.0004; the punishment lands on MBON10, a compartment the visual Kenyon cells do not reach, so the calibration
itself is impossible there and the rule keeps its default strength, which the output now states); the
dense-code control (no competition) raises the seed-0 leakage from 0.02 to 0.19, so the sparse code is still
what gives the specificity there is, as for smell.

### 8.3 Visual memory changes choice, moderately

Through the same valence map (one pairing). For the seed-0 object pair the reciprocal T-maze index peaks at 0.27
near gain 8 to 16; over ten independently drawn pairs it is 0.20 +- 0.11 at gain 8 (0.01 to 0.38) and
0.23 +- 0.19 at gain 16 (0.00 to 0.67), against smell's 0.34 +- 0.03 and 0.59 +- 0.05. That gap is not the
memory. The untrained circuit already prefers one object of most visual pairs (innate |bias| 0.31 +- 0.13 at
gain 8, against 0.05 +- 0.04 for odour pairs), because a 17-cell code gives the two objects unequal valence
sums, and the reciprocal index falls with that bias (correlation -0.92 at gain 8: after the preferred object is
punished, the fly can still pick it). The size of what the pairing writes, the drop in the punished object's
approach score, is 0.10 +- 0.03, against 0.10 +- 0.01 for smell: the memory writes a change of the same mean
size in both senses (three times more variable across visual pairs); it is the innate preference of the pair
that caps the visual index. Correction
(2026-09-19): an earlier version read "visual memory shifts choice less strongly than smell, consistent with
visual conditioning being harder in flies"; that comparison was one object pair with a large innate bias, and
is withdrawn.

Reach into the descending neurons, through the cache's MBON -> DN and MBON -> interneuron -> DN routes (a relay
is an interneuron, never another MBON or a DN; an earlier version let MBONs and DNs relay, and excluding them
left the direct route unchanged, moved the pooled two-hop values by 2 to 7% (smell 0.036 to 0.037, vision 0.019
to 0.021, driven DNs 1,285 to 1,256) and the single-cell and untouched figures by up to 20% (DNa02 two-hop
0.018 to 0.021 smell, 0.009 to 0.011 vision), with every comparison below keeping its sign under both
versions). Smell reaches the steering neurons MORE than vision in absolute terms: two-hop relative
change 0.037 (smell) against 0.021 (vision); direct 0.031 against 0.014; DNa02 two-hop 0.021 against 0.011;
DNa03 0.012 against 0.010. Vision is more trained-specific: its trained object moves the two-hop drive 13 times
more than an untouched object (0.021 against 0.0016), smell 7.6 times (0.037 against 0.0049). Correction
(2026-09-19): an earlier version said the change "reaches the descending neurons more here than for smell,
because the visual output cells sit on shorter paths to steering"; the outputs say the opposite, and no
path-length measurement exists to support the mechanism, so both are withdrawn. Either way the signal is small
and does not close the loop to action (section 9.2, 10.4).

## 9. Putting the memory back inside the recurrent brain (2026-09-17)

Sections 7 and 8 run the mushroom body feedforward. This section tests whether the memory can live inside the
full recurrent brain that plays Doom (138,968 neurons, 4.64 M connections). Code: `flybrain/eval/mb_sparse.py`
(9.1; results in `outputs/mb/kcsparse2_*.json`) and `flybrain/eval/mb_embed.py` (9.2; `outputs/mb/embed2.json`).
Both need a GPU.

### 9.1 Why it cannot be done the naive way

The recurrent gain-matched model cannot compute a sparse, stimulus-specific Kenyon-cell code. Driving the
projection neurons (re-imposed every substep, so the odour is not overwritten between integration steps) and
sweeping the Kenyon-cell threshold gives either every cell active or every cell silent, with or without the
APL, and the raw cross-odour cosine of the Kenyon rates stays at 1.00: the six odours produce the same pattern.

| Kenyon-cell threshold (resting potential) | fraction active, APL intact / APL silenced | mean Kenyon rate | raw cross-odour cosine |
|---|---|---|---|
| default | 1.00 / 1.00 | 0.68 | 1.000 |
| raised a little (-2) | 0.85 / 1.00 | 0.11 | 0.999 |
| raised more (-5) | 0.00 / 0.00 | 0.006 | 0.999 |
| raised further (-10) | 0.00 / 0.00 | 0.000 | 0.999 |

Correction (2026-09-19). An earlier version of this table also reported a "centered cross-odour cosine" of
-0.20 and a "nearest-neighbour odour decoding" of 0.00 against a chance of 0.17, and presented them as two
further independent tests. The second review showed both were artefacts of the test design, not measurements:
with one presentation per odour, subtracting the per-cell mean across six odours forces the mean cross-odour
cosine to exactly -1/(6-1) = -0.20 for any code, and a decoder that excludes the self-match with one sample
per class cannot return anything but 0. They carried no evidence and are removed. `mb_sparse.py` now presents
each odour five times with rate jitter and scores a leave-one-out nearest-centroid decoder on held-out
presentations (chance 1/6; a separable code scores 1, checked on synthetic codes in `tests/test_mb.py`); that
rerun needs GPU time and is pending. The conclusion rests on what was measured: every cell at the same rate,
or silent, is no code. This is a property of the uniform rate model, not the fly; real Kenyon cells are
feedforward coincidence detectors.

### 9.2 The faithful embedding, and what it shows

So the Kenyon-cell code is computed the way the cell works (real PN->KC wiring plus k-winners-take-all) and
injected at the Kenyon-cell layer of the full recurrent core, pinned every substep; the learned KC->MBON
changes are applied to the core's own edges; and the memory is read at the output cells and descending neurons
through the full recurrent brain. The memory lives on the connectome's real synapses; only Kenyon-cell activity
is computed feedforward, which is disclosed.

| endpoint, inside the full recurrent brain (one pairing, odour A + PPL101) | value |
|---|---|
| MBON11 fraction of input from Kenyon cells (full connectome; `MBON11_input_frac_from_KC` in `olf5_binary.json`; an earlier version printed 0.88 with no source) | 0.86 |
| unpaired odours, mean drop at MBON11 (wiring-derived) | 0.10 |
| paired odour A, drop at MBON11 (calibrated, capped) | 0.68 |
| paired-over-unpaired specificity ratio | 7 to 1 |
| descending-neuron drive, relative change pooled over the 1,322 descending neurons (trained) | 0.0003 |
| descending-neuron drive, largest single-neuron change (on a scale of 5) | 0.006 |
| descending neurons changing by more than 1% of the peak rate | 0 |

Reading. The memory does express inside the full recurrent brain, and specifically: the trained odour drops
about seven times more than unpaired odours at MBON11, on the connectome's real synapses. The specificity is
the wiring-derived result; the paired magnitude (0.68) is a calibrated quantity, and it caps at 0.68 even at
the maximum learning rate, because once the KC->MBON11 synapses are fully depressed the recurrent loop still
restores part of the output. The honest negative: the change does not reach the descending neurons. The
relative change pooled across the 1,322 descending neurons is 0.0003 (an earlier version of this text called
that number "whole-brain"; it was never computed over the whole brain. `outputs/mb/embed2.json` is the
pre-review output and still stores this DN-pooled value under the old key `rel_change_whole_brain`;
`mb_embed.py` now writes `rel_change_DN_population`, a true whole-core `rel_change_whole_brain` and the
subgraph's MBON11 Kenyon-input fraction, but that rerun needs GPU time and is pending, as for 9.1), the
single most-affected steering neuron moves 0.006 on a scale of 5, and not one descending neuron changes by even
1% of the peak firing rate. The memory-to-action loop is not closed inside this model; the behaviour results of
sections 7 and 8 go through the published valence map. Section 10.4 shows from the connectome alone that no
memory-specific route from MBON11 to the steering neurons exists, so this is anatomy, not a modelling shortfall.

A methodology note, in the spirit of reporting what broke: the adversarial review of this code found that an
initial version pinned the Kenyon-cell code only once per decision, but the core runs six substeps per
decision, so the injected code drifted and the memory looked completely absent. Pinning every substep fixed it.
The same drift bug was found and fixed in the sparse-code test of 9.1. The injection has to hold at the
integration timescale, not the decision timescale.

### 9.3 Caveats (worst first)

1. The memory expresses at the output cell but not at the descending neurons (largest change 0.006 of 5): it
   does not yet change what the modelled fly does. Closing that loop is unsolved, and section 10.4 finds the
   connectome gives it no dedicated route to close.
2. Kenyon-cell activity is injected feedforward, not computed by the recurrent model, because the recurrent
   model cannot produce the code (9.1). This is faithful to Kenyon-cell physiology but is a modelling choice.
3. The paired magnitude is calibrated and caps near 0.68; only the specificity ratio and the descending-neuron
   readout are wiring-derived claims.
4. The descending-neuron readout runs on the weight-thresholded Doom subgraph, and the operating point drives
   many cells near their rate ceiling, both of which can only reduce the apparent reach of the memory.

## 10. Experiments on the simulated fly (2026-09-18/19)

Five further things you can do to the section-7 circuit once it exists: teach it in real time, lesion it,
give it a partial cue, trace where its memory could go, and take away the one hand-set part of its Kenyon
code. All run on the CPU in seconds on the wiring cache; results in `outputs/mb/online.json`, `lesion.json`,
`recall.json`, `pathway.json`, `apl.json`. Each output carries its own verdict string and says which of its
numbers are calibrated, which are closed forms of the rule, and which come from the wiring. These modules
went through the same adversarial review as sections 7 to 9; the labelling below is what survived it.

### 10.1 Teaching with a clock (`mb_online.py`)

Sections 7 to 9 have no time axis: one rule call is one pairing block. Here the odour switches its Kenyon cells
on for 5 s, dopamine arrives as four 50 ms pulses at 2 Hz starting at a set delay, and a KC->MBON synapse
weakens only where a decaying eligibility trace of recent Kenyon activity (time constant 0.8 s) and dopamine
coincide. The rule is a trace-gated multiplicative depression, NOT the Gkanias rule of section 7 (it has no
silent-cell recovery term, so the same-compartment failure of 7.4 does not arise here); the two are not
interchangeable. One constant is anchored so the standard protocol (pulses from 0.2 s after odour onset) gives
Hige's 0.90 at MBON11. With a binary code every active synapse then sees the same trace and the same dopamine,
so the MBON11 drop for ANY schedule is a closed form of the anchor and the timings; the module prints that
closed form next to each measured value and they coincide. The timing and dose tables are therefore the rule,
not the wiring; what the wiring contributes is the unpaired odour (0.06) and the behaviour.

| dopamine train, relative to a 5 s odour | drop at MBON11 |
|---|---|
| ends 2.0, 1.0, 0.5 s BEFORE odour onset | 0.00, 0.00, 0.00 (zero by construction: no trace before Kenyon activity) |
| starts 0, 0.2, 0.5, 1, 2 s after onset (during the odour) | 0.85, 0.90, 0.94, 0.96, 0.97 |
| starts 0.2, 1, 2 s after odour OFFSET | 0.74, 0.38, 0.13 (the trace decaying) |
| pulses at 0.2 s: 0, 1, 2, 4, 8, 16 | 0.00, 0.20, 0.53, 0.90, 1.00, 1.00 |

Controls, both zero by the mechanism: dopamine with no odour 0.00, odour with no dopamine 0.00. Behaviour
through the valence map (gain 8): P(avoid A) 0.50 untrained, 0.68 after forward teaching, 0.50 after a train
that ends 2.2 s before the odour; reciprocal index 0.35 forward, 0.00 backward; avoidance against pulses
0.50, 0.54, 0.61, 0.68, 0.69, 0.68. Caveats: the backward zero is a property of a forward-only trace, not a
measured timing curve; the bidirectional rule of Cohn et al. 2015 (dopamine before the odour potentiates) is
not modelled; single odour draw.

### 10.2 Lesions (`mb_lesion.py`)

The rule is calibrated once on the intact circuit and held fixed; each lesion is applied, the same one pairing
is run, and the memory (MBON11) and the behaviour (P(avoid A) at gain 8, before and after) are read. Three
rows are identities of the rule or the readout and cannot fail once the dopamine-to-MBON compartment map is
taken as given: no punishment dopamine means no weight change; zeroing MBON11's valence removes the cell
carrying 97% of the depression (that share is the compartment map, itself a wiring result of 7.3 and 7.4, so
this row is an identity given the map, not independent of the wiring); and no KC->MBON synapses means nothing
to depress. Only the APL and KCg-m rows test the connectome here.

| lesion | real fly | wiring test? | unpaired leak at MBON11 | MBON11 drive to A lost | learned shift in P(avoid A), fraction of intact |
|---|---|---|---|---|---|
| none | learns and avoids | (baseline) | 0.08 | 0 | +0.18 (1.00) |
| PPL1-gamma1pedc silenced | memory abolished (Aso 2010, 2012) | no, identity | 0 | 0 | 0.00 (0.00) |
| MBON11 silenced in the readout | avoidance lost (Aso 2014, Perisse 2016) | no, identity | 0.08 | 0 | -0.00 (-0.02) |
| APL removed (dense code) | specificity lost (Lin 2014) | yes | 0.41 | not applicable (dense code raises drive) | +0.07 (0.37) |
| KCg-m Kenyon cells silenced | short-term memory impaired (Aso 2014) | yes | 0.02 | 0.75 | +0.15 (0.82) |
| all KC->MBON synapses cut | nothing stored | no, identity | none | 1.00 | 0.00 (0.00) |

Reading: the two wiring-dependent rows go the way the animal goes. Without APL sparsening the memory leaks onto
unpaired odours five times more (0.08 -> 0.41) and the learned avoidance drops to 37% of intact; silencing the
main gamma Kenyon cells removes three quarters of MBON11's drive to the odour and leaves 82% of the learned
avoidance, a mild impairment. The behavioural readout is the change in P(avoid A), not its absolute value: the
untrained baseline is lesion-specific (0.43 to 0.55 here), so an absolute 0.5 means nothing. Caveats: the MBON11
row lesions the readout, not the circuit; the paired drop itself is calibration-locked (0.90 whenever any
trained synapse survives) and cannot express graded impairment; single odour draw.

### 10.3 A partial cue (`mb_recall.py`)

Teach the full six-glomerulus odour once, then present only some of its glomeruli.

| glomeruli of A presented | 6/6 | 5/6 | 4/6 | 3/6 | 2/6 | 1/6 |
|---|---|---|---|---|---|---|
| recall at MBON11, fraction of the full-odour drop | 1.00 | 0.81 | 0.65 | 0.50 | 0.42 | 0.29 |

Recall is graded (slope 0.83 against cue fraction), never complete. What this measures, exactly: with a binary
code and one pairing, the recall fraction EQUALS the share of the partial cue's MBON11 drive that passes through
the trained odour's Kenyon cells (the module prints both; they agree to the third decimal in the recorded
output). So the curve is
the connectome's partial-cue overlap, and completion is excluded for this feedforward circuit by construction;
the finding is the shape of the graded curve, not a test that could have shown completion. A recurrent or
attractor stage would be needed for a degraded cue to re-create the whole memory.

### 10.4 Where the memory could go: MBON11 to the steering neurons (`mb_pathway.py`)

From the full connectome alone, no model: the routes from MBON-gamma1pedc (2 cells) to the established steering
descending neurons DNa02 and DNa03 (4 cells). A bridge is an interneuron; MBONs and DNs are not counted as
relays. The bridge that matters is one that both receives a meaningful share of its input from MBON11 (so the
memory can modulate it) and drives the steering neurons.

| quantity | value |
|---|---|
| direct MBON11 -> DNa02/DNa03 synapses | 0 |
| interneuron bridges (types) | 19 (16) |
| strongest bridge by memory-modulated throughput | CRE021: 72 synapses onto the steering DNs, but MBON11 is 0.19% of its input |
| largest raw capacity | AOTU019: 1,058 synapses onto the steering DNs, 5 from MBON11 (0.01% of its input) |
| most MBON11-specific bridge | SMP272: 0.22% of its input from MBON11, 3 synapses onto the steering DNs |
| share of the steering DNs' 84,206 input synapses that MBON11 can modulate | 0.0003% |

Reading: the connectome offers no memory-specific route from this compartment to steering, which is the
anatomical version of section 9.2's negative. Prediction, testable in a fly: silencing the top bridge types
(CRE021, AOTU019, SMP148) should NOT selectively abolish learned odour avoidance while leaving naive behaviour
intact; a null result supports diffuse summation over many compartments or a longer route, a positive hit means
the behavioural path uses connections below this connectome's confidence threshold. AOTU019 in particular is a
poor target: it steers, but MBON11 barely touches it, so a deficit there would not be memory-specific.

### 10.5 The Kenyon code from the real APL loop (`mb_apl.py`)

The one hand-set part of section 7 is the top-5% rule that stands in for APL inhibition. Here the code is
computed the way the circuit does it: the projection-neuron drive enters, the Kenyon cells excite the two APL
cells (KC -> APL synapses from the connectome), the APL inhibits every Kenyon cell back in proportion to its own
APL -> KC synapse count, and the loop relaxes to a fixed point (convergence checked). One parameter is set, the
inhibition gain, chosen so the code lands at the same density as the top-5% baseline; which cells survive is
decided by the wiring. Ten seeds, fresh odours and a fresh shuffle each; cross-odour cosine, lower is more
distinct, at matched density (about 4.6% active):

| Kenyon code | cross-odour cosine |
|---|---|
| top-5% of the PN->KC drive, no APL (section 7) | 0.100 +- 0.015, the most distinct in all ten seeds |
| APL loop, every APL->KC weight set to the same value | 0.107 +- 0.014 |
| APL loop, real per-cell APL->KC weights | 0.130 +- 0.016 |
| APL loop, real weights shuffled across Kenyon cells | 0.179 +- 0.016, worse in all ten seeds |

Reading: the real loop does sparsen the code from one gain, so the sparseness LEVEL no longer has to be set by
hand. But the per-cell APL->KC weights do not create the odour identity: a structureless uniform threshold is
as distinct as the real weights in nine of ten seeds, and the pure feedforward top-k beats both. The real weights
are not random (shuffling them hurts), but "beats random" is not "makes the code". Mechanistically this is
near-forced: with two APL cells the loop's inhibition is rank 2, a near-global modulation that cannot select
winners cell by cell. Honest conclusion: the PN->KC wiring gives the odour identity; the APL loop sets only how
many cells fire. This module went through two reviews of its own (a density confound in the comparison and a
one-sided test were found and removed).

### 10.6 Caveats for this section (worst first)

1. 10.1's timing and dose curves are closed forms of an anchored rule; they demonstrate the trace mechanism and
   contain no connectome information. Only the unpaired odour and the behaviour there depend on the wiring.
2. 10.2 has two real tests in six rows; three rows are identities kept as floors and one is the intact baseline.
3. 10.3 cannot show completion by construction; it measures the partial-cue overlap.
4. 10.4 is anatomy at minconf 0.5: capacity, not proven necessity, and a route below the threshold is invisible.
5. 10.1 to 10.3 are single odour draws; 10.5 has ten.
