# Results — a connectome-constrained agent that plays ViZDoom (2026-09-14/15)

**Status: E1M1 (the real first level of Doom) delivered — 57 % of runs reach the exit, reactive, not a
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
| connectome agent v1, **blindfolded** (black frames) | — | 179 | 168 [156, 183] |
| connectome agent v1, flat-grey frames | — | 186 | 174 [159, 192] |

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
kills normal, 16.3 blindfolded** — it presses attack 95–99 % of the time; a copy of the teacher's
dominant habit, not a reactive policy. Cloning the labels-aiming script instead (60 k decisions:
69 % fire, 31 % turns; frozen wiring + linear readout, held-out agreement 0.667): **12.7 kills with
vision vs 10.4 blindfolded** (IQM 11.0 [10.0, 12.1] vs 9.0 [8.4, 9.8]) — a real but small
reactivity gap; it turns 8 % of the time vs the script's 31 %. Enemy azimuth is less decodable
from the descending neurons on this scenario (Gate 0 R² 0.41 vs 0.46). The nonlinear-readout
variant collapsed onto "fire" (99 %): 16.2 kills with vision vs 16.5 blind — not reactive. A
class-balanced cloning run (turn decisions up-weighted 1/frequency) over-corrected: with vision it
turns 68 % of the time (mostly right) and scores 8.3 kills, blindfolded it fires and scores 15.9 —
strongly input-dependent, but not a competent policy. **defend_the_line stays unsolved with the
frozen-wiring recipe**: enemy azimuth is only weakly linearly available in the descending neurons
(Gate 0 R² 0.41), and every readout either ignores it (fire-only) or over-reacts to it.

## 5. take_cover: pushing past the linear-readout plateau

DAgger round (`bc3b_frozen_lin_d1`: 40 k student-visited states labelled by the teacher, readout
retrained; held-out agreement 0.784): **505 mean / 465 IQM [380, 569]**, blindfolded 179 — no
better than v1 (565 / 509 [438, 603]); the linear readout of the frozen wiring plateaus at
~500–560. Two variants: (a) unfreezing the 4.86 M synapse magnitudes at LR 1e-4 with a drift
penalty, warm-started from v1, destabilised within two epochs (gradient norms 470 → 970, KL rising
0.39 → 0.92, agreement falling) and was stopped; (b) **frozen wiring with a 256-unit nonlinear
readout** (`bc4b_frozen_mlp`, 4 epochs on teacher + DAgger data, held-out agreement 0.793):
**633 mean / 555 IQM [475, 632]**, blindfolded 171 — the best agent so far, wiring still untouched.
Demo video (5 episodes, mean 656): `connectome_take_cover_frozen_mlp_v2.mp4` (server `outputs/videos/`).
A second DAgger round on this agent (`bc4c_frozen_mlp_d2`, 40 k more student-visited states,
held-out agreement 0.792) scored **488 / 435 IQM [364, 521]**, blindfolded 171 — worse; DAgger does
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

Training: imitation of a privileged scripted navigator (map geometry + player position; 98 % exits) —
200 demonstration runs (144,808 decisions, 5 % random-movement bursts for coverage), then 3 DAgger rounds
(the agent drives, the navigator labels; 48 runs each). Labels the navigator produced from hidden state
(random un-stick wiggles, off-grid positions) are masked from the loss. Round 3 was chosen on 32 validation
runs (seeds 10,000+); the table below is a separate, untouched block.

### 6.1 Final test — 100 runs each, seeds 20,000–20,099, never used before

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
| navigator, wide starts + sticky | 95 % [90, 99] | 98 % [96, 100] | 100 % | 4 | 1 | — |
| replay, wide starts + sticky | 0 % | 35 % [31, 40] | 27 % | 42 | 58 | — |

Reading: the agent finishes the real level in 57 % of runs and on average gets 92 % of the way; the median
run reaches the exit. Widening the starts and forcing 10 % repeated actions does not hurt (64 %, within
noise), while the same perturbation drops the blind replay of a recorded run from 6 % to 0 % exits — the
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

`videos/e1m1_fly_dashboard_runs_40000-40002.mp4` — the first three of twelve consecutive runs (seeds
40,000–40,011, not selected; 7 of the 12 reached the exit): run 1 dies at 72 % of the route, runs 2 and 3
reach the exit. `videos/e1m1_fly_dashboard_best_40004.mp4` — the fastest of those twelve (71 s of game
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
3. The head has 5.4 M parameters (a 256-unit MLP on 21,022 inputs) — a large trained component, though it
   sees only the connectome's rates and never the frames.
4. The eye is a hand-built motion/contrast model, not fly vision; "fly plays Doom" is not claimed.
5. Whether fly wiring beats random wiring is untested on E1M1 (Gate 0, `M0_report.md`, argues against it
   on take_cover — and that measurement used a float16 feature cache, so it should be redone).
6. Skill 1 only; one level; dying is the dominant failure.
