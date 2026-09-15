# DOOM-x-Fly

I took the fruit fly brain scan that came out in September (MaleCNS, from Janelia and Google), simulated the wiring as-is, and got it to play the first level of Doom.

Not a fly. Not "the brain thinking". The scan is a wiring map: 138,968 neurons and 4.6 million synapses, plus which synapses excite and which inhibit. I run that map as a big recurrent network and never change a connection. In front of it there's an eye model I wrote that turns the game frame into inputs for the fly's real motion-detecting cells. Behind it there's a small trained layer that looks at 10,511 of the simulated neurons and picks one of 8 buttons. The game is the real shareware E1M1, easiest skill, nothing modified.

## Does it actually play?

100 runs from starting positions it never saw during training:

| | reached the exit | got through (avg) |
|---|---|---|
| random buttons | 0% | 13% |
| always forward | 0% | 14% |
| replaying a recorded winning run, blind | 6% | 48% |
| **the fly-scan agent** | **57%** | **92%** |
| same agent, its own frames in random order | 0% | 22% |
| same agent, black screen | 0% | 8% |
| scripted player that can see the map (ceiling) | 98% | 99% |

So it finishes the level more often than not, and it's really using what it sees: scramble its frames and it drops to zero. It's also not just replaying a memorised route, since a blind replay of a good run only works 6% of the time. When it fails it's usually because it died. It's decent at finding its way and bad at fighting.

All the numbers, error bars and the failed attempts are in [docs/RESULTS.md](docs/RESULTS.md).

## Videos

Two clips, with the game, the simulated neurons lighting up at their real positions in the brain, the map, and what the decision layer is doing:

- the fastest of 12 consecutive runs (71 seconds)
- the first three of those 12, unedited: it dies once, then makes it twice

Too big for git. Links coming.

## What's real and what's mine

The neurons, synapses and their signs come straight from the scan. The only thing I tune in the wiring is one global scale factor so the network doesn't blow up or go silent.

The eye model is hand-built and is not fly vision. The decision layer is a normal small neural net (256 hidden units, reads 1,312 descending neurons and 9,199 visual projection neurons). It was trained by copying a scripted player that can see the map, then three rounds of "let the agent drive, have the script correct it". It never sees the screen, only neuron activity. The agent also needs a bit of randomness in its button choices (temperature 1.25); if it always takes its top choice it walks into a wall and stays there.

The big open question: does the fly's specific wiring matter, or would any random wiring with the same size do the same? I haven't tested that on this level. An earlier test on a simpler scenario suggested random wiring carries at least as much signal. So don't read this as "the fly brain is smart". Read it as "the scan, run unchanged, passes enough of the picture through to steer with".

## How to run it

You need the MaleCNS v1.0 flat connectome files (CC BY 4.0, four Feather files) in `$FLYBRAIN_DATA/malecns_v1/`, and the shareware `DOOM1.WAD` v1.9 in `$FLYBRAIN_DATA/wads/`. Python 3.11, PyTorch with CUDA, ViZDoom 1.3, the rest is in `pyproject.toml`. Set `FLYBRAIN_DATA` and `FLYBRAIN_OUT`, then:

```bash
python -m flybrain.data.import_malecns
python -m flybrain.data.subgraph --w-min 5
python -m flybrain.vision.geometry
python -m flybrain.env.wadmap --wad $FLYBRAIN_DATA/wads/doom1.wad --map E1M1
python -m flybrain.train.level_bc record --episodes 200 --workers 22 --wiggle 10 --wiggle-moves --out $FLYBRAIN_OUT/e1m1/demo_nav_v2.npz
FLYBRAIN_GPU_SHARED=0 bash scripts/e1m1_rounds.sh 0 3
FLYBRAIN_GPU_SHARED=0 bash scripts/e1m1_final.sh $FLYBRAIN_OUT/e1m1/v2_r3/student.pt r3_T125 1.25
python -m flybrain.eval.dashboard --ckpt $FLYBRAIN_OUT/e1m1/v2_r3/student.pt --temperature 1.25 --seeds 40004 --out demo.mp4
```

Training takes a couple of hours on one A30. `python -m pytest tests/` runs the tests. The scripts `sync.sh`, `remote.sh`, `detach.sh` and `gpu_window.sh` are for my own two-machine setup, ignore them. Don't run the whole-brain step of `flybrain/model/bench.py` on CPU, it tries to build a dense 139k x 139k gradient.

## Where things are

The package is still called `flybrain` from before the project had a name.

- `flybrain/data` – reading the scan, cutting the subgraph, rewired controls
- `flybrain/model` – the sparse kernel and the rate network
- `flybrain/vision` – the eye model
- `flybrain/env` – ViZDoom wrappers, WAD parsing, the E1M1 environment and the scripted navigator
- `flybrain/train` – the student, imitation/DAgger, evaluation
- `flybrain/eval` – probes and the video renderer
- `docs/` – results and the data import report

## Credits

MaleCNS v1.0 by HHMI Janelia Research Campus and Google Research (CC BY 4.0). ViZDoom by Marek Wydmuch and others. Doom shareware by id Software. DOOMFLY and the other fly-connectome demos are what made me want to check whether one of these could pass a blindfold test. 
