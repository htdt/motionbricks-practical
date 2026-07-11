# MotionBricks — installing and running the motion generator

Stage 2 (BAKE.md) needs a motion source. The one this pipeline is built on —
and the one all gate thresholds were validated against — is **NVIDIA
MotionBricks** (<https://nvlabs.github.io/motionbricks/>), a real-time
generative motion model over the Unitree G1 humanoid skeleton (29 DOF). It
ships as a subproject of
[GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl).
This document is the complete download → install → generate manual; the
`motionbricks/` directory of this repo contains the tooling that drives it.

Why this model fits the pipeline: its inference API is **keyframe-conditioned
inbetweening** — give it 4 context frames and a 4-frame target keyframe
window, it generates the motion between. That is exactly the shape BAKE.md's
pose-library + move-spec design needs. Everything runs offline; the model
never ships with the game (baked clips do).

Licensing: MotionBricks code is Apache-2.0; the pretrained checkpoints are
under the NVIDIA Open Model License (commercial use permitted, generated
outputs are yours). Check the upstream `legal/` directory for the current
terms.

## 0. Requirements

- Linux, CUDA-capable GPU (the stack loads ~2.3 GB of fp32 checkpoints;
  8 GB VRAM is comfortable)
- Python 3.10+, conda (or any venv manager)
- [Git LFS](https://git-lfs.com/) — checkpoints and meshes are LFS objects
- `xvfb` for headless runs (`apt install xvfb`)
- ~6 GB disk

## 1. Download

```bash
git lfs install
git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
cd GR00T-WholeBodyControl

# checkpoints (~2.2 GB) and G1 meshes are excluded from a normal clone; fetch them:
git lfs pull --include="motionbricks/out/**" --exclude=""
git lfs pull --include="motionbricks/assets/skeletons/g1/meshes/**" --exclude=""
```

Verify you got real files, not LFS pointer stubs (~1 KB means pointer —
re-run the pulls):

```bash
ls -lh motionbricks/out/G1-clip.ckpt                                     # ~7.5 MB
ls -lh motionbricks/out/motionbricks_vqvae/version_1/checkpoints/*.ckpt  # ~273 MB
ls -lh motionbricks/out/motionbricks_pose/version_1/checkpoints/*.ckpt   # ~1.6 GB
ls -lh motionbricks/out/motionbricks_root/version_1/checkpoints/*.ckpt   # ~391 MB
```

The mesh pull matters even for headless generation: the G1 MuJoCo scene
(`assets/skeletons/g1/scene_29dof.xml`) references the mesh files, and every
tool here loads that scene for FK.

## 2. Install

```bash
conda create -n motionbricks python=3.10 -y
conda activate motionbricks
cd motionbricks
pip install -e .
```

That single install covers everything the tooling needs (torch, mujoco,
scipy, matplotlib come with it).

## 3. Add this repo's tooling

The generation tools run **from inside** `GR00T-WholeBodyControl/motionbricks/`
(they import the installed `motionbricks` package and use its relative
checkpoint/asset paths). Copy them in:

```bash
git clone https://github.com/htdt/motionbricks-practical.git
cp motionbricks-practical/motionbricks/*.py \
   motionbricks-practical/motionbricks/*.json \
   GR00T-WholeBodyControl/motionbricks/
```

| File | What it is |
|---|---|
| `mbstack.py` | glue: builds the full inference stack the way the upstream demo does |
| `probe_api.py` | install smoke-test (run this first) |
| `posekit.py` | pose library: mine/inspect/save keyframe poses (BAKE.md §2) |
| `pose_library.json` | starter library — 15 poses (stance, strikes, kicks, hits, …) |
| `movegen.py` | keyframe-conditioned move generation, best-of-N + QA gates (BAKE.md §3–4) |
| `qposops.py` | pure-numpy qpos ops: world-space ground/air gates + the spec's `"post"` edit block (BAKE.md §3a, §4); self-tests with `python qposops.py` |
| `moves_example.json` | example move spec — a 17-move fighting-game set |
| `bake_moves.py` | canonicalize + trim + export motion JSONs + manifest (BAKE.md §5) |
| `pipeline.py` | one-command rebake ritual: movegen → bake → certify → prebake |
| `make_motion_json.py` | MuJoCo Z-up → three.js Y-up conversion + the G1→humanoid bone map |

## 4. Smoke test

```bash
cd GR00T-WholeBodyControl/motionbricks
xvfb-run -a python probe_api.py
```

First run takes a minute (checkpoint loading). Success ends with
`PROBE OK — MotionBricks stack is functional.` and prints the layout facts
the tooling relies on: qpos is `(36,)` = root pos(3) + root quat WXYZ(4) +
29 dof, 30 fps, generation chunks of 6–16 four-frame tokens.

`xvfb-run -a` is required on headless machines — the demo stack initializes a
MuJoCo rendering context even with the viewer disabled.

## 5. Generate a move set

```bash
# sanity: the starter pose library
python posekit.py list

# generate every move in the spec, 8 seeds each, keep the best per move
xvfb-run -a python movegen.py --spec moves_example.json --seeds 8
# iterate on one move without regenerating the rest:
xvfb-run -a python movegen.py --spec moves_example.json --only uppercut --seeds 8

# canonicalize + export engine-consumable JSONs + manifest
python bake_moves.py --in-dir out/moves --out-dir baked --spec moves_example.json
```

`movegen.py` prints per-seed gate values (keyframe arrival error, foot skate,
jitter, limit violations, plus the world-space ground/air gates — thresholds
and meaning in BAKE.md §4) and writes `out/moves/<move>.npz` + `<move>.json`
(all seeds' gates + frame data). `bake_moves.py` turns those into
`baked/<move>.json` motion clips (canonicalized, trimmed, Y-up, ALIGN.md clip
format) plus `baked/manifest.json` (frames, fps, loop flag, frame data) — the
exact inputs Stage 1 certification (`certify.mjs --clips`) and the Stage 3
runtime consume. Generation is seconds per seed on a consumer GPU; a full
17-move × 8-seed run is roughly a coffee break.

The whole rebake ritual — regenerate changed moves, re-bake the spec (post
edits, trims, loop flags), re-certify, re-prebake — is one command:

```bash
python pipeline.py --spec your_moves.json --moves slide --seeds 8 --groundfix \
    --char hero.glb --practical ~/src/motionbricks-practical
```

Every stage is one of the plain tools above (run any of them by hand); a
failed certification aborts the chain before prebake. Without `--char` the
chain stops after the bake with fresh clips + manifest.

## 6. The end-to-end path

```
rigged character GLB                                (yours / Mixamo / Tripo / Meshy …)
        │
        ▼
node certify.mjs character.glb --clips baked/idle_stance.json,baked/walk_fwd.json
        │  certificate JSON, exit 0                 (ALIGN.md — Stage 1)
        ▼
xvfb-run -a python movegen.py --spec your_moves.json && python bake_moves.py
        │  baked/*.json + manifest.json             (BAKE.md — Stage 2, this doc)
        ▼
runtime: Retargeter per clip, state machine, root-motion integration
                                                    (INTEGRATE.md — Stage 3)
```

The two halves are independent: certification needs *some* baked clips to
probe with (generate the example set once and reuse it for every character),
and generation never needs a character at all.

## 7. Extending

**New poses** (BAKE.md §2): the starter library covers a basic combat set.
To mine more, point `POSEKIT_CSV_DIR` at a directory of G1-retargeted LAFAN1
mocap CSVs (root pos + XYZW quat + 29 dof per row — e.g. the `g1/` folder of
<https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset>), then:

```bash
python posekit.py scan dance1_subject1.csv        # heuristic candidates per tag
python posekit.py sheet dance1_subject1.csv --frames 120,340,560   # look first!
python posekit.py save spin_apex dance1_subject1.csv 340
python posekit.py wrists spin_apex --left 65,0,0 --right -65,0,0
```

Wrist angles are always authored, never mined — see the note in `posekit.py`
and BAKE.md §2d.

**New moves** (BAKE.md §3): add entries to your spec JSON. `"type":
"keyframes"` chains library poses (`tokens` = chunk length in 4-frame tokens,
6–16; `dxy` = root displacement in meters, forward/left; bookend every move
in one shared contact pose — BAKE.md §3). `"type": "mode"` rolls out a native
clip-holder skill — the mode list is **checkpoint-dependent**, and
`probe_api.py` (its `CLIPS:` line) is the source of truth. The shipped
checkpoint carries walk variants only (`idle`, `slow_walk`, `walk`,
`walk_left/right`, stealth/injured/zombie/... walks) — there is **no `run`
mode**.

**Missing locomotion verbs** (a run cycle, a sprint) are generated as
keyframe moves from mined poses instead — the recipe that produced a genuine
2.9 m/s run cycle in the field: mine the two contact poses of a LAFAN1 sprint
(`sprint1_subject2.csv`), then chain them with per-step displacement matching
the real stride:

```jsonc
{"name": "run_loop", "type": "keyframes", "start": "run_A", "loop": true,
 "steps": [{"pose": "run_B", "tokens": 6, "dxy": [1.6, 0.0]},
           {"pose": "run_A", "tokens": 6, "dxy": [1.6, 0.0]},
           {"pose": "run_B", "tokens": 6, "dxy": [1.6, 0.0]},
           {"pose": "run_A", "tokens": 6, "dxy": [1.6, 0.0]}]}
```

`dxy` is what sets the true speed (1.6 m per 6-token step ≈ 2.9 m/s at
30 fps) — the prior alone will not run; it needs the keyframes to pull it.

**Different canonical skeleton entirely**: everything in ALIGN.md/BAKE.md/
INTEGRATE.md is stated as data contracts, so another source works — but you
inherit the job MotionBricks solves here: quality-gated, keyframe-controllable
motion on a fixed skeleton. Port `movegen.py`'s gate battery regardless; the
gates are what make generation shippable.
