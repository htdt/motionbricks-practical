# Kimodo — installing and running the human-skeleton motion generator

The second motion source this pipeline ships working tooling for — and the
recommended one since its release — is **NVIDIA Kimodo**
(<https://research.nvidia.com/labs/sil/projects/kimodo/>), a text- and
constraint-conditioned motion diffusion model trained on 700 hours of studio
mocap (Bones Rigplay). Unlike MotionBricks (MOTIONBRICKS.md), which generates
on the Unitree G1 *robot* skeleton, Kimodo generates on **SOMA — a full human
skeleton** (77 joints incl. head, neck, fingers, toes), which removes the G1
path's structural gaps in one move: no more synthesized head channels, no
amplitude timidity (`yLift` hacks), no wrist-noise authoring discipline, and
its training data explicitly covers **videogame combat**.

Why it fits the pipeline: generation is *offline*, output is joint positions
+ rotations on one canonical skeleton, and its constraint system (full-body
keyframes at arbitrary frames, end-effector targets, 2D root paths) is a
superset of the keyframe conditioning BAKE.md's design needs. Where the
MotionBricks flow *requires* mined keyframes (its only steering channel is
the inbetweening API), Kimodo flips the priority: **text prompts do the
authoring** ("A person throws a rising uppercut punch, then returns to the
fighting stance") and keyframe constraints are reserved for what prompts
can't pin down — e.g. bookending every move with THE stance pose so clips
chain in-game.

Licensing mirrors MotionBricks: Apache-2.0 code, weights under the NVIDIA
Open Model License (commercial use OK, outputs are yours). The SMPL-X variant
is under the more restrictive R&D license — use the SOMA models.

## 0. Requirements

- Linux, CUDA GPU. The diffusion model itself is small (**≈2.5 GB VRAM
  peak**); the 17 GB VRAM figure in Kimodo's README is the Llama-3-8B-based
  text encoder, which runs fine on CPU (`TEXT_ENCODER_DEVICE=cpu`) — an
  RTX 3060 12 GB does the whole MK move-set without breaking a sweat.
- ≥ 20 GB free RAM if the text encoder runs on CPU (bf16 8B model),
  or a ≥ 17 GB-VRAM GPU to keep everything on-device.
- Python 3.10+, PyTorch 2.0+ (CUDA build), cmake (the MotionCorrection
  post-process extension builds at install).
- ~35 GB disk: 16 GB Llama text encoder, ~3 GB Kimodo weights, venv.

## 1. Install

```bash
git clone https://github.com/nv-tlabs/kimodo
python3 -m venv kimenv && kimenv/bin/pip install cmake ninja
kimenv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
cd kimodo
# --no-build-isolation so the build sees the venv's cmake; install build
# backends its git dependencies need first
../kimenv/bin/pip install hatchling scikit-build-core
PATH=$PWD/../kimenv/bin:$PATH ../kimenv/bin/pip install --no-build-isolation -e ".[all]"
```

Kimodo model weights auto-download from Hugging Face on first generation
(`nvidia/Kimodo-SOMA-RP-v1.1` — use this one; the SEED variants are
benchmark-reduced, the G1 variant reintroduces the robot skeleton).

### 1a. The gated text encoder

Kimodo's text conditioning is LLM2Vec on **`meta-llama/Meta-Llama-3-8B-Instruct`,
which is HF-gated**. Two ways through:

1. Your HF account has Llama-3 access → `hf auth login` and everything just
   works.
2. No access → build a local `TEXT_ENCODERS_DIR` from the public
   byte-identical `NousResearch/Meta-Llama-3-8B-Instruct` mirror plus the two
   (ungated) McGill-NLP LLM2Vec adapter repos, with `adapter_config.json`
   re-pointed at the local base. `kimodo/setup_text_encoder.py` in this repo
   automates the layout; then export
   `TEXT_ENCODERS_DIR=<...>/text_encoders` for every Kimodo command.
   Keep `_name_or_path = meta-llama/Meta-Llama-3-8B-Instruct` in the local
   config — LLM2Vec keys its prompt formatting off that string.

### 1b. Run the text encoder as a service

Loading 16 GB of Llama per CLI call is the dominant cost of naive batch
generation. Start the encoder **once** as a service; every Kimodo process
finds it automatically (mode `auto` probes `http://127.0.0.1:9550`):

```bash
TEXT_ENCODER_DEVICE=cpu TEXT_ENCODERS_DIR=... kimenv/bin/kimodo_textencoder &
```

With the service warm, a full 100-step generation of a 2.5 s move × 8
samples takes **~30 s on an RTX 3060** (≈22 diffusion it/s); the text encode
adds a few seconds per unique prompt on CPU.

## 2. Sanity-check generation

```bash
TEXT_ENCODER_DEVICE=cpu kimenv/bin/kimodo_gen "A person walks forward." \
    --model Kimodo-SOMA-RP-v1.1 --duration 3.0 --seed 42 --output walk
```

The output NPZ contains `posed_joints [T,77,3]`, `global_rot_mats
[T,77,3,3]`, `local_rot_mats`, `root_positions`, `global_root_heading
[T,2]`, `foot_contacts [T,4]` at 30 fps.

**Conventions** (verified empirically — `kimodo/validate_axes.py` re-checks
them on any NPZ): Y-up, ground at y=0, meters, root = `Hips` at ~0.95 m
standing; heading angle t stored as `[cos t, sin t]` with facing direction
`(sin t, 0, cos t)` — heading 0 faces **+Z**. Note this is already
three.js-style axes: no basis conversion at bake time (the G1 path's
MuJoCo-Z-up shuffle is gone), only a yaw canonicalization to the baked-clip
convention (forward = +X).

**The rest-pose trap (cost one shipped defect — read this).** In Kimodo NPZ
outputs, the T-pose is the zero pose: `global_rot_mats` of *every* joint is
the **identity** at rest. The `standard_t_pose_global_offsets_rots.p` asset
in the repo is a *different* (BVH-export) convention — baking it as the
clips' `restQuat` gives garbage orientation anchors, which shipped as
visibly skewed fists and toes-up feet. Two facts to bake by:

- **Feet**: identity rest is also the correct *semantic* anchor — at a
  generated standing frame the foot's global rotation returns to ~identity
  (measured 0.1°). The T-pose toe droop (~14°) is anatomy present in every
  frame; do NOT "level" it (that injects exactly 14° of toes-up skew).
- **Hands**: one correction is needed — the SOMA T-pose hand bends ~18° off
  the forearm axis while game rigs bind hands straight, so the identity
  anchor hyperextends the character's wrist whenever the mocap wrist is
  straight. `bake_kimodo.py` pre-rotates the hand rest onto the forearm
  axis (absolute bend tracking). This is only valid because `retarget.js`
  maps wrist deltas through **world axes at rest/bind** (`handM`/`handT`),
  never through raw bone-local axes — source and character rigs disagree
  arbitrarily on bone frames.

All three mistakes are caught mechanically by `qa_endeffectors.mjs` (§4) —
run it after any bake or retargeter change.

## 3. The tooling in `kimodo/` (this repo)

BAKE.md's contract, reimplemented for a text-first generator:

| file | role |
|---|---|
| `kimogen.py` | move-set generation: prompt + optional stance-bookend fullbody constraints → best-of-8 → QA gates → NPZ + gate/frame-data JSON per move |
| `moveset_mk.json` | the validated 17-move Mortal Kombat spec (prompts, travel intents, apex gates, bookend flags) |
| `bake_kimodo.py` | canonicalized NPZ → browser motion JSON (+`srcMap`) + manifest — the Kimodo `bake_moves.py` |
| `validate_axes.py` | prints/verifies the axis & heading conventions on any output NPZ |
| `setup_text_encoder.py` | builds the local mirror `TEXT_ENCODERS_DIR` (§1a) |

```bash
# 1. generate the idle first, unconstrained
TEXT_ENCODERS_DIR=... python kimogen.py gen --spec moveset_mk.json --only idle_stance
# 2. its medoid frame becomes THE stance pose (constraint + gate reference)
python kimogen.py stance
# 3. everything else, bookended to that stance
TEXT_ENCODERS_DIR=... python kimogen.py gen --spec moveset_mk.json
python kimogen.py report          # gate table
# 4. bake for the runtime (INTEGRATE.md)
python bake_kimodo.py --web ../web/moves_kimodo
```

### What changed vs the MotionBricks flow, stage by stage

| BAKE.md stage | MotionBricks (G1) | Kimodo (SOMA) |
|---|---|---|
| pose library | mined from LAFAN1→G1 CSVs (`posekit.py`) | one stance pose, extracted from Kimodo's own idle generation |
| move authoring | keyframe schedule per move (only control channel) | text prompt per move; fullbody keyframes only as bookends |
| wrists | authored channels, model output discarded | model output used as-is (mocap-clean hands + real fingers) |
| duration | pinned in 4-frame tokens (24-frame floor) | seconds, free (10 s max per prompt) |
| cleanup | external contact-lock IK required (teacher skates 0.34 m/s) | built-in post-process (`MotionCorrection`); measured skate 0.003–0.03 m/s |
| gates | keyframe-hit error, skate, jitter, limits | travel intent, apex (kick height / root rise/dip/floor), stance-match at both ends, skate, jitter |
| frame data | derived from keyframe-arrival frames | derived from strike-limb tip speed peaks |

The stance-bookend trick replaces the G1 flow's "every move starts and ends
on the stance keyframe" design rule: constrain frames `[0, T-1]` to the
stance pose (somaskel30 axis-angle + root height, authored in Kimodo's
native +Z frame), CFG `separated (2.0, 2.0)`. Prompts that end "...then
returns to the fighting stance" plus the constraint give end poses within
0.05–0.2 m of the reference — inside a 5-frame crossfade's coverage.

## 4. Retargeting Kimodo output (Stage 1 / Stage 3 hookup)

`retarget.js` ships a `SOMA_SRC` source map (the SOMA-77 joints are
anatomical, so limb roles are 1:1) and the `Retargeter` accepts the map from
the clip itself: `bake_kimodo.py` writes `srcMap` into every motion JSON, so
runtimes don't branch on the source family — a G1 clip and a SOMA clip play
through the same code. Rest pose comes from the SOMA standard T-pose assets,
ground-lifted and yawed to face +X like every canonicalized clip. Everything
in ALIGN.md (certification) and INTEGRATE.md (runtime) applies unchanged;
`yLift`-style amplitude hacks become no-ops (human clips jump for real).

**End-effector fidelity gates** — `qa_endeffectors.mjs <char.glb> <movesDir>
--gate` retargets every baked clip headless and measures, on the character:
median foot pitch vs bind on frames where the *source* foot is grounded and
still (flat by definition; gate ≤ 8° per clip, ≥15 contact frames), and
median |character wrist bend − source wrist bend| (knuckle direction vs
forearm axis, gate ≤ 30° per clip cap / ≤ 12° aggregate). This is the gate
that catches rest-anchor mistakes — a wrong rest convention reads as 15–90°
medians on *every* clip, unmistakably. Wire it into the project test suite
next to the game QA; run it for every new character × move-set pair.

## 5. Known limits

- ≤ 10 s per prompt; ≤ 20 constrained frames per constraint type.
- Prompts outside the training distribution (locomotion, gestures, everyday
  actions, object interaction, **videogame combat**, dance, stylized walks)
  degrade fast — see the BONES-SEED prompt style for calibration; "A person
  ..." phrasing works best.
- The model won't reliably invent multi-phase choreography from one prompt —
  chain prompts (multi-prompt generation) or split into separate moves.
- Post-processing (foot-skate cleanup + constraint optimization) is on by
  default and should stay on; it is what turns near-miss constraints into
  hits.
