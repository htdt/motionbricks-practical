# Kimodo — installing and running the motion generator

The pipeline's motion source is **NVIDIA Kimodo**
(<https://research.nvidia.com/labs/sil/projects/kimodo/>), a text- and
constraint-conditioned motion diffusion model trained on 700 hours of studio
mocap (Bones Rigplay), generating on **SOMA — a full human skeleton**
(77 joints incl. head, neck, fingers, toes). Its training data explicitly
covers **videogame combat**, locomotion, gestures and dance.

Why it fits the pipeline: generation is *offline*, output is joint positions
+ rotations on one canonical skeleton, and it is steerable on exactly the two
channels Stage 2 needs: **text prompts do the authoring** ("A person throws a
rising uppercut punch, then returns to the fighting stance") and its
constraint system (full-body keyframes at arbitrary frames, end-effector
targets, 2D root paths) pins down what prompts can't — e.g. bookending every
move with THE stance pose so clips chain in-game.

Licensing: Apache-2.0 code, weights under the NVIDIA Open Model License
(commercial use OK, outputs are yours). The SMPL-X variant is under the more
restrictive R&D license — use the SOMA models.

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
[T,2]`, and (after SOMA-77 expansion) `foot_contacts [T,6]` at 30 fps. The
generation gate also accepts Kimodo's compact four-contact layout.

**Conventions** (verified empirically — `kimodo/validate_axes.py` re-checks
them on a generated forward-walk NPZ): Y-up, ground at y=0, meters, root =
`Hips` at ~0.95 m
standing; heading angle t stored as `[cos t, sin t]` with facing direction
`(sin t, 0, cos t)` — heading 0 faces **+Z**. Note this is already
three.js-style axes: no basis conversion at bake time, only a yaw
canonicalization to the baked-clip convention (forward = +X).

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

**Forearm roll.** The aim-based limb transfer takes each bone's roll from
the body-frame rebase (chest delta for arms). With big torso pitches — a
deep jump crouch — the chest delta projects onto the forearm axis and the
forearm *spins in place* (measured: up to 45°/frame of twist with <5°/frame
of direction change; disabling the source path costs up to **179°** of roll
on the regenerated move set), taking the fist with it. Any clip carrying
quaternions therefore rebuilds the forearm from the source forearm's full
world delta — true mocap pronation/supination, stable by nature — then
rotates minimally onto the aimed direction. This is automatic and is the
ONLY quaternion path; the old per-clip `foreRollSrc` switch was removed
(evidence/README.md).

**Wrist articulation gain (`handFollow`).** Even with correct anchors and
axes, raw mocap wrist channels read poorly on game hands: every twitch,
roll, and stylistic flex of the performer shows on a fingerless fist mesh as
a "broken" wrist (the balancing hand during a side kick was the reported
case). The clip JSON carries `handFollow` — the retargeter slerps between
the hand rigidly riding the forearm (0) and the full source wrist (1).
`bake_kimodo.py` bakes **0.3** for prediction-only clips: hands stay aligned
with the forearm, keeping a hint of wrist life. This is an intentional
STYLIZATION, not a correctness guard — measured, it discards 22.8° mean /
60.8° p95 of real wrist articulation, so QA reports raw transfer fidelity at
`handFollow = 1` and the stylization delta separately, and clips with
**authored hand constraints are baked at 1.0** (an exact target must never
be damped; the constraint IK applies the authored orientation exactly at
constrained frames).

**Deleted guards.** The retargeter used to carry an anatomical hand clamp
(85°/70°), a torso-capsule arm displacement, a 40°/frame temporal continuity
slew, and an optional ground lift. All four were ablated per-intervention on
the regenerated move set across both certified reference rigs and deleted:
the temporal guards never engaged (Δ = 0°) while making poses depend on
playback history; the clamp clipped VALID authored wrists by up to 15.5°;
the capsule displaced valid near-face guard poses with zero measured torso
penetration in the unguarded baseline. The transfer is now unguarded and
deterministic; torso clearance, ground penetration, branch flips, and
contact-frame skate are measured by QA instead of silently corrected.
Numbers and reproduction commands: `evidence/README.md`.

## 3. The tooling in `kimodo/` (this repo)

The Stage 2 implementation (contract and workflow: BAKE.md):

| file | role |
|---|---|
| `kimogen.py` | move-set generation: text + any compatible constraint combination → best-of-8 → QA gates incl. per-constraint adherence → NPZ + gate/frame-data JSON + resolved canonical constraint records per move |
| `kimoconstraints.py` | the constraint layer: upstream-schema validation, end-effector normalization, conflict detection, stance-bookend merging, FK resolution of authored targets, canonical-transform + loop-trim handling |
| `moveset_mk.json` | the validated 17-move Mortal Kombat spec (prompts, travel intents, apex gates, bookend flags) |
| `moveset_e2e.json` / `make_e2e_spec.py` | the deterministic GPU end-to-end constraint suite: one move per authoring mode, poses lifted from the upstream demo files |
| `run_e2e.sh` / `e2e_check.mjs` | reproducible e2e driver + expected pass/fail matrix (incl. the deliberate failures) |
| `bake_kimodo.py` | canonicalized NPZ → browser motion JSON (+`srcMap`, foot-contact predictions, resolved constraint records) + manifest |
| `test_constraints.py` / `test_tools.py` | fast model-free tests: every constraint family, invalid cases, conflicts, transforms, trims, adherence math, the agent-guide vocabulary |
| `validate_axes.py` | validates the axis & heading conventions on a forward-walk NPZ |
| `setup_text_encoder.py` | builds the local mirror `TEXT_ENCODERS_DIR` (§1a) |

Constraint authoring — which control to use when: **ANIMATION_AGENT.md**;
schema, validation and conflict rules: **BAKE.md §3**. `kimogen.py` passes
`root_margin=0.01` to Kimodo's post-processing: the upstream default (0.04 m)
leaves corrected roots exactly 4 cm off authored waypoints, outside the 2 cm
adherence gate (`--root-margin 0` = exact).

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

### The stance bookend, mechanically

The bookend constraint (BAKE.md §2) pins frames `[0, T-1]` to the stance
pose as a fullbody keyframe constraint — somaskel30 axis-angle + root
height, authored in Kimodo's **native +Z frame** (constraints are applied
before canonicalization, so a stance extracted from a canonicalized clip
must be de-rotated back). CFG `separated (2.0, 2.0)`.

**Prediction vs conditioning vs correction, measured.** An unconstrained
hand/foot (and every foot-contact label) is a model PREDICTION. An authored
constraint CONDITIONS the prediction through the diffusion guidance, and
built-in post-processing (`MotionCorrection`) then CORRECTS the result onto
the target: measured on the regenerated sets, fullbody keyframes and
end-effector targets land at **0.0000 m / 0.00°** and root waypoints at
exactly the configured `root_margin` (kimogen: 0.01 m; foot skate after
correction: 0.003–0.03 m/s). The four shorthand EE types, fullbody, and
root2d all get correction masks; a generic `end-effector` constraint would
condition but NOT be corrected, which is why the wrapper normalizes it into
shorthands. Each baked record carries this provenance
(`conditioned+corrected`).

## 4. Retargeting Kimodo output (Stage 1 / Stage 3 hookup)

`retarget.js` ships the `SOMA_SRC` source map (the SOMA-77 joints are
anatomical, so limb roles are 1:1) and the `Retargeter` accepts the map from
the clip itself: `bake_kimodo.py` writes `srcMap` into every motion JSON, so
runtimes don't branch on the source family — any source meeting the clip
contract plays through the same code. Rest pose comes from the SOMA standard
T-pose joint positions, ground-lifted and yawed to face +X like every
canonicalized clip. Everything in ALIGN.md (certification) and INTEGRATE.md
(runtime) applies unchanged.

**Constraint accuracy gates** — `qa_constraints.mjs <char.glb> <movesDir>
--gate` measures every stage separately so a later stage cannot hide an
earlier failure: (1) authored target → final SOMA output (pos ≤ 5 mm, rot ≤
2°, root XZ ≤ 2 cm, from the baked records), (2) SOMA → UNGUARDED character
transfer (full-quaternion + swing/twist fidelity at `handFollow = 1`, plus
round-trip recovery), (3) the delta each retained style modifier introduces
(and whether it touches an authored constrained frame — reported, never
silent), (4) final character vs the mapped authored target through the
constraint IK (pos p95 ≤ 2 cm / max ≤ 4 cm, rot p95 ≤ 5° / max ≤ 10°;
geometrically unreachable targets are clamped explicitly, reported, and fail
unless the move declares `reach_policy: "clamp"`). Plus: no NaN, no invalid
quaternion, no one-frame branch flip, no new contact-frame foot skate (using
the stored predicted contact channels), and sequential == direct-seek
determinism. Emits machine-readable JSON (`--json`) and a table; a metric
that cannot be computed is a failure, never a skip.

**Perceptual end-effector gates** — `qa_endeffectors.mjs <char.glb>
<movesDir> --gate` measures, on the character: median foot pitch vs bind on
frames where the *source* foot is grounded and still (flat by definition;
gate ≤ 10° per clip, ≥15 contact frames), and median |character wrist bend −
source wrist bend| at RAW transfer (knuckle direction vs forearm axis, gate
≤ 40° per clip cap / ≤ 15° aggregate), with the `handFollow` stylization
delta reported separately. This catches rest-anchor mistakes — a wrong rest
convention reads as 15–90° medians on *every* clip, unmistakably. Run both
tools for every new character × move-set pair.

## 5. Known limits

- ≤ 10 s per prompt; < 20 constrained frames per constraint type (dense
  root2d paths exempt) — the wrapper enforces both before loading the model.
- Constraint-only generation (no text) is supported by the installed API —
  an empty prompt's text features are explicitly zeroed — and exposed by
  omitting `prompt` in the move spec.
- The training distribution covers locomotion, gestures, everyday actions,
  object interaction, **videogame combat**, dance, and stylized walks. Prompts
  far outside those categories degrade fast; plain "A person ..." phrasing
  works best.
- The model won't reliably invent multi-phase choreography from one prompt —
  chain prompts (multi-prompt generation) or split into separate moves.
- Post-processing (foot-skate cleanup + constraint optimization) is on by
  default and should stay on; it is what turns conditioned near-misses into
  exact hits (see §3's measured provenance).
