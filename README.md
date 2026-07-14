# kimodo-practical — Kimodo → game animations, end to end

A practical extension of [NVIDIA Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/):
the manual and tooling for turning its generated motion into shipped game
animations for **any humanoid character, in any engine**. Motion is generated
once on the canonical skeleton, quality-gated, retargeted onto a **certified**
character rig, and shipped as ordinary baked clips. The game runtime contains
zero neural code; gameplay consumes generated frame data and root motion rather
than per-clip hand-tuned timing or displacement constants.

Written to be followed end-to-end by an autonomous coding agent (or a human).
It is engine-agnostic and game-agnostic: the reference implementation is
three.js, and it has been validated by building a complete two-player fighting
game from generated motion, but every stage states its contract in plain data
(JSON formats, gate thresholds, state-machine shapes) that ports anywhere.

## The pipeline

```
 Stage 1 — ALIGN            Stage 2 — BAKE                Stage 3 — INTEGRATE
 certify the rig            author clips on the           play baked clips:
 (rigmap roles, probe       canonical skeleton:           state machine, root-
 battery, gates,            prompts → generation →        motion integration,
 certificate JSON)          QA gates → best-of-N →        crossfades, frame-data
                            canonicalize + frame data     gameplay, QA harness
        │                          │                              │
   ALIGN.md                  BAKE.md + KIMODO.md             INTEGRATE.md
```

The motion generator is **NVIDIA Kimodo** — a text- and constraint-conditioned
motion diffusion model over the SOMA human skeleton (77 joints), trained on
700 hours of studio mocap that explicitly covers videogame combat. It is used
entirely offline; the model never ships with the game. KIMODO.md covers
download, install, conventions, and running the generation tools in `kimodo/`
end to end. The wrapper exposes Kimodo's **full authoring surface**: text
prompts, arbitrary full-body keyframes, hand/foot end-effector targets, sparse
root waypoints and dense paths, any compatible combination, and
constraint-only generation (no text) — each authored constraint is validated
before the model loads, gated for adherence after generation (hand/foot
targets to 5 mm/2° on the source skeleton), carried through canonicalization
and baking as resolved provenance records, and landed exactly on the final
character by a deterministic constraint IK (p95 gates: 2 cm/5° end to end).

Two principles carry the whole design:

1. **Don't try to make retargeting always right — make it impossible to ship
   wrong.** Stage 1 is a certification battery, not a fixed mapping: absolute
   gates + round-trip metrics, pass-or-reject. An uncertified character never
   enters animation production.
2. **Clips are character-agnostic.** Everything in Stage 2 targets the
   canonical skeleton, not a character. A new character costs one
   certification run (seconds); a new move costs one generation run. The two
   never multiply.

## What's in this repo

| File | What it is |
|---|---|
| `ALIGN.md` | Stage 1 manual: rig resolution, retargeting, certification battery |
| `BAKE.md` | Stage 2 manual: move specs, prompts + the full constraint schema, generation gates, baking |
| `KIMODO.md` | Stage 2 setup: download/install/run the Kimodo generator, its conventions and traps |
| `INTEGRATE.md` | Stage 3 manual: runtime layers, root motion, combat timing, QA |
| `ANIMATION_AGENT.md` | one-page decision guide for authoring agents: which control (text / pose / end-effector / waypoints / path / combinations) to use when |
| `rigmap.js` | bone → canonical-role resolution for arbitrary humanoid rigs |
| `retarget.js` | the two-skeleton position-based retargeter (unguarded, deterministic; the guard ablation and removals: `evidence/README.md`) |
| `ik.js` | target-space constraint IK: analytic two-bone solve that lands authored hand/foot targets exactly on any certified rig, deterministic under seek/speed/bake |
| `align.js` | probe mining, inverse recovery, gates, `certifyRig` |
| `glbskel.mjs` | GLB → bone hierarchy + animation sampler in node (no browser) |
| `certify.mjs` | certification CLI, writes `<char.glb>.retarget_certificate.json` |
| `prebake.mjs` | Stage 3 pre-bake for non-three.js engines: character GLB + baked manifest → new GLB with one glTF animation per clip + `rootmotion.json` (INTEGRATE.md §9) |
| `qametrics.mjs` | shared measurement machinery: captures, fidelity/accuracy/kinematics/clearance/skate metrics |
| `qa_constraints.mjs` | stage-separated constraint accuracy gates: authored target → final SOMA → unguarded rig → shipped rig, with determinism, branch-flip, and contact-skate gates; JSON + table output |
| `qa_endeffectors.mjs` | perceptual end-effector gates: foot flatness + wrist-bend tracking (raw transfer, with the stylization delta reported separately) |
| `ablate.mjs` | transfer-modifier ablation runner — the evidence generator behind keep/delete decisions |
| `selftest.mjs` / `selftest_constraints.mjs` | zero-asset self-tests (synthetic rigs, procedural motion, IK, determinism, sabotage cases) |
| `evidence/` | recorded ablation/QA evidence + reproduction commands for every guard decision |
| `kimodo/` | Stage 2 generation tools (run against a Kimodo install): `kimogen.py` (text + full constraint authoring + adherence gates), `kimoconstraints.py` (schema validation, conflict detection, FK resolution, canonicalization), `bake_kimodo.py`, the validated MK move spec, the deterministic e2e constraint suite (`make_e2e_spec.py`, `run_e2e.sh`), axis validator, text-encoder setup |

The JS scripts are the complete Stage 1 implementation and the runtime
retarget layer used in Stage 3; the Python tools in `kimodo/` are the complete
Stage 2 implementation. The pipeline is source-agnostic at its waist: every
baked clip carries a `srcMap` (canonical role → source joint name) and the
retargeter drives any certified rig from it with no runtime branching, so a
different motion source — a mocap library, another text-to-motion model,
hand-keyed animation — only has to meet the plain-data clip contract stated
in BAKE.md.

## Quick start

```bash
npm install        # three + @gltf-transform/core, nothing else
npm test           # selftests: synthetic certification + constraint IK/determinism, no assets
```

Certify a character against motion clips (format in ALIGN.md):

```bash
node certify.mjs character.glb --clips baked/idle_stance.json,baked/walk_fwd.json
# → character.glb.retarget_certificate.json, exit 0 = certified
```

The clips come from the generator — KIMODO.md walks through producing a full
move set (`baked/*.json` + `manifest.json`) from scratch.

Retarget any motion source onto any humanoid GLB in the browser:

```js
import { loadGLBSkeleton, buildBoneOrder, Retargeter } from './retarget.js';

const { hips } = await loadGLBSkeleton(GLTFLoader, './character.glb', scene);
const orderedBones = buildBoneOrder(hips);
const bones = {}; orderedBones.forEach(b => bones[b.name] = b);
const rt = new Retargeter({ bones, orderedBones, hips, hipsParent: hips.parent, data: motion });
rt.applyFrame(f);   // pose the rig for frame f, call per render tick
```

## For agents: how to use this repo

You were probably sent here to add animated characters to a game. The order
of work is fixed and each stage gates the next:

1. Read **KIMODO.md**. Install the generator and produce the example move
   set — you need baked clips before you can certify anything.
2. Read **ALIGN.md**. Certify every character rig. If certification fails,
   fix or regenerate the rig — do not proceed with an uncertified character;
   every downstream artifact would be built on a broken mapping.
3. Read **BAKE.md**. Author the game's move set (prompts + constraints on the
   canonical skeleton), gate every generated clip numerically, then look at
   filmstrips before accepting. Output: canonicalized clips + a manifest +
   frame data.
4. Read **INTEGRATE.md**. Wire the clips into the game with the three-layer
   architecture (clip / entity / game). Build the deterministic QA harness
   *before* tuning gameplay — it is what makes the rest debuggable.

Throughout: prefer regenerating a failed artifact over patching around it at
runtime; every gate threshold in these docs was validated in practice — treat
a gate failure as a real defect, not noise.

## Assumptions & limits

- **Humanoids only**: two legs, two arms, one spine chain. Toes, shoulders and
  neck are optional and degrade gracefully; missing core roles are a hard fail.
- **Y-up, meters**, feet flat and pointing forward at bind pose. Uniform
  armature scale (0.01-scaled FBX→GLB exports are handled).
- Verified against Mixamo, Tripo3D, Meshy AI and UE-style rigs, driven from
  the Kimodo SOMA-77 human skeleton (clips carry their source map — `srcMap` —
  so any source meeting the clip contract plays through the same runtime);
  runs in the browser and node.

## License

MIT
