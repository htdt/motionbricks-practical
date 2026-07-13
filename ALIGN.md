# Stage 1 — ALIGN: rig resolution, retargeting, certification

Rig-agnostic humanoid **skeleton alignment**: resolve any humanoid rig's bones
to canonical roles, retarget motion between two arbitrary skeletons with a
position-based retargeter, and **certify** the correspondence with a probe-pose
battery — absolute gates + round-trip metrics — so a bad retarget is caught
before it ships, not after.

The certification mindset: **don't try to make retargeting always right — make
it impossible to ship wrong.** Certification is cheap (seconds) and rig
generation is usually cheap too (AI rigging services charge cents), so
pass-or-regenerate converges to effectively 100% certified. The invariant the
rest of the pipeline gets: *no character enters animation production without a
certificate* — twisted wrists and broken knees become pre-production
rejections, not shipped bugs.

```
any motion source (role-keyed positions + quats per frame)
        │  srcMap: canonical role -> source joint name
        ▼
   Retargeter ──────────────► posed target rig (THREE.Bone hierarchy)
        ▲                          │
   rigmap.js roles                 ▼
   (name patterns +       certification battery
    topology fallback)    gates + round-trip + certificate JSON
```

## Files

| File | What it is | Environment |
|---|---|---|
| `rigmap.js` | bone → canonical role resolution (Mixamo ± prefix, Tripo, UE, VRoid, Meshy naming; spine-chain walk; full topology fallback for unnamed bones; partial-chain completion) | browser + node |
| `retarget.js` | the two-skeleton retargeter (`Retargeter`, `loadGLBSkeleton`, `buildBoneOrder`, `SOMA_SRC`) | browser + node |
| `align.js` | alignment & certification: probe mining, inverse recovery, gates, calibration, `certifyRig` | browser + node |
| `glbskel.mjs` | GLB → `THREE.Bone` hierarchy + animation sampler without a browser (gltf-transform) | node only |
| `certify.mjs` | certification CLI, writes `<char.glb>.retarget_certificate.json` | node only |
| `selftest.mjs` | zero-asset self-test (synthetic rigs, procedural motion, sabotage case) | node only |

## Quick start

```bash
npm install            # three + @gltf-transform/core
npm test               # selftest: synthetic source→target certification, no assets

# certify a character against probe clips (role-keyed motion JSON, see format):
node certify.mjs character.glb --clips clipA.json,clipB.json
# → character.glb.retarget_certificate.json, exit 0 = certified
```

Pick probe clips that cover range of motion — a walk, a kick, a jump, a reach.
The battery mines its probe poses from whatever clips you give it, so clips
that never bend a knee can't certify knees. The source role map comes from
`--srcmap map.json`, or a `srcMap` field on the first clip, or defaults to the
SOMA skeleton.

Retarget in the browser (any humanoid GLB, any motion source):

```js
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { loadGLBSkeleton, buildBoneOrder, Retargeter } from './retarget.js';

const { hips } = await loadGLBSkeleton(GLTFLoader, './character.glb', scene);
const orderedBones = buildBoneOrder(hips);
const bones = {}; orderedBones.forEach(b => bones[b.name] = b);
const rt = new Retargeter({ bones, orderedBones, hips, hipsParent: hips.parent, data: motion });
rt.applyFrame(f);      // pose the rig for frame f; call per render tick
```

GLB→GLB in node (one rig's animation clip driving another rig):

```js
import { loadGLBBones, animationSampler } from './glbskel.mjs';
import { rigFromBones, srcMapFromRig, snapshotMotion, certifyRig, mineProbeFrames } from './align.js';
import { Retargeter } from './retarget.js';

const src = await loadGLBBones('walker.glb');
const srcT = rigFromBones(src.bones);
const sampler = animationSampler(src.doc, src.byNode, src.wrapper, 0);
const motion = snapshotMotion(srcT.orderedBones, f => sampler.apply(f / 30), N, 30);
const srcMap = srcMapFromRig(srcT.rig.map);

const tgt = rigFromBones((await loadGLBBones('fighter.glb')).bones);
const cert = certifyRig(tgt, mineProbeFrames([motion], srcMap), { srcMap });   // gates
const rt = new Retargeter({ ...tgt, data: motion, srcMap });                    // playback
```

## Motion data format

A motion source is a plain object of world-space joint data (Y-up meters):

```js
{
  names: [string],          // joint names (index-aligned with pos/quat rows)
  fps: 30, numFrames: N, mode: 'clip name',
  rest:     [[x,y,z], …],   // world joint positions at the source's rest pose
  restQuat: [[x,y,z,w], …], // world joint orientations at rest (optional*)
  pos:  [ [[x,y,z], …], … ],    // per frame × per joint
  quat: [ [[x,y,z,w], …], … ],  // per frame × per joint (optional*)
}
```

\* Without quats, feet ride the shin and hands ride the forearm; with quats you
get true ankle orientation (heel-strike/toe-off) and wrist articulation.

The `srcMap` names which joints play which canonical role
(`Hips, Chest, L/R UpLeg/Leg/Foot, L/R Arm/ForeArm/Hand`, plus optional
`*Anchor` roles for the hip/shoulder frame lines). Defaults to SOMA
(`SOMA_SRC`); `srcMapFromRig(rig.map)` builds one for any resolved GLB rig.

This format is the pipeline's **hourglass waist**: every motion source
(generative model, mocap library, another rig's clips) converts into it once,
and every character consumes it through one certified retarget. New sources
and new rig families each cost one adapter, never source × character.

## The retargeter (what "alignment" means here)

Position-based, body-frame-relative — knees/elbows point where the source's do
and cannot invert:

- **Limbs** aim each bone at its child joint along the source's direction,
  rebased on the transferred pelvis (legs) / chest (arms) frame so roll follows
  the body.
- **Body-frame deltas are yaw-rebased** onto the character's bind heading (as is
  root translation) — otherwise a source *pitch* becomes partial *roll* on any
  rig whose bind heading differs from the source's.
- **Feet** take the source ankle's true orientation, transferred
  pelvis-relative (valid because rest feet are flat/forward on both sides).
- **Hands** ride the character's forearm and add the source's own
  wrist-vs-forearm delta — anchored locally, so a T-pose bind vs a hanging
  source arm cannot cock the wrist ("odd fists"), and the deviation never
  approaches the 180° clamp-branch flip.
- **Root translation** scales by the functional **leg-length ratio**
  (hip→ankle drop), keeping planted feet planted across different leg-to-hip
  proportions.
- **Guards** (all optional, on by default): anatomical hand twist/swing clamps;
  a torso-capsule that arms can graze but not pass through; a **temporal
  continuity guard** (40°/frame cap on hands/forearms/feet, sequential playback
  only, auto-bypassed on random frame access) that suppresses aim-antipode
  flips; an optional ground clamp (`guards.ground`).

## Certification battery

`mineProbeFrames(clips, srcMap)` picks deterministic probe poses from real
clips (per clip: highest left/right foot = kick apex, lowest/highest pelvis =
crouch/jump, widest hand span = reach, largest hip-vs-shoulder yaw = twist,
plus the source rest pose). `certifyRig(target, probes, { srcMap })` then
measures:

| Gate | Default | Catches |
|---|---|---|
| `sideConsistency` | must hold | mirrored/swapped role maps — these are perfectly *cycle-consistent*, so an absolute anchor (toe direction at bind) is required |
| `boneStretchPct` | ≤ 1 % | scale/FK corruption (aim transfer preserves lengths by construction) |
| `footFlatDeg` | ≤ 6° | grounded feet tilting differently than the source's own sole tilt |
| `footGroundFrac` | ≤ 0.10 | planted feet sinking/floating: wrong root scale, leg-proportion errors (also cycle-invisible) |
| `twistFrac` | ≤ 1.0 | per-bone twist beyond anatomical limits (`TWIST_LIMITS`: UpLeg 100°, Leg 95°, Arm 130°, ForeArm 165°, Hand 92°) — candy-wrapper artifacts |
| `capsuleClearance` | ≥ 0.85 | limbs passing through the torso |
| `roundTripMean` / `P95` | ≤ 0.05 / 0.10 m | transfer losses measurable by inverting the map: posed target → recovered source-skeleton joint positions vs the actual source frame |

The **inverse** (`recoverCanonicalPose`) uses only static calibration (bind
frames, source segment lengths, root scale, yaw rebase): chain-root anchors ride
their body frame rigidly, limb segments invert the aim map. Round-trip error is
*necessary but not sufficient* — an invertible-but-wrong map scores zero, which
is exactly why the absolute gates above exist. The battery is validated by
sabotage tests: a mirrored map, a disabled yaw rebase, and a wrong root scale
must each FAIL certification (see `selftest.mjs`).

`calibrateHandClamp` measures the rest-pose wrist twist bias and tightens the
hand clamp per character; it runs inside `certifyRig` by default.

Certificates look like:

```json
{
  "ok": true,
  "rig": { "ok": true, "missing": [], "rolesResolved": 20, "spineChain": ["Spine02","Spine01","Spine"] },
  "scale": { "srcHipY": 0.8, "charHipY": 1.012, "scaleRoot": 1.16 },
  "calibration": { "biasTwistDeg": 0, "handClampDeg": { "twist": 85 } },
  "probes": [ { "tag": "walk:kickL@223", "roundTripMean": 0.012, "...": "…" } ],
  "gates": { "sideConsistency": true, "boneStretchPct": 0, "footFlatDeg": 0.3,
             "footGroundFrac": 0.036, "twistFrac": 0.67, "capsuleClearance": 0.99,
             "roundTripMean": 0.0139, "roundTripP95": 0.0416 },
  "thresholds": { "…": "…" }, "failures": []
}
```

## When certification fails

Failures map to three repair tiers, cheapest first:

1. **Role remap** — `rig.missing` non-empty or `sideConsistency` fired:
   oddly-named bones defeated the name patterns. Inspect `rig.map`, add or
   correct entries (rigmap accepts explicit overrides), re-run. Structural
   quirks (extra roll bones, split spines) usually land here too.
2. **Clamp/scale calibration** — `twistFrac` or `footGroundFrac` marginal:
   check the character binds with flat, forward-pointing feet and uniform
   armature scale; `calibrateHandClamp` handles wrist bias automatically.
3. **Regenerate the rig** — anything structural that survives 1–2. With AI
   rigging services a re-rig costs cents; a manual repair costs hours and
   produces a one-off. Regeneration is almost always the right call.

## Assumptions & limits

- **Humanoids only**: two legs, two arms, one spine chain; `rigmap` hard-fails
  on missing core roles (toes/shoulders/neck are optional and degrade
  gracefully).
- **Y-up, meters**, feet flat and pointing forward at bind. Uniform armature
  scale (0.01-scaled FBX→GLB exports are handled).
- The `Retargeter` reads the target's **bind pose at construction** — construct
  it on a freshly loaded (or `resetBindPose`-restored) skeleton. `certifyRig`
  handles this itself via the `bindPose` snapshot in `rigFromBones`.
- The continuity guard needs *sequential* `applyFrame` calls; any jump larger
  than one frame resets it (by design, so pose batteries measure raw transfer).
- The inverse recovers joint *positions*, not orientations — orientation
  correctness is covered by the foot-flat/twist gates instead.

## Tests

`npm test` runs the standalone selftest with no assets: a synthetic UE-style
source (+X-facing) is certified onto a synthetic Mixamo-style target
(+Z-facing, different proportions, 0.01 armature scale), and a
mirrored-role-map sabotage must fail. Extend the same pattern for your own
integration tests: certify your real rigs against your real clips and assert
gate values, then render one clip on multiple rig families side by side and
eyeball it — the battery catches transfer defects, your eyes catch aesthetic
ones.
