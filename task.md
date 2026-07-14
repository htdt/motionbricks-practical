# Task: complete Kimodo constraint authoring and end-effector fidelity

## Objective

Expose Kimodo's full motion-authoring capability through this repository and
make hand/foot accuracy measurable from authored constraint to final character.
Support text, arbitrary full-body keyframes, hand/foot end-effector targets,
root waypoints/dense paths, constraint-only generation where Kimodo supports it,
and arbitrary compatible combinations of those inputs.

At the same time, simplify the retargeter. The hand-orientation anchor bug was
fixed over several iterations, while clamps, damping, temporal smoothing, and
collision guards accumulated around it. Re-establish a correct unguarded
baseline, ablate every guard and transfer modifier, and remove anything that no
longer provides a measured benefit or that reduces constraint/source fidelity.

## Current gaps

- `kimogen.py` accepts a text prompt and can synthesize one special constraint:
  the shared full-body stance at frames `0` and `T-1`.
- Arbitrary full-body keyframes cannot be declared in the move spec.
- Hand/foot end-effector constraints cannot be declared in the move spec.
- Root waypoints and dense root paths cannot be declared in the move spec.
  The existing `travel` field is a result gate, not a generation constraint.
- Mixed constraint families are supported by Kimodo but are not passed through
  by this wrapper, except for text plus the stance bookend.
- Generated constraints are not preserved in a normalized, post-trim form for
  runtime use or end-to-end QA.
- Kimodo predicts hands, feet, their rotations, and foot contacts when they are
  unconstrained. For constrained end-effectors, an authored target conditions
  the prediction and `MotionCorrection` adjusts the result. The baked format
  does not record that provenance clearly and currently drops foot contacts.
- The retargeter transfers predicted source limb directions and hand/foot
  rotations, but does not solve a final target-character IK constraint. A SOMA
  wrist or ankle hitting its target therefore does not prove that a character
  with different proportions hits the mapped world-space target.
- `qa_endeffectors.mjs` checks foot pitch and wrist-bend transfer. It does not
  measure authored end-effector position/rotation error on SOMA or after
  retargeting.
- The hand QA multiplies the expected source bend by `handFollow`. This makes
  intentional damping look like perfect source transfer and cannot reveal how
  much fidelity the modifier discards.
- The current `--noguards` diagnostic disables only `handClamp`,
  `torsoCapsule`, and `continuity`; it does not provide a true raw-transfer
  baseline or isolate `handFollow`, `foreRollSrc`, and `ground`.
- The repository contains no generated move set and target-character assets,
  so the normal unit tests cannot certify a real character x move-set result.

## Required authoring support

### 1. Move-spec constraint schema

Add an upstream-compatible per-move constraint input. Support either an inline
JSON array or a path to a Kimodo constraints JSON file. Do not invent another
kinematic representation when Kimodo's existing schema can be used.

At minimum, accept:

- `fullbody`: arbitrary sparse full-body keyframes at arbitrary frames;
- `left-hand`, `right-hand`, `left-foot`, and `right-foot` shorthand;
- `end-effector`: any supported subset of left/right hands and feet;
- `root2d`: sparse waypoints and dense paths, with optional heading;
- multiple constraint objects, including multiple families in one move.

An inline example should be possible in a move:

```jsonc
{
  "name": "reach_while_walking",
  "duration": 3.0,
  "prompt": "A person walks forward and reaches for an object.",
  "constraints": [
    {
      "type": "root2d",
      "frame_indices": [0, 45, 89],
      "smooth_root_2d": [[0, 0], [0.4, 0.2], [0.9, 0.2]]
    },
    {
      "type": "end-effector",
      "frame_indices": [60],
      "joint_names": ["RightHand"],
      "root_positions": [[0, 0.95, 0]],
      "local_joints_rot": ["... complete SOMA pose used to derive the target ..."]
    }
  ]
}
```

Use the exact joint-name casing and expansion rules expected by the installed
Kimodo version. Examples and tests must contain valid complete arrays rather
than the illustrative placeholder above.

Allow `stance_bookend` to remain as a convenience that produces a `fullbody`
constraint and merges it with explicitly supplied constraints. Reject duplicate
or contradictory constraints with a useful move/frame/type diagnostic; never
silently let one overwrite another.

Verify and support constraint-only generation if the selected Kimodo model/API
supports it. Represent the absence of text explicitly rather than substituting
a motion-changing prompt. If the installed API cannot do it, fail clearly and
document that exact upstream limitation.

### 2. Validation and coordinate handling

Validate before loading the model:

- known constraint types and supported end-effectors;
- integer, in-range, sorted, unique frame indices;
- matching array lengths and exact SOMA-30/SOMA-77 shapes;
- finite positions, rotations, headings, and paths;
- valid rotations and nonzero heading vectors;
- the Kimodo guidance limit for sparse constrained frames (currently fewer
  than 20 per constraint type, except dense root paths);
- constraint reachability basics and conflicts that can be detected cheaply;
- mutually exclusive inline constraints and constraint-file fields.

Constraints are authored in Kimodo's native Y-up, meter, +Z-facing canonical
frame, while accepted clips are later canonicalized to +X and may be loop
trimmed. Use one tested transform path for positions, root heading, and global
rotations. Do not introduce hand/foot-specific axis corrections.

When a clip is trimmed, shift retained constraint frames, remove constraints
outside the retained interval, and reject a trim that removes a required
constraint. Recanonicalize the retained targets using the same transform as the
motion.

### 3. Preserve resolved constraint provenance

Store a normalized, baked representation of resolved targets in the motion
JSON and/or manifest. It must be directly usable by QA and runtime code without
reloading the authoring skeleton or repeating FK. For each constrained frame,
record as applicable:

- constraint family and original authoring source;
- canonical frame index;
- canonical root/path target and heading;
- canonical full-body target;
- end-effector role, world position, and world rotation;
- whether position and rotation are constrained;
- whether the final value came from free prediction, conditioned prediction,
  or post-processing correction.

Preserve Kimodo's foot-contact predictions in the accepted NPZ and baked clip
with an explicit joint/channel mapping. Treat contact labels as predictions and
QA/cleanup evidence, not as authored end-effector targets.

## End-effector accuracy contract

Measure the following stages separately. A later stage must not hide an earlier
failure.

1. **Authored target -> final SOMA output:** position and rotation error at every
   constrained hand/foot frame after Kimodo post-processing.
2. **SOMA output -> unguarded target rig:** position and rotation transfer error
   using the corrected rest anchors and world-axis mapping, with no clamp,
   damping, temporal smoothing, collision displacement, or ground adjustment.
3. **Unguarded -> shipped target rig:** the delta introduced by each retained
   guard or intentional style modifier.
4. **Final target -> mapped authored target:** end-to-end position and rotation
   error on the rendered character.

Report mean, median, p95, and maximum errors per role, clip, constraint type,
and character. Do not reduce hand orientation to wrist bend alone; compare full
quaternions as well as swing and twist. Keep the existing wrist-bend and foot-
pitch metrics as useful perceptual diagnostics.

Use the full source wrist demand as the fidelity reference. Report
`handFollow`/stylization error separately; never multiply the expected value by
`handFollow` in the raw transfer metric.

Initial hard gates for reachable authored targets:

- final SOMA end-effector position: max `<= 0.005 m`;
- final SOMA end-effector rotation: max `<= 2 deg`;
- final character end-effector position against the mapped target: p95
  `<= 0.02 m`, max `<= 0.04 m`;
- final character end-effector rotation: p95 `<= 5 deg`, max `<= 10 deg`;
- constrained root waypoint/path samples: max `<= 0.02 m` in XZ;
- no NaN, invalid quaternion, one-frame orientation branch flip, or new foot
  skate on contact frames.

If a target is geometrically unreachable on a character, clamp it explicitly,
mark it unreachable in the report, and fail the authored-target gate unless the
move opts into a documented reach policy. Do not count clamped targets as
accurate hits.

## Target-character IK

Add a target-space end-effector solve if measurement confirms that direction-
based retargeting alone misses reachable mapped targets.

- Map canonical SOMA targets through the same root-yaw, root-scale, and bind
  transform used by the retargeter.
- Use an analytic two-bone or equivalently deterministic solve for arms and
  legs, preserving the predicted elbow/knee pole direction.
- Apply authored hand/foot orientation after the positional solve.
- Define behavior when two or more effectors and/or a root target interact.
- Do not move a constrained root or already pinned end-effector as an implicit
  side effect of satisfying another target.
- Ensure exact constrained frames do not pop: derive deterministic blending or
  solve weights from constraint data, not playback history.
- Applying frame `f` must produce the same constrained pose whether frames are
  evaluated sequentially, sought directly, baked offline, or played at a
  different speed.
- If full-time IK following of the predicted hand/foot trajectories performs
  better than constrained-frame-only IK, it may be used, but the choice must be
  justified by the same ablation metrics and must not increase unconstrained
  motion artifacts.

For clips with authored hand-rotation constraints, do not apply a global
`handFollow: 0.3` after the exact orientation. Use full authored orientation at
the constrained frame, with any transition derived deterministically around it.

## Guard and transfer-modifier cleanup

### Establish the baseline

The baseline is the corrected orientation transfer, not the historical broken
anchor:

- normalized foot rest anchor;
- straight hand/forearm semantic rest anchor;
- source-to-character world-axis wrist mapping;
- valid source forearm roll and source ankle quaternion transfer;
- all optional clamps, smoothing, damping, collision displacement, and ground
  adjustment disabled.

Add a diagnostic mode that can reproduce this baseline exactly. Replace the
ambiguous `--noguards` behavior with an explicit configuration dump and an
ablation runner.

### Ablate every current intervention

Evaluate each intervention alone and in the shipped combination:

- `handClamp` (`85 deg` twist / `70 deg` swing);
- `torsoCapsule` arm-direction displacement;
- `continuity` and its `40 deg/frame` history-dependent slew cap;
- optional `ground` root lift and its temporal settling;
- `handFollow`, especially the baked value `0.3`;
- `foreRollSrc` as a conditional transfer path;
- any new IK reach clamp or constraint blending introduced by this task.

For every intervention, measure:

- hand/foot target position and full-orientation error;
- source-motion fidelity on unconstrained frames;
- angular velocity, angular acceleration, and branch flips;
- elbow/knee direction and limb hyperextension;
- torso penetration and ground penetration;
- contact-frame foot skate;
- determinism under sequential playback, direct seek, loop wrap, speed-up, and
  offline prebake.

Specifically test the hypothesis that `handClamp`, `continuity`, and
`handFollow` were compensating for the old hand-anchor error. Check whether:

- the clamp clips valid authored wrist orientations;
- continuity creates lag or misses exact keyframes;
- `handFollow: 0.3` discards valid source/authored motion;
- the torso capsule moves a constrained hand away from its target;
- ground lift violates root, foot, or full-body constraints;
- `foreRollSrc` can become the single correct quaternion path rather than a
  per-clip compatibility switch.

### Removal rule

Delete an intervention when it has no reproducible benefit on the representative
suite, duplicates a now-correct transfer path, makes results history-dependent,
or materially worsens target/source fidelity. Remove its option, state, docs,
and tests together; do not leave dead compatibility switches.

Retain a guard only when its failure class is still reproducible from valid
inputs after the anchor fix and its benefit is larger than its measured
degradation. Document the evidence, scope it to the smallest affected bones and
frames, and make sure it cannot modify an authored target silently. A conflict
between a safety guard and an exact target must be reported, not hidden.

Prefer deterministic geometric correctness to temporal filtering. Quaternion
sign normalization is acceptable; pose results that depend on which frame was
applied previously are not acceptable for constrained clips or offline baking.

## QA and reporting changes

- Extend generation reports with per-constraint adherence metrics and explicit
  post-processing status.
- Extend `qa_endeffectors.mjs`, or replace it with a more general constraint QA,
  to measure source-stage and retargeted-stage position plus full-rotation
  errors.
- Use stored foot-contact channels instead of only height/speed heuristics when
  they are present; retain the heuristic as a cross-check/fallback.
- Print raw baseline, each guard delta, and final result side by side.
- Fail if a metric cannot be computed; do not silently skip a constrained role
  or a character lacking a required measurement fixture.
- Keep perceptual metrics (foot pitch, wrist bend, torso clearance) separate
  from exact constraint-adherence metrics.
- Emit machine-readable JSON suitable for CI and a concise human table.

## Tests

### Fast tests without model weights

- Spec validation for every constraint family and invalid shape/value/frame
  case.
- Loading inline and file-based constraints.
- Merging stance bookends with arbitrary constraints and detecting conflicts.
- Mixed full-body + end-effector + root2d constraints.
- Constraint-only behavior.
- Native +Z authoring -> canonical +X conversion for positions, headings, and
  rotations.
- Loop trimming/frame shifting with retained and removed constraints.
- Constraint metadata and foot-contact bake round trips.
- Synthetic arm/leg targets on rigs with different bind headings, scales, limb
  proportions, bone-local axes, and missing finger bones.
- Known wrist rotations that previously reproduced skewed fists, broken wrists,
  and 90-180 degree flips.
- Direct seek and sequential playback produce identical constrained frames.
- A deliberately restored old rest anchor must fail the new gates.
- A deliberately over-damped, clamped, or history-smoothed result must fail raw
  source/constraint fidelity even if it looks stable.

### GPU/end-to-end tests

Generate deterministic examples for:

- text only;
- one arbitrary full-body target;
- each hand and foot independently;
- both hands, both feet, and hands + feet;
- sparse root waypoints and a dense curved path;
- text + pose + end-effectors + path in one move;
- constraints without text, if supported;
- reachable and deliberately unreachable targets.

Run the accepted outputs on at least two certified character rigs with different
bind headings, proportions, and hand skeleton detail. Include the complete
representative move set, not only hand-picked easy clips. Store the commands,
model version, seeds, reports, and threshold results needed to reproduce the
decision about every guard.

## Documentation

Update `README.md`, `BAKE.md`, `KIMODO.md`, `ALIGN.md`, and `INTEGRATE.md` to:

- list exactly which authoring modes the wrapper supports;
- show valid examples for every constraint type and a mixed example;
- explain native constraint coordinates and baked canonical coordinates;
- distinguish prediction, authored conditioning, post-processing, retargeting,
  and target-space IK;
- explain what foot contacts are and that they are predicted;
- state the final guard/modifier set and the ablation evidence for keeping it;
- remove obsolete continuity/guard playback instructions when the associated
  behavior is deleted;
- document all accuracy gates and how to run them for each character x move-set
  pair.

### Concise agent-facing authoring guide

Create a short, standalone instruction for an animation-authoring agent (for
example `ANIMATION_AGENT.md`). It is an operational decision guide, not another
implementation manual. Keep it to roughly one page and link to the detailed
manuals for schemas and troubleshooting.

The guide must list the controls that this wrapper actually implements and say
when to use each one:

- **Text prompt:** motion intent, action, timing, mood, and style where variation
  is acceptable.
- **Full-body target pose:** an exact key pose, start/end bookend,
  in-betweening, or a transition that must match another clip.
- **Hand/foot end-effector target:** a spatial contact such as grabbing,
  touching, planting, stepping, kicking a known point, or interacting with an
  object/environment while leaving the rest of the body free.
- **Root waypoint:** sparse navigation goals where Kimodo should choose the
  natural path between points.
- **Dense root path:** a continuous trajectory that must follow a specific
  curve.
- **Combined constraints:** requirements from different families that must hold
  simultaneously, using the smallest non-conflicting set of constraints.
- **Constraint-only generation:** spatially driven motion without semantic text,
  if the installed model/API supports it.
- **`stance_bookend`:** the convenience form for clips that must start and end
  in the shared stance; do not manually duplicate those keyframes.

It must also give the agent these selection rules:

1. Use text for semantic intent; do not rely on wording for a measurable world-
   space requirement.
2. Use the narrowest constraint that expresses the requirement. Prefer an
   end-effector target over a full-body pose when only a hand or foot must hit a
   point.
3. Use a full-body pose when the whole silhouette/joint arrangement matters.
4. Use waypoints for destinations and a dense path only when the complete route
   matters.
5. Combine controls only when each adds an independent requirement. Detect and
   reject contradictions rather than increasing guidance weights blindly.
6. Keep sparse constraints below Kimodo's supported per-type limit, keep post-
   processing enabled, and allow enough duration for targets to be reachable.
7. Treat hand/foot targets as authored constraints, foot contacts as model
   predictions, and unconstrained limbs as predictions; never describe all
   three as equivalent inputs.
8. For every generated move, state which controls were selected and why, then
   check the corresponding constraint and motion-quality gates before accepting
   it.

Include one minimal mixed example and a compact input/output checklist covering
frame indices, Y-up meters, Kimodo's native +Z authoring frame, reachability,
conflicts, post-processing, and the generated QA report. The guide must be
generated from or tested against the supported spec vocabulary so it cannot
claim features that the wrapper does not implement.

## Definition of done

- Every authoring mode above works through the move spec and can be combined.
- Constraint files saved by the compatible Kimodo demo/API can be consumed
  without manual rewriting.
- Resolved constraints survive canonicalization, trimming, baking, runtime
  loading, and QA with traceable provenance.
- Reachable authored hand/foot targets pass the SOMA and final-character
  position/rotation gates.
- Unreachable or conflicting targets fail explicitly with useful diagnostics.
- Raw transfer fidelity is measured independently of stylization and guards.
- Every existing guard and transfer modifier has a recorded ablation result;
  unnecessary or degrading mechanisms are removed completely.
- Constrained frames are deterministic under seek, playback, speed changes,
  looping, and prebake.
- Fast JavaScript/Python tests and the reproducible GPU/end-to-end suite pass.
- The concise agent-facing guide accurately lists every available control and
  gives an unambiguous rule for when to use it.
- Documentation reflects the implemented behavior rather than upstream Kimodo
  capability alone.
