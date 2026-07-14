# ANIMATION_AGENT — choosing motion-authoring controls

Operational decision guide for an agent authoring moves through this
wrapper's move spec (`kimodo/kimogen.py`). Schemas, coordinate math, and
troubleshooting live in BAKE.md (spec + workflow), KIMODO.md (model,
conventions, gates), and ALIGN.md (rig certification) — this page only tells
you **which control to use when**. Every control listed here is implemented
and tested; `kimodo/test_constraints.py` validates this vocabulary.

## The controls

| control | move-spec field | use it for |
|---|---|---|
| **Text prompt** | `prompt` | motion intent: action, timing, mood, style — anywhere natural variation is acceptable |
| **Full-body target pose** | `constraints: [{"type":"fullbody", ...}]` | an exact key pose at a known frame: start/end bookends, in-betweening, matching another clip's pose |
| **Hand/foot end-effector target** | `left-hand` / `right-hand` / `left-foot` / `right-foot`, or `end-effector` + `joint_names` (exact tokens: `LeftHand`, `RightHand`, `LeftFoot`, `RightFoot`) | a spatial contact: grab, touch, plant, step, kick a known point, interact with an object — while the rest of the body stays free |
| **Root waypoints** | `root2d` with a few `frame_indices` | sparse navigation goals; Kimodo chooses the natural path between them |
| **Dense root path** | `root2d` with one entry per frame (+ optional `global_root_heading`) | a continuous trajectory that must follow a specific curve |
| **Combined constraints** | several objects in `constraints` | independent requirements from different families that must hold simultaneously |
| **Constraint-only generation** | omit `prompt` | spatially driven motion with no semantic text (supported by the installed model: an empty prompt is explicitly zeroed, never replaced by wording) |
| **`stance_bookend`** | `stance_bookend: true` | clips that must start and end in the shared stance — never duplicate those keyframes by hand |
| **Constraint file** | `constraints_file` (mutually exclusive with inline `constraints`) | reuse a JSON saved by the Kimodo demo/API verbatim |

## Selection rules

1. Use **text** for semantic intent. Never rely on wording for a measurable
   world-space requirement — gate it with a constraint instead.
2. Use the **narrowest constraint** that expresses the requirement: an
   end-effector target beats a full-body pose when only one hand or foot
   must hit a point.
3. Use a **full-body pose** only when the whole silhouette/joint arrangement
   matters.
4. Use **waypoints** for destinations; a **dense path** only when the entire
   route matters.
5. **Combine** controls only when each adds an independent requirement. The
   validator rejects duplicates and contradictions (same-frame double pins,
   root disagreements, impossible root speeds) — fix the conflict, never
   raise guidance weights blindly.
6. Keep sparse constraints **under 20 frames per type** (dense root paths
   exempt), keep post-processing on, and give the move enough `duration`
   for targets to be reachable (< 5 m/s root travel between pinned frames).
7. Vocabulary discipline: hand/foot **targets are authored constraints**;
   **foot contacts are model predictions** (QA evidence, never inputs);
   unconstrained limbs are predictions. Don't describe the three as
   equivalent.
8. For every generated move, state which controls you selected and why,
   then check the generation report's constraint-adherence gates and the
   motion-quality gates before accepting
   (`kimogen.py report`, `qa_constraints.mjs`, `qa_endeffectors.mjs`).

## Things the schema will enforce anyway

- Constraints are authored in Kimodo's **native frame**: Y-up, meters,
  heading 0 faces **+Z**, root starts near the XZ origin. (Accepted clips are
  canonicalized to +X later — the wrapper re-expresses your targets for you.)
- `frame_indices` are integers in `[0, duration·fps)`, sorted, unique.
- `fullbody` / end-effector constraints carry one **complete SOMA pose** per
  frame (`local_joints_rot` `[T,30|77,3]` axis-angle radians +
  `root_positions` `[T,3]`); the world target is the FK of that pose. An
  end-effector constraint **also pins the root XZ, root height, and heading**
  implied by its pose at those frames — co-framed constraints must agree.
- A `root2d` heading is a `[cos θ, sin θ]` pair per frame.
- Unreachable or contradictory authoring fails loudly, at validation when
  cheap (root speed, conflicts), else at the adherence gates.

## Minimal mixed example

```jsonc
{
  "name": "reach_while_walking",
  "duration": 3.0,
  "prompt": "A person walks forward and reaches for an object.",
  "constraints": [
    { "type": "root2d", "frame_indices": [0, 45, 89],
      "smooth_root_2d": [[0.0, 0.0], [0.4, 0.2], [0.9, 0.2]] },
    { "type": "right-hand", "frame_indices": [60],
      "local_joints_rot": [[ /* complete 30-joint axis-angle pose */ ]],
      "root_positions": [[0.55, 0.95, 0.2]] }
  ]
}
```

(Real complete pose arrays: see `kimodo/moveset_e2e.json`, generated from the
upstream demo poses by `make_e2e_spec.py`.)

## Input/output checklist

Before generating: frame indices in range and sorted · Y-up meters · native
+Z authoring frame · root speed between pins < 5 m/s · no same-frame double
pins · sparse frames per type < 20 · duration long enough to reach targets.

After generating: `accepted: true` in `out/moves/<move>.json` · adherence
gates green (`constraints.summary`: EE pos ≤ 5 mm, EE rot ≤ 2°, root XZ
≤ 2 cm) · `<move>.resolved_constraints.json` present (canonical targets for
runtime/QA) · then bake and run `qa_constraints.mjs <char> <movesDir> --gate`
per character.
