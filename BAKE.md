# Stage 2 — BAKE: author, generate, gate, and bake motion clips

Everything in this stage happens on the **canonical skeleton** — the skeleton
of your motion source, not of any character. Clips authored here drive every
certified character forever; a new character never requires regenerating a
clip, and a new clip never requires touching a character.

The idea in one line: **author moves as text prompts plus any compatible
combination of Kimodo constraints (full-body keyframes, hand/foot targets,
root waypoints/paths — or constraints alone), generate N samples per move,
gate the results numerically (including per-constraint adherence), keep the
best, and bake canonicalized clips + resolved constraint records + frame
data.** The one-page control-selection guide for authoring agents:
ANIMATION_AGENT.md.

```
 move spec              stance pose             generation             bake
 prompts + intents ──►  extracted from the ──►  N samples/move   ──►  validate frame,
 (JSON, per move)       generated idle          → QA gates            frame data,
                        (bookend constraint)    → best-of-N           manifest
  moveset_mk.json        kimogen.py stance       kimogen.py gen        bake_kimodo.py
```

## 0. Prerequisites

- **NVIDIA Kimodo** installed and generating — a text- and
  constraint-conditioned diffusion model over the SOMA *human* skeleton
  (77 joints, head + fingers). **KIMODO.md is the complete
  download/install/run manual**, including its axis conventions and the
  rest-pose trap; the `kimodo/` directory holds the tools referenced
  throughout this stage (`kimogen.py`, `bake_kimodo.py`, the validated MK
  move spec).

  Alternatives fit the same pipeline — cutting clips from a mocap library,
  another text-to-motion model, hand-keyed animation — anything that can emit
  the clip format below plays through the same Stage 1/Stage 3 machinery.
- A **certified character** to preview on (Stage 1, `certify.mjs` — see
  ALIGN.md). Never evaluate motion on an uncertified rig: you can't tell
  motion defects from transfer defects.

## 1. The clip contract

Every baked clip is a motion JSON in the ALIGN.md format (world-space joint
positions + quaternions per frame, Y-up meters, with `rest`/`restQuat`), and
**canonicalized**: frame 0 pelvis at the origin, heading = 0, canonical
forward = **+X**. `kimogen.py` canonicalizes before evaluating gates (and
re-canonicalizes after loop trimming); `bake_kimodo.py` refuses clips that do
not satisfy the contract. Raw generations inherit arbitrary world frames, and
the runtime (Stage 3) depends on every clip agreeing on origin and forward.

Each clip also carries:

- `srcMap` — canonical source role → joint name in this clip. This is what
  makes the runtime source-agnostic: the retargeter reads the map from the
  clip, never assumes a skeleton.
- `handFollow` — the wrist-articulation stylization gain (what it does and
  the ablation evidence: KIMODO.md §2). Clips with authored hand constraints
  are always baked at 1.0.
- `contacts` + `contactJoints` — Kimodo's per-frame foot-contact
  **predictions** with their explicit joint mapping. QA and cleanup evidence,
  never authored targets.
- `constraints` — the resolved constraint records for every authored
  constraint, re-expressed in the clip's canonical frame with the exact
  rigid transform + loop trim the motion got: family, authoring source,
  canonical frame index, world position/rotation targets (end-effectors),
  root/path/heading targets, position/rotation constrained flags, and
  post-processing provenance (`conditioned+corrected` vs `conditioned`).
  Directly usable by runtime constraint IK (`ik.js`) and constraint QA
  (`qa_constraints.mjs`) — no authoring skeleton, no FK.

Root motion stays **in the clip data** (the pelvis actually travels). The
runtime decides how to consume it; never bake clips "in place".

Alongside the clips, emit a `manifest.json`: the ordered list of moves, each
with its file, frame count, fps, loop flag, and frame data (§5).

## 2. The stance pose (the bookend)

**Bookend transition-sensitive one-shots with the same pose.** Set
`stance_bookend` when a move must start and end on the shared contact pose;
cyclic locomotion instead closes on its best pose-space loop, while terminal or
prompt-only reactions can omit the constraint. *Which* shared pose is arbitrary:
pick your game's natural contact pose (the MK set uses a fighting stance; an
endless runner would use a run-contact pose).

The pose is not authored by hand — it is **extracted from the generation
itself**: generate the idle first (unconstrained), then take its medoid frame
(the pose the idle keeps returning to) as THE stance:

```bash
python kimogen.py gen --spec moveset_mk.json --only idle_stance
python kimogen.py stance          # → out/stance_pose.json
```

Every move with `"stance_bookend": true` is then generated with a full-body
keyframe constraint pinning frames `[0, T−1]` to that pose, and its prompt
ends "...then returns to the fighting stance". Constraint + prompt together
give end poses within 0.05–0.2 m of the reference — inside a 5-frame
crossfade's coverage.

## 3. Write the move spec

One JSON spec per move set (`kimodo/moveset_mk.json` is the validated 17-move
reference). Per move:

```jsonc
{"fps": 30, "moves": [
  // cyclic (idles, walks): loop → trim to the best pose-space cycle at bake
  {"name": "walk_fwd", "duration": 4.0, "loop": true, "travel": "fwd",
   "prompt": "A person in a fighting stance with fists raised walks forward cautiously toward their opponent."},
  // one-shot attack: bookended, gated on its apex, with frame data
  {"name": "kick_high", "duration": 2.5, "travel": "in_place",
   "stance_bookend": true, "strike": "foot", "height": "high",
   "apex": {"kind": "ankle_height", "min": 1.1},
   "prompt": "A person in a fighting stance throws a high roundhouse kick at head height, then returns to the fighting stance."}
]}
```

- `travel` — the move's net-root-displacement intent, gated in world space:
  `"fwd"`/`"back"` on sign, `"in_place"` on magnitude, `null` = don't gate.
- `apex` — the move's defining physical moment, gated absolutely:
  `ankle_height` (kicks), `root_rise` (jumps), `root_dip` (crouches, sweeps),
  `root_floor` (knockdowns), bounds in meters. One bound is required per kind
  (`max` for `root_floor`, `min` for the rest); the other is optional and caps
  the gate from the far side — e.g. `{"kind": "ankle_height", "min": 0.4,
  "max": 0.9}` rejects both statues and karate-kick winners.
- `strike` (`"hand"`/`"foot"`) + `height` — enables frame-data extraction
  (§5) and tags the move for gameplay rules.

### Constraints: pinning what prompts can't

Any move can carry Kimodo constraints — inline (`constraints`: a JSON array
in the exact upstream schema) or from a file saved by the Kimodo demo/API
(`constraints_file`; the two are mutually exclusive). Every family the
installed Kimodo supports passes through:

```jsonc
{"name": "reach_while_walking", "duration": 3.0,
 "prompt": "A person walks forward and reaches for an object.",
 "constraints": [
   // sparse root waypoints (dense per-frame arrays = an exact path);
   // optional global_root_heading pins facing as [cos t, sin t] pairs
   {"type": "root2d", "frame_indices": [0, 45, 89],
    "smooth_root_2d": [[0.0, 0.0], [0.4, 0.2], [0.9, 0.2]]},
   // hand/foot end-effector target: a complete SOMA pose per frame; the
   // world target is its FK. Shorthands: left-hand right-hand left-foot
   // right-foot; or "end-effector" + joint_names (exact tokens LeftHand,
   // RightHand, LeftFoot, RightFoot) — the wrapper splits the generic form
   // into shorthands so MotionCorrection corrects them.
   {"type": "right-hand", "frame_indices": [60],
    "local_joints_rot": ["... [1,30,3] axis-angle ..."],
    "root_positions": [[0.55, 0.95, 0.2]]},
   // arbitrary full-body keyframe at any frame
   {"type": "fullbody", "frame_indices": [110],
    "local_joints_rot": ["... [1,30,3] ..."], "root_positions": [[0.62, 0.95, 0.28]]}
 ]}
```

Real complete examples for every family (and their mixed combination) are in
`kimodo/moveset_e2e.json`. The essentials:

- **Native authoring frame**: Y-up, meters, heading 0 faces **+Z**, root
  starts near the XZ origin. Accepted clips are canonicalized to +X later —
  the wrapper re-expresses resolved targets with the same transform.
- **Everything validates before the model loads**: known types and
  end-effector tokens, integer/sorted/unique in-range frames, exact
  SOMA-30/77 array shapes, finite values, radians (a magnitude > 2π is
  rejected as degrees), plausible hip heights, non-zero headings, Kimodo's
  <20 sparse-frames-per-type guidance (dense root paths exempt), root-speed
  reachability (≤ 5 m/s between pinned frames), and conflict detection.
- **Conflicts are rejected, never overwritten**: two same-type constraints on
  one frame, fullbody + end-effector on the same frame, co-framed
  end-effectors with disagreeing roots (an EE constraint also pins root XZ,
  height, and heading from its pose), root2d vs pose XZ disagreements, and
  heading contradictions all fail with a move/constraint/frame diagnostic.
- **`stance_bookend` stays a convenience**: it produces a fullbody constraint
  at frames `[0, T−1]` and merges with authored constraints under the same
  conflict rules.
- **Constraint-only moves** omit `prompt`: the installed API zeroes empty
  text explicitly (no placeholder wording is ever substituted).
- **`reach_policy: "clamp"`** (optional, per move) documents that mapped
  end-effector targets beyond a character's limb reach may be explicitly
  clamped at retarget time; the default is strict (character QA fails
  unreachable targets).
- **Loop moves**: retained constraint frames shift with the trim; a trim that
  would drop a required constraint is rejected (constraints may opt out with
  `"required": false` — e.g. dense paths on loop clips).

Generation measures **per-constraint adherence** on every sample and hard-
gates the winner (`ee_pos_max ≤ 5 mm`, `ee_rot_max ≤ 2°`, `root_xz_max ≤
2 cm`, fullbody EE positions, heading error); the report records per-record
errors, and `out/moves/<move>.resolved_constraints.json` stores the resolved
canonical targets that the bake embeds into the clip.

Design rules that survived a full move-set in production:

- **One action per prompt.** "Throws a rising uppercut, then returns to the
  fighting stance" — never a combo. The model won't reliably invent
  multi-phase choreography from one prompt; split it into separate moves.
- **Prompt in the training distribution's voice.** "A person ..." phrasing;
  concrete physical verbs; the target height/direction stated plainly.
- **Expect amplitude undershoot.** Generative priors are conservative. Say
  "at head height", gate the apex numerically, and reword stronger (or raise
  `duration` so the move has room) when every sample fails the gate.
- **Respect the prior's natural tempo.** If the game needs a 12-frame jab,
  generate the natural ~1.5 s move and play it faster in-engine (Stage 3
  handles this) — don't fight the prior with a tiny `duration`.

## 4. Generate with best-of-N

Per move, generate N samples (default 8), gate each candidate, keep the best:

```bash
python kimogen.py gen --spec moveset_mk.json        # everything, bookended
python kimogen.py report                            # gate table so far
```

| gate | meaning | reject |
|---|---|---|
| non-finite / shape | malformed sample | any |
| contact evidence | at least one foot-contact label on ≥5% of frames | missing contact signal |
| travel intent | net root X vs the spec's `travel` | wrong sign, or drift on `in_place` |
| apex | the move's defining moment (`ankle_height`, `root_rise`, …) in world space | outside `min`/`max` |
| stance match | end-effector error vs the stance pose at both bookends | > 0.22 m |
| foot skate | mean horizontal foot/toe speed during labeled ground contact | > 0.12 m/s |
| jitter | mean 2nd difference of joint positions | > 0.015 m/frame² |

Passing samples are ranked by a score (skate + jitter + end-stance error);
the winner's NPZ + gate/frame-data JSON land in `out/moves/`. If no sample
passes, `kimogen.py` exits nonzero, writes only the diagnostic report, and
removes any stale NPZ for that move so a previous generation cannot be baked
by accident.

**Gate what you actually care about, in world space.** A cautionary tale
(learned on a previous motion source, still the operating rule): a "jump"
selected by smoothness gates alone never left the ground — foot-skate gates
*reward* staying planted, and a pelvis-relative error metric can't see root
height. Every move's defining physical property (flight, displacement, floor
time, apex height) gets an absolute world-space gate, and those are **hard
rejects, not score terms** — a sample with a defining physical defect must
not win best-of-N no matter how smooth it is.

**The reject loop.** Apex unreachable on all samples → stronger prompt
wording or longer duration. Wrong action entirely → the prompt is outside
the training distribution; rephrase toward plain physical description. One
sample does something weird → more samples. Always verify visually with a
stick-figure filmstrip *on the canonical skeleton* before baking — cheaper
than debugging through the retargeter.

## 5. Bake + frame data

```bash
python bake_kimodo.py --web <movesDir>    # clips + manifest.json
```

`kimogen.py` has already canonicalized the motion and trimmed loops;
`bake_kimodo.py` verifies that frame, builds the normalized rest pose (identity
rest + straightened hand anchors — the trap and its fix: KIMODO.md §2), and
injects `srcMap`/`handFollow`, the predicted foot-contact channels
(`contacts` + `contactJoints`), and the resolved constraint records
(`constraints`, from `<move>.resolved_constraints.json`; frame/fps mismatches
are rejected). The manifest carries each move's file, frame count, fps, loop
flag, frame data, constraint counts + generation adherence, and any
`reachPolicy` — runtimes and the pre-bake tool consume it as-is and never
re-derive which clips loop.
The bake rejects missing or failed generation reports by default. Imported
example NPZs can opt out explicitly with `--allow-ungated`; generated moves
must never use that escape hatch, and an explicit failed report is always
rejected. It also requires every move in the spec;
use a smaller spec for a smaller deliverable (`--allow-missing` is reserved
for deliberate partial previews).

Frame data is derived per attack from the generation itself — never
hand-authored:

- **startup / active** — from the strike-limb tip speed profile: the active
  window spans the fast phase of the swing,
- **contact** — the single frame the strike visually lands: max extension of
  the striking tip from the root within the active window,
- **recovery** — the rest.

`contact` exists because the active window alone is not precise enough for
impact events: generated motion has real wind-up, so the speed-derived window
opens while the limb is still travelling — measured on the MK set, visual
impact falls 5–9 frames *after* `active[0]`, usually right at the window's
end. Stage 3 syncs damage/hitstop/sfx to it (INTEGRATE.md §6).

Keep the baked clips character-agnostic; retargeting happens at load/run time
through the certified retargeter (or, equivalently, pre-bake per-character
engine-native clips — glTF animations, engine `AnimationClip`s — by running
the retargeter offline once per character and exporting; the runtime cost is
identical to hand-authored animation either way).

## 6. Visual QA rubric

Certification (Stage 1) catches *rig* problems; this catches *motion*
problems. Render every move on a certified character (contact sheet + video)
and check:

- the prompted apex actually visible and at the intended amplitude?
- feet planted during stances/guards (no skate)?
- no wrist flips or knee pops?
- root travel matches intent (lunge forward, knockback back, in-place else)?

A weak move is **a prompt/spec edit + regenerate** (~a minute), never a
runtime patch. Run `qa_constraints.mjs` (stage-separated constraint accuracy:
authored → SOMA → unguarded rig → shipped rig, plus determinism/flip/skate
gates) and `qa_endeffectors.mjs` (perceptual foot-pitch/wrist-bend gates)
after any bake — together they catch rest-anchor skew, constraint misses,
and transfer regressions mechanically, per character × move set.

## 7. New character = zero work here

Clips know nothing about characters. For a new model: certify it (Stage 1),
then play the existing clips through its certified retarget. Regeneration is
only ever needed for new *moves*.
