# Stage 2 — BAKE: author, generate, gate, and bake motion clips

Everything in this stage happens on the **canonical skeleton** — the skeleton
of your motion source, not of any character. Clips authored here drive every
certified character forever; a new character never requires regenerating a
clip, and a new clip never requires touching a character.

The idea in one line: **author sparse keyframe poses, let your motion source
generate the movement between them, gate the results numerically, keep the
best of N attempts, and bake canonicalized clips + frame data.**

```
pose library          move spec              generation             bake
mine/author    ──►    keyframe schedule ──►  N seeds per move  ──►  canonicalize,
keyframe poses        (JSON, per move)       → QA gates             frame data,
(verified by eye)                            → best-of-N            manifest
   posekit.py           moves_*.json            movegen.py           bake_moves.py
```

## 0. Prerequisites

- A motion source. The validated path — the one this repo ships working
  tooling for — is **NVIDIA MotionBricks**, a keyframe-conditioned generative
  inbetweener over the Unitree G1 skeleton (takes 4 context frames + a
  4-frame target keyframe window, generates the motion between).
  **MOTIONBRICKS.md is the complete download/install/run manual**; the
  `motionbricks/` directory holds the tools referenced throughout this stage
  (`posekit.py`, `movegen.py`, `bake_moves.py`, a starter pose library and an
  example move spec). Alternatives that fit the same pipeline: cutting clips
  from a mocap library, a text-to-motion model, hand-keyed animation —
  anything that can emit the clip format below.
- A **certified character** to preview on (Stage 1, `certify.mjs` — see
  ALIGN.md). Never evaluate motion on an uncertified rig: you can't tell
  motion defects from transfer defects.

## 1. The clip contract

Every baked clip is a motion JSON in the ALIGN.md format (world-space joint
positions + quaternions per frame, Y-up meters, with `rest`/`restQuat`), and
**canonicalized**: frame 0 pelvis at the origin, heading = 0, canonical
forward = **+X**. Canonicalize at bake time — raw generations and mocap cuts
inherit arbitrary world frames, and the runtime (Stage 3) depends on every
clip agreeing on origin and forward.

Root motion stays **in the clip data** (the pelvis actually travels). The
runtime decides how to consume it; never bake clips "in place".

Alongside the clips, emit a `manifest.json`: the ordered list of moves, each
with its file, frame count, fps, loop flag, and frame data (§6).

## 2. Build a pose library

A keyframe is a full-body pose on the canonical skeleton. Tooling:
`posekit.py` (scan / sheet / save / show / list / wrists — see its docstring
and MOTIONBRICKS.md §7); a starter library `pose_library.json` ships with the
repo. Store poses as small **windows** (e.g. 4 consecutive frames — whatever
your generator's constraint API takes), in two flavors:

- a **moving window** (consecutive frames from a real clip) keeps the pose's
  momentum — right for strike apexes and anything the motion should *swing
  through*;
- a **held pose** (one frame tiled) has zero velocity — right for stances,
  guards, crouches: poses the motion should *arrive at and stop*.

### 2a. Mine from mocap (preferred)

Scan whatever mocap you have with cheap heuristics (highest foot = kick apex,
lowest pelvis = crouch, widest hand span = reach, …) to shortlist candidate
frames, then **render a contact sheet and look at it before saving anything**
— heuristics shortlist, your eyes decide. Save the chosen frames under
semantic names (`kick_high`, `stance`, `hit_recoil`).

### 2b. Author novel poses (fallback)

For poses no mocap has: copy the nearest library pose, edit joint angles,
render, iterate — cap it at ~3 rounds; if it's still wrong, the pose is
probably outside your generator's reachable set. Stay inside the source
skeleton's joint limits.

### 2c. Overshoot deliberately

Generative priors are conservative — they undershoot amplitude. Keyframes
*pull* the generation toward them, so author apex poses at or slightly past
what you actually want (head-height ankle, full lunge). If generation can't
reach the pose (arrival-error gate fails on every seed), back the pose off.

### 2d. Unregularized channels are authored, never generated

Rule learned the hard way: any channel your generator doesn't regularize
(for a robot-skeleton source that was the **wrists** — tiny links, barely
represented in the model's features, output was 40°+/frame noise) must be
**discarded from the generated data and rebuilt** as smooth interpolation
between explicitly authored per-keyframe values. Filtering noise leaves
smoothed noise; authored control channels make flicker impossible by
construction. If such a channel ever looks wrong, the fix is a pose-library
edit and a regenerate — never a runtime filter.

## 3. Write the move spec

One JSON spec per move set (`moves_example.json` is a complete 17-move
reference). Two move types:

```jsonc
{"moves": [
  // keyframe-driven (attacks, reactions, poses)
  {"name": "uppercut", "type": "keyframes",
   "start": "stance",              // context pose the move begins from
   "steps": [                      // each step = one generation chunk
     {"pose": "crouch_deep", "tokens": 6},   // chunk length in 4-frame tokens
     {"pose": "strike_rising", "tokens": 6},
     {"pose": "stance", "tokens": 6}         // ALWAYS return to the bookend pose
   ]},
  // native-skill rollout (locomotion — the model's own prior is the source)
  {"name": "walk_fwd", "type": "mode", "mode": "walk", "dir": "fwd",
   "chunks": 4, "loop": true}
]}
```

Useful per-step fields: target pose, duration (`tokens`: pin it, or omit and
let the model predict), root displacement `dxy` in meters (for lunges and
knockback), heading change `turn` in degrees. Move-level `loop: true` means
"trim the result to its best pose-space cycle" (idles, walks).

Move-level fields for the world-space gates (§4): `"min_airborne": 6` rejects
any seed without at least that many consecutive frames of BOTH ankles above
`"airborne_z"` (default 0.12 m) — this is how a jump is made to actually fly.
`"trim": [A, B]` cuts the baked clip to that frame range (§5) — a messy
recovery segment is dropped the same way on every rebake instead of by a
hand edit that goes stale.

### 3a. Declarative post-generation edits

Some clips only become readable after qpos-space surgery — the field example
was a slide whose extended leg vanished into the torso silhouette on a chase
camera. Doing that surgery in a one-off script means it is silently lost on
every regeneration. Put it in the move spec instead — a `"post"` block, a
list of ops applied per seed, in order, BEFORE gating (the gates must judge
exactly what ships), implemented in `qposops.py`:

```jsonc
{"name": "slide", "type": "keyframes",
 "steps": [ ... ],
 "post": [
   {"blend_to_pose": "slide_lunge", "frames": [12, 28], "max_w": 0.9},
   {"yaw_twist": 0.62, "frames": [12, 28]},
   {"ground_clamp": true}
 ]}
```

- `blend_to_pose` blends the joint dofs toward a pose-library pose over the
  frame window with a raised-cosine envelope peaking at `max_w`. Root and
  authored channels (wrists, §2d) are never touched.
- `yaw_twist` adds a yaw rotation (radians, about world up, in place about
  the pelvis) with the same envelope — reshapes the silhouette without
  moving the root path.
- `ground_clamp` lifts the root by the smoothed per-frame ankle penetration
  (same fix as `--groundfix`, §4, but scoped to this move).

Frame indices refer to the clip the filmstrip shows (after loop trimming);
pin `tokens` on the steps so the layout is stable across seeds. Post edits
are for readability surgery on an otherwise-good generation — a *weak pose*
is still a pose-library edit and a regenerate (§7).

Design rules that survived a full move-set in production:

- **Bookend every move with the same pose.** Start from it, end on it.
  This is what lets clips chain and crossfade in-game with a single short
  blend. *Which* pose is arbitrary — pick your game's natural contact pose,
  not this repo's `stance` (a fighting stance, right for the combat starter
  set only). The field endless-runner bookended every move in a mined
  run-contact pose (`run_A`) and chained/crossfaded cleanly; set it per move
  with `"start"` and end the step list on it.
- **One action per step.** `stance → kick → stance`, never
  `stance → kick_and_recover`. Use an intermediate held pose (a deep crouch
  inside an uppercut) to shape the path.
- **Respect your generator's minimum chunk length.** If the shortest natural
  generation is ~24 frames and the game needs a 12-frame jab, generate 24 and
  play it faster in-engine (Stage 3 handles this) — don't fight the prior.
- **Ground-contact changes are fine** (stance → flying hit → on the ground →
  get up) but give those steps extra duration.

## 4. Generate with best-of-N

Per move, run every seed (N = 8–16; make sampling stochastic so seeds
actually differ) through the whole keyframe schedule, gate each candidate,
keep the best. `movegen.py --spec <spec> --seeds 8` does all of this and
prints the gate table per seed. Generation is typically seconds per move on
a consumer GPU — seeds are cheap, debugging a bad clip downstream is not.

| gate | meaning | healthy | reject |
|---|---|---|---|
| keyframe arrival error | mean end-effector distance (wrists/ankles/torso, root-relative) between the generated arrival frame and the keyframe | 0.03–0.06 m | > 0.1 m = keyframe not reached |
| foot skate | mean horizontal ankle speed during ground contact | source-prior level | rising above it |
| jitter | mean 2nd difference of joint angles | ≤ 0.03 rad | visible vibration |
| limit violations | fraction of frames outside joint range | 0.0 | any |
| min world ankle height | lowest world ankle Z anywhere in the clip | ≥ −0.03 m | below = floor penetration, seed rejected |
| airborne frames | longest run of consecutive frames with both ankles above 0.12 m | ≥ the move's `min_airborne` | below = the "jump" never flew, seed rejected |

**Gate what you actually care about, in world space.** A cautionary tale: a
"jump" selected by the first four gates alone never left the ground —
foot-skate gates *reward* staying planted, and a pelvis-relative arrival
error can't see root height. The same blindness lets ground-contact moves
ship ankles below the floor: the inbetweener has no ground-contact
constraint, so an authored slide pose drove the source ankles to −0.17 m and
only the downstream certification gate caught it. The last two table rows are
the standing fix, and they are **hard rejects, not score terms** — a seed
with a defining physical defect must not win best-of-N no matter how clean
its keyframe arrival is. Declare flight per move (`"min_airborne"`, §3);
penetration is rejected everywhere. When a move has any other defining
physical property (displacement, floor time), assert it the same way.

**`--groundfix`.** Small penetrations on otherwise-best seeds don't have to
cost a regeneration: `movegen.py --groundfix` lifts the root per frame by the
maximum ankle penetration (smoothed, and re-maxed so smoothing never leaves
residual contact) before gating. Use it as the default for move sets with
ground work; per-move, the same fix is available as a `"ground_clamp"` post
op (§3a).

**The reject loop.** Keyframe unreachable on all seeds → reduce overshoot or
add an intermediate step. Duration feels wrong → pin a different length. One
seed does something weird → more seeds. Always verify visually with a
stick-figure filmstrip *on the canonical skeleton* before baking — cheaper
than debugging through the retargeter.

## 5. Bake

Canonicalize (§1), compute per-clip metadata, write clips + `manifest.json`
(`bake_moves.py` — output goes straight into `certify.mjs --clips` and the
Stage 3 runtime). The manifest carries each move's file, frame count, fps,
`loop` flag (from the spec) and frame data — runtimes and the pre-bake tool
consume it as-is and never re-derive which clips loop.

Trim at bake time, declaratively: the spec's `"trim": [A, B]` (§3) or
`bake_moves.py --trim move=A:B` keeps that half-open frame range (blank/null
end = to the clip's end). The cut happens BEFORE canonicalization — frame 0
of the trimmed clip is the canonical origin — and startup/active/recovery
shift with it. The field lesson behind this: hand-editing a baked JSON plus
the manifest is redone (and half-forgotten) on every rebake.
Keep the baked clips character-agnostic; retargeting happens at load/run time
through the certified retargeter (or, equivalently, pre-bake per-character
engine-native clips — glTF animations, engine `AnimationClip`s — by running
the retargeter offline once per character and exporting; the runtime cost is
identical to hand-authored animation either way).

## 6. Frame data (for gameplay)

Derive combat/gameplay timing from the generation itself — never hand-author
it:

- **startup** = frames until the first keyframe arrival (the apex),
- **active** = a small window around that arrival,
- **recovery** = the rest.

Sanity-check against strike-limb tip velocity peaks. Store per move in the
manifest. Stage 3 builds hit detection, reach, and interruption rules purely
from this data.

## 7. Visual QA rubric

Certification (Stage 1) catches *rig* problems; this catches *motion*
problems. Render every move on a certified character (contact sheet + video)
and check:

- keyframe apex actually visible and at the intended amplitude?
- feet planted during stances/guards (no skate)?
- no wrist flips or knee pops at chunk seams?
- root travel matches intent (lunge forward, knockback back, in-place else)?

Weak poses are **pose-library edits + regenerate** (~minutes), never runtime
patches.

## 8. New character = zero work here

Clips know nothing about characters. For a new model: certify it (Stage 1),
then play the existing clips through its certified retarget. Regeneration is
only ever needed for new *moves*.
