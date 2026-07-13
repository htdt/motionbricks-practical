# Stage 2 — BAKE: author, generate, gate, and bake motion clips

Everything in this stage happens on the **canonical skeleton** — the skeleton
of your motion source, not of any character. Clips authored here drive every
certified character forever; a new character never requires regenerating a
clip, and a new clip never requires touching a character.

The idea in one line: **author moves as text prompts (plus a shared bookend
constraint), generate N samples per move, gate the results numerically, keep
the best, and bake canonicalized clips + frame data.**

```
 move spec              stance pose             generation             bake
 prompts + intents ──►  extracted from the ──►  N samples/move   ──►  canonicalize,
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
forward = **+X**. Canonicalize at bake time — raw generations inherit
arbitrary world frames, and the runtime (Stage 3) depends on every clip
agreeing on origin and forward.

Each clip also carries:

- `srcMap` — canonical source role → joint name in this clip. This is what
  makes the runtime source-agnostic: the retargeter reads the map from the
  clip, never assumes a skeleton.
- `handFollow`, `foreRollSrc` — wrist-articulation gain and forearm-roll
  source flags for the retargeter (what they fix and why: KIMODO.md §2).

Root motion stays **in the clip data** (the pelvis actually travels). The
runtime decides how to consume it; never bake clips "in place".

Alongside the clips, emit a `manifest.json`: the ordered list of moves, each
with its file, frame count, fps, loop flag, and frame data (§5).

## 2. The stance pose (the bookend)

**Bookend every move with the same pose.** Start from it, end on it — this is
what lets clips chain and crossfade in-game with a single short blend. *Which*
pose is arbitrary: pick your game's natural contact pose (the MK set uses a
fighting stance; an endless runner would use a run-contact pose).

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
  `root_floor` (knockdowns) with a `min`/`max` in meters.
- `strike` (`"hand"`/`"foot"`) + `height` — enables frame-data extraction
  (§5) and tags the move for gameplay rules.

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
| nan / shape | malformed sample | any |
| travel intent | net root X vs the spec's `travel` | wrong sign, or drift on `in_place` |
| apex | the move's defining moment (`ankle_height`, `root_rise`, …) in world space | outside `min`/`max` |
| stance match | end-effector error vs the stance pose at both bookends | > 0.25 m |
| foot skate | mean horizontal ankle speed during ground contact | above source-prior level |
| jitter | mean 2nd difference of joint angles | visible vibration |

Passing samples are ranked by a score (skate + jitter + end-stance error);
the winner's NPZ + gate/frame-data JSON land in `out/moves/`.

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

Canonicalization (yaw to +X facing), the rest-pose normalization (identity
rest + straightened hand anchors — the trap and its fix: KIMODO.md §2), and
`srcMap`/`handFollow`/`foreRollSrc` injection all happen here. Loop moves are
trimmed to their best pose-space cycle before export. The manifest carries
each move's file, frame count, fps, loop flag and frame data — runtimes and
the pre-bake tool consume it as-is and never re-derive which clips loop.

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
runtime patch. Run `qa_endeffectors.mjs` (end-effector fidelity gates) after
any bake — it catches rest-anchor skew mechanically.

## 7. New character = zero work here

Clips know nothing about characters. For a new model: certify it (Stage 1),
then play the existing clips through its certified retarget. Regeneration is
only ever needed for new *moves*.
