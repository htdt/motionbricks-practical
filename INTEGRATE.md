# Stage 3 — INTEGRATE: baked clips → a playable game

How to turn a baked move set (BAKE.md output) into interactive gameplay with
correct root motion, responsive timing, and verified animation quality — in
any engine. The reference implementation is three.js using this repo's
`retarget.js` at runtime; the same architecture ports to engine-native
animation players unchanged (§9).

The idea in one line: **clips play in place through the certified retargeter,
a tiny entity layer integrates their baked root motion and crossfades between
states, and gameplay timing falls out of the generated frame data — zero
neural code, zero hand-tuned motion constants.**

## 1. Architecture: three layers, strictly separated

```
                     per character
   manifest.json ──► one Retargeter PER CLIP  (inPlace: true)
                            │ applyFrame(f)
   state machine ───────────┤
     · entity position       │ world anchor = bind + entity position
     · facing                │
     · cursor (float, speed) ▼
   gameplay (frame data) ──► bones posed ──► render
```

- **Clip layer** — one `Retargeter` instance per (character, clip).
  Constructed **once at load, all before the first `applyFrame`**:
  construction captures the bind pose from the current bone transforms, so
  posing the skeleton first makes the "bind" garbage.
- **Entity layer** — the character's world position + facing. The clip's root
  motion is *integrated* into the entity position; the clip itself plays in
  place. This layer owns collision, arena bounds, separation constraints.
- **Game layer** — state machine, hit/interaction resolution from frame data,
  scores/AI/UI. Nothing here touches bones.

## 2. Facing: bake it, don't mirror it

Rotate the character's model to its gameplay facing **before constructing its
retargeters**, then never rotate for facing again:

```js
model.rotation.y = THREE.MathUtils.degToRad(yawDeg);
model.updateWorldMatrix(true, true);      // BEFORE new Retargeter(...)
```

The retargeter yaw-rebases everything onto the bind heading it sees at
construction, so "canonical clip forward (+X)" automatically becomes "the
direction this character faces". No runtime mirroring, no negative scale, no
per-frame yaw math. (Rig families differ in bind facing — check with one
screenshot and set the offset once.)

For games with free-rotating characters, the same principle holds per state
change: re-anchor heading when the entity turns, keep the clip in place.

## 3. Root motion: integrate horizontal, pass vertical through

Construct every retargeter with `inPlace: true` — it zeroes root X/Z but
**keeps Y**, so crouches dip and jumps rise with no extra work. Per tick,
integrate the clip's horizontal root delta into the entity:

```js
let dx = pos[f][pelvis][0] - pos[prevF][pelvis][0];
if (f < prevF)   // loop wrap (walk cycles): tail delta + head delta
  dx = (pos[N-1][pelvis][0] - pos[prevF][pelvis][0]) + (pos[f][pelvis][0] - pos[0][pelvis][0]);
entity.x += entity.facing * dx * rt.scaleRoot;

rt.hipsBindWorldPos.x = bind0.x + entity.x;   // pin the anchor (bind0 cloned at load)
rt.applyFrame(f);
```

What this buys, for free, from the baked data: walk speed *is* the baked
cycle displacement, attack lunges advance exactly as generated, hit reactions
knock back by their baked recoil. Movement constants cease to exist as code.
Clamp the entity to the arena and enforce separation/collision at the entity
layer — pushing, cornering, and shoving emerge from the same integration.

**Vertical amplification.** If the source prior under-jumps, scale the
vertical: `rt.yLift = k` multiplies root displacement above the source's
**rest height** — note, rest height, *not* a lowered stance height; computing
k from the stance gives ~40% less lift than expected. Set it per-clip, only
on clips that should be amplified (a victory pose also sits above rest height
and must not be).

## 4. Playback speed & the continuity-guard trap

Generators have a minimum natural clip length; games want snappier timing.
Play clips faster instead of regenerating: attacks ~1.4–1.6×, reactions
~1.2–1.4×, cycles ~1.25×, via a float cursor (`cursor += speed` per tick).

**The one real bug this causes:** at speed > 1 the integer frame index skips,
and the retargeter's temporal continuity guard (the 40°/frame slew limiter
that suppresses hand/forearm/foot flips) silently disengages — it only arms
on a *sequential* frame stream. Symptom: one-frame 90–180° hand flips that
never appeared at 1×. Fix: apply every skipped frame; the last write wins
visually and the guard sees a continuous stream:

```js
let steps = f - lastAppliedF; if (steps < 0) steps += N;
for (let k = steps - 1; k >= 1; k--) rt.applyFrame((f - k + N) % N);
rt.applyFrame(f);
```

Also reset the guard on every clip switch (`rt._lastF = null;
rt._contPrev = {}`) so it never bridges two different clips.

## 5. State machine over the clip set

States = clip names. Three shapes cover everything:

- **loops** (idle, walks): cursor wraps; use the wrap-aware root delta (§3).
- **one-shots** (attacks, jumps, reactions): cursor clamps at the last frame
  → done → back to idle. While active the character is *busy*
  (uninterruptible) — that commitment is what makes startup frames feel like
  gameplay. Define the exceptions explicitly: hit reactions always interrupt;
  terminal states (KO, victory) freeze on a data-derived frame (e.g. lowest
  pelvis = lying flat) rather than a hand-picked number.
- **holds** (crouch, guard): play to the keyframe-arrival frame (from frame
  data), pin the cursor while the input is held, on release jump to the
  return segment. The jump is a pose cut — cover it with the same crossfade
  as state changes.

**Crossfade** on every state change: snapshot all bone local quaternions +
hips position, then for ~5 frames blend snapshot → newly-applied pose with
smoothstep (`slerpQuaternions`). Because every clip bookends in the same
pose (BAKE.md rule — whatever contact pose fits the game), ~5 frames is
enough *everywhere* — walks cut mid-stride, hits interrupt attacks, all
covered by one mechanism.

## 6. Gameplay from frame data — no hand-authored numbers

Everything timing- and range-related derives from the bake outputs:

- **Timing**: `frame_data` per attack (`startup`, `active: [a,b]`,
  `recovery`). An attack can only connect while the cursor is inside the
  active window; check *every* active frame (a target can move into range
  mid-swing); latch a hit flag so one swing hits once.
- **Reach**: derive per attack at load — max forward offset of any strike-tip
  joint from the pelvis over the active window, × `rt.scaleRoot`, + a target
  body allowance, **floored just above your minimum entity separation**. The
  floor matters: a point-blank move (an uppercut) has almost no forward tip
  offset and would otherwise be unlandable. Generous hit ranges are usually
  genre-correct.
- **Rules from pose data**: states like "airborne" (pelvis above stance
  height during the jump window) or "downed" come from the clip data, not
  flags you maintain by hand.
- **Impact feel**: a few ticks of hitstop (freeze both entities' cursors) +
  a short camera shake on heavy hits. Cheap, transforms how hits read.

## 7. Determinism & the QA harness

Make the game loop externally steppable — this is what makes it *testable*:

```js
// ?capture=1 disables the internal clock; a test driver owns time
window.__step   = (input) => { gameTick(input ?? {}); render(); };
window.__state  = () => ({ /* per entity: state, frame, x, hipY, … */ });
window.__force  = (who, move) => { /* play any clip directly */ };
```

Any AI must use a seeded RNG (an LCG, not `Math.random`) so a scripted run
replays identically. Showcase videos are then just a scripted input list
against the seeded game, screenshotted every tick.

Drive every move through real inputs and assert:

| check | gate |
|---|---|
| NaN anywhere in pose/position | none allowed |
| locomotion travel | exceeds a floor over a fixed tick count, correct sign |
| jump rise / crouch dip / knockdown floor | hip-height deltas vs idle |
| one-shot integrity | cursor monotonic while busy (no spurious interruption) |
| per-tick worst bone world-rotation step | ≤ ~2× the guard cap outside crossfades |

On the rotation metric: the guard caps 40° per *applied* frame and sped-up
clips apply ~2 frames per tick, so ≤ ~80°/tick is guard-limited slew (smooth
by eye); genuine flips show as 90–180° single-tick spikes.

**Visual rubric** (contact sheets per move, consecutive-tick close-ups for
anything the metrics flag): apex pose visible? feet planted in stance? no
wrist flips or knee pops at state changes? interactions plausible at the
distance they register? root travel matches intent?

## 8. Gotchas index (each cost a real debug cycle)

1. **Continuity guard disengages at speed > 1** → apply intermediate frames (§4).
2. **Vertical amplification scales above source *rest* height, not stance** (§3).
3. **Point-blank moves need a reach floor** above minimum separation (§6).
4. **Construct all retargeters before any `applyFrame`** — bind capture (§1).
5. **Best-of-N can select away the move's point** (a "jump" that never leaves
   the ground) → gate defining physical properties in world space (BAKE.md §4).
6. **Weak generated poses are pose-library edits**, not runtime patches
   (BAKE.md §7).
7. **QA scenarios must actually exercise the interaction** — spawn entities in
   reach or every hit test silently passes as a whiff.

## 9. Porting to other engines (Godot / Bevy / Babylon / …)

Nothing above is three.js-specific. Per engine: pre-bake the retargeted clips
to engine-native animation assets (glTF animations → `AnimationPlayer` /
`AnimationClip` / `AnimationGroup`) instead of retargeting at runtime, keep
the identical entity/state/gameplay layers, and let the engine's animation
system do the crossfades (e.g. an `AnimationTree` crossfade replaces §5's
snapshot blend — same ~5-frame duration, same shared-bookend assumption).
Frame-data JSON and the derived-reach formula port unchanged. The runtime
cost is identical to hand-authored animation — that is the point of the
design.

`prebake.mjs` is the shipped pre-bake tool:

```bash
node prebake.mjs character.glb --manifest baked/manifest.json --out character_anim.glb \
     [--rootmotion rootmotion.json] [--ylift jump_gap=1.5]
```

It runs the certified retargeter offline over every clip in the manifest
(`inPlace: true`, continuity guard armed, sequential frames) and writes the
same GLB with one glTF animation per move (per-joint local rotations + hips
local translation), plus a `rootmotion.json` with each clip's absolute pelvis
X/Z per frame in character-scaled meters, hip height, loop flag, and frame
data — everything the entity layer needs to integrate root motion exactly as
in §3. `--ylift` applies §3's per-clip vertical amplification at bake time.
Field-validated with Babylon's glTF loader and `AnimationGroup` weight
blending, used unchanged.
