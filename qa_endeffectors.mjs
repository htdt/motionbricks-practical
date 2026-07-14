// End-effector transfer-fidelity gates: catch skewed fists/feet numerically.
//
// The Retargeter anchors its full-orientation transfers at the SOURCE REST
// pose (feet: FULLQ pelvis-relative; hands: ride-the-forearm + rest-anchored
// wrist delta). That is only correct if the rest pose is semantically neutral
// — feet FLAT, hands STRAIGHT along the forearm. The SOMA standard T-pose is
// NOT (toes ~14° down, hands ~18° forward), which shipped as toes-up feet and
// bent "skewed" fists until this gate existed. bake_kimodo.py now normalizes
// the rest; this script PROVES it on the character, per clip, headless:
//
//   foot pitch  — character toe-segment angle vs horizontal on frames where
//                 the SOURCE foot is grounded & still (flat by definition).
//   hand skew   — angle between the character's hand direction and the
//                 direction the source demands (source hand-vs-forearm
//                 deviation applied to the character's forearm). Measured on
//                 a RAW transfer run (handFollow = 1): the expected value is
//                 NEVER scaled by the clip's stylization gain, so damping
//                 cannot masquerade as perfect transfer. The gain's own
//                 effect is reported separately as `styleDelta` (the wrist
//                 articulation the shipped handFollow discards) and is not
//                 part of the transfer gate.
//
// These are PERCEPTUAL diagnostics; exact constraint-adherence gates live in
// qa_constraints.mjs and must be run alongside.
//
// Usage: node qa_endeffectors.mjs <char.glb> <movesDir> [--gate]
//        e.g. node qa_endeffectors.mjs ../web/fighter.glb ../web/moves_kimodo --gate
// Gates: median |foot pitch| <= 10 deg on contact frames,
//        median hand skew   <= 40 deg over all frames (per side, per clip),
//        with aggregate medians <= 4 / 15 deg.
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { loadGLBBones } from './glbskel.mjs';
import { Retargeter } from './retarget.js';
import { rigFromBones, resetBindPose } from './align.js';

const [charPath, movesDir] = process.argv.slice(2);
const GATE = process.argv.includes('--gate');
if (!charPath || !movesDir) {
  console.error('usage: node qa_endeffectors.mjs <char.glb> <movesDir> [--gate]');
  process.exit(2);
}

// Gate design: the defect class this catches is CONSTANT anchor skew — a
// wrong rest convention reads as 15° (feet) to 65-90° (hands) medians on
// EVERY clip (that shipped once). Per-clip caps flag it loudly; aggregate
// medians keep typical fidelity honest. Thresholds are calibrated across
// two certified rigs (Tripo fighter: agg 1.1/9.1; Mixamo: agg 1.0/13.5) so
// transient fast-swing and exact authored-wrist clips pass without per-clip
// exceptions while the old-anchor defect (65-90° on nearly every clip) cannot.
const FOOT_PITCH_MAX = 10;      // deg, per-clip median over contact frames
const FOOT_MIN_FRAMES = 15;     // fewer contact frames -> unreliable, skip clip
const HAND_SKEW_MAX = 40;       // deg, per-clip median cap
const AGG_FOOT_MAX = 4;         // deg, median of per-clip medians
const AGG_HAND_MAX = 15;        // deg, median of per-clip medians

const { bones: loadedBones, wrapper } = await loadGLBBones(charPath);
const target = rigFromBones(loadedBones);
const bones = target.bones;
const manifest = JSON.parse(fs.readFileSync(path.join(movesDir, 'manifest.json'), 'utf8'));
if (!Array.isArray(manifest.moves) || !manifest.moves.length)
  throw new Error('manifest must contain a non-empty moves array');
const manifestNames = manifest.moves.map(mv => mv.name);
if (manifestNames.some(name => typeof name !== 'string' || !name) ||
    new Set(manifestNames).size !== manifestNames.length)
  throw new Error('manifest move names must be present and unique');
if (manifest.moves.some(mv => typeof mv.file !== 'string' || !mv.file))
  throw new Error('every manifest move must provide a clip file');

const deg = (r) => r * 180 / Math.PI;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const wpos = (b) => b.getWorldPosition(new THREE.Vector3());

let failures = 0;
const rows = [];

for (const mv of manifest.moves) {
  const motion = JSON.parse(fs.readFileSync(path.join(movesDir, mv.file), 'utf8'));
  const S = motion.srcMap;
  if (!S) { console.log(`[FAIL] ${mv.name}: clip carries no srcMap`); failures++; continue; }
  const idx = Object.create(null); motion.names.forEach((n, i) => idx[n] = i);

  // fresh skeleton state per clip: reset to bind, then construct (constructor
  // snapshots bind from current transforms). RAW transfer run: handFollow = 1
  // so the transfer gates measure fidelity, not the stylization gain.
  resetBindPose(target);
  const rt = new Retargeter({ ...target, data: motion, handFollow: 1 });
  const R = rt.R;
  const shippedFollow = motion.handFollow ?? 1;

  // character measurement fixtures (captured at bind)
  const fix = {};
  for (const side of ['Left', 'Right']) {
    const foot = bones[R[`${side}Foot`]];
    const toe = R[`${side}ToeBase`] ? bones[R[`${side}ToeBase`]] : null;
    // bind pitch of the ankle->toe segment (the ankle sits above the toes,
    // so a flat character foot is NOT horizontal here) — errors are measured
    // relative to this
    let bindFootPitch = 0;
    if (foot && toe) {
      const d0 = wpos(toe).sub(wpos(foot)).normalize();
      bindFootPitch = deg(Math.asin(THREE.MathUtils.clamp(d0.y, -1, 1)));
    }
    const fore = bones[R[`${side}ForeArm`]];
    const hand = bones[R[`${side}Hand`]];
    // hand direction probe: prefer the bone child straightest along the
    // forearm at bind (middle-finger chain); rigs without finger bones
    // (Tripo) fall back to tracking the hand's world-quat delta from bind
    // applied to the bind knuckle direction (extrapolated along the forearm)
    let handChild = null, best = -2;
    let handBindQ = null, handRefDir = null, bindHandDev = 0;
    if (hand && fore) {
      const fdir = wpos(hand).sub(wpos(fore)).normalize();
      for (const c of hand.children.filter(c => c.isBone)) {
        const d = wpos(c).sub(wpos(hand)).normalize().dot(fdir);
        if (d > best) { best = d; handChild = c; }
      }
      handBindQ = hand.getWorldQuaternion(new THREE.Quaternion());
      handRefDir = fdir.clone();                 // bind hands extend along the forearm
      // rigs with finger bones measure the real knuckle chain, which carries
      // the mesh's own bind curl — subtract it so 0 = bind posture on every rig
      if (handChild) bindHandDev = deg(Math.acos(THREE.MathUtils.clamp(
        wpos(handChild).sub(wpos(hand)).normalize().dot(fdir), -1, 1)));
    }
    fix[side] = { foot, toe, fore, hand, handChild, handBindQ, handRefDir, bindHandDev, bindFootPitch };
  }

  const footPitch = { Left: [], Right: [] };
  const handSkew = { Left: [], Right: [] };
  const styleDelta = { Left: [], Right: [] };

  // source contact detection: toe near clip-min height and slow
  const srcToeMin = {};
  for (const side of ['Left', 'Right']) {
    const ti = idx[side === 'Left' ? 'LeftToeBase' : 'RightToeBase'];
    if (ti === undefined) throw new Error(`${mv.name}: source clip has no ${side}ToeBase joint`);
    srcToeMin[side] = Math.min(...motion.pos.map(f => f[ti][1]));
  }

  const N = motion.numFrames ?? motion.pos.length;
  for (let f = 0; f < N; f++) {
    rt.applyFrame(f);
    wrapper.updateMatrixWorld(true);
    for (const side of ['Left', 'Right']) {
      const { foot, toe, fore, hand, handChild } = fix[side];
      const sToe = idx[`${side}ToeBase`];
      const sFore = idx[S[`${side}ForeArm`]], sWrist = idx[S[`${side}Hand`]];
      const sMid = idx[`${side}HandMiddle1`] ?? idx[`${side}HandMiddleEnd`];

      // ---- foot pitch on source-contact frames
      if (foot && toe) {
        const p = motion.pos[f][sToe], speed = f ? Math.hypot(
          p[0] - motion.pos[f - 1][sToe][0], p[2] - motion.pos[f - 1][sToe][2]) * rt.fps : 0;
        if (p[1] < srcToeMin[side] + 0.03 && speed < 0.3) {
          const d = wpos(toe).sub(wpos(foot)).normalize();
          const pitch = deg(Math.asin(THREE.MathUtils.clamp(d.y, -1, 1)));
          footPitch[side].push(pitch - fix[side].bindFootPitch);
        }
      }

      // ---- hand skew: |char wrist bend - source wrist bend|, raw angles
      // between the knuckle direction and the forearm axis. The bake anchors
      // hands for ABSOLUTE bend tracking (straight source wrist ↔ straight
      // character hand); this run transfers the FULL source wrist
      // (handFollow=1), so the expected value is the unscaled source bend —
      // intentional damping can never masquerade as perfect transfer here.
      const { handBindQ, handRefDir } = fix[side];
      if (hand && handBindQ && sMid !== undefined) {
        const srcFore = v3(motion.pos[f][sWrist]).sub(v3(motion.pos[f][sFore])).normalize();
        const srcHand = v3(motion.pos[f][sMid]).sub(v3(motion.pos[f][sWrist])).normalize();
        const devSrc = deg(Math.acos(THREE.MathUtils.clamp(srcHand.dot(srcFore), -1, 1)));
        const charFore = wpos(hand).sub(wpos(fore)).normalize();
        const got = handChild
          ? wpos(handChild).sub(wpos(hand)).normalize()
          : handRefDir.clone().applyQuaternion(
              hand.getWorldQuaternion(new THREE.Quaternion()).multiply(handBindQ.clone().invert()));
        const devChar = deg(Math.acos(THREE.MathUtils.clamp(got.dot(charFore), -1, 1)))
          - fix[side].bindHandDev;
        handSkew[side].push(Math.abs(devChar - devSrc));
        // the shipped stylization gain discards (1-handFollow) of the source
        // wrist deviation — reported separately, never folded into the gate
        styleDelta[side].push((1 - shippedFollow) * devSrc);
      }
    }
  }

  for (const side of ['Left', 'Right']) {
    const fp = median(footPitch[side].map(Math.abs));
    const hs = median(handSkew[side]);
    const sd = median(styleDelta[side]);
    const fpOk = footPitch[side].length < FOOT_MIN_FRAMES || !(fp > FOOT_PITCH_MAX);
    const hsOk = Number.isFinite(hs) && hs <= HAND_SKEW_MAX;
    if (!fpOk || !hsOk) failures++;
    rows.push({ clip: mv.name, side, footPitchMed: +fp.toFixed(1),
                contactFrames: footPitch[side].length,
                handSkewMed: +hs.toFixed(1),
                styleDeltaMed: Number.isFinite(sd) ? +sd.toFixed(1) : null,
                ok: fpOk && hsOk });
  }
}

const aggFoot = median(rows.filter(r => r.contactFrames >= FOOT_MIN_FRAMES).map(r => r.footPitchMed));
const aggHand = median(rows.map(r => r.handSkewMed).filter(v => !Number.isNaN(v)));
if (aggFoot > AGG_FOOT_MAX || aggHand > AGG_HAND_MAX) failures++;
if (!rows.length || !Number.isFinite(aggFoot) || !Number.isFinite(aggHand)) failures++;

console.log(`\nend-effector fidelity — ${path.basename(charPath)} vs ${movesDir}`);
console.log(`gates: per-clip |foot pitch| med <= ${FOOT_PITCH_MAX} deg (>=${FOOT_MIN_FRAMES} contact frames), ` +
  `hand-bend skew med <= ${HAND_SKEW_MAX} deg; aggregate medians <= ${AGG_FOOT_MAX} / ${AGG_HAND_MAX} deg`);
for (const r of rows)
  console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${r.clip.padEnd(13)} ${r.side.padEnd(5)}` +
    ` footPitch=${String(r.footPitchMed).padStart(5)} (${r.contactFrames}f)` +
    ` handSkew=${String(r.handSkewMed).padStart(5)}` +
    ` styleΔ=${String(r.styleDeltaMed ?? '-').padStart(5)}`);
console.log(`aggregate: foot ${aggFoot?.toFixed(1)} deg, hand ${aggHand?.toFixed(1)} deg`);
console.log(failures ? `\n${failures} FAILING gates` : '\nALL END-EFFECTOR GATES PASS');
if (GATE && failures) process.exit(1);
