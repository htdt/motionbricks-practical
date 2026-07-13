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
//                 deviation applied to the character's forearm), i.e. pure
//                 transfer error, valid for any wrist posture.
//
// Usage: node qa_endeffectors.mjs <char.glb> <movesDir> [--gate]
//        e.g. node qa_endeffectors.mjs ../web/fighter.glb ../web/moves_kimodo --gate
// Gates: median |foot pitch| <= 8 deg on contact frames,
//        median hand skew   <= 12 deg over all frames (per side, per clip).
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { loadGLBBones } from './glbskel.mjs';
import { buildBoneOrder, Retargeter } from './retarget.js';

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
// transient fast-swing outliers (a hand mid jump-swing) pass without
// per-clip exceptions while the defect class cannot.
const FOOT_PITCH_MAX = 10;      // deg, per-clip median over contact frames
const FOOT_MIN_FRAMES = 15;     // fewer contact frames -> unreliable, skip clip
const HAND_SKEW_MAX = 35;       // deg, per-clip median cap
const AGG_FOOT_MAX = 4;         // deg, median of per-clip medians
const AGG_HAND_MAX = 15;        // deg, median of per-clip medians

const { byName, wrapper } = await loadGLBBones(charPath);
const bones = {};
for (const [n, b] of byName) bones[n] = b;
const manifest = JSON.parse(fs.readFileSync(path.join(movesDir, 'manifest.json'), 'utf8'));

// snapshot the pristine bind pose once; restore before every clip (the
// Retargeter constructor reads "bind" from current transforms)
const bindSnap = Object.values(bones).map(b => ({
  b, p: b.position.clone(), q: b.quaternion.clone(), s: b.scale.clone(),
}));
function restoreBind() {
  for (const { b, p, q, s } of bindSnap) { b.position.copy(p); b.quaternion.copy(q); b.scale.copy(s); }
  wrapper.updateMatrixWorld(true);
}

const deg = (r) => r * 180 / Math.PI;
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : NaN; };
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const wpos = (b) => b.getWorldPosition(new THREE.Vector3());

let failures = 0;
const rows = [];

for (const mv of manifest.moves) {
  const motion = JSON.parse(fs.readFileSync(path.join(movesDir, mv.file), 'utf8'));
  const S = motion.srcMap;
  if (!S) { console.log(`[skip] ${mv.name}: clip carries no srcMap`); continue; }
  const idx = {}; motion.names.forEach((n, i) => idx[n] = i);

  // fresh skeleton state per clip: reset to bind, then construct (constructor
  // snapshots bind from current transforms)
  restoreBind();
  const orderedBones = buildBoneOrder(wrapper.children[0].isBone ? wrapper.children[0] : wrapper);
  const NOGUARDS = process.argv.includes('--noguards');   // diagnostic mode
  const rt = new Retargeter({ bones, orderedBones, data: motion,
    guards: NOGUARDS ? { handClamp: false, torsoCapsule: false, continuity: false } : {} });
  const R = rt.R;

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
  const handTwist = { Left: [], Right: [] };

  // source contact detection: toe near clip-min height and slow
  const srcToeMin = {};
  for (const side of ['Left', 'Right']) {
    const ti = idx[side === 'Left' ? 'LeftToeBase' : 'RightToeBase'];
    srcToeMin[side] = Math.min(...motion.pos.map(f => f[ti][1]));
  }

  // source anatomical rest deviation of the hand (from the baked rest pose):
  // subtracted so devSrc means "deviation from straight-along-forearm"
  const srcRestHandDev = {};
  for (const side of ['Left', 'Right']) {
    const w = idx[`${side}Hand`], m = idx[`${side}HandMiddle1`], fo = idx[S[`${side}ForeArm`]];
    if (m === undefined) { srcRestHandDev[side] = 0; continue; }
    const hd = v3(motion.rest[m]).sub(v3(motion.rest[w])).normalize();
    const fd = v3(motion.rest[w]).sub(v3(motion.rest[fo])).normalize();
    srcRestHandDev[side] = deg(Math.acos(THREE.MathUtils.clamp(hd.dot(fd), -1, 1)));
  }

  for (let f = 0; f < motion.numFrames; f++) {
    rt.applyFrame(f);
    wrapper.updateMatrixWorld(true);
    for (const side of ['Left', 'Right']) {
      const { foot, toe, fore, hand, handChild } = fix[side];
      const sToe = idx[`${side}ToeBase`], sToeEnd = idx[`${side}ToeEnd`];
      const sFore = idx[S[`${side}ForeArm`]], sWrist = idx[S[`${side}Hand`]];
      const sMid = idx[`${side}HandMiddle1`];

      // ---- foot pitch on source-contact frames
      if (foot && toe) {
        const p = motion.pos[f][sToe], speed = f ? Math.hypot(
          p[0] - motion.pos[f - 1][sToe][0], p[2] - motion.pos[f - 1][sToe][2]) * motion.fps : 0;
        if (p[1] < srcToeMin[side] + 0.03 && speed < 0.3) {
          const d = wpos(toe).sub(wpos(foot)).normalize();
          const pitch = deg(Math.asin(THREE.MathUtils.clamp(d.y, -1, 1)));
          footPitch[side].push(pitch - fix[side].bindFootPitch);
        }
      }

      // ---- hand skew: |char wrist bend - handFollow×(source wrist bend)|,
      // raw angles between the knuckle direction and the forearm axis. The
      // bake anchors hands for ABSOLUTE bend tracking (straight source wrist
      // ↔ straight character hand) scaled by the clip's wrist articulation
      // gain (handFollow; 1 = full transfer) — a scalar comparison, free of
      // rotation-axis/twist ambiguity.
      const { handBindQ, handRefDir } = fix[side];
      if (hand && handBindQ && sMid !== undefined) {
        const srcFore = v3(motion.pos[f][sWrist]).sub(v3(motion.pos[f][sFore])).normalize();
        const srcHand = v3(motion.pos[f][sMid]).sub(v3(motion.pos[f][sWrist])).normalize();
        const devSrc = (motion.handFollow ?? 1) *
          deg(Math.acos(THREE.MathUtils.clamp(srcHand.dot(srcFore), -1, 1)));
        const charFore = wpos(hand).sub(wpos(fore)).normalize();
        const got = handChild
          ? wpos(handChild).sub(wpos(hand)).normalize()
          : handRefDir.clone().applyQuaternion(
              hand.getWorldQuaternion(new THREE.Quaternion()).multiply(handBindQ.clone().invert()));
        const devChar = deg(Math.acos(THREE.MathUtils.clamp(got.dot(charFore), -1, 1)))
          - fix[side].bindHandDev;
        handSkew[side].push(Math.abs(devChar - devSrc));

        // full-orientation skew incl. TWIST: world rotation delta of the
        // character hand (vs bind) against the source wrist (vs rest) —
        // rest and bind are the same canonical pose, so the deltas must
        // agree; the angle of D_char·D_src⁻¹ is the total error
        const qW = motion.quat[f][sWrist], rW = motion.restQuat[sWrist];
        const dSrc = new THREE.Quaternion(qW[0], qW[1], qW[2], qW[3])
          .multiply(new THREE.Quaternion(rW[0], rW[1], rW[2], rW[3]).invert());
        const dChar = hand.getWorldQuaternion(new THREE.Quaternion())
          .multiply(handBindQ.clone().invert());
        const dq = dChar.multiply(dSrc.invert());
        handTwist[side].push(deg(2 * Math.acos(Math.min(1, Math.abs(dq.w)))));
      }
    }
  }

  for (const side of ['Left', 'Right']) {
    const fp = median(footPitch[side].map(Math.abs));
    const hs = median(handSkew[side]);
    const ht = median(handTwist[side]);
    const fpOk = footPitch[side].length < FOOT_MIN_FRAMES || !(fp > FOOT_PITCH_MAX);
    const hsOk = !(hs > HAND_SKEW_MAX);
    if (!fpOk || !hsOk) failures++;
    rows.push({ clip: mv.name, side, footPitchMed: +fp.toFixed(1),
                contactFrames: footPitch[side].length,
                handSkewMed: +hs.toFixed(1), handTwistMed: +ht.toFixed(1),
                ok: fpOk && hsOk });
  }
}

const aggFoot = median(rows.filter(r => r.contactFrames >= FOOT_MIN_FRAMES).map(r => r.footPitchMed));
const aggHand = median(rows.map(r => r.handSkewMed).filter(v => !Number.isNaN(v)));
if (aggFoot > AGG_FOOT_MAX || aggHand > AGG_HAND_MAX) failures++;

console.log(`\nend-effector fidelity — ${path.basename(charPath)} vs ${movesDir}`);
console.log(`gates: per-clip |foot pitch| med <= ${FOOT_PITCH_MAX} deg (>=${FOOT_MIN_FRAMES} contact frames), ` +
  `hand-bend skew med <= ${HAND_SKEW_MAX} deg; aggregate medians <= ${AGG_FOOT_MAX} / ${AGG_HAND_MAX} deg`);
for (const r of rows)
  console.log(`${r.ok ? 'ok  ' : 'FAIL'} ${r.clip.padEnd(13)} ${r.side.padEnd(5)}` +
    ` footPitch=${String(r.footPitchMed).padStart(5)} (${r.contactFrames}f)` +
    ` handSkew=${String(r.handSkewMed).padStart(5)}` +
    ` handTwist=${String(r.handTwistMed).padStart(5)}`);
console.log(`aggregate: foot ${aggFoot?.toFixed(1)} deg, hand ${aggHand?.toFixed(1)} deg`);
console.log(failures ? `\n${failures} FAILING gates` : '\nALL END-EFFECTOR GATES PASS');
if (GATE && failures) process.exit(1);
