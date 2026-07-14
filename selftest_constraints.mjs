// Constraint-transfer self-test — NO external assets. Proves the target-space
// IK, the baseline diagnostic mode, and the constraint QA metrics on
// synthetic rigs with different bind headings, scales, limb proportions and
// no finger bones, including the mandated sabotage cases:
//   - a deliberately restored OLD hand rest anchor must FAIL the fidelity gates;
//   - a deliberately over-damped (handFollow) result must FAIL raw source
//     fidelity even though it looks stable;
//   - known wrist rotations that once shipped skewed fists must transfer
//     within tolerance through the world-axis mapping;
//   - direct seek and sequential playback must produce identical constrained
//     frames (baseline + IK path);
//   - unreachable targets are clamped EXPLICITLY and reported.
// Usage: node selftest_constraints.mjs
import * as THREE from 'three';
import { rigFromBones, srcMapFromRig, snapshotMotion, resetBindPose } from './align.js';
import { baselineOptions, Retargeter } from './retarget.js';
import { ConstraintIK, mapRecordTargets } from './ik.js';
import { captureRun, captureDelta, sourceFidelity, constraintAccuracy, quatAngleDeg } from './qametrics.mjs';

let failures = 0;
function check(label, ok, detail = '') {
  if (ok) console.log(`ok   ${label}`);
  else { failures++; console.log(`FAIL ${label}${detail ? '  — ' + detail : ''}`); }
}

// same convention as selftest.mjs: every local offset stored divided by the
// armature scale, like real scaled-armature exports (cm under a 0.01 node)
function buildRig(spec, armatureScale = 1) {
  const bones = {};
  const wrapper = new THREE.Group();
  wrapper.scale.setScalar(armatureScale);
  for (const [name, [parent, pos]] of Object.entries(spec)) {
    const b = new THREE.Bone();
    b.name = name;
    b.position.set(...pos).multiplyScalar(1 / armatureScale);
    bones[name] = b;
    if (parent) bones[parent].add(b); else wrapper.add(b);
  }
  wrapper.updateMatrixWorld(true);
  return { bones, wrapper, all: Object.values(bones) };
}

// SOURCE: UE-style names, +X facing, 1.5 m tall (same family as selftest.mjs)
const L = -0.09, R = +0.09;
const srcSpec = {
  pelvis: [null, [0, 0.86, 0]],
  spine_01: ['pelvis', [0, 0.14, 0]],
  spine_02: ['spine_01', [0, 0.14, 0]],
  spine_03: ['spine_02', [0, 0.14, 0]],
  neck_01: ['spine_03', [0, 0.10, 0]], head: ['neck_01', [0, 0.09, 0]],
  clavicle_l: ['spine_03', [0.01, 0.04, L * 0.5]], upperarm_l: ['clavicle_l', [0, 0, L]],
  lowerarm_l: ['upperarm_l', [0, 0, L * 2.4]], hand_l: ['lowerarm_l', [0, 0, L * 2.2]],
  clavicle_r: ['spine_03', [0.01, 0.04, R * 0.5]], upperarm_r: ['clavicle_r', [0, 0, R]],
  lowerarm_r: ['upperarm_r', [0, 0, R * 2.4]], hand_r: ['lowerarm_r', [0, 0, R * 2.2]],
  thigh_l: ['pelvis', [0, -0.04, L]], calf_l: ['thigh_l', [0, -0.40, 0]],
  foot_l: ['calf_l', [0, -0.38, 0]], ball_l: ['foot_l', [0.12, -0.04, 0]],
  thigh_r: ['pelvis', [0, -0.04, R]], calf_r: ['thigh_r', [0, -0.40, 0]],
  foot_r: ['calf_r', [0, -0.38, 0]], ball_r: ['foot_r', [0.12, -0.04, 0]],
};
// TARGETS: Mixamo-style names, +Z facing, different proportions; one 0.01
// armature, one plain; neither has finger bones.
const tgtSpec = {
  Hips: [null, [0, 1.02, 0]],
  Spine: ['Hips', [0, 0.16, 0]], Spine1: ['Spine', [0, 0.16, 0]], Spine2: ['Spine1', [0, 0.16, 0]],
  Neck: ['Spine2', [0, 0.10, 0]], Head: ['Neck', [0, 0.10, 0]],
  LeftShoulder: ['Spine2', [0.07, 0.04, 0]], LeftArm: ['LeftShoulder', [0.13, 0, 0]],
  LeftForeArm: ['LeftArm', [0.27, 0, 0]], LeftHand: ['LeftForeArm', [0.25, 0, 0]],
  RightShoulder: ['Spine2', [-0.07, 0.04, 0]], RightArm: ['RightShoulder', [-0.13, 0, 0]],
  RightForeArm: ['RightArm', [-0.27, 0, 0]], RightHand: ['RightForeArm', [-0.25, 0, 0]],
  LeftUpLeg: ['Hips', [0.11, -0.05, 0]], LeftLeg: ['LeftUpLeg', [0, -0.48, 0]],
  LeftFoot: ['LeftLeg', [0, -0.44, 0]], LeftToeBase: ['LeftFoot', [0, -0.05, 0.13]],
  RightUpLeg: ['Hips', [-0.11, -0.05, 0]], RightLeg: ['RightUpLeg', [0, -0.48, 0]],
  RightFoot: ['RightLeg', [0, -0.44, 0]], RightToeBase: ['RightFoot', [0, -0.05, 0.13]],
};
const shortSpec = JSON.parse(JSON.stringify(tgtSpec));      // shorter arms: IK must matter
shortSpec.LeftForeArm[1] = [0.24, 0, 0]; shortSpec.LeftHand[1] = [0.22, 0, 0];
shortSpec.RightForeArm[1] = [-0.24, 0, 0]; shortSpec.RightHand[1] = [-0.22, 0, 0];

const src = buildRig(srcSpec);
const srcT = rigFromBones(src.all);
const srcMap = srcMapFromRig(srcT.rig.map);

const N = 60, FPS = 30;
function pose(f) {
  const t = f / FPS;
  const s = (w, ph = 0) => Math.sin(2 * Math.PI * w * t + ph);
  src.bones.spine_02.rotation.y = 0.25 * s(0.5);
  src.bones.upperarm_l.rotation.x = 0.9 * s(0.7);
  src.bones.lowerarm_l.rotation.y = 0.5 * (1 - Math.cos(2 * Math.PI * 0.7 * t)) / 2;
  src.bones.upperarm_r.rotation.x = -0.8 * s(0.7, 1);
  src.bones.hand_l.rotation.z = 0.4 * s(1.2);
  src.bones.hand_l.rotation.x = 0.5 * s(0.9);              // wrist roll: the skewed-fist class
  src.bones.thigh_l.rotation.z = 0.4 * s(0.8);
  src.bones.calf_l.rotation.z = -0.3 * (1 - Math.cos(2 * Math.PI * 0.8 * t)) / 2;
  src.wrapper.updateMatrixWorld(true);
}
const clip = snapshotMotion(srcT.orderedBones, pose, N, FPS, 'ctest');
clip.srcMap = srcMap;

// authored end-effector records in SOURCE space at two frames: the wrist
// world pose of the source itself (reachable by construction on the source)
function eeRecordAt(frame, role = 'LeftHand') {
  const idx = clip.names.indexOf(srcMap[role]);
  const p = clip.pos[frame][idx], q = clip.quat[frame][idx];
  return { family: 'end-effector', type: role === 'LeftHand' ? 'left-hand' : 'right-hand',
    source: 'inline', originalType: 'test', required: true, provenance: 'conditioned+corrected',
    frame, role, pos: [...p], quat: [...q], posConstrained: true, rotConstrained: true };
}
clip.constraints = [eeRecordAt(20), eeRecordAt(45)];

// ---------------------------------------------------------------- baseline --
const tgt = rigFromBones(buildRig(tgtSpec, 0.01).all);
{
  const cap = captureRun(tgt, clip, { config: baselineOptions(clip) });
  check('baseline config dump reflects raw transfer', cap.config.handFollow === 1
    && cap.config.foreRollSrc === true && !('guards' in cap.config));
  const fid = sourceFidelity(cap);
  check('baseline wrist fidelity tight on rig with different bind heading',
    fid.LeftHand.rot.p95 < 8, `p95 ${fid.LeftHand.rot.p95}°`);
  check('baseline ankle fidelity tight', fid.LeftFoot.rot.p95 < 8, `p95 ${fid.LeftFoot.rot.p95}°`);

  // determinism: seek == sequential in baseline
  const seek = captureRun(tgt, clip, { config: baselineOptions(clip), order: 'seek' });
  const d = captureDelta(cap, seek);
  check('baseline: direct seek == sequential playback', d.maxRotDeg < 1e-6, `Δ ${d.maxRotDeg}°`);
}

// -------------------------------------------------------------- sabotage ----
{
  // OLD anchor restored: bend the stored hand rest 18° (the SOMA T-pose
  // trap: a baked restQuat inconsistent with the quat channels). Every
  // rest-RELATIVE quantity self-cancels under this corruption, so the
  // detector is the SEMANTIC-ANCHOR property the fix established: when the
  // source stands at its rest pose (frame 0 here), the character's hand must
  // sit exactly at its bind relation to the forearm. The corrupted anchor
  // breaks that by the injected 18°. (In production the geometry-referenced
  // qa_endeffectors gates catch this class on real meshes.)
  const bad = { ...clip, restQuat: clip.restQuat.map(q => [...q]) };
  const hi = clip.names.indexOf(srcMap.LeftHand);
  const bend = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), THREE.MathUtils.degToRad(18));
  const q0 = new THREE.Quaternion(...clip.restQuat[hi]).multiply(bend);
  bad.restQuat[hi] = [q0.x, q0.y, q0.z, q0.w];
  const relAtRest = (data) => {
    const cap = captureRun(tgt, data, { config: baselineOptions(data) });
    const row = cap.rows[0].roles.LeftHand;                 // frame 0 = source rest pose
    const rt = cap.rt;
    const foreName = rt.R.LeftForeArm, handName = rt.R.LeftHand;
    const bindRel = rt.bindWorldQ[foreName].clone().invert().multiply(rt.bindWorldQ[handName]);
    const nowRel = row.foreQuat.clone().invert().multiply(row.quat);
    return quatAngleDeg(bindRel, nowRel);
  };
  const good = relAtRest(clip), badDev = relAtRest(bad);
  check('sabotage: restored old hand rest anchor FAILS the semantic-anchor gate',
    good < 2 && badDev > 10, `bind deviation at source rest: good ${good.toFixed(1)}° bad ${badDev.toFixed(1)}°`);
}
{
  // over-damped wrists look "stable" but must fail RAW fidelity (the raw
  // metric never scales the expectation by handFollow)
  const fid = sourceFidelity(captureRun(tgt, clip, { config: { ...baselineOptions(clip), handFollow: 0.2 } }));
  check('sabotage: over-damped handFollow FAILS raw source fidelity',
    fid.LeftHand.rot.p95 > 10, `p95 ${fid.LeftHand.rot.p95}°`);
}
{
  // the shipped configuration itself must be history-free: sequential,
  // seek, and reversed application all yield identical poses
  const a = captureRun(tgt, clip, { config: {} });
  const b = captureRun(tgt, clip, { config: {}, order: 'seek' });
  const d = captureDelta(a, b);
  check('shipped config: seek == sequential (no temporal state left)',
    d.maxRotDeg < 1e-6, `Δ ${d.maxRotDeg}°`);
}

// ------------------------------------------------------------ constraint IK --
for (const [label, spec, scale] of [['same-proportions rig', tgtSpec, 0.01],
  ['short-armed rig', shortSpec, 1]]) {
  const target = rigFromBones(buildRig(spec, scale).all);
  const noIK = captureRun(target, clip, { config: baselineOptions(clip) });
  const ikCap = captureRun(target, clip, { config: baselineOptions(clip), ik: true });
  const before = constraintAccuracy(noIK), after = constraintAccuracy(ikCap);
  const b = before.summary.LeftHand, a = after.summary.LeftHand;
  check(`${label}: all authored targets reachable`, a.unreachable === 0,
    `${a.unreachable} unreachable`);
  check(`${label}: IK reaches mapped authored targets (pos)`,
    a.pos !== null && a.pos.max <= 0.005, `before ${b.pos?.max}m -> after ${a.pos?.max}m`);
  check(`${label}: IK applies exact authored orientation at the key`,
    a.rot !== null && a.rot.max <= 0.01, `before ${b.rot?.max}° -> after ${a.rot?.max}°`);
  check(`${label}: IK improves or preserves position accuracy`,
    (a.pos?.max ?? Infinity) <= (b.pos?.max ?? Infinity) + 1e-9);

  // determinism through the IK path
  const seek = captureRun(target, clip, { config: baselineOptions(clip), ik: true, order: 'seek' });
  const d = captureDelta(ikCap, seek);
  check(`${label}: IK seek == sequential`, d.maxRotDeg < 1e-6, `Δ ${d.maxRotDeg}°`);

  // blend window: outside ±6 frames of a key the pose is untouched
  const dOut = quatAngleDeg(noIK.rows[0].roles.LeftHand.quat, ikCap.rows[0].roles.LeftHand.quat);
  check(`${label}: pose untouched outside the blend window`, dOut < 1e-6, `Δ ${dOut}°`);
}

// unreachable: a target far outside any arm span is clamped + reported
{
  const target = rigFromBones(buildRig(tgtSpec, 0.01).all);
  const farClip = { ...clip, constraints: [{ ...eeRecordAt(20), pos: [5, 1, 0] }] };
  const cap = captureRun(target, farClip, { config: baselineOptions(farClip), ik: true });
  const solve = cap.solves.find(s => s.weight === 1);
  check('unreachable target is reported, not silently hit', solve && solve.reachable === false);
  const acc = constraintAccuracy(cap);
  check('unreachable target excluded from accuracy stats and counted',
    acc.summary.LeftHand.unreachable === 1);
}

// root/pinned-effector isolation: solving the LEFT hand must not move the
// right hand, either foot, or the root
{
  const target = rigFromBones(buildRig(tgtSpec, 0.01).all);
  const one = { ...clip, constraints: [eeRecordAt(20)] };
  const noIK = captureRun(target, one, { config: baselineOptions(one) });
  const ikCap = captureRun(target, one, { config: baselineOptions(one), ik: true });
  let moved = 0;
  for (const role of ['RightHand', 'LeftFoot', 'RightFoot']) {
    for (let f = 0; f < N; f++) {
      if (noIK.rows[f].roles[role].pos.distanceTo(ikCap.rows[f].roles[role].pos) > 1e-9) moved++;
    }
  }
  const hipsMoved = noIK.rows.some((r, f) => r.hips.distanceTo(ikCap.rows[f].hips) > 1e-9);
  check('IK never moves other effectors or the root as a side effect',
    moved === 0 && !hipsMoved, `${moved} disturbed frames, hipsMoved=${hipsMoved}`);
}

// in-place playback (prebake mode): with the root's horizontal travel zeroed,
// mapped targets must shed the same displacement — the constrained hand still
// lands exactly even though the clip has walked half a meter away
{
  const travel = snapshotMotion(srcT.orderedBones, (f) => {
    pose(f);
    src.bones.pelvis.position.x = 0.6 * (f / (N - 1));
    src.wrapper.updateMatrixWorld(true);
  }, N, FPS, 'ctravel');
  travel.srcMap = srcMap;
  const idxT = travel.names.indexOf(srcMap.LeftHand);
  travel.constraints = [{ family: 'end-effector', type: 'left-hand', source: 'inline',
    originalType: 'test', required: true, provenance: 'x', frame: 45, role: 'LeftHand',
    pos: [...travel.pos[45][idxT]], quat: [...travel.quat[45][idxT]],
    posConstrained: true, rotConstrained: true }];
  const target = rigFromBones(buildRig(tgtSpec, 0.01).all);
  const cap = captureRun(target, travel, { config: { ...baselineOptions(travel), inPlace: true }, ik: true });
  const acc = constraintAccuracy(cap);
  check('inPlace (prebake) IK still lands the travelled constraint exactly',
    acc.summary.LeftHand.unreachable === 0 && acc.summary.LeftHand.pos.max <= 0.005,
    JSON.stringify(acc.summary.LeftHand));
  // restore the source rig for any later block
  src.all.forEach(b => b.rotation.set(0, 0, 0));
  src.bones.pelvis.position.set(0, 0.86, 0);
  src.wrapper.updateMatrixWorld(true);
}

// mapped targets go through the same yaw/scale/bind transform as the root
{
  const target = rigFromBones(buildRig(tgtSpec, 0.01).all);
  resetBindPose(target);
  const rt = new Retargeter({ ...target, data: clip, ...baselineOptions(clip) });
  const t = mapRecordTargets(rt, clip.constraints)[0];
  const rec = clip.constraints[0];
  const srcRestHips = new THREE.Vector3(...clip.rest[clip.names.indexOf(srcMap.Hips)]);
  const expect = new THREE.Vector3(...rec.pos).sub(srcRestHips)
    .applyQuaternion(rt.rootYaw).multiplyScalar(rt.scaleRoot).add(rt.hipsBindWorldPos);
  check('mapSrcPoint == root-yaw ∘ root-scale ∘ bind translate',
    t.pos.distanceTo(expect) < 1e-9);
}

console.log(failures ? `\n${failures} FAILURES` : '\nconstraint selftest passed (no external assets used)');
process.exit(failures ? 1 : 0);
