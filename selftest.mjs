// Standalone self-test — NO external assets. Proves the module ships: builds
// two synthetic humanoids from scratch (different naming families, different
// bind facings, different proportions, one with a 0.01-scaled armature),
// procedurally animates the source, and runs the full alignment certification
// source→target, plus a mirrored-map sabotage that must FAIL. A prebake leg
// (INTEGRATE.md §9) round-trips the target through a generated GLB.
// Usage: node selftest.mjs   (needs only node_modules: three + @gltf-transform/core)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import * as THREE from 'three';
import { Document, NodeIO } from '@gltf-transform/core';
import {
  rigFromBones, srcMapFromRig, snapshotMotion, mineProbeFrames,
  certifyRig, checkSideConsistency, resetBindPose, DEFAULT_GATES,
} from './align.js';
import { prebake } from './prebake.mjs';

let failures = 0;
function check(label, ok, detail = '') {
  if (ok) console.log(`ok   ${label}`);
  else { failures++; console.log(`FAIL ${label}${detail ? '  — ' + detail : ''}`); }
}

// build a bone hierarchy from { name: [parentName, [x,y,z]] } (parents first);
// wraps in a Group with the given uniform scale (bone offsets given in WORLD
// units are divided by it, like scaled-armature exports store them)
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
  // child local offsets were divided once too much (parents already carry the
  // scale correction exactly once, at the wrapper) — restore all but the root
  for (const [name, [parent]] of Object.entries(spec)) {
    if (parent) bones[name].position.multiplyScalar(armatureScale);
  }
  wrapper.updateMatrixWorld(true);
  return { bones, wrapper, all: Object.values(bones) };
}

// SOURCE: UE-style names, binds facing +X (left limbs at −Z), height ~1.5
// (offsets are parent-relative; world = accumulated)
const L = -0.09, R = +0.09;                       // lateral z for a +X-facing rig
const srcSpec = {
  pelvis: [null, [0, 0.86, 0]],
  spine_01: ['pelvis', [0, 0.14, 0]],
  spine_02: ['spine_01', [0, 0.14, 0]],
  spine_03: ['spine_02', [0, 0.14, 0]],
  neck_01: ['spine_03', [0, 0.10, 0]],
  head: ['neck_01', [0, 0.09, 0]],
  clavicle_l: ['spine_03', [0.01, 0.04, L * 0.5]], upperarm_l: ['clavicle_l', [0, 0, L]],
  lowerarm_l: ['upperarm_l', [0, 0, L * 2.4]], hand_l: ['lowerarm_l', [0, 0, L * 2.2]],
  clavicle_r: ['spine_03', [0.01, 0.04, R * 0.5]], upperarm_r: ['clavicle_r', [0, 0, R]],
  lowerarm_r: ['upperarm_r', [0, 0, R * 2.4]], hand_r: ['lowerarm_r', [0, 0, R * 2.2]],
  thigh_l: ['pelvis', [0, -0.04, L]], calf_l: ['thigh_l', [0, -0.40, 0]],
  foot_l: ['calf_l', [0, -0.38, 0]], ball_l: ['foot_l', [0.12, -0.04, 0]],
  thigh_r: ['pelvis', [0, -0.04, R]], calf_r: ['thigh_r', [0, -0.40, 0]],
  foot_r: ['calf_r', [0, -0.38, 0]], ball_r: ['foot_r', [0.12, -0.04, 0]],
};

// TARGET: Mixamo-style names, binds facing +Z (left limbs at +X), DIFFERENT
// proportions (longer legs, shorter torso, wider shoulders), 0.01 armature
const tgtSpec = {
  Hips: [null, [0, 1.02, 0]],
  Spine: ['Hips', [0, 0.16, 0]],
  Spine1: ['Spine', [0, 0.16, 0]],
  Spine2: ['Spine1', [0, 0.16, 0]],
  Neck: ['Spine2', [0, 0.10, 0]],
  Head: ['Neck', [0, 0.10, 0]],
  LeftShoulder: ['Spine2', [0.07, 0.04, 0]], LeftArm: ['LeftShoulder', [0.13, 0, 0]],
  LeftForeArm: ['LeftArm', [0.27, 0, 0]], LeftHand: ['LeftForeArm', [0.25, 0, 0]],
  RightShoulder: ['Spine2', [-0.07, 0.04, 0]], RightArm: ['RightShoulder', [-0.13, 0, 0]],
  RightForeArm: ['RightArm', [-0.27, 0, 0]], RightHand: ['RightForeArm', [-0.25, 0, 0]],
  LeftUpLeg: ['Hips', [0.11, -0.05, 0]], LeftLeg: ['LeftUpLeg', [0, -0.48, 0]],
  LeftFoot: ['LeftLeg', [0, -0.44, 0]], LeftToeBase: ['LeftFoot', [0, -0.05, 0.13]],
  RightUpLeg: ['Hips', [-0.11, -0.05, 0]], RightLeg: ['RightUpLeg', [0, -0.48, 0]],
  RightFoot: ['RightLeg', [0, -0.44, 0]], RightToeBase: ['RightFoot', [0, -0.05, 0.13]],
};

const src = buildRig(srcSpec);
const srcT = rigFromBones(src.all);
check('source rig resolves (UE names, +X facing)', srcT.rig.ok, 'missing=' + srcT.rig.missing);

const tgt = buildRig(tgtSpec, 0.01);
const tgtT = rigFromBones(tgt.all);
check('target rig resolves (Mixamo names, +Z facing, 0.01 armature)', tgtT.rig.ok,
  'missing=' + tgtT.rig.missing);
check('target world scale sane despite armature', Math.abs(
  tgt.bones.Hips.getWorldPosition(new THREE.Vector3()).y - 1.02) < 1e-6);

// procedurally animate the source: torso twist, arm swings + elbow bends,
// leg swings + knee bends, slight pelvis bob — moderate, humanoid-plausible
const N = 90, FPS = 30;
const basePelvisY = src.bones.pelvis.position.y;
function pose(f) {
  const t = f / FPS;
  const s = (w, ph = 0) => Math.sin(2 * Math.PI * w * t + ph);
  src.bones.pelvis.position.y = basePelvisY - 0.05 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.spine_02.rotation.y = 0.30 * s(0.5);
  src.bones.upperarm_l.rotation.x = 0.55 * s(1);          // swing about the facing axis
  src.bones.upperarm_r.rotation.x = -0.55 * s(1, Math.PI / 3);
  src.bones.lowerarm_l.rotation.y = 0.45 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.lowerarm_r.rotation.y = -0.45 * (1 - Math.cos(2 * Math.PI * t + 1)) / 2;
  src.bones.hand_l.rotation.z = 0.30 * s(1.5);
  src.bones.thigh_l.rotation.z = 0.45 * s(1);              // hip flexion about lateral (z)
  src.bones.thigh_r.rotation.z = -0.45 * s(1);
  src.bones.calf_l.rotation.z = -0.35 * (1 - Math.cos(2 * Math.PI * t)) / 2;
  src.bones.calf_r.rotation.z = -0.35 * (1 + Math.cos(2 * Math.PI * t)) / 2;
  src.bones.foot_l.rotation.z = 0.15 * s(1, 1);
  src.wrapper.updateMatrixWorld(true);
}
const motion = snapshotMotion(srcT.orderedBones, pose, N, FPS, 'selftest');
check('motion snapshot has quats + rest', Array.isArray(motion.quat) && motion.rest.length === srcT.orderedBones.length);

const srcMap = srcMapFromRig(srcT.rig.map);
const probes = mineProbeFrames([motion], srcMap);
check('probes mined from synthetic clip', probes.numFrames === 7, `got ${probes.numFrames}`);

// prone probe: supine pose (horizontal hips→chest axis) with the hands held
// above the belly. Locks in the capsule-clearance fix: the gate must measure
// the full radial clearance — an upright-torso shortcut (zeroing the radial's
// Y component) reads hands-over-chest as "inside the torso" and false-fails
// every lying pose (fall/knockdown clips) while leaving upright probes intact.
{
  const b = src.bones;
  // point `bone` so the direction toward its child (dLocal in bind space) hits
  // the world direction w: q = pq⁻¹ · R(pq·dLocal → w) · pq
  const aimWorld = (bone, dLocal, w) => {
    src.wrapper.updateMatrixWorld(true);
    const pq = bone.parent.getWorldQuaternion(new THREE.Quaternion());
    const d0 = dLocal.clone().applyQuaternion(pq).normalize();
    const R = new THREE.Quaternion().setFromUnitVectors(d0, w.clone().normalize());
    bone.quaternion.copy(pq.clone().invert().multiply(R).multiply(pq));
  };
  const proneFrame = () => {
    src.all.forEach(bone => bone.rotation.set(0, 0, 0));
    b.pelvis.position.set(0, 0.15, 0);                 // lying height
    b.pelvis.rotation.z = Math.PI / 2;                 // supine: torso axis → −X, chest up
    // forearms folded across the chest: elbows out and lifted, each wrist
    // crossing just past the midline while resting clear above the ribcage —
    // the clearance is almost purely vertical (relative to the world), which
    // is exactly the component an upright-torso shortcut throws away
    aimWorld(b.upperarm_l, new THREE.Vector3(0, 0, -1), new THREE.Vector3(0.88, 0.48, 0));
    aimWorld(b.lowerarm_l, new THREE.Vector3(0, 0, -1), new THREE.Vector3(0.37, 0.28, 0.88));
    aimWorld(b.upperarm_r, new THREE.Vector3(0, 0, 1), new THREE.Vector3(0.88, 0.48, 0));
    aimWorld(b.lowerarm_r, new THREE.Vector3(0, 0, 1), new THREE.Vector3(0.37, 0.28, -0.88));
    src.wrapper.updateMatrixWorld(true);
  };
  const snap = snapshotMotion(srcT.orderedBones, proneFrame, 1, FPS, 'prone');
  probes.pos.push(snap.pos[0]);
  probes.quat.push(snap.quat[0]);
  probes.probeTags.push('prone:sprawl');
  probes.probeGrounded.push({ Left: false, Right: false });
  probes.numFrames += 1;
}

// full certification source→target
const cert = certifyRig(tgtT, probes, { srcMap });
check('synthetic source→target certifies', cert.pass, cert.failures.join('; '));
check('  side consistency verified', cert.gates.sideConsistency === true);
check('  bone stretch ~0', cert.gates.boneStretchPct < 0.1, `${cert.gates.boneStretchPct}%`);
check('  round trip tight', cert.gates.roundTripMean < 0.05 && cert.gates.roundTripP95 < 0.10,
  `mean ${cert.gates.roundTripMean} p95 ${cert.gates.roundTripP95}`);
console.log('     gates:', JSON.stringify(cert.gates));

// the prone probe must clear the capsule gate on its true (un-zeroed) radial —
// re-introducing the upright-torso assumption in measureGates drops it to ~0
{
  const prone = cert.probes.find(p => p.tag === 'prone:sprawl');
  check('prone probe capsule clearance measured', prone && prone.capsuleClearance !== null);
  check('prone probe passes capsule gate', prone &&
    prone.capsuleClearance >= DEFAULT_GATES.capsuleClearance,
  `clearance ${prone?.capsuleClearance} < ${DEFAULT_GATES.capsuleClearance}`);
}

// sabotage: mirrored legs must FAIL certification via the absolute side gate
{
  const t2 = rigFromBones(buildRig(tgtSpec, 0.01).all);
  const m = t2.rig.map;
  for (const r of ['UpLeg', 'Leg', 'Foot', 'ToeBase']) {
    [m['Left' + r], m['Right' + r]] = [m['Right' + r], m['Left' + r]];
  }
  const side = checkSideConsistency(t2);
  check('sabotage mirrored legs: side gate fires', side !== null && !side.ok);
  const cert2 = certifyRig(t2, probes, { srcMap });
  check('sabotage mirrored legs: certification FAILS', !cert2.pass);
}

resetBindPose(tgtT);

// ---------------------------------------------------------------- prebake leg
// INTEGRATE.md §9: write the target skeleton to a GLB and the synthetic source
// animation (with real forward travel) to a baked-clip JSON + manifest, run
// prebake, and verify the GLB gained the animation and the root-motion export
// reproduces the source pelvis curve × scaleRoot.
{
  // pristine bind, then re-pose with horizontal travel so root motion is real
  src.all.forEach(b => b.rotation.set(0, 0, 0));
  src.bones.pelvis.position.set(0, basePelvisY, 0);
  src.wrapper.updateMatrixWorld(true);
  const travelClip = snapshotMotion(srcT.orderedBones, (f) => {
    pose(f);
    src.bones.pelvis.position.x = 0.02 * f;
    src.wrapper.updateMatrixWorld(true);
  }, N, FPS, 'selftest_clip');

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'mb-selftest-'));
  try {
    // target skeleton as a GLB (joints-only skin is all loadGLBBones needs)
    const doc = new Document();
    doc.createBuffer();
    const scene = doc.createScene('scene');
    const skin = doc.createSkin('skin');
    const nodes = {};
    for (const [name, [parent, pos]] of Object.entries(tgtSpec)) {
      nodes[name] = doc.createNode(name).setTranslation(pos);
      (parent ? nodes[parent] : scene).addChild(nodes[name]);
      skin.addJoint(nodes[name]);
    }
    const glbPath = path.join(dir, 'char.glb');
    await new NodeIO().write(glbPath, doc);

    fs.writeFileSync(path.join(dir, 'selftest_clip.json'),
      JSON.stringify({ ...travelClip, srcMap }));
    const manifestPath = path.join(dir, 'manifest.json');
    fs.writeFileSync(manifestPath, JSON.stringify({ moves: [{
      name: 'selftest_clip', file: 'selftest_clip.json', loop: true,
      frame_data: { startup: 10, active: [10, 14], recovery: N - 14 },
    }] }));

    const { outPath, rmPath } = await prebake({
      glb: glbPath, manifest: manifestPath,
      out: path.join(dir, 'char_anim.glb'), log: () => {},
    });

    const outDoc = await new NodeIO().read(outPath);
    const anims = outDoc.getRoot().listAnimations();
    check('prebake: one glTF animation per manifest clip',
      anims.length === 1 && anims[0].getName() === 'selftest_clip');
    const channels = anims[0]?.listChannels() ?? [];
    const rot = channels.filter(c => c.getTargetPath() === 'rotation');
    const hipsT = channels.find(c => c.getTargetPath() === 'translation'
      && c.getTargetNode()?.getName() === 'Hips');
    check('prebake: rotation channel per joint', rot.length === Object.keys(tgtSpec).length,
      `got ${rot.length} of ${Object.keys(tgtSpec).length}`);
    check('prebake: hips translation channel present', !!hipsT);
    check('prebake: sampler covers every frame',
      !!hipsT && hipsT.getSampler().getInput().getArray().length === N);

    const rm = JSON.parse(fs.readFileSync(rmPath, 'utf8'));
    const clip = rm.clips.selftest_clip;
    const k = rm.scaleRoot;
    check('prebake: scaleRoot sane', Number.isFinite(k) && k > 0.5 && k < 2, `scaleRoot ${k}`);
    const pi = travelClip.names.indexOf(srcMap.Hips);
    let worst = 0;
    for (let f = 0; f < N; f++) {
      worst = Math.max(worst,
        Math.abs(clip.pelvisXZ[f][0] - travelClip.pos[f][pi][0] * k),
        Math.abs(clip.pelvisXZ[f][1] - travelClip.pos[f][pi][2] * k),
        Math.abs(clip.hipY[f] - travelClip.pos[f][pi][1] * k));
    }
    check('prebake: root motion matches source pelvis curve × scaleRoot',
      worst < 1e-9, `worst |err| ${worst}`);
    check('prebake: loop flag + frame data pass through',
      clip.loop === true && clip.frameData?.startup === 10);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

console.log(failures ? `\n${failures} FAILURES` : '\nselftest passed (no external assets used)');
process.exit(failures ? 1 : 0);
