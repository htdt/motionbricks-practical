import * as THREE from 'three';
import { resolveRig } from './rigmap.js';
import { Retargeter, buildBoneOrder, SOMA_SRC } from './retarget.js';

// Skeleton ALIGNMENT & CERTIFICATION (pipeline Stage 1, see ALIGN.md).
//
// Aligns two arbitrary humanoid skeletons: a SOURCE described by motion data
// (per-frame world joint positions + optional world quats — the SOMA canonical
// skeleton from Kimodo, or any GLB rig sampled through its own animation)
// and a TARGET GLB rig resolved by rigmap. The forward map is the position-based
// Retargeter; this module adds what certification needs on top:
//
//  - srcMapFromRig / snapshotMotion: make ANY resolved rig usable as a source,
//    so alignment is genuinely skeleton↔skeleton, not SOMA→character only.
//  - mineProbeFrames: a deterministic probe-pose battery pulled from real clips
//    (kick apex, crouch, jump top, max reach, max torso twist, rest, …).
//  - recoverCanonicalPose: the INVERSE map — reconstruct source-skeleton joint
//    positions from the POSED TARGET alone (static calibration only: bind
//    frames, segment lengths, scale, yaw rebase). Round-trip MAE = how much of
//    the source pose survives the forward transfer, in source meters.
//  - absolute gates (round-trip is necessary but NOT sufficient): bone-length
//    stretch, foot flatness vs the source's own foot tilt, per-bone twist
//    limits, torso-capsule clearance.
//  - calibrateHandClamp: rest-pose wrist-bias measurement -> per-character
//    clamp tightening (the one continuous parameter the retargeter exposes).
//  - certifyRig: run the battery, emit a retarget_certificate-style report.

// canonical source roles measured by the inverse map / round trip
export const MEASURE_ROLES = [
  'Hips', 'Chest',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot',
  'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
];
// chains reconstructed by the inverse map: parent role -> child roles
const CHAIN = {
  Hips: ['LeftUpLeg', 'RightUpLeg', 'Chest'],
  Chest: ['LeftArm', 'RightArm'],
  LeftUpLeg: ['LeftLeg'], LeftLeg: ['LeftFoot'],
  RightUpLeg: ['RightLeg'], RightLeg: ['RightFoot'],
  LeftArm: ['LeftForeArm'], LeftForeArm: ['LeftHand'],
  RightArm: ['RightForeArm'], RightForeArm: ['RightHand'],
};

export const DEFAULT_GATES = {
  boneStretchPct: 1.0,      // max bone-length change vs bind, % (FK should give ~0)
  footFlatDeg: 6.0,         // grounded probes: |char sole tilt − source sole tilt|
  footGroundFrac: 0.10,     // grounded probes: |foot height − bind height| / hip height
  twistFrac: 1.0,           // max single-bone twist as a fraction of its anatomical limit
  capsuleClearance: 0.85,   // min hand/elbow distance to torso axis, × capsuleR
  roundTripMean: 0.05,      // m at source scale, mean over probes+roles
  roundTripP95: 0.10,       // m at source scale, 95th percentile
};

// anatomical twist limits (deg) about each bone's own axis, vs its bind relation.
// ForeArm is large on purpose: pronation/supination IS forearm twist. Hands are
// clamped to 85° by the retargeter, so 92° means the clamp itself broke. Feet are
// gated by footFlat when grounded and legitimately free when airborne.
export const TWIST_LIMITS = {
  UpLeg: 100, Leg: 95, Arm: 130, ForeArm: 165, Hand: 92,
};

const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function frameQ(origin, upPt, rA, rB) {
  const up = upPt.clone().sub(origin).normalize();
  const r0 = rA.clone().sub(rB).normalize();
  const fwd = new THREE.Vector3().crossVectors(r0, up).normalize();
  const right = new THREE.Vector3().crossVectors(up, fwd).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd));
}

// ---------------------------------------------------------------- rig helpers

// resolve a rig and package everything the Retargeter needs
export function rigFromBones(allBones) {
  const rig = resolveRig(allBones);
  if (!rig.map.Hips) throw new Error('align: cannot resolve Hips; missing=' + rig.missing);
  const byName = {};
  allBones.forEach(b => { byName[b.name] = b; });
  const hips = byName[rig.map.Hips];
  if (!hips.parent) {                    // retargeter needs a parent space for the root
    const g = new THREE.Group();
    g.add(hips);
    g.updateMatrixWorld(true);
  }
  const orderedBones = buildBoneOrder(hips);
  const bones = {};
  orderedBones.forEach(b => { bones[b.name] = b; });
  // snapshot the pristine bind pose: the Retargeter reads "bind" from the bones'
  // CURRENT state at construction, and applyFrame leaves the skeleton posed — so
  // any re-certification of a reused target must restore this first.
  const bindPose = orderedBones.map(b => ({
    b, p: b.position.clone(), q: b.quaternion.clone(), s: b.scale.clone(),
  }));
  return { rig, bones, orderedBones, hips, hipsParent: hips.parent, bindPose };
}

// restore the bind pose captured by rigFromBones
export function resetBindPose(target) {
  if (!target.bindPose) return;
  for (const { b, p, q, s } of target.bindPose) {
    b.position.copy(p); b.quaternion.copy(q); b.scale.copy(s);
  }
  target.hips.updateWorldMatrix(true, true);
}

// canonical source map for a rig acting as the MOTION SOURCE.
// Anchor roles fall back inside the Retargeter when identical to limb roles.
export function srcMapFromRig(rigMap) {
  const S = {};
  for (const role of MEASURE_ROLES) if (rigMap[role]) S[role] = rigMap[role];
  S.Chest = rigMap.Chest ?? rigMap.Hips;
  S.LeftHipAnchor = rigMap.LeftUpLeg; S.RightHipAnchor = rigMap.RightUpLeg;
  S.LeftShoulderAnchor = rigMap.LeftArm; S.RightShoulderAnchor = rigMap.RightArm;
  return S;
}

// Sample a POSED rig into retargeter motion data. poseFrame(f) must put the
// skeleton into frame f (e.g. glbskel's animationSampler.apply); world matrices
// must be current when it returns. Captures world pos + quat of every bone.
export function snapshotMotion(orderedBones, poseFrame, numFrames, fps = 30, mode = 'glb') {
  const names = orderedBones.map(b => b.name);
  const grab = () => orderedBones.map(b => {
    const p = b.getWorldPosition(new THREE.Vector3());
    return [p.x, p.y, p.z];
  });
  const grabQ = () => orderedBones.map(b => {
    const q = b.getWorldQuaternion(new THREE.Quaternion());
    return [q.x, q.y, q.z, q.w];
  });
  // rest = the un-posed skeleton as currently bound
  orderedBones[0].updateWorldMatrix(true, true);
  const rest = grab(), restQuat = grabQ();
  const pos = [], quat = [];
  for (let f = 0; f < numFrames; f++) {
    poseFrame(f);
    pos.push(grab()); quat.push(grabQ());
  }
  return { names, fps, numFrames, mode, rest, restQuat, pos, quat };
}

// -------------------------------------------------------------- probe mining

// Deterministic probe battery from real motion clips: for each clip pick the
// most articulated moments. Returns probe motion data (same shape as clip data,
// so it feeds a Retargeter directly) + tags. Frame 0 is always the source REST
// pose (the closest thing to a T-pose probe the source skeleton defines).
export function mineProbeFrames(clips, srcMap = SOMA_SRC) {
  const first = clips[0];
  const idx = {}; first.names.forEach((n, i) => { idx[n] = i; });
  const S = srcMap;
  const j = (P, role) => V(P[idx[S[role]]]);

  const rows = [{ tag: 'rest', pos: first.rest, quat: first.restQuat, grounded: { Left: true, Right: true } }];
  for (const clip of clips) {
    const N = clip.numFrames;
    const restFootY = Math.min(j(clip.rest, 'LeftFoot').y, j(clip.rest, 'RightFoot').y);
    const score = {
      kickL: (P) => j(P, 'LeftFoot').y,
      kickR: (P) => j(P, 'RightFoot').y,
      crouch: (P) => -j(P, 'Hips').y,
      jumpTop: (P) => Math.min(j(P, 'LeftFoot').y, j(P, 'RightFoot').y),
      reach: (P) => j(P, 'LeftHand').distanceTo(j(P, 'RightHand')),
      twist: (P) => {
        const hipDir = j(P, 'LeftUpLeg').sub(j(P, 'RightUpLeg')).setY(0).normalize();
        const shDir = j(P, 'LeftArm').sub(j(P, 'RightArm')).setY(0).normalize();
        return Math.abs(Math.asin(THREE.MathUtils.clamp(hipDir.clone().cross(shDir).y, -1, 1)));
      },
    };
    for (const [tag, fn] of Object.entries(score)) {
      let best = -Infinity, bf = 0;
      for (let f = 0; f < N; f++) { const s = fn(clip.pos[f]); if (s > best) { best = s; bf = f; } }
      const P = clip.pos[bf];
      rows.push({
        tag: `${clip.mode ?? 'clip'}:${tag}@${bf}`,
        grounded: {
          Left: j(P, 'LeftFoot').y < restFootY + 0.03,
          Right: j(P, 'RightFoot').y < restFootY + 0.03,
        },
        pos: P, quat: clip.quat ? clip.quat[bf] : undefined,
      });
    }
  }
  const hasQuat = rows.every(r => Array.isArray(r.quat));
  return {
    names: first.names, fps: first.fps ?? 30, numFrames: rows.length, mode: 'probes',
    rest: first.rest, restQuat: first.restQuat,
    pos: rows.map(r => r.pos),
    quat: hasQuat ? rows.map(r => r.quat) : undefined,
    probeTags: rows.map(r => r.tag), probeGrounded: rows.map(r => r.grounded),
  };
}

// ----------------------------------------------------------- inverse recovery

// Reconstruct SOURCE-skeleton joint positions from the retargeter's posed
// TARGET, using only static calibration (bind/rest frames, source segment
// lengths + rest offsets, root scale, yaw rebase). Call right after
// rt.applyFrame(f). Returns { role: THREE.Vector3 } in source world space.
//
// Two recovery rules:
//  - CHAIN-ROOT ANCHORS (Chest, UpLeg, Arm) ride their body frame rigidly in
//    the source (pelvis→hip, torso→shoulder are near-rigid offsets), so they
//    are recovered by rotating the source REST offset with the body delta read
//    off the posed hips/chest bones. Mapping the target's own chest→shoulder
//    direction instead would bake the rigs' proportion difference into the
//    metric as a constant false error.
//  - LIMB SEGMENTS are recovered by inverting the aim map: the target's
//    segment direction, conjugated back into the source frame, times the
//    source segment length.
export function recoverCanonicalPose(rt) {
  const S = rt.S, R = rt.R, rest = rt.data.rest, idx = rt.idx;
  const srcRest = (role) => V(rest[idx[S[role]]]);
  const cpos = (role) => rt._framePos(rt.bones[R[role]]).clone();

  // body deltas NOW, read from the posed bones (the forward pass writes exactly
  // pelvisDelta·bindWorldQ into hips, chestDelta·bindWorldQ into the chain top);
  // un-yaw-rebase to get back the source's own deltas.
  const unYaw = (q) => rt.rootYawInv.clone().multiply(q).multiply(rt.rootYaw);
  const boneQ = (b) => (rt.animWorld[b.uuid] ?? b.getWorldQuaternion(new THREE.Quaternion())).clone();
  const pelvisDeltaChar = boneQ(rt.hips).multiply(rt.bindWorldQ[rt.hips.name].clone().invert());
  const pelvisDeltaSrc = unYaw(pelvisDeltaChar);
  const chestDeltaChar = boneQ(rt.chestBone).multiply(rt.bindWorldQ[rt.chestBone.name].clone().invert());
  const chestDeltaSrc = unYaw(chestDeltaChar);
  const FpRec = pelvisDeltaSrc.clone().multiply(rt.FpRest);
  const FcNow = pelvisDeltaChar.clone().multiply(rt.FcBind);
  const toSrc = FpRec.clone().multiply(FcNow.clone().invert());   // char world dir -> source world dir

  // root: invert the forward translation map
  const disp = cpos('Hips').sub(rt.hipsBindWorldPos)
    .applyQuaternion(rt.rootYaw.clone().invert())
    .multiplyScalar(1 / rt.scaleRoot);
  const out = { Hips: srcRest('Hips').add(disp) };

  const ANCHOR_DELTA = { Chest: pelvisDeltaSrc, LeftUpLeg: pelvisDeltaSrc, RightUpLeg: pelvisDeltaSrc,
    LeftArm: chestDeltaSrc, RightArm: chestDeltaSrc };
  const walk = (parentRole) => {
    for (const child of CHAIN[parentRole] ?? []) {
      if (!(S[child] in idx) || (child !== 'Chest' && !R[child])) continue;
      if (ANCHOR_DELTA[child]) {
        const off = srcRest(child).sub(srcRest(parentRole)).applyQuaternion(ANCHOR_DELTA[child]);
        out[child] = out[parentRole].clone().add(off);
      } else {
        const segLen = srcRest(child).distanceTo(srcRest(parentRole));
        const dChar = cpos(child).sub(cpos(parentRole));
        const dSrc = dChar.lengthSq() > 1e-12
          ? dChar.normalize().applyQuaternion(toSrc) : new THREE.Vector3(0, 1, 0);
        out[child] = out[parentRole].clone().add(dSrc.multiplyScalar(segLen));
      }
      walk(child);
    }
  };
  walk('Hips');
  return out;
}

// round-trip position error for one applied frame: recovered source joints vs
// the actual source frame. Returns { perRole, mean, max } in source meters.
export function roundTripError(rt, f) {
  rt.applyFrame(f);
  const rec = recoverCanonicalPose(rt);
  const P = rt.data.pos[((f % rt.numFrames) + rt.numFrames) % rt.numFrames];
  const perRole = {};
  let sum = 0, n = 0, max = 0;
  for (const role of MEASURE_ROLES) {
    if (!rec[role] || !(rt.S[role] in rt.idx)) continue;
    const e = rec[role].distanceTo(V(P[rt.idx[rt.S[role]]]));
    perRole[role] = e; sum += e; n++; if (e > max) max = e;
  }
  return { perRole, mean: sum / Math.max(n, 1), max };
}

// ------------------------------------------------------------------- gates

const LIMB_ROLES = ['LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg',
  'LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm', 'LeftHand', 'RightHand'];

// per-frame gate measurements on the posed target. Call after rt.applyFrame(f).
export function measureGates(rt, f, grounded) {
  const R = rt.R;
  const m = { boneStretchPct: 0, twistFrac: 0, twistDeg: 0, twistWorstRole: null,
    footFlatDeg: null, footGroundFrac: null, capsuleClearance: null };

  // bone stretch: world length of every aimed segment vs bind length
  for (const name in rt.aimByBone) {
    const child = rt.aimByBone[name][3];
    const len = rt._framePos(rt.bones[child]).distanceTo(rt._framePos(rt.bones[name]));
    const pct = Math.abs(len - rt.bindLen[name]) / rt.bindLen[name] * 100;
    if (pct > m.boneStretchPct) m.boneStretchPct = pct;
  }

  // twist vs bind, about each bone's own bind axis, as a fraction of the
  // role's anatomical limit
  for (const role of LIMB_ROLES) {
    const name = R[role];
    if (!name || !rt.bones[name]) continue;
    const limit = TWIST_LIMITS[role.replace(/^(Left|Right)/, '')];
    if (!limit) continue;
    const b = rt.bones[name];
    const parentQ = rt.animWorld[b.parent.uuid] ?? rt.bindWorldQ[b.parent.name] ?? new THREE.Quaternion();
    const local = parentQ.clone().invert().multiply(rt.animWorld[b.uuid]);
    const rel = rt.bindLocalQ[name].clone().invert().multiply(local);
    if (rel.w < 0) rel.set(-rel.x, -rel.y, -rel.z, -rel.w);
    let axis = rt.clampAxis?.[name];
    if (!axis) {
      const child = rt.aimByBone[name]?.[3];
      const p0 = rt._framePos(b);
      const dirWorld = child ? rt._framePos(rt.bones[child]).clone().sub(p0)
        : p0.clone().sub(rt._framePos(b.parent));
      axis = dirWorld.normalize().applyQuaternion(rt.animWorld[b.uuid].clone().invert()).normalize();
    }
    const r = new THREE.Vector3(rel.x, rel.y, rel.z);
    const proj = axis.clone().multiplyScalar(r.dot(axis));
    const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, rel.w).normalize();
    const deg = THREE.MathUtils.radToDeg(2 * Math.acos(THREE.MathUtils.clamp(Math.abs(twist.w), -1, 1)));
    if (deg / limit > m.twistFrac) { m.twistFrac = deg / limit; m.twistDeg = deg; m.twistWorstRole = role; }
  }

  // foot flatness: for each foot GROUNDED IN THE SOURCE this frame, the char
  // sole tilt must match the source's own sole tilt.
  // tilt = angle the bind up-axis has rotated away from world up.
  {
    const upTilt = (qNow, qBind) => {
      const up = new THREE.Vector3(0, 1, 0)
        .applyQuaternion(qBind.clone().invert()).applyQuaternion(qNow);
      return THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(up.y, -1, 1)));
    };
    let worst = null;
    for (const side of ['Left', 'Right']) {
      if (!grounded?.[side]) continue;
      const name = R[side + 'Foot'];
      if (!name) continue;
      const charTilt = upTilt(rt.animWorld[rt.bones[name].uuid], rt.bindWorldQ[name]);
      let srcTilt = 0;
      const srcJoint = rt.S[side + 'Foot'];
      if (rt.hasQuat && srcJoint in rt.idx) {
        const q = rt.data.quat[((f % rt.numFrames) + rt.numFrames) % rt.numFrames][rt.idx[srcJoint]];
        const q0 = rt.data.restQuat[rt.idx[srcJoint]];
        srcTilt = upTilt(new THREE.Quaternion(...q), new THREE.Quaternion(...q0));
      }
      worst = Math.max(worst ?? 0, Math.abs(charTilt - srcTilt));

      // grounding: a foot planted in the source must sit at its bind height on
      // the character too (as a fraction of hip height). Catches wrong root
      // scale and gross leg-proportion errors — both cycle-consistent.
      const bindY = rt.groundBones.find(([gb]) => gb.name === name)?.[1];
      if (bindY !== undefined) {
        const frac = Math.abs(rt._framePos(rt.bones[name]).y - bindY) / rt.hipsBindWorldPos.y;
        m.footGroundFrac = Math.max(m.footGroundFrac ?? 0, frac);
      }
    }
    m.footFlatDeg = worst;
  }

  // capsule clearance: hand/elbow horizontal distance to the hips→chest axis
  const capA = rt._framePos(rt.hips), capB = rt._framePos(rt.chestBone);
  let minClear = Infinity;
  for (const role of ['LeftHand', 'RightHand', 'LeftForeArm', 'RightForeArm']) {
    const name = R[role];
    if (!name || !rt.bones[name]) continue;
    const p = rt._framePos(rt.bones[name]);
    const ab = capB.clone().sub(capA);
    const t = THREE.MathUtils.clamp(p.clone().sub(capA).dot(ab) / ab.lengthSq(), 0, 1);
    if (t <= 0 || t >= 1) continue;                    // beside, not within, the torso span
    // radial is already ⊥ to the torso axis (closest-point construction), so
    // its length IS the clearance. Do not zero radial.y: that assumes an
    // upright torso — for prone/ground poses (horizontal hips→chest axis) it
    // discards true clearance and false-fails the gate. For a vertical axis
    // the perpendicular has ~no Y component, so upright behavior is unchanged.
    const radial = p.clone().sub(capA.clone().add(ab.multiplyScalar(t)));
    minClear = Math.min(minClear, radial.length() / rt.capsuleR);
  }
  m.capsuleClearance = Number.isFinite(minClear) ? minClear : null;
  return m;
}

// Absolute L/R side check at BIND, anchored on the toe (forward) direction —
// deliberately independent of the role map and of the retarget frames, because
// a mirrored role map is perfectly cycle-consistent (forward and inverse both
// mirror) and slips through every relative metric. Requires toe bones; returns
// null (not checked) without them.
export function checkSideConsistency(target) {
  const { rig, bones } = target;
  const R = rig.map;
  if (!R.LeftToeBase || !R.RightToeBase || !R.LeftFoot || !R.RightFoot) return null;
  const wp = (n) => bones[n].getWorldPosition(new THREE.Vector3());
  const fwd = wp(R.LeftToeBase).sub(wp(R.LeftFoot))
    .add(wp(R.RightToeBase).sub(wp(R.RightFoot)));
  fwd.y = 0;
  if (fwd.lengthSq() < 1e-8) return null;
  fwd.normalize();
  const leftDir = new THREE.Vector3(0, 1, 0).cross(fwd).normalize();
  const pairs = [['LeftUpLeg', 'RightUpLeg'], ['LeftArm', 'RightArm'],
    ['LeftFoot', 'RightFoot'], ['LeftHand', 'RightHand']];
  const wrong = [];
  for (const [l, r] of pairs) {
    if (!R[l] || !R[r]) continue;
    if (wp(R[l]).sub(wp(R[r])).dot(leftDir) <= 0) wrong.push(l + '/' + r);
  }
  return { ok: wrong.length === 0, wrong };
}

// ----------------------------------------------------------- hand calibration

// Rest-pose wrist TWIST bias: with the source at rest, any twist of the hand
// about its own axis (vs its bind relation to the forearm) is a systematic
// source↔char wrist-convention offset — the "twisted wrist" family of bugs.
// Tighten the twist clamp so bias + motion can't exceed anatomy. Swing is NOT
// calibrated here: at source rest the arms hang while a T-pose bind sticks them
// out, so the rest swing deviation is pose mismatch, not a convention bias.
export function calibrateHandClamp(mk) {
  const rt = mk({});
  if (!rt.hasQuat) return null;
  rt.applyFrame(0);                                   // probe frame 0 == rest
  let twist = 0;
  for (const role of ['LeftHand', 'RightHand']) {
    const name = rt.R[role];
    if (!name || !rt.bones[name] || !rt.fullqByBone[name]) continue;
    const b = rt.bones[name];
    const parentQ = rt.animWorld[b.parent.uuid];
    const local = parentQ.clone().invert().multiply(rt.animWorld[b.uuid]);
    const rel = rt.bindLocalQ[name].clone().invert().multiply(local);
    if (rel.w < 0) rel.set(-rel.x, -rel.y, -rel.z, -rel.w);
    const axis = rt.clampAxis[name] ?? new THREE.Vector3(0, 1, 0);
    const r = new THREE.Vector3(rel.x, rel.y, rel.z);
    const proj = axis.clone().multiplyScalar(r.dot(axis));
    const tq = new THREE.Quaternion(proj.x, proj.y, proj.z, rel.w).normalize();
    const ang = 2 * Math.acos(THREE.MathUtils.clamp(Math.abs(tq.w), -1, 1));
    twist = Math.max(twist, THREE.MathUtils.radToDeg(ang));
  }
  return {
    biasTwistDeg: twist,
    handClampDeg: { twist: THREE.MathUtils.clamp(85 - twist, 35, 85) },
  };
}

// -------------------------------------------------------------- certification

// Run the full battery for one target rig against a source probe set.
//  target: { rig, bones, orderedBones, hips, hipsParent } from rigFromBones()
//  probes: mineProbeFrames() output
//  opts:   { srcMap, gates, calibrate }
export function certifyRig(target, probes, opts = {}) {
  const gates = { ...DEFAULT_GATES, ...(opts.gates ?? {}) };
  const mk = (extra) => {
    resetBindPose(target);               // Retargeter reads bind from current state
    return new Retargeter({
      bones: target.bones, orderedBones: target.orderedBones,
      hips: target.hips, hipsParent: target.hipsParent,
      rig: target.rig, data: probes, srcMap: opts.srcMap,
      // probes are UNRELATED poses at consecutive indices — the temporal
      // continuity guard must not smooth across them; gates measure raw transfer
      guards: { continuity: false, ...(opts.guards ?? {}) },
      ...extra,
    });
  };

  const calib = opts.calibrate === false ? null : calibrateHandClamp(mk);
  const rt = mk(calib ? { handClampDeg: calib.handClampDeg } : {});
  const side = checkSideConsistency(target);

  const probeResults = [];
  const rtErrs = [];
  const agg = { boneStretchPct: 0, twistFrac: 0, twistDeg: 0, twistWorstRole: null,
    footFlatDeg: 0, footGroundFrac: 0, capsuleClearance: Infinity };
  for (let f = 0; f < probes.numFrames; f++) {
    const grounded = probes.probeGrounded?.[f] ?? false;
    const rte = roundTripError(rt, f);                 // applies the frame
    const g = measureGates(rt, f, grounded);
    for (const e of Object.values(rte.perRole)) rtErrs.push(e);
    agg.boneStretchPct = Math.max(agg.boneStretchPct, g.boneStretchPct);
    if (g.twistFrac > agg.twistFrac) {
      agg.twistFrac = g.twistFrac; agg.twistDeg = g.twistDeg; agg.twistWorstRole = g.twistWorstRole;
    }
    if (g.footFlatDeg !== null) agg.footFlatDeg = Math.max(agg.footFlatDeg, g.footFlatDeg);
    if (g.footGroundFrac !== null) agg.footGroundFrac = Math.max(agg.footGroundFrac, g.footGroundFrac);
    if (g.capsuleClearance !== null) agg.capsuleClearance = Math.min(agg.capsuleClearance, g.capsuleClearance);
    probeResults.push({
      tag: probes.probeTags?.[f] ?? String(f), grounded,
      roundTripMean: +rte.mean.toFixed(4), roundTripMax: +rte.max.toFixed(4),
      boneStretchPct: +g.boneStretchPct.toFixed(3),
      twistFrac: +g.twistFrac.toFixed(2), twistDeg: +g.twistDeg.toFixed(1),
      twistWorstRole: g.twistWorstRole,
      footFlatDeg: g.footFlatDeg === null ? null : +g.footFlatDeg.toFixed(1),
      footGroundFrac: g.footGroundFrac === null ? null : +g.footGroundFrac.toFixed(3),
      capsuleClearance: g.capsuleClearance === null ? null : +g.capsuleClearance.toFixed(2),
    });
  }
  rtErrs.sort((a, b) => a - b);
  const mean = rtErrs.reduce((s, e) => s + e, 0) / Math.max(rtErrs.length, 1);
  const p95 = rtErrs[Math.min(rtErrs.length - 1, Math.floor(rtErrs.length * 0.95))] ?? 0;
  if (!Number.isFinite(agg.capsuleClearance)) agg.capsuleClearance = null;

  const failures = [];
  if (side && !side.ok)
    failures.push(`sideConsistency: mirrored pairs ${side.wrong.join(', ')}`);
  if (agg.boneStretchPct > gates.boneStretchPct)
    failures.push(`boneStretch ${agg.boneStretchPct.toFixed(2)}% > ${gates.boneStretchPct}%`);
  if (agg.footFlatDeg > gates.footFlatDeg)
    failures.push(`footFlat ${agg.footFlatDeg.toFixed(1)}° > ${gates.footFlatDeg}°`);
  if (agg.footGroundFrac > gates.footGroundFrac)
    failures.push(`footGround ${agg.footGroundFrac.toFixed(3)} > ${gates.footGroundFrac}`);
  if (agg.twistFrac > gates.twistFrac)
    failures.push(`twist ${agg.twistDeg.toFixed(1)}° on ${agg.twistWorstRole} ` +
      `(${(agg.twistFrac * 100).toFixed(0)}% of anatomical limit)`);
  if (agg.capsuleClearance !== null && agg.capsuleClearance < gates.capsuleClearance)
    failures.push(`capsuleClearance ${agg.capsuleClearance.toFixed(2)} < ${gates.capsuleClearance}`);
  if (mean > gates.roundTripMean)
    failures.push(`roundTripMean ${mean.toFixed(3)} > ${gates.roundTripMean}`);
  if (p95 > gates.roundTripP95)
    failures.push(`roundTripP95 ${p95.toFixed(3)} > ${gates.roundTripP95}`);

  resetBindPose(target);                 // leave the skeleton clean for the caller
  const roles = Object.keys(target.rig.map).length;
  return {
    rig: {
      ok: target.rig.ok, missing: target.rig.missing, rolesResolved: roles,
      spineChain: target.rig.spineChain, roles: target.rig.map,
    },
    scale: { srcHipY: +rt.srcHipY.toFixed(3), charHipY: +rt.hipsBindWorldPos.y.toFixed(3), scaleRoot: +rt.scaleRoot.toFixed(3) },
    calibration: calib ? {
      biasTwistDeg: +calib.biasTwistDeg.toFixed(1),
      handClampDeg: { twist: +calib.handClampDeg.twist.toFixed(1) },
    } : null,
    probes: probeResults,
    gates: {
      sideConsistency: side ? side.ok : null,
      boneStretchPct: +agg.boneStretchPct.toFixed(3),
      footFlatDeg: +agg.footFlatDeg.toFixed(1),
      footGroundFrac: +agg.footGroundFrac.toFixed(3),
      twistFrac: +agg.twistFrac.toFixed(2), twistDeg: +agg.twistDeg.toFixed(1),
      twistWorstRole: agg.twistWorstRole,
      capsuleClearance: agg.capsuleClearance === null ? null : +agg.capsuleClearance.toFixed(2),
      roundTripMean: +mean.toFixed(4), roundTripP95: +p95.toFixed(4),
    },
    thresholds: gates,
    pass: failures.length === 0,
    failures,
  };
}
