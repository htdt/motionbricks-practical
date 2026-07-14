// Shared measurement machinery for constraint QA (qa_constraints.mjs) and
// the guard/modifier ablation runner (ablate.mjs).
//
// One `captureRun` = one (character, clip, transfer-config) evaluation:
// construct a fresh Retargeter on a bind-reset skeleton, apply every frame,
// and record everything the metrics need — end-effector world poses, raw
// (pre-guard) demands, local quaternions for flip detection, foot heights
// for skate/penetration, hips trajectory, and IK solve diagnostics. Metrics
// are pure functions over captures, so any two configurations can be
// compared without re-running the transfer.
import * as THREE from 'three';
import { Retargeter, baselineOptions } from './retarget.js';
import { ConstraintIK, mapRecordTargets } from './ik.js';
import { resetBindPose } from './align.js';

export const EE_ROLES = ['LeftHand', 'RightHand', 'LeftFoot', 'RightFoot'];

export const deg = (r) => (r * 180) / Math.PI;
// angle between two rotations. The dot is normalized by the actual quat
// magnitudes: products of many near-unit quats drift to |q| = 1 ± 1e-7, and
// an unnormalized acos then reports a false ~0.05° angle between IDENTICAL
// rotations — enough to fail an exact-determinism gate.
export const quatAngleDeg = (a, b) => {
  const dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  const n = Math.sqrt(a.lengthSq() * b.lengthSq());
  return deg(2 * Math.acos(Math.min(1, Math.abs(dot) / (n || 1))));
};

export function stats(values) {
  const v = values.filter(Number.isFinite);
  if (!v.length) return null;
  const s = [...v].sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return {
    n: v.length,
    mean: +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(5),
    median: +q(0.5).toFixed(5),
    p95: +q(0.95).toFixed(5),
    max: +s[s.length - 1].toFixed(5),
  };
}

// swing/twist decomposition angle split of the rotation taking quat a to
// quat b, about `axis` expressed in a's local frame
export function swingTwistSplitDeg(a, b, axisLocal) {
  const rel = a.clone().invert().multiply(b);
  if (rel.w < 0) rel.set(-rel.x, -rel.y, -rel.z, -rel.w);
  const r = new THREE.Vector3(rel.x, rel.y, rel.z);
  const proj = axisLocal.clone().normalize().multiplyScalar(r.dot(axisLocal.clone().normalize()));
  const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, rel.w);
  if (twist.lengthSq() < 1e-12) twist.identity(); else twist.normalize();
  const swing = rel.clone().multiply(twist.clone().invert());
  const ang = (q) => deg(2 * Math.acos(Math.min(1, Math.abs(q.w))));
  return { swing: ang(swing), twist: ang(twist) };
}

// ---------------------------------------------------------------- capture --

// options: { config: Retargeter options overrides, ik: bool, ikOptions,
//            order: 'sequential' | 'seek' | int[] (explicit frame order) }
export function captureRun(target, clip, { config = {}, ik = false, ikOptions = {}, order = 'sequential' } = {}) {
  resetBindPose(target);
  const rt = new Retargeter({ ...target, data: clip, ...config });
  const solver = ik && clip.constraints?.length ? new ConstraintIK(rt, clip.constraints, ikOptions) : null;
  const N = rt.numFrames;
  const bones = {};
  for (const role of EE_ROLES) {
    const name = rt.R[role];
    if (!name || !rt.bones[name]) throw new Error(`character lacks a bone for ${role}`);
    bones[role] = rt.bones[name];
  }
  const foreOf = { LeftHand: 'LeftForeArm', RightHand: 'RightForeArm' };
  const toeOf = { LeftFoot: 'LeftToeBase', RightFoot: 'RightToeBase' };
  const chainRootOf = { LeftHand: 'LeftArm', RightHand: 'RightArm',
    LeftFoot: 'LeftUpLeg', RightFoot: 'RightUpLeg' };
  const groundBind = rt.groundBones.map(([b, y]) => [b, y]);

  let frames;
  if (Array.isArray(order)) frames = order;
  else if (order === 'seek') {
    // deterministic scrambled order touching every frame exactly once
    frames = [];
    for (let k = 0; k < N; k++) frames.push((k * 7919) % N);
    frames = [...new Set(frames)];
    for (let f = 0; f < N; f++) if (!frames.includes(f)) frames.push(f);
  } else frames = Array.from({ length: N }, (_, f) => f);

  const rows = new Array(N);
  const solves = [];
  for (const f of frames) {
    rt.applyFrame(f);
    if (solver) solves.push(...solver.apply(f));
    const row = { f, roles: {}, hips: rt.frameState.hipsWorld.clone() };
    for (const role of EE_ROLES) {
      const b = bones[role];
      const entry = {
        pos: rt._framePos(b).clone(),
        quat: (rt.animWorld[b.uuid] ?? b.getWorldQuaternion(new THREE.Quaternion())).clone(),
        local: b.quaternion.clone(),
        raw: rt.rawTargets[b.name]?.clone() ?? null,
      };
      const foreName = foreOf[role] && rt.R[foreOf[role]];
      if (foreName && rt.bones[foreName]) {
        entry.forePos = rt._framePos(rt.bones[foreName]).clone();
        entry.foreQuat = (rt.animWorld[rt.bones[foreName].uuid]
          ?? rt.bones[foreName].getWorldQuaternion(new THREE.Quaternion())).clone();
        entry.foreLocal = rt.bones[foreName].quaternion.clone();
      }
      const toeName = toeOf[role] && rt.R[toeOf[role]];
      if (toeName && rt.bones[toeName]) entry.toePos = rt._framePos(rt.bones[toeName]).clone();
      const rootName = rt.R[chainRootOf[role]];
      if (rootName && rt.bones[rootName]) entry.chainRootPos = rt._framePos(rt.bones[rootName]).clone();
      row.roles[role] = entry;
    }
    // torso-capsule clearance of hands/forearms (fraction of capsuleR) and
    // deepest ground penetration below bind height (m) this frame
    {
      const capA = rt._framePos(rt.hips), capB = rt._framePos(rt.chestBone);
      const ab = capB.clone().sub(capA);
      let minClear = Infinity;
      for (const role of ['LeftHand', 'RightHand']) {
        for (const p of [row.roles[role].pos, row.roles[role].forePos]) {
          if (!p) continue;
          const t = THREE.MathUtils.clamp(p.clone().sub(capA).dot(ab) / ab.lengthSq(), 0, 1);
          if (t <= 0 || t >= 1) continue;
          const radial = p.clone().sub(capA.clone().add(ab.clone().multiplyScalar(t)));
          minClear = Math.min(minClear, radial.length() / rt.capsuleR);
        }
      }
      row.capsuleClearance = Number.isFinite(minClear) ? minClear : null;
      let pen = 0;
      for (const [b, bindY] of groundBind) pen = Math.max(pen, bindY - rt._framePos(b).y);
      row.groundPen = pen;
    }
    rows[f] = row;
  }
  const capture = { rt, clip, rows, solves, N, fps: rt.fps, config: rt.configDump(), ik: !!solver };
  resetBindPose(target);
  return capture;
}

// -------------------------------------------------------------- metrics ----

// source foot-contact mask per frame/side, from the clip's stored PREDICTED
// contact channels when present (explicit joint mapping), else the
// height+speed heuristic as fallback — callers can require the channels.
export function contactMask(clip, { requireChannels = false } = {}) {
  const N = clip.numFrames ?? clip.pos.length;
  const mask = { Left: new Array(N).fill(false), Right: new Array(N).fill(false) };
  if (Array.isArray(clip.contacts) && Array.isArray(clip.contactJoints)) {
    for (let f = 0; f < N; f++) {
      for (const side of ['Left', 'Right']) {
        mask[side][f] = clip.contactJoints.some((j, k) =>
          j.startsWith(side) && clip.contacts[f][k] > 0.5);
      }
    }
    mask.source = 'channels';
    return mask;
  }
  if (requireChannels) throw new Error('clip carries no foot-contact channels');
  const idx = Object.create(null);
  clip.names.forEach((n, i) => { idx[n] = i; });
  for (const side of ['Left', 'Right']) {
    const ti = idx[`${side}ToeBase`];
    if (ti === undefined) throw new Error(`clip lacks ${side}ToeBase for contact heuristic`);
    const minY = Math.min(...clip.pos.map((fr) => fr[ti][1]));
    for (let f = 0; f < N; f++) {
      const p = clip.pos[f][ti];
      const speed = f ? Math.hypot(p[0] - clip.pos[f - 1][ti][0], p[2] - clip.pos[f - 1][ti][2]) * (clip.fps ?? 30) : 0;
      mask[side][f] = p[1] < minY + 0.03 && speed < 0.3;
    }
  }
  mask.source = 'heuristic';
  return mask;
}

// horizontal EE speeds (m/s) per role over a capture, plus per-frame values
export function footSkate(capture, mask) {
  const out = {};
  for (const side of ['Left', 'Right']) {
    const role = `${side}Foot`;
    const speeds = [];
    for (let f = 1; f < capture.N; f++) {
      if (!mask[side][f] || !mask[side][f - 1]) continue;
      const a = capture.rows[f - 1].roles[role], b = capture.rows[f].roles[role];
      const pa = a.toePos ?? a.pos, pb = b.toePos ?? b.pos;
      speeds.push(Math.hypot(pb.x - pa.x, pb.z - pa.z) * capture.fps);
    }
    out[side] = stats(speeds);
  }
  return out;
}

// per-role angular velocity / acceleration (deg per frame) + branch flips
export function angularKinematics(capture) {
  const out = {};
  for (const role of EE_ROLES) {
    const vel = [], acc = [];
    let flips = 0;
    for (let f = 1; f < capture.N; f++) {
      const v = quatAngleDeg(capture.rows[f - 1].roles[role].local, capture.rows[f].roles[role].local);
      vel.push(v);
      if (v > 90) flips++;                     // a >90°/frame local step is a branch flip
      if (f >= 2) acc.push(Math.abs(v - vel[vel.length - 2]));
    }
    out[role] = { vel: stats(vel), acc: stats(acc), flips };
  }
  return out;
}

// pose validity: NaN / non-unit local quats anywhere in the capture
export function poseValidity(capture) {
  let nan = 0, denorm = 0;
  for (const row of capture.rows) {
    for (const role of EE_ROLES) {
      const q = row.roles[role].local;
      const n2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
      if (!Number.isFinite(n2)) nan++;
      else if (Math.abs(n2 - 1) > 1e-3) denorm++;
    }
  }
  return { nan, denorm, ok: nan === 0 && denorm === 0 };
}

// max per-bone world-orientation difference between two captures (deg) and
// max EE position difference (m) — determinism / guard-delta measurements
export function captureDelta(a, b) {
  if (a.N !== b.N) throw new Error('captures cover different frame counts');
  let rot = 0, pos = 0;
  const perRole = {};
  for (const role of EE_ROLES) {
    const rotV = [], posV = [];
    for (let f = 0; f < a.N; f++) {
      const ra = a.rows[f].roles[role], rb = b.rows[f].roles[role];
      rotV.push(quatAngleDeg(ra.quat, rb.quat));
      posV.push(ra.pos.distanceTo(rb.pos));
    }
    perRole[role] = { rot: stats(rotV), pos: stats(posV) };
    rot = Math.max(rot, perRole[role].rot.max);
    pos = Math.max(pos, perRole[role].pos.max);
  }
  return { perRole, maxRotDeg: +rot.toFixed(4), maxPosM: +pos.toFixed(5) };
}

// stage-2 metric: transfer fidelity of the capture against the SOURCE clip.
// Absolute world orientations legitimately differ between rigs (proportions
// change the aimed forearm/shin), so fidelity is measured on quantities the
// transfer is DESIGNED to preserve exactly, both zero-error at the raw
// baseline by the conjugation identity of the world-axis mapping:
//
//  - hands: the wrist-vs-forearm ARTICULATION deviation from the rest/bind
//    relation. The source's local deviation and the character's are conjugate
//    quaternions at the baseline, so their rotation angles (and swing/twist
//    magnitudes about the respective bone axes) must match. Uses the FULL
//    source demand — never scaled by handFollow: stylization shows up here
//    as error, and is reported as its own stage-3 delta instead.
//  - feet: the world deviation from rest/bind (pelvis-conjugated at baseline,
//    same angle-preservation argument).
//
// `demandTracking` additionally reports how far the emitted pose deviates
// from the transfer's own raw demand (rawTargets): exactly 0 at the true
// baseline; any clamp/damping/smoothing shows up here.
export function sourceFidelity(capture) {
  const { rt, clip } = capture;
  const idx = Object.create(null);
  clip.names.forEach((n, i) => { idx[n] = i; });
  const q4 = (a) => new THREE.Quaternion(a[0], a[1], a[2], a[3]).normalize();
  const angleOf = (q) => deg(2 * Math.acos(Math.min(1, Math.abs(q.w))));
  const out = {};
  for (const role of EE_ROLES) {
    const boneName = rt.R[role];
    const srcJoint = rt.S[role];
    const sj = idx[srcJoint];
    if (sj === undefined || !clip.quat) { out[role] = null; continue; }
    const isHand = role.endsWith('Hand');
    const rotErr = [], swingErr = [], twistErr = [], demandErr = [];
    // char-side rest relations
    const axisChar = rt.twistAxis?.[boneName] ?? new THREE.Vector3(0, 1, 0);
    let bindRel = null, srcRestRel = null, foreName = null, sf = null, axisSrc = null;
    if (isHand) {
      foreName = rt.R[role === 'LeftHand' ? 'LeftForeArm' : 'RightForeArm'];
      sf = idx[rt.S[role === 'LeftHand' ? 'LeftForeArm' : 'RightForeArm']];
      bindRel = rt.bindWorldQ[foreName].clone().invert().multiply(rt.bindWorldQ[boneName]);
      srcRestRel = q4(clip.restQuat[sf]).invert().multiply(q4(clip.restQuat[sj]));
      // source wrist twist axis: the forearm->wrist segment direction at
      // rest, in the source wrist's rest-local frame
      const rw = new THREE.Vector3(...clip.rest[sj]), rf = new THREE.Vector3(...clip.rest[sf]);
      axisSrc = rw.clone().sub(rf).normalize()
        .applyQuaternion(q4(clip.restQuat[sj]).invert()).normalize();
    }
    for (let f = 0; f < capture.N; f++) {
      const row = capture.rows[f].roles[role];
      let relC, relS;
      if (isHand) {
        if (!row.foreQuat) { relC = null; relS = null; }
        else {
          relC = bindRel.clone().invert()
            .multiply(row.foreQuat.clone().invert().multiply(row.quat));
          relS = srcRestRel.clone().invert()
            .multiply(q4(clip.quat[f][sf]).invert().multiply(q4(clip.quat[f][sj])));
        }
      } else {
        relC = rt.bindWorldQ[boneName].clone().invert().multiply(row.quat);
        relS = q4(clip.restQuat[sj]).invert().multiply(q4(clip.quat[f][sj]));
      }
      if (relC && relS) {
        rotErr.push(Math.abs(angleOf(relC) - angleOf(relS)));
        const stC = swingTwistSplitDeg(new THREE.Quaternion(), relC, axisChar);
        const stS = swingTwistSplitDeg(new THREE.Quaternion(), relS, isHand ? axisSrc : axisChar);
        swingErr.push(Math.abs(stC.swing - stS.swing));
        twistErr.push(Math.abs(stC.twist - stS.twist));
      }
      if (row.raw) demandErr.push(quatAngleDeg(row.quat, row.raw));
    }
    out[role] = { rot: stats(rotErr), swing: stats(swingErr), twist: stats(twistErr),
      demandTracking: stats(demandErr) };
  }
  // (limb position/direction fidelity — elbow/knee placement — is measured
  // by align.js round-trip recovery; the ablation runner reports it alongside)
  return out;
}

// limb extension ratio (|chain root -> end| / total bind length): 1.0 = a
// fully straight limb; sustained ~1.0 with the source limb bent = hyperextension
export function limbExtension(capture) {
  const { rt } = capture;
  const chains = { LeftHand: ['LeftArm', 'LeftForeArm'], RightHand: ['RightArm', 'RightForeArm'],
    LeftFoot: ['LeftUpLeg', 'LeftLeg'], RightFoot: ['RightUpLeg', 'RightLeg'] };
  const out = {};
  for (const role of EE_ROLES) {
    const [rA, rB] = chains[role];
    const nA = rt.R[rA], nB = rt.R[rB];
    if (!nA || !nB || !rt.bindLen[nA] || !rt.bindLen[nB]) continue;
    const total = rt.bindLen[nA] + rt.bindLen[nB];
    const ratios = [];
    for (const row of capture.rows) {
      const e = row.roles[role];
      if (e.chainRootPos) ratios.push(e.pos.distanceTo(e.chainRootPos) / total);
    }
    out[role] = stats(ratios);
  }
  return out;
}

// torso-capsule clearance / ground penetration aggregates over a capture
export function bodyClearance(capture) {
  const clear = capture.rows.map(r => r.capsuleClearance).filter(v => v !== null);
  const pen = capture.rows.map(r => r.groundPen);
  return {
    capsuleClearanceMin: clear.length ? +Math.min(...clear).toFixed(3) : null,
    groundPenMax: +Math.max(...pen).toFixed(4),
  };
}

// stage-4 metric: character EE pose vs the MAPPED authored targets
export function constraintAccuracy(capture) {
  const { rt, clip } = capture;
  const targets = mapRecordTargets(rt, clip.constraints ?? []);
  const rowsByKey = new Map(capture.rows.map((r) => [r.f, r]));
  const perRole = {}, perType = {}, perRoleType = {};
  const add = (buckets, key, posErr, rotErr, reachable) => {
    buckets[key] ??= { pos: [], rot: [], unreachable: 0 };
    if (reachable === false) { buckets[key].unreachable++; return; }
    if (posErr !== null) buckets[key].pos.push(posErr);
    if (rotErr !== null) buckets[key].rot.push(rotErr);
  };
  const summarize = (buckets) => Object.fromEntries(Object.entries(buckets).map(
    ([key, v]) => [key, { pos: stats(v.pos), rot: stats(v.rot), unreachable: v.unreachable }]));
  const details = [];
  for (const t of targets) {
    const row = rowsByKey.get(t.frame);
    if (!row) throw new Error(`no captured frame ${t.frame} for constraint on ${t.role}`);
    const got = row.roles[t.role];
    const posErr = t.posConstrained ? got.pos.distanceTo(t.pos) : null;
    const rotErr = t.rotConstrained ? quatAngleDeg(got.quat, t.quat) : null;
    const solve = capture.solves.find(
      (s) => s.role === t.role && s.frame === t.frame && s.weight === 1);
    const reachable = solve ? solve.reachable : null;
    details.push({
      role: t.role, type: t.type, frame: t.frame, source: t.source,
      posErr: posErr === null ? null : +posErr.toFixed(5),
      rotErrDeg: rotErr === null ? null : +rotErr.toFixed(3),
      reachable,
    });
    add(perRole, t.role, posErr, rotErr, reachable);
    add(perType, t.type, posErr, rotErr, reachable);
    add(perRoleType, `${t.role}:${t.type}`, posErr, rotErr, reachable);
  }
  return {
    details,
    summary: summarize(perRole),
    summaryByType: summarize(perType),
    summaryByRoleType: summarize(perRoleType),
    count: targets.length,
  };
}
