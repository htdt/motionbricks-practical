import * as THREE from 'three';

// Target-space end-effector IK for AUTHORED constraints (task: retargeting by
// limb direction alone cannot guarantee that a character with different
// proportions hits a mapped world-space target — measured miss on certified
// rigs is ~2-6 cm at authored hand targets, see qa_constraints.mjs).
//
// Design rules (the constraint IK contract):
//  - Canonical SOMA targets are mapped through the SAME root-yaw, root-scale
//    and bind transform the retargeter uses (rt.mapSrcPoint / rt.mapSrcQuat).
//  - Analytic two-bone solve per limb, preserving the retargeted (predicted)
//    elbow/knee pole direction; the authored hand/foot ORIENTATION is applied
//    after the positional solve.
//  - Chains are disjoint (arm/arm/leg/leg) and the root/spine is never
//    touched: satisfying one target can never move a constrained root or
//    another pinned end-effector as a side effect. A combined-reach failure
//    is REPORTED as unreachable, never hidden by moving the torso.
//  - Deterministic: solve weights derive only from constraint frame data
//    (symmetric windows clipped to half the gap between records), never from
//    playback history. Applying frame f produces the same pose whether frames
//    run sequentially, are sought directly, play at any speed, or bake offline.
//  - Geometrically unreachable targets are clamped to the reachable sphere
//    explicitly and marked `reachable: false` — QA fails them unless the move
//    opts into a documented reach policy.
//
// Usage:
//   const ik = new ConstraintIK(rt, clip.constraints);   // once per clip
//   rt.applyFrame(f);  ik.apply(f);                      // every frame

const CHAINS = {
  LeftHand: ['LeftArm', 'LeftForeArm', 'LeftHand'],
  RightHand: ['RightArm', 'RightForeArm', 'RightHand'],
  LeftFoot: ['LeftUpLeg', 'LeftLeg', 'LeftFoot'],
  RightFoot: ['RightUpLeg', 'RightLeg', 'RightFoot'],
};
const DEFAULT_WINDOW = 6;                       // frames of blend-in/out around a key

const smooth = (t) => t * t * (3 - 2 * t);      // C1 ease, exact 0/1 at the ends

// map baked end-effector records into character space once per clip
export function mapRecordTargets(rt, records) {
  const out = [];
  const srcHipsIdx = rt.idx[rt.S.Hips];
  const srcRestHips = new THREE.Vector3(...rt.data.rest[srcHipsIdx]);
  for (const rec of records ?? []) {
    if (rec.family !== 'end-effector' || !CHAINS[rec.role]) continue;
    if (!Number.isInteger(rec.frame) || rec.frame < 0 || rec.frame >= rt.numFrames)
      throw new Error(`constraint record frame ${rec.frame} outside clip (${rt.numFrames} frames)`);
    const boneName = rt.R[rec.role];
    const srcJoint = rt.S[rec.role];
    if (!boneName || !rt.bones[boneName]) throw new Error(`rig lacks a bone for constrained role ${rec.role}`);
    const pos = rt.mapSrcPoint(new THREE.Vector3(rec.pos[0], rec.pos[1], rec.pos[2]));
    if (rt.inPlace) {
      // an in-place retargeter zeroes the root's horizontal displacement;
      // the mapped target must shed the same displacement (at the key frame)
      // or the character would reach toward where the clip has TRAVELLED
      const hips = rt.data.pos[rec.frame][srcHipsIdx];
      const disp = new THREE.Vector3(hips[0], hips[1], hips[2]).sub(srcRestHips)
        .applyQuaternion(rt.rootYaw).multiplyScalar(rt.scaleRoot);
      pos.x -= disp.x; pos.z -= disp.z;
    }
    const quat = rec.quat ? rt.mapSrcQuat(
      new THREE.Quaternion(rec.quat[0], rec.quat[1], rec.quat[2], rec.quat[3]).normalize(),
      srcJoint, boneName) : null;
    out.push({
      role: rec.role, frame: rec.frame, pos, quat,
      posConstrained: rec.posConstrained !== false,
      rotConstrained: rec.rotConstrained !== false && quat !== null,
      source: rec.source, type: rec.type,
    });
  }
  return out;
}

export class ConstraintIK {
  constructor(rt, records, { window = DEFAULT_WINDOW, reachPolicy = 'fail' } = {}) {
    if (!rt || typeof rt.applyFrame !== 'function') throw new Error('ConstraintIK requires a Retargeter');
    if (!Number.isInteger(window) || window < 0) throw new Error(`window must be a non-negative integer; got ${window}`);
    if (!['fail', 'clamp'].includes(reachPolicy)) throw new Error(`unknown reachPolicy "${reachPolicy}"`);
    this.rt = rt;
    this.reachPolicy = reachPolicy;
    this.byRole = new Map();                    // role -> sorted [{frame, pos, quat, wL, wR}]
    for (const t of mapRecordTargets(rt, records)) {
      if (!this.byRole.has(t.role)) this.byRole.set(t.role, []);
      this.byRole.get(t.role).push(t);
    }
    for (const list of this.byRole.values()) {
      list.sort((a, b) => a.frame - b.frame);
      // deterministic blend windows: clipped to half the gap toward the
      // neighboring record so windows can never overlap or interact
      for (let i = 0; i < list.length; i++) {
        const prevGap = i > 0 ? list[i].frame - list[i - 1].frame : Infinity;
        const nextGap = i < list.length - 1 ? list[i + 1].frame - list[i].frame : Infinity;
        list[i].wL = Math.min(window, Math.floor(prevGap / 2));
        list[i].wR = Math.min(window, Math.floor(nextGap / 2));
      }
    }
    // chain bones + bind lengths per constrained role
    this.chains = {};
    for (const role of this.byRole.keys()) {
      const [rA, rB, rC] = CHAINS[role];
      const nA = rt.R[rA], nB = rt.R[rB], nC = rt.R[rC];
      if (!nA || !nB || !nC || !rt.bones[nA] || !rt.bones[nB] || !rt.bones[nC])
        throw new Error(`rig cannot solve constrained role ${role}: missing ${rA}/${rB}/${rC}`);
      const L1 = rt.bindLen[nA], L2 = rt.bindLen[nB];
      if (!Number.isFinite(L1) || !Number.isFinite(L2) || L1 <= 0 || L2 <= 0)
        throw new Error(`rig has degenerate bind lengths for ${role}`);
      this.chains[role] = { a: rt.bones[nA], b: rt.bones[nB], c: rt.bones[nC], L1, L2 };
    }
    this.lastSolves = [];                       // diagnostics for the frame just applied
  }

  // active record + weight for a role at frame f (nearest record; ties -> earlier)
  _active(role, f) {
    let best = null, bestD = Infinity;
    for (const t of this.byRole.get(role)) {
      const d = Math.abs(f - t.frame);
      if (d < bestD) { best = t; bestD = d; }
    }
    if (!best) return null;
    const span = f < best.frame ? best.wL : best.wR;
    if (bestD > span) return null;
    const w = bestD === 0 ? 1 : smooth(1 - bestD / (span + 1));
    return { rec: best, w };
  }

  // world quaternion of a bone as of the already-applied frame
  _worldQ(b) {
    return (this.rt.animWorld[b.uuid] ?? b.getWorldQuaternion(new THREE.Quaternion())).clone();
  }

  _parentQ(b) {
    return b.parent
      ? (this.rt.animWorld[b.parent.uuid] ?? b.parent.getWorldQuaternion(new THREE.Quaternion())).clone()
      : new THREE.Quaternion();
  }

  // Apply the constraint solve for frame f. Call directly after
  // rt.applyFrame(f); mutates the same bones, returns per-role diagnostics.
  apply(f) {
    const rt = this.rt;
    f = Number.isFinite(f) ? (((Math.round(f) % rt.numFrames) + rt.numFrames) % rt.numFrames) : 0;
    this.lastSolves = [];
    for (const [role, ] of this.byRole) {
      const act = this._active(role, f);
      if (!act) continue;
      const { rec, w } = act;
      const { a, b, c, L1, L2 } = this.chains[role];
      const pA = rt._framePos(a).clone(), pB = rt._framePos(b).clone(), pC = rt._framePos(c).clone();
      const qA = this._worldQ(a), qB = this._worldQ(b), qC = this._worldQ(c);

      const solve = { role, frame: f, keyFrame: rec.frame, weight: +w.toFixed(4), reachable: true };
      let qA2 = qA, qB2 = qB;
      if (rec.posConstrained) {
        const d = rec.pos.clone().sub(pA);
        const dist = d.length();
        const maxReach = L1 + L2, minReach = Math.abs(L1 - L2) + 1e-9;
        solve.targetDist = +dist.toFixed(4);
        solve.reachable = dist <= maxReach + 1e-9 && dist >= minReach;
        // clamp to the reachable sphere EXPLICITLY (never silently): the
        // solve continues toward the clamped point, QA sees reachable=false
        const reach = THREE.MathUtils.clamp(dist, minReach, maxReach - 1e-9);
        if (dist < 1e-9) { this.lastSolves.push(solve); continue; }
        const dHat = d.clone().normalize();
        // pole = the retargeted elbow/knee direction, projected off the
        // shoulder->target axis (preserves the predicted pole)
        const ab = pB.clone().sub(pA);
        const pole = ab.clone().sub(dHat.clone().multiplyScalar(ab.dot(dHat)));
        if (pole.lengthSq() < 1e-10) {
          // degenerate: limb exactly along target axis — derive a stable
          // fallback from the current end-bone offset, else any axis ⊥ target
          const cb = pC.clone().sub(pA).sub(dHat.clone().multiplyScalar(pC.clone().sub(pA).dot(dHat)));
          if (cb.lengthSq() > 1e-10) pole.copy(cb);
          else {
            const up = Math.abs(dHat.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
            pole.copy(up.sub(dHat.clone().multiplyScalar(up.dot(dHat))));
          }
        }
        pole.normalize();
        const cosA = THREE.MathUtils.clamp(
          (L1 * L1 + reach * reach - L2 * L2) / (2 * L1 * reach), -1, 1);
        const sinA = Math.sqrt(Math.max(0, 1 - cosA * cosA));
        const pB2 = pA.clone()
          .add(dHat.clone().multiplyScalar(cosA * L1))
          .add(pole.clone().multiplyScalar(sinA * L1));
        const pT = pA.clone().add(dHat.clone().multiplyScalar(reach));
        // upper segment: minimal swing current->solved direction (keeps twist)
        const swingA = new THREE.Quaternion().setFromUnitVectors(
          ab.normalize(), pB2.clone().sub(pA).normalize());
        qA2 = swingA.clone().multiply(qA);
        // lower segment: minimal swing of its post-swingA direction onto the target
        const bcAfter = pC.clone().sub(pB).applyQuaternion(swingA).normalize();
        const swingB = new THREE.Quaternion().setFromUnitVectors(
          bcAfter, pT.clone().sub(pB2).normalize());
        qB2 = swingB.multiply(swingA).multiply(qB);
        solve.posErrBefore = +pC.distanceTo(rec.pos).toFixed(4);
      }
      // blend by the deterministic weight (w=1 exactly at the key frame)
      const qAf = qA.clone().slerp(qA2, w);
      const qBf = qB.clone().slerp(qB2, w);
      // end bone: exact authored orientation at the key, else keep the
      // retargeted WORLD orientation (parent changes must not drag the hand)
      const qCf = rec.rotConstrained ? qC.clone().slerp(rec.quat, w) : qC;

      const pwA = this._parentQ(a);
      a.quaternion.copy(pwA.clone().invert().multiply(qAf));
      rt.animWorld[a.uuid] = qAf.clone();
      b.quaternion.copy(qAf.clone().invert().multiply(qBf));
      rt.animWorld[b.uuid] = qBf.clone();
      c.quaternion.copy(qBf.clone().invert().multiply(qCf));
      rt.animWorld[c.uuid] = qCf.clone();
      // downstream world-position cache is stale for this subtree
      rt.animWorldPos = {};
      if (rec.posConstrained && w === 1) {
        const pC2 = rt._framePos(c);
        solve.posErrAfter = +pC2.distanceTo(rec.pos).toFixed(5);
      }
      this.lastSolves.push(solve);
    }
    return this.lastSolves;
  }
}
