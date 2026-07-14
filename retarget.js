import * as THREE from 'three';
import { resolveRig } from './rigmap.js';

// Position-based, pelvis-relative retargeter — RIG-AGNOSTIC version.
//
// Reads the source-skeleton joint WORLD POSITIONS, builds a pelvis frame and a chest
// frame from positions, and re-applies the motion on the character: limbs are aimed at
// their child joint; the spine is bent/twisted from the pelvis frame (bottom) to the
// chest frame (top). Driving from positions means knees/elbows point where the source's
// do and cannot invert, and the chest frame (built from the shoulder line) carries the
// torso twist.
//
// The character side is addressed through CANONICAL ROLES resolved by rigmap.js
// (Mixamo with/without prefix, Tripo3D rig specs, topology fallback), so the same
// retargeter drives any humanoid GLB. Spine blend weights adapt to the actual
// spine-chain length.
//
// Twist/roll handling:
//  - limb aims are REBASED on the transferred pelvis (legs) / chest (arms) frame, so the
//    roll around each bone follows the body and setFromUnitVectors only contributes the
//    residual swing — hands riding the forearm keep a natural palm facing;
//  - forearms with quaternion sources take their roll from the source forearm's own
//    world delta (true pronation/supination) — the ONLY quaternion path since the
//    ablation showed the chest-rebase projection off by up to 179° (evidence/README.md);
//  - feet take the source ankle's TRUE orientation (data.quat), transferred
//    pelvis-relative, giving real heel-strike/toe-off instead of inheriting the shin;
//  - HANDS ride the character's forearm and add the source's own wrist-vs-forearm
//    delta mapped through world axes at rest/bind;
//  - neck/head are damped toward the overall body heading instead of riding chest twist 1:1.
//
// The transfer is DETERMINISTIC and stateless per frame: applying frame f yields
// the same pose under sequential playback, direct seek, any speed, or offline
// baking. The historical guards (handClamp, torsoCapsule, continuity slew,
// ground lift) were ablated on the regenerated move set across two certified
// rigs and deleted: none had a reproducible benefit after the rest-anchor fix,
// the clamp clipped valid authored wrists by up to 15.5°, the capsule displaced
// valid near-face guard poses with zero measured torso penetration in the
// baseline, and the temporal guards made results depend on playback history
// (evidence/README.md). Torso clearance, ground penetration, branch flips, and
// foot skate remain MEASURED QA metrics (qametrics.mjs) instead of silent
// runtime corrections.

export function buildBoneOrder(root) {
  const out = [];
  (function walk(b) { if (b.isBone) out.push(b); b.children.forEach(walk); })(root);
  return out;
}

export async function loadGLBSkeleton(GLTFLoader, url, scene) {
  const gltf = await new GLTFLoader().loadAsync(url);
  const model = gltf.scene;
  model.traverse(o => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  if (scene) scene.add(model);
  model.updateWorldMatrix(true, true);
  let skinned = null;
  model.traverse(o => { if (o.isSkinnedMesh && !skinned) skinned = o; });
  if (!skinned?.skeleton) throw new Error(`${url}: no skinned mesh with a skeleton`);
  const bones = Object.create(null);
  skinned.skeleton.bones.forEach(b => {
    if (!b.name) throw new Error(`${url}: every skeleton bone must have a name`);
    if (hasOwn(bones, b.name)) throw new Error(`${url}: duplicate skeleton bone name "${b.name}"`);
    bones[b.name] = b;
  });
  const rig = resolveRig(skinned.skeleton.bones);
  if (!rig.ok) throw new Error(`${url}: incomplete humanoid rig; missing=${rig.missing.join(',')}`);
  const hips = bones[rig.map.Hips];
  return { model, skinned, bones, hips, rig, animations: gltf.animations };
}

// SOURCE-SKELETON MAP: canonical source roles -> joint names in the motion data.
// Baked clips carry their own map (data.srcMap); align.js builds the same map for
// any GLB rig via rigmap, which is what makes the retargeter a general
// two-skeleton aligner.
// *Anchor roles are frame-building references (may coincide with limb roles on
// rigs that don't have separate yaw/pitch links):
//   Hips/Chest        pelvis + chest frame origins
//   L/RHipAnchor      the hip line (pelvis frame right axis)
//   L/RShoulderAnchor the shoulder line (chest frame right axis + origin midpoint)
//
// The default is SOMA (NVIDIA Kimodo somaskel77 output — a human skeleton, so the
// anatomical joint centers ARE the named joints; anchors coincide with limbs
// and fall back inside the constructor).
export const SOMA_SRC = {
  Hips: 'Hips', Chest: 'Chest',
  LeftHipAnchor: 'LeftLeg', RightHipAnchor: 'RightLeg',
  LeftShoulderAnchor: 'LeftArm', RightShoulderAnchor: 'RightArm',
  LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftShin', LeftFoot: 'LeftFoot',
  RightUpLeg: 'RightLeg', RightLeg: 'RightShin', RightFoot: 'RightFoot',
  LeftArm: 'LeftArm', LeftForeArm: 'LeftForeArm', LeftHand: 'LeftHand',
  RightArm: 'RightArm', RightForeArm: 'RightForeArm', RightHand: 'RightHand',
};

const SOURCE_REQUIRED = [
  'Hips', 'Chest',
  'LeftHipAnchor', 'RightHipAnchor', 'LeftShoulderAnchor', 'RightShoulderAnchor',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot',
  'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
];
const SOURCE_SEGMENTS = [
  ['LeftUpLeg', 'LeftLeg'], ['LeftLeg', 'LeftFoot'],
  ['RightUpLeg', 'RightLeg'], ['RightLeg', 'RightFoot'],
  ['LeftArm', 'LeftForeArm'], ['LeftForeArm', 'LeftHand'],
  ['RightArm', 'RightForeArm'], ['RightForeArm', 'RightHand'],
];
const TARGET_REQUIRED = [
  'Hips', 'Chest',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'RightUpLeg', 'RightLeg', 'RightFoot',
  'LeftArm', 'LeftForeArm', 'LeftHand', 'RightArm', 'RightForeArm', 'RightHand',
];
const hasOwn = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);

// limb roles: aim from a source joint toward its child joint (both given as
// canonical source roles, resolved through the source map).
// base: which transferred body-frame delta carries the bone's twist/roll while the
// aim only adds the residual swing ('pelvis' for legs, 'chest' for arms).
const AIM = {
  LeftUpLeg:    ['LeftUpLeg',   'LeftLeg',      'pelvis'],
  LeftLeg:      ['LeftLeg',     'LeftFoot',     'pelvis'],
  RightUpLeg:   ['RightUpLeg',  'RightLeg',     'pelvis'],
  RightLeg:     ['RightLeg',    'RightFoot',    'pelvis'],
  LeftArm:      ['LeftArm',     'LeftForeArm',  'chest'],
  LeftForeArm:  ['LeftForeArm', 'LeftHand',     'chest'],
  RightArm:     ['RightArm',    'RightForeArm', 'chest'],
  RightForeArm: ['RightForeArm','RightHand',    'chest'],
};
const AIM_CHILD = {
  LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftFoot', RightUpLeg: 'RightLeg', RightLeg: 'RightFoot',
  LeftArm: 'LeftForeArm', LeftForeArm: 'LeftHand', RightArm: 'RightForeArm', RightForeArm: 'RightHand',
};
// full-orientation transfer: target role -> source role. FEET only:
// both rigs' rest feet are flat on the ground pointing forward, so the
// pelvis-relative absolute transfer is anchored correctly.
const FULLQ = {
  LeftFoot: 'LeftFoot',
  RightFoot: 'RightFoot',
};
// HANDS are transferred FOREARM-RELATIVE instead: the old chest-relative
// absolute transfer re-anchored the hand to the character's T-pose bind, but
// near source rest the arms HANG — the anchor mismatch cocked the wrists ~70°
// against the forearm (pinned on the clamp: "odd fists" at rest-ish poses) and
// let the deviation wander past 180°, where any clamp flips branch (the
// one-frame ~176° fist flicker). Riding the forearm and applying the source's
// own wrist-vs-forearm delta keeps the deviation small and continuous.
const HAND_LOCAL = {
  LeftHand:  'LeftForeArm',
  RightHand: 'RightForeArm',
};
// head/neck: damped toward the overall body heading instead of riding chest twist 1:1
const HEAD_DAMP = { Neck: 0.45, Head: 0.2 };
// per-role anatomical twist axes are still derived at bind (twistAxis) for
// QA's swing/twist metrics; there is no runtime clamp anymore

const _v = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function frameQ(origin, upPt, rA, rB) {
  const up = upPt.clone().sub(origin);
  const r0 = rA.clone().sub(rB);
  if (up.lengthSq() < 1e-12 || r0.lengthSq() < 1e-12)
    throw new Error('cannot build body frame from coincident joints');
  up.normalize(); r0.normalize();
  const fwd = new THREE.Vector3().crossVectors(r0, up);
  if (fwd.lengthSq() < 1e-12) throw new Error('cannot build body frame from collinear joints');
  fwd.normalize();
  const right = new THREE.Vector3().crossVectors(up, fwd).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd));
}

// The reference "raw transfer" configuration: corrected rest anchors,
// world-axis wrist mapping, full source wrist transfer. Since the guard
// cleanup (evidence/README.md) the only shipped modifier on top of this
// baseline is the per-clip `handFollow` stylization gain, so the baseline is
// simply handFollow = 1. Kept as a named helper so diagnostics stay explicit
// about which configuration they measured.
export function baselineOptions(data) {
  void data;
  return { handFollow: 1 };
}

export class Retargeter {
  constructor({ bones, orderedBones, hips, hipsParent, data, inPlace, rig, srcMap, handFollow, ...rest }) {
    if ('guards' in rest || 'foreRollSrc' in rest)
      throw new Error('retarget guards and the foreRollSrc switch were removed after the '
        + 'guard ablation (evidence/README.md): the transfer is unguarded and source '
        + 'forearm roll is automatic for quaternion clips');
    if (!data || !Array.isArray(data.names) || !Array.isArray(data.rest) || !Array.isArray(data.pos))
      throw new Error('motion data requires names, rest, and pos arrays');
    if (!Array.isArray(orderedBones) || !orderedBones.length)
      throw new Error('target requires a non-empty orderedBones array');
    if (!bones || typeof bones !== 'object') throw new Error('target requires a bone-name map');
    const targetNames = new Set();
    for (const b of orderedBones) {
      if (!b?.isBone || !b.name) throw new Error('every ordered target bone must be named');
      if (targetNames.has(b.name)) throw new Error(`target bone names must be unique; duplicate "${b.name}"`);
      if (bones[b.name] !== b) throw new Error(`target bone map does not match ordered bone "${b.name}"`);
      targetNames.add(b.name);
    }
    this.bones = bones; this.orderedBones = orderedBones;
    this.rig = rig ?? resolveRig(orderedBones);
    this.R = this.rig.map;                    // role -> actual bone name
    if (!this.rig.ok)
      throw new Error('rigmap: incomplete humanoid rig; missing=' + this.rig.missing.join(','));
    const missingTarget = TARGET_REQUIRED.filter(role => !this.R[role] || !bones[this.R[role]]);
    if (missingTarget.length)
      throw new Error('target bone map is missing resolved roles: ' + missingTarget.join(','));
    this.roleOf = Object.create(null);          // actual bone name -> role
    for (const [role, name] of Object.entries(this.R)) this.roleOf[name] = role;

    this.hips = hips ?? bones[this.R.Hips];
    if (!this.hips) throw new Error(`target bones do not contain resolved Hips "${this.R.Hips}"`);
    this.hipsParent = hipsParent ?? this.hips.parent;
    if (!this.hipsParent)
      throw new Error('target Hips must have a parent transform (use rigFromBones for standalone rigs)');
    this.data = data;
    this.idx = Object.create(null);
    data.names.forEach((n, i) => {
      if (typeof n !== 'string' || !n) throw new Error(`motion joint ${i} has no valid name`);
      if (hasOwn(this.idx, n)) throw new Error(`motion joint names must be unique; duplicate "${n}"`);
      this.idx[n] = i;
    });
    if (data.rest.length !== data.names.length)
      throw new Error(`motion rest has ${data.rest.length} joints; expected ${data.names.length}`);
    this.numFrames = data.numFrames ?? data.pos.length;
    if (!Number.isInteger(this.numFrames) || this.numFrames < 1 || data.pos.length !== this.numFrames)
      throw new Error(`motion numFrames (${this.numFrames}) must match pos length (${data.pos.length})`);
    this.fps = data.fps ?? 30;
    if (!Number.isFinite(this.fps) || this.fps <= 0) throw new Error(`motion fps must be positive; got ${this.fps}`);
    const hasQuat = Array.isArray(data.quat), hasRestQuat = Array.isArray(data.restQuat);
    if (hasQuat !== hasRestQuat)
      throw new Error('motion must provide quat and restQuat together, or omit both');
    if (hasQuat && (data.quat.length !== this.numFrames || data.restQuat.length !== data.names.length))
      throw new Error('motion quaternion arrays do not match names/numFrames');
    // source-skeleton map (canonical source role -> data joint name); baked
    // clips carry their own (data.srcMap), default SOMA. Anchor roles fall back
    // to their limb roles when the source has no separate anchor joints
    // (a GLB-rig source: shoulder line = the Arm joints, etc.).
    this.S = { ...(srcMap ?? data.srcMap ?? SOMA_SRC) };
    for (const [anchor, fb] of [['LeftHipAnchor', 'LeftUpLeg'], ['RightHipAnchor', 'RightUpLeg'],
      ['LeftShoulderAnchor', 'LeftArm'], ['RightShoulderAnchor', 'RightArm']]) {
      if (!hasOwn(this.idx, this.S[anchor])) this.S[anchor] = this.S[fb];
    }
    for (const role of SOURCE_REQUIRED) {
      if (!hasOwn(this.idx, this.S[role]))
        throw new Error(`srcMap: source joint for ${role} ("${this.S[role]}") not in motion data`);
    }
    const validVec = (v, n) => Array.isArray(v) && v.length === n && v.every(Number.isFinite);
    const validQuat = (q) => validVec(q, 4) && q.reduce((sum, x) => sum + x * x, 0) > 1e-12;
    for (const role of SOURCE_REQUIRED) {
      const i = this.idx[this.S[role]];
      if (!validVec(data.rest[i], 3)) throw new Error(`motion rest joint "${this.S[role]}" is not a finite vec3`);
      if (hasQuat && !validQuat(data.restQuat[i]))
        throw new Error(`motion restQuat joint "${this.S[role]}" is not a valid quaternion`);
      for (let f = 0; f < this.numFrames; f++) {
        if (!validVec(data.pos[f]?.[i], 3))
          throw new Error(`motion pos frame ${f}, joint "${this.S[role]}" is not a finite vec3`);
        if (hasQuat && !validQuat(data.quat[f]?.[i]))
          throw new Error(`motion quat frame ${f}, joint "${this.S[role]}" is not a valid quaternion`);
      }
    }
    const dist2 = (a, b) => {
      const dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
      return dx * dx + dy * dy + dz * dz;
    };
    for (const [parentRole, childRole] of SOURCE_SEGMENTS) {
      const a = this.idx[this.S[parentRole]], b = this.idx[this.S[childRole]];
      if (dist2(data.rest[a], data.rest[b]) < 1e-12)
        throw new Error(`source rest segment ${parentRole}→${childRole} has zero length`);
      for (let f = 0; f < this.numFrames; f++) {
        if (dist2(data.pos[f][a], data.pos[f][b]) < 1e-12)
          throw new Error(`source segment ${parentRole}→${childRole} collapses at frame ${f}`);
      }
    }
    this.inPlace = inPlace ?? false;
    if (typeof this.inPlace !== 'boolean') throw new Error(`inPlace must be boolean; got ${this.inPlace}`);
    this.yLift = 1;                            // extra gain on upward root displacement (jumps)
    this.hasQuat = hasQuat;
    this.animWorld = {};
    this.animWorldPos = {};                    // bone uuid -> world position this frame

    // spine blend weights from the actual chain (bottom→top): (i+1)/n
    this.spineBlend = {};
    const chain = this.rig.spineChain ?? [];
    chain.forEach((name, i) => { this.spineBlend[name] = (i + 1) / chain.length; });

    // per-role tables resolved to actual bone names + source data names
    this.aimByBone = {};      // actual name -> [srcJointA, srcJointB, base, childActualName, role]
    for (const [role, [srcA, srcB, base]] of Object.entries(AIM)) {
      const name = this.R[role], child = this.R[AIM_CHILD[role]];
      const a = this.S[srcA], b = this.S[srcB];
      if (name && child && bones[name] && bones[child] && hasOwn(this.idx, a) && hasOwn(this.idx, b))
        this.aimByBone[name] = [a, b, base, child, role];
    }
    this.fullqByBone = {};    // actual name -> source joint
    for (const [role, srcRole] of Object.entries(FULLQ)) {
      const name = this.R[role], src = this.S[srcRole];
      if (name && bones[name] && hasOwn(this.idx, src)) this.fullqByBone[name] = src;
    }
    this.handByBone = {};     // actual name -> [srcWrist, srcForearm, forearmActualName, role]
    for (const [role, foreRole] of Object.entries(HAND_LOCAL)) {
      const name = this.R[role], foreName = this.R[foreRole];
      const sw = this.S[role], sf = this.S[foreRole];
      if (name && foreName && bones[name] && bones[foreName] && hasOwn(this.idx, sw) && hasOwn(this.idx, sf))
        this.handByBone[name] = [sw, sf, foreName, role];
    }
    this.headDamp = {};
    for (const [role, k] of Object.entries(HEAD_DAMP)) {
      if (this.R[role] && bones[this.R[role]]) this.headDamp[this.R[role]] = k;
    }

    // character bind data (world quats/dirs at bind pose)
    this.bindWorldQ = {}; this.bindLocalQ = {}; this.bindDir = {}; this.bindLen = {};
    for (const b of orderedBones) {
      this.bindLocalQ[b.name] = b.quaternion.clone();
      this.bindWorldQ[b.name] = b.getWorldQuaternion(new THREE.Quaternion());
    }
    this.hipsBindWorldPos = this.hips.getWorldPosition(new THREE.Vector3());
    // per-bone twist axis in bind-local space (the bone's own world direction
    // at bind: to its first bone child, else along its parent segment) —
    // consumed by the QA swing/twist metrics, not by any runtime clamp
    this.twistAxis = {};
    for (const role of ['LeftHand', 'RightHand']) {
      const name = this.R[role];
      if (!name || !bones[name]) continue;
      const b = bones[name];
      const p0 = b.getWorldPosition(new THREE.Vector3());
      const child = b.children.find(c => c.isBone);
      const ref = child ? child.getWorldPosition(new THREE.Vector3())
        : p0.clone().add(p0.clone().sub(b.parent.getWorldPosition(new THREE.Vector3())));
      const dirWorld = ref.sub(p0).normalize();
      this.twistAxis[name] = dirWorld.applyQuaternion(
        this.bindWorldQ[name].clone().invert()).normalize();
    }
    for (const name in this.aimByBone) {
      const child = this.aimByBone[name][3];
      const p0 = bones[name].getWorldPosition(new THREE.Vector3());
      const p1 = bones[child].getWorldPosition(new THREE.Vector3());
      this.bindLen[name] = p1.distanceTo(p0);
      if (!Number.isFinite(this.bindLen[name]) || this.bindLen[name] < 1e-6)
        throw new Error(`target segment "${name}" has zero or invalid bind length`);
      this.bindDir[name] = p1.sub(p0).normalize();
    }

    // torso capsule (bind): axis hips→chest, radius from hip/shoulder half-span
    const chestName = this.R.Chest ?? (chain.length ? chain[chain.length - 1] : this.R.Hips);
    this.chestBone = bones[chestName];
    const la = this.R.LeftArm && bones[this.R.LeftArm], ra = this.R.RightArm && bones[this.R.RightArm];
    const lu = bones[this.R.LeftUpLeg], ru = bones[this.R.RightUpLeg];
    let halfShoulder = 0.16, halfHip = 0.1;
    if (la && ra) halfShoulder = la.getWorldPosition(new THREE.Vector3())
      .distanceTo(ra.getWorldPosition(new THREE.Vector3())) / 2;
    if (lu && ru) halfHip = lu.getWorldPosition(new THREE.Vector3())
      .distanceTo(ru.getWorldPosition(new THREE.Vector3())) / 2;
    // torso is roughly as deep as the hip half-span; keep a small clearance
    this.capsuleR = Math.min(halfShoulder, halfHip * 1.6) * 0.95;

    // pelvis + chest frames: character bind, source rest
    this.FcBind = this._charPelvisFrame();
    this.FpRest = this._srcPelvisFrame(data.rest);
    this.FchestRest = this._srcChestFrame(data.rest);

    this.srcHipY = data.rest[this.idx[this.S.Hips]][1] || 0.8;
    // root displacement scale: FUNCTIONAL LEG LENGTH ratio (hip→ankle drop at
    // bind), not raw hip-height ratio — with aim-transferred knee angles, the
    // pelvis must drop in proportion to the legs or planted feet sink/float on
    // rigs whose leg-to-hip proportion differs from the source's.
    this.scaleRoot = this.hipsBindWorldPos.y / this.srcHipY;
    const srcAnkles = [this.S.LeftFoot, this.S.RightFoot].filter(n => hasOwn(this.idx, n));
    const charAnkles = [this.R.LeftFoot, this.R.RightFoot].filter(n => n && bones[n]);
    if (srcAnkles.length && charAnkles.length) {
      const srcAnkY = Math.min(...srcAnkles.map(n => data.rest[this.idx[n]][1]));
      const charAnkY = Math.min(...charAnkles.map(n =>
        bones[n].getWorldPosition(new THREE.Vector3()).y));
      const srcLeg = this.srcHipY - srcAnkY, charLeg = this.hipsBindWorldPos.y - charAnkY;
      if (srcLeg > 0.2 && charLeg > 0.05) this.scaleRoot = charLeg / srcLeg;
    }
    if (!Number.isFinite(this.scaleRoot) || this.scaleRoot <= 0)
      throw new Error(`cannot derive a positive root scale; got ${this.scaleRoot}`);

    // foot/toe bones with their bind-pose world heights — consumed by the QA
    // ground-penetration metric (the runtime ground-lift guard was deleted:
    // it never engaged on the representative suite and its temporal settling
    // was history-dependent; see evidence/README.md)
    this.groundBones = [];
    for (const role of ['LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase']) {
      const name = this.R[role];
      if (name && bones[name])
        this.groundBones.push([bones[name], bones[name].getWorldPosition(new THREE.Vector3()).y]);
    }

    // root displacement must be re-based from the source's rest heading onto the
    // character's bind heading (yaw only), like every direction already is —
    // otherwise a character whose bind faces +Z slides 90° off its facing when
    // driven by a +X-facing source (it "moonwalks" sideways).
    const fwdC = new THREE.Vector3(0, 0, 1).applyQuaternion(this.FcBind); fwdC.y = 0; fwdC.normalize();
    const fwdG = new THREE.Vector3(0, 0, 1).applyQuaternion(this.FpRest); fwdG.y = 0; fwdG.normalize();
    const dyaw = Math.atan2(fwdC.x, fwdC.z) - Math.atan2(fwdG.x, fwdG.z);
    this.rootYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dyaw);
    this.rootYawInv = this.rootYaw.clone().invert();
    this._tmp = new THREE.Vector3();
    this.IDENT = new THREE.Quaternion();

    this.FcBindInv = this.FcBind.clone().invert();
    this.handFollow = handFollow ?? data.handFollow ?? 1;
    if (!Number.isFinite(this.handFollow) || this.handFollow < 0 || this.handFollow > 1)
      throw new Error(`handFollow must be between 0 and 1; got ${this.handFollow}`);
    // forearm roll comes from the source's own quaternions whenever the clip
    // carries them — the single quaternion path (the chest-rebase projection
    // measured up to 179° of roll error on real clips; evidence/README.md)
    this.foreRollSrc = this.hasQuat;
    if (this.hasQuat) {
      this.restQInv = {};
      for (const name in this.fullqByBone) {
        const sj = this.fullqByBone[name];
        if (this.idx[sj] === undefined) continue;
        this.restQInv[sj] = this._q(data.restQuat[this.idx[sj]]).invert();
      }
      // hand transfer constants. Per frame the hand target is
      //   foreNowChar · M · srcForeNow⁻¹ · srcWristNow · T
      // with M = bindFore⁻¹·srcForeRest and T = srcWristRest⁻¹·bindHand:
      // the source wrist-vs-forearm delta is mapped through the WORLD axes
      // at rest/bind (both are the same canonical pose), never through raw
      // bone-local axes — source and character rigs disagree arbitrarily on
      // bone frames. (The old form  foreNow·srcForeNow⁻¹·srcWristNow·
      // restW⁻¹·restF·bindFore⁻¹·bindHand  spliced the source-local delta
      // straight into the character chain; with near-rigid wrists both
      // coincide, but real mocap wrist deviations got applied about wrong
      // axes: cocked "skewed fists", caught by qa_endeffectors.mjs.)
      this.handM = {}; this.handT = {}; this.handRide = {};
      for (const name in this.handByBone) {
        const [sw, sf, foreName] = this.handByBone[name];
        this.handM[name] = this.bindWorldQ[foreName].clone().invert()
          .multiply(this._q(data.restQuat[this.idx[sf]]));
        this.handT[name] = this._q(data.restQuat[this.idx[sw]]).invert()
          .multiply(this.bindWorldQ[name]);
        this.handRide[name] = this.bindWorldQ[foreName].clone().invert()
          .multiply(this.bindWorldQ[name]);
      }
      // wrist articulation gain: 1 = transfer the source wrist fully, 0 =
      // the hand rigidly rides the forearm (bind relation). Mocap wrist
      // channels read poorly on fingerless fist meshes — every twitch and
      // roll shows as a "broken" fist — so human-mocap sources bake a low
      // value (clip JSON handFollow) and keep only a hint of wrist life.
      // Authored-wrist sources omit the key and default to full transfer.
    }
  }

  // The complete effective transfer configuration — what a diagnostic run
  // must record so results are attributable to an exact setup, replacing the
  // old ambiguous "--noguards" notion. After the guard cleanup the only
  // modifier left beyond the raw transfer is handFollow.
  configDump() {
    return {
      handFollow: this.handFollow,
      foreRollSrc: this.foreRollSrc,             // derived: quaternion clips only
      inPlace: this.inPlace,
      yLift: this.yLift,
      hasQuat: this.hasQuat,
      scaleRoot: +this.scaleRoot.toFixed(4),
      capsuleR: +this.capsuleR.toFixed(4),       // QA clearance metric radius
      rootYawDeg: +THREE.MathUtils.radToDeg(
        2 * Math.acos(THREE.MathUtils.clamp(Math.abs(this.rootYaw.w), -1, 1))).toFixed(2),
      srcMap: { ...this.S },
    };
  }

  // map a canonical source-space world POINT into character world space with
  // the same root-yaw + root-scale + bind transform the root translation
  // gets (constraint targets, QA expectations)
  mapSrcPoint(p) {
    const srcRestHips = _v(this.data.rest[this.idx[this.S.Hips]]);
    return p.clone().sub(srcRestHips).applyQuaternion(this.rootYaw)
      .multiplyScalar(this.scaleRoot).add(this.hipsBindWorldPos);
  }

  // map a canonical source-space world ROTATION of source joint sj onto
  // character bone `name`: world-axis delta from source rest, yaw-rebased,
  // applied to the character's bind orientation
  mapSrcQuat(q, sj, name) {
    const rest = this.hasQuat ? this._q(this.data.restQuat[this.idx[sj]]) : new THREE.Quaternion();
    const delta = q.clone().multiply(rest.invert());
    return this._yawRebase(delta).multiply(this.bindWorldQ[name]);
  }

  // express a world-axis body delta about the character's own heading
  _yawRebase(q) { return this.rootYaw.clone().multiply(q).multiply(this.rootYawInv); }

  _gp(P, name) { return _v(P[this.idx[name]]); }
  _q(a) { return new THREE.Quaternion(a[0], a[1], a[2], a[3]).normalize(); }
  _cp(name) { return this.bones[name].getWorldPosition(new THREE.Vector3()); }

  _srcPelvisFrame(P) {
    return frameQ(this._gp(P, this.S.Hips), this._gp(P, this.S.Chest),
      this._gp(P, this.S.LeftHipAnchor), this._gp(P, this.S.RightHipAnchor));
  }
  _srcChestFrame(P) {
    const ls = this._gp(P, this.S.LeftShoulderAnchor), rs = this._gp(P, this.S.RightShoulderAnchor);
    const mid = ls.clone().add(rs).multiplyScalar(0.5);
    return frameQ(this._gp(P, this.S.Chest), mid, ls, rs);
  }
  _charPelvisFrame() {
    const chest = this.chestBone ? this.chestBone.name : this.R.Hips;
    return frameQ(this._cp(this.R.Hips), this._cp(chest),
      this._cp(this.R.LeftUpLeg), this._cp(this.R.RightUpLeg));
  }
  // world position of a bone as of THIS frame's already-applied parents:
  // parent world pos + parent world quat * (world-scaled local offset). Bones are
  // processed in DFS order, so a bone's parent is always finalized before the bone.
  // The scale factor matters: Mixamo/Meshy exports carry a 0.01-scaled armature,
  // so raw local offsets are in centimeters while the scene is in meters.
  _framePos(b) {
    if (this.animWorldPos[b.uuid]) return this.animWorldPos[b.uuid];
    let p;
    if (b === this.hips) {
      p = this.hips.position.clone();
      this.hipsParent.localToWorld(p);
    } else {
      const pq = this.animWorld[b.parent.uuid] ?? b.parent.getWorldQuaternion(new THREE.Quaternion());
      p = b.position.clone().multiply(this._worldScale(b.parent)).applyQuaternion(pq)
        .add(this._framePos(b.parent));
    }
    this.animWorldPos[b.uuid] = p;
    return p;
  }

  // parent world scale at bind (constant: animation only writes rotations).
  // Assumes uniform-ish scale, which is what armature exports carry.
  _worldScale(b) {
    if (!this._scaleCache) this._scaleCache = {};
    return this._scaleCache[b.uuid] ??= b.getWorldScale(new THREE.Vector3());
  }

  applyFrame(f) {
    f = Number.isFinite(f) ? (((Math.round(f) % this.numFrames) + this.numFrames) % this.numFrames) : 0;
    const P = this.data.pos[f];
    this.animWorldPos = {};

    // body-frame deltas are YAW-REBASED onto the character's bind heading (same
    // rebase the root translation gets): the source's world-axis delta must act
    // about the character's own body axes, or a source PITCH becomes a partial
    // ROLL on any rig whose bind heading differs from the source's rest heading.
    // Invisible in yaw-dominated motion (walking); breaks crouches / kick apexes.
    const Fp = this._srcPelvisFrame(P);
    const pelvisDelta = this._yawRebase(Fp.clone().multiply(this.FpRest.clone().invert()));
    const Fc = pelvisDelta.clone().multiply(this.FcBind);            // char pelvis frame (limb aim)
    const FpInv = Fp.clone().invert();
    const Fchest = this._srcChestFrame(P);
    const chestDelta = this._yawRebase(Fchest.clone().multiply(this.FchestRest.clone().invert()));

    // root translation (yaw-rebased from the source rest heading onto char bind heading)
    this._tmp.copy(this._gp(P, this.S.Hips)).sub(_v(this.data.rest[this.idx[this.S.Hips]]));
    this._tmp.applyQuaternion(this.rootYaw);
    if (this.inPlace) { this._tmp.x = 0; this._tmp.z = 0; }
    this._tmp.multiplyScalar(this.scaleRoot);
    if (this.yLift !== 1 && this._tmp.y > 0) this._tmp.y *= this.yLift;
    this._tmp.add(this.hipsBindWorldPos);
    const hipsWorld = this._tmp.clone();
    const hp = this._tmp.clone(); this.hipsParent.worldToLocal(hp); this.hips.position.copy(hp);

    // per-frame transfer state, exposed for QA / constraint IK
    this.frameState = { f, Fp, FpInv, Fc, pelvisDelta, chestDelta, hipsWorld };
    this.rawTargets = {};                     // bone name -> pre-guard demand (world quat)

    for (const b of this.orderedBones) {
      const parentWorld = (b.parent && this.animWorld[b.parent.uuid]) ? this.animWorld[b.parent.uuid]
        : (b.parent ? b.parent.getWorldQuaternion(new THREE.Quaternion()) : this.IDENT);
      let target;
      const role = this.roleOf[b.name];
      if (b === this.hips) {
        target = pelvisDelta.clone().multiply(this.bindWorldQ[b.name]);
      } else if (this.spineBlend[b.name] !== undefined) {
        const D = pelvisDelta.clone().slerp(chestDelta, this.spineBlend[b.name]);
        target = D.multiply(this.bindWorldQ[b.name]);
      } else if (this.aimByBone[b.name]) {
        // rebase on the moving body frame so the bone's roll follows the pelvis/chest
        // naturally; the aim then only adds the residual swing (near-zero twist artifact)
        const [sj, scj, base, childName, aimRole] = this.aimByBone[b.name];
        const D = base === 'chest' ? chestDelta : pelvisDelta;
        const dLimb = this._gp(P, scj).sub(this._gp(P, sj)).normalize();
        const dTarget = dLimb.applyQuaternion(FpInv).applyQuaternion(Fc).normalize();
        const baseDir = this.bindDir[b.name].clone().applyQuaternion(D).normalize();
        const qAim = new THREE.Quaternion().setFromUnitVectors(baseDir, dTarget);
        target = qAim.multiply(D.clone()).multiply(this.bindWorldQ[b.name]);
        // FOREARM ROLL from the source (every quaternion clip): the
        // body-rebase roll above projects chest pitch onto the forearm
        // axis — during a deep jump crouch the forearm SPINS up to 45°/frame
        // while pointing the same way, and the fist (riding it) spins too.
        // Rebuild the forearm from the source forearm's full world delta
        // (carries the true mocap pronation/supination — stable), then
        // rotate minimally onto the aimed direction. The residual minrot is
        // small (source dir ≈ aimed dir), so it adds no twist artifact.
        if (this.foreRollSrc &&
            (aimRole === 'LeftForeArm' || aimRole === 'RightForeArm')) {
          const dSrc = this._yawRebase(this._q(this.data.quat[f][this.idx[sj]])
            .multiply(this._q(this.data.restQuat[this.idx[sj]]).invert()));
          const candDir = this.bindDir[b.name].clone().applyQuaternion(dSrc).normalize();
          const fix = new THREE.Quaternion().setFromUnitVectors(candDir, dTarget);
          target = fix.multiply(dSrc).multiply(this.bindWorldQ[b.name]);
        }
      } else if (this.hasQuat && this.handByBone[b.name]) {
        // hands: FOREARM-RELATIVE — ride the character's forearm and add the
        // source's own wrist-vs-forearm delta (identity at source rest).
        // See HAND_LOCAL above for why not chest-anchored.
        const [sw, sf, foreName] = this.handByBone[b.name];
        const foreNow = this.animWorld[this.bones[foreName].uuid] ?? this.bindWorldQ[foreName];
        target = foreNow.clone()
          .multiply(this.handM[b.name])
          .multiply(this._q(this.data.quat[f][this.idx[sf]]).invert())
          .multiply(this._q(this.data.quat[f][this.idx[sw]]))
          .multiply(this.handT[b.name]);
        this.rawTargets[b.name] = target.clone();   // full source wrist demand
        if (this.handFollow < 1) {
          const ride = foreNow.clone().multiply(this.handRide[b.name]);
          target = ride.slerp(target, this.handFollow);
        }
      } else if (this.hasQuat && this.fullqByBone[b.name] && this.restQInv[this.fullqByBone[b.name]]) {
        // feet: true source ankle orientation, transferred pelvis-relative
        const sq = this.fullqByBone[b.name];
        const qNow = this._q(this.data.quat[f][this.idx[sq]]);
        target = Fc.clone().multiply(FpInv).multiply(qNow).multiply(this.restQInv[sq])
          .multiply(this.FpRest).multiply(this.FcBindInv).multiply(this.bindWorldQ[b.name]);
        this.rawTargets[b.name] = target.clone();
      } else if (this.headDamp[b.name] !== undefined) {
        const D = pelvisDelta.clone().slerp(chestDelta, this.headDamp[b.name]);
        target = D.multiply(this.bindWorldQ[b.name]);
      } else {
        target = parentWorld.clone().multiply(this.bindLocalQ[b.name]);   // shoulders/fingers/toes ride parent
      }
      const localQ = parentWorld.clone().invert().multiply(target);
      b.quaternion.copy(localQ);
      this.animWorld[b.uuid] = target;
    }
  }
}
