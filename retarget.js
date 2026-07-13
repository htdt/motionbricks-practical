import * as THREE from 'three';
import { resolveRig } from './rigmap.js';

// Position-based, pelvis-relative retargeter — RIG-AGNOSTIC version.
//
// Reads the G1 joint WORLD POSITIONS, builds a pelvis frame and a chest frame from
// positions, and re-applies the motion on the character: limbs are aimed at their child
// joint; the spine is bent/twisted from the pelvis frame (bottom) to the chest frame
// (top). Driving from positions means knees/elbows point where the G1's do and cannot
// invert, and the chest frame (built from the shoulder line) carries the torso twist.
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
//  - feet take the G1 ankle body's TRUE orientation (data.quat), transferred
//    pelvis-relative, giving real heel-strike/toe-off instead of inheriting the shin;
//  - HANDS take the G1 wrist body's TRUE orientation, transferred chest-relative, then
//    CLAMPED to anatomical limits relative to the forearm (twist/swing decomposition) —
//    fixes the "hand spinning on the wrist" artifact;
//  - neck/head are damped toward the overall body heading instead of riding chest twist 1:1.
//
// Body-clip guard: a torso capsule (hips→chest axis, radius from the shoulder span)
// pushes arm aim directions outward when the elbow/hand target would land inside the
// torso — arms can graze the body but no longer pass through it.

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
  const bones = {};
  skinned.skeleton.bones.forEach(b => { bones[b.name] = b; });
  const rig = resolveRig(skinned.skeleton.bones);
  const hips = rig.map.Hips ? bones[rig.map.Hips] : bones['Hips'];
  return { model, skinned, bones, hips, rig, animations: gltf.animations };
}

// SOURCE-SKELETON MAP: canonical source roles -> joint names in the motion data.
// The default is the Unitree G1 (MotionBricks canonical skeleton), preserving the
// original behavior; align.js builds the same map for any GLB rig via rigmap, which
// is what makes the retargeter a general two-skeleton aligner.
// *Anchor roles are frame-building references (may coincide with limb roles on rigs
// that don't have separate yaw/pitch links):
//   Hips/Chest        pelvis + chest frame origins
//   L/RHipAnchor      the hip line (pelvis frame right axis)
//   L/RShoulderAnchor the shoulder line (chest frame right axis + origin midpoint)
// NOTE (G1): thigh/upper-arm segments start at the hip_ROLL / shoulder_ROLL links —
// the anatomical joint centers (the yaw links sit several cm along the limb).
export const G1_SRC = {
  Hips: 'pelvis', Chest: 'torso_link',
  LeftHipAnchor: 'left_hip_yaw_link', RightHipAnchor: 'right_hip_yaw_link',
  LeftShoulderAnchor: 'left_shoulder_pitch_link', RightShoulderAnchor: 'right_shoulder_pitch_link',
  LeftUpLeg: 'left_hip_roll_link', LeftLeg: 'left_knee_link', LeftFoot: 'left_ankle_roll_link',
  RightUpLeg: 'right_hip_roll_link', RightLeg: 'right_knee_link', RightFoot: 'right_ankle_roll_link',
  LeftArm: 'left_shoulder_roll_link', LeftForeArm: 'left_elbow_link', LeftHand: 'left_wrist_yaw_link',
  RightArm: 'right_shoulder_roll_link', RightForeArm: 'right_elbow_link', RightHand: 'right_wrist_yaw_link',
};

// SOMA source map (NVIDIA Kimodo somaskel77 output — a human skeleton, so the
// anatomical joint centers ARE the named joints; anchors coincide with limbs
// and fall back inside the constructor). Baked Kimodo clips carry this map in
// their JSON (data.srcMap), so players don't need to know the source family.
export const SOMA_SRC = {
  Hips: 'Hips', Chest: 'Chest',
  LeftHipAnchor: 'LeftLeg', RightHipAnchor: 'RightLeg',
  LeftShoulderAnchor: 'LeftArm', RightShoulderAnchor: 'RightArm',
  LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftShin', LeftFoot: 'LeftFoot',
  RightUpLeg: 'RightLeg', RightLeg: 'RightShin', RightFoot: 'RightFoot',
  LeftArm: 'LeftArm', LeftForeArm: 'LeftForeArm', LeftHand: 'LeftHand',
  RightArm: 'RightArm', RightForeArm: 'RightForeArm', RightHand: 'RightHand',
};

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
// arm roles whose aim goes through the torso-capsule guard
const CLIP_GUARD = new Set(['LeftArm', 'LeftForeArm', 'RightArm', 'RightForeArm']);
// full-orientation transfer: role -> [source role, base frame]. FEET only:
// both rigs' rest feet are flat on the ground pointing forward, so the
// pelvis-relative absolute transfer is anchored correctly.
const FULLQ = {
  LeftFoot:  ['LeftFoot',  'pelvis'],
  RightFoot: ['RightFoot', 'pelvis'],
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
// bones under the temporal continuity guard (see applyFrame): hands + feet
// (full-orientation) and forearms (aim can flip at the antipode)
const CONT_ROLES = ['LeftHand', 'RightHand', 'LeftForeArm', 'RightForeArm', 'LeftFoot', 'RightFoot'];
// anatomical clamps for full-orientation bones, relative to their parent's bind
// relation (deg): twist = rotation about the bone axis, swing = the rest.
const CLAMP = {
  LeftHand:  { twist: 85, swing: 70 },
  RightHand: { twist: 85, swing: 70 },
};
// head/neck: damped toward the overall body heading instead of riding chest twist 1:1
const HEAD_DAMP = { Neck: 0.45, Head: 0.2 };

const _v = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function frameQ(origin, upPt, rA, rB) {
  const up = upPt.clone().sub(origin).normalize();
  const r0 = rA.clone().sub(rB).normalize();
  const fwd = new THREE.Vector3().crossVectors(r0, up).normalize();
  const right = new THREE.Vector3().crossVectors(up, fwd).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, up, fwd));
}

// swing-twist decomposition of q about axis (unit). returns {swing, twist}
function swingTwist(q, axis) {
  const r = new THREE.Vector3(q.x, q.y, q.z);
  const proj = axis.clone().multiplyScalar(r.dot(axis));
  const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, q.w).normalize();
  if (r.dot(axis) < 0 && q.w < 0) { /* keep shortest */ }
  const swing = q.clone().multiply(twist.clone().invert());
  return { swing, twist };
}

function clampAngle(q, maxDeg) {
  const maxRad = THREE.MathUtils.degToRad(maxDeg);
  const w = THREE.MathUtils.clamp(q.w, -1, 1);
  const ang = 2 * Math.acos(Math.abs(w));
  if (ang <= maxRad || ang === 0) return q;
  return new THREE.Quaternion().slerp(q, maxRad / ang);   // identity→q, partial
}

export class Retargeter {
  constructor({ bones, orderedBones, hips, hipsParent, data, inPlace, rig, guards, handClampDeg, srcMap }) {
    this.bones = bones; this.orderedBones = orderedBones;
    this.rig = rig ?? resolveRig(orderedBones);
    this.R = this.rig.map;                    // role -> actual bone name
    if (!this.R.Hips) throw new Error('rigmap: could not resolve Hips; missing=' + this.rig.missing);
    this.roleOf = {};                          // actual bone name -> role
    for (const [role, name] of Object.entries(this.R)) this.roleOf[name] = role;

    this.hips = hips ?? bones[this.R.Hips];
    this.hipsParent = hipsParent ?? this.hips.parent;
    this.data = data;
    this.idx = {}; data.names.forEach((n, i) => this.idx[n] = i);
    // source-skeleton map (canonical source role -> data joint name); default G1.
    // Anchor roles fall back to their limb roles when the source has no separate
    // anchor joints (a GLB-rig source: shoulder line = the Arm joints, etc.).
    this.S = { ...(srcMap ?? data.srcMap ?? G1_SRC) };
    for (const [anchor, fb] of [['LeftHipAnchor', 'LeftUpLeg'], ['RightHipAnchor', 'RightUpLeg'],
      ['LeftShoulderAnchor', 'LeftArm'], ['RightShoulderAnchor', 'RightArm']]) {
      if (!(this.S[anchor] in this.idx)) this.S[anchor] = this.S[fb];
    }
    for (const role of ['Hips', 'Chest', 'LeftUpLeg', 'LeftLeg', 'RightUpLeg', 'RightLeg']) {
      if (!(this.S[role] in this.idx))
        throw new Error(`srcMap: source joint for ${role} ("${this.S[role]}") not in motion data`);
    }
    this.numFrames = data.numFrames; this.fps = data.fps;
    this.inPlace = inPlace ?? false;
    this.guards = { handClamp: true, torsoCapsule: true, ground: false, continuity: true, ...(guards ?? {}) };
    // temporal continuity guard: cap per-frame rotation of hands/forearms/feet
    // during SEQUENTIAL playback (auto-bypassed on random frame access, so the
    // probe battery measures the raw transfer). Catches the two flip sources:
    // clamp branch changes and aim antipodes.
    this.maxStepDeg = 40;
    this._contPrev = {};
    this._lastF = null;
    this.yLift = 1;                            // extra gain on upward root displacement (jumps)
    // per-character override of the hand twist/swing limits (deg). The G1 holds its
    // wrists with a constant bias vs some characters' bind hands, so the transfer can
    // sit ON the clamp — tightening it per character is how that bias is absorbed.
    this.clampLim = {};
    for (const role in CLAMP) this.clampLim[role] = { ...CLAMP[role], ...(handClampDeg ?? {}) };
    this.hasQuat = Array.isArray(data.quat) && Array.isArray(data.restQuat);
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
      if (name && child && bones[name] && bones[child] && a in this.idx && b in this.idx)
        this.aimByBone[name] = [a, b, base, child, role];
    }
    this.fullqByBone = {};    // actual name -> [srcJoint, base, role]
    for (const [role, [srcRole, base]] of Object.entries(FULLQ)) {
      const name = this.R[role], src = this.S[srcRole];
      if (name && bones[name] && src in this.idx) this.fullqByBone[name] = [src, base, role];
    }
    this.handByBone = {};     // actual name -> [srcWrist, srcForearm, forearmActualName, role]
    for (const [role, foreRole] of Object.entries(HAND_LOCAL)) {
      const name = this.R[role], foreName = this.R[foreRole];
      const sw = this.S[role], sf = this.S[foreRole];
      if (name && foreName && bones[name] && bones[foreName] && sw in this.idx && sf in this.idx)
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
    // twist axis for clamped bones, in the bone's bind-local space: the bone's own
    // world direction at bind (to its first bone child, else along its parent segment)
    this.clampAxis = {};
    for (const role in CLAMP) {
      const name = this.R[role];
      if (!name || !bones[name]) continue;
      const b = bones[name];
      const p0 = b.getWorldPosition(new THREE.Vector3());
      const child = b.children.find(c => c.isBone);
      const ref = child ? child.getWorldPosition(new THREE.Vector3())
        : p0.clone().add(p0.clone().sub(b.parent.getWorldPosition(new THREE.Vector3())));
      const dirWorld = ref.sub(p0).normalize();
      this.clampAxis[name] = dirWorld.applyQuaternion(
        this.bindWorldQ[name].clone().invert()).normalize();
    }
    for (const name in this.aimByBone) {
      const child = this.aimByBone[name][3];
      const p0 = bones[name].getWorldPosition(new THREE.Vector3());
      const p1 = bones[child].getWorldPosition(new THREE.Vector3());
      this.bindLen[name] = p1.distanceTo(p0);
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
    this.FchestBind = this._charChestFrame();
    this.FpRest = this._srcPelvisFrame(data.rest);
    this.FchestRest = this._srcChestFrame(data.rest);

    this.srcHipY = data.rest[this.idx[this.S.Hips]][1] || 0.8;
    this.g1HipY = this.srcHipY;               // legacy alias
    // root displacement scale: FUNCTIONAL LEG LENGTH ratio (hip→ankle drop at
    // bind), not raw hip-height ratio — with aim-transferred knee angles, the
    // pelvis must drop in proportion to the legs or planted feet sink/float on
    // rigs whose leg-to-hip proportion differs from the source's.
    this.scaleRoot = this.hipsBindWorldPos.y / this.srcHipY;
    const srcAnkles = [this.S.LeftFoot, this.S.RightFoot].filter(n => n in this.idx);
    const charAnkles = [this.R.LeftFoot, this.R.RightFoot].filter(n => n && bones[n]);
    if (srcAnkles.length && charAnkles.length) {
      const srcAnkY = Math.min(...srcAnkles.map(n => data.rest[this.idx[n]][1]));
      const charAnkY = Math.min(...charAnkles.map(n =>
        bones[n].getWorldPosition(new THREE.Vector3()).y));
      const srcLeg = this.srcHipY - srcAnkY, charLeg = this.hipsBindWorldPos.y - charAnkY;
      if (srcLeg > 0.2 && charLeg > 0.05) this.scaleRoot = charLeg / srcLeg;
    }

    // ground clamp: foot/toe bones with their bind-pose world heights — during
    // animation none of them may sink below its bind height (proportion mismatch
    // between the G1's legs and the character's puts feet underground otherwise)
    this.groundBones = [];
    for (const role of ['LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase']) {
      const name = this.R[role];
      if (name && bones[name])
        this.groundBones.push([bones[name], bones[name].getWorldPosition(new THREE.Vector3()).y]);
    }
    this.groundOffset = 0;

    // root displacement must be re-based from the G1's rest heading onto the
    // character's bind heading (yaw only), like every direction already is —
    // otherwise a character whose bind faces +Z slides 90° off its facing when
    // driven by the +X-facing G1 (it "moonwalks" sideways).
    const fwdC = new THREE.Vector3(0, 0, 1).applyQuaternion(this.FcBind); fwdC.y = 0; fwdC.normalize();
    const fwdG = new THREE.Vector3(0, 0, 1).applyQuaternion(this.FpRest); fwdG.y = 0; fwdG.normalize();
    const dyaw = Math.atan2(fwdC.x, fwdC.z) - Math.atan2(fwdG.x, fwdG.z);
    this.rootYaw = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), dyaw);
    this.rootYawInv = this.rootYaw.clone().invert();
    this._tmp = new THREE.Vector3();
    this.IDENT = new THREE.Quaternion();

    this.FcBindInv = this.FcBind.clone().invert();
    this.FchestBindInv = this.FchestBind.clone().invert();
    if (this.hasQuat) {
      this.restQInv = {};
      for (const name in this.fullqByBone) {
        const g1 = this.fullqByBone[name][0];
        if (this.idx[g1] === undefined) continue;
        this.restQInv[g1] = this._q(data.restQuat[this.idx[g1]]).invert();
      }
      // hand transfer constants. Per frame the hand target is
      //   foreNowChar · M · srcForeNow⁻¹ · srcWristNow · T
      // with M = bindFore⁻¹·srcForeRest and T = srcWristRest⁻¹·bindHand:
      // the source wrist-vs-forearm delta is mapped through the WORLD axes
      // at rest/bind (both are the same canonical pose), never through raw
      // bone-local axes — source and character rigs disagree arbitrarily on
      // bone frames. (The old form  foreNow·srcForeNow⁻¹·srcWristNow·
      // restW⁻¹·restF·bindFore⁻¹·bindHand  spliced the source-local delta
      // straight into the character chain; with near-rigid wrists — the
      // authored-wrist G1 clips — both coincide, but real mocap wrist
      // deviations got applied about wrong axes: cocked "skewed fists" on
      // Kimodo clips, caught by qa_endeffectors.mjs.)
      this.handM = {}; this.handT = {};
      for (const name in this.handByBone) {
        const [sw, sf, foreName] = this.handByBone[name];
        this.handM[name] = this.bindWorldQ[foreName].clone().invert()
          .multiply(this._q(data.restQuat[this.idx[sf]]));
        this.handT[name] = this._q(data.restQuat[this.idx[sw]]).invert()
          .multiply(this.bindWorldQ[name]);
      }
    }
    this.contBones = new Set(CONT_ROLES.map(r => this.R[r]).filter(n => n && bones[n]));
  }

  // express a world-axis body delta about the character's own heading
  _yawRebase(q) { return this.rootYaw.clone().multiply(q).multiply(this.rootYawInv); }

  _gp(P, name) { return _v(P[this.idx[name]]); }
  _q(a) { return new THREE.Quaternion(a[0], a[1], a[2], a[3]); }
  _cp(name) { return this.bones[name].getWorldPosition(new THREE.Vector3()); }
  _role(role) { return this.bones[this.R[role]]; }

  _srcPelvisFrame(P) {
    return frameQ(this._gp(P, this.S.Hips), this._gp(P, this.S.Chest),
      this._gp(P, this.S.LeftHipAnchor), this._gp(P, this.S.RightHipAnchor));
  }
  _srcChestFrame(P) {
    const ls = this._gp(P, this.S.LeftShoulderAnchor), rs = this._gp(P, this.S.RightShoulderAnchor);
    const mid = ls.clone().add(rs).multiplyScalar(0.5);
    return frameQ(this._gp(P, this.S.Chest), mid, ls, rs);
  }
  // legacy aliases (debug.js / check_heading.mjs)
  _g1PelvisFrame(P) { return this._srcPelvisFrame(P); }
  _g1ChestFrame(P) { return this._srcChestFrame(P); }
  _charPelvisFrame() {
    const chest = this.chestBone ? this.chestBone.name : this.R.Hips;
    return frameQ(this._cp(this.R.Hips), this._cp(chest),
      this._cp(this.R.LeftUpLeg), this._cp(this.R.RightUpLeg));
  }
  _charChestFrame() {
    const ls = this._cp(this.R.LeftArm), rs = this._cp(this.R.RightArm);
    const mid = ls.clone().add(rs).multiplyScalar(0.5);
    return frameQ(this._cp(this.chestBone.name), mid, ls, rs);
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

  // push an aim direction out of the torso capsule if the child joint would land
  // inside it. axisA/axisB: capsule segment (hips/chest world pos this frame).
  _capsuleGuard(fromPos, dir, len, axisA, axisB) {
    const end = fromPos.clone().add(dir.clone().multiplyScalar(len));
    const ab = axisB.clone().sub(axisA);
    const t = THREE.MathUtils.clamp(end.clone().sub(axisA).dot(ab) / ab.lengthSq(), 0, 1.15);
    const closest = axisA.clone().add(ab.multiplyScalar(t));
    const radial = end.clone().sub(closest);
    radial.y = 0;                                     // push out horizontally only
    const d = radial.length();
    if (d >= this.capsuleR || d < 1e-6) return dir;
    // move the endpoint to the capsule surface, re-normalize the aim
    const pushed = end.add(radial.multiplyScalar((this.capsuleR - d) / d));
    return pushed.sub(fromPos).normalize();
  }

  // clamp a full-orientation target so the bone stays within anatomical limits
  // relative to its parent (twist about the bone's bind direction, swing = rest)
  _clampToParent(targetWorldQ, b, role, parentWorld) {
    const lim = this.clampLim[role];
    if (!lim || !this.guards.handClamp) return targetWorldQ;
    const bindLocal = this.bindLocalQ[b.name];
    const local = parentWorld.clone().invert().multiply(targetWorldQ);
    const rel = bindLocal.clone().invert().multiply(local);      // deviation from bind, in bind-local space
    if (rel.w < 0) rel.set(-rel.x, -rel.y, -rel.z, -rel.w);
    const axis = this.clampAxis[b.name] ?? new THREE.Vector3(0, 1, 0);
    const { swing, twist } = swingTwist(rel, axis);
    const cs = clampAngle(swing, lim.swing);
    const ct = clampAngle(twist, lim.twist);
    const relClamped = cs.multiply(ct);
    return parentWorld.clone().multiply(bindLocal).multiply(relClamped);
  }

  applyFrame(f) {
    f = Number.isFinite(f) ? (((Math.round(f) % this.numFrames) + this.numFrames) % this.numFrames) : 0;
    const P = this.data.pos[f];
    this.animWorldPos = {};
    // continuity guard only makes sense across consecutive frames
    const seq = this._lastF !== null && ((f - this._lastF + this.numFrames) % this.numFrames) <= 1;
    if (!seq) this._contPrev = {};
    this._lastF = f;

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

    // capsule axis for this frame is computed lazily (hips is known now; chest
    // becomes known once the spine bones are processed — arms come after in DFS)
    let capA = null, capB = null;

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
        const dG1 = this._gp(P, scj).sub(this._gp(P, sj)).normalize();
        let dTarget = dG1.applyQuaternion(FpInv).applyQuaternion(Fc).normalize();
        if (this.guards.torsoCapsule && CLIP_GUARD.has(aimRole)) {
          if (!capA) { capA = this._framePos(this.hips); capB = this._framePos(this.chestBone); }
          dTarget = this._capsuleGuard(this._framePos(b), dTarget, this.bindLen[b.name], capA, capB);
        }
        const baseDir = this.bindDir[b.name].clone().applyQuaternion(D).normalize();
        const qAim = new THREE.Quaternion().setFromUnitVectors(baseDir, dTarget);
        target = qAim.multiply(D.clone()).multiply(this.bindWorldQ[b.name]);
      } else if (this.hasQuat && this.handByBone[b.name]) {
        // hands: FOREARM-RELATIVE — ride the character's forearm and add the
        // source's own wrist-vs-forearm delta (identity at source rest), then
        // clamp as a safety net. See HAND_LOCAL above for why not chest-anchored.
        const [sw, sf, foreName, hRole] = this.handByBone[b.name];
        const foreNow = this.animWorld[this.bones[foreName].uuid] ?? this.bindWorldQ[foreName];
        target = foreNow.clone()
          .multiply(this.handM[b.name])
          .multiply(this._q(this.data.quat[f][this.idx[sf]]).invert())
          .multiply(this._q(this.data.quat[f][this.idx[sw]]))
          .multiply(this.handT[b.name]);
        target = this._clampToParent(target, b, hRole, parentWorld);
      } else if (this.hasQuat && this.fullqByBone[b.name] && this.restQInv[this.fullqByBone[b.name][0]]) {
        // feet: true source ankle orientation, transferred pelvis-relative
        const [g1] = this.fullqByBone[b.name];
        const qNow = this._q(this.data.quat[f][this.idx[g1]]);
        target = Fc.clone().multiply(FpInv).multiply(qNow).multiply(this.restQInv[g1])
          .multiply(this.FpRest).multiply(this.FcBindInv).multiply(this.bindWorldQ[b.name]);
      } else if (this.headDamp[b.name] !== undefined) {
        const D = pelvisDelta.clone().slerp(chestDelta, this.headDamp[b.name]);
        target = D.multiply(this.bindWorldQ[b.name]);
      } else {
        target = parentWorld.clone().multiply(this.bindLocalQ[b.name]);   // shoulders/fingers/toes ride parent
      }
      let localQ = parentWorld.clone().invert().multiply(target);
      if (this.contBones.has(b.name)) {
        if (this.guards.continuity && seq) {
          const prev = this._contPrev[b.name];
          if (prev) {
            const ang = prev.angleTo(localQ);
            const maxStep = THREE.MathUtils.degToRad(this.maxStepDeg);
            if (ang > maxStep) {
              localQ = prev.clone().slerp(localQ, maxStep / ang);
              target = parentWorld.clone().multiply(localQ);   // children must see the guarded pose
            }
          }
        }
        this._contPrev[b.name] = localQ.clone();
      }
      b.quaternion.copy(localQ);
      this.animWorld[b.uuid] = target;
    }

    if (this.guards.ground && this.groundBones.length) {
      // deepest penetration below each foot bone's bind height, pre-offset
      let pen = 0;
      for (const [b, bindY] of this.groundBones) {
        const d = bindY - this._framePos(b).y;
        if (d > pen) pen = d;
      }
      // rise quickly to cover a strike-through, settle slowly when airborne/clear
      this.groundOffset += (pen - this.groundOffset) * (pen > this.groundOffset ? 0.5 : 0.06);
      if (this.groundOffset > 1e-4) {
        const hw = hipsWorld.clone(); hw.y += this.groundOffset;
        this.hipsParent.worldToLocal(hw); this.hips.position.copy(hw);
      }
    }
  }
}
