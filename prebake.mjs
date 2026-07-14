// Pre-bake retargeted clips into engine-native GLB animations (INTEGRATE.md §9).
// For engines other than three.js (Babylon, Godot, ...): run the certified
// retargeter offline once per character and export ordinary glTF animations —
// the runtime then plays baked clips with its own animation system and never
// touches retargeting code.
//
// Usage:
//   node prebake.mjs <char.glb> --manifest baked/manifest.json --out char_anim.glb \
//        [--rootmotion rootmotion.json] [--ylift jump_gap=1.5,other=1.2]
//
// Per clip: constructs a Retargeter (inPlace: true — root X/Z zeroed, Y kept),
// applies every frame through the stateless retargeter and deterministic IK, and records
// each joint's local rotation plus the hips' local translation into a new glTF
// animation named after the move. Horizontal root motion is exported separately
// to <rootmotion.json>: pelvis X/Z displacement from source rest in
// character-scaled meters (× scaleRoot), for the entity layer to integrate
// (INTEGRATE.md §3).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, resetBindPose } from './align.js';
import { Retargeter, SOMA_SRC } from './retarget.js';
import { ConstraintIK } from './ik.js';

export async function prebake({ glb, manifest, out, rootmotion, ylift = {}, log = console.log }) {
  if (typeof glb !== 'string' || !glb || typeof manifest !== 'string' || !manifest)
    throw new Error('prebake requires glb and manifest paths');
  if (out !== undefined && (typeof out !== 'string' || !out))
    throw new Error('prebake out must be a non-empty path');
  if (rootmotion !== undefined && (typeof rootmotion !== 'string' || !rootmotion))
    throw new Error('prebake rootmotion must be a non-empty path');
  if (typeof log !== 'function') throw new Error('prebake log must be a function');
  if (!ylift || typeof ylift !== 'object' || Array.isArray(ylift))
    throw new Error('ylift must be a move-to-gain object');
  const ext = path.extname(glb);
  if (ext.toLowerCase() !== '.glb') throw new Error(`prebake input must be a .glb file; got ${glb}`);
  const outPath = out ?? path.join(path.dirname(glb), `${path.basename(glb, ext)}_anim.glb`);
  const rmPath = rootmotion ?? path.join(path.dirname(outPath), 'rootmotion.json');
  const glbAbs = path.resolve(glb), manifestAbs = path.resolve(manifest);
  const outAbs = path.resolve(outPath), rmAbs = path.resolve(rmPath);
  if (outAbs === glbAbs) throw new Error('prebake output must not overwrite the input GLB');
  if (outAbs === manifestAbs) throw new Error('prebake output must not overwrite the manifest');
  if (rmAbs === glbAbs || rmAbs === manifestAbs || rmAbs === outAbs)
    throw new Error('root-motion output must not overwrite an input or the animation GLB');

  const manifestData = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  const bakedDir = path.dirname(manifest);
  const moves = manifestData.moves ?? manifestData;   // accept {moves:[...]} or [...]
  if (!Array.isArray(moves) || !moves.length) throw new Error('manifest must contain a non-empty moves array');
  if (moves.some(mv => !mv || typeof mv !== 'object' || Array.isArray(mv)))
    throw new Error('every manifest move must be an object');
  const moveNames = moves.map(mv => mv.name ?? mv.move);
  if (moveNames.some(name => typeof name !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(name)) ||
      new Set(moveNames).size !== moveNames.length)
    throw new Error('manifest move names must be safe and unique');
  if (moves.some(mv => mv.file !== undefined && (typeof mv.file !== 'string' || !mv.file)))
    throw new Error('manifest clip files must be non-empty strings');
  if (moves.some(mv => mv.loop !== undefined && typeof mv.loop !== 'boolean'))
    throw new Error('manifest loop flags must be boolean');
  const unknownYLift = Object.keys(ylift).filter(name => !moveNames.includes(name));
  if (unknownYLift.length) throw new Error(`ylift names not in manifest: ${unknownYLift.join(', ')}`);

  const { doc, bones, byNode } = await loadGLBBones(glb);
  const target = rigFromBones(bones);
  const existingAnimations = doc.getRoot().listAnimations();
  for (const animation of existingAnimations) animation.dispose();
  if (existingAnimations.length) log(`removed ${existingAnimations.length} input animation(s)`);
  const nodeForBone = new Map();               // THREE.Bone -> glTF Node
  for (const [node, bone] of byNode) nodeForBone.set(bone, node);

  const buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer();
  const rootMotion = { scaleRoot: null, clips: {} };

  for (const mv of moves) {
    const name = mv.name ?? mv.move;
    const file = mv.file ?? `${name}.json`;
    const clipPath = path.resolve(bakedDir, file);
    if (clipPath === outAbs || clipPath === rmAbs)
      throw new Error(`${name}: an output path would overwrite its source clip`);
    const data = JSON.parse(fs.readFileSync(clipPath, 'utf8'));
    const srcMap = data.srcMap ?? SOMA_SRC;
    const N = data.numFrames ?? data.pos.length;
    const fps = data.fps ?? 30;
    if (mv.frames !== undefined && mv.frames !== N)
      throw new Error(`${name}: manifest frames=${mv.frames}, clip numFrames=${N}`);
    if (mv.fps !== undefined && mv.fps !== fps)
      throw new Error(`${name}: manifest fps=${mv.fps}, clip fps=${fps}`);

    resetBindPose(target);                     // pristine bind before each construction
    const rt = new Retargeter({ ...target, data, inPlace: true, srcMap: data.srcMap });
    // clips with authored constraints bake through the same deterministic
    // constraint IK the runtime uses — offline frames match playback exactly
    const ik = Array.isArray(data.constraints) && data.constraints.length
      ? new ConstraintIK(rt, data.constraints,
          mv.reachPolicy === 'clamp' ? { reachPolicy: 'clamp' } : {})
      : null;
    if (ylift[name] !== undefined) {
      if (!Number.isFinite(ylift[name]) || ylift[name] <= 0)
        throw new Error(`ylift for ${name} must be positive; got ${ylift[name]}`);
      rt.yLift = ylift[name];
    }
    if (rootMotion.scaleRoot !== null && Math.abs(rootMotion.scaleRoot - rt.scaleRoot) > 1e-6)
      throw new Error(`${name}: scaleRoot differs across clips; the manifest mixes source skeletons`);
    rootMotion.scaleRoot = rt.scaleRoot;

    // sample: local rotation for every joint, local translation for hips
    const times = new Float32Array(N);
    for (let f = 0; f < N; f++) times[f] = f / fps;
    const quats = new Map();                   // bone -> Float32Array(N*4)
    for (const b of target.orderedBones) quats.set(b, new Float32Array(N * 4));
    const prevQuat = new Map();                 // bone -> last emitted xyzw (sign-continuous)
    const hipsPos = new Float32Array(N * 3);

    const hipsIdx = data.names.indexOf(srcMap.Hips);
    const rmXZ = new Array(N);
    for (let f = 0; f < N; f++) {
      rt.applyFrame(f);
      if (ik) ik.apply(f);
      for (const b of target.orderedBones) {
        const q = quats.get(b);
        const raw = [b.quaternion.x, b.quaternion.y, b.quaternion.z, b.quaternion.w];
        const prev = prevQuat.get(b);
        const sign = prev && prev[0] * raw[0] + prev[1] * raw[1] +
          prev[2] * raw[2] + prev[3] * raw[3] < 0 ? -1 : 1;
        const emitted = raw.map(v => v * sign);
        q.set(emitted, f * 4);
        prevQuat.set(b, emitted);
      }
      hipsPos[f * 3] = target.hips.position.x;
      hipsPos[f * 3 + 1] = target.hips.position.y;
      hipsPos[f * 3 + 2] = target.hips.position.z;
      const srcRoot = data.pos[f][hipsIdx], srcRestRoot = data.rest[hipsIdx];
      let rootDy = srcRoot[1] - srcRestRoot[1];
      if (rootDy > 0) rootDy *= rt.yLift;
      rmXZ[f] = [
        (srcRoot[0] - srcRestRoot[0]) * rt.scaleRoot,
        (srcRoot[2] - srcRestRoot[2]) * rt.scaleRoot,
        rt.hipsBindWorldPos.y + rootDy * rt.scaleRoot,
      ];
    }

    const anim = doc.createAnimation(name);
    const input = doc.createAccessor().setType('SCALAR').setArray(times).setBuffer(buffer);
    for (const b of target.orderedBones) {
      const node = nodeForBone.get(b);
      if (!node) continue;
      const out = doc.createAccessor().setType('VEC4').setArray(quats.get(b)).setBuffer(buffer);
      const s = doc.createAnimationSampler().setInput(input).setOutput(out).setInterpolation('LINEAR');
      const ch = doc.createAnimationChannel().setTargetNode(node).setTargetPath('rotation').setSampler(s);
      anim.addSampler(s).addChannel(ch);
    }
    const hipsNode = nodeForBone.get(target.hips);
    const outT = doc.createAccessor().setType('VEC3').setArray(hipsPos).setBuffer(buffer);
    const sT = doc.createAnimationSampler().setInput(input).setOutput(outT).setInterpolation('LINEAR');
    const chT = doc.createAnimationChannel().setTargetNode(hipsNode).setTargetPath('translation').setSampler(sT);
    anim.addSampler(sT).addChannel(chT);

    rootMotion.clips[name] = {
      fps, numFrames: N, loop: !!mv.loop,
      frameData: mv.frame_data ?? mv.frameData ?? null,
      pelvisXZ: rmXZ.map(v => [v[0], v[1]]),
      hipY: rmXZ.map(v => v[2]),
    };
    log(`baked ${name}: ${N} frames @ ${fps}fps, loop=${!!mv.loop}`);
  }

  resetBindPose(target);
  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.mkdirSync(path.dirname(rmPath), { recursive: true });
  await new NodeIO().write(outPath, doc);
  fs.writeFileSync(rmPath, JSON.stringify(rootMotion));
  log(`wrote ${outPath} (${moves.length} animations) + ${rmPath}, scaleRoot=${rootMotion.scaleRoot?.toFixed(3)}`);
  return { outPath, rmPath, rootMotion };
}

// ------------------------------------------------------------------- CLI
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const known = new Set(['manifest', 'out', 'rootmotion', 'ylift']);
  const opts = Object.create(null), positional = [];
  let parseError = null;
  for (let i = 0; i < args.length && !parseError; i++) {
    const arg = args[i];
    if (!arg.startsWith('--')) { positional.push(arg); continue; }
    const name = arg.slice(2);
    if (!known.has(name)) { parseError = `unknown option --${name}`; continue; }
    if (Object.hasOwn(opts, name)) { parseError = `duplicate option --${name}`; continue; }
    const value = args[++i];
    if (!value || value.startsWith('--')) { parseError = `--${name} requires a value`; continue; }
    opts[name] = value;
  }
  const glb = positional.length === 1 ? positional[0] : null;
  if (parseError || !glb || !opts.manifest) {
    if (parseError) console.error(parseError);
    console.error('usage: node prebake.mjs <char.glb> --manifest baked/manifest.json --out out.glb [--rootmotion rm.json] [--ylift move=k,...]');
    process.exit(2);
  }
  const ylift = {};
  for (const kv of (opts.ylift ?? '').split(',').filter(Boolean)) {
    const parts = kv.split('=');
    if (parts.length !== 2 || !/^[A-Za-z0-9][A-Za-z0-9_-]*$/.test(parts[0]) || parts[1].trim() === '') {
      console.error(`invalid --ylift entry "${kv}"; expected move=positiveNumber`);
      process.exit(2);
    }
    ylift[parts[0]] = Number(parts[1]);
  }
  try {
    await prebake({ glb, manifest: opts.manifest, out: opts.out,
      rootmotion: opts.rootmotion, ylift });
  } catch (e) {
    console.error(e.message);
    process.exit(2);
  }
}
