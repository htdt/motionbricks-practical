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
// applies every frame sequentially (continuity guard stays armed), and records
// each joint's local rotation plus the hips' local translation into a new glTF
// animation named after the move. Horizontal root motion is exported separately
// to <rootmotion.json>: absolute pelvis X/Z per frame in character-scaled
// meters (× scaleRoot), for the entity layer to integrate (INTEGRATE.md §3).
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { NodeIO } from '@gltf-transform/core';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, resetBindPose } from './align.js';
import { Retargeter, SOMA_SRC } from './retarget.js';

export async function prebake({ glb, manifest, out, rootmotion, ylift = {}, log = console.log }) {
  const outPath = out ?? glb.replace(/\.glb$/, '_anim.glb');
  const rmPath = rootmotion ?? path.join(path.dirname(outPath), 'rootmotion.json');

  const manifestData = JSON.parse(fs.readFileSync(manifest, 'utf8'));
  const bakedDir = path.dirname(manifest);
  const moves = manifestData.moves ?? manifestData;   // accept {moves:[...]} or [...]

  const { doc, bones, byNode } = await loadGLBBones(glb);
  const target = rigFromBones(bones);
  const nodeForBone = new Map();               // THREE.Bone -> glTF Node
  for (const [node, bone] of byNode) nodeForBone.set(bone, node);

  const buffer = doc.getRoot().listBuffers()[0] ?? doc.createBuffer();
  const rootMotion = { scaleRoot: null, clips: {} };

  for (const mv of moves) {
    const name = mv.name ?? mv.move;
    const file = mv.file ?? `${name}.json`;
    const data = JSON.parse(fs.readFileSync(path.join(bakedDir, file), 'utf8'));
    const srcMap = data.srcMap ?? SOMA_SRC;
    const N = data.numFrames ?? data.pos.length;
    const fps = data.fps ?? 30;

    resetBindPose(target);                     // pristine bind before each construction
    const rt = new Retargeter({ ...target, data, inPlace: true, srcMap: data.srcMap });
    if (ylift[name]) rt.yLift = ylift[name];
    rootMotion.scaleRoot = rt.scaleRoot;

    // sample: local rotation for every joint, local translation for hips
    const times = new Float32Array(N);
    for (let f = 0; f < N; f++) times[f] = f / fps;
    const quats = new Map();                   // bone -> Float32Array(N*4)
    for (const b of target.orderedBones) quats.set(b, new Float32Array(N * 4));
    const hipsPos = new Float32Array(N * 3);

    const hipsIdx = data.names.indexOf(srcMap.Hips);
    const rmXZ = new Array(N);
    for (let f = 0; f < N; f++) {
      rt.applyFrame(f);
      for (const b of target.orderedBones) {
        const q = quats.get(b);
        q[f * 4] = b.quaternion.x; q[f * 4 + 1] = b.quaternion.y;
        q[f * 4 + 2] = b.quaternion.z; q[f * 4 + 3] = b.quaternion.w;
      }
      hipsPos[f * 3] = target.hips.position.x;
      hipsPos[f * 3 + 1] = target.hips.position.y;
      hipsPos[f * 3 + 2] = target.hips.position.z;
      rmXZ[f] = [
        data.pos[f][hipsIdx][0] * rt.scaleRoot,
        data.pos[f][hipsIdx][2] * rt.scaleRoot,
        data.pos[f][hipsIdx][1] * rt.scaleRoot,   // world hip height (info/QA)
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
  await new NodeIO().write(outPath, doc);
  fs.writeFileSync(rmPath, JSON.stringify(rootMotion));
  log(`wrote ${outPath} (${moves.length} animations) + ${rmPath}, scaleRoot=${rootMotion.scaleRoot?.toFixed(3)}`);
  return { outPath, rmPath, rootMotion };
}

// ------------------------------------------------------------------- CLI
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  const glb = args.find(a => !a.startsWith('--'));
  const opt = (name, dflt) => {
    const i = args.indexOf('--' + name);
    return i >= 0 ? args[i + 1] : dflt;
  };
  if (!glb || !opt('manifest', null)) {
    console.error('usage: node prebake.mjs <char.glb> --manifest baked/manifest.json --out out.glb [--rootmotion rm.json] [--ylift move=k,...]');
    process.exit(2);
  }
  const ylift = {};
  for (const kv of (opt('ylift', '') || '').split(',').filter(Boolean)) {
    const [k, v] = kv.split('=');
    ylift[k] = parseFloat(v);
  }
  await prebake({
    glb,
    manifest: opt('manifest'),
    out: opt('out', undefined),
    rootmotion: opt('rootmotion', undefined),
    ylift,
  });
}
