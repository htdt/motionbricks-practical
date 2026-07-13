// Alignment certification CLI (pipeline Stage 1, see ALIGN.md).
// Usage: node certify.mjs <char.glb> --clips a.json,b.json [--srcmap map.json] [--out cert.json]
// Clips are motion JSONs in the format documented in ALIGN.md; probe poses are
// mined from them deterministically. The source role map is taken from
// --srcmap, else from a `srcMap` field on the first clip, else defaults to the
// SOMA skeleton (SOMA_SRC). Writes <char.glb>.retarget_certificate.json
// next to the GLB unless --out is given.
// Exit code 0 = certified, 1 = gate failure, 2 = rig resolution failure.
import fs from 'node:fs';
import path from 'node:path';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, mineProbeFrames, certifyRig } from './align.js';

const args = process.argv.slice(2);
const glb = args.find(a => !a.startsWith('--'));
const opt = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : dflt;
};
const clipsArg = opt('clips', null);
if (!glb || !clipsArg) {
  console.error('usage: node certify.mjs <char.glb> --clips a.json,b.json [--srcmap map.json] [--out cert.json]');
  process.exit(2);
}
const clipPaths = clipsArg.split(',');
const out = opt('out', glb + '.retarget_certificate.json');

const clips = clipPaths.map(p => JSON.parse(fs.readFileSync(p, 'utf8')));
const srcmapPath = opt('srcmap', null);
const srcMap = srcmapPath ? JSON.parse(fs.readFileSync(srcmapPath, 'utf8'))
  : clips[0].srcMap ?? undefined;
const probes = mineProbeFrames(clips, srcMap);

let target;
try {
  const { bones } = await loadGLBBones(glb);
  target = rigFromBones(bones);
} catch (e) {
  console.error(JSON.stringify({ ok: false, error: 'rig resolution failed: ' + e.message }));
  process.exit(2);
}

const cert = certifyRig(target, probes, srcMap ? { srcMap } : {});
const doc = {
  ok: cert.pass,
  character: path.resolve(glb),
  generatedAt: new Date().toISOString(),
  probeClips: clipPaths.map(p => path.basename(p)),
  ...cert,
};
fs.writeFileSync(out, JSON.stringify(doc, null, 2));
console.log(JSON.stringify({
  ok: cert.pass, certificate: out,
  gates: cert.gates, failures: cert.failures,
  rig: { rolesResolved: cert.rig.rolesResolved, missing: cert.rig.missing },
}, null, 2));
process.exit(cert.pass ? 0 : 1);
