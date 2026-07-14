#!/usr/bin/env node
// Transfer-modifier ABLATION RUNNER — the evidence generator behind every
// keep-or-delete decision. The historical guards (handClamp, torsoCapsule,
// continuity, ground) and the chest-rebase forearm-roll path were measured
// with this tool and DELETED; their recorded numbers live in
// evidence/README.md and the pre-removal tree in git history. What remains
// to ablate is the retained per-clip stylization (handFollow) and the
// constraint IK, each alone on the raw baseline and in the shipped
// combination, measuring per configuration:
//
//   - hand/foot authored-target position + full-orientation error (stage 4)
//   - source-motion fidelity (full-quaternion wrist/ankle error vs the source
//     demand, swing/twist split, round-trip position recovery)
//   - angular velocity / acceleration and orientation branch flips
//   - limb extension (hyperextension detector)
//   - torso-capsule clearance and ground penetration
//   - contact-frame foot skate (predicted contact channels when present)
//   - determinism: sequential vs direct-seek vs frame-stride playback
//
// Usage: node ablate.mjs <char.glb> <movesDir> [--json out.json] [--clips a,b] [--ik]
// The JSON output is the recorded ablation evidence referenced by KIMODO.md.
import fs from 'node:fs';
import path from 'node:path';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, resetBindPose, roundTripError } from './align.js';
import { Retargeter, baselineOptions } from './retarget.js';
import {
  captureRun, contactMask, footSkate, angularKinematics, poseValidity,
  captureDelta, sourceFidelity, constraintAccuracy, limbExtension, bodyClearance,
  stats, quatAngleDeg, EE_ROLES,
} from './qametrics.mjs';

const args = process.argv.slice(2);
const withValue = new Set(['--json', '--clips']);
const positional = [];
let jsonOut = null, onlyClips = null, USE_IK = false, usageError = null;
for (let i = 0; i < args.length; i++) {
  const a = args[i];
  if (a === '--ik') { USE_IK = true; continue; }
  if (withValue.has(a)) {
    const v = args[++i];
    if (!v || v.startsWith('--')) { usageError = `${a} requires a value`; break; }
    if (a === '--json') jsonOut = v; else onlyClips = new Set(v.split(',').filter(Boolean));
    continue;
  }
  if (a.startsWith('--')) { usageError = `unknown option ${a}`; break; }
  positional.push(a);
}
const [charPath, movesDir] = positional;
if (usageError || !charPath || !movesDir) {
  if (usageError) console.error(usageError);
  console.error('usage: node ablate.mjs <char.glb> <movesDir> [--json out.json] [--clips a,b] [--ik]');
  process.exit(2);
}

const { bones: loadedBones } = await loadGLBBones(charPath);
const target = rigFromBones(loadedBones);
const manifest = JSON.parse(fs.readFileSync(path.join(movesDir, 'manifest.json'), 'utf8'));
const moves = manifest.moves.filter(mv => !onlyClips || onlyClips.has(mv.name));
if (!moves.length) throw new Error('no clips selected');

// each retained intervention alone on top of the exact baseline + shipped
function configsFor(clip) {
  const base = baselineOptions(clip);
  const configs = {
    baseline: { config: base },
    handFollow: { config: { ...base, handFollow: clip.handFollow ?? 0.3 } },
  };
  if (USE_IK) configs.constraintIK = { config: base, ik: true };
  configs.shipped = { config: {}, ik: USE_IK };   // all retained modifiers together
  return configs;
}

const aggAll = (per) => {
  const merged = Object.values(per).filter(Boolean);
  return merged.length ? +Math.max(...merged.map(s => s?.max ?? 0)).toFixed(4) : null;
};

const report = { character: path.resolve(charPath), movesDir: path.resolve(movesDir),
  ik: USE_IK, clips: {} };

for (const mv of moves) {
  const clip = JSON.parse(fs.readFileSync(path.join(movesDir, mv.file), 'utf8'));
  const mask = contactMask(clip);
  const out = {};
  report.clips[mv.name] = out;
  let baselineCap = null;
  for (const [name, spec] of Object.entries(configsFor(clip))) {
    const ik = !!spec.ik && (clip.constraints ?? []).some(r => r.family === 'end-effector');
    const cap = captureRun(target, clip, { config: spec.config, ik });
    if (name === 'baseline') baselineCap = cap;
    const fid = sourceFidelity(cap);
    const kin = angularKinematics(cap);
    const row = {
      config: cap.config,
      fidelityRot: Object.fromEntries(EE_ROLES.map(r => [r, fid[r]?.rot ?? null])),
      fidelityTwist: Object.fromEntries(EE_ROLES.map(r => [r, fid[r]?.twist ?? null])),
      velMax: aggAll(Object.fromEntries(EE_ROLES.map(r => [r, kin[r]?.vel]))),
      accMax: aggAll(Object.fromEntries(EE_ROLES.map(r => [r, kin[r]?.acc]))),
      flips: EE_ROLES.reduce((a, r) => a + (kin[r]?.flips ?? 0), 0),
      extension: limbExtension(cap),
      clearance: bodyClearance(cap),
      skate: footSkate(cap, mask),
      validity: poseValidity(cap),
    };
    // round-trip source-position recovery
    {
      resetBindPose(target);
      const rt = new Retargeter({ ...target, data: clip, ...spec.config });
      const errs = [];
      for (let f = 0; f < cap.N; f++) errs.push(roundTripError(rt, f).mean);
      row.roundTrip = stats(errs);
      resetBindPose(target);
    }
    if ((clip.constraints ?? []).length) row.constraintAccuracy = constraintAccuracy(cap).summary;
    if (name !== 'baseline') row.deltaVsBaseline = (() => {
      const d = captureDelta(baselineCap, cap);
      return { maxRotDeg: d.maxRotDeg, maxPosM: d.maxPosM };
    })();
    // determinism: direct seek, 2x-speed stride, and a loop-wrap evaluation
    // order must reproduce the same poses.
    const seek = captureRun(target, clip, { config: spec.config, ik, order: 'seek' });
    const strideFrames = Array.from({ length: Math.ceil(cap.N / 2) }, (_, k) => 2 * k).filter(f => f < cap.N);
    const stride = captureRun(target, clip, { config: spec.config, ik, order: strideFrames });
    const loopFrames = [cap.N - 1, ...Array.from({ length: cap.N - 1 }, (_, f) => f)];
    const loopWrap = captureRun(target, clip, { config: spec.config, ik, order: loopFrames });
    let strideRot = 0;
    for (const f of strideFrames) {
      for (const role of EE_ROLES) {
        const a = cap.rows[f].roles[role], b = stride.rows[f]?.roles[role];
        if (!b) continue;
        strideRot = Math.max(strideRot, quatAngleDeg(a.quat, b.quat));
      }
    }
    row.determinism = {
      seekMaxRotDeg: captureDelta(cap, seek).maxRotDeg,
      strideMaxRotDeg: +strideRot.toFixed(4),
      loopWrapMaxRotDeg: captureDelta(cap, loopWrap).maxRotDeg,
      historyDependent: captureDelta(cap, seek).maxRotDeg > 1e-4 || strideRot > 1e-4
        || captureDelta(cap, loopWrap).maxRotDeg > 1e-4,
    };
    out[name] = row;
  }
}

// ------------------------------------------------------- aggregate summary
const configs = Object.keys(Object.values(report.clips)[0] ?? {});
const summary = {};
for (const cfg of configs) {
  const rows = Object.values(report.clips).map(c => c[cfg]).filter(Boolean);
  const fmax = (get) => {
    const v = rows.map(get).filter(Number.isFinite);
    return v.length ? +Math.max(...v).toFixed(4) : null;
  };
  const fmean = (get) => {
    const v = rows.map(get).filter(Number.isFinite);
    return v.length ? +(v.reduce((a, b) => a + b, 0) / v.length).toFixed(4) : null;
  };
  summary[cfg] = {
    handRotMean: fmean(r => (r.fidelityRot.LeftHand?.mean + r.fidelityRot.RightHand?.mean) / 2),
    handRotP95: fmax(r => Math.max(r.fidelityRot.LeftHand?.p95 ?? 0, r.fidelityRot.RightHand?.p95 ?? 0)),
    footRotMean: fmean(r => (r.fidelityRot.LeftFoot?.mean + r.fidelityRot.RightFoot?.mean) / 2),
    roundTripMean: fmean(r => r.roundTrip?.mean),
    velMax: fmax(r => r.velMax), accMax: fmax(r => r.accMax),
    flips: rows.reduce((a, r) => a + r.flips, 0),
    extensionMax: fmax(r => Math.max(...Object.values(r.extension).map(s => s?.max ?? 0))),
    clearanceMin: +Math.min(...rows.map(r => r.clearance.capsuleClearanceMin ?? Infinity)).toFixed(3),
    groundPenMax: fmax(r => r.clearance.groundPenMax),
    skateMeanMax: fmax(r => Math.max(r.skate.Left?.mean ?? 0, r.skate.Right?.mean ?? 0)),
    historyDependent: rows.some(r => r.determinism.historyDependent),
    deltaRotMax: fmax(r => r.deltaVsBaseline?.maxRotDeg),
    constraintPosMax: fmax(r => Math.max(...Object.values(r.constraintAccuracy ?? {}).map(s => s?.pos?.max ?? 0))),
    constraintRotMax: fmax(r => Math.max(...Object.values(r.constraintAccuracy ?? {}).map(s => s?.rot?.max ?? 0))),
  };
}
report.summary = summary;

const pad = (s, n) => String(s ?? '-').padEnd(n);
console.log(`\nablation — ${path.basename(charPath)} vs ${movesDir} (${moves.length} clips)`);
console.log(pad('config', 17) + pad('handRot mean/p95', 18) + pad('rt mean', 9) +
  pad('velMax', 8) + pad('flips', 6) + pad('ext', 7) + pad('clear', 7) +
  pad('skate', 7) + pad('histDep', 8) + pad('Δrot', 8) + 'consPos/Rot');
for (const [cfg, s] of Object.entries(summary)) {
  console.log(pad(cfg, 17) +
    pad(`${s.handRotMean}/${s.handRotP95}`, 18) +
    pad(s.roundTripMean, 9) + pad(s.velMax, 8) + pad(s.flips, 6) +
    pad(s.extensionMax, 7) + pad(s.clearanceMin, 7) + pad(s.skateMeanMax, 7) +
    pad(s.historyDependent ? 'YES' : 'no', 8) + pad(s.deltaRotMax ?? '-', 8) +
    `${s.constraintPosMax ?? '-'}/${s.constraintRotMax ?? '-'}`);
}
if (jsonOut) {
  fs.mkdirSync(path.dirname(path.resolve(jsonOut)), { recursive: true });
  fs.writeFileSync(jsonOut, JSON.stringify(report, null, 1));
  console.log(`\nevidence -> ${jsonOut}`);
}
