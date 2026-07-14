#!/usr/bin/env node
// Constraint QA: end-effector accuracy from AUTHORED CONSTRAINT to FINAL
// CHARACTER, measured stage by stage so a later stage can never hide an
// earlier failure:
//
//   stage 1  authored target -> final SOMA output   (from the baked clip's
//            resolved records vs its own pos/quat channels; root2d adherence
//            comes from the generation report through the manifest)
//   stage 2  SOMA output -> UNGUARDED target rig    (corrected anchors +
//            world-axis mapping, all guards/damping/smoothing off; rotation
//            error against the FULL source demand — never scaled by
//            handFollow — plus round-trip position recovery)
//   stage 3  unguarded -> shipped                   (the delta each RETAINED
//            style modifier introduces — after the guard cleanup that is
//            handFollow only; deleted guards: evidence/README.md)
//   stage 4  final character -> mapped authored target (with constraint IK;
//            gates: pos p95<=0.02m max<=0.04m, rot p95<=5deg max<=10deg)
//
// plus motion-validity gates on the shipped result: no NaN, no invalid
// quaternion, no one-frame orientation branch flip, no new foot skate on
// (predicted) contact frames, and sequential==seek determinism.
//
// Usage: node qa_constraints.mjs <char.glb> <movesDir> [--gate] [--json out.json] [--no-ik]
// Exit: 0 ok, 1 gate failure (with --gate), 2 usage/data error.
import fs from 'node:fs';
import path from 'node:path';
import * as THREE from 'three';
import { loadGLBBones } from './glbskel.mjs';
import { rigFromBones, resetBindPose, roundTripError } from './align.js';
import { Retargeter, baselineOptions } from './retarget.js';
import {
  captureRun, contactMask, footSkate, angularKinematics, poseValidity,
  captureDelta, sourceFidelity, constraintAccuracy, stats, quatAngleDeg, EE_ROLES,
} from './qametrics.mjs';

const args = process.argv.slice(2);
const positional = args.filter(a => !a.startsWith('--'));
const GATE = args.includes('--gate');
const NO_IK = args.includes('--no-ik');
const jsonIdx = args.indexOf('--json');
const jsonOut = jsonIdx >= 0 ? args[jsonIdx + 1] : null;
const [charPath, movesDir] = positional;
if (!charPath || !movesDir || (jsonIdx >= 0 && (!jsonOut || jsonOut.startsWith('--')))) {
  console.error('usage: node qa_constraints.mjs <char.glb> <movesDir> [--gate] [--json out.json] [--no-ik]');
  process.exit(2);
}

// ---- accuracy contract (task gates)
const G = {
  somaPosMax: 0.005, somaRotMaxDeg: 2.0,          // stage 1, per authored EE frame
  rootXZMax: 0.02,                                 // stage 1, root waypoints (from gen report)
  charPosP95: 0.02, charPosMax: 0.04,              // stage 4 position
  charRotP95: 5.0, charRotMax: 10.0,               // stage 4 rotation
  skateTolerance: 0.03,                            // m/s of NEW skate on contact frames
  seekTolDeg: 1e-4,                                // sequential vs seek determinism
};

const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);
const q4 = (a) => new THREE.Quaternion(a[0], a[1], a[2], a[3]).normalize();
const summarizeRows = (rows, keyOf) => {
  const groups = {};
  for (const row of rows) {
    const key = keyOf(row);
    if (!key) continue;
    groups[key] ??= { pos: [], rot: [], rootXZ: [] };
    if (Number.isFinite(row.posErr)) groups[key].pos.push(row.posErr);
    if (Number.isFinite(row.rotErrDeg)) groups[key].rot.push(row.rotErrDeg);
    if (Number.isFinite(row.rootXZErr)) groups[key].rootXZ.push(row.rootXZErr);
  }
  return Object.fromEntries(Object.entries(groups).map(([key, values]) => [key, {
    pos: stats(values.pos), rot: stats(values.rot), rootXZ: stats(values.rootXZ),
  }]));
};

const { bones: loadedBones } = await loadGLBBones(charPath);
const target = rigFromBones(loadedBones);
const manifest = JSON.parse(fs.readFileSync(path.join(movesDir, 'manifest.json'), 'utf8'));
if (!Array.isArray(manifest.moves) || !manifest.moves.length)
  throw new Error('manifest must contain a non-empty moves array');

let failures = 0;
const report = { character: path.resolve(charPath), movesDir: path.resolve(movesDir),
  gates: G, clips: {}, failures: [] };
const fail = (clip, msg) => { failures++; report.failures.push(`${clip}: ${msg}`); };

for (const mv of manifest.moves) {
  const clip = JSON.parse(fs.readFileSync(path.join(movesDir, mv.file), 'utf8'));
  const idx = Object.create(null);
  clip.names.forEach((n, i) => { idx[n] = i; });
  const records = clip.constraints ?? [];
  const eeRecords = records.filter(r => r.family === 'end-effector');
  const out = { constrained: records.length > 0 };
  report.clips[mv.name] = out;

  try {
    // ------------------------------------------------ stage 1: SOMA output
    if (records.length) {
      const s1 = [];
      for (const rec of records) {
        if (!Number.isInteger(rec.frame) || rec.frame < 0 || rec.frame >= clip.pos.length)
          throw new Error(`constraint ${rec.type} has out-of-range frame ${rec.frame}`);
        if (Array.isArray(rec.rootXZ)) {
          const hips = idx.Hips;
          if (hips === undefined) throw new Error('clip lacks Hips for root constraint measurement');
          const got = clip.pos[rec.frame][hips];
          const rootXZErr = Math.hypot(got[0] - rec.rootXZ[0], got[2] - rec.rootXZ[1]);
          s1.push({ type: rec.type, family: rec.family, source: rec.source,
            frame: rec.frame, role: 'Hips', rootXZErr: +rootXZErr.toFixed(5) });
          // The hard 2 cm root contract applies to authored root waypoints /
          // dense paths. Pose/EE records preserve their implicit root pins as
          // provenance and report the delta, but their accuracy contract is
          // the authored pose or end-effector target.
          if (rec.family === 'root2d' && rootXZErr > G.rootXZMax)
            fail(mv.name, `stage1 ${rec.type}@${rec.frame} root XZ err ${rootXZErr.toFixed(4)}m > ${G.rootXZMax}`);
        }
        if (rec.family === 'end-effector') {
          const j = idx[rec.role];
          if (j === undefined) throw new Error(`clip lacks source joint for ${rec.role}`);
          const posErr = v3(clip.pos[rec.frame][j]).distanceTo(v3(rec.pos));
          const rotErr = quatAngleDeg(q4(clip.quat[rec.frame][j]), q4(rec.quat));
          s1.push({ type: rec.type, family: rec.family, source: rec.source,
            frame: rec.frame, role: rec.role,
            posErr: +posErr.toFixed(5), rotErrDeg: +rotErr.toFixed(3) });
          if (posErr > G.somaPosMax)
            fail(mv.name, `stage1 ${rec.role}@${rec.frame} SOMA pos err ${posErr.toFixed(4)}m > ${G.somaPosMax}`);
          if (rotErr > G.somaRotMaxDeg)
            fail(mv.name, `stage1 ${rec.role}@${rec.frame} SOMA rot err ${rotErr.toFixed(2)}° > ${G.somaRotMaxDeg}`);
        } else if (rec.family === 'fullbody' && rec.ee) {
          for (const [name, t] of Object.entries(rec.ee)) {
            const j = idx[name];
            if (j === undefined) throw new Error(`clip lacks source joint for fullbody role ${name}`);
            const posErr = v3(clip.pos[rec.frame][j]).distanceTo(v3(t.pos));
            const rotErr = quatAngleDeg(q4(clip.quat[rec.frame][j]), q4(t.quat));
            s1.push({ type: 'fullbody', family: rec.family, source: rec.source,
              frame: rec.frame, role: name,
              posErr: +posErr.toFixed(5), rotErrDeg: +rotErr.toFixed(3) });
            if (posErr > G.somaPosMax)
              fail(mv.name, `stage1 fullbody ${name}@${rec.frame} SOMA pos err ${posErr.toFixed(4)}m > ${G.somaPosMax}`);
            if (rotErr > G.somaRotMaxDeg)
              fail(mv.name, `stage1 fullbody ${name}@${rec.frame} SOMA rot err ${rotErr.toFixed(2)}° > ${G.somaRotMaxDeg}`);
          }
        }
      }
      const adherence = mv.constraints?.adherence;
      if (!adherence) throw new Error('manifest lacks generation-time constraint adherence metrics');
      if (adherence.root_xz_max !== undefined && adherence.root_xz_max > G.rootXZMax)
        fail(mv.name, `stage1 root waypoint err ${adherence.root_xz_max}m > ${G.rootXZMax}`);
      out.stage1 = {
        records: s1,
        summaryByRole: summarizeRows(s1, row => row.role),
        summaryByType: summarizeRows(s1, row => row.type),
        summaryByRoleType: summarizeRows(s1, row => `${row.role}:${row.type}`),
        generationAdherence: adherence,
      };
    }

    // -------------------------------------- stage 2: unguarded transfer rig
    const baseline = captureRun(target, clip, { config: baselineOptions(clip) });
    out.baselineConfig = baseline.config;
    out.stage2 = { fidelity: sourceFidelity(baseline) };
    {
      resetBindPose(target);
      const rt = new Retargeter({ ...target, data: clip, ...baselineOptions(clip) });
      const rtErrs = [], rtByRole = {};
      for (let f = 0; f < baseline.N; f++) {
        const recovered = roundTripError(rt, f);
        rtErrs.push(recovered.mean);
        for (const [role, error] of Object.entries(recovered.perRole)) {
          rtByRole[role] ??= [];
          rtByRole[role].push(error);
        }
      }
      out.stage2.roundTripMean = stats(rtErrs);
      out.stage2.roundTripPositionByRole = Object.fromEntries(
        Object.entries(rtByRole).map(([role, values]) => [role, stats(values)]));
      resetBindPose(target);
    }
    for (const role of EE_ROLES) {
      const s = out.stage2.fidelity[role];
      if (!s?.rot) throw new Error(`stage2 fidelity for ${role} could not be measured`);
    }

    // -------------- stage 3: deltas of each RETAINED modifier vs baseline
    // (the deleted guards' recorded deltas: evidence/README.md)
    const interventions = {
      handFollow: { config: { ...baselineOptions(clip), handFollow: clip.handFollow ?? 0.3 } },
    };
    out.stage3 = {};
    for (const [name, spec] of Object.entries(interventions)) {
      const cap = captureRun(target, clip, { config: spec.config });
      const delta = captureDelta(baseline, cap);
      out.stage3[name] = { maxRotDeg: delta.maxRotDeg, maxPosM: delta.maxPosM };
      // a guard that moves an authored constrained frame is a reported
      // conflict, never a silent adjustment
      for (const rec of eeRecords) {
        const a = baseline.rows[rec.frame].roles[rec.role];
        const b = cap.rows[rec.frame].roles[rec.role];
        const dRot = quatAngleDeg(a.quat, b.quat), dPos = a.pos.distanceTo(b.pos);
        if (dRot > 0.5 || dPos > 0.002) {
          out.stage3[name].constrainedFrameDelta ??= [];
          out.stage3[name].constrainedFrameDelta.push({
            role: rec.role, frame: rec.frame, dRotDeg: +dRot.toFixed(2), dPosM: +dPos.toFixed(4) });
        }
      }
    }

    // ------------------------------------------- stage 4: shipped character
    const useIK = !NO_IK && eeRecords.length > 0;
    const shipped = captureRun(target, clip, { config: {}, ik: useIK });
    out.shippedConfig = shipped.config;
    out.ik = useIK;
    if (eeRecords.length) {
      const acc = constraintAccuracy(shipped);
      out.stage4 = acc;
      const reachPolicy = mv.reachPolicy ?? 'fail';
      for (const [role, s] of Object.entries(acc.summary)) {
        if (s.unreachable && reachPolicy !== 'clamp')
          fail(mv.name, `stage4 ${role}: ${s.unreachable} authored target(s) unreachable on this rig`);
        if (s.pos && (s.pos.p95 > G.charPosP95 || s.pos.max > G.charPosMax))
          fail(mv.name, `stage4 ${role} pos p95=${s.pos.p95} max=${s.pos.max} exceeds ${G.charPosP95}/${G.charPosMax}m`);
        if (s.rot && (s.rot.p95 > G.charRotP95 || s.rot.max > G.charRotMax))
          fail(mv.name, `stage4 ${role} rot p95=${s.rot.p95} max=${s.rot.max} exceeds ${G.charRotP95}/${G.charRotMax}°`);
      }
    } else if (records.length) {
      out.stage4 = { applicable: false,
        reason: 'no authored end-effector records (fullbody/root targets are measured at stage 1)' };
    }

    // ------------------------------------------------ motion validity gates
    const validity = poseValidity(shipped);
    const kinematics = angularKinematics(shipped);
    out.validity = validity;
    out.kinematics = Object.fromEntries(Object.entries(kinematics)
      .map(([r, k]) => [r, { velMax: k.vel?.max, accMax: k.acc?.max, flips: k.flips }]));
    if (!validity.ok) fail(mv.name, `invalid pose data: ${validity.nan} NaN, ${validity.denorm} denormalized`);
    for (const [role, k] of Object.entries(kinematics)) {
      if (k.flips > 0) fail(mv.name, `${k.flips} orientation branch flip(s) on ${role}`);
    }
    const mask = contactMask(clip);
    out.contactSource = mask.source;
    const skBase = footSkate(baseline, mask), skShip = footSkate(shipped, mask);
    out.skate = { baseline: skBase, shipped: skShip };
    for (const side of ['Left', 'Right']) {
      if (skBase[side] && skShip[side] &&
          skShip[side].mean > skBase[side].mean + G.skateTolerance)
        fail(mv.name, `new foot skate on ${side} contact frames: ` +
          `${skShip[side].mean} vs baseline ${skBase[side].mean} m/s`);
    }

    // --------------------------------------------- determinism (seek==seq)
    const seek = captureRun(target, clip, { config: {}, ik: useIK, order: 'seek' });
    const det = captureDelta(shipped, seek);
    out.determinism = { maxRotDeg: det.maxRotDeg, maxPosM: det.maxPosM };
    if (det.maxRotDeg > G.seekTolDeg)
      fail(mv.name, `constrained pose depends on frame order: seek delta ${det.maxRotDeg}° ` +
        '(sequential playback vs direct seek must be identical)');
  } catch (e) {
    fail(mv.name, `metric could not be computed: ${e.message}`);
    continue;
  }
}

// Character-level aggregates across clips. Per-clip records above remain the
// source of detail; these buckets make role/type comparisons available
// directly without asking CI consumers to reconstruct them. Explicitly
// unreachable stage-4 targets stay in clip diagnostics but are excluded from
// accuracy statistics, as required by the authored-target contract.
const allStage1 = [], allStage4 = [];
for (const [clip, result] of Object.entries(report.clips)) {
  for (const row of result.stage1?.records ?? []) allStage1.push({ ...row, clip });
  for (const row of result.stage4?.details ?? []) {
    if (row.reachable !== false) allStage4.push({ ...row, clip });
  }
}
report.summary = {
  authoredToSoma: {
    byRole: summarizeRows(allStage1, row => row.role),
    byType: summarizeRows(allStage1, row => row.type),
    byRoleType: summarizeRows(allStage1, row => `${row.role}:${row.type}`),
  },
  finalCharacterToAuthored: {
    byRole: summarizeRows(allStage4, row => row.role),
    byType: summarizeRows(allStage4, row => row.type),
    byRoleType: summarizeRows(allStage4, row => `${row.role}:${row.type}`),
  },
};

// ------------------------------------------------------------- human table
console.log(`\nconstraint QA — ${path.basename(charPath)} vs ${movesDir}`);
console.log('stage gates: SOMA pos<=5mm rot<=2°; char pos p95<=20mm max<=40mm, rot p95<=5° max<=10°\n');
const pad = (s, n) => String(s ?? '-').padEnd(n);
const clipWidth = Math.max(15, ...Object.keys(report.clips).map(name => name.length + 2));
console.log(pad('clip', clipWidth) + pad('cons', 5) + pad('unreach', 8) + pad('s1 posMax', 11) + pad('s4 posP95', 11) +
  pad('s4 posMax', 11) + pad('s4 rotP95', 11) + pad('flips', 7) + pad('seekΔ°', 9) + 'skate Δ');
for (const [name, c] of Object.entries(report.clips)) {
  const s1 = c.stage1?.records?.filter(r => r.posErr !== undefined) ?? [];
  const s1max = s1.length ? Math.max(...s1.map(r => r.posErr)) : null;
  const s4 = c.stage4?.summary ? Object.values(c.stage4.summary) : [];
  const s4PosP95 = s4.map(s => s.pos?.p95).filter(Number.isFinite);
  const s4PosMax = s4.map(s => s.pos?.max).filter(Number.isFinite);
  const s4RotP95 = s4.map(s => s.rot?.p95).filter(Number.isFinite);
  const s4p95 = s4PosP95.length ? Math.max(...s4PosP95) : null;
  const s4max = s4PosMax.length ? Math.max(...s4PosMax) : null;
  const s4rot = s4RotP95.length ? Math.max(...s4RotP95) : null;
  const unreachable = s4.reduce((n, s) => n + (s.unreachable ?? 0), 0);
  const flips = c.kinematics ? Object.values(c.kinematics).reduce((a, k) => a + (k.flips ?? 0), 0) : '-';
  const skD = c.skate?.shipped?.Left && c.skate?.baseline?.Left
    ? Math.max(c.skate.shipped.Left.mean - c.skate.baseline.Left.mean,
        (c.skate.shipped.Right?.mean ?? 0) - (c.skate.baseline.Right?.mean ?? 0)).toFixed(3)
    : '-';
  console.log(pad(name, clipWidth) + pad(c.constrained ? 'yes' : '-', 5) +
    pad(unreachable || '-', 8) +
    pad(s1max === null ? '-' : s1max.toFixed(4), 11) +
    pad(s4p95 === null ? '-' : s4p95.toFixed(4), 11) +
    pad(s4max === null ? '-' : s4max.toFixed(4), 11) +
    pad(s4rot === null ? '-' : s4rot.toFixed(2), 11) +
    pad(flips, 7) + pad(c.determinism?.maxRotDeg ?? '-', 9) + skD);
}
console.log('\nstage-3 guard deltas (max over clips, deg / m):');
const agg = {};
for (const c of Object.values(report.clips)) {
  for (const [g, d] of Object.entries(c.stage3 ?? {})) {
    agg[g] ??= { rot: 0, pos: 0, conflicts: 0 };
    agg[g].rot = Math.max(agg[g].rot, d.maxRotDeg);
    agg[g].pos = Math.max(agg[g].pos, d.maxPosM);
    agg[g].conflicts += d.constrainedFrameDelta?.length ?? 0;
  }
}
for (const [g, d] of Object.entries(agg))
  console.log(`  ${pad(g, 17)} rotΔ ${String(d.rot).padStart(8)}°  posΔ ${String(d.pos).padStart(8)}m` +
    (d.conflicts ? `  ⚠ modifies ${d.conflicts} authored constrained frame(s)` : ''));

if (report.failures.length) {
  console.log(`\n${report.failures.length} FAILING gates:`);
  for (const f of report.failures) console.log('  FAIL ' + f);
} else {
  console.log('\nALL CONSTRAINT QA GATES PASS');
}
if (jsonOut) {
  fs.mkdirSync(path.dirname(path.resolve(jsonOut)), { recursive: true });
  fs.writeFileSync(jsonOut, JSON.stringify(report, null, 1));
  console.log(`report -> ${jsonOut}`);
}
if (GATE && failures) process.exit(1);
