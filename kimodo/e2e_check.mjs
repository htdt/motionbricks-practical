#!/usr/bin/env node
// Assert the e2e suite's expected outcome matrix (run by run_e2e.sh from the
// repo root). Expected results are part of the suite: an expected FAILURE
// that silently passes is as much a defect as an unexpected one.
import fs from 'node:fs';
import path from 'node:path';

const HERE = path.dirname(new URL(import.meta.url).pathname);
const ROOT = path.join(HERE, '..');
let bad = 0;
const check = (label, ok, detail = '') => {
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${label}${!ok && detail ? ' — ' + detail : ''}`);
  if (!ok) bad++;
};

// ---- generation: every move accepted EXCEPT e2e_unreachable
const spec = JSON.parse(fs.readFileSync(path.join(HERE, 'moveset_e2e.json'), 'utf8'));
for (const mv of spec.moves) {
  const rep = JSON.parse(fs.readFileSync(path.join(HERE, 'out/moves', `${mv.name}.json`), 'utf8'));
  if (mv.name === 'e2e_unreachable') {
    check('e2e_unreachable is rejected by the gates (expected failure)', rep.accepted === false);
    check('  ...with its root adherence measured, not skipped',
      rep.gates?.constraints?.root_xz_max !== undefined);
  } else {
    check(`${mv.name} accepted`, rep.accepted === true,
      JSON.stringify(rep.gates?.constraints ?? rep.gates?.malformed ?? ''));
    const cons = rep.gates?.constraints;
    if (cons) check(`  ${mv.name} constraint adherence ok`, cons.constraint_ok === true, JSON.stringify(cons));
  }
}
// constraint-only moves really carried no text
for (const name of ['e2e_constraint_only', 'e2e_constraint_only_pose']) {
  const rep = JSON.parse(fs.readFileSync(path.join(HERE, 'out/moves', `${name}.json`), 'utf8'));
  check(`${name} recorded absent prompt explicitly`, rep.prompt === null);
}

// ---- character QA matrices
const fighter = JSON.parse(fs.readFileSync(path.join(ROOT, 'evidence/qa_e2e_fighter.json'), 'utf8'));
check('fighter: all constraint QA gates pass', fighter.failures.length === 0,
  fighter.failures.slice(0, 3).join('; '));

const char2 = JSON.parse(fs.readFileSync(path.join(ROOT, 'evidence/qa_e2e_char2.json'), 'utf8'));
const unexpected = char2.failures.filter(f => !f.startsWith('e2e_hand_rot:'));
check('char2: only the strict-reach move fails (short arms, documented)',
  unexpected.length === 0 && char2.failures.length > 0, unexpected.slice(0, 3).join('; '));
check('char2: the strict failure is the unreachable diagnostic',
  char2.failures.every(f => !f.startsWith('e2e_hand_rot:') || f.includes('unreachable')));
// clamp-policy moves report their unreachable targets without failing
const mixed = char2.clips.e2e_mixed;
check('char2: clamp-policy move reports unreachable targets without failing',
  mixed?.stage4?.details?.some(d => d.reachable === false));

// ---- certification
for (const rig of ['fighter', 'char2']) {
  const cert = JSON.parse(fs.readFileSync(path.join(ROOT, `evidence/cert_e2e_${rig}.json`), 'utf8'));
  check(`${rig} certified against the e2e clips`, cert.ok === true, cert.failures?.join('; '));
}

console.log(bad ? `\n${bad} E2E EXPECTATION FAILURES` : '\nE2E SUITE MATCHES THE EXPECTED MATRIX');
process.exit(bad ? 1 : 0);
