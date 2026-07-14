#!/usr/bin/env bash
# GPU end-to-end constraint suite: generate every authoring mode
# deterministically, bake, run the stage-separated constraint QA on two
# certified rigs, and assert the expected pass/fail matrix (including the
# EXPECTED failures: e2e_unreachable at generation, e2e_hand_rot's strict
# reach targets on the short-armed rig).
#
# Reproducibility: model Kimodo-SOMA-RP-v1.1, seed 42, 8 samples,
# 100 denoising steps, root margin 0.01 m. Reports land in ../evidence/.
#
# Usage: PY=<kimenv python> FIGHTER=<fighter.glb> CHAR2=<char2.glb> ./run_e2e.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-$HOME/Downloads/ani_test/kimenv/bin/python}
FIGHTER=${FIGHTER:-$HOME/Downloads/ani_test/web/fighter.glb}
CHAR2=${CHAR2:-$HOME/Downloads/ani_test/char2/Meshy_AI_Harbinger_of_the_With_biped_Character_output.glb}

$PY make_e2e_spec.py
# This driver must never certify stale generation output. Clear only its own
# e2e namespace before loading the model; a setup/model failure then leaves no
# artifacts for the bake step to mistake for a fresh run.
mkdir -p out/moves
find out/moves -maxdepth 1 -type f -name 'e2e_*' -delete
rm -rf out/web_e2e

# e2e_unreachable is EXPECTED to be rejected, so a complete generation run
# exits exactly 1. The checker below proves that it was the only rejection.
set +e
$PY kimogen.py gen --spec moveset_e2e.json --samples 8 --seed 42
gen_status=$?
set -e
if [[ $gen_status -ne 1 ]]; then
  echo "[reject] expected generation exit 1 (the deliberate unreachable move); got $gen_status" >&2
  exit 1
fi
$PY bake_kimodo.py --spec moveset_e2e.json --web out/web_e2e --allow-missing

cd ..
node certify.mjs "$FIGHTER" --clips kimodo/out/web_e2e/e2e_text.json,kimodo/out/web_e2e/e2e_waypoints.json --out evidence/cert_e2e_fighter.json
node certify.mjs "$CHAR2"   --clips kimodo/out/web_e2e/e2e_text.json,kimodo/out/web_e2e/e2e_waypoints.json --out evidence/cert_e2e_char2.json
node qa_constraints.mjs "$FIGHTER" kimodo/out/web_e2e --json evidence/qa_e2e_fighter.json
node qa_constraints.mjs "$CHAR2"   kimodo/out/web_e2e --json evidence/qa_e2e_char2.json
node qa_endeffectors.mjs "$FIGHTER" kimodo/out/web_e2e --gate
node qa_endeffectors.mjs "$CHAR2"   kimodo/out/web_e2e --gate
node ablate.mjs "$FIGHTER" kimodo/out/web_e2e --ik --json evidence/ablation_fighter_e2e.json
node ablate.mjs "$CHAR2"   kimodo/out/web_e2e --ik --json evidence/ablation_char2_e2e.json

node kimodo/e2e_check.mjs
