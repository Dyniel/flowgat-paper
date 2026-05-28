#!/bin/bash
# ---------------------------------------------------------------------------
# SUBMIT_E7_FOLLOWUP.sh — Phase E7 post-training analysis.
#
# Trains have already completed (job 24 array done 2026-05-20). Predictions
# are dumped in results/predictions/subend_*/seed_*/. This script kicks off
# three independent CPU jobs that finalise the sUbend analysis:
#
#   job 25 : direction-identifiability diagnostic (re-run with the FIXED
#            cosserat_sweep_diagnostic.py — earlier run produced an empty
#            aggregate.csv because aggregate() used the hardcoded
#            cosserat_sweep_* variant list.)
#   job 26 : ΔP / magnitude-collapse investigation (explains the dP_MAE
#            inversion observed on sUbend).
#   job 27 : paired bootstrap CIs over case_id for the four pairwise
#            contrasts that the SR narrative needs.
#
# All three are independent CPU jobs. Submit in parallel.
#
# Usage:
#   bash jobs/SUBMIT_E7_FOLLOWUP.sh
# ---------------------------------------------------------------------------

set -euo pipefail

JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"

submit() {
    local script="$1"
    local out
    out="$(sbatch "$script")"
    echo "$out" | awk '{print $NF}'
}

J25="$(submit "$JOBS_DIR/25_subend_aggregate.sh")"
echo "[submit] job 25 (subend diag,    fixed)  -> $J25"

J26="$(submit "$JOBS_DIR/26_subend_dp_investigation.sh")"
echo "[submit] job 26 (subend dP inv)          -> $J26"

J27="$(submit "$JOBS_DIR/27_subend_bootstrap.sh")"
echo "[submit] job 27 (subend bootstrap CIs)   -> $J27"

J28="$(submit "$JOBS_DIR/28_clinical_headline_fig.sh")"
echo "[submit] job 28 (clinical headline Fig 1)-> $J28"

echo
echo "[submit] Watch with:"
echo "   squeue -u $USER --format='%.10i %.20j %.10T %.10M %.20R'"
