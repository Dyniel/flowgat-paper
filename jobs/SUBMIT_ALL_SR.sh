#!/bin/bash
# ---------------------------------------------------------------------------
# SUBMIT_ALL_SR.sh — single entry point for the SR-reframe pipeline.
#
# Submits, in dependency order:
#   job 22 : aggregate cosserat sweep + mesh refinement (CPU,   ~30 min)
#   job 23 : build sUbend NPZs (noleak + 3 variants)   (CPU,    ~30-60 min)
#   job 24 : train 4 variants x 3 seeds on sUbend      (GPU,    ~4-8h x 12 array)
#   job 25 : aggregate sUbend diagnostic               (CPU,    ~10 min)
#
# Each downstream job is scheduled with --dependency=afterok:<previous>.
# Job 22 and the rest run independently of one another; if you only want
# the sUbend pipeline (skip 22), call:   bash jobs/SUBMIT_ALL_SR.sh --no-agg
#
# Usage:
#   bash jobs/SUBMIT_ALL_SR.sh            # everything
#   bash jobs/SUBMIT_ALL_SR.sh --no-agg   # skip cosserat+meshref aggregation
#
# Prerequisite: extraction of newCFD_dataset.zip must already be complete.
# Run on the login node (issues sbatch, does not block):
#   /users/scratch1/dancies/conda_envs/py312/bin/python data/external/extract_subend.py \
#     --outer data/external/newCFD_dataset.zip \
#     --dest  data/external/newCFD_dataset --procs 8
# ---------------------------------------------------------------------------

set -euo pipefail

JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="$(cd "$JOBS_DIR/.." && pwd)"
cd "$PAPER_DIR"

SKIP_AGG=0
if [[ "${1:-}" == "--no-agg" ]]; then
    SKIP_AGG=1
fi

submit() {
    local script="$1"; shift
    local sbatch_args=("$@")
    local out
    out="$(sbatch "${sbatch_args[@]}" "$script")"
    echo "$out" | awk '{print $NF}'
}

declare -a DEP_FLAGS=()

if (( SKIP_AGG == 0 )); then
    J22="$(submit "$JOBS_DIR/22_aggregate_cosserat_meshref.sh")"
    echo "[submit] job 22 (cosserat+meshref agg) -> $J22"
fi

# sUbend extraction is assumed done on login node before submitting.
if ! ls "$PAPER_DIR/data/external/newCFD_dataset"/sUbend_*/CFD/Frame_00.vtk >/dev/null 2>&1; then
    echo "[submit] WARN: extracted sUbend frames not found under data/external/newCFD_dataset/"
    echo "[submit] WARN: job 23 will fail unless extraction completes before it starts."
fi

J23="$(submit "$JOBS_DIR/23_subend_build_npz.sh")"
echo "[submit] job 23 (sUbend NPZ build)        -> $J23"

J24="$(submit "$JOBS_DIR/24_train_subend.sh" "--dependency=afterok:$J23")"
echo "[submit] job 24 (sUbend training array)   -> $J24  (after $J23)"

J25="$(submit "$JOBS_DIR/25_subend_aggregate.sh" "--dependency=afterok:$J24")"
echo "[submit] job 25 (sUbend aggregate)        -> $J25  (after $J24)"

echo
echo "[submit] Pipeline submitted. Watch with:"
echo "   squeue -u $USER --format='%.10i %.20j %.10T %.10M %.20R'"
echo
if (( SKIP_AGG == 0 )); then
    echo "Independent jobs:"
    echo "   22: $J22"
fi
echo "Dependency chain (sUbend):"
echo "   23 (build)  -> $J23"
echo "   24 (train)  -> $J24"
echo "   25 (agg)    -> $J25"
