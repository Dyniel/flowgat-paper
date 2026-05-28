#!/bin/bash
#SBATCH --job-name=paper_diag
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 05_diagnostics.sh — Phase A2/A3/A4. Runs three CPU analyses:
#   1. peak_loc_diagnostic   (degeneracy check on dumped predictions)
#   2. stratified_analyze    (per-pathology mean ± std)
#   3. bootstrap_ci          (paired bootstrap CIs, withleak vs noleak)
#
# Submit AFTER 04_dump_predictions.sh:
#   sbatch -d afterok:<dump_jobid> jobs/05_diagnostics.sh
# ---------------------------------------------------------------------------

set -euo pipefail
unalias python 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

module load trytonp/conda/py313_25.9.1-3
eval "$(conda shell.bash hook)"
conda activate /users/scratch1/$USER/conda_envs/py312

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
SRC_DIR="$PAPER_DIR/src"
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"
RESULTS_DIR="$PAPER_DIR/results"

mkdir -p "$RESULTS_DIR/diagnostics" "$RESULTS_DIR/stratified" "$RESULTS_DIR/bootstrap"

echo "==========================================================================="
echo " DIAGNOSTICS | $(date) | host=$(hostname)"
echo "==========================================================================="

# A2: degeneracy check
echo "[05] (1/3) peak_loc_diagnostic"
"$PYBIN" "$SRC_DIR/peak_loc_diagnostic.py" \
  --predictions_dir "$RESULTS_DIR/predictions" \
  --out_dir         "$RESULTS_DIR/diagnostics" \
  --fig_dir         "$RESULTS_DIR/figures"

# A3: per-pathology stratification (test only — val has 4 cases, no FSI split)
echo "[05] (2/3) stratified_analyze (test)"
"$PYBIN" "$SRC_DIR/stratified_analyze.py" \
  --per_seed_dir "$RESULTS_DIR/per_seed" \
  --out_dir      "$RESULTS_DIR/stratified" \
  --split        test

# A4: paired bootstrap CI on test
echo "[05] (3/3) bootstrap_ci (test)"
"$PYBIN" "$SRC_DIR/bootstrap_ci.py" \
  --per_seed_dir "$RESULTS_DIR/per_seed" \
  --out_dir      "$RESULTS_DIR/bootstrap" \
  --split        test \
  --n_boot       10000 \
  --n_perm       10000 \
  --seed         42

echo "[05] DONE"
echo "Artifacts:"
echo "  $RESULTS_DIR/diagnostics/peak_loc_diagnostic.{csv,md}"
echo "  $RESULTS_DIR/diagnostics/peak_loc_dispersion.csv"
echo "  $RESULTS_DIR/figures/diag_peak_loc_normalized.{png,pdf}"
echo "  $RESULTS_DIR/stratified/stratified_test.{csv,md}"
echo "  $RESULTS_DIR/bootstrap/bootstrap_test.{csv,md}"
