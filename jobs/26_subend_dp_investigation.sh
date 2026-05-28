#!/bin/bash
#SBATCH --job-name=paper_subend_dp_inv
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=24G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 26_subend_dp_investigation.sh — Phase E7 follow-up: investigate why
# the noleak variant on sUbend has LOWER dP_MAE than withleak (opposite
# of every other domain we have looked at). The script measures per-frame
# magnitude collapse: max||u_pred||, max||u_true||, the Bernoulli dP they
# generate, and the correlation between predicted and true dP across cases.
#
# Pure CPU job. Walks results/predictions/subend_* (1800 NPZs total,
# 24 var×seed×split×case combos).
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
OUT="$RESULTS_DIR/diagnostics/subend"
FIG="$RESULTS_DIR/figures"
mkdir -p "$OUT" "$FIG"

echo "==========================================================================="
echo " SUBEND DP INVESTIGATION | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/subend_dp_investigation.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --out_dir          "$OUT" \
  --figures_dir      "$FIG" \
  --variants subend_withleak subend_leak_dir_only subend_leak_mag_only subend_noleak

echo "[26] DONE"
ls -la "$OUT" | grep dp_invest
echo
echo "----- subend dp_investigation.md -----"
cat "$OUT/dp_investigation.md" 2>/dev/null || true
