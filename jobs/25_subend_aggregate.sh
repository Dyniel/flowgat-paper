#!/bin/bash
#SBATCH --job-name=paper_subend_agg
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 25_subend_aggregate.sh — Phase E7 step 3: unified direction-identifiability
# metrics for the sUbend predictions.
#
# Reuses src/cosserat_sweep_diagnostic.py with --variants overridden to the
# sUbend variant names so we keep ONE implementation across Cosserat / sUbend
# / Womersley (all funneled through src/womersley_metrics.compute_case).
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
mkdir -p "$OUT"

echo "==========================================================================="
echo " SUBEND AGGREGATE | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/cosserat_sweep_diagnostic.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$OUT" \
  --variants subend_withleak subend_leak_dir_only subend_leak_mag_only subend_noleak

echo "[25] DONE"
ls -la "$OUT"
echo
echo "----- subend summary.md -----"
cat "$OUT/summary.md" 2>/dev/null || true
