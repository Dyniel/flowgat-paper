#!/bin/bash
#SBATCH --job-name=paper_flowgat_nobc_cosserat_agg
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 34_aggregate_flowgat_nobc_cosserat.sh — unified direction-identifiability
# metrics for FlowGAT-no-BC × Cosserat predictions (job 31 outputs).
# Mirrors job 25; pointed at cosserat_sweep_*_nobc variant names.
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
OUT="$RESULTS_DIR/diagnostics/flowgat_nobc_cosserat"
mkdir -p "$OUT"

echo "==========================================================================="
echo " FLOWGAT_NOBC_COSSERAT AGGREGATE | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/cosserat_sweep_diagnostic.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$OUT" \
  --variants \
    cosserat_sweep_withleak_nobc \
    cosserat_sweep_leak_dir_only_nobc \
    cosserat_sweep_leak_mag_only_nobc \
    cosserat_sweep_noleak_nobc

echo "[34] DONE"
ls -la "$OUT"
echo
echo "----- flowgat_nobc_cosserat summary.md -----"
cat "$OUT/summary.md" 2>/dev/null || true
