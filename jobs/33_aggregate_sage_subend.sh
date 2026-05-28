#!/bin/bash
#SBATCH --job-name=paper_sage_subend_agg
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 33_aggregate_sage_subend.sh — unified direction-identifiability metrics
# for SAGE × sUbend predictions (job 30 outputs).
# Mirrors job 25 but pointed at sage_subend_* variant names.
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
OUT="$RESULTS_DIR/diagnostics/sage_subend"
mkdir -p "$OUT"

echo "==========================================================================="
echo " SAGE_SUBEND AGGREGATE | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/cosserat_sweep_diagnostic.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$OUT" \
  --variants sage_subend_withleak sage_subend_leak_dir_only sage_subend_leak_mag_only sage_subend_noleak

echo "[33] DONE"
ls -la "$OUT"
echo
echo "----- sage_subend summary.md -----"
cat "$OUT/summary.md" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Trampoline: launch final stage (job 31 + its agg 34). See note in
# 32_aggregate_sage_cosserat.sh — chained inline to satisfy
# AssocMaxSubmitJobLimit.
# ---------------------------------------------------------------------------
cd "$PAPER_DIR"
JOB31=$(sbatch --parsable jobs/31_train_flowgat_nobc_cosserat.sh)
echo "[33 trampoline] submitted 31_train_flowgat_nobc_cosserat jobid=$JOB31"
JOB34=$(sbatch --parsable --dependency=afterok:$JOB31 jobs/34_aggregate_flowgat_nobc_cosserat.sh)
echo "[33 trampoline] submitted 34_aggregate_flowgat_nobc_cosserat jobid=$JOB34 (afterok:$JOB31)"
