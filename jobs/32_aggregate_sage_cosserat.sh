#!/bin/bash
#SBATCH --job-name=paper_sage_cosserat_agg
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 32_aggregate_sage_cosserat.sh — unified direction-identifiability metrics
# for SAGE × Cosserat predictions (job 29 outputs).
# Mirrors job 25 but pointed at sage_cosserat_* variant names.
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
OUT="$RESULTS_DIR/diagnostics/sage_cosserat"
mkdir -p "$OUT"

echo "==========================================================================="
echo " SAGE_COSSERAT AGGREGATE | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/cosserat_sweep_diagnostic.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$OUT" \
  --variants sage_cosserat_withleak sage_cosserat_leak_dir_only sage_cosserat_leak_mag_only sage_cosserat_noleak

echo "[32] DONE"
ls -la "$OUT"
echo
echo "----- sage_cosserat summary.md -----"
cat "$OUT/summary.md" 2>/dev/null || true

# ---------------------------------------------------------------------------
# Trampoline: launch next stage (job 30 + its agg 33). We do this from inside
# the agg job rather than at top-level because the cluster's
# AssocMaxSubmitJobLimit counts each `--array` task individually; submitting
# all three (29+30+31) at once exceeds the cap. By chaining inline we keep
# at most ~13 jobs (one 12-task array + one agg) in the queue at a time.
# ---------------------------------------------------------------------------
cd "$PAPER_DIR"
JOB30=$(sbatch --parsable jobs/30_train_sage_subend.sh)
echo "[32 trampoline] submitted 30_train_sage_subend jobid=$JOB30"
JOB33=$(sbatch --parsable --dependency=afterok:$JOB30 jobs/33_aggregate_sage_subend.sh)
echo "[32 trampoline] submitted 33_aggregate_sage_subend jobid=$JOB33 (afterok:$JOB30)"
