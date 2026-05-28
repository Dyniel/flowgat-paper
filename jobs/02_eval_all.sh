#!/bin/bash
#SBATCH --job-name=paper_eval_all
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 02_eval_all.sh — re-evaluate all 6 (variant, seed) checkpoints.
#
# Useful only if you want to redo evals with different metrics / splits
# without retraining. The 01 job already produces per-seed CSVs at training
# time, so this is mostly a re-run safety net for analysis.
#
# Usage:
#   sbatch jobs/02_eval_all.sh
# ---------------------------------------------------------------------------

set -euo pipefail
unalias python 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

module load trytonp/conda/py313_25.9.1-3
module load trytonp/nvidia_hpc_sdk/24.3
eval "$(conda shell.bash hook)"
conda activate /users/scratch1/$USER/conda_envs/py312

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
SRC_DIR="$PAPER_DIR/src"
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"
RESULTS_DIR="$PAPER_DIR/results"
PER_SEED_DIR="$RESULTS_DIR/per_seed"
mkdir -p "$PER_SEED_DIR"

for VARIANT in withleak noleak; do
  CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
  for SEED in 1337 2026 777; do
    BEST="$RESULTS_DIR/checkpoints/${VARIANT}/seed_${SEED}/best.pt"
    if [[ ! -f "$BEST" ]]; then
      echo "[02] SKIP missing $BEST"
      continue
    fi
    for SPLIT in val test; do
      OUT="$PER_SEED_DIR/${VARIANT}_${SPLIT}_seed${SEED}.csv"
      echo "[02] $VARIANT seed=$SEED split=$SPLIT -> $OUT"
      "$PYBIN" "$SRC_DIR/evaluate.py" \
        --config "$CFG" --ckpt "$BEST" --split "$SPLIT" --out "$OUT"
    done
  done
done

echo "[02] DONE."
