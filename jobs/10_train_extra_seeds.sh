#!/bin/bash
#SBATCH --job-name=paper_train_extra
#SBATCH --time=14:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 10_train_extra_seeds.sh — Phase D / S1 — add 2 extra seeds to all 4 variants
# to widen bootstrap CIs on headline metrics (3 -> 5 seeds per variant).
#
# Array index -> (variant, seed):
#   0: withleak       42       4: withleak       3407
#   1: leak_dir_only  42       5: leak_dir_only  3407
#   2: leak_mag_only  42       6: leak_mag_only  3407
#   3: noleak         42       7: noleak         3407
#
# Walltime cap 14h: noleak historically up to ~10h, others 5-6h.
#
# After all 8 finish:
#   sbatch jobs/08_aggregate_decomp.sh     # re-aggregate with 5 seeds
#   sbatch -d afterok:<agg_jobid> jobs/09_figures_final.sh   # refresh Fig 2/3/5/6
#
# Submit:
#   sbatch jobs/10_train_extra_seeds.sh
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

PAIRS=(
  "withleak 42"
  "leak_dir_only 42"
  "leak_mag_only 42"
  "noleak 42"
  "withleak 3407"
  "leak_dir_only 3407"
  "leak_mag_only 3407"
  "noleak 3407"
)
PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="$(echo "$PAIR" | awk '{print $1}')"
SEED="$(echo "$PAIR" | awk '{print $2}')"

CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
RESULTS_DIR="$PAPER_DIR/results"
PER_SEED_DIR="$RESULTS_DIR/per_seed"
CKPT_DIR="$RESULTS_DIR/checkpoints/${VARIANT}/seed_${SEED}"
mkdir -p "$PER_SEED_DIR" "$CKPT_DIR"

echo "==========================================================================="
echo " TRAIN-EXTRA-SEED | variant=${VARIANT} seed=${SEED} | $(date) | host=$(hostname)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# ---- Train ----------------------------------------------------------------
"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

# ---- Evaluate val + test --------------------------------------------------
BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[10] ERROR: missing $BEST_CKPT" >&2
  exit 1
fi

for SPLIT in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SPLIT}_seed${SEED}.csv"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" --ckpt "$BEST_CKPT" --split "$SPLIT" --out "$OUT_CSV"
done

echo "[10] DONE | variant=${VARIANT} seed=${SEED}"
