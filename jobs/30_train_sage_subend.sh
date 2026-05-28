#!/bin/bash
#SBATCH --job-name=paper_train_sage_subend
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-11%4
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 30_train_sage_subend.sh — Phase E8: SAGE × sUbend (4th domain).
#
# 4 variants × 3 seeds = 12 tasks, %4 throttle.
#
# Pairs with job 24 (FlowGAT × sUbend). Combined with job 29, this covers
# SAGE on all four domains, fully closing the architecture-coverage gap
# flagged in the Limitations of main.tex.
#
# Array idx → (variant, seed):
#   0-2:  sage_subend_withleak       {1337, 2026, 777}
#   3-5:  sage_subend_leak_dir_only  {1337, 2026, 777}
#   6-8:  sage_subend_leak_mag_only  {1337, 2026, 777}
#   9-11: sage_subend_noleak         {1337, 2026, 777}
#
# Submit:  sbatch jobs/30_train_sage_subend.sh
# Walltime per run: ~4–8h (sUbend frames ~50k nodes, 25 frames × 15 cases).
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

VARIANTS=(
  "sage_subend_withleak"      "sage_subend_withleak"      "sage_subend_withleak"
  "sage_subend_leak_dir_only" "sage_subend_leak_dir_only" "sage_subend_leak_dir_only"
  "sage_subend_leak_mag_only" "sage_subend_leak_mag_only" "sage_subend_leak_mag_only"
  "sage_subend_noleak"        "sage_subend_noleak"        "sage_subend_noleak"
)
SEEDS=(1337 2026 777 1337 2026 777 1337 2026 777 1337 2026 777)

VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
RESULTS_DIR="$PAPER_DIR/results"
PER_SEED_DIR="$RESULTS_DIR/per_seed"
CKPT_DIR="$RESULTS_DIR/checkpoints/${VARIANT}/seed_${SEED}"
mkdir -p "$PER_SEED_DIR" "$CKPT_DIR"

echo "==========================================================================="
echo " SAGE_SUBEND TRAIN | variant=${VARIANT} seed=${SEED} | $(date)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[30] ERROR: missing $BEST_CKPT" >&2
  exit 1
fi

for SP in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SP}_seed${SEED}.csv"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" --ckpt "$BEST_CKPT" --split "$SP" --out "$OUT_CSV"
done

"$PYBIN" "$SRC_DIR/dump_predictions.py" \
  --config "$CFG" --ckpt "$BEST_CKPT" \
  --out_dir "$RESULTS_DIR/predictions" \
  --variant "$VARIANT" --seed "$SEED" || \
  echo "[30] WARN: dump_predictions failed"

echo "[30] DONE | variant=${VARIANT} seed=${SEED}"
