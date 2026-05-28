#!/bin/bash
#SBATCH --job-name=paper_dump_existing
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-5
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 14_dump_existing.sh — dump predictions for leak_dir_only / leak_mag_only
# (checkpoints already trained, but predictions/*.npz never produced).
# Needed by physics_diagnostics.py.
#
# Array idx → (variant, seed):
#   0-2: leak_dir_only {1337, 2026, 777}
#   3-5: leak_mag_only {1337, 2026, 777}
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
  "leak_dir_only" "leak_dir_only" "leak_dir_only"
  "leak_mag_only" "leak_mag_only" "leak_mag_only"
)
SEEDS=(1337 2026 777 1337 2026 777)
VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
CKPT="$PAPER_DIR/results/checkpoints/${VARIANT}/seed_${SEED}/best.pt"
OUT_DIR="$PAPER_DIR/results/predictions/${VARIANT}/seed_${SEED}"

if [[ ! -f "$CKPT" ]]; then
  echo "[14] missing $CKPT — skipping"
  exit 0
fi

echo "[14] DUMP $VARIANT seed=$SEED → $OUT_DIR"
"$PYBIN" "$SRC_DIR/dump_predictions.py" \
  --config "$CFG" --ckpt "$CKPT" \
  --out_dir "$OUT_DIR" \
  --variant "$VARIANT" --seed "$SEED"
echo "[14] DONE $VARIANT seed=$SEED"
