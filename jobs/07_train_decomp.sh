#!/bin/bash
#SBATCH --job-name=paper_train_decomp
#SBATCH --time=10:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-5
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 07_train_decomp.sh — train 6 (variant, seed) pairs for feature decomposition.
#
# Array index → (variant, seed):
#   0: leak_dir_only 1337     3: leak_mag_only 1337
#   1: leak_dir_only 2026     4: leak_mag_only 2026
#   2: leak_dir_only 777      5: leak_mag_only 777
#
# Withleak runs took 5-6h; leak_dir_only should be similar (direction leak
# alone is highly predictive). leak_mag_only may take longer (closer to noleak
# which took ~10h). Walltime cap = 10h to cover both.
#
# Submit AFTER 06_build_decomp_data.sh:
#   sbatch -d afterok:<build_jobid> jobs/07_train_decomp.sh
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
  "leak_dir_only 1337"
  "leak_dir_only 2026"
  "leak_dir_only 777"
  "leak_mag_only 1337"
  "leak_mag_only 2026"
  "leak_mag_only 777"
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
echo " TRAIN-DECOMP | variant=${VARIANT} seed=${SEED} | $(date) | host=$(hostname)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Train
"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

# Eval val + test
BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[07] ERROR: missing $BEST_CKPT" >&2
  exit 1
fi

for SPLIT in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SPLIT}_seed${SEED}.csv"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" --ckpt "$BEST_CKPT" --split "$SPLIT" --out "$OUT_CSV"
done

echo "[07] DONE | variant=${VARIANT} seed=${SEED}"
