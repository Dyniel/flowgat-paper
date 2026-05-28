#!/bin/bash
#SBATCH --job-name=paper_train_centerline
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 11_train_local_tangent.sh — Phase E1: train noleak_centerline (3 seeds).
#
# This variant tests the *second rung* of the direction-prior ladder:
# x[:,3:6] = iteratively-refined medial centerline tangent (per-node)
# instead of global PCA tangent (the existing noleak).
#
# Array index → seed:
#   0: 1337     1: 2026     2: 777
#
# Prerequisite: data/npz_noleak_centerline/ built by
#   python src/make_npz_local_tangent.py \
#     --src_dir data/npz_noleak --withleak_dir data/npz_withleak \
#     --out_dir data/npz_noleak_centerline
# (assistant runs this on login node, no GPU needed)
#
# Submit:
#   sbatch jobs/11_train_local_tangent.sh
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

VARIANT="noleak_centerline"
SEEDS=(1337 2026 777)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
RESULTS_DIR="$PAPER_DIR/results"
PER_SEED_DIR="$RESULTS_DIR/per_seed"
CKPT_DIR="$RESULTS_DIR/checkpoints/${VARIANT}/seed_${SEED}"
mkdir -p "$PER_SEED_DIR" "$CKPT_DIR"

echo "==========================================================================="
echo " TRAIN-CENTERLINE | variant=${VARIANT} seed=${SEED} | $(date) | host=$(hostname)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Train
"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

# Eval val + test
BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[11] ERROR: missing $BEST_CKPT" >&2
  exit 1
fi
for SPLIT in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SPLIT}_seed${SEED}.csv"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" --ckpt "$BEST_CKPT" --split "$SPLIT" --out "$OUT_CSV"
done

# Dump predictions for downstream physics-diagnostic analysis.
# NOTE: --out_dir is the *base* dir; dump_predictions.py appends /<variant>/seed_<seed>/.
"$PYBIN" "$SRC_DIR/dump_predictions.py" \
  --config "$CFG" --ckpt "$BEST_CKPT" \
  --out_dir "$RESULTS_DIR/predictions" \
  --variant "$VARIANT" --seed "$SEED"

echo "[11] DONE | variant=${VARIANT} seed=${SEED}"
