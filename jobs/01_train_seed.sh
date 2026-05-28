#!/bin/bash
#SBATCH --job-name=paper_train
#SBATCH --time=14:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 01_train_seed.sh — train ONE (variant, seed) pair and write per-seed eval.
#
# Required env vars:
#   VARIANT   = withleak | noleak
#   SEED      = e.g. 1337 / 2026 / 777
# Optional:
#   EPOCHS    = override training.epochs (default: from config = 1500)
#   SMOKE     = 1 to run 3-epoch smoke test
#
# Usage (one of 6 paper runs):
#   sbatch --export=ALL,VARIANT=withleak,SEED=1337 jobs/01_train_seed.sh
#   sbatch --export=ALL,VARIANT=noleak,SEED=2026   jobs/01_train_seed.sh
#
# Smoke (3 epochs, sanity):
#   sbatch --export=ALL,VARIANT=withleak,SEED=1337,SMOKE=1 jobs/01_train_seed.sh
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

VARIANT="${VARIANT:?VARIANT (withleak|noleak) must be set}"
SEED="${SEED:?SEED must be set}"
EPOCHS="${EPOCHS:-}"
SMOKE="${SMOKE:-0}"

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
SRC_DIR="$PAPER_DIR/src"
CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"
RESULTS_DIR="$PAPER_DIR/results"
PER_SEED_DIR="$RESULTS_DIR/per_seed"
CKPT_DIR="$RESULTS_DIR/checkpoints/${VARIANT}/seed_${SEED}"

mkdir -p "$PER_SEED_DIR" "$CKPT_DIR"

echo "==========================================================================="
echo " FlowGAT-Paper TRAIN | variant=${VARIANT} seed=${SEED} | $(date)"
echo "==========================================================================="
echo "HOSTNAME=$(hostname)"
echo "CFG=$CFG"
echo "CKPT_DIR=$CKPT_DIR"
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ---- Train ----------------------------------------------------------------
TRAIN_ARGS=( --config "$CFG" --seeds "$SEED" )
TRAIN_ARGS+=( --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}" )
[[ -n "$EPOCHS" ]] && TRAIN_ARGS+=( --override "training.epochs=$EPOCHS" )
[[ "$SMOKE" == "1" ]] && TRAIN_ARGS+=( --smoke )

echo "[01] Training: ${TRAIN_ARGS[*]}"
"$PYBIN" "$SRC_DIR/train.py" "${TRAIN_ARGS[@]}"

# ---- Evaluate val + test --------------------------------------------------
BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[01] ERROR: best.pt not found at $BEST_CKPT" >&2
  exit 1
fi

for SPLIT in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SPLIT}_seed${SEED}.csv"
  echo "[01] Eval split=$SPLIT -> $OUT_CSV"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" \
    --ckpt   "$BEST_CKPT" \
    --split  "$SPLIT" \
    --out    "$OUT_CSV"
done

echo "[01] DONE | variant=${VARIANT} seed=${SEED}"
