#!/bin/bash
#SBATCH --job-name=paper_dump
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --array=0-5
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 04_dump_predictions.sh — dump pred/y/pos for each (variant, seed) on
# val+test, for use by all downstream analyses.
#
# Array index maps to (variant, seed):
#   0: withleak 1337     3: noleak 1337
#   1: withleak 2026     4: noleak 2026
#   2: withleak 777      5: noleak 777
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
  "withleak 1337"
  "withleak 2026"
  "withleak 777"
  "noleak   1337"
  "noleak   2026"
  "noleak   777"
)
PAIR="${PAIRS[$SLURM_ARRAY_TASK_ID]}"
VARIANT="$(echo "$PAIR" | awk '{print $1}')"
SEED="$(echo "$PAIR" | awk '{print $2}')"

CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
CKPT="$PAPER_DIR/results/checkpoints/${VARIANT}/seed_${SEED}/best.pt"
OUT_DIR="$PAPER_DIR/results/predictions"

echo "==========================================================================="
echo " DUMP | variant=${VARIANT} seed=${SEED} | $(date) | host=$(hostname)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

if [[ ! -f "$CKPT" ]]; then
  echo "[04] ERROR: missing checkpoint $CKPT" >&2
  exit 1
fi

"$PYBIN" "$SRC_DIR/dump_predictions.py" \
  --config "$CFG" \
  --ckpt   "$CKPT" \
  --variant "$VARIANT" \
  --seed   "$SEED" \
  --out_dir "$OUT_DIR" \
  --splits val test

echo "[04] DONE | variant=${VARIANT} seed=${SEED}"
