#!/bin/bash
#SBATCH --job-name=paper_recover_centerline
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --array=0-2
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 15_recover_centerline.sh — eval + dump predictions for noleak_centerline.
#
# Background:
#   - seed=1337, seed=777 were CANCELLED by SLURM time limit at epoch ~1453
#     of 1500. best.pt was already saved at the model's true peak (val-best
#     epoch around ~800-900). No retraining needed.
#   - seed=2026 trained fully; eval CSVs were written; only the
#     dump_predictions step failed (--variant/--seed args bug in original
#     11_train_local_tangent.sh, since fixed). best.pt is intact.
#
# This recovery just runs evaluate.py + dump_predictions.py from best.pt
# for all 3 seeds. ~15 min per seed on A100.
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
CKPT="$PAPER_DIR/results/checkpoints/${VARIANT}/seed_${SEED}/best.pt"
PER_SEED_DIR="$PAPER_DIR/results/per_seed"

if [[ ! -f "$CKPT" ]]; then
  echo "[15] ERROR: $CKPT missing"
  exit 1
fi

echo "=== RECOVER | variant=${VARIANT} seed=${SEED} | $(date) ==="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# Eval (overwrites existing CSVs if any; idempotent)
for SP in val test; do
  OUT_CSV="$PER_SEED_DIR/${VARIANT}_${SP}_seed${SEED}.csv"
  "$PYBIN" "$SRC_DIR/evaluate.py" \
    --config "$CFG" --ckpt "$CKPT" --split "$SP" --out "$OUT_CSV"
done

# Dump predictions (--out_dir is the BASE; script appends /<variant>/seed_<seed>/)
"$PYBIN" "$SRC_DIR/dump_predictions.py" \
  --config "$CFG" --ckpt "$CKPT" \
  --out_dir "$PAPER_DIR/results/predictions" \
  --variant "$VARIANT" --seed "$SEED"

echo "[15] DONE | variant=${VARIANT} seed=${SEED}"
