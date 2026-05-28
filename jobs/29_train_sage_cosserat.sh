#!/bin/bash
#SBATCH --job-name=paper_train_sage_cosserat
#SBATCH --time=08:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-11%4
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 29_train_sage_cosserat.sh — Phase E8: SAGE × Cosserat sweep.
#
# 4 variants × 3 seeds = 12 tasks, capped at 4 concurrent (%4) to leave
# headroom for other projects on the GPU partition.
#
# Closes the architecture-coverage gap on the curved-tube domain (Discussion
# limitation in main.tex:1209–1213). Same data + features as the FlowGAT
# Cosserat runs (job 19); only the backbone differs (flow_sage, no attention,
# no edge-bias, no hard-no-slip head).
#
# Array idx → (variant, seed):
#   0-2:  sage_cosserat_withleak       {1337, 2026, 777}
#   3-5:  sage_cosserat_leak_dir_only  {1337, 2026, 777}
#   6-8:  sage_cosserat_leak_mag_only  {1337, 2026, 777}
#   9-11: sage_cosserat_noleak         {1337, 2026, 777}
#
# Submit:  sbatch jobs/29_train_sage_cosserat.sh
# Walltime per run: ~2–4h (Cosserat is ~23k nodes; smaller than VMR SAGE).
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
  "sage_cosserat_withleak"      "sage_cosserat_withleak"      "sage_cosserat_withleak"
  "sage_cosserat_leak_dir_only" "sage_cosserat_leak_dir_only" "sage_cosserat_leak_dir_only"
  "sage_cosserat_leak_mag_only" "sage_cosserat_leak_mag_only" "sage_cosserat_leak_mag_only"
  "sage_cosserat_noleak"        "sage_cosserat_noleak"        "sage_cosserat_noleak"
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
echo " SAGE_COSSERAT TRAIN | variant=${VARIANT} seed=${SEED} | $(date)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[29] ERROR: missing $BEST_CKPT" >&2
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
  echo "[29] WARN: dump_predictions failed"

echo "[29] DONE | variant=${VARIANT} seed=${SEED}"
