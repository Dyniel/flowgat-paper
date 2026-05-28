#!/bin/bash
#SBATCH --job-name=paper_train_flowgat_nobc_cosserat
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-11%4
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 31_train_flowgat_nobc_cosserat.sh — Phase E8: FlowGAT × Cosserat, no-slip
# head OFF. Explicit ablation of the hard-no-slip boundary condition.
#
# 4 variants × 3 seeds = 12 tasks, %4 throttle.
#
# Companion to job 19 (FlowGAT × Cosserat, head ON). Together they isolate
# the causal role of wall-zeroing for the asymmetric direction/magnitude
# pattern; if the pattern reproduces with head OFF, the no-slip head is a
# design detail rather than the mechanism — matching the implicit signal
# from the SAGE runs (jobs 13, 29, 30), whose backbone lacks the head.
#
# Array idx → (variant, seed):
#   0-2:  cosserat_sweep_withleak_nobc       {1337, 2026, 777}
#   3-5:  cosserat_sweep_leak_dir_only_nobc  {1337, 2026, 777}
#   6-8:  cosserat_sweep_leak_mag_only_nobc  {1337, 2026, 777}
#   9-11: cosserat_sweep_noleak_nobc         {1337, 2026, 777}
#
# Submit:  sbatch jobs/31_train_flowgat_nobc_cosserat.sh
# Walltime per run: ~1.5–3h (mirrors job 19 Cosserat budget).
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
  "cosserat_sweep_withleak_nobc"      "cosserat_sweep_withleak_nobc"      "cosserat_sweep_withleak_nobc"
  "cosserat_sweep_leak_dir_only_nobc" "cosserat_sweep_leak_dir_only_nobc" "cosserat_sweep_leak_dir_only_nobc"
  "cosserat_sweep_leak_mag_only_nobc" "cosserat_sweep_leak_mag_only_nobc" "cosserat_sweep_leak_mag_only_nobc"
  "cosserat_sweep_noleak_nobc"        "cosserat_sweep_noleak_nobc"        "cosserat_sweep_noleak_nobc"
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
echo " FLOWGAT_NOBC_COSSERAT TRAIN | variant=${VARIANT} seed=${SEED} | $(date)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

"$PYBIN" "$SRC_DIR/train.py" \
  --config "$CFG" --seeds "$SEED" \
  --override "training.ckpt_dir=$RESULTS_DIR/checkpoints/${VARIANT}"

BEST_CKPT="$CKPT_DIR/best.pt"
if [[ ! -f "$BEST_CKPT" ]]; then
  echo "[31] ERROR: missing $BEST_CKPT" >&2
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
  echo "[31] WARN: dump_predictions failed"

echo "[31] DONE | variant=${VARIANT} seed=${SEED}"
