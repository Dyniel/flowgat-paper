#!/bin/bash
#SBATCH --job-name=paper_recover_dumps
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 16_recover_dumps.sh — dump predictions for all variants that finished
# training but failed at the dump_predictions step (due to a --variant/--seed
# arg bug in the original 11/12/13 scripts; since fixed).
#
# Runs sequentially in ONE slurm allocation (no array, since we are capped
# at 5 concurrent pending+running tasks). Each dump is ~5-10 min, total
# ~2-3h for 15 (variant, seed) pairs.
#
# Skip-if-exists: if a dump dir already has .npz files, that (variant,seed)
# is skipped. Safe to re-run.
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

dump_one() {
  local VARIANT=$1
  local SEED=$2
  local CFG="$PAPER_DIR/configs/${VARIANT}.yaml"
  local CKPT="$PAPER_DIR/results/checkpoints/${VARIANT}/seed_${SEED}/best.pt"
  local OUT_DIR="$PAPER_DIR/results/predictions/${VARIANT}/seed_${SEED}"

  if [[ ! -f "$CKPT" ]]; then
    echo "[16] SKIP $VARIANT seed=$SEED — no best.pt"
    return 0
  fi
  if compgen -G "$OUT_DIR/*.npz" > /dev/null; then
    local n=$(ls "$OUT_DIR"/*.npz 2>/dev/null | wc -l)
    echo "[16] SKIP $VARIANT seed=$SEED — already has $n .npz files"
    return 0
  fi

  echo "===[16] DUMP $VARIANT seed=$SEED -> $OUT_DIR ==="
  "$PYBIN" "$SRC_DIR/dump_predictions.py" \
    --config "$CFG" --ckpt "$CKPT" \
    --out_dir "$PAPER_DIR/results/predictions" \
    --variant "$VARIANT" --seed "$SEED"
  echo "===[16] DONE $VARIANT seed=$SEED ==="
}

# (variant, seed) pairs to recover. Order: smallest-mesh first for warm-up.
PAIRS=(
  # Womersley (small meshes, ~23k nodes, fast dumps)
  "womersley_withleak 1337"      "womersley_withleak 2026"      "womersley_withleak 777"
  "womersley_leak_dir_only 1337" "womersley_leak_dir_only 2026" "womersley_leak_dir_only 777"
  "womersley_leak_mag_only 1337" "womersley_leak_mag_only 2026" "womersley_leak_mag_only 777"
  "womersley_noleak 1337"        "womersley_noleak 2026"        "womersley_noleak 777"
  # SAGE on VMR (large meshes, slower dumps)
  "sage_withleak 1337"           "sage_withleak 2026"           "sage_withleak 777"
  "sage_noleak 1337"             "sage_noleak 2026"             "sage_noleak 777"
)

START_TIME=$(date +%s)
for PAIR in "${PAIRS[@]}"; do
  read -r VARIANT SEED <<<"$PAIR"
  dump_one "$VARIANT" "$SEED" || echo "[16] WARN: $VARIANT seed=$SEED failed (continuing)"
  ELAPSED=$(( $(date +%s) - START_TIME ))
  echo "  [elapsed ${ELAPSED}s]"
done

echo "[16] all done"
