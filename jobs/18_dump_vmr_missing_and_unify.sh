#!/bin/bash
#SBATCH --job-name=paper_unify
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 18_dump_vmr_missing_and_unify.sh — closes the metric-coverage gap.
#
# Phase E5+ step: dump VMR predictions for the three variants we don't have
# (withleak, leak_dir_only, leak_mag_only) × 3 seeds, then re-run the unified
# metrics (PP_dir, PP_peak, cos_signed) over *every* available variant so the
# paper can quote a symmetric VMR-vs-Womersley comparison.
#
# Why a single job: dump needs GPU (model.forward), the metrics that follow
# are pure CPU and only take seconds — so we keep them in the same allocation
# instead of orchestrating dependencies.
#
# Submit:
#   sbatch jobs/18_dump_vmr_missing_and_unify.sh
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

PRED_DIR="$PAPER_DIR/results/predictions"
DIAG_DIR="$PAPER_DIR/results/diagnostics/all_variants"
FIG_DIR="$PAPER_DIR/results/figures"

mkdir -p "$DIAG_DIR" "$FIG_DIR"

echo "==========================================================================="
echo " UNIFIED METRICS (VMR + Womersley) | $(date) | host=$(hostname)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# STAGE 1 — dump missing VMR predictions (GPU)
# ---------------------------------------------------------------------------
SEEDS=(1337 2026 777)
VARIANTS=(withleak leak_dir_only leak_mag_only)

for V in "${VARIANTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    CKPT="$PAPER_DIR/results/checkpoints/${V}/seed_${S}/best.pt"
    OUT_SEED="$PRED_DIR/${V}/seed_${S}"

    if [[ ! -f "$CKPT" ]]; then
      echo "[18] SKIP missing ckpt: $CKPT"
      continue
    fi

    # Skip if predictions already exist and look complete (val + test)
    if ls "$OUT_SEED"/val_*.npz >/dev/null 2>&1 \
       && ls "$OUT_SEED"/test_*.npz >/dev/null 2>&1; then
      echo "[18] SKIP existing dumps for ${V} seed=${S}"
      continue
    fi

    echo
    echo "----- dump ${V} seed=${S} -----"
    "$PYBIN" "$SRC_DIR/dump_predictions.py" \
      --config "$PAPER_DIR/configs/${V}.yaml" \
      --ckpt   "$CKPT" \
      --variant "$V" \
      --seed   "$S" \
      --out_dir "$PRED_DIR" \
      --splits val test
  done
done

echo
echo "[18] STAGE 1 done — dumps:"
for V in "${VARIANTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    O="$PRED_DIR/${V}/seed_${S}"
    if [[ -d "$O" ]]; then
      N=$(ls "$O" 2>/dev/null | wc -l)
      echo "    ${V}/seed_${S}: $N files"
    fi
  done
done

# ---------------------------------------------------------------------------
# STAGE 2 — unified metrics across ALL variants (CPU only, but already in
# allocation so we just run it here)
# ---------------------------------------------------------------------------
echo
echo "==========================================================================="
echo " STAGE 2 — unified PP_dir / PP_peak / cos_signed across all variants"
echo "==========================================================================="

ALL_VARIANTS=(
  withleak leak_dir_only leak_mag_only noleak noleak_centerline
  sage_withleak sage_noleak
  womersley_withleak womersley_leak_dir_only
  womersley_leak_mag_only womersley_noleak
)

"$PYBIN" "$SRC_DIR/womersley_metrics.py" \
  --predictions_root "$PRED_DIR" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$DIAG_DIR" \
  --variants         "${ALL_VARIANTS[@]}"

echo
echo "==========================================================================="
echo " STAGE 3 — phase analysis re-run on unified per_case (Womersley subset)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/womersley_phase_analysis.py" \
  --per_case_csv "$DIAG_DIR/per_case.csv" \
  --out_dir      "$DIAG_DIR" \
  --fig_dir      "$FIG_DIR" \
  --n_bins       4

echo
echo "[18] DONE"
echo "==========================================================================="
echo "Artifacts:"
ls -la "$DIAG_DIR" 2>/dev/null
echo
echo "----- aggregate.csv (variant, split rows) -----"
head -20 "$DIAG_DIR/aggregate.csv" 2>/dev/null | column -t -s, 2>/dev/null || head -20 "$DIAG_DIR/aggregate.csv"
echo
echo "----- summary.md (head) -----"
head -80 "$DIAG_DIR/summary.md" 2>/dev/null || true
