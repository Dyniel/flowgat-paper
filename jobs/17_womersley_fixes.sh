#!/bin/bash
#SBATCH --job-name=paper_wm_fix
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 17_womersley_fixes.sh — Phase E5, Step 1+3 in STRATEGY_CP.md.
#
# Pure CPU job. Operates on already-dumped predictions in
# results/predictions/womersley_* — does NOT require GPUs, training, or
# evaluation re-runs.
#
# Produces:
#   results/diagnostics/womersley/{per_case,aggregate,phase_per_case,
#                                  phase_binned,phase_flip_summary}.csv
#   results/diagnostics/womersley/{summary,phase_summary}.md
#   results/figures/womersley_phase_{angle,cos}.{png,pdf}
#
# Submit:
#   sbatch jobs/17_womersley_fixes.sh
# ---------------------------------------------------------------------------

set -euo pipefail
unalias python 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH

module load trytonp/conda/py313_25.9.1-3
eval "$(conda shell.bash hook)"
conda activate /users/scratch1/$USER/conda_envs/py312

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
SRC_DIR="$PAPER_DIR/src"
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"

RESULTS_DIR="$PAPER_DIR/results"
DIAG_DIR="$RESULTS_DIR/diagnostics/womersley"
FIG_DIR="$RESULTS_DIR/figures"

mkdir -p "$DIAG_DIR" "$FIG_DIR"

echo "==========================================================================="
echo " WOMERSLEY FIXES | $(date) | host=$(hostname)"
echo "==========================================================================="

echo "[17] (1/2) womersley_metrics — PP_dir, PP_peak, signed cosine"
"$PYBIN" "$SRC_DIR/womersley_metrics.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$DIAG_DIR"

echo
echo "[17] (2/2) womersley_phase_analysis — phase-aware diagnostic + figures"
"$PYBIN" "$SRC_DIR/womersley_phase_analysis.py" \
  --per_case_csv "$DIAG_DIR/per_case.csv" \
  --out_dir      "$DIAG_DIR" \
  --fig_dir      "$FIG_DIR" \
  --n_bins       4

echo
echo "[17] DONE"
echo "==========================================================================="
echo "Artifacts:"
ls -la "$DIAG_DIR" 2>/dev/null
echo
echo "Figures:"
ls -la "$FIG_DIR"/womersley_phase_*.* 2>/dev/null

echo
echo "----- summary.md (head) -----"
head -60 "$DIAG_DIR/summary.md" 2>/dev/null || true
echo
echo "----- phase_summary.md (head) -----"
head -60 "$DIAG_DIR/phase_summary.md" 2>/dev/null || true
