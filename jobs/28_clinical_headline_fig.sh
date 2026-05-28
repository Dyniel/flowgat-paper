#!/bin/bash
#SBATCH --job-name=paper_fig_clinical
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 28_clinical_headline_fig.sh — proposed Fig 1 generator.
#
# Produces results/figures/fig_clinical_headline.{pdf,png} and an
# accompanying _values.csv with the underlying numbers, so reviewers
# can compare the figure against the headline table in the paper.
#
# Pure CPU; reads 48 per-seed aggregate JSONs (4 domains x 4 variants x
# 3 seeds; some domains are partial coverage, missing values render as
# absent bars rather than zero).
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
PER_SEED="$RESULTS_DIR/per_seed"
FIG="$RESULTS_DIR/figures"
mkdir -p "$FIG"

echo "==========================================================================="
echo " FIG CLINICAL HEADLINE | $(date)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/make_fig_clinical_headline.py" \
  --per_seed_dir "$PER_SEED" \
  --out_dir      "$FIG"

echo "[28] DONE"
ls -la "$FIG" | grep clinical_headline
