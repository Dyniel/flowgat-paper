#!/bin/bash
#SBATCH --job-name=paper_analyze
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 03_analyze.sh — aggregate per-seed CSVs/JSONs into 3-seed tables, write
# leakage-delta dataframe, generate paper figures, write manifest.json.
#
# Run after all 6 (variant, seed) trainings finish (or even partially — script
# is robust to missing seeds and reports n_seeds in summary).
#
# Usage:
#   sbatch jobs/03_analyze.sh
# Or interactively:
#   bash jobs/03_analyze.sh
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
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"

echo "[03] Aggregate per-seed -> 3-seed tables"
"$PYBIN" "$PAPER_DIR/src/analyze.py" --paper-dir "$PAPER_DIR" --seeds "1337,2026,777"

echo "[03] Generate paper figures"
"$PYBIN" "$PAPER_DIR/src/make_figures.py" --paper-dir "$PAPER_DIR"

echo "[03] DONE."
ls -la "$PAPER_DIR/results/aggregated/"
ls -la "$PAPER_DIR/results/figures/" 2>/dev/null || true
