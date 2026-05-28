#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --time=00:20:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 09_figures_final.sh — regenerate Fig 2/3/4/5/6 from the latest aggregated
# tables + dumped predictions. Submit AFTER 05_diagnostics.sh (and after
# 08_aggregate_decomp.sh if Phase B was run).
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

echo "==========================================================================="
echo " FIGURES v2 | $(date) | host=$(hostname)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/make_figures_v2.py" --paper-dir "$PAPER_DIR"

ls -lh "$PAPER_DIR/results/figures/" | tail -20
echo "[09] DONE"
