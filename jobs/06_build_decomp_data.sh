#!/bin/bash
#SBATCH --job-name=paper_build_decomp
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 06_build_decomp_data.sh — build leak_dir_only + leak_mag_only NPZ datasets
# from existing npz_withleak + npz_noleak. Phase B prerequisite.
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
DATA_DIR="$PAPER_DIR/data"

echo "==========================================================================="
echo " BUILD DECOMP DATA | $(date) | host=$(hostname)"
echo "==========================================================================="

"$PYBIN" "$SRC_DIR/make_npz_leak_decomp.py" \
  --withleak_dir "$DATA_DIR/npz_withleak" \
  --noleak_dir   "$DATA_DIR/npz_noleak" \
  --out_dir_dir  "$DATA_DIR/npz_leak_dir_only" \
  --out_dir_mag  "$DATA_DIR/npz_leak_mag_only"

echo "[06] DONE"
ls -lh "$DATA_DIR/npz_leak_dir_only" | head -3
ls -lh "$DATA_DIR/npz_leak_mag_only" | head -3
