#!/bin/bash
#SBATCH --job-name=paper_build_data
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 00_build_data.sh — rebuild withleak + noleak NPZ datasets from VMR raw VTU.
#
# This is OPTIONAL for the CP submission run — the package ships with
# data/npz_withleak/ and data/npz_noleak/ already built. This script exists so
# reviewers can rebuild from scratch end-to-end.
#
# Skips if NPZ already exist (idempotent). Force rebuild with FORCE=1.
#
# Usage (review only):
#   sbatch jobs/00_build_data.sh
#   sbatch --export=ALL,FORCE=1 jobs/00_build_data.sh
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
DATA_DIR="$PAPER_DIR/data"
WITHLEAK="$DATA_DIR/npz_withleak"
NOLEAK="$DATA_DIR/npz_noleak"
VMR_RAW="${VMR_RAW:-/users/scratch1/$USER/data/datav2}"

PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"
FORCE="${FORCE:-0}"

echo "[00] Rebuild data — VMR_RAW=$VMR_RAW FORCE=$FORCE"

# --- withleak (vmr_npz_v2 schema): x[:,3:6]=unit(y), x[:,8]=u_mean/u_char ---
if [[ "$FORCE" == "1" || ! -f "$WITHLEAK/split.json" ]]; then
  echo "[00] Building $WITHLEAK from $VMR_RAW (with-leak schema)"
  rm -rf "$WITHLEAK"
  mkdir -p "$WITHLEAK"
  "$PYBIN" "$SRC_DIR/preprocess_vmr.py" \
    --vmr-root "$VMR_RAW" \
    --out-dir  "$WITHLEAK" \
    --fsi-vtu \
    --skip-coronary
else
  echo "[00] withleak already built — skipping (use FORCE=1 to rebuild)"
fi

# --- noleak: copy mesh/labels/split from withleak, rebuild x as geom-only ---
if [[ "$FORCE" == "1" || ! -f "$NOLEAK/split.json" ]]; then
  echo "[00] Building $NOLEAK from $WITHLEAK (no-leak schema)"
  rm -rf "$NOLEAK"
  mkdir -p "$NOLEAK"
  "$PYBIN" "$SRC_DIR/make_npz_noleak.py" \
    --src "$WITHLEAK" \
    --dst "$NOLEAK"
else
  echo "[00] noleak already built — skipping (use FORCE=1 to rebuild)"
fi

echo "[00] Counts:"
echo "  withleak: $(ls $WITHLEAK/*.npz 2>/dev/null | wc -l) NPZ"
echo "  noleak:   $(ls $NOLEAK/*.npz 2>/dev/null | wc -l) NPZ"

echo "[00] DONE."
