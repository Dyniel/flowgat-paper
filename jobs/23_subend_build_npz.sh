#!/bin/bash
#SBATCH --job-name=paper_subend_build
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=64G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 23_subend_build_npz.sh — Phase E7 step 1.
#
# Pure CPU job. Reads extracted Suk-sUbend VTK frames under
#   data/external/newCFD_dataset/sUbend_*/CFD/Frame_*.vtk
# and builds the project NPZ schema under
#   data/npz_subend/                        (noleak base)
#   data/npz_subend_withleak/               (variant)
#   data/npz_subend_leak_dir_only/          (variant)
#   data/npz_subend_leak_mag_only/          (variant)
#
# Stratified-subsamples each 1.13M-node hex mesh to MAX_NODES nodes
# (keeps all wall nodes), preserves the 25-frame pulsatile axis as t_phase
# in meta, and emits per-frame NPZs with a case-disjoint split.json.
#
# Submit:
#   sbatch jobs/23_subend_build_npz.sh
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

SRC_ROOT="$PAPER_DIR/data/external/newCFD_dataset"
OUT_BASE="$PAPER_DIR/data/npz_subend"

MAX_NODES="${MAX_NODES:-50000}"
KNN="${KNN:-16}"
WALL_T="${WALL_T:-0.01}"
N_TRAIN="${N_TRAIN:-9}"
N_VAL="${N_VAL:-3}"
N_TEST="${N_TEST:-3}"

echo "==========================================================================="
echo " SUBEND BUILD | $(date) | host=$(hostname)"
echo "==========================================================================="
echo "[23] MAX_NODES=$MAX_NODES KNN=$KNN WALL_T=$WALL_T  split=$N_TRAIN/$N_VAL/$N_TEST"

if ! ls "$SRC_ROOT"/sUbend_*/CFD/Frame_00.vtk >/dev/null 2>&1; then
    echo "[23] ERROR: no sUbend Frame_00.vtk files found under $SRC_ROOT"
    echo "[23] Run the extractor first: data/external/extract_subend.py"
    exit 1
fi

echo "[23] (1/2) build noleak NPZs"
"$PYBIN" "$SRC_DIR/make_npz_subend.py" \
  --src_root "$SRC_ROOT" \
  --out_dir  "$OUT_BASE" \
  --max_nodes "$MAX_NODES" \
  --knn "$KNN" \
  --wall_speed_threshold "$WALL_T" \
  --n_train "$N_TRAIN" \
  --n_val "$N_VAL" \
  --n_test "$N_TEST"

echo
echo "[23] (2/2) build 3 leakage variants"
"$PYBIN" "$SRC_DIR/make_npz_subend_variants.py" \
  --src_dir "$OUT_BASE" \
  --out_root "$PAPER_DIR/data"

echo
echo "[23] DONE"
echo "==========================================================================="
ls -d "$PAPER_DIR/data/npz_subend"* 2>/dev/null
for d in "$PAPER_DIR/data/npz_subend"*; do
    n=$(ls "$d"/*.npz 2>/dev/null | wc -l)
    echo "  $d : $n npz"
done
