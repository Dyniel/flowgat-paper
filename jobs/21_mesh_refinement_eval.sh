#!/bin/bash
#SBATCH --job-name=paper_meshref_eval
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --array=0-35
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%A_%a.out

# ---------------------------------------------------------------------------
# 21_mesh_refinement_eval.sh — Concern 4: Womersley mesh-refinement eval.
#
# Evaluation only.  No training.
#
# Array idx -> 4 variants x 3 seeds x 3 resolutions = 36 tasks:
#   variants:    withleak, leak_dir_only, leak_mag_only, noleak
#   seeds:       1337, 2026, 777
#   resolutions: 1x, 2x, 4x
#
# Prerequisite datasets, generated on the login node:
#   python src/make_npz_womersley.py --out_dir data/npz_womersley_meshref_1x \
#     --n_train 24 --n_val 4 --n_test 6 --seed 2026 \
#     --n_axial 80 --n_radial 12 --n_angular 24
#   python src/make_npz_womersley.py --out_dir data/npz_womersley_meshref_2x \
#     --n_train 24 --n_val 4 --n_test 6 --seed 2026 \
#     --n_axial 160 --n_radial 24 --n_angular 48
#   python src/make_npz_womersley.py --out_dir data/npz_womersley_meshref_4x \
#     --n_train 24 --n_val 4 --n_test 6 --seed 2026 \
#     --n_axial 320 --n_radial 48 --n_angular 96
#
# Submit:
#   sbatch jobs/21_mesh_refinement_eval.sh
#
# After all array tasks complete, aggregate on the login node:
#   python src/mesh_refinement_diagnostic.py --mode aggregate \
#     --out_dir results/diagnostics/mesh_refinement
# ---------------------------------------------------------------------------

set -euo pipefail
unalias python 2>/dev/null || true
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
unset PYTHONPATH

module load trytonp/conda/py313_25.9.1-3
module load trytonp/nvidia_hpc_sdk/24.3
eval "$(conda shell.bash hook)"
conda activate /users/scratch1/$USER/conda_envs/py312

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
SRC_DIR="$PAPER_DIR/src"
PYBIN="/users/scratch1/$USER/conda_envs/py312/bin/python"
OUT_DIR="$PAPER_DIR/results/diagnostics/mesh_refinement"

VARIANTS=(withleak leak_dir_only leak_mag_only noleak)
SEEDS=(1337 2026 777)
RESOLUTIONS=(1x 2x 4x)

IDX="${SLURM_ARRAY_TASK_ID}"
VARIANT="${VARIANTS[$((IDX / 9))]}"
SEED="${SEEDS[$(((IDX / 3) % 3))]}"
RESOLUTION="${RESOLUTIONS[$((IDX % 3))]}"

echo "==========================================================================="
echo " MESHREF | variant=${VARIANT} seed=${SEED} resolution=${RESOLUTION} | $(date)"
echo "==========================================================================="
"$PYBIN" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd "$PAPER_DIR"
mkdir -p "$OUT_DIR/parts"

"$PYBIN" "$SRC_DIR/mesh_refinement_diagnostic.py" \
  --mode eval \
  --variant "$VARIANT" \
  --seed "$SEED" \
  --resolution "$RESOLUTION" \
  --data_root "$PAPER_DIR/data" \
  --ckpt_root "$PAPER_DIR/results/checkpoints" \
  --config_root "$PAPER_DIR/configs" \
  --out_dir "$OUT_DIR" \
  --chunk_nodes 256000 \
  --div_k 16 \
  --helm_k 16 \
  --helm_cg_tol 1e-6 \
  --helm_cg_maxiter 500

echo "[21] DONE | variant=${VARIANT} seed=${SEED} resolution=${RESOLUTION}"
