#!/bin/bash
#SBATCH --job-name=paper_agg_cs_meshref
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 22_aggregate_cosserat_meshref.sh — SR-reframe step 1.
#
# Pure CPU. Runs two post-hoc aggregations on already-dumped predictions and
# already-evaluated mesh-refinement parts:
#
#   (a) Cosserat sweep diagnostic (4 variants x 3 seeds)
#       -> results/diagnostics/cosserat_sweep/{per_case,aggregate,
#                                              stratified_by_de,
#                                              stratified_by_eps}.csv
#       -> results/diagnostics/cosserat_sweep/summary.md
#
#   (b) Mesh-refinement aggregate (4 variants x 3 seeds x 3 resolutions)
#       -> results/diagnostics/mesh_refinement/{per_case,aggregate}.csv
#       -> results/diagnostics/mesh_refinement/summary.md
#
# Submit:
#   sbatch jobs/22_aggregate_cosserat_meshref.sh
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
CS_OUT="$RESULTS_DIR/diagnostics/cosserat_sweep"
MR_OUT="$RESULTS_DIR/diagnostics/mesh_refinement"

mkdir -p "$CS_OUT" "$MR_OUT"

echo "==========================================================================="
echo " AGGREGATE | $(date) | host=$(hostname)"
echo "==========================================================================="

echo "[22] (1/2) Cosserat sweep diagnostic"
"$PYBIN" "$SRC_DIR/cosserat_sweep_diagnostic.py" \
  --predictions_root "$RESULTS_DIR/predictions" \
  --data_root        "$PAPER_DIR/data" \
  --out_dir          "$CS_OUT"

echo
echo "[22] (2/2) Mesh refinement aggregate"
"$PYBIN" "$SRC_DIR/mesh_refinement_diagnostic.py" \
  --mode aggregate \
  --out_dir "$MR_OUT"

echo
echo "[22] DONE"
echo "==========================================================================="
echo "Cosserat artifacts:"
ls -la "$CS_OUT" 2>/dev/null
echo
echo "Mesh refinement artifacts:"
ls -la "$MR_OUT" 2>/dev/null | grep -v parts

echo
echo "----- cosserat_sweep/summary.md -----"
cat "$CS_OUT/summary.md" 2>/dev/null || true
echo
echo "----- mesh_refinement/summary.md (head) -----"
head -80 "$MR_OUT/summary.md" 2>/dev/null || true
