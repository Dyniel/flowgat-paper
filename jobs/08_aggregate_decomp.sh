#!/bin/bash
#SBATCH --job-name=paper_agg_decomp
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 08_aggregate_decomp.sh — re-run analyze.py on all 4 variants and refresh
# stratified + bootstrap analyses. Submit AFTER 07_train_decomp.sh.
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

echo "==========================================================================="
echo " AGGREGATE 4-VARIANT | $(date) | host=$(hostname)"
echo "==========================================================================="

# 1) Re-run analyze.py with the 4 variants now visible in per_seed/.
"$PYBIN" "$SRC_DIR/analyze.py" \
  --paper-dir "$PAPER_DIR" \
  --seeds     "1337,2026,777" \
  --variants  "withleak,leak_dir_only,leak_mag_only,noleak"

# 2) Stratified analysis (now with 4 variants)
"$PYBIN" "$SRC_DIR/stratified_analyze.py" \
  --per_seed_dir "$RESULTS_DIR/per_seed" \
  --out_dir      "$RESULTS_DIR/stratified" \
  --split        test

# 3) Pairwise bootstrap CIs: each non-withleak variant vs withleak, and vs noleak.
for VAR in leak_dir_only leak_mag_only noleak; do
  "$PYBIN" "$SRC_DIR/bootstrap_ci.py" \
    --per_seed_dir "$RESULTS_DIR/per_seed" \
    --out_dir      "$RESULTS_DIR/bootstrap" \
    --split        test \
    --var_a        "withleak" \
    --var_b        "$VAR" \
    --n_boot       10000 --n_perm 10000 --seed 42
  mv "$RESULTS_DIR/bootstrap/bootstrap_test.csv" \
     "$RESULTS_DIR/bootstrap/bootstrap_test_withleak_vs_${VAR}.csv"
  mv "$RESULTS_DIR/bootstrap/bootstrap_test.md" \
     "$RESULTS_DIR/bootstrap/bootstrap_test_withleak_vs_${VAR}.md"
done

# Also: leak_dir vs leak_mag (which sub-leak dominates)
"$PYBIN" "$SRC_DIR/bootstrap_ci.py" \
  --per_seed_dir "$RESULTS_DIR/per_seed" \
  --out_dir      "$RESULTS_DIR/bootstrap" \
  --split        test \
  --var_a        "leak_dir_only" \
  --var_b        "leak_mag_only" \
  --n_boot       10000 --n_perm 10000 --seed 42
mv "$RESULTS_DIR/bootstrap/bootstrap_test.csv" \
   "$RESULTS_DIR/bootstrap/bootstrap_test_dir_vs_mag.csv"
mv "$RESULTS_DIR/bootstrap/bootstrap_test.md" \
   "$RESULTS_DIR/bootstrap/bootstrap_test_dir_vs_mag.md"

echo "[08] DONE"
