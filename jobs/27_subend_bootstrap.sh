#!/bin/bash
#SBATCH --job-name=paper_subend_boot
#SBATCH --time=00:15:00
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --output=/users/scratch1/%u/flowgat_paper/logs/%x_%j.out

# ---------------------------------------------------------------------------
# 27_subend_bootstrap.sh — Phase E7 step 4: paired bootstrap CIs for the
# sUbend domain (4th replication of the asymmetric-leakage pattern).
#
# Resampling unit: case_id (n=15 sUbend cases) — much narrower CIs than
# the n=5 VMR baseline. We run all four pairwise contrasts that the
# reframed SR narrative needs:
#
#   - withleak vs noleak             — headline "leakage matters"
#   - leak_dir_only vs noleak        — direction-only equals withleak?
#   - leak_mag_only vs noleak        — magnitude-only equals noleak?
#   - withleak vs leak_dir_only      — does magnitude add anything?
#
# CPU-only; 10k bootstrap + 10k permutation iterations, takes <1 minute.
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
OUT_BASE="$RESULTS_DIR/bootstrap"
mkdir -p "$OUT_BASE"

echo "==========================================================================="
echo " SUBEND BOOTSTRAP | $(date)"
echo "==========================================================================="

run_pair () {
  local a="$1"
  local b="$2"
  local tag="$3"
  for sp in val test; do
    local out="$OUT_BASE/subend_${tag}_${sp}"
    mkdir -p "$out"
    echo "[27] bootstrap $a vs $b on $sp -> $out"
    "$PYBIN" "$SRC_DIR/bootstrap_ci.py" \
      --per_seed_dir "$PER_SEED" \
      --out_dir      "$out" \
      --split        "$sp" \
      --var_a        "$a" \
      --var_b        "$b" \
      --n_boot       10000 \
      --n_perm       10000 \
      --seed         42
  done
}

run_pair subend_withleak       subend_noleak         withleak_vs_noleak
run_pair subend_leak_dir_only  subend_noleak         dir_only_vs_noleak
run_pair subend_leak_mag_only  subend_noleak         mag_only_vs_noleak
run_pair subend_withleak       subend_leak_dir_only  withleak_vs_dir_only

echo "[27] DONE"
ls -1 "$OUT_BASE" | grep subend_ || true
