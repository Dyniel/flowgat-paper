#!/bin/bash
# Login-node orchestrator for the Cosserat sweep. This is not a SLURM batch.

set -euo pipefail

PAPER_DIR="${PAPER_DIR:-/users/scratch1/$USER/flowgat_paper}"
PYBIN="${PYBIN:-/users/scratch1/$USER/conda_envs/py312/bin/python}"

cd "$PAPER_DIR"

run_diagnostic() {
  echo "[20] step 5/5: run Cosserat sweep diagnostic"
  "$PYBIN" src/cosserat_sweep_diagnostic.py \
    --predictions_root "$PAPER_DIR/results/predictions" \
    --data_root "$PAPER_DIR/data" \
    --out_dir "$PAPER_DIR/results/diagnostics/cosserat_sweep"
}

if [[ "${1:-}" == "diagnostic" ]]; then
  run_diagnostic
  exit 0
fi

echo "[20] step 1/5: generate base npz_cosserat_sweep"
"$PYBIN" src/make_npz_cosserat_sweep.py \
  --out_dir "$PAPER_DIR/data/npz_cosserat_sweep" \
  --n_cases 300 \
  --seed 2026

echo "[20] step 2/5: generate leakage variants"
"$PYBIN" src/make_npz_cosserat_sweep_variants.py \
  --src_dir "$PAPER_DIR/data/npz_cosserat_sweep" \
  --out_root "$PAPER_DIR/data"

echo "[20] step 3/5: submit SLURM training array"
SBATCH_OUT="$(sbatch jobs/19_train_cosserat_sweep.sh)"
JOBID="$(printf '%s\n' "$SBATCH_OUT" | awk '{print $NF}')"
echo "[20] submitted JOBID=$JOBID"

echo "[20] step 4/5: now wait for JOBID=$JOBID to finish, then run step 5:"
echo "  bash jobs/20_cosserat_sweep_pipeline.sh diagnostic"
