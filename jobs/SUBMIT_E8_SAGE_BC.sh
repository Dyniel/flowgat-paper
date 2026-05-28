#!/bin/bash
# ---------------------------------------------------------------------------
# SUBMIT_E8_SAGE_BC.sh — Phase E8 follow-up launcher.
#
# Closes two reviewer-attack surfaces flagged in Limitations of main.tex:
#   (a) SAGE × {Cosserat, sUbend} — architecture coverage on curved domains
#   (b) FlowGAT × Cosserat no-BC ablation — explicit no-slip-head control
#
# Strategy:
# - Three train arrays (jobs 29, 30, 31), each 12 tasks (4 var × 3 seeds).
# - Each array uses `--array=0-11%4`: max 4 concurrent inside that array.
# - This launcher submits only STAGE 1 (job 29 + its agg job 32). Each agg
#   then trampolines: job 32 sbatch's job 30 + 33; job 33 sbatch's job 31
#   + 34. This keeps the in-queue count ≤ ~13 slots at any one time, which
#   matters because the cluster's AssocMaxSubmitJobLimit counts each
#   array task as a separate submission slot. Submitting all three arrays
#   up-front (29+30+31 = 36 slots + 3 aggs) would exceed the limit.
# - Aggregation jobs (32, 33, 34) are CPU-only (`--gres=gpu:0`).
#
# Total walltime (rough): Cosserat sweep ~3h × 3 batches = ~9h + sUbend
# ~8h × 3 batches = ~24h + nobc_cosserat ~2h × 3 = ~6h → ~40h wall.
#
# Usage:
#   bash jobs/SUBMIT_E8_SAGE_BC.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
cd "$PAPER_DIR"

echo "==========================================================================="
echo " Phase E8 — SAGE + no-slip BC follow-up launcher | $(date)"
echo "==========================================================================="

# --- Pre-flight: sanity-check that data dirs + configs exist ---------------
required_dirs=(
  data/npz_cosserat_sweep
  data/npz_cosserat_sweep_withleak
  data/npz_cosserat_sweep_leak_dir_only
  data/npz_cosserat_sweep_leak_mag_only
  data/npz_subend
  data/npz_subend_withleak
  data/npz_subend_leak_dir_only
  data/npz_subend_leak_mag_only
)
for d in "${required_dirs[@]}"; do
  if [[ ! -d "$PAPER_DIR/$d" ]]; then
    echo "[launcher] ERROR: missing data dir $d" >&2
    exit 1
  fi
done

required_configs=(
  sage_cosserat_withleak sage_cosserat_leak_dir_only
  sage_cosserat_leak_mag_only sage_cosserat_noleak
  sage_subend_withleak sage_subend_leak_dir_only
  sage_subend_leak_mag_only sage_subend_noleak
  cosserat_sweep_withleak_nobc cosserat_sweep_leak_dir_only_nobc
  cosserat_sweep_leak_mag_only_nobc cosserat_sweep_noleak_nobc
)
for c in "${required_configs[@]}"; do
  if [[ ! -f "$PAPER_DIR/configs/${c}.yaml" ]]; then
    echo "[launcher] ERROR: missing config configs/${c}.yaml" >&2
    exit 1
  fi
done

mkdir -p "$PAPER_DIR/logs"

# --- STAGE 1: SAGE × Cosserat (12 tasks, %4) + its agg ---------------------
# Job 32 (agg) will trampoline-submit stage 2 (30 + 33) when it runs,
# and job 33 will in turn trampoline-submit stage 3 (31 + 34).
JOB29=$(sbatch --parsable jobs/29_train_sage_cosserat.sh)
echo "[launcher] submitted 29_train_sage_cosserat       jobid=$JOB29   (array 0-11%4)"

JOB32=$(sbatch --parsable --dependency=afterok:$JOB29 jobs/32_aggregate_sage_cosserat.sh)
echo "[launcher] submitted 32_aggregate_sage_cosserat   jobid=$JOB32   (afterok:$JOB29, trampolines→30+33)"

echo
echo "==========================================================================="
echo " Stage 1 submitted. Stages 2 and 3 will be auto-launched by the aggs."
echo "==========================================================================="
echo "  Stage 1: 29 (sage_cosserat)        jobid=$JOB29 → 32 (agg) jobid=$JOB32"
echo "  Stage 2: 30 (sage_subend)          submitted by 32 when it runs → 33 (agg)"
echo "  Stage 3: 31 (flowgat_nobc_cosserat) submitted by 33 when it runs → 34 (agg)"
echo
echo "Inspect queue:   squeue -u \$USER -o '%.10i %.30j %.2t %.10M %R'"
echo "Cancel stage 1:  scancel $JOB29 $JOB32"
echo "Cancel later stages once submitted: scancel them from squeue."
echo
echo "Outputs after completion:"
echo "  results/diagnostics/sage_cosserat/{per_case.csv,aggregate.csv,summary.md}"
echo "  results/diagnostics/sage_subend/{per_case.csv,aggregate.csv,summary.md}"
echo "  results/diagnostics/flowgat_nobc_cosserat/{per_case.csv,aggregate.csv,summary.md}"
echo "  results/per_seed/sage_cosserat_*_{val,test}_seed*.{csv,json}    (24 files)"
echo "  results/per_seed/sage_subend_*_{val,test}_seed*.{csv,json}      (24 files)"
echo "  results/per_seed/cosserat_sweep_*_nobc_*.{csv,json}             (24 files)"
