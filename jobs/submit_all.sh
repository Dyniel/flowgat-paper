#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh — submit all 6 (variant, seed) training jobs in parallel,
# then chain a dependency-locked analyze job at the end.
#
# Run from login node (NOT inside SLURM):
#   bash jobs/submit_all.sh
#
# Optional smoke test (3 epochs each, ~10 min total walltime):
#   SMOKE=1 bash jobs/submit_all.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
JOBS="$PAPER_DIR/jobs"
SMOKE="${SMOKE:-0}"

JOB_IDS=()
for VARIANT in withleak noleak; do
  for SEED in 1337 2026 777; do
    EXPORT="ALL,VARIANT=${VARIANT},SEED=${SEED}"
    [[ "$SMOKE" == "1" ]] && EXPORT="${EXPORT},SMOKE=1"
    NAME="paper_${VARIANT}_s${SEED}"
    JID=$(sbatch --parsable --job-name="$NAME" --export="$EXPORT" "$JOBS/01_train_seed.sh")
    echo "[submit_all] $NAME -> job $JID"
    JOB_IDS+=("$JID")
  done
done

DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
ANALYZE_JID=$(sbatch --parsable --dependency="afterany:${DEPS}" --job-name=paper_analyze "$JOBS/03_analyze.sh")
echo "[submit_all] analyze -> job $ANALYZE_JID (waits on ${#JOB_IDS[@]} train jobs: $DEPS)"

cat <<EOF

==========================================================================
Submitted ${#JOB_IDS[@]} training jobs + 1 analyze job.

Monitor:
  squeue -u \$USER

When all done, see:
  $PAPER_DIR/results/aggregated/summary.md
  $PAPER_DIR/results/manifest.json
  $PAPER_DIR/results/figures/

Logs:
  $PAPER_DIR/logs/
==========================================================================
EOF
