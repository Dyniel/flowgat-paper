#!/bin/bash
# ---------------------------------------------------------------------------
# retrain_noleak.sh — submit only the 3 noleak seeds with extended walltime.
#
# Use this if the initial run hit SLURM time limit (8h was insufficient for
# noleak — model still slowly improving at ep ~1140/1500). 14h walltime gives
# ~2× margin.
#
# Withleak is NOT retrained (already self-stopped, results stable).
#
# Usage from login node:
#   bash jobs/retrain_noleak.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PAPER_DIR="/users/scratch1/$USER/flowgat_paper"
JOBS="$PAPER_DIR/jobs"

# Backup old noleak results before retrain so we have provenance
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$PAPER_DIR/results/_archive_noleak_${TS}"
mkdir -p "$BACKUP_DIR"
echo "[retrain] backing up old noleak artifacts to $BACKUP_DIR"
[ -d "$PAPER_DIR/results/checkpoints/noleak" ] && \
  cp -r "$PAPER_DIR/results/checkpoints/noleak" "$BACKUP_DIR/checkpoints_noleak"
for f in "$PAPER_DIR/results/per_seed/noleak_"*; do
  [ -f "$f" ] && cp "$f" "$BACKUP_DIR/"
done

# Clear old noleak checkpoints to force fresh run
echo "[retrain] clearing old noleak checkpoints"
rm -rf "$PAPER_DIR/results/checkpoints/noleak"
rm -f  "$PAPER_DIR/results/per_seed/noleak_"*

JOB_IDS=()
for SEED in 1337 2026 777; do
  EXPORT="ALL,VARIANT=noleak,SEED=${SEED}"
  NAME="paper_noleak_s${SEED}"
  JID=$(sbatch --parsable --job-name="$NAME" --export="$EXPORT" "$JOBS/01_train_seed.sh")
  echo "[retrain] $NAME -> job $JID"
  JOB_IDS+=("$JID")
done

DEPS=$(IFS=:; echo "${JOB_IDS[*]}")
ANALYZE_JID=$(sbatch --parsable --dependency="afterany:${DEPS}" --job-name=paper_analyze "$JOBS/03_analyze.sh")
echo "[retrain] analyze -> job $ANALYZE_JID (waits on ${#JOB_IDS[@]} jobs: $DEPS)"

cat <<EOF

==========================================================================
Retraining noleak: ${#JOB_IDS[@]} jobs (each 14h walltime, ~10-12h actual).
Withleak left intact (self-stopped, 3-seed mean PP@10=0.231).

Old noleak artifacts archived in:
  $BACKUP_DIR

Monitor:
  squeue -u \$USER

When done, the analyze job will refresh:
  $PAPER_DIR/results/aggregated/summary.md
  $PAPER_DIR/results/figures/
==========================================================================
EOF
