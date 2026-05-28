#!/bin/bash
# ---------------------------------------------------------------------------
# PHASE_E_SUBMIT.sh — drip-feed orchestrator for Phase E GPU work.
#
# The cluster enforces AssocMaxSubmitJobLimit (~5 pending+running array
# tasks at once). We therefore poll squeue and submit each job only when
# there is room. Inter-array dependencies (afterany) preserve ordering.
#
# Run on the login node (foreground; keep terminal open):
#   bash jobs/PHASE_E_SUBMIT.sh
# Or detach with nohup:
#   nohup bash jobs/PHASE_E_SUBMIT.sh > logs/phase_e_submit.log 2>&1 &
#
# Already-submitted jobs (e.g., from a partial earlier attempt) are
# detected by name; the orchestrator only submits jobs it hasn't yet
# launched. Idempotent — safe to re-run.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."

# Max array tasks (pending+running) the cluster lets us hold at once.
# Leave 1 slot of headroom below the hard limit.
MAX_QUEUED=4
POLL_SECONDS=60

# Job sequence: (script, depends-on-name-or-empty, job-name)
# Order is enforced by afterany dependencies so even if drip-feed catches
# multiple jobs at once they execute in this order.
JOBS=(
  "14_dump_existing.sh:paper_dump_existing"
  "11_train_local_tangent.sh:paper_train_centerline"
  "12_train_womersley.sh:paper_train_womersley"
  "13_train_sage.sh:paper_train_sage"
)

queued_tasks() {
  squeue -u "$USER" -h -t PD,R -r 2>/dev/null | wc -l
}

already_submitted_id() {
  # Return the most-recent jobid for an existing job-name, or empty.
  local jname=$1
  squeue -u "$USER" -h -n "$jname" -o "%i" 2>/dev/null | head -1
}

declare -A SUBMITTED
PREV_JID=""

for entry in "${JOBS[@]}"; do
  script="${entry%%:*}"
  jname="${entry##*:}"

  # Skip if already in queue (idempotency).
  existing=$(already_submitted_id "$jname")
  if [[ -n "$existing" ]]; then
    echo "[skip] $jname already in queue as $existing"
    PREV_JID=$(echo "$existing" | awk -F'_' '{print $1}')  # strip array suffix
    SUBMITTED[$jname]=$PREV_JID
    continue
  fi

  echo "[next] waiting to submit $script ($jname)..."
  while true; do
    q=$(queued_tasks)
    if (( q <= MAX_QUEUED )); then
      cmd=(sbatch --parsable)
      if [[ -n "$PREV_JID" ]]; then
        cmd+=(-d "afterany:$PREV_JID")
      fi
      cmd+=("jobs/$script")
      if jid=$("${cmd[@]}" 2>&1); then
        if [[ "$jid" =~ ^[0-9]+$ ]]; then
          echo "  submitted -> $jid  (queue had $q tasks)"
          SUBMITTED[$jname]=$jid
          PREV_JID=$jid
          break
        else
          echo "  sbatch returned non-numeric: $jid — retry in ${POLL_SECONDS}s"
        fi
      else
        echo "  sbatch failed: $jid — retry in ${POLL_SECONDS}s"
      fi
    else
      echo "  queue full ($q tasks) — sleeping ${POLL_SECONDS}s"
    fi
    sleep "$POLL_SECONDS"
  done
done

echo
echo "=== summary ==="
for k in "${!SUBMITTED[@]}"; do
  echo "  $k -> ${SUBMITTED[$k]}"
done
echo
echo "Monitor:  squeue -u \$USER -t PD,R"
echo "Logs:     ls -lt logs/"
