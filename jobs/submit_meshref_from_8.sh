#!/bin/bash

JOB_SCRIPT="/users/scratch1/dancies/flowgat_paper/jobs/21_mesh_refinement_eval.sh"
JOB_NAME="paper_meshref_eval"
MAX_ACTIVE=4

for IDX in $(seq 8 35); do
  while true; do
    ACTIVE=$(squeue -u "$USER" -h -n "$JOB_NAME" | wc -l)

    if [ "$ACTIVE" -lt "$MAX_ACTIVE" ]; then
      break
    fi

    echo "[submitter] active=$ACTIVE >= $MAX_ACTIVE; waiting..."
    sleep 60
  done

  echo "[submitter] submitting IDX=$IDX"
  sbatch --qos=short --array="$IDX" "$JOB_SCRIPT"
  sleep 3
done

echo "[submitter] submitted IDX 8..35"
