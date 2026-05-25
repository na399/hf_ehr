#!/usr/bin/env bash
set -euo pipefail

LONGLEAF_HOST="${LONGLEAF_HOST:-longleaf}"
ONYEN="${ONYEN:-na399}"

if [[ $# -eq 0 ]]; then
  ssh "${LONGLEAF_HOST}" "squeue -u '${ONYEN}' -o '%.18i %.12P %.45j %.10T %.12M %.8D %R'"
  exit 0
fi

JOB_ID="$1"
ssh "${LONGLEAF_HOST}" "squeue -j '${JOB_ID}' -o '%.18i %.12P %.45j %.10T %.12M %.8D %R'; sacct -j '${JOB_ID}' --format=JobID,JobName%45,Partition,State,ExitCode,Elapsed,MaxRSS,ReqMem -P || true; seff '${JOB_ID}' || true"
