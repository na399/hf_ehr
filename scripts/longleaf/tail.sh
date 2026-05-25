#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <job_id> [lines]" >&2
  exit 2
fi

JOB_ID="$1"
LINES="${2:-120}"
LONGLEAF_HOST="${LONGLEAF_HOST:-longleaf}"

if [[ -f ".longleaf/${JOB_ID}.env" ]]; then
  # shellcheck disable=SC1090
  source ".longleaf/${JOB_ID}.env"
fi

REMOTE_PROJECT="${REMOTE_PROJECT:-/nas/longleaf/home/na399/users/hf_ehr/src/hf_ehr}"
ssh "${LONGLEAF_HOST}" "cd '${REMOTE_PROJECT}' && for f in logs/*-${JOB_ID}.out logs/*-${JOB_ID}.err; do if [ -f \"\$f\" ]; then echo '===== '\$f; tail -n '${LINES}' \"\$f\"; fi; done"
