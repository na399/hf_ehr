#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <job_id>" >&2
  exit 2
fi

LONGLEAF_HOST="${LONGLEAF_HOST:-longleaf}"
ssh "${LONGLEAF_HOST}" "scancel '$1'"
