#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <job_id> [remote_run_dir]" >&2
  exit 2
fi

JOB_ID="$1"
REMOTE_RUN_DIR="${2:-}"
LONGLEAF_HOST="${LONGLEAF_HOST:-longleaf}"

if [[ -f ".longleaf/${JOB_ID}.env" ]]; then
  # shellcheck disable=SC1090
  source ".longleaf/${JOB_ID}.env"
fi

REMOTE_PROJECT="${REMOTE_PROJECT:-/nas/longleaf/home/na399/users/hf_ehr/src/hf_ehr}"
DEST="temp/longleaf/${JOB_ID}"
mkdir -p "${DEST}"

rsync -az --include="*-${JOB_ID}.out" --include="*-${JOB_ID}.err" --exclude="*" \
  "${LONGLEAF_HOST}:${REMOTE_PROJECT}/logs/" "${DEST}/logs/"

if [[ -n "${REMOTE_RUN_DIR}" ]]; then
  mkdir -p "${DEST}/run"
  rsync -az \
    --include="logs/" \
    --include="logs/info.log" \
    --include="logs/artifacts/" \
    --include="logs/artifacts/config.yaml" \
    --include="ckpts/" \
    --include="ckpts/*.ckpt.index" \
    --exclude="*.ckpt" \
    --exclude="*" \
    "${LONGLEAF_HOST}:${REMOTE_RUN_DIR%/}/" "${DEST}/run/"
  ssh "${LONGLEAF_HOST}" "find '${REMOTE_RUN_DIR%/}' -maxdepth 3 -type f | sed 's#^#/#' | head -n 200" > "${DEST}/remote_manifest.txt"
fi

echo "Pulled compact evidence to ${DEST}"
