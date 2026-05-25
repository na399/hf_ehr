#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 jobs/<job>.slurm" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

JOB_FILE="$1"
test -f "${JOB_FILE}" || { echo "Missing job file: ${JOB_FILE}" >&2; exit 2; }

ONYEN="${ONYEN:-na399}"
LONGLEAF_HOST="${LONGLEAF_HOST:-longleaf}"
PROJECT_NAME="${PROJECT_NAME:-hf_ehr}"
REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/${ONYEN}/users/hf_ehr}"
REMOTE_PROJECT="${REMOTE_PROJECT:-${REMOTE_BASE}/src/${PROJECT_NAME}}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/${PROJECT_NAME}}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

mkdir -p .longleaf

ssh "${LONGLEAF_HOST}" "mkdir -p '${REMOTE_PROJECT}' '${REMOTE_BASE}/cache' '${REMOTE_BASE}/runs' '${REMOTE_BASE}/logs'"

rsync -az \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.longleaf/' \
  --exclude 'cache/' \
  --exclude 'logs/' \
  --exclude 'runs/' \
  --exclude 'slurm_logs/' \
  --exclude 'temp/' \
  --exclude 'wandb/' \
  --exclude 'hf_ehr/notebooks/' \
  ./ "${LONGLEAF_HOST}:${REMOTE_PROJECT}/"

EXPORTS="ALL,REMOTE_BASE=${REMOTE_BASE},UV_SHARED_VENV=${UV_SHARED_VENV},UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT},UV_CACHE_DIR=${UV_CACHE_DIR},HF_HOME=${HF_HOME},TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE},SENTENCE_TRANSFORMERS_HOME=${SENTENCE_TRANSFORMERS_HOME}"

append_optional_export() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    EXPORTS="${EXPORTS},${name}=${!name}"
  fi
}

for name in \
  MEDS_READER_DIR TOKENIZER_CONFIG TOKENIZER_CACHE_DIR OUTPUT_DIR \
  TOKENIZER_SMOKE_MAX_SUBJECTS TOKENIZER_SMOKE_OUTPUT \
  EHRSHOT_ROOT EHRSHOT_ASSETS LABELS_DIR LABEL_TASK LABEL_TASKS \
  PATH_TO_CKPT MODEL_NAME BATCH_SIZE DEVICE EMBED_STRAT PADDING_SIDE \
  BOOTSTRAP_SAMPLES SUBJECT_SPLITS MIN_PRIOR_HISTORY_DAYS; do
  append_optional_export "${name}"
done

SUBMIT_OUTPUT="$(ssh "${LONGLEAF_HOST}" "cd '${REMOTE_PROJECT}' && mkdir -p logs && sbatch --export='${EXPORTS}' '${JOB_FILE}'")"
echo "${SUBMIT_OUTPUT}"

JOB_ID="$(printf '%s\n' "${SUBMIT_OUTPUT}" | awk '/Submitted batch job/ {print $4}')"
if [[ -z "${JOB_ID}" ]]; then
  echo "Could not parse job id from sbatch output" >&2
  exit 1
fi

cat > ".longleaf/${JOB_ID}.env" <<EOF
JOB_ID=${JOB_ID}
JOB_FILE=${JOB_FILE}
LONGLEAF_HOST=${LONGLEAF_HOST}
REMOTE_BASE=${REMOTE_BASE}
REMOTE_PROJECT=${REMOTE_PROJECT}
EOF

echo "Wrote .longleaf/${JOB_ID}.env"
