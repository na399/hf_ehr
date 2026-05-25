#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_mimiciv_ehrshot_labels
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
MEDS_READER_DIR="${MEDS_READER_DIR:-/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader}"
EHRSHOT_ROOT="${EHRSHOT_ROOT:-${REMOTE_BASE}/ehrshot-mimiciv}"
EHRSHOT_ASSETS="${EHRSHOT_ASSETS:-${EHRSHOT_ROOT}/EHRSHOT_ASSETS}"
LABELS_DIR="${LABELS_DIR:-${EHRSHOT_ASSETS}/benchmark_mimiciv}"
MIN_PRIOR_HISTORY_DAYS="${MIN_PRIOR_HISTORY_DAYS:-730}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${LABELS_DIR}" logs
test -d "${MEDS_READER_DIR}"

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

python hf_ehr/eval/generate_mimiciv_ehrshot_labels.py \
  --path_to_meds_reader_extract "${MEDS_READER_DIR}" \
  --path_to_labels_dir "${LABELS_DIR}" \
  --min_prior_history_days "${MIN_PRIOR_HISTORY_DAYS}"

test -s "${LABELS_DIR}/death/all_labels.csv"
test -s "${LABELS_DIR}/long_los/all_labels.csv"
echo "MIMIC-IV EHRSHOT-style labels ready under ${LABELS_DIR}"
