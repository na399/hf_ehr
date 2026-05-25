#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_mimiciv_tokenization_smoke
#SBATCH --partition=general
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
MEDS_READER_DIR="${MEDS_READER_DIR:-/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-${REMOTE_BASE}/cache/tokenizers/clmbr_mimiciv/tokenizer_config.json}"
TOKENIZER_CACHE_DIR="${TOKENIZER_CACHE_DIR:-${REMOTE_BASE}/cache/tokenizer_coverage/clmbr_mimiciv_medsnorm}"
TOKENIZER_SMOKE_MAX_SUBJECTS="${TOKENIZER_SMOKE_MAX_SUBJECTS:-1024}"
TOKENIZER_SMOKE_OUTPUT="${TOKENIZER_SMOKE_OUTPUT:-${REMOTE_BASE}/reports/mimiciv_tokenization_smoke_${SLURM_JOB_ID}.json}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${REMOTE_BASE}/cache" "${TOKENIZER_CACHE_DIR}" "$(dirname "${TOKENIZER_SMOKE_OUTPUT}")" logs
test -d "${MEDS_READER_DIR}"
test -s "${TOKENIZER_CONFIG}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Writing tokenization smoke report to ${TOKENIZER_SMOKE_OUTPUT}"

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

python hf_ehr/eval/mimiciv_tokenization_smoke.py \
  --path_to_meds_reader_extract "${MEDS_READER_DIR}" \
  --path_to_tokenizer_config "${TOKENIZER_CONFIG}" \
  --tokenizer_cache_dir "${TOKENIZER_CACHE_DIR}" \
  --max_subjects "${TOKENIZER_SMOKE_MAX_SUBJECTS}" \
  --output_json "${TOKENIZER_SMOKE_OUTPUT}"

test -s "${TOKENIZER_SMOKE_OUTPUT}"
echo "MIMIC-IV CLMBR tokenization smoke passed: ${TOKENIZER_SMOKE_OUTPUT}"
