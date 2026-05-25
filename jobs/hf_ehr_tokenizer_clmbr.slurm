#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_tokenizer_clmbr
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${REMOTE_BASE}/cache/tokenizers/clmbr_mimiciv}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-/nas/longleaf/home/na399/users/hf_ehr/cache/tokenizers/clmbr_mimiciv/tokenizer_config.json}"
ORIG_CLMBR_JSON="${ORIG_CLMBR_JSON:-${REMOTE_BASE}/cache/source/clmbr_v8_original_dictionary.json}"

export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${REMOTE_BASE}/cache" "${TOKENIZER_DIR}" logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
echo "Writing tokenizer to ${TOKENIZER_CONFIG}"

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

CONFIG_PATH="${REMOTE_BASE}/configs/clmbr_mimiciv.yaml"
mkdir -p "$(dirname "${CONFIG_PATH}")"
cat > "${CONFIG_PATH}" <<EOF
data:
  tokenizer:
    name: CLMBRTokenizer
    path_to_config: ${TOKENIZER_CONFIG}
EOF

if [[ -s "${ORIG_CLMBR_JSON}" ]]; then
  python hf_ehr/tokenizers/create_clmbr.py \
    --path_to_tokenizer_config "${CONFIG_PATH}" \
    --path_to_orig_clmbr_json "${ORIG_CLMBR_JSON}"
else
  python hf_ehr/tokenizers/create_clmbr.py --path_to_tokenizer_config "${CONFIG_PATH}"
fi
test -s "${TOKENIZER_CONFIG}"
echo "Tokenizer config ready: ${TOKENIZER_CONFIG}"
