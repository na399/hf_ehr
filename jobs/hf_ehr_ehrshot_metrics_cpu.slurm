#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_mimiciv_ehrshot_metrics
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
MEDS_READER_DIR="${MEDS_READER_DIR:-/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader}"
EHRSHOT_ROOT="${EHRSHOT_ROOT:-${REMOTE_BASE}/ehrshot-mimiciv}"
EHRSHOT_ASSETS="${EHRSHOT_ASSETS:-${EHRSHOT_ROOT}/EHRSHOT_ASSETS}"
FEATURES_DIR="${FEATURES_DIR:-${EHRSHOT_ASSETS}/features_mimiciv}"
RESULTS_DIR="${RESULTS_DIR:-${EHRSHOT_ASSETS}/results_mimiciv}"
SUBJECT_SPLITS="${SUBJECT_SPLITS:-${MEDS_READER_DIR}/metadata/subject_splits.parquet}"
MODEL_NAME="${MODEL_NAME:-mimiciv_gpt_small_10ep_medsnorm_clmbr_512}"
LABEL_TASKS="${LABEL_TASKS:-long_los death}"
EMBED_STRAT="${EMBED_STRAT:-last_nonpad}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-200}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${RESULTS_DIR}" logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"

for label_task in ${LABEL_TASKS}; do
  test -s "${FEATURES_DIR}/${MODEL_NAME}_${label_task}_last_chunk:last_embed:${EMBED_STRAT}_features_1.pkl"
done
test -s "${SUBJECT_SPLITS}"

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

python hf_ehr/eval/ehrshot_mimiciv_probe_metrics.py \
  --path_to_features_dir "${FEATURES_DIR}" \
  --path_to_results_dir "${RESULTS_DIR}" \
  --path_to_subject_splits "${SUBJECT_SPLITS}" \
  --model_name "${MODEL_NAME}" \
  --embed_strat "${EMBED_STRAT}" \
  --tasks ${LABEL_TASKS} \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

echo "MIMIC-IV EHRSHOT-style probe metrics completed."
