#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_ehrshot_smoke_l40s
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
MEDS_READER_DIR="${MEDS_READER_DIR:-/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader}"
EHRSHOT_ROOT="${EHRSHOT_ROOT:-${REMOTE_BASE}/ehrshot-mimiciv}"
EHRSHOT_ASSETS="${EHRSHOT_ASSETS:-${EHRSHOT_ROOT}/EHRSHOT_ASSETS}"
LABELS_DIR="${LABELS_DIR:-${EHRSHOT_ASSETS}/benchmark_mimiciv}"
LABEL_TASK="${LABEL_TASK:-long_los}"
PATH_TO_CKPT="${PATH_TO_CKPT:-${REMOTE_BASE}/runs/mimiciv_gpt_small_10ep_medsnorm/ckpts/last.ckpt}"
MODEL_NAME="${MODEL_NAME:-mimiciv_gpt_small_10ep_medsnorm_clmbr_512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-cuda:0}"
EMBED_STRAT="${EMBED_STRAT:-last_nonpad}"
PADDING_SIDE="${PADDING_SIDE:-right}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${EHRSHOT_ASSETS}/features_mimiciv" "${EHRSHOT_ASSETS}/tokenized_timelines_mimiciv" "${EHRSHOT_ASSETS}/models" logs

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi || true

test -s "${PATH_TO_CKPT}"
test -d "${MEDS_READER_DIR}"
test -s "${LABELS_DIR}/${LABEL_TASK}/all_labels.csv"

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

python hf_ehr/eval/ehrshot.py \
  --database_backend meds_reader \
  --path_to_database "${MEDS_READER_DIR}" \
  --path_to_labels_dir "${LABELS_DIR}/${LABEL_TASK}" \
  --path_to_features_dir "${EHRSHOT_ASSETS}/features_mimiciv" \
  --path_to_tokenized_timelines_dir "${EHRSHOT_ASSETS}/tokenized_timelines_mimiciv" \
  --path_to_model "${PATH_TO_CKPT}" \
  --model_name "${MODEL_NAME}_${LABEL_TASK}" \
  --batch_size "${BATCH_SIZE}" \
  --embed_strat "${EMBED_STRAT}" \
  --chunk_strat last \
  --device "${DEVICE}" \
  --add_special_tokens \
  --padding_side "${PADDING_SIDE}" \
  --patient_idx_start 0 \
  --patient_idx_end 256

echo "EHRSHOT feature smoke completed."
