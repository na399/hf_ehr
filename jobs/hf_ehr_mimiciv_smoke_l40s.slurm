#!/usr/bin/env bash
#SBATCH --job-name=hf_ehr_mimiciv_smoke_l40s
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REMOTE_BASE="${REMOTE_BASE:-/nas/longleaf/home/na399/users/hf_ehr}"
MEDS_READER_DIR="${MEDS_READER_DIR:-/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader}"
TOKENIZER_CONFIG="${TOKENIZER_CONFIG:-${REMOTE_BASE}/cache/tokenizers/clmbr_mimiciv/tokenizer_config.json}"
TOKENIZER_CACHE_DIR="${TOKENIZER_CACHE_DIR:-${REMOTE_BASE}/cache/tokenizer_seq_lengths/mimiciv_smoke_clmbr_medsnorm}"
OUTPUT_DIR="${OUTPUT_DIR:-${REMOTE_BASE}/runs/mimiciv_gpt_small_smoke_${SLURM_JOB_ID}}"
UV_SHARED_VENV="${UV_SHARED_VENV:-${REMOTE_BASE}/.venvs/hf_ehr}"
UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${UV_SHARED_VENV}}"
UV_CACHE_DIR="${UV_CACHE_DIR:-${REMOTE_BASE}/cache/uv}"
HF_HOME="${HF_HOME:-${REMOTE_BASE}/cache/huggingface}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-${HF_HOME}/sentence-transformers}"

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=online
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export PYTHONPATH="$PWD"
export UV_PROJECT_ENVIRONMENT UV_CACHE_DIR HF_HOME TRANSFORMERS_CACHE SENTENCE_TRANSFORMERS_HOME

mkdir -p "${REMOTE_BASE}/cache" "${TOKENIZER_CACHE_DIR}" "${OUTPUT_DIR}" logs
test -d "${MEDS_READER_DIR}"
test -s "${TOKENIZER_CONFIG}"

echo "Job ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi || true

if command -v uv >/dev/null 2>&1; then
  mkdir -p "$(dirname "${UV_SHARED_VENV}")" "${UV_CACHE_DIR}"
  flock "${UV_SHARED_VENV}.lock" uv venv "${UV_SHARED_VENV}" --python 3.10
  flock "${UV_SHARED_VENV}.lock" uv pip install --python "${UV_SHARED_VENV}/bin/python" -e .
else
  python3 -m venv "${UV_SHARED_VENV}"
  "${UV_SHARED_VENV}/bin/python" -m pip install -e .
fi

source "${UV_SHARED_VENV}/bin/activate"

python hf_ehr/scripts/run.py \
  +data=meds_dev \
  +trainer=single_gpu \
  +model=gpt2-base \
  +tokenizer=clmbr \
  model.config_kwargs.n_layer=6 \
  model.config_kwargs.n_head=8 \
  model.config_kwargs.n_embd=512 \
  model.is_gradient_checkpointing=True \
  data.dataset.path_to_meds_reader_extract="${MEDS_READER_DIR}" \
  data.dataset.is_debug=True \
  data.tokenizer.path_to_config="${TOKENIZER_CONFIG}" \
  +data.tokenizer.cache_dir="${TOKENIZER_CACHE_DIR}" \
  data.dataloader.mode=approx \
  data.dataloader.approx_batch_sampler.max_tokens=256 \
  data.dataloader.max_length=256 \
  data.dataloader.n_workers=0 \
  data.dataloader.seq_length_n_procs=1 \
  data.dataloader.seq_length_chunk_size=250 \
  data.dataloader.precompute_splits=[val] \
  trainer.devices=[0] \
  trainer.distributed_backend=auto \
  trainer.precision=bf16-mixed \
  trainer.accumulate_grad_batches=1 \
  trainer.max_epochs=4 \
  trainer.min_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=1 \
  callbacks.early_stopping.monitor=val/loss \
  callbacks.early_stopping.metric_mode=min \
  callbacks.early_stopping.patience=0 \
  callbacks.early_stopping.min_delta=1000000 \
  callbacks.model_checkpointing.save_top_k_val_loss=1 \
  callbacks.model_checkpointing.save_most_recent_k=1 \
  callbacks.model_checkpointing.save_most_recent_every_n_train_steps=1 \
  callbacks.model_checkpointing.save_last_checkpoint=True \
  callbacks.model_checkpointing.save_start_checkpoint=False \
  callbacks.model_checkpointing.save_epoch_checkpoint=False \
  callbacks.model_checkpointing.save_step_checkpoint=False \
  logging.wandb.is_wandb=True \
  logging.wandb.entity=na399-ai \
  logging.wandb.project=hf_ehr_longleaf \
  logging.mlflow.is_mlflow=False \
  main.path_to_output_dir="${OUTPUT_DIR}"
