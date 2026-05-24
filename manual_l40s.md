# Single L40S Training Manual

This guide describes a data-agnostic workflow for training `hf_ehr` models on one NVIDIA L40S GPU. It assumes a MEDS dataset, `uv` for Python environments, Hydra configs, and PyTorch Lightning.

## Principles

- Start with an end-to-end smoke run before launching a full run.
- Use a MEDS reader database for training, not raw MEDS files directly.
- Keep run outputs, tokenizer caches, and temporary scripts separate per experiment.
- Prefer `bf16-mixed` on L40S.
- Use approximate token batching and tune `max_tokens` before changing the model.
- Keep checkpointing lean: one best validation checkpoint plus `last.ckpt`.
- Treat DataLoader worker settings as hardware and dataset dependent. If worker processes fail during validation, retry with `data.dataloader.n_workers=0`.

## Environment

Create and use a local `uv` virtual environment:

```bash
uv venv .venv
.venv/bin/python -m pip install -e .
```

Check CUDA visibility:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

The L40S supports bf16. Use:

```bash
trainer.precision=bf16-mixed
```

Use this allocator setting unless you have evidence another setting is better:

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
```

## Inputs

Use placeholders in scripts and docs:

```bash
MEDS_DIR=/path/to/meds
MEDS_READER_DIR=/path/to/meds_reader
TOKENIZER_CONFIG=/path/to/tokenizer_config.json
TOKENIZER_CACHE_DIR=/path/to/tokenizer_cache
OUTPUT_DIR=/path/to/run_output
```

Do not hard-code site, user, project, or dataset-specific paths in committed configs.

## Convert MEDS To MEDS Reader

Convert once per dataset:

```bash
.venv/bin/meds_reader_convert "$MEDS_DIR" "$MEDS_READER_DIR" --num_threads 8
```

Optionally verify:

```bash
.venv/bin/meds_reader_verify "$MEDS_DIR" "$MEDS_READER_DIR"
```

The training code expects a `subject_splits.parquet` under the reader metadata. Existing MEDS split labels may be `train`, `tuning`, and `held_out`; `MEDSDataset` maps those to train, val, and test.

## Tokenizer

Provide a tokenizer config explicitly:

```bash
data.tokenizer.path_to_config="$TOKENIZER_CONFIG"
```

Use a separate tokenizer cache for each dataset/tokenizer/debug-mode combination:

```bash
+data.tokenizer.cache_dir="$TOKENIZER_CACHE_DIR"
```

The first approximate-batching run creates sequence-length caches. Subsequent compatible runs should load them.

## Smoke Run

Run a short smoke before any full run. The goal is to prove:

- MEDS reader opens.
- Tokenizer loads.
- Sequence-length cache generation works.
- Forward, backward, validation, early stopping, and checkpoint saving work.
- GPU memory is stable.

Recommended smoke overrides:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=disabled \
HYDRA_FULL_ERROR=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
PYTHONPATH="$PWD" \
.venv/bin/python hf_ehr/scripts/run.py \
  +data=meds_dev \
  +trainer=single_gpu \
  +model=gpt2-base \
  +tokenizer=clmbr \
  model.config_kwargs.n_layer=6 \
  model.config_kwargs.n_head=8 \
  model.config_kwargs.n_embd=512 \
  model.is_gradient_checkpointing=True \
  data.dataset.path_to_meds_reader_extract="$MEDS_READER_DIR" \
  data.dataset.is_debug=True \
  data.tokenizer.path_to_config="$TOKENIZER_CONFIG" \
  +data.tokenizer.cache_dir="$TOKENIZER_CACHE_DIR" \
  data.dataloader.mode=approx \
  data.dataloader.approx_batch_sampler.max_tokens=256 \
  data.dataloader.max_length=256 \
  data.dataloader.n_workers=0 \
  data.dataloader.seq_length_n_procs=4 \
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
  logging.wandb.is_wandb=False \
  logging.mlflow.is_mlflow=False \
  main.path_to_output_dir="$OUTPUT_DIR"
```

Expected smoke evidence:

- Logs show model parameter count.
- Logs show train, val, and test dataset sizes.
- Logs show approximate batch counts.
- At least one `val/loss` is logged.
- Early stopping triggers if the forced smoke settings above are used.
- `ckpts/last.ckpt` exists.
- A best validation checkpoint exists if `val/loss` was logged.

## Full GPT Small Run

For a single L40S, start with GPT small and a 512-token context:

```bash
CUDA_VISIBLE_DEVICES=0 \
WANDB_MODE=online \
HYDRA_FULL_ERROR=1 \
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
PYTHONPATH="$PWD" \
.venv/bin/python hf_ehr/scripts/run.py \
  +data=meds_dev \
  +trainer=single_gpu \
  +model=gpt2-base \
  +tokenizer=clmbr \
  model.config_kwargs.n_layer=6 \
  model.config_kwargs.n_head=8 \
  model.config_kwargs.n_embd=512 \
  model.is_gradient_checkpointing=True \
  data.dataset.path_to_meds_reader_extract="$MEDS_READER_DIR" \
  data.dataset.is_debug=False \
  data.tokenizer.path_to_config="$TOKENIZER_CONFIG" \
  +data.tokenizer.cache_dir="$TOKENIZER_CACHE_DIR" \
  data.dataloader.mode=approx \
  data.dataloader.approx_batch_sampler.max_tokens=512 \
  data.dataloader.max_length=512 \
  data.dataloader.n_workers=0 \
  data.dataloader.seq_length_n_procs=16 \
  data.dataloader.seq_length_chunk_size=5000 \
  data.dataloader.precompute_splits=[val] \
  trainer.devices=[0] \
  trainer.distributed_backend=auto \
  trainer.precision=bf16-mixed \
  trainer.max_epochs=10 \
  trainer.min_epochs=1 \
  trainer.limit_train_batches=null \
  trainer.limit_val_batches=1 \
  trainer.val_check_interval=1.0 \
  trainer.check_val_every_n_epoch=1 \
  callbacks.early_stopping.monitor=val/loss \
  callbacks.early_stopping.metric_mode=min \
  callbacks.early_stopping.patience=3 \
  callbacks.early_stopping.min_delta=0.0 \
  callbacks.model_checkpointing.save_top_k_val_loss=1 \
  callbacks.model_checkpointing.save_most_recent_k=0 \
  callbacks.model_checkpointing.save_most_recent_every_n_train_steps=10000 \
  callbacks.model_checkpointing.save_last_checkpoint=True \
  callbacks.model_checkpointing.save_start_checkpoint=False \
  callbacks.model_checkpointing.save_epoch_checkpoint=False \
  callbacks.model_checkpointing.save_step_checkpoint=False \
  callbacks.model_checkpointing.every_n_train_nonPAD_tokens=null \
  logging.wandb.is_wandb=True \
  logging.mlflow.is_mlflow=False \
  main.path_to_output_dir="$OUTPUT_DIR"
```

With `trainer.accumulate_grad_batches="__PLACEHOLDER__"`, the code derives accumulation from:

```text
trainer.target_tokens_per_update / data.dataloader.approx_batch_sampler.max_tokens
```

For the default `target_tokens_per_update=65_536` and `max_tokens=512`, this becomes `128`.

## Running As A User Service

For long runs, use `systemd-run --user` so the job survives terminal disconnects:

```bash
systemd-run --user \
  --unit=<unit_name> \
  --description='<description>' \
  --working-directory="$PWD" \
  --setenv=CUDA_VISIBLE_DEVICES=0 \
  --setenv=WANDB_MODE=online \
  --setenv=HYDRA_FULL_ERROR=1 \
  --setenv=PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 \
  --setenv=PYTHONPATH="$PWD" \
  .venv/bin/python hf_ehr/scripts/run.py <hydra_overrides>
```

Monitor:

```bash
systemctl --user status <unit_name> --no-pager
journalctl --user -u <unit_name> -n 160 --no-pager
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
```

Stop:

```bash
systemctl --user stop <unit_name>
```

## L40S Memory Knobs

Start here:

- `trainer.precision=bf16-mixed`
- `model.is_gradient_checkpointing=True`
- `data.dataloader.max_length=512`
- `data.dataloader.approx_batch_sampler.max_tokens=512`
- GPT small dimensions: 6 layers, 8 heads, 512 hidden

If CUDA OOM occurs:

1. Lower `data.dataloader.approx_batch_sampler.max_tokens`.
2. Lower `data.dataloader.max_length`.
3. Keep `trainer.target_tokens_per_update` fixed and let accumulation increase.
4. Keep gradient checkpointing enabled.
5. Only then reduce model dimensions.

For GPT causal LM, the vocabulary projection and cross-entropy can dominate memory. Smaller hidden size and shorter context both help.

## Sequence-Length Cache Notes

Approximate batching needs patient sequence lengths. The fast path can count MEDS parquet rows for code-only tokenizers and writes `seq_length_per_patient.json` under the tokenizer cache.

Use:

```bash
data.dataloader.seq_length_n_procs=16
data.dataloader.seq_length_chunk_size=5000
data.dataloader.precompute_splits=[val]
```

`precompute_splits=[val]` avoids building test cache during training. Add `test` only before explicit test/eval runs.

Peak CPU RAM can be high while building full-dataset length caches. If this is a problem, reduce `seq_length_n_procs` or use a smaller first full-run cache job.

## DataLoader Workers

Use `data.dataloader.n_workers=0` if worker processes fail during CUDA validation or if the MEDS reader is not fork-safe in the current environment.

Once a smoke and full startup are stable, try:

```bash
data.dataloader.n_workers=2
```

Then compare throughput and stability. Do not assume more workers is better.

## Loss Expectations

GPT uses causal next-token cross-entropy. For a vocabulary of size `V`, a random baseline is approximately:

```text
ln(V)
```

Examples:

- `V=32,000` gives baseline `~10.37`
- `V=64,000` gives baseline `~11.07`

The logged `ppl` is clamped at 100, so it is not useful until loss is below `ln(100) ~= 4.61`.

Very low losses such as `1.3` or lower are unusual for a normal held-out clinical-code next-token LM and should trigger leakage or data-construction checks.

## Checkpoints

Recommended full-run checkpoint policy:

```bash
callbacks.model_checkpointing.save_top_k_val_loss=1
callbacks.model_checkpointing.save_last_checkpoint=True
callbacks.model_checkpointing.save_most_recent_k=0
callbacks.model_checkpointing.save_start_checkpoint=False
callbacks.model_checkpointing.save_epoch_checkpoint=False
callbacks.model_checkpointing.save_step_checkpoint=False
callbacks.model_checkpointing.every_n_train_nonPAD_tokens=null
```

This keeps storage bounded while preserving resumability.

## Before Calling A Run Healthy

Confirm:

- W&B run exists if enabled.
- `model_parameter_count` is expected.
- `train/loss` logs after real training starts.
- `val/loss` logs at epoch end.
- `ckpts/last.ckpt` exists after the first checkpoint event.
- GPU memory is stable for multiple steps past the first few batches.
- Output directory and tokenizer cache are separate from prior experiments.
