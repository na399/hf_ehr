# MIMIC-IV EHRSHOT Lessons Learned

Date: 2026-05-25

This note summarizes the MIMIC-IV EHRSHOT debugging pass for the Longleaf
GPT-small run and records the checks that should happen before trusting future
training or evaluation results. It is intentionally PHI-free: keep raw MIMIC-IV
data, generated labels, tokenized timelines, and feature payloads on Longleaf.

## Bottom Line

The bad EHRSHOT metrics were not just a weak downstream classifier. The first
feature artifacts were degenerate because the MIMIC-IV MEDS reader emitted codes
like `LOINC//3023314` and `RxNorm//46287424//end`, while the CLMBR tokenizer
configuration used single-slash vocabulary codes like `LOINC/3023314` and
`RxNorm/46287424`.

That mismatch made evaluation timelines all PAD, and likely made the original
GPT-small training run mostly learn from special/PAD tokens instead of clinical
events. Patching evaluation can make old checkpoints run with non-empty tokens,
but it cannot create clinical signal that was never learned during training.

## Incident Evidence

The diagnostic job on the existing EHRSHOT artifacts found:

- Labels existed for both tasks: `59,352` labels per task across `45,039`
  patients.
- Raw MEDS histories were non-empty. In a 1,000-label sample, raw event counts
  had a median around `420` and a max around `33,797`.
- Tokenized timelines were empty: token counts were all zero and the combined
  timeline arrays had shape `[59352, 512]` with every row ending in `[PAD]`.
- Frozen feature rows were effectively identical: the feature matrix had only
  two rounded unique rows and sample cosine similarity to the first row was
  approximately `1.0`.
- EHRSHOT-style metrics on those old features were near random:
  `long_los` AUROC `0.500000`, `death` AUROC `0.500077`.

That pattern is a tokenizer/data-interface failure, not a subtle modeling issue.

## What We Missed

1. Tokenizer compatibility was assumed after the CLMBR config loaded.
   Loading a tokenizer config is not enough. We must prove that raw dataset
   codes produce non-special token IDs before training or feature generation.

2. Sequence-length precompute could look mechanically successful while counting
   the wrong code namespace.
   The fast MEDS parquet path counted tokenizer vocabulary codes directly, so it
   missed double-slash MEDS reader variants until normalization was added.

3. Completed SLURM jobs were treated as stronger evidence than artifact content.
   `COMPLETED (0:0)` only proves the process exited cleanly. It does not prove
   token coverage, feature diversity, or meaningful labels.

4. The evaluation cache key was too weak.
   Tokenized timeline caches did not include label CSV content, padding mode, or
   special-token settings, so reruns could reuse stale artifacts after label or
   tokenization changes.

5. Evaluation did not initially follow the available benchmark split contract.
   MIMIC MEDS already has `metadata/subject_splits.parquet`; metrics should use
   train/tuning/held-out splits instead of ad hoc patient hashing.

6. Label generation missed outcome-task eligibility details.
   The MIMIC-style outcome labels should enforce at least 730 days of prior
   observed history before the prediction time.

7. Causal LM labels included PAD positions.
   The collator used `input_ids` directly as causal labels. PAD positions should
   be masked to `-100` so they do not contribute to the loss.

## Fixes Added

- Added MEDS-to-CLMBR code normalization:
  - `VOCAB//concept` can resolve to `VOCAB/concept`.
  - interval boundary codes like `VOCAB//concept//start` and
    `VOCAB//concept//end` can resolve to the base CLMBR code.
- Versioned the CLMBR tokenizer metadata with `meds_code_normalization` so stale
  sequence-length caches are invalidated.
- Updated the fast MEDS sequence-length path to count raw double-slash variants.
- Added `last_nonpad`, right-padding, and `--add_special_tokens` support for
  EHRSHOT feature extraction.
- Added label-file digests and tokenization settings to the EHRSHOT timeline
  cache key.
- Added safe rerun behavior for copied model directories instead of silently
  deleting existing outputs.
- Updated probe metrics to use MEDS subject splits and EHRSHOT-style
  `MaxAbsScaler` plus validation-tuned logistic regression.
- Updated MIMIC-IV label prep to enforce `--min_prior_history_days 730` by
  default.
- Masked PAD labels in causal LM training.
- Added Longleaf wrapper support for safe job-specific overrides such as
  `PATH_TO_CKPT`, `MODEL_NAME`, `OUTPUT_DIR`, and `TOKENIZER_CACHE_DIR`.

## Current Interpretation Of Existing Checkpoints

The old checkpoint at:

```text
/nas/longleaf/home/na399/users/hf_ehr/runs/mimiciv_gpt_small_10ep/ckpts/last.ckpt
```

can be evaluated with the patched code by overriding:

```bash
PATH_TO_CKPT=/nas/longleaf/home/na399/users/hf_ehr/runs/mimiciv_gpt_small_10ep/ckpts/last.ckpt \
MODEL_NAME=mimiciv_gpt_small_10ep_clmbr_512_patched \
./scripts/longleaf/submit.sh jobs/hf_ehr_ehrshot_smoke_l40s.slurm
```

This is useful as a salvage test, not proof that the checkpoint is valid. If the
training run used the broken tokenizer path, the model weights likely did not
learn useful clinical-token representations. The patched eval smoke should be
judged by token coverage and feature diversity before metrics.

## Required Checks Before Future Full Runs

Run these checks before submitting a long 10-epoch job or a full EHRSHOT feature
generation job.

### Tokenization Smoke

Verify on Longleaf, inside a SLURM job:

- MEDS reader opens.
- CLMBR tokenizer loads.
- A sample of raw patient timelines has nonzero raw event counts.
- The same sample has nonzero non-special/non-PAD token counts.
- Common raw codes are mapped to tokenizer IDs.
- Tokenized timelines have more than one unique row.
- `[PAD]` is not the only observed token ID.

Use the repo-local CPU smoke before any GPU training:

```bash
./scripts/longleaf/submit.sh jobs/hf_ehr_mimiciv_tokenization_smoke_cpu.slurm
```

### Training Smoke

Verify the smoke job evidence before full training:

- Approximate batching reports nonzero token counts.
- Forward and backward passes run.
- Validation runs.
- W&B initializes under `na399-ai/hf_ehr_longleaf` when requested.
- Early stopping callback is wired.
- `ckpts/last.ckpt` is written.
- A best validation checkpoint is written when configured.
- No CUDA OOM or traceback appears.

### EHRSHOT Smoke

Verify before full EHRSHOT extraction:

- Generated labels load for each task.
- Subject splits are read from `metadata/subject_splits.parquet`.
- MEDS reader opens.
- Checkpoint, tokenizer, and model load.
- Tokenized timelines are written for the smoke slice.
- Feature pickle is written.
- Tokenized smoke rows have nonzero lengths.
- Feature rows have meaningful diversity.
- No traceback or CUDA OOM appears.

### Metrics

Do not report metrics until the artifact diagnostics pass. At minimum, include:

- Label counts and positive prevalence by train/tuning/held-out split.
- Tokenized timeline shape and non-PAD length summary.
- Unique token ID count.
- Feature matrix shape and row-diversity summary.
- AUROC, AUPRC, Brier, and a dummy-prior baseline for each task.

## Longleaf Practice

- Use only repo-local wrappers for control-plane actions:
  `scripts/longleaf/submit.sh`, `status.sh`, `tail.sh`, and `cancel.sh`.
- Do not run training, feature extraction, or data preparation on the login node.
- Keep MIMIC-IV data and restricted artifacts on Longleaf.
- Pull only compact logs, metrics JSON, and summaries.
- Use distinct output names for repaired runs, for example
  `mimiciv_gpt_small_10ep_medsnorm`, so old broken checkpoints are not silently
  overwritten or resumed.

## Practical Rule

Before trusting any EHR foundation model result, prove this chain:

```text
raw MEDS events -> mapped tokenizer tokens -> non-empty batches -> model learns/evaluates -> diverse features -> split-correct metrics
```

If any link is missing, a clean SLURM exit code is not enough.
