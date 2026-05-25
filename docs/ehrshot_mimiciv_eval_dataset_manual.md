# MIMIC-IV EHRSHOT-Style Eval Dataset Manual

This manual describes how to prepare the local evaluation labels used by
`hf_ehr/eval/ehrshot.py` when evaluating a trained `hf_ehr` checkpoint on the
MIMIC-IV MEDS reader extract. It does not require the credentialed Stanford
EHRSHOT asset bundle. It creates EHRSHOT-style label CSVs directly from the
MIMIC-IV MEDS data.

## Inputs

The preparation step expects a MEDS reader extract with subject timelines. Set
these paths for your environment:

```bash
export PATH_BASE=<project-root>
export MEDS_READER_DIR=<meds-reader-extract>
export EHRSHOT_ROOT="${PATH_BASE}/ehrshot-mimiciv"
export EHRSHOT_ASSETS="${EHRSHOT_ROOT}/EHRSHOT_ASSETS"
export LABELS_DIR="${EHRSHOT_ASSETS}/benchmark_mimiciv"
```

Do not copy PHI or restricted source data out of the environment where it is
approved to live.

## Outputs

Labels are written under the project EHRSHOT-style workspace:

```text
${LABELS_DIR}/
```

The generator writes one `all_labels.csv` per task:

```text
benchmark_mimiciv/
  death/
    all_labels.csv
  long_los/
    all_labels.csv
```

Each CSV uses the format consumed by `femr.labelers.load_labeled_patients`:

```csv
patient_id,prediction_time,value,label_type
10000719,2140-04-17T00:14,false,boolean
```

## Label Definitions

The script is `hf_ehr/eval/generate_mimiciv_ehrshot_labels.py`.

It scans each subject timeline and identifies admission intervals from these
MEDS visit interval codes:

```text
Visit//9201
Visit//262
```

For each valid admission interval:

- `prediction_time` is admission start plus 48 hours.
- Admissions ending before that prediction time are skipped.
- Admissions where death already occurred before prediction time are skipped.
- Admissions without at least 730 days of prior observed history before
  prediction time are skipped by default.
- `death` is `true` if `MEDS_DEATH` occurs during the admission interval.
- `long_los` is `true` if admission duration is greater than 7 days.

The default tasks are:

```text
death long_los
```

## Local Dry Run

Use a tiny subject cap only for syntax and schema checks. Do this only against a
local non-restricted toy extract, or inside the approved compute environment if
the extract is restricted.

```bash
uv run python hf_ehr/eval/generate_mimiciv_ehrshot_labels.py \
  --path_to_meds_reader_extract "${MEDS_READER_DIR}" \
  --path_to_labels_dir temp/ehrshot-labels \
  --max_subjects 100
```

Expected files:

```text
temp/ehrshot-labels/death/all_labels.csv
temp/ehrshot-labels/long_los/all_labels.csv
```

## Prepare Labels

Run the label generator in the approved environment that can access the MEDS
reader extract:

```bash
uv run python hf_ehr/eval/generate_mimiciv_ehrshot_labels.py \
  --path_to_meds_reader_extract "${MEDS_READER_DIR}" \
  --path_to_labels_dir "${LABELS_DIR}" \
  --min_prior_history_days 730
```

Successful output should include lines like:

```text
Wrote <N> death labels from <M> subjects to .../benchmark_mimiciv/death/all_labels.csv
Wrote <N> long_los labels from <M> subjects to .../benchmark_mimiciv/long_los/all_labels.csv
```

## Sanity Checks

After label preparation completes, check that each file exists and has more than
a header:

```bash
for task in long_los death; do
  label="${LABELS_DIR}/${task}/all_labels.csv"
  echo "TASK=$task"
  wc -l "$label"
  head -n 3 "$label"
done
```

Expected:

- The first line is `patient_id,prediction_time,value,label_type`.
- `value` is boolean text: `true` or `false`.
- There is one subdirectory per task.
- There is no combined mixed-task `all_labels.csv` at the benchmark root used by
  eval jobs.

## Eval Usage

The generated labels are consumed by `hf_ehr/eval/ehrshot.py` using the MEDS
reader backend:

```bash
python hf_ehr/eval/ehrshot.py \
  --database_backend meds_reader \
  --path_to_database "${MEDS_READER_DIR}" \
  --path_to_labels_dir "${LABELS_DIR}/long_los" \
  --path_to_features_dir "${EHRSHOT_ASSETS}/features_mimiciv" \
  --path_to_tokenized_timelines_dir "${EHRSHOT_ASSETS}/tokenized_timelines_mimiciv" \
  --path_to_model "${PATH_BASE}/runs/mimiciv_gpt_small_10ep_medsnorm/ckpts/last.ckpt" \
  --model_name mimiciv_gpt_small_10ep_medsnorm_clmbr_512_long_los \
  --batch_size 32 \
  --embed_strat last_nonpad \
  --chunk_strat last \
  --device cuda:0 \
  --add_special_tokens \
  --padding_side right
```

For a smoke run, add:

```bash
--patient_idx_start 0 --patient_idx_end 256
```

## Safety Notes

- Keep MIMIC-IV data and generated restricted artifacts in the approved compute
  environment.
- Pull only compact logs or summaries unless explicitly approved.
- The public Stanford EHRSHOT benchmark scripts are not required for this
  MIMIC-native label preparation flow.

## Appendix: Longleaf Workflow

When the approved compute environment is Longleaf, use the repo-local SLURM
wrappers instead of running data preparation or GPU feature generation directly
on the login node.

Submit the CPU label-prep job from the local repo:

```bash
./scripts/longleaf/submit.sh jobs/hf_ehr_ehrshot_prep.slurm
```

The job:

1. Uses the configured MEDS reader extract.
2. Creates the MIMIC-IV EHRSHOT-style workspace.
3. Installs the repo into the shared `uv` environment.
4. Runs `generate_mimiciv_ehrshot_labels.py`.
5. Fails if either task CSV is missing or empty.

Monitor it with:

```bash
./scripts/longleaf/status.sh <job_id>
./scripts/longleaf/tail.sh <job_id> 120
```

For Longleaf sanity checks, run the generic check remotely with `LABELS_DIR`
forwarded:

```bash
ssh longleaf "LABELS_DIR='${LABELS_DIR}' bash -s" <<'REMOTE'
for task in long_los death; do
  label="${LABELS_DIR}/${task}/all_labels.csv"
  echo "TASK=$task"
  wc -l "$label"
  head -n 3 "$label"
done
REMOTE
```

For feature generation on Longleaf, submit the smoke job first, then the full
job only after the smoke evidence passes:

```bash
./scripts/longleaf/submit.sh jobs/hf_ehr_ehrshot_smoke_l40s.slurm
./scripts/longleaf/submit.sh jobs/hf_ehr_ehrshot_full_l40s.slurm
```
