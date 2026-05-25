from pathlib import Path
import ast


REPO = Path(__file__).resolve().parents[1]
READER_PATH = "/nas/longleaf/home/na399/users/mimiciv_meds_reader/mimiciv_reader"
REMOTE_BASE = "/nas/longleaf/home/na399/users/hf_ehr"


def read(path: str) -> str:
    return (REPO / path).read_text()


def test_submit_wrapper_validates_job_and_forwards_cache_contract():
    submit = read("scripts/longleaf/submit.sh")

    assert "Usage:" in submit
    assert "test -f" in submit
    assert "rsync" in submit
    assert "sbatch" in submit
    for name in [
        "UV_SHARED_VENV",
        "UV_PROJECT_ENVIRONMENT",
        "UV_CACHE_DIR",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
    ]:
        assert name in submit


def test_submit_wrapper_allows_safe_job_specific_overrides():
    submit = read("scripts/longleaf/submit.sh")

    assert "append_optional_export" in submit
    for name in [
        "OUTPUT_DIR",
        "TOKENIZER_CACHE_DIR",
        "TOKENIZER_SMOKE_MAX_SUBJECTS",
        "TOKENIZER_SMOKE_OUTPUT",
        "PATH_TO_CKPT",
        "MODEL_NAME",
        "LABEL_TASKS",
        "MIN_PRIOR_HISTORY_DAYS",
    ]:
        assert name in submit


def test_job_files_have_expected_hydra_and_resource_contracts():
    tokenizer = read("jobs/hf_ehr_tokenizer_clmbr.slurm")
    tokenizer_smoke = read("jobs/hf_ehr_mimiciv_tokenization_smoke_cpu.slurm")
    smoke = read("jobs/hf_ehr_mimiciv_smoke_l40s.slurm")
    full = read("jobs/hf_ehr_mimiciv_gpt_small_10ep_l40s.slurm")
    ehrshot_prep = read("jobs/hf_ehr_ehrshot_prep.slurm")
    ehrshot_smoke = read("jobs/hf_ehr_ehrshot_smoke_l40s.slurm")
    ehrshot_full = read("jobs/hf_ehr_ehrshot_full_l40s.slurm")
    ehrshot_metrics = read("jobs/hf_ehr_ehrshot_metrics_cpu.slurm")
    ehrshot_diag = read("jobs/hf_ehr_ehrshot_diagnostics_cpu.slurm")

    assert "#SBATCH --partition=general" in tokenizer
    assert "hf_ehr/tokenizers/create_clmbr.py" in tokenizer
    assert "--path_to_orig_clmbr_json" in tokenizer
    assert "clmbr_mimiciv/tokenizer_config.json" in tokenizer

    assert "#SBATCH --partition=general" in tokenizer_smoke
    assert "mimiciv_tokenization_smoke.py" in tokenizer_smoke
    assert "TOKENIZER_SMOKE_MAX_SUBJECTS" in tokenizer_smoke
    assert "TOKENIZER_SMOKE_OUTPUT" in tokenizer_smoke
    assert "clmbr_mimiciv/tokenizer_config.json" in tokenizer_smoke
    assert "tokenizer_coverage/clmbr_mimiciv_medsnorm" in tokenizer_smoke
    assert READER_PATH in tokenizer_smoke

    for gpu_job in [smoke, full]:
        assert "#SBATCH --partition=l40-gpu" in gpu_job
        assert "#SBATCH --qos=gpu_access" in gpu_job
        assert "#SBATCH --gres=gpu:nvidia_l40s:1" in gpu_job
        assert READER_PATH in gpu_job
        assert REMOTE_BASE in gpu_job
        assert "+data=meds_dev" in gpu_job
        assert "+trainer=single_gpu" in gpu_job
        assert "+model=gpt2-base" in gpu_job
        assert "+tokenizer=clmbr" in gpu_job
        assert "model.config_kwargs.n_layer=6" in gpu_job
        assert "model.is_gradient_checkpointing=True" in gpu_job
        assert "trainer.precision=bf16-mixed" in gpu_job
        assert "callbacks.model_checkpointing.save_last_checkpoint=True" in gpu_job
        assert "WANDB_MODE=online" in gpu_job
        assert "logging.wandb.is_wandb=True" in gpu_job
        assert "logging.wandb.entity=na399-ai" in gpu_job
        assert "logging.wandb.project=hf_ehr_longleaf" in gpu_job
        assert "data.dataloader.seq_length_n_procs=1" in gpu_job

    assert "data.dataset.is_debug=True" in smoke
    assert "trainer.limit_train_batches=2" in smoke
    assert "trainer.limit_val_batches=1" in smoke
    assert "trainer.max_epochs=4" in smoke
    assert "callbacks.early_stopping.patience=0" in smoke
    assert "callbacks.early_stopping.min_delta=1000000" in smoke
    assert "callbacks.model_checkpointing.save_most_recent_k=1" in smoke
    assert "mimiciv_smoke_clmbr_medsnorm" in smoke

    assert "data.dataset.is_debug=False" in full
    assert "trainer.max_epochs=10" in full
    assert "trainer.limit_train_batches=null" in full
    assert "trainer.limit_val_batches=1" in full
    assert "callbacks.early_stopping.patience=3" in full
    assert "callbacks.model_checkpointing.save_most_recent_k=0" in full
    assert "runs/mimiciv_gpt_small_10ep_medsnorm" in full
    assert "mimiciv_full_clmbr_512_medsnorm" in full

    assert "#SBATCH --partition=general" in ehrshot_prep
    assert "generate_mimiciv_ehrshot_labels.py" in ehrshot_prep
    assert "--min_prior_history_days" in ehrshot_prep
    assert "MIN_PRIOR_HISTORY_DAYS" in ehrshot_prep
    assert READER_PATH in ehrshot_prep
    assert "benchmark_mimiciv" in ehrshot_prep
    assert "death/all_labels.csv" in ehrshot_prep
    assert "long_los/all_labels.csv" in ehrshot_prep

    for ehrshot_job in [ehrshot_smoke, ehrshot_full]:
        assert "#SBATCH --partition=l40-gpu" in ehrshot_job
        assert "#SBATCH --qos=gpu_access" in ehrshot_job
        assert "#SBATCH --gres=gpu:nvidia_l40s:1" in ehrshot_job
        assert "/nas/longleaf/home/na399/users/hf_ehr" in ehrshot_job
        assert "runs/mimiciv_gpt_small_10ep_medsnorm/ckpts/last.ckpt" in ehrshot_job
        assert "mimiciv_gpt_small_10ep_medsnorm_clmbr_512" in ehrshot_job
        assert "EHRSHOT_ASSETS" in ehrshot_job
        assert READER_PATH in ehrshot_job
        assert "benchmark_mimiciv" in ehrshot_job
        assert "features_mimiciv" in ehrshot_job
        assert "tokenized_timelines_mimiciv" in ehrshot_job
        assert "UV_SHARED_VENV" in ehrshot_job
        assert "hf_ehr/eval/ehrshot.py" in ehrshot_job
        assert "--database_backend meds_reader" in ehrshot_job
        assert "--batch_size \"${BATCH_SIZE}\"" in ehrshot_job
        assert "EMBED_STRAT" in ehrshot_job
        assert "last_nonpad" in ehrshot_job
        assert "--embed_strat \"${EMBED_STRAT}\"" in ehrshot_job
        assert "--chunk_strat last" in ehrshot_job
        assert "--add_special_tokens" in ehrshot_job
        assert "--padding_side \"${PADDING_SIDE}\"" in ehrshot_job

    assert "--patient_idx_start 0" in ehrshot_smoke
    assert "--patient_idx_end 256" in ehrshot_smoke
    assert "LABEL_TASK" in ehrshot_smoke
    assert "${MODEL_NAME}_${LABEL_TASK}" in ehrshot_smoke
    assert "7_eval.sh" not in ehrshot_smoke

    assert "LABEL_TASKS" in ehrshot_full
    assert "${MODEL_NAME}_${label_task}" in ehrshot_full
    assert "7_eval.sh" not in ehrshot_full

    assert "#SBATCH --partition=general" in ehrshot_metrics
    assert "ehrshot_mimiciv_probe_metrics.py" in ehrshot_metrics
    assert "features_mimiciv" in ehrshot_metrics
    assert "results_mimiciv" in ehrshot_metrics
    assert "subject_splits.parquet" in ehrshot_metrics
    assert "--path_to_subject_splits" in ehrshot_metrics
    assert "--embed_strat \"${EMBED_STRAT}\"" in ehrshot_metrics
    assert "last_nonpad" in ehrshot_metrics
    assert "mimiciv_gpt_small_10ep_medsnorm_clmbr_512" in ehrshot_metrics
    assert "LABEL_TASKS" in ehrshot_metrics
    assert "long_los death" in ehrshot_metrics
    assert "UV_SHARED_VENV" in ehrshot_metrics

    assert "#SBATCH --partition=general" in ehrshot_diag
    assert "ehrshot_mimiciv_diagnostics.py" in ehrshot_diag
    assert "features_mimiciv" in ehrshot_diag
    assert "tokenized_timelines_mimiciv" in ehrshot_diag
    assert "diagnostics.json" in ehrshot_diag
    assert "subject_splits.parquet" in ehrshot_diag
    assert "--path_to_subject_splits" in ehrshot_diag
    assert "--embed_strat \"${EMBED_STRAT}\"" in ehrshot_diag
    assert "last_nonpad" in ehrshot_diag
    assert "UV_SHARED_VENV" in ehrshot_diag


def test_agents_policy_blocks_local_heavy_work_and_data_exfiltration():
    policy = read("AGENTS.md")

    assert "Do not run training" in policy
    assert "Longleaf login node" in policy
    assert "Do not copy PHI" in policy
    assert "scripts/longleaf/submit.sh" in policy


def test_create_clmbr_imports_optional_for_runtime_annotation():
    tree = ast.parse(read("hf_ehr/tokenizers/create_clmbr.py"))
    typing_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "typing"
        for alias in node.names
    }

    assert "Optional" in typing_imports


def test_clmbr_tokenizer_normalizes_meds_reader_codes():
    source = read("hf_ehr/data/tokenization.py")

    assert "clmbr_meds_code_candidates" in source
    assert "clmbr_meds_raw_variants" in source
    assert "meds_code_normalization" in source


def test_causal_lm_collator_masks_padding_labels():
    source = read("hf_ehr/data/tokenization.py")

    assert "tokens['labels'] = tokens['input_ids'].clone()" in source
    assert "tokens['labels'][tokens['attention_mask'] == 0] = -100" in source


def test_ehrshot_eval_requires_explicit_overwrite_or_unique_rerun_dir():
    source = read("hf_ehr/eval/ehrshot.py")

    assert "--overwrite_existing_model_dir" in source
    assert "get_unique_path" in source
    assert "get_file_digest" in source
    assert "labels={label_cache_key}" in source
    assert "shutil.rmtree(path_to_model_ehrshot_dir)" in source
    assert "args.overwrite_existing_model_dir" in source
