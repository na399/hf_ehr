# End-to-End Training Guide for HF-EHR

## Overview

This guide provides comprehensive instructions for training EHR foundation models on OMOP CDM v5.4 data using the HF-EHR framework. The system supports both NVIDIA GPUs (L40s) and Apple Silicon (M3 Pro) with automatic device detection.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Model Prioritization](#model-prioritization)
5. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd hf_ehr

# Install package in development mode
pip install -e .

# Install MEDS packages (if not already installed)
pip install meds_reader meds_etl
```

### 2. Configure Environment

Copy and edit the environment configuration:

```bash
cp .env.example .env
```

Edit `.env` with your paths:

```bash
# Data directories (adjust to your setup)
DATA_DIR=./data
OUTPUT_DIR=./outputs
CACHE_DIR=./cache
LOG_DIR=./logs

# OMOP CDM data location
OMOP_DATA_DIR=./data/Synthea27NjParquet
MEDS_DATA_DIR=./data/synthea_meds
MEDS_READER_DIR=./data/synthea_meds_reader

# Device configuration (auto-detects by default)
FORCE_DEVICE=auto  # Options: auto, cuda, mps, cpu

# For L40s users
CUDA_VISIBLE_DEVICES=0

# Logging
WANDB_PROJECT=hf-ehr-training
WANDB_ENTITY=your-entity
WANDB_MODE=online  # or offline for local logging
```

### 3. Verify Setup

Run the comprehensive test suite:

```bash
python scripts/test_e2e.py
```

Expected output:
```
✓ PyTorch version: 2.5.1
✓ Device type: mps (or cuda for L40s)
✓ Synthea Data: PASSED
✓ Model Loading: PASSED
✓ ALL TESTS PASSED!
```

---

## Data Preparation

### Converting OMOP CDM v5.4 to MEDS Format

The framework requires data in MEDS format for efficient training. Use the provided converter:

```bash
# Basic conversion
python scripts/convert_omop_to_meds.py

# With custom paths
python scripts/convert_omop_to_meds.py \
  --input ./data/Synthea27NjParquet \
  --meds-output ./data/synthea_meds \
  --reader-output ./data/synthea_meds_reader \
  --train-split 0.8 \
  --val-split 0.1

# Force overwrite existing data
python scripts/convert_omop_to_meds.py --force
```

This will:
1. Convert OMOP parquet files to MEDS format (~5-10 minutes for Synthea)
2. Create MEDS Reader database for efficient loading
3. Generate train/val/test splits (80%/10%/10% by default)

### Expected Data Structure

After conversion, you should have:

```
data/
├── Synthea27NjParquet/      # Original OMOP data
│   ├── person.parquet
│   ├── condition_occurrence.parquet
│   ├── drug_exposure.parquet
│   └── ...
├── synthea_meds/            # MEDS format data
│   └── data/
└── synthea_meds_reader/     # MEDS Reader database
    ├── metadata/
    │   └── subject_splits.parquet
    └── ...
```

---

## Model Training

### Quick Start

Train a model with automatic configuration:

```bash
# Train GPT-2 base model
python scripts/train_local.py \
  --model gpt2 \
  --size base \
  --context-length 512 \
  --epochs 10

# Train Mamba tiny model (recommended for testing)
python scripts/train_local.py \
  --model mamba \
  --size tiny \
  --context-length 1024 \
  --epochs 5

# Debug mode (very fast, for testing)
python scripts/train_local.py \
  --model gpt2 \
  --size base \
  --debug \
  --wandb-offline
```

### Advanced Training Options

```bash
python scripts/train_local.py \
  --model llama \              # Model architecture
  --size base \                # Model size
  --tokenizer clmbr \          # Tokenizer choice
  --context-length 2048 \      # Sequence length
  --batch-size auto \          # Auto-detect optimal batch size
  --epochs 20 \                # Number of epochs
  --force-device mps \         # Force specific device
  --wandb-offline \            # Offline logging
  --force-refresh              # Start training from scratch
```

### Device-Specific Configurations

The system automatically detects and optimizes for your device:

#### L40s (48GB VRAM) - CUDA
- Larger batch sizes (32-64 for base models)
- BF16 mixed precision
- More dataloader workers (8)

#### M3 Pro (36GB Unified) - MPS
- Smaller batch sizes (8-16 for base models)
- FP16 mixed precision
- Fewer dataloader workers (4)

---

## Model Prioritization

Models are prioritized based on performance and compatibility:

### 1. **Mamba** (Priority 1)
Best for long sequences with linear complexity.

```bash
python scripts/train_local.py --model mamba --size tiny --context-length 4096
```

Sizes: `tiny`, `small`, `medium`, `large`

**Note**: Requires `mamba-ssm` for optimized kernels (optional):
```bash
pip install mamba-ssm causal-conv1d  # CUDA only
```

### 2. **Llama** (Priority 2)
Industry-standard architecture with RoPE embeddings.

```bash
python scripts/train_local.py --model llama --size base --context-length 2048
```

Sizes: `base` only (add more configs as needed)

### 3. **GPT-2** (Priority 3)
Well-tested baseline model.

```bash
python scripts/train_local.py --model gpt2 --size base --context-length 1024
```

Sizes: `base`, `medium`, `large`, `xlarge`

### 4. **Hyena** (Priority 4)
Experimental architecture with convolutions.

```bash
python scripts/train_local.py --model hyena --size small --context-length 2048
```

Sizes: `tiny`, `small`, `medium`, `large`, `xlarge`

---

## Monitoring Training

### WandB Integration

Training metrics are automatically logged to Weights & Biases:

1. Set up WandB (first time only):
```bash
wandb login
```

2. Configure in `.env`:
```bash
WANDB_PROJECT=hf-ehr-training
WANDB_ENTITY=your-team
WANDB_MODE=online
```

3. View runs at: https://wandb.ai/your-team/hf-ehr-training

### Local Monitoring

For offline training:
```bash
python scripts/train_local.py --wandb-offline

# View logs
tail -f logs/training.log
```

---

## Troubleshooting

### Common Issues

#### Out of Memory (OOM)

Reduce batch size or context length:
```bash
python scripts/train_local.py --batch-size 4 --context-length 256
```

Or use gradient accumulation:
```bash
# In configs/trainer/mps.yaml or configs/trainer/single_gpu.yaml
accumulate_grad_batches: 8
```

#### MPS Fallback Errors

Some operations may not be supported on MPS. Force CPU if needed:
```bash
export FORCE_DEVICE=cpu
python scripts/train_local.py
```

#### MEDS Conversion Fails

Check OMOP data structure:
```bash
ls -la data/Synthea27NjParquet/*.parquet
```

Required tables:
- person.parquet
- condition_occurrence.parquet
- drug_exposure.parquet
- measurement.parquet
- observation.parquet

### Performance Tips

#### For L40s (CUDA)
- Use larger batch sizes: `--batch-size 32`
- Enable BF16 (automatic)
- Use multiple workers in dataloader

#### For M3 Pro (MPS)
- Use smaller batch sizes: `--batch-size 8`
- Use gradient accumulation for effective larger batches
- Monitor unified memory usage

#### General
- Start with shorter sequences: `--context-length 512`
- Use debug mode for testing: `--debug`
- Profile memory usage: Check `nvidia-smi` or Activity Monitor

---

## Next Steps

After successful training:

1. **Evaluate Model**: Use validation perplexity to assess performance
2. **Fine-tune**: Adapt for downstream tasks
3. **Deploy**: Export to HuggingFace format

### Export Trained Model

```python
from transformers import AutoModelForCausalLM

# Load checkpoint
model = AutoModelForCausalLM.from_pretrained("outputs/runs/your-model/")

# Push to HuggingFace Hub
model.push_to_hub("your-org/model-name")
```

---

## Support

For issues or questions:
- Check logs: `tail -f logs/*.log`
- Run tests: `python scripts/test_e2e.py`
- Review configs: `hf_ehr/configs/`

## Summary of Changes Made

1. **Device Detection**: Automatic CUDA/MPS/CPU detection with fallbacks
2. **Environment Config**: Replaced 87 hard-coded paths with environment variables
3. **OMOP→MEDS Pipeline**: Complete conversion toolchain for OMOP CDM v5.4
4. **Model Configs**: Prioritized Mamba > Llama > GPT > Hyena
5. **Training Scripts**: Unified interface for all platforms
6. **Documentation**: Comprehensive guides for setup and training

The system is now ready for end-to-end training on both L40s (CUDA) and M3 Pro (MPS) devices!