# End-to-End Training Guide for HF-EHR

## Overview

This guide provides comprehensive instructions for training EHR foundation models on OMOP CDM v5.4 data using the HF-EHR framework. 

**Important:** This version has been simplified to focus exclusively on single CUDA GPU training for improved stability and performance. Distributed training, MPS (Apple Silicon), and CPU support have been removed.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Tokenizer Creation](#tokenizer-creation)
4. [Model Training](#model-training)
5. [Model Prioritization](#model-prioritization)
6. [Troubleshooting](#troubleshooting)

## Complete Workflow

```bash
# Step 1: Setup environment
uv venv && source .venv/bin/activate
uv pip install -e .

# Step 2: Convert OMOP to MEDS
python scripts/convert_omop_to_meds.py

# Step 3: Create tokenizer
python scripts/create_tokenizers.py --type clmbr

# Step 4: Train model
python scripts/train_local.py --model gpt2 --size base
```

---

## Environment Setup

### 1. Install Dependencies

```bash
# Clone the repository
git clone <repo-url>
cd hf_ehr

# Create virtual env
uv venv
source .venv/bin/activate

# Install package in development mode
uv pip install -e .
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

# Device configuration (CUDA only)
CUDA_VISIBLE_DEVICES=0  # Select GPU device ID

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
✓ Device type: cuda
✓ CUDA available: True
✓ GPU: NVIDIA L40s (or your GPU model)
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

## Tokenizer Creation

### Overview

Before training models, you need to create a tokenizer that converts clinical codes and events into tokens. HF-EHR supports three tokenizer types:

1. **CLMBRTokenizer** - Fast, uses pre-defined vocabulary from CLMBR papers (~5 seconds)
2. **DescTokenizer** - Description-based embeddings, more semantic (~30 minutes)
3. **CookbookTokenizer** - Custom tokenizer with numerical binning (~10 minutes)

### Quick Start

Create the default CLMBR tokenizer (recommended for beginners):

```bash
# Quick tokenizer creation
python scripts/create_tokenizers.py --type clmbr

# Or create all tokenizers
python scripts/create_tokenizers.py --type all
```

### Detailed Tokenizer Creation

#### 1. CLMBRTokenizer (Recommended)

The fastest tokenizer, uses a pre-defined vocabulary from the CLMBR/MOTOR/EHRSHOT papers:

```bash
# Basic creation
python scripts/create_tokenizers.py --type clmbr

# With vocabulary size limit (in thousands)
python scripts/create_tokenizers.py --type clmbr --vocab-size 8

# Direct creation (advanced)
python hf_ehr/tokenizers/create_clmbr.py \
  --path_to_tokenizer_config hf_ehr/configs/tokenizer/clmbr.yaml
```

**Pros:** Fast creation, well-tested, good baseline performance
**Cons:** Fixed vocabulary, may miss dataset-specific codes

#### 2. DescTokenizer

Creates embeddings based on code descriptions:

```bash
# Basic creation
python scripts/create_tokenizers.py --type desc

# With more workers for faster processing
python scripts/create_tokenizers.py --type desc --n-procs 10

# Direct creation (advanced)
python hf_ehr/tokenizers/create_desc.py \
  --path_to_dataset_config hf_ehr/configs/data/synthea_omop.yaml \
  --path_to_tokenizer_config hf_ehr/configs/tokenizer/desc.yaml \
  --n_procs 10
```

**Pros:** Semantic understanding, better for rare codes
**Cons:** Slower creation, requires code descriptions

#### 3. CookbookTokenizer

Custom tokenizer with numerical binning for lab values:

```bash
# Basic creation
python scripts/create_tokenizers.py --type cookbook

# With custom binning
python scripts/create_tokenizers.py --type cookbook --n-buckets 5

# Direct creation (advanced)
python hf_ehr/tokenizers/create_cookbook.py \
  --path_to_dataset_config hf_ehr/configs/data/synthea_omop.yaml \
  --path_to_tokenizer_config hf_ehr/configs/tokenizer/cookbook.yaml \
  --n_buckets_for_numerical_range_codes 10 \
  --n_procs 10
```

**Pros:** Handles numerical values well, customizable
**Cons:** Slowest creation, may overfit to training data

### Tokenizer Storage

Tokenizers are stored in:
```
cache/tokenizers/
├── clmbr_synthea/
│   ├── tokenizer_config.json
│   └── vocab.json
├── desc_synthea/
│   └── ...
└── cookbook_synthea/
    └── ...
```

### Verifying Tokenizer Creation

Check if your tokenizer was created successfully:

```bash
# List available tokenizers
ls -la cache/tokenizers/

# Check tokenizer config
python -c "
import json
with open('cache/tokenizers/clmbr_synthea/tokenizer_config.json') as f:
    config = json.load(f)
    print(f'Tokenizer has {len(config[\"tokens\"])} tokens')
"
```

### Performance Considerations

| Tokenizer | Creation Time | Memory Usage | Recommended For |
|-----------|--------------|--------------|-----------------|
| CLMBR | ~5 seconds | Low | Quick experiments, baseline |
| Desc | ~30 minutes | Medium | Production models |
| Cookbook | ~10 minutes | High | Custom domains, lab-heavy data |

### Troubleshooting Tokenizer Creation

#### Out of Memory
Reduce the number of processes:
```bash
python scripts/create_tokenizers.py --type cookbook --n-procs 2
```

#### Missing Dependencies
Install required packages:
```bash
uv pip install meds-reader transformers
```

#### HuggingFace Authentication (for CLMBR)
The CLMBR tokenizer downloads from HuggingFace. If needed:
```bash
huggingface-cli login
```

---

## Model Training

### Prerequisites

Before training, ensure you have:
1. ✅ Converted OMOP data to MEDS format (see [Data Preparation](#data-preparation))
2. ✅ Created a tokenizer (see [Tokenizer Creation](#tokenizer-creation))

### Quick Start

Train a model with automatic configuration:

```bash
# First create tokenizer (if not done)
python scripts/create_tokenizers.py --type clmbr

# Train GPT-2 base model
python scripts/train_local.py \
  --model gpt2 \
  --size base \
  --tokenizer clmbr_synthea \
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
  --tokenizer clmbr_synthea \  # Tokenizer choice
  --context-length 2048 \      # Sequence length
  --batch-size auto \          # Auto-detect optimal batch size
  --epochs 20 \                # Number of epochs
  --force-device mps \         # Force specific device
  --wandb-offline \            # Offline logging
  --force-refresh              # Start training from scratch
```

### GPU Memory Guidelines

Recommended batch sizes by GPU memory:

#### High-End GPUs (40GB+ VRAM)
**Examples:** L40s (48GB), A100 (40/80GB), H100 (80GB)
- Batch sizes: 32-64 for base models
- BF16 mixed precision (automatic)
- 8 dataloader workers
- Context length: Up to 4096

#### Mid-Range GPUs (24GB VRAM)
**Examples:** RTX 4090, RTX 3090, A10
- Batch sizes: 16-32 for base models  
- Mixed precision enabled
- 4 dataloader workers
- Context length: Up to 2048

#### Entry GPUs (16GB VRAM)
**Examples:** V100, T4, RTX 4070 Ti
- Batch sizes: 4-8 for base models
- Use gradient accumulation
- 2 dataloader workers
- Context length: 512-1024

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

#### CUDA Not Available

Ensure CUDA is properly installed:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If False, reinstall PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
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

#### Memory Optimization
- Monitor GPU memory: `nvidia-smi -l 1`
- Use gradient accumulation for effective larger batches
- Enable mixed precision (automatic with CUDA)
- Clear cache between epochs if needed

#### Speed Optimization  
- Use larger batch sizes when memory allows
- Pin memory for faster data transfer (`pin_memory: true`)
- Adjust number of dataloader workers based on CPU cores
- Use torch.compile() for PyTorch 2.0+ (experimental)

#### Debugging
- Start with shorter sequences: `--context-length 512`
- Use debug mode for testing: `--debug`
- Set `CUDA_LAUNCH_BLOCKING=1` for better error messages
- Use `torch.cuda.empty_cache()` to free memory

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

## Summary of Configuration

### Simplified Architecture
This version has been streamlined for single CUDA GPU training:

1. **Single GPU Focus**: All distributed training code removed for improved stability
2. **CUDA Only**: Exclusively supports NVIDIA GPUs with CUDA
3. **Fixed Issues**: Resolved tokenizer special tokens and interval calculation bugs
4. **Optimized Settings**: Pre-configured for single GPU performance
5. **Clean Codebase**: Removed complexity from multi-device support

### Key Features
- ✅ OMOP CDM v5.4 → MEDS conversion pipeline
- ✅ Three tokenizer types (CLMBR, Desc, Cookbook)
- ✅ Support for GPT-2, Llama, Mamba, Hyena architectures
- ✅ Automatic mixed precision training
- ✅ WandB integration for experiment tracking
- ✅ Checkpoint saving and resumption

### System Requirements
- NVIDIA GPU with CUDA support
- CUDA toolkit 11.8 or higher
- PyTorch 2.0+ with CUDA support
- Sufficient GPU memory for chosen model size

The system is optimized for high-performance single GPU training on NVIDIA hardware!