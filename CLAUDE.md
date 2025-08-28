# CLAUDE.md - Development Notes

## Recent Changes (2025-08-28)

### Stanford Internal References Removed
All Stanford-specific internal references have been removed from the codebase:

#### ✅ Removed Components:
1. **Carina System**
   - Deleted `hf_ehr/scripts/carina/` directory entirely
   - Removed `is_carina` flags from all configuration files
   - Removed `rewrite_paths_for_carina_from_config()` function
   - Cleaned up SLURM partition detection code

2. **Internal Paths**
   - Replaced all `/share/pi/nigam/*` paths with environment variables
   - Updated 19 YAML config files to use `${oc.env:VAR,default}` syntax
   - Updated 21 Python files to use local paths
   - Cleaned documentation to remove Stanford-specific instructions

3. **Path Replacements**
   - `/share/pi/nigam/data/` → `${oc.env:DATA_DIR,./data}/`
   - `/share/pi/nigam/mwornow/hf_ehr/cache/` → `${oc.env:CACHE_DIR,./cache}/`
   - Output paths → `${oc.env:OUTPUT_DIR,./outputs}/`
   - Wandb entity → `${oc.env:WANDB_ENTITY,na399-ai}`

### BF16 Support for All Models
Enhanced numerical stability across all model architectures:

#### 🔧 Implementation:
1. **Universal BF16 Support**
   - Added BF16 support to all model types: GPT, BERT, Hyena, T5, Based, Mamba, Llama
   - Automatically enabled for GPUs with compute capability ≥ 8.0
   - Provides better gradient stability than FP16
   - Flash Attention 2 made optional (requires separate installation)

2. **Modified Files**
   - `hf_ehr/models/mamba.py` - BF16 support
   - `hf_ehr/models/gpt.py` - BF16 (Flash Attention 2 optional)
   - `hf_ehr/models/bert.py` - BF16 support
   - `hf_ehr/models/hyena.py` - BF16 support
   - `hf_ehr/models/t5.py` - BF16 support + fixed config_kwargs
   - `hf_ehr/models/based.py` - BF16 support + replaced custom library with standard GPT2
   - `hf_ehr/models/llama.py` - BF16 support + fixed gated repo issue

3. **Model Fixes Applied**
   - **T5**: Added missing `config_kwargs: {}` to architecture config
   - **Based**: Removed `based` library dependency, uses standard AutoModelForCausalLM
   - **Llama**: Creates LlamaConfig directly instead of downloading from Meta's gated repo

4. **Benefits**
   - **Better dynamic range**: BF16 has same exponent bits as FP32
   - **Prevents NaN losses**: More stable gradient flow
   - **Hardware optimization**: Excellent support on modern GPUs (L40S, H100, A100)
   - **Consistent behavior**: All models now use same precision strategy
   - **No external dependencies**: All models work without additional libraries

### Successful BF16 Testing Results
Validated BF16 implementation across all model architectures:

#### ✅ Test Results (All Models):
| Model | BF16 | Training | Notes |
|-------|------|----------|-------|
| **GPT** | ✅ | ✅ Success | Flash Attention 2 made optional |
| **BERT** | ✅ | ✅ Success | BF16 working correctly |
| **Hyena** | ✅ | ✅ Success | BF16 working correctly |
| **Mamba** | ✅ | ✅ Success | BF16 working correctly, 5 epochs stable |
| **T5** | ✅ | ✅ Success | Fixed config_kwargs issue |
| **Based** | ✅ | ✅ Success | Using standard GPT2 instead of custom library |
| **Llama** | ✅ | ✅ Success | Creates config from scratch, no Meta download |

- **Dataset**: Synthea MEDS/OMOP (23 train, 3 val, 2 test samples)
- **GPU**: L40S (48GB VRAM) with compute capability 8.9
- **All models completed 1 epoch without errors**
- **Wandb tracking**: Successfully logged to na399-ai/hf-ehr-training

## Previous Changes (2024-08-28)

### Single GPU Simplification
The codebase has been significantly simplified to focus exclusively on single CUDA GPU training:

#### ✅ Removed Components:
1. **Distributed Training**
   - Removed all `torch.distributed` imports and operations
   - Removed `rank_zero_only` utilities
   - Removed DDP, FSDP, and multi-GPU strategy code
   - Removed distributed barriers and all_reduce operations
   - Cleaned up conditional blocks that checked for rank

2. **Multi-Device Support**
   - Removed MPS (Apple Silicon) support
   - Removed CPU fallback code
   - Hard-coded `accelerator='cuda'` and `devices=1`
   - Removed device detection logic

3. **Configuration Files**
   - Deleted `mps.yaml`, `multi_gpu_2.yaml`, `multi_gpu_4.yaml`, `fsdp.yaml`
   - Updated `single_gpu.yaml` to be CUDA-only
   - Removed `distributed_backend` from all configs

#### 🔧 Fixed Issues:
1. **Tokenizer Fixes**
   - Fixed `CookbookTokenizer` special tokens not being included in vocabulary
   - Fixed tokenizer interval calculation for overlapping visits (negative intervals)
   - Ensured base special tokens are preserved when subclasses add custom tokens

2. **Training Pipeline**
   - Set `strategy='auto'` for single GPU
   - Fixed devices handling to support integer values
   - Removed `use_distributed_sampler` logic
   - Set `pin_memory=True` for CUDA optimization

#### 📁 Modified Files:
- `/hf_ehr/scripts/run.py` - Removed distributed training code
- `/hf_ehr/trainer/loaders.py` - Simplified for single GPU
- `/hf_ehr/models/base.py` - Removed distributed operations
- `/hf_ehr/data/tokenization.py` - Fixed tokenizer issues
- `/hf_ehr/configs/trainer/single_gpu.yaml` - CUDA-only config
- `/hf_ehr/configs/config.yaml` - Removed distributed settings

## Training Instructions

### Prerequisites
- NVIDIA GPU with CUDA support (L40s, H100, etc.)
- CUDA toolkit installed
- PyTorch with CUDA support

### Quick Start
```bash
# Setup environment
uv venv && source .venv/bin/activate
uv pip install -e .

# Convert OMOP to MEDS
python scripts/convert_omop_to_meds.py

# Create tokenizer (use cookbook for full vocabulary)
python hf_ehr/tokenizers/create_cookbook.py \
    --path_to_dataset_config hf_ehr/configs/data/synthea_omop.yaml \
    --path_to_tokenizer_config hf_ehr/configs/tokenizer/cookbook_synthea.yaml

# Run training
cd /home/natthawut/hf_ehr
python -m hf_ehr.scripts.run \
    +model=gpt2-debug \
    +data=synthea_omop \
    +tokenizer=cookbook_synthea \
    +trainer=single_gpu \
    data.dataloader.max_length=512 \
    trainer.optimizer.lr=1e-4 \
    trainer.max_epochs=10 \
    data.dataloader.batch_size=4 \
    logging.wandb.entity=na399-ai \
    logging.wandb.project=hf-ehr-training \
    main.path_to_output_dir=./outputs/my-run
```

## Known Limitations
- Only supports single CUDA GPU training
- No distributed training support
- No MPS or CPU support
- Requires NVIDIA GPU with sufficient VRAM

## Future Work
- Consider adding back distributed training as optional feature
- Add automatic mixed precision (AMP) optimization
- Implement gradient checkpointing for memory efficiency