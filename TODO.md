# TODO.md - Development Tasks

## Completed ✅
- [x] Remove distributed training code from run.py
- [x] Simplify loaders.py for single GPU
- [x] Update trainer configs for CUDA only
- [x] Fix CookbookTokenizer special tokens inclusion
- [x] Fix tokenizer interval calculation for overlapping visits
- [x] Remove MPS and CPU support code
- [x] Test training pipeline end-to-end
- [x] Remove all Stanford internal references (Carina, /share/pi/nigam paths)
- [x] Add BF16 support for Mamba models
- [x] Test Mamba model training on Synthea MEDS data
- [x] Replace internal paths with environment variables
- [x] Add BF16 support for all models (GPT, BERT, Hyena, T5, Based, Mamba, Llama)
- [x] Fix T5 model config_kwargs merge issue
- [x] Fix Based model library dependency (replaced with standard GPT2)
- [x] Fix Llama model gated repository access (create config directly)
- [x] Test all models with BF16 successfully

## High Priority 🔴
- [ ] Add comprehensive error handling for CUDA out of memory
- [ ] Implement automatic batch size finder
- [ ] Add gradient checkpointing option for large models
- [ ] Create script to automatically detect optimal training parameters
- [ ] Add validation for CUDA availability at startup
- [x] Add automatic precision detection (BF16 vs FP16 based on GPU capability)
- [ ] Clean up eval README.md to remove remaining Stanford paths
- [ ] Install flash_attn package for improved GPT/Llama performance
- [ ] Fix DataLoader worker CUDA initialization errors

## Medium Priority 🟡
- [ ] Optimize dataloader for single GPU (remove multi-GPU overhead)
- [ ] Add profiling tools for memory and compute usage
- [ ] Implement early stopping based on validation loss
- [ ] Add model checkpoint resumption with proper error recovery
- [ ] Create benchmarking script for different model sizes

## Low Priority 🟢
- [ ] Add option to re-enable distributed training (as plugin)
- [ ] Document performance benchmarks for different GPU types
- [ ] Add support for gradient accumulation strategies
- [ ] Implement learning rate finder
- [ ] Add tensorboard integration alongside wandb

## Documentation 📚
- [ ] Update main README.md with single GPU focus
- [ ] Add troubleshooting guide for common CUDA errors
- [ ] Create performance tuning guide
- [ ] Add examples for different model architectures
- [ ] Document memory requirements for each model size

## Testing 🧪
- [ ] Add unit tests for tokenizer fixes
- [ ] Create integration tests for training pipeline
- [ ] Add smoke tests for different model configurations
- [ ] Implement continuous integration for CUDA environments
- [ ] Add memory leak detection tests

## Optimization 🚀
- [ ] Profile and optimize data loading pipeline
- [ ] Implement mixed precision training optimization
- [ ] Add torch.compile() support for PyTorch 2.0+
- [ ] Optimize tokenizer caching mechanism
- [ ] Implement efficient sequence packing

## Known Issues 🐛
- [ ] DataLoader workers occasionally crash with CUDA initialization error (seen with Llama)
- [ ] Memory fragmentation with long training runs
- [ ] Tokenizer cache not properly invalidated on config changes
- [ ] Checkpoint saving can fail with large models
- [ ] Flash Attention 2 requires separate installation (not included)
- [ ] Based library removed - using standard GPT2 instead

## Future Features 💡
- [ ] Multi-GPU support as optional plugin
- [ ] Distributed training via separate module
- [ ] Support for quantized models (INT8, INT4)
- [ ] Integration with PEFT methods (LoRA, QLoRA)
- [ ] Support for streaming datasets
- [ ] Real-time training metrics dashboard

## Notes
- Priority on stability and performance for single GPU setup
- Focus on NVIDIA GPUs (L40s, H100, A100)
- Maintain compatibility with HuggingFace ecosystem