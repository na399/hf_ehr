# How to Run

## EHRSHOT

### Few-shot
Usage: `python ehrshot.py <path_to_ckpt> <model_name> <batch_size> <device>`

where...
- `<path_to_ckpt>` is the path to the checkpoint to load
- `<model_name>` is the name of the model
- `<batch_size>` is the batch size to use
- `<device>` is the device (cpu/gpu) where to run the program

```sh
# Example commands with local paths
# Replace <OUTPUT_DIR> with your actual output directory

# gpt2-base
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-512--clmbr 32
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-1024--clmbr 8
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-2048--clmbr 8
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-4096--clmbr 4

# hyena-medium
python ehrshot.py <OUTPUT_DIR>/runs/hyena-medium-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-medium-1024--clmbr 32
python ehrshot.py <OUTPUT_DIR>/runs/hyena-medium-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-medium-4096--clmbr 8
python ehrshot.py <OUTPUT_DIR>/runs/hyena-medium-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-medium-8192--clmbr 4
python ehrshot.py <OUTPUT_DIR>/runs/hyena-medium-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-medium-16384--clmbr 1

# mamba-tiny
python ehrshot.py <OUTPUT_DIR>/runs/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-1024--clmbr 16
python ehrshot.py <OUTPUT_DIR>/runs/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-4096--clmbr 16
python ehrshot.py <OUTPUT_DIR>/runs/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-8192--clmbr 8
python ehrshot.py <OUTPUT_DIR>/runs/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-16k--clmbr 1

# gpt-base-clmbr_k
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_8k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_8k 16 cuda:0
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_16k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_16k 16 cuda:0
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_64k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_64k 8 cuda:0
python ehrshot.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_96k/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt gpt-base-1024--clmbr_96k 1 cuda:0

# llama-base
python ehrshot.py <OUTPUT_DIR>/runs/llama-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-512--clmbr 16
python ehrshot.py <OUTPUT_DIR>/runs/llama-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-1024--clmbr 16
python ehrshot.py <OUTPUT_DIR>/runs/llama-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-2048--clmbr 8
python ehrshot.py <OUTPUT_DIR>/runs/llama-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=100000000-persist.ckpt llama-base-4096--clmbr 1
```

### Zero-shot

Need to run `7b_eval_zero_shot.sh` from the `ehrshot-benchmark` repo.

Timings on GPU node:
- gpt2-512: 0.5hr per rollout over 2195 test patients in guo_los (batch_size=8)
- gpt2-4k: 12hr per rollout over 2192 test patients in guo_los (batch_size=8)

Timings on a100 node:
- mamba-16k: 6hr per rollout over 2195 test patients in guo_los (batch_size=1)

Timings on h100 node:
- mamba-1k: 0.25hr per rollout over 2195 test patients in guo_los (batch_size=64)
- mamba-16k: 4hr per rollout over 2195 test patients in guo_los (batch_size=8)
- llama-4k: 1hr per rollout over 2195 test patients in guo_los (batch_size=16)

## MIMIC-4

```sh
# Example commands with local paths
# Replace <OUTPUT_DIR> with your actual output directory

# gpt2-base
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-512--clmbr 32
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-1024--clmbr 8
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-2048--clmbr 8
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt gpt2-base-4096--clmbr 4

# hyena-large
python mimic4.py <OUTPUT_DIR>/runs/hyena-large-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-large-1024--clmbr 32
python mimic4.py <OUTPUT_DIR>/runs/hyena-large-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-large-4096--clmbr 8
python mimic4.py <OUTPUT_DIR>/runs/hyena-large-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-large-8192--clmbr 4
python mimic4.py <OUTPUT_DIR>/runs/hyena-large-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt hyena-large-16384--clmbr 1

# mamba-tiny
python mimic4.py <OUTPUT_DIR>/runs/mamba-tiny-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-1024--clmbr 16
python mimic4.py <OUTPUT_DIR>/runs/mamba-tiny-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-4096--clmbr 16
python mimic4.py <OUTPUT_DIR>/runs/mamba-tiny-8192--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-8192--clmbr 8
python mimic4.py <OUTPUT_DIR>/runs/mamba-tiny-16384--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt mamba-tiny-16384--clmbr 1

# gpt-base-clmbr_k
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_8k/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt gpt-base-1024--clmbr_8k 16
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_16k/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt gpt-base-1024--clmbr_16k 16
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_64k/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt gpt-base-1024--clmbr_64k 8
python mimic4.py <OUTPUT_DIR>/runs/gpt-base-1024--clmbr_96k/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt gpt-base-1024--clmbr_96k 1

# llama-base
python mimic4.py <OUTPUT_DIR>/runs/llama-base-512--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt llama-base-512--clmbr 16
python mimic4.py <OUTPUT_DIR>/runs/llama-base-1024--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=1000000000-persist.ckpt llama-base-1024--clmbr 16
python mimic4.py <OUTPUT_DIR>/runs/llama-base-2048--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt llama-base-2048--clmbr 8
python mimic4.py <OUTPUT_DIR>/runs/llama-base-4096--clmbr/ckpts/train-tokens-total_nonPAD-ckpt_val=2000000000-persist.ckpt llama-base-4096--clmbr 1
```

## Environment Variables

Set these environment variables before running:
```bash
export OUTPUT_DIR=./outputs  # Or your preferred output directory
export DATA_DIR=./data       # Or your data directory
export CACHE_DIR=./cache     # Or your cache directory
```