#!/usr/bin/env python3
"""
Local training script with automatic device detection.
Works on both CUDA (L40s) and MPS (M3 Pro) devices.
"""

import argparse
import sys
import os
from pathlib import Path
import subprocess
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hf_ehr.utils.device import get_device_config, get_optimal_batch_size
from hf_ehr.utils.config_loader import (
    get_run_output_dir, FORCE_DEVICE, DEFAULT_BATCH_SIZE
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train model with automatic device detection')
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'bert', 'llama', 'mamba', 'hyena'],
        help='Model architecture'
    )
    parser.add_argument(
        '--size',
        type=str,
        default='base',
        choices=['tiny', 'small', 'base', 'medium', 'large'],
        help='Model size'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='clmbr',
        choices=['clmbr', 'desc', 'cookbook'],
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--context-length',
        type=int,
        default=512,
        help='Context length for training'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='synthea_test',
        help='Dataset configuration name'
    )
    parser.add_argument(
        '--batch-size',
        type=str,
        default='auto',
        help='Batch size (auto for automatic detection)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--force-device',
        type=str,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Force specific device'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with smaller data'
    )
    parser.add_argument(
        '--wandb-offline',
        action='store_true',
        help='Run wandb in offline mode'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force restart training from scratch'
    )
    
    args = parser.parse_args()
    
    # Get device configuration
    if args.force_device != 'auto':
        os.environ['FORCE_DEVICE'] = args.force_device
    
    device_config = get_device_config()
    logger.info(f"Device detected: {device_config['device_name']}")
    logger.info(f"Device type: {device_config['device_type']}")
    
    # Determine batch size
    model_name = f"{args.model}-{args.size}"
    if args.batch_size == 'auto':
        batch_size = get_optimal_batch_size(
            model_name, 
            args.context_length,
            device_config['device_type']
        )
    else:
        batch_size = int(args.batch_size)
    
    logger.info(f"Using batch size: {batch_size}")
    
    # Select appropriate trainer config based on device
    if device_config['device_type'] == 'mps':
        trainer_config = 'mps'
    elif device_config['device_type'] == 'cuda':
        trainer_config = 'single_gpu'
    else:
        trainer_config = 'single_gpu'  # CPU fallback
    
    # Get output directory
    output_dir = get_run_output_dir(model_name, args.context_length, args.tokenizer)
    logger.info(f"Output directory: {output_dir}")
    
    # Build hydra command
    cmd = [
        sys.executable,
        str(Path(__file__).parent / 'run.py'),
        f'+data={args.data}',
        f'+trainer={trainer_config}',
        f'+model={model_name}',
        f'+tokenizer={args.tokenizer}',
        f'data.dataloader.batch_size={batch_size}',
        f'data.dataloader.max_length={args.context_length}',
        f'trainer.max_epochs={args.epochs}',
        f'main.path_to_output_dir={output_dir}',
        f'logging.wandb.name={model_name}-{args.context_length}-{args.tokenizer}'
    ]
    
    # Add model-specific context length settings
    if args.model == 'gpt2':
        cmd.append(f'model.config_kwargs.n_positions={args.context_length}')
    elif args.model in ['bert', 'llama']:
        cmd.append(f'model.config_kwargs.max_position_embeddings={args.context_length}')
    elif args.model == 'hyena':
        cmd.append(f'model.config_kwargs.max_seq_len={args.context_length}')
    
    # Add debug settings
    if args.debug:
        cmd.append('data.dataset.is_debug=true')
        cmd.append('trainer.limit_train_batches=10')
        cmd.append('trainer.limit_val_batches=5')
    
    # Add wandb settings
    if args.wandb_offline:
        cmd.append('logging.wandb.mode=offline')
    
    # Force refresh if requested
    if args.force_refresh:
        cmd.append('main.is_force_restart=true')
    
    # Log the command
    logger.info("Running command:")
    logger.info(" ".join(cmd))
    
    # Run the training
    try:
        result = subprocess.run(cmd, check=True)
        logger.info("Training completed successfully!")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())