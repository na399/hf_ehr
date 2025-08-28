"""
Device detection and configuration utility.
Provides CUDA-first support with MPS fallback for Apple Silicon.
"""
import torch
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_device_config() -> Dict[str, Any]:
    """
    Get optimal device configuration with CUDA priority.
    
    Returns:
        Dictionary with device configuration for PyTorch Lightning trainer
    """
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        logger.info(f"CUDA device detected: {device_name} with {total_memory:.1f}GB memory")
        
        return {
            'accelerator': 'gpu',
            'devices': 1 if device_count == 1 else list(range(device_count)),
            'precision': 'bf16' if torch.cuda.is_bf16_supported() else 16,
            'device_type': 'cuda',
            'device_name': device_name,
            'memory_gb': total_memory
        }
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("MPS device detected (Apple Silicon)")
        
        return {
            'accelerator': 'mps',
            'devices': 1,
            'precision': 16,  # MPS doesn't support bf16
            'device_type': 'mps',
            'device_name': 'Apple Silicon (MPS)',
            'memory_gb': 36  # Approximate for M3 Pro
        }
    
    else:
        logger.warning("No GPU detected, falling back to CPU")
        
        return {
            'accelerator': 'cpu',
            'devices': 1,
            'precision': 32,
            'device_type': 'cpu',
            'device_name': 'CPU',
            'memory_gb': None
        }


def get_optimal_batch_size(model_name: str, context_length: int = 512, device_type: str = None) -> int:
    """
    Get optimal batch size based on model, context length, and device.
    
    Args:
        model_name: Name of the model (e.g., 'mamba-tiny', 'llama-base')
        context_length: Sequence length for the model
        device_type: Override device type (if None, auto-detect)
    
    Returns:
        Recommended batch size
    """
    if device_type is None:
        device_config = get_device_config()
        device_type = device_config['device_type']
        memory_gb = device_config.get('memory_gb', 0)
    else:
        memory_gb = 48 if device_type == 'cuda' else 36
    
    # Base batch sizes for context_length=512
    batch_configs = {
        'cuda': {  # L40s with 48GB
            'mamba-tiny': 64,
            'mamba-small': 32,
            'mamba-medium': 24,
            'mamba-large': 16,
            'llama-base': 32,
            'gpt2-base': 48,
            'gpt2-medium': 32,
            'gpt2-large': 24,
            'hyena-small': 48,
            'hyena-medium': 32,
            'hyena-large': 24,
            'bert-base': 48,
        },
        'mps': {  # M3 Pro with 36GB unified memory
            'mamba-tiny': 16,
            'mamba-small': 8,
            'mamba-medium': 6,
            'mamba-large': 4,
            'llama-base': 8,
            'gpt2-base': 12,
            'gpt2-medium': 8,
            'gpt2-large': 6,
            'hyena-small': 12,
            'hyena-medium': 8,
            'hyena-large': 6,
            'bert-base': 12,
        },
        'cpu': {  # CPU fallback
            'mamba-tiny': 4,
            'mamba-small': 2,
            'llama-base': 2,
            'gpt2-base': 4,
            'hyena-large': 2,
            'bert-base': 4,
        }
    }
    
    base_batch_size = batch_configs.get(device_type, {}).get(model_name, 4)
    
    # Adjust for context length (quadratic scaling for attention models)
    if 'mamba' not in model_name:  # Mamba has linear scaling
        scaling_factor = (512 / context_length) ** 2
    else:
        scaling_factor = 512 / context_length
    
    adjusted_batch_size = max(1, int(base_batch_size * scaling_factor))
    
    logger.info(f"Optimal batch size for {model_name} on {device_type} with context_length={context_length}: {adjusted_batch_size}")
    
    return adjusted_batch_size


def get_num_workers(device_type: str = None) -> int:
    """
    Get optimal number of dataloader workers based on device.
    
    Args:
        device_type: Override device type (if None, auto-detect)
    
    Returns:
        Recommended number of workers
    """
    if device_type is None:
        device_config = get_device_config()
        device_type = device_config['device_type']
    
    worker_config = {
        'cuda': 8,   # More workers for GPU
        'mps': 4,    # Moderate for unified memory
        'cpu': 2     # Fewer for CPU
    }
    
    return worker_config.get(device_type, 4)


def synchronize_device():
    """Synchronize device for accurate timing measurements."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.synchronize()
    # CPU doesn't need synchronization


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory statistics in GB
    """
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }
    elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
        # MPS has limited memory introspection
        return {
            'allocated': torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch.mps, 'current_allocated_memory') else 0,
            'reserved': 0,
            'max_allocated': 0
        }
    else:
        return {
            'allocated': 0,
            'reserved': 0,
            'max_allocated': 0
        }