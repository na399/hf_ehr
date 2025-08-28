"""
Configuration loader with environment variable support.
Replaces hard-coded paths with configurable environment variables.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    logger.info(f"Loaded environment from {env_path}")
else:
    logger.info("No .env file found, using system environment variables")


def get_path(env_var: str, default: str = None) -> Path:
    """
    Get a path from environment variable with fallback.
    
    Args:
        env_var: Environment variable name
        default: Default path if env var not set
    
    Returns:
        Resolved Path object
    """
    path_str = os.getenv(env_var, default)
    if path_str is None:
        raise ValueError(f"Environment variable {env_var} not set and no default provided")
    
    path = Path(path_str).expanduser().resolve()
    
    # Create directory if it doesn't exist (except for input data)
    if env_var not in ['DATA_DIR', 'OMOP_DATA_DIR'] and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return path


# Core paths - replace hard-coded /share/pi/nigam paths
DATA_DIR = get_path('DATA_DIR', './data')
OUTPUT_DIR = get_path('OUTPUT_DIR', './outputs')
CACHE_DIR = get_path('CACHE_DIR', './cache')
LOG_DIR = get_path('LOG_DIR', './logs')

# OMOP and MEDS paths
OMOP_DATA_DIR = get_path('OMOP_DATA_DIR', DATA_DIR / 'Synthea27NjParquet')
MEDS_DATA_DIR = get_path('MEDS_DATA_DIR', DATA_DIR / 'synthea_meds')
MEDS_READER_DIR = get_path('MEDS_READER_DIR', DATA_DIR / 'synthea_meds_reader')

# Model cache paths
TRANSFORMERS_CACHE = get_path('TRANSFORMERS_CACHE', CACHE_DIR / 'transformers')
HF_HOME = get_path('HF_HOME', CACHE_DIR / 'huggingface')

# Set environment variables for libraries
os.environ['TRANSFORMERS_CACHE'] = str(TRANSFORMERS_CACHE)
os.environ['HF_HOME'] = str(HF_HOME)

# Device configuration
FORCE_DEVICE = os.getenv('FORCE_DEVICE', 'auto')

# Training configuration
DEFAULT_BATCH_SIZE = os.getenv('DEFAULT_BATCH_SIZE', 'auto')
DEFAULT_NUM_WORKERS = os.getenv('DEFAULT_NUM_WORKERS', 'auto')

# Logging configuration
WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'hf-ehr-training')
WANDB_ENTITY = os.getenv('WANDB_ENTITY', None)
WANDB_MODE = os.getenv('WANDB_MODE', 'online')

# Debug settings
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
VERBOSE = os.getenv('VERBOSE', 'false').lower() == 'true'


def get_run_output_dir(model_name: str, context_length: int, tokenizer: str) -> Path:
    """
    Get output directory for a training run.
    
    Args:
        model_name: Model name (e.g., 'mamba-tiny')
        context_length: Context length
        tokenizer: Tokenizer name
    
    Returns:
        Path to run output directory
    """
    run_name = f"{model_name}-{context_length}--{tokenizer}"
    run_dir = OUTPUT_DIR / 'runs' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_tokenizer_cache_dir(tokenizer_name: str) -> Path:
    """
    Get cache directory for tokenizer.
    
    Args:
        tokenizer_name: Name of tokenizer
    
    Returns:
        Path to tokenizer cache directory
    """
    tokenizer_dir = CACHE_DIR / 'tokenizers' / tokenizer_name
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    return tokenizer_dir


def get_slurm_log_path(job_name: str) -> Path:
    """
    Get path for SLURM logs.
    
    Args:
        job_name: Name of the job
    
    Returns:
        Path to log file
    """
    log_path = LOG_DIR / 'slurm' / f"{job_name}_%A.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


# Log configuration on import
if logger.level <= logging.INFO:
    logger.info(f"Configuration loaded:")
    logger.info(f"  DATA_DIR: {DATA_DIR}")
    logger.info(f"  OUTPUT_DIR: {OUTPUT_DIR}")
    logger.info(f"  CACHE_DIR: {CACHE_DIR}")
    logger.info(f"  LOG_DIR: {LOG_DIR}")
    logger.info(f"  FORCE_DEVICE: {FORCE_DEVICE}")