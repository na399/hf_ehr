#!/usr/bin/env python3
"""
Simplified tokenizer creation for OMOP/MEDS data.
Automatically configures and creates tokenizers for the converted MEDS data.
"""

import argparse
import os
import sys
import subprocess
import json
import logging
from pathlib import Path
from typing import Optional, List
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hf_ehr.utils.config_loader import (
    MEDS_READER_DIR, CACHE_DIR
)
from hf_ehr.utils.device import get_num_workers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ensure_configs_exist():
    """Ensure necessary config files exist."""
    config_dir = Path(__file__).parent.parent / 'hf_ehr' / 'configs'
    
    # Create synthea_omop dataset config if it doesn't exist
    dataset_config = config_dir / 'data' / 'synthea_omop.yaml'
    if not dataset_config.exists():
        logger.info(f"Creating dataset config: {dataset_config}")
        dataset_config.parent.mkdir(exist_ok=True)
        dataset_config.write_text("""# @package _global_

data:
  dataset:
    name: MEDSDataset
    path_to_meds_reader_extract: ${oc.env:MEDS_READER_DIR,./data/synthea_meds_reader}
    is_debug: false
    seed: 42
""")
    
    # Create tokenizer configs with environment variables
    tokenizer_configs = {
        'clmbr_synthea.yaml': """# @package _global_

data:
  tokenizer:
    name: CLMBRTokenizer
    path_to_config: ${oc.env:TOKENIZER_CACHE_DIR,./cache/tokenizers}/clmbr_synthea/tokenizer_config.json
""",
        'desc_synthea.yaml': """# @package _global_

data:
  tokenizer:
    name: DescTokenizer
    path_to_config: ${oc.env:TOKENIZER_CACHE_DIR,./cache/tokenizers}/desc_synthea/tokenizer_config.json
""",
        'cookbook_synthea.yaml': """# @package _global_

data:
  tokenizer:
    name: CookbookTokenizer
    path_to_config: ${oc.env:TOKENIZER_CACHE_DIR,./cache/tokenizers}/cookbook_synthea/tokenizer_config.json
"""
    }
    
    tokenizer_dir = config_dir / 'tokenizer'
    for filename, content in tokenizer_configs.items():
        filepath = tokenizer_dir / filename
        if not filepath.exists():
            logger.info(f"Creating tokenizer config: {filepath}")
            filepath.write_text(content)
    
    return dataset_config, tokenizer_dir


def create_clmbr_tokenizer(
    tokenizer_config_path: Path,
    vocab_size: Optional[int] = None,
    force: bool = False
) -> bool:
    """Create CLMBR tokenizer."""
    logger.info("Creating CLMBR tokenizer...")
    
    # Check if tokenizer already exists
    output_dir = Path(os.environ.get('TOKENIZER_CACHE_DIR', './cache/tokenizers')) / 'clmbr_synthea'
    if output_dir.exists() and not force:
        logger.info(f"CLMBR tokenizer already exists at {output_dir}")
        logger.info("Use --force to recreate")
        return True
    
    # Run the creation script
    cmd = [
        sys.executable,
        'hf_ehr/tokenizers/create_clmbr.py',
        '--path_to_tokenizer_config', str(tokenizer_config_path)
    ]
    
    if vocab_size:
        cmd.extend(['--k', str(vocab_size)])
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"CLMBR tokenizer creation failed: {result.stderr}")
            return False
        
        logger.info("CLMBR tokenizer created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create CLMBR tokenizer: {e}")
        return False


def create_desc_tokenizer(
    dataset_config_path: Path,
    tokenizer_config_path: Path,
    n_procs: int = 5,
    force: bool = False
) -> bool:
    """Create Desc tokenizer."""
    logger.info("Creating Desc tokenizer...")
    
    # Check if tokenizer already exists
    output_dir = Path(os.environ.get('TOKENIZER_CACHE_DIR', './cache/tokenizers')) / 'desc_synthea'
    if output_dir.exists() and not force:
        logger.info(f"Desc tokenizer already exists at {output_dir}")
        logger.info("Use --force to recreate")
        return True
    
    # Run the creation script
    cmd = [
        sys.executable,
        'hf_ehr/tokenizers/create_desc.py',
        '--path_to_dataset_config', str(dataset_config_path),
        '--path_to_tokenizer_config', str(tokenizer_config_path),
        '--n_procs', str(n_procs)
    ]
    
    if force:
        cmd.append('--is_force_refresh')
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("This may take ~30 minutes...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Desc tokenizer creation failed: {result.stderr}")
            return False
        
        logger.info("Desc tokenizer created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Desc tokenizer: {e}")
        return False


def create_cookbook_tokenizer(
    dataset_config_path: Path,
    tokenizer_config_path: Path,
    n_buckets: int = 10,
    n_procs: int = 5,
    force: bool = False
) -> bool:
    """Create Cookbook tokenizer."""
    logger.info("Creating Cookbook tokenizer...")
    
    # Check if tokenizer already exists
    output_dir = Path(os.environ.get('TOKENIZER_CACHE_DIR', './cache/tokenizers')) / 'cookbook_synthea'
    if output_dir.exists() and not force:
        logger.info(f"Cookbook tokenizer already exists at {output_dir}")
        logger.info("Use --force to recreate")
        return True
    
    # Run the creation script
    cmd = [
        sys.executable,
        'hf_ehr/tokenizers/create_cookbook.py',
        '--path_to_dataset_config', str(dataset_config_path),
        '--path_to_tokenizer_config', str(tokenizer_config_path),
        '--n_buckets_for_numerical_range_codes', str(n_buckets),
        '--n_procs', str(n_procs)
    ]
    
    if force:
        cmd.append('--is_force_refresh')
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        logger.info("This may take ~10 minutes...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Cookbook tokenizer creation failed: {result.stderr}")
            return False
        
        logger.info("Cookbook tokenizer created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create Cookbook tokenizer: {e}")
        return False


def verify_meds_data() -> bool:
    """Verify MEDS Reader data exists."""
    meds_reader_dir = Path(MEDS_READER_DIR)
    
    if not meds_reader_dir.exists():
        logger.error(f"MEDS Reader directory not found: {meds_reader_dir}")
        logger.error("Please run: python scripts/convert_omop_to_meds.py")
        return False
    
    # Check for subject splits
    splits_file = meds_reader_dir / 'metadata' / 'subject_splits.parquet'
    if not splits_file.exists():
        logger.warning(f"Subject splits not found: {splits_file}")
        logger.warning("You may need to recreate splits")
    
    logger.info(f"Found MEDS Reader data at: {meds_reader_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Create tokenizers for OMOP/MEDS data')
    parser.add_argument(
        '--type',
        choices=['clmbr', 'desc', 'cookbook', 'all'],
        default='clmbr',
        help='Type of tokenizer to create (default: clmbr)'
    )
    parser.add_argument(
        '--vocab-size',
        type=int,
        help='Vocabulary size in thousands (for CLMBR only)'
    )
    parser.add_argument(
        '--n-buckets',
        type=int,
        default=10,
        help='Number of buckets for numerical binning (for Cookbook only)'
    )
    parser.add_argument(
        '--n-procs',
        type=int,
        default=None,
        help='Number of processes for parallel processing'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate tokenizer even if it exists'
    )
    
    args = parser.parse_args()
    
    # Auto-detect workers if not specified
    if args.n_procs is None:
        args.n_procs = get_num_workers()
    
    logger.info(f"Using {args.n_procs} worker processes")
    
    # Verify MEDS data exists
    if not verify_meds_data():
        return 1
    
    # Ensure configs exist
    dataset_config, tokenizer_dir = ensure_configs_exist()
    
    # Track success
    success = True
    start_time = time.time()
    
    # Create requested tokenizers
    if args.type in ['clmbr', 'all']:
        config_path = tokenizer_dir / 'clmbr_synthea.yaml'
        if not create_clmbr_tokenizer(config_path, args.vocab_size, args.force):
            success = False
    
    if args.type in ['desc', 'all']:
        config_path = tokenizer_dir / 'desc_synthea.yaml'
        if not create_desc_tokenizer(
            dataset_config, config_path, args.n_procs, args.force
        ):
            success = False
    
    if args.type in ['cookbook', 'all']:
        config_path = tokenizer_dir / 'cookbook_synthea.yaml'
        if not create_cookbook_tokenizer(
            dataset_config, config_path, args.n_buckets, args.n_procs, args.force
        ):
            success = False
    
    # Summary
    elapsed = time.time() - start_time
    if success:
        logger.info("=" * 60)
        logger.info(f"✓ Tokenizer creation completed in {elapsed:.1f} seconds")
        logger.info(f"Tokenizers saved to: {os.environ.get('TOKENIZER_CACHE_DIR', './cache/tokenizers')}")
        logger.info("Ready for model training!")
        return 0
    else:
        logger.error("Tokenizer creation failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())