#!/usr/bin/env python3
"""
Convert OMOP CDM v5.4 parquet files to MEDS format.
Optimized for Synthea test data but works with any OMOP CDM v5.4 dataset.
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import subprocess
import time
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hf_ehr.utils.config_loader import (
    OMOP_DATA_DIR, MEDS_DATA_DIR, MEDS_READER_DIR
)
from hf_ehr.utils.device import get_device_config, get_num_workers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_omop_data(data_dir: Path) -> bool:
    """
    Check if OMOP CDM data exists and has required tables.
    
    Args:
        data_dir: Path to OMOP data directory
    
    Returns:
        True if valid OMOP data found
    """
    required_tables = [
        'person.parquet',
        'condition_occurrence.parquet',
        'drug_exposure.parquet',
        'measurement.parquet',
        'observation.parquet',
        'visit_occurrence.parquet'
    ]
    
    if not data_dir.exists():
        logger.error(f"OMOP data directory does not exist: {data_dir}")
        return False
    
    missing_tables = []
    for table in required_tables:
        table_path = data_dir / table
        if not table_path.exists():
            missing_tables.append(table)
    
    if missing_tables:
        logger.warning(f"Missing OMOP tables: {missing_tables}")
        logger.warning("Continuing anyway - conversion may work with partial data")
    
    found_tables = [f for f in data_dir.glob('*.parquet')]
    logger.info(f"Found {len(found_tables)} parquet files in {data_dir}")
    
    return len(found_tables) > 0


def convert_omop_to_meds(
    input_dir: Path,
    output_dir: Path,
    num_workers: Optional[int] = None,
    force: bool = False
) -> bool:
    """
    Convert OMOP CDM data to MEDS format using meds_etl.
    
    Args:
        input_dir: Path to OMOP data directory
        output_dir: Path for MEDS output
        num_workers: Number of worker processes
        force: Force overwrite if output exists
    
    Returns:
        True if conversion successful
    """
    # Check if output already exists
    if output_dir.exists() and not force:
        logger.warning(f"MEDS output directory already exists: {output_dir}")
        logger.warning("Use --force to overwrite")
        return False
    
    # Auto-detect optimal workers if not specified
    if num_workers is None:
        device_config = get_device_config()
        if device_config['device_type'] == 'cuda':
            num_workers = 16  # Higher for GPU systems
        elif device_config['device_type'] == 'mps':
            num_workers = 4   # Lower for M3 Pro
        else:
            num_workers = 2   # Minimal for CPU
    
    logger.info(f"Converting OMOP to MEDS with {num_workers} workers")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    
    # Remove existing output if force
    if output_dir.exists() and force:
        logger.info(f"Removing existing output directory: {output_dir}")
        subprocess.run(['rm', '-rf', str(output_dir)], check=False)
    
    # Run meds_etl_omop conversion
    start_time = time.time()
    try:
        cmd = [
            'meds_etl_omop',
            str(input_dir),
            str(output_dir),
            '--num_proc', str(num_workers)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"MEDS conversion failed: {result.stderr}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"MEDS conversion completed in {elapsed:.1f} seconds")
        
    except FileNotFoundError:
        logger.error("meds_etl_omop not found. Please install: pip install meds_etl")
        return False
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False
    
    return True


def convert_meds_to_reader(
    meds_dir: Path,
    reader_dir: Path,
    num_workers: Optional[int] = None,
    force: bool = False
) -> bool:
    """
    Convert MEDS data to MEDS Reader format for efficient training.
    
    Args:
        meds_dir: Path to MEDS data directory
        reader_dir: Path for MEDS Reader output
        num_workers: Number of worker processes
        force: Force overwrite if output exists
    
    Returns:
        True if conversion successful
    """
    # Check if output already exists
    if reader_dir.exists() and not force:
        logger.warning(f"MEDS Reader directory already exists: {reader_dir}")
        logger.warning("Use --force to overwrite")
        return False
    
    # Auto-detect optimal workers if not specified
    if num_workers is None:
        num_workers = get_num_workers()
    
    logger.info(f"Converting MEDS to MEDS Reader with {num_workers} workers")
    logger.info(f"Input: {meds_dir}")
    logger.info(f"Output: {reader_dir}")
    
    # Remove existing output if force
    if reader_dir.exists() and force:
        logger.info(f"Removing existing output directory: {reader_dir}")
        subprocess.run(['rm', '-rf', str(reader_dir)], check=False)
    
    # Run meds_reader_convert
    start_time = time.time()
    try:
        cmd = [
            'meds_reader_convert',
            str(meds_dir),
            str(reader_dir),
            '--num_threads', str(num_workers)
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"MEDS Reader conversion failed: {result.stderr}")
            return False
        
        elapsed = time.time() - start_time
        logger.info(f"MEDS Reader conversion completed in {elapsed:.1f} seconds")
        
    except FileNotFoundError:
        logger.error("meds_reader_convert not found. Please install: pip install meds_reader")
        return False
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return False
    
    # Verify the conversion
    try:
        cmd = ['meds_reader_verify', str(meds_dir), str(reader_dir)]
        logger.info("Verifying MEDS Reader conversion...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("MEDS Reader verification successful")
        else:
            logger.warning(f"MEDS Reader verification failed: {result.stderr}")
            
    except Exception as e:
        logger.warning(f"Could not verify conversion: {e}")
    
    return True


def create_splits(
    reader_dir: Path,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> bool:
    """
    Create train/val/test splits for MEDS Reader data.
    
    Args:
        reader_dir: Path to MEDS Reader directory
        train_split: Proportion for training (0-1)
        val_split: Proportion for validation (0-1)
    
    Returns:
        True if splits created successfully
    """
    logger.info(f"Creating splits: train={train_split:.0%}, val={val_split:.0%}, test={1-train_split-val_split:.0%}")
    
    # Use the existing split_meds_dataset.py script
    split_script = Path(__file__).parent / 'datasets' / 'split_meds_dataset.py'
    
    if not split_script.exists():
        # Create the split script inline if it doesn't exist
        logger.info("Creating split script...")
        split_script.parent.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            '-c',
            f"""
import meds_reader
import polars as pl
import os

database = meds_reader.SubjectDatabase('{reader_dir}')
subject_ids = list(database)
n_patients = len(subject_ids)

splits = [
    ('train' if idx < {train_split} * n_patients else 
     'tuning' if idx < ({train_split} + {val_split}) * n_patients else 
     'held_out', subject_ids[idx])
    for idx in range(len(subject_ids))
]

df = pl.DataFrame(splits, schema=["split", "subject_id"])
print(f"Total patients: {{n_patients}}")
print(f"Train patients: {{df.filter(pl.col('split') == 'train').shape[0]}}")
print(f"Val patients: {{df.filter(pl.col('split') == 'tuning').shape[0]}}")
print(f"Test patients: {{df.filter(pl.col('split') == 'held_out').shape[0]}}")

os.makedirs(os.path.join('{reader_dir}', 'metadata'), exist_ok=True)
df.write_parquet(os.path.join('{reader_dir}', 'metadata', 'subject_splits.parquet'))
"""
        ]
    else:
        cmd = [
            sys.executable,
            str(split_script),
            '--path_to_meds_reader', str(reader_dir),
            '--train_split_size', str(train_split),
            '--val_split_size', str(val_split)
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to create splits: {result.stderr}")
            return False
        
        logger.info("Splits created successfully")
        if result.stdout:
            logger.info(result.stdout)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to create splits: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert OMOP CDM to MEDS format')
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=OMOP_DATA_DIR,
        help=f'Input OMOP data directory (default: {OMOP_DATA_DIR})'
    )
    parser.add_argument(
        '--meds-output', '-m',
        type=Path,
        default=MEDS_DATA_DIR,
        help=f'MEDS output directory (default: {MEDS_DATA_DIR})'
    )
    parser.add_argument(
        '--reader-output', '-r',
        type=Path,
        default=MEDS_READER_DIR,
        help=f'MEDS Reader output directory (default: {MEDS_READER_DIR})'
    )
    parser.add_argument(
        '--num-workers', '-n',
        type=int,
        default=None,
        help='Number of worker processes (default: auto-detect)'
    )
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.8,
        help='Training set proportion (default: 0.8)'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Validation set proportion (default: 0.1)'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force overwrite existing outputs'
    )
    parser.add_argument(
        '--skip-reader',
        action='store_true',
        help='Skip MEDS Reader conversion (only do OMOP->MEDS)'
    )
    
    args = parser.parse_args()
    
    # Log device configuration
    device_config = get_device_config()
    logger.info(f"Running on: {device_config['device_name']}")
    
    # Step 1: Check OMOP data
    logger.info("=" * 60)
    logger.info("Step 1: Checking OMOP data")
    if not check_omop_data(args.input):
        logger.error("Invalid OMOP data, exiting")
        return 1
    
    # Step 2: Convert OMOP to MEDS
    logger.info("=" * 60)
    logger.info("Step 2: Converting OMOP to MEDS")
    if not convert_omop_to_meds(
        args.input, 
        args.meds_output,
        args.num_workers,
        args.force
    ):
        logger.error("OMOP to MEDS conversion failed")
        return 1
    
    if args.skip_reader:
        logger.info("Skipping MEDS Reader conversion as requested")
        return 0
    
    # Step 3: Convert MEDS to MEDS Reader
    logger.info("=" * 60)
    logger.info("Step 3: Converting MEDS to MEDS Reader")
    if not convert_meds_to_reader(
        args.meds_output,
        args.reader_output,
        args.num_workers,
        args.force
    ):
        logger.error("MEDS to Reader conversion failed")
        return 1
    
    # Step 4: Create train/val/test splits
    logger.info("=" * 60)
    logger.info("Step 4: Creating train/val/test splits")
    if not create_splits(
        args.reader_output,
        args.train_split,
        args.val_split
    ):
        logger.error("Failed to create splits")
        return 1
    
    logger.info("=" * 60)
    logger.info("✓ Conversion pipeline completed successfully!")
    logger.info(f"  MEDS data: {args.meds_output}")
    logger.info(f"  MEDS Reader: {args.reader_output}")
    logger.info("Ready for training!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())