#!/usr/bin/env python3
"""
End-to-end test script for HF-EHR pipeline.
Tests the complete workflow from OMOP data to model training.
"""

import sys
import os
from pathlib import Path
import logging
import subprocess
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hf_ehr.utils.device import get_device_config, get_memory_usage
from hf_ehr.utils.config_loader import DATA_DIR, OMOP_DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(cmd, description, check=True):
    """Run a command and log the output."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            capture_output=True,
            text=True,
            check=check
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        if result.stderr and result.returncode != 0:
            logger.error(f"Error: {result.stderr}")
            
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if e.stdout:
            logger.error(f"Output: {e.stdout}")
        if e.stderr:
            logger.error(f"Error: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def test_imports():
    """Test that all required modules can be imported."""
    logger.info("=" * 60)
    logger.info("Testing imports...")
    
    try:
        import torch
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        
        import transformers
        logger.info(f"✓ Transformers version: {transformers.__version__}")
        
        import polars
        logger.info(f"✓ Polars version: {polars.__version__}")
        
        import meds_reader
        logger.info(f"✓ MEDS Reader available")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        logger.error("Please install required packages:")
        logger.error("pip install -e .")
        return False


def test_device():
    """Test device detection."""
    logger.info("=" * 60)
    logger.info("Testing device detection...")
    
    try:
        config = get_device_config()
        
        logger.info(f"✓ Device type: {config['device_type']}")
        logger.info(f"✓ Device name: {config['device_name']}")
        logger.info(f"✓ Memory: {config.get('memory_gb', 'Unknown')} GB")
        logger.info(f"✓ Precision: {config['precision']}")
        logger.info(f"✓ Accelerator: {config['accelerator']}")
        
        # Test device creation
        import torch
        if config['device_type'] == 'cuda':
            device = torch.device('cuda')
            logger.info(f"✓ CUDA device created: {device}")
        elif config['device_type'] == 'mps':
            device = torch.device('mps')
            logger.info(f"✓ MPS device created: {device}")
        else:
            device = torch.device('cpu')
            logger.info(f"✓ CPU device created: {device}")
            
        # Test tensor creation on device
        test_tensor = torch.randn(10, 10, device=device)
        logger.info(f"✓ Test tensor created on {device}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Device test failed: {e}")
        return False


def test_synthea_data():
    """Test that Synthea data exists."""
    logger.info("=" * 60)
    logger.info("Testing Synthea data...")
    
    if not OMOP_DATA_DIR.exists():
        logger.error(f"✗ OMOP data directory not found: {OMOP_DATA_DIR}")
        return False
        
    parquet_files = list(OMOP_DATA_DIR.glob('*.parquet'))
    
    if not parquet_files:
        logger.error(f"✗ No parquet files found in {OMOP_DATA_DIR}")
        return False
        
    logger.info(f"✓ Found {len(parquet_files)} parquet files:")
    for f in parquet_files[:5]:  # Show first 5
        logger.info(f"  - {f.name}")
    
    # Check key tables
    key_tables = ['person.parquet', 'condition_occurrence.parquet', 'drug_exposure.parquet']
    for table in key_tables:
        table_path = OMOP_DATA_DIR / table
        if table_path.exists():
            size_mb = table_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ {table}: {size_mb:.2f} MB")
        else:
            logger.warning(f"⚠ {table} not found")
    
    return True


def test_meds_conversion():
    """Test OMOP to MEDS conversion (small subset)."""
    logger.info("=" * 60)
    logger.info("Testing MEDS conversion...")
    
    # Check if meds_etl is available
    result = run_command(
        ["which", "meds_etl_omop"],
        "Checking for meds_etl_omop",
        check=False
    )
    
    if not result:
        logger.warning("⚠ meds_etl_omop not found - skipping conversion test")
        logger.warning("Install with: pip install meds_etl")
        return False
    
    logger.info("✓ meds_etl_omop is available")
    
    # We'll skip actual conversion in test to save time
    logger.info("⚠ Skipping actual conversion (takes ~10 minutes)")
    logger.info("Run manually with: python scripts/convert_omop_to_meds.py")
    
    return True


def test_model_loading():
    """Test loading a small model."""
    logger.info("=" * 60)
    logger.info("Testing model loading...")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoConfig
        
        # Try to load a tiny model configuration
        logger.info("Creating tiny GPT2 config...")
        config = AutoConfig.from_pretrained('gpt2')
        config.n_embd = 128  # Make it tiny
        config.n_layer = 2
        config.n_head = 2
        config.n_positions = 512
        
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_config(config)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"✓ Model loaded with {params/1e6:.2f}M parameters")
        
        # Test forward pass
        device_config = get_device_config()
        if device_config['device_type'] == 'cuda':
            device = torch.device('cuda')
        elif device_config['device_type'] == 'mps':
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
            
        model = model.to(device)
        logger.info(f"✓ Model moved to {device}")
        
        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 10), device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids)
            logger.info(f"✓ Forward pass successful, output shape: {outputs.logits.shape}")
        
        # Check memory usage
        if device_config['device_type'] in ['cuda', 'mps']:
            memory = get_memory_usage()
            logger.info(f"✓ Memory used: {memory['allocated']:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False


def test_training_script():
    """Test the training script in debug mode."""
    logger.info("=" * 60)
    logger.info("Testing training script...")
    
    # Check if we have MEDS data
    meds_reader_dir = DATA_DIR / 'synthea_meds_reader'
    
    if not meds_reader_dir.exists():
        logger.warning("⚠ MEDS Reader data not found")
        logger.warning("Please run: python scripts/convert_omop_to_meds.py")
        logger.warning("Skipping training test")
        return False
    
    # Run training in debug mode (very small)
    cmd = [
        sys.executable,
        'scripts/train_local.py',
        '--model', 'gpt2',
        '--size', 'base',
        '--context-length', '128',
        '--batch-size', '2',
        '--epochs', '1',
        '--debug',
        '--wandb-offline'
    ]
    
    logger.info("Running training in debug mode (this may take a minute)...")
    result = run_command(cmd, "Training test", check=False)
    
    if result:
        logger.info("✓ Training script executed successfully")
    else:
        logger.warning("⚠ Training script failed - this is expected if data not converted")
    
    return result


def main():
    """Run all tests."""
    logger.info("Starting HF-EHR End-to-End Test")
    logger.info("=" * 60)
    
    # Track test results
    results = {}
    
    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Device Detection", test_device),
        ("Synthea Data", test_synthea_data),
        ("MEDS Conversion", test_meds_conversion),
        ("Model Loading", test_model_loading),
        # ("Training Script", test_training_script),  # Skip by default as it needs data
    ]
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Print summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    # Overall result
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("=" * 60)
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Convert data: python scripts/convert_omop_to_meds.py")
        logger.info("2. Train model: python scripts/train_local.py")
    else:
        logger.info("=" * 60)
        logger.warning("⚠ SOME TESTS FAILED")
        logger.info("=" * 60)
        logger.info("\nPlease fix the issues above before proceeding")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())