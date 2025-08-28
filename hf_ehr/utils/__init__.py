# Utils package for HF-EHR

# Import from the parent utils.py for backward compatibility
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
if parent_dir / 'utils.py' in [Path(p) for p in sys.modules.get('hf_ehr.utils', []).__dict__.get('__path__', [])]:
    pass  # Already loaded
else:
    # Try to import from parent utils.py file
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("hf_ehr.utils_file", parent_dir / "utils.py")
        utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(utils_module)
        
        # Export commonly used functions
        get_rel_path = utils_module.get_rel_path
        get_tokenizer_info_from_config_yaml = utils_module.get_tokenizer_info_from_config_yaml
        get_dataset_info_from_config_yaml = utils_module.get_dataset_info_from_config_yaml
        lr_warmup_with_constant_plateau = utils_module.lr_warmup_with_constant_plateau
    except:
        # Fallback if import fails
        pass