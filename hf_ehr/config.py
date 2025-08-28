import datetime
import json
import socket
import os
from typing import TypedDict, Dict, Optional, List, Any, Literal, Union, Tuple, Callable
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from dataclasses import dataclass, asdict, field
import logging
from tqdm import tqdm

# FEMR splits for Stanford STARR-OMOP
SPLIT_SEED: int = 97
SPLIT_TRAIN_CUTOFF: float = 70
SPLIT_VAL_CUTOFF: float = 85

def copy_file(src: str, dest: str, is_overwrite_if_exists: bool = False) -> None:
    """Copy a file or directory if it does not exist."""
    if is_overwrite_if_exists or not os.path.exists(os.path.join(dest, os.path.basename(src))):
        if os.path.isdir(src):
            logger.info(f"Copying directory from `{src}` to `{dest}`.")
            os.system(f'cp -r {src} {dest}')
        else:
            logger.info(f"Copying file from `{src}` to `{dest}`.")
            os.system(f'cp {src} {dest}')

def wrapper_with_logging(func: Callable, func_name: str, *args: Any, **kwargs: Any) -> None:
    """
    Wrapper function to log the execution of another function and optionally handle multiprocessing.
    
    Parameters:
    - func: The function to be executed.
    - func_name: The name of the function, used for logging.
    - *args, **kwargs: Arguments and keyword arguments for the function.
    
    Returns:
    - None
    """
    logging.info(f"Starting {func_name} with args={args} and kwargs={kwargs}...")
    
    try:
        # Call the function directly, let the function itself manage multiprocessing if needed
        func(*args, **kwargs)
        logging.info(f"Finished {func_name} successfully.")
    except Exception as e:
        logging.error(f"Error in {func_name}: {str(e)}")
        raise

#############################################
#
# hf_ehr's internal Event type -- generic object for representing a clinical event
#
#############################################
@dataclass()
class Event():
    code: str # LOINC/1234
    value: Optional[Any] = None # 123.45 or 'YES' or None
    unit: Optional[str] = None # mg/dL or None
    start: Optional[datetime.datetime] = None # 2023-08-13
    end: Optional[datetime.datetime] = None # 2023-08-13
    omop_table: Optional[str] = None  # 'measurement' or 'observation' or None
    
    def to_dict(self):
        return asdict(self)

#############################################
#
# Token Stats
#
#############################################
@dataclass()
class TCEStat():
    """A statistic about a token. 
    
    Examples:
        - count_occurrences: Number of times a token occurs in a dataset
        - count_patients: Number of unique patients with the token in a dataset
        - ppl: Average perplexity of the token in a dataset
    """
    type: Literal['count_occurrences', 'count_patients', 'ppl'] # type of this stat

    def to_dict(self) -> dict:
        return asdict(self)
    
@dataclass()
class CountOccurrencesTCEStat(TCEStat):
    """Count of the # of times a token occurs in a dataset split."""
    dataset: Optional[str] = None
    split: Optional[str] = None
    count: Optional[str] = None
    type: str = 'count_occurrences'

@dataclass()
class CountPatientsTCEStat(TCEStat):
    """Count of the # of unique patients with a token in a dataset split."""
    dataset: Optional[str] = None
    split: Optional[str] = None
    count: Optional[int] = None
    type: str = 'count_patients'

@dataclass()
class PPLTCEStat(TCEStat):
    """Average perplexity of a token in a dataset split."""
    dataset: Optional[str] = None
    split: Optional[str] = None
    model: Optional[str] = None
    ppl: Optional[float] = None
    type: str = 'ppl'

#############################################
#
# Token Types
#
#############################################
@dataclass()
class TokenizerConfigEntry():
    """A token in the tokenizer config. One entry per unique token. Includes metadata and stats about the token."""
    code: str # LOINC/1234 -- raw code
    type: Literal['numerical_range', 'categorical', 'code'] # type of this token
    description: Optional[str] = None # 'Glucose' -- description of the code
    tokenization: Dict[str, Any] = field(default_factory=dict) # various info helpful for tokenizing
    stats: List[TCEStat] = field(default_factory=list) # various stats about this token

    def to_dict(self) -> dict:
        return asdict(self)

    def to_token(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def get_stat(self, type_: str, settings: Optional[dict])-> Optional[TCEStat]:
        """Returns stat with specified type (required) and settings (optional)"""
        for s in self.stats:
            if s.type == type_:
                if settings is not None:
                    for key, val in settings.items():
                        if getattr(s, key) != val:
                            break
                return s
        return None

@dataclass()
class CodeTCE(TokenizerConfigEntry):
    """A vanilla code token. 
    
    Examples:
        - LOINC/1234
        - ICD10/1234
        - SNOMED/1234
    """
    type: str = 'code'

    def to_token(self) -> str:
        # LOINC/1234
        return f"{self.code}"

@dataclass()
class NumericalRangeTCE(TokenizerConfigEntry):
    """A token that represents a numerical range.
    
    Examples:
        - LOINC/1234 || mg/dL || 0.0 - 100.0
        - ICD10/1234 || kg || 0.0 - 100.0
        - SNOMED/1234 || kg || 0.0 - 100.0
    """
    type: str = 'numerical_range'
    tokenization: Dict[str, Any] = field(default_factory=lambda: {
        'unit' : None, 'range_start': None, 'range_end' : None, 
    })
    
    def to_token(self)-> str:
        # LOINC/1234 || mg/dL || 0.0 - 100.0
        return f"{self.code} || {self.tokenization['unit']} || {self.tokenization['range_start']} - {self.tokenization['range_end']}"

@dataclass()
class CategoricalTCE(TokenizerConfigEntry):
    """A token that represents a specific (set of) categorical values.
    
    Examples:
        - LOINC/1234 || category1,category2,category3
        - ICD10/1234 || category1,category2,category3
        - SNOMED/1234 || category1,category2,category3
    """
    type: str = 'categorical'
    tokenization: Dict[str, Any] = field(default_factory=lambda: {
        'categories': [] 
    })

    def to_token(self)-> str:
        # LOINC/1234 || category1,category2,category3
        return f"{self.code} || {','.join(self.tokenization['categories'])}"


#############################################
#
# Tokenizer config helpers
#
#############################################
def save_tokenizer_config_to_path(path_to_tokenizer_config: str, tokenizer_config: List[TokenizerConfigEntry], metadata: Optional[Dict] = None) -> None:
    """Given a path to a `tokenizer_config.json` file, saves the JSON config to disk."""
    json.dump({
        'timestamp' : str(datetime.datetime.now().isoformat()),
        'metadata' : metadata if metadata else {},
        'tokens' : [ x.to_dict() for x in tokenizer_config ], # NOTE: Takes ~30 seconds for 1.5M tokens
    }, open(path_to_tokenizer_config, 'w'), indent=2) # NOTE: Saving takes a few minutes

def load_tokenizer_config_and_metadata_from_path(path_to_tokenizer_config: str) -> Tuple[List[TokenizerConfigEntry], Dict[str, Any]]:
    return load_tokenizer_config_from_path(path_to_tokenizer_config, is_return_metadata=True) # type: ignore

def load_tokenizer_config_from_path(path_to_tokenizer_config: str, is_return_metadata: bool = False) -> Union[List[TokenizerConfigEntry], Tuple[List[TokenizerConfigEntry], Dict[str, Any]]]:
    """Given a path to a `tokenizer_config.json` file, loads the JSON config and parses into Python objects."""
    raw_data = json.load(open(path_to_tokenizer_config, 'r'))
    raw_metadata: Dict[str, Any] = raw_data['metadata']
    raw_tokens: Dict[str, Any] = raw_data['tokens']
    
    # Parse token Dict => TokenizerConfigEntry objects
    config: List[TokenizerConfigEntry] = []
    for entry in raw_tokens:
        raw_stats = entry.pop('stats')
        stats: List[TCEStat] = []
        for stat in raw_stats:
            if stat['type'] == 'count_occurrences':
                stats.append(CountOccurrencesTCEStat(**stat))
            elif stat['type'] == 'count_patients':
                stats.append(CountPatientsTCEStat(**stat))
            elif stat['type'] == 'ppl':
                stats.append(PPLTCEStat(**stat))
            else:
                raise ValueError(f"Unknown stat type: {stat}")
        if entry['type'] == 'code':
            config.append(CodeTCE(**entry, stats=stats))
        elif entry['type'] == 'numerical_range':
            config.append(NumericalRangeTCE(**entry, stats=stats))
        elif entry['type'] == 'categorical':
            config.append(CategoricalTCE(**entry, stats=stats))
        else:
            raise ValueError(f"Unknown token type: {entry}")
    if is_return_metadata:
        return config, raw_metadata
    else:
        return config

#############################################
#
# Evaluations
#
#############################################

EHRSHOT_LABELING_FUNCTION_2_PAPER_NAME = {
    # Guo et al. 2023
    "guo_los": "Long LOS",
    "guo_readmission": "30-day Readmission",
    "guo_icu": "ICU Admission",
    # New diagnosis
    "new_pancan": "Pancreatic Cancer",
    "new_celiac": "Celiac",
    "new_lupus": "Lupus",
    "new_acutemi": "Acute MI",
    "new_hypertension": "Hypertension",
    "new_hyperlipidemia": "Hyperlipidemia",
    # Instant lab values
    "lab_thrombocytopenia": "Thrombocytopenia",
    "lab_hyperkalemia": "Hyperkalemia",
    "lab_hypoglycemia": "Hypoglycemia",
    "lab_hyponatremia": "Hyponatremia",
    "lab_anemia": "Anemia",
    # # Custom tasks
    "chexpert": "Chest X-ray Findings",
    # MIMIC-IV tasks
    "mimic4_los" : "Long LOS (MIMIC-IV)",
    "mimic4_readmission" : "30-day Readmission (MIMIC-IV)",
    "mimic4_mortality" : "Inpatient Mortality (MIMIC-IV)",
}

EHRSHOT_TASK_GROUP_2_PAPER_NAME = {
    "operational_outcomes": "Operational Outcomes",
    "lab_values": "Anticipating Lab Test Results",
    "new_diagnoses": "Assignment of New Diagnoses",
    "chexpert": "Anticipating Chest X-ray Findings",
}

EHRSHOT_TASK_GROUP_2_LABELING_FUNCTION = {
    "operational_outcomes": [
        "guo_los",
        "guo_readmission",
        "guo_icu",
        "mimic4_los",
        "mimic4_mortality",
        "mimic4_readmission",
    ],
    "lab_values": [
        "lab_thrombocytopenia",
        "lab_hyperkalemia",
        "lab_hypoglycemia",
        "lab_hyponatremia",
        "lab_anemia"
    ],
    "new_diagnoses": [
        "new_hypertension",
        "new_hyperlipidemia",
        "new_pancan",
        "new_celiac",
        "new_lupus",
        "new_acutemi"
    ],
    "chexpert": [
        "chexpert"
    ],
}

