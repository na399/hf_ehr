#!/usr/bin/env python3
"""
Generate downstream task labels from Synthea MEDS data.
Adapts existing MIMIC labelers for Synthea's simpler structure.
"""

import os
import sys
import datetime
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import meds_reader
import meds
from hf_ehr.utils.config_loader import MEDS_READER_DIR, DATA_DIR
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheaMortalityLabeler:
    """
    Mortality prediction labeler for Synthea data.
    Predicts whether a patient will die during their hospital admission.
    """
    def __init__(self, time_after_admission: datetime.timedelta = datetime.timedelta(hours=24)):
        self.time_after_admission = time_after_admission
    
    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        """Generate mortality labels for a subject."""
        labels = []
        
        # Track admissions and death
        admissions = []
        death_time = None
        
        for event in subject.events:
            # Look for admission events (encounters in Synthea)
            if event.code in ['Visit/IP', 'Visit/ER', 'SNOMED/183452005']:  # Inpatient/ER visits
                if hasattr(event, 'time') and event.time:
                    admissions.append(event.time)
            
            # Check for death
            if event.code == meds.death_code or 'death' in str(event.code).lower():
                death_time = event.time if hasattr(event, 'time') else None
        
        # Generate labels for each admission
        for admission_time in admissions:
            prediction_time = admission_time + self.time_after_admission
            
            # Determine if death occurs within 30 days of admission
            is_death = False
            if death_time and death_time <= admission_time + datetime.timedelta(days=30):
                is_death = True
            
            labels.append(meds.Label(
                subject_id=subject.subject_id,
                prediction_time=prediction_time,
                boolean_value=is_death
            ))
        
        return labels


class SyntheaLengthOfStayLabeler:
    """
    Length of stay prediction labeler for Synthea data.
    Predicts whether a patient will have a long admission (>3 days).
    """
    def __init__(self, 
                 time_after_admission: datetime.timedelta = datetime.timedelta(hours=24),
                 long_stay_threshold: datetime.timedelta = datetime.timedelta(days=3)):
        self.time_after_admission = time_after_admission
        self.long_stay_threshold = long_stay_threshold
    
    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        """Generate length of stay labels for a subject."""
        labels = []
        
        # Track admission start/end pairs
        admissions = {}  # encounter_id -> (start, end)
        
        for event in subject.events:
            # Track encounter starts
            if event.code in ['Visit/IP', 'Visit/ER']:
                encounter_id = getattr(event, 'encounter_id', None) or id(event)
                if encounter_id not in admissions:
                    admissions[encounter_id] = [event.time, None]
            
            # Track encounter ends (discharge)
            if 'discharge' in str(event.code).lower():
                # Find the matching admission
                for enc_id, times in admissions.items():
                    if times[1] is None:  # Not yet ended
                        times[1] = event.time
                        break
        
        # Generate labels for complete admissions
        for encounter_id, (start_time, end_time) in admissions.items():
            if start_time and end_time:
                prediction_time = start_time + self.time_after_admission
                
                # Check if it's a long stay
                length_of_stay = end_time - start_time
                is_long_stay = length_of_stay > self.long_stay_threshold
                
                labels.append(meds.Label(
                    subject_id=subject.subject_id,
                    prediction_time=prediction_time,
                    boolean_value=is_long_stay
                ))
        
        return labels


class SyntheaReadmissionLabeler:
    """
    30-day readmission prediction labeler for Synthea data.
    """
    def __init__(self, 
                 time_after_discharge: datetime.timedelta = datetime.timedelta(hours=24),
                 readmission_window: datetime.timedelta = datetime.timedelta(days=30)):
        self.time_after_discharge = time_after_discharge
        self.readmission_window = readmission_window
    
    def label(self, subject: meds_reader.Subject) -> List[meds.Label]:
        """Generate readmission labels for a subject."""
        labels = []
        
        # Collect all admissions with times
        admissions = []
        for event in subject.events:
            if event.code in ['Visit/IP']:  # Only inpatient admissions
                if hasattr(event, 'time') and event.time:
                    admissions.append(event.time)
        
        # Sort admissions chronologically
        admissions.sort()
        
        # Check for readmissions
        for i in range(len(admissions) - 1):
            discharge_time = admissions[i] + datetime.timedelta(days=1)  # Assume 1-day stay if no discharge
            next_admission = admissions[i + 1]
            
            prediction_time = discharge_time + self.time_after_discharge
            
            # Check if readmitted within window
            is_readmitted = (next_admission - discharge_time) <= self.readmission_window
            
            labels.append(meds.Label(
                subject_id=subject.subject_id,
                prediction_time=prediction_time,
                boolean_value=is_readmitted
            ))
        
        return labels


def apply_labeler(db: meds_reader.SubjectDatabase, labeler, label_name: str) -> List[meds.Label]:
    """Apply a labeler to all subjects in the database."""
    all_labels = []
    
    # Get all subject IDs
    subject_ids = list(db)
    
    for subject_id in tqdm(subject_ids, desc=f"Generating {label_name} labels"):
        try:
            subject = db[subject_id]
            labels = labeler.label(subject)
            all_labels.extend(labels)
        except Exception as e:
            logger.warning(f"Failed to label subject {subject_id}: {e}")
            continue
    
    return all_labels


def main():
    """Generate all labels for Synthea data."""
    
    # Check if MEDS Reader data exists
    if not MEDS_READER_DIR.exists():
        logger.error(f"MEDS Reader data not found at {MEDS_READER_DIR}")
        logger.error("Please run: python scripts/convert_omop_to_meds.py")
        return 1
    
    # Create output directory
    labels_dir = DATA_DIR / 'synthea_labels'
    if labels_dir.exists():
        logger.info(f"Removing existing labels directory: {labels_dir}")
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True)
    
    # Initialize labelers
    labelers = {
        'mortality': SyntheaMortalityLabeler(
            time_after_admission=datetime.timedelta(hours=24)
        ),
        'long_los': SyntheaLengthOfStayLabeler(
            time_after_admission=datetime.timedelta(hours=24),
            long_stay_threshold=datetime.timedelta(days=3)
        ),
        'readmission': SyntheaReadmissionLabeler(
            time_after_discharge=datetime.timedelta(hours=24),
            readmission_window=datetime.timedelta(days=30)
        ),
    }
    
    # Open database and generate labels
    logger.info(f"Opening MEDS database: {MEDS_READER_DIR}")
    with meds_reader.SubjectDatabase(str(MEDS_READER_DIR), num_threads=4) as db:
        logger.info(f"Database contains {len(list(db))} subjects")
        
        for label_name, labeler in labelers.items():
            logger.info(f"\nGenerating {label_name} labels...")
            
            # Generate labels
            labels = apply_labeler(db, labeler, label_name)
            
            logger.info(f"Generated {len(labels)} {label_name} labels")
            
            if labels:
                # Save as parquet (more efficient than CSV)
                output_path = labels_dir / f'{label_name}.parquet'
                
                # Convert to PyArrow table
                label_dicts = []
                for label in labels:
                    label_dict = {
                        'subject_id': label.subject_id,
                        'prediction_time': label.prediction_time.isoformat() if label.prediction_time else None,
                        'boolean_value': label.boolean_value,
                        'numeric_value': getattr(label, 'numeric_value', None)
                    }
                    label_dicts.append(label_dict)
                
                # Create table and save
                import pandas as pd
                df = pd.DataFrame(label_dicts)
                df.to_parquet(output_path, index=False)
                
                logger.info(f"Saved {label_name} labels to {output_path}")
                
                # Print statistics
                positive_labels = sum(1 for l in labels if l.boolean_value)
                negative_labels = len(labels) - positive_labels
                logger.info(f"  Positive: {positive_labels} ({positive_labels/len(labels)*100:.1f}%)")
                logger.info(f"  Negative: {negative_labels} ({negative_labels/len(labels)*100:.1f}%)")
            else:
                logger.warning(f"No {label_name} labels generated")
    
    logger.info(f"\n✅ Label generation complete! Labels saved to {labels_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())