"""
OMOP CDM Concept Hierarchy Utilities

This module provides functionality to work with OMOP concept hierarchies
for hierarchical code rollup in tokenization.
"""

import pandas as pd
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass
class ConceptInfo:
    """Information about an OMOP concept"""
    concept_id: int
    concept_name: str
    domain_id: str
    vocabulary_id: str
    concept_class_id: str
    concept_code: str
    standard_concept: Optional[str]


class OMOPHierarchy:
    """
    Manages OMOP concept hierarchy for code rollup operations.
    
    Provides methods to:
    - Load concept and concept_ancestor tables
    - Find parent/ancestor concepts
    - Perform hierarchical rollup for rare codes
    """
    
    def __init__(self, data_dir: str = "./data/synpuf100k_omop"):
        """
        Initialize OMOP hierarchy manager.
        
        Args:
            data_dir: Directory containing concept.parquet and concept_ancestor.parquet
        """
        self.data_dir = Path(data_dir)
        self.concept_df: Optional[pd.DataFrame] = None
        self.ancestor_df: Optional[pd.DataFrame] = None
        self.concept_lookup: Dict[int, ConceptInfo] = {}
        self.code_to_concept_id: Dict[str, int] = {}
        self.ancestor_cache: Dict[int, List[int]] = {}
        
    def load_tables(self) -> None:
        """Load concept and concept_ancestor tables from parquet files"""
        concept_path = self.data_dir / "concept.parquet"
        ancestor_path = self.data_dir / "concept_ancestor.parquet"
        
        if not concept_path.exists():
            raise FileNotFoundError(f"Concept table not found at {concept_path}")
        if not ancestor_path.exists():
            raise FileNotFoundError(f"Concept ancestor table not found at {ancestor_path}")
            
        print(f"Loading OMOP concept tables from {self.data_dir}")
        self.concept_df = pd.read_parquet(concept_path)
        self.ancestor_df = pd.read_parquet(ancestor_path)
        
        print(f"Loaded {len(self.concept_df):,} concepts and {len(self.ancestor_df):,} ancestor relationships")
        
        # Build lookup tables for fast access
        self._build_lookups()
        
    def _build_lookups(self) -> None:
        """Build fast lookup dictionaries from loaded tables"""
        print("Building concept lookup tables...")
        
        # Build concept lookup by ID
        for _, row in self.concept_df.iterrows():
            concept_info = ConceptInfo(
                concept_id=row['concept_id'],
                concept_name=row['concept_name'],
                domain_id=row['domain_id'],
                vocabulary_id=row['vocabulary_id'],
                concept_class_id=row['concept_class_id'],
                concept_code=row['concept_code'],
                standard_concept=row.get('standard_concept')
            )
            self.concept_lookup[row['concept_id']] = concept_info
            
            # Build code to concept_id lookup (vocab_id/code format)
            code_key = f"{row['vocabulary_id']}/{row['concept_code']}"
            self.code_to_concept_id[code_key] = row['concept_id']
            
        print(f"Built lookups for {len(self.concept_lookup):,} concepts")
        
    def get_concept_id(self, code: str) -> Optional[int]:
        """
        Get concept ID for a code in format 'vocabulary_id/concept_code'
        
        Args:
            code: Code in format like 'NDC/12345678901' or 'ICD9CM/250.00'
            
        Returns:
            concept_id if found, None otherwise
        """
        return self.code_to_concept_id.get(code)
        
    def get_concept_info(self, concept_id: int) -> Optional[ConceptInfo]:
        """Get concept information by concept_id"""
        return self.concept_lookup.get(concept_id)
        
    def get_ancestors(self, concept_id: int, max_level: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Get all ancestors of a concept.
        
        Args:
            concept_id: The descendant concept ID
            max_level: Maximum separation level to include (None for all)
            
        Returns:
            List of (ancestor_concept_id, min_levels_of_separation) tuples, sorted by level
        """
        if concept_id in self.ancestor_cache:
            ancestors = self.ancestor_cache[concept_id]
        else:
            # Find ancestors from ancestor table
            ancestor_rows = self.ancestor_df[self.ancestor_df['descendant_concept_id'] == concept_id]
            
            if len(ancestor_rows) == 0:
                ancestors = []
            else:
                # Sort by separation level (closest first)
                ancestor_rows = ancestor_rows.sort_values('min_levels_of_separation')
                ancestors = [(row['ancestor_concept_id'], row['min_levels_of_separation']) 
                           for _, row in ancestor_rows.iterrows()]
            
            # Cache the result
            self.ancestor_cache[concept_id] = ancestors
        
        # Filter by max_level if specified
        if max_level is not None:
            ancestors = [(anc_id, level) for anc_id, level in ancestors if level <= max_level]
            
        return ancestors
        
    def find_rollup_target(self, 
                          original_code: str,
                          target_concept_ids: Set[int],
                          max_rollup_level: int = 3) -> Optional[Tuple[int, int]]:
        """
        Find the best rollup target for a code not in the target vocabulary.
        
        Args:
            original_code: Original code in 'vocab_id/code' format
            target_concept_ids: Set of concept IDs that are in the target vocabulary
            max_rollup_level: Maximum hierarchy levels to traverse for rollup
            
        Returns:
            (target_concept_id, rollup_level) if rollup found, None otherwise
        """
        # Get concept ID for the original code
        concept_id = self.get_concept_id(original_code)
        if concept_id is None:
            return None
            
        # If the code is already in target vocabulary, no rollup needed
        if concept_id in target_concept_ids:
            return (concept_id, 0)
            
        # Find ancestors and look for one in target vocabulary
        ancestors = self.get_ancestors(concept_id, max_level=max_rollup_level)
        
        for ancestor_id, level in ancestors:
            if ancestor_id in target_concept_ids:
                return (ancestor_id, level)
                
        return None
        
    def get_high_level_categories(self) -> Dict[str, List[int]]:
        """
        Get high-level category concepts that can serve as fallback rollup targets.
        
        Returns:
            Dictionary mapping category names to list of concept IDs
        """
        categories = {
            'drug_products': [],
            'clinical_findings': [],
            'procedures': [],
            'observations': []
        }
        
        # Find top-level concepts with many descendants
        if self.ancestor_df is not None:
            # Get concepts with most descendants (likely high-level categories)
            desc_counts = self.ancestor_df[
                self.ancestor_df['min_levels_of_separation'] > 0
            ].groupby('ancestor_concept_id').size()
            
            top_ancestors = desc_counts.nlargest(50).index.tolist()
            
            for concept_id in top_ancestors:
                concept_info = self.get_concept_info(concept_id)
                if concept_info is None:
                    continue
                    
                name_lower = concept_info.concept_name.lower()
                
                # Categorize based on concept name and vocabulary
                if any(term in name_lower for term in ['drug', 'medication', 'pill', 'oral', 'injectable']):
                    categories['drug_products'].append(concept_id)
                elif any(term in name_lower for term in ['clinical', 'finding', 'disease', 'disorder']):
                    categories['clinical_findings'].append(concept_id)
                elif any(term in name_lower for term in ['procedure', 'surgery', 'operation']):
                    categories['procedures'].append(concept_id)
                elif any(term in name_lower for term in ['observation', 'measurement', 'test']):
                    categories['observations'].append(concept_id)
                    
        return categories
        
    def analyze_rollup_coverage(self, 
                               original_codes: List[str],
                               target_concept_ids: Set[int]) -> Dict[str, Any]:
        """
        Analyze how well hierarchical rollup would work for a set of codes.
        
        Args:
            original_codes: List of codes to analyze
            target_concept_ids: Target vocabulary concept IDs
            
        Returns:
            Dictionary with coverage statistics
        """
        stats = {
            'total_codes': len(original_codes),
            'codes_in_target': 0,
            'codes_with_rollup': 0,
            'codes_without_rollup': 0,
            'rollup_levels': {1: 0, 2: 0, 3: 0},
            'coverage_rate': 0.0
        }
        
        for code in original_codes:
            rollup_result = self.find_rollup_target(code, target_concept_ids)
            
            if rollup_result is None:
                concept_id = self.get_concept_id(code)
                if concept_id in target_concept_ids:
                    stats['codes_in_target'] += 1
                else:
                    stats['codes_without_rollup'] += 1
            else:
                target_id, level = rollup_result
                if level == 0:
                    stats['codes_in_target'] += 1
                else:
                    stats['codes_with_rollup'] += 1
                    if level in stats['rollup_levels']:
                        stats['rollup_levels'][level] += 1
                        
        stats['coverage_rate'] = (stats['codes_in_target'] + stats['codes_with_rollup']) / stats['total_codes']
        
        return stats


def load_omop_hierarchy(data_dir: str = "./data/synpuf100k_omop") -> OMOPHierarchy:
    """
    Convenience function to load and return a configured OMOP hierarchy manager.
    
    Args:
        data_dir: Directory containing OMOP parquet files
        
    Returns:
        Configured OMOPHierarchy instance with tables loaded
    """
    hierarchy = OMOPHierarchy(data_dir)
    hierarchy.load_tables()
    return hierarchy