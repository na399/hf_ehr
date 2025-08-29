"""
Polars-Native OMOP CDM Concept Hierarchy Utilities

This module provides high-performance, parallel-processing functionality
for working with OMOP concept hierarchies using Polars DataFrames.
"""

import polars as pl
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import os
from pathlib import Path
from tqdm import tqdm
import time


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


class PolarsOMOPHierarchy:
    """
    High-performance OMOP concept hierarchy manager using Polars for parallel processing.
    
    Provides vectorized operations for:
    - Batch concept ID resolution
    - Parallel rollup target finding
    - Efficient hierarchy traversal
    """
    
    def __init__(self, data_dir: str = "./data/synpuf100k_omop"):
        """
        Initialize Polars-based OMOP hierarchy manager.
        
        Args:
            data_dir: Directory containing concept.parquet and concept_ancestor.parquet
        """
        self.data_dir = Path(data_dir)
        
        # Use lazy frames for memory efficiency
        self.concept_lazy = None
        self.ancestor_lazy = None
        
        # Clinical tables for standard concept mappings
        self.drug_exposure_lazy = None
        self.condition_occurrence_lazy = None
        self.procedure_occurrence_lazy = None
        
        # Cached DataFrames for frequently used operations
        self.concept_df = None
        self.ancestor_df = None
        self.source_to_standard_mapping = None
        
        print(f"🚀 Initializing Polars OMOP hierarchy from {self.data_dir}")
        self._validate_files()
        self._load_lazy_frames()
        
    def _validate_files(self) -> None:
        """Validate that required OMOP files exist"""
        concept_path = self.data_dir / "concept.parquet"
        ancestor_path = self.data_dir / "concept_ancestor.parquet"
        
        if not concept_path.exists():
            raise FileNotFoundError(f"Concept table not found at {concept_path}")
        if not ancestor_path.exists():
            raise FileNotFoundError(f"Concept ancestor table not found at {ancestor_path}")
            
        print(f"✅ OMOP files validated: {concept_path.name}, {ancestor_path.name}")
        
    def _load_lazy_frames(self) -> None:
        """Load OMOP tables as lazy Polars DataFrames for memory efficiency"""
        concept_path = self.data_dir / "concept.parquet"
        ancestor_path = self.data_dir / "concept_ancestor.parquet"
        
        # Clinical tables for standard concept mappings
        drug_exposure_path = self.data_dir / "drug_exposure.parquet"
        condition_occurrence_path = self.data_dir / "condition_occurrence.parquet"
        procedure_occurrence_path = self.data_dir / "procedure_occurrence.parquet"
        
        # Create lazy frames - data not loaded until needed
        self.concept_lazy = pl.scan_parquet(str(concept_path))
        self.ancestor_lazy = pl.scan_parquet(str(ancestor_path))
        
        # Load clinical tables if they exist
        if drug_exposure_path.exists():
            self.drug_exposure_lazy = pl.scan_parquet(str(drug_exposure_path))
        
        if condition_occurrence_path.exists():
            self.condition_occurrence_lazy = pl.scan_parquet(str(condition_occurrence_path))
            
        if procedure_occurrence_path.exists():
            self.procedure_occurrence_lazy = pl.scan_parquet(str(procedure_occurrence_path))
        
        # Get table sizes for progress tracking
        concept_count = self.concept_lazy.select(pl.len()).collect().item()
        ancestor_count = self.ancestor_lazy.select(pl.len()).collect().item()
        
        # Count clinical tables
        clinical_tables = []
        if self.drug_exposure_lazy is not None:
            drug_count = self.drug_exposure_lazy.select(pl.len()).collect().item()
            clinical_tables.append(f"drug_exposure: {drug_count:,}")
            
        if self.condition_occurrence_lazy is not None:
            condition_count = self.condition_occurrence_lazy.select(pl.len()).collect().item() 
            clinical_tables.append(f"condition_occurrence: {condition_count:,}")
            
        if self.procedure_occurrence_lazy is not None:
            procedure_count = self.procedure_occurrence_lazy.select(pl.len()).collect().item()
            clinical_tables.append(f"procedure_occurrence: {procedure_count:,}")
        
        clinical_info = ", ".join(clinical_tables) if clinical_tables else "none"
        print(f"📊 Loaded lazy frames: {concept_count:,} concepts, {ancestor_count:,} ancestors")
        print(f"📊 Clinical tables: {clinical_info}")
        
    def batch_resolve_concept_ids(self, codes: List[str]) -> pl.DataFrame:
        """
        Resolve concept IDs for a batch of codes using parallel joins.
        
        Args:
            codes: List of codes in 'vocabulary_id/concept_code' format
            
        Returns:
            DataFrame with columns: code, concept_id, vocabulary_id, concept_code
        """
        if not codes:
            return pl.DataFrame(schema={"code": pl.Utf8, "concept_id": pl.Int64})
        
        print(f"🔍 Resolving concept IDs for {len(codes):,} codes using parallel joins...")
        
        with tqdm(total=3, desc="Concept ID resolution") as pbar:
            # Step 1: Create DataFrame from input codes
            codes_df = pl.DataFrame({"code": codes})
            pbar.set_description("Creating codes DataFrame")
            pbar.update(1)
            
            # Step 2: Parse vocabulary_id and concept_code from combined codes
            pbar.set_description("Parsing code components")
            
            # Safer parsing approach - filter first, then split
            codes_with_separator = codes_df.filter(pl.col("code").str.contains("/"))
            
            if codes_with_separator.height == 0:
                print("⚠️  No codes with '/' separator found")
                return pl.DataFrame(schema={"code": pl.Utf8, "concept_id": pl.Int64})
            
            # Split codes and extract parts safely
            codes_parsed_df = codes_with_separator.with_columns([
                pl.col("code").str.split_exact("/", 1).struct.field("field_0").alias("vocabulary_id"),
                pl.col("code").str.split_exact("/", 1).struct.field("field_1").alias("concept_code")
            ]).filter(
                pl.col("vocabulary_id").is_not_null() & 
                pl.col("concept_code").is_not_null() &
                (pl.col("vocabulary_id") != "") &
                (pl.col("concept_code") != "")
            )
            pbar.update(1)
            
            # Step 3: Join with concept table to get concept_ids
            pbar.set_description("Joining with concept table")
            concept_select = self.concept_lazy.select([
                "concept_id", "vocabulary_id", "concept_code", "concept_name", "domain_id"
            ]).collect()  # Collect the lazy frame first
            
            result_df = codes_parsed_df.join(
                concept_select, 
                on=["vocabulary_id", "concept_code"], 
                how="left"
            )
            pbar.update(1)
        
        found_count = result_df.filter(pl.col("concept_id").is_not_null()).height
        print(f"✅ Found concept IDs for {found_count:,}/{len(codes):,} codes ({found_count/len(codes)*100:.1f}%)")
        
        return result_df
        
    def batch_find_rollup_targets(self, 
                                excluded_concept_ids: List[int], 
                                target_concept_ids: Set[int],
                                max_rollup_level: int = 3) -> pl.DataFrame:
        """
        Find rollup targets for excluded concepts using parallel hierarchy traversal.
        
        Args:
            excluded_concept_ids: Concept IDs that need rollup targets
            target_concept_ids: Set of concept IDs in the target vocabulary
            max_rollup_level: Maximum hierarchy levels to traverse
            
        Returns:
            DataFrame with columns: descendant_concept_id, ancestor_concept_id, rollup_level
        """
        if not excluded_concept_ids or not target_concept_ids:
            return pl.DataFrame(schema={
                "descendant_concept_id": pl.Int64,
                "ancestor_concept_id": pl.Int64,
                "rollup_level": pl.Int64
            })
        
        print(f"🔄 Finding rollup targets for {len(excluded_concept_ids):,} concepts...")
        print(f"   Target vocabulary: {len(target_concept_ids):,} concepts")
        print(f"   Max rollup levels: {max_rollup_level}")
        
        with tqdm(total=4, desc="Rollup target finding") as pbar:
            # Step 1: Create DataFrames with consistent data types
            pbar.set_description("Creating input DataFrames")
            excluded_df = pl.DataFrame({"descendant_concept_id": excluded_concept_ids}, schema={"descendant_concept_id": pl.Int64})
            target_df = pl.DataFrame({"ancestor_concept_id": list(target_concept_ids)}, schema={"ancestor_concept_id": pl.Int64})
            pbar.update(1)
            
            # Step 2: Filter ancestor table for relevant relationships
            pbar.set_description("Filtering ancestor relationships")
            relevant_ancestors = (self.ancestor_lazy
                .with_columns([
                    pl.col("descendant_concept_id").cast(pl.Int64),
                    pl.col("ancestor_concept_id").cast(pl.Int64),
                    pl.col("min_levels_of_separation").cast(pl.Int64)
                ])
                .filter(
                    pl.col("descendant_concept_id").is_in(excluded_concept_ids) &
                    pl.col("ancestor_concept_id").is_in(list(target_concept_ids)) &
                    (pl.col("min_levels_of_separation") <= max_rollup_level) &
                    (pl.col("min_levels_of_separation") > 0)
                )
                .collect()
            )
            pbar.update(1)
            
            # Step 3: Find best rollup target (closest ancestor) for each descendant
            pbar.set_description("Finding closest ancestors")
            best_rollups = (relevant_ancestors
                .group_by("descendant_concept_id")
                .agg([
                    pl.col("ancestor_concept_id").sort_by("min_levels_of_separation").first().alias("ancestor_concept_id"),
                    pl.col("min_levels_of_separation").min().alias("rollup_level")
                ])
            )
            pbar.update(1)
            
            # Step 4: Join to ensure targets are valid
            pbar.set_description("Validating rollup targets")
            final_rollups = best_rollups.join(target_df, on="ancestor_concept_id", how="inner")
            pbar.update(1)
        
        rollup_count = final_rollups.height
        success_rate = rollup_count / len(excluded_concept_ids) * 100 if excluded_concept_ids else 0
        
        print(f"✅ Found rollup targets: {rollup_count:,}/{len(excluded_concept_ids):,} ({success_rate:.1f}%)")
        
        # Print rollup level distribution
        if rollup_count > 0:
            level_dist = final_rollups.group_by("rollup_level").agg(pl.count().alias("count")).sort("rollup_level")
            print("📊 Rollup level distribution:")
            for row in level_dist.iter_rows():
                level, count = row
                print(f"   Level {level}: {count:,} codes")
        
        return final_rollups
        
    def get_concept_info_batch(self, concept_ids: List[int]) -> pl.DataFrame:
        """
        Get concept information for a batch of concept IDs.
        
        Args:
            concept_ids: List of concept IDs to lookup
            
        Returns:
            DataFrame with concept information
        """
        if not concept_ids:
            return pl.DataFrame(schema={
                "concept_id": pl.Int64,
                "concept_name": pl.Utf8,
                "domain_id": pl.Utf8,
                "vocabulary_id": pl.Utf8
            })
            
        concept_ids_df = pl.DataFrame({"concept_id": concept_ids})
        
        result = (concept_ids_df
            .join(
                self.concept_lazy.select([
                    "concept_id", "concept_name", "domain_id", 
                    "vocabulary_id", "concept_class_id", "concept_code"
                ]),
                on="concept_id",
                how="left"
            )
            .collect()
        )
        
        return result
        
    def extract_source_to_standard_mappings(self) -> pl.DataFrame:
        """
        Extract source → standard concept mappings from OMOP clinical tables.
        This recovers the mappings lost during OMOP → MEDS conversion.
        
        Returns:
            DataFrame with columns: source_concept_id, standard_concept_id, domain_type
        """
        print("🔗 Extracting source → standard concept mappings from OMOP clinical tables...")
        
        all_mappings = []
        
        # Extract drug mappings: NDC → RxNorm
        if self.drug_exposure_lazy is not None:
            print("   📊 Processing drug_exposure mappings...")
            drug_mappings = (self.drug_exposure_lazy
                .select(["drug_source_concept_id", "drug_concept_id"])
                .unique()
                .filter(
                    pl.col("drug_source_concept_id").is_not_null() &
                    pl.col("drug_concept_id").is_not_null() &
                    (pl.col("drug_source_concept_id") != 0) &
                    (pl.col("drug_concept_id") != 0)
                )
                .rename({
                    "drug_source_concept_id": "source_concept_id", 
                    "drug_concept_id": "standard_concept_id"
                })
                .with_columns(pl.lit("Drug").alias("domain_type"))
                .collect()
            )
            all_mappings.append(drug_mappings)
            print(f"      ✅ Found {drug_mappings.height:,} unique drug mappings")
        
        # Extract condition mappings: ICD9CM → SNOMED  
        if self.condition_occurrence_lazy is not None:
            print("   📊 Processing condition_occurrence mappings...")
            condition_mappings = (self.condition_occurrence_lazy
                .select(["condition_source_concept_id", "condition_concept_id"])
                .unique()
                .filter(
                    pl.col("condition_source_concept_id").is_not_null() &
                    pl.col("condition_concept_id").is_not_null() &
                    (pl.col("condition_source_concept_id") != 0) &
                    (pl.col("condition_concept_id") != 0)
                )
                .rename({
                    "condition_source_concept_id": "source_concept_id",
                    "condition_concept_id": "standard_concept_id"
                })
                .with_columns(pl.lit("Condition").alias("domain_type"))
                .collect()
            )
            all_mappings.append(condition_mappings)
            print(f"      ✅ Found {condition_mappings.height:,} unique condition mappings")
        
        # Extract procedure mappings: CPT4 → SNOMED
        if self.procedure_occurrence_lazy is not None:
            print("   📊 Processing procedure_occurrence mappings...")
            procedure_mappings = (self.procedure_occurrence_lazy
                .select(["procedure_source_concept_id", "procedure_concept_id"])
                .unique()
                .filter(
                    pl.col("procedure_source_concept_id").is_not_null() &
                    pl.col("procedure_concept_id").is_not_null() &
                    (pl.col("procedure_source_concept_id") != 0) &
                    (pl.col("procedure_concept_id") != 0)
                )
                .rename({
                    "procedure_source_concept_id": "source_concept_id", 
                    "procedure_concept_id": "standard_concept_id"
                })
                .with_columns(pl.lit("Procedure").alias("domain_type"))
                .collect()
            )
            all_mappings.append(procedure_mappings)
            print(f"      ✅ Found {procedure_mappings.height:,} unique procedure mappings")
        
        # Combine all mappings
        if not all_mappings:
            print("⚠️  No clinical tables found - standard concept mapping not available")
            return pl.DataFrame(schema={
                "source_concept_id": pl.Int64,
                "standard_concept_id": pl.Int64,
                "domain_type": pl.Utf8
            })
        
        # Concatenate and deduplicate
        combined_mappings = pl.concat(all_mappings).unique(["source_concept_id", "standard_concept_id"])
        
        total_mappings = combined_mappings.height
        print(f"✅ Extracted {total_mappings:,} total source → standard concept mappings")
        
        # Print breakdown by domain
        domain_breakdown = (combined_mappings
            .group_by("domain_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        
        print("📊 Mappings by domain:")
        for row in domain_breakdown.iter_rows(named=True):
            domain = row['domain_type']
            count = row['count']
            print(f"   • {domain}: {count:,} mappings")
        
        # Cache for reuse
        self.source_to_standard_mapping = combined_mappings
        
        return combined_mappings
        
    def batch_resolve_via_standard_concepts(self, codes: List[str]) -> pl.DataFrame:
        """
        Resolve codes to their standard concepts using the source → standard mappings.
        
        Args:
            codes: List of codes in 'vocabulary_id/concept_code' format
            
        Returns:
            DataFrame with columns: code, source_concept_id, standard_concept_id, domain_type
        """
        print(f"🔍 Resolving {len(codes):,} codes to standard concepts...")
        
        # Step 1: Get source concept IDs from MEDS codes
        source_concepts_df = self.batch_resolve_concept_ids(codes)
        
        # Step 2: Load source → standard mappings if not cached
        if self.source_to_standard_mapping is None:
            self.source_to_standard_mapping = self.extract_source_to_standard_mappings()
        
        # Step 3: Join with source → standard mappings
        standard_resolution_df = (source_concepts_df
            .select(["code", "concept_id"])
            .rename({"concept_id": "source_concept_id"})
            .join(
                self.source_to_standard_mapping,
                on="source_concept_id",
                how="left"
            )
        )
        
        # Count success rates
        total_codes = standard_resolution_df.height
        with_standard = standard_resolution_df.filter(pl.col("standard_concept_id").is_not_null()).height
        success_rate = with_standard / total_codes * 100 if total_codes > 0 else 0
        
        print(f"✅ Standard concept resolution: {with_standard:,}/{total_codes:,} ({success_rate:.1f}% success)")
        
        # Print breakdown by domain
        domain_stats = (standard_resolution_df
            .filter(pl.col("domain_type").is_not_null())
            .group_by("domain_type")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )
        
        if domain_stats.height > 0:
            print("📊 Standard mappings by domain:")
            for row in domain_stats.iter_rows(named=True):
                domain = row['domain_type']
                count = row['count']
                print(f"   • {domain}: {count:,} mappings")
        
        return standard_resolution_df
        
    def analyze_rollup_coverage(self, 
                               original_codes: List[str],
                               target_concept_ids: Set[int],
                               max_rollup_level: int = 3) -> Dict[str, Any]:
        """
        Analyze rollup coverage for a set of codes using parallel processing.
        
        Args:
            original_codes: List of codes to analyze
            target_concept_ids: Target vocabulary concept IDs
            max_rollup_level: Maximum rollup levels to consider
            
        Returns:
            Dictionary with coverage statistics
        """
        print(f"📈 Analyzing rollup coverage for {len(original_codes):,} codes...")
        
        start_time = time.time()
        
        # Step 1: Resolve concept IDs for all codes
        concepts_df = self.batch_resolve_concept_ids(original_codes)
        
        # Step 2: Classify codes
        codes_with_concepts = concepts_df.filter(pl.col("concept_id").is_not_null())
        codes_in_target = codes_with_concepts.filter(pl.col("concept_id").is_in(target_concept_ids))
        excluded_concepts = codes_with_concepts.filter(~pl.col("concept_id").is_in(target_concept_ids))
        
        # Step 3: Find rollup targets for excluded codes
        rollup_results = self.batch_find_rollup_targets(
            excluded_concepts["concept_id"].to_list(),
            target_concept_ids,
            max_rollup_level
        )
        
        analysis_time = time.time() - start_time
        
        # Compile statistics
        stats = {
            'analysis_time_seconds': analysis_time,
            'total_codes': len(original_codes),
            'codes_with_concept_ids': codes_with_concepts.height,
            'codes_in_target_vocab': codes_in_target.height,
            'codes_needing_rollup': excluded_concepts.height,
            'codes_with_rollup_found': rollup_results.height,
            'codes_without_rollup': excluded_concepts.height - rollup_results.height,
            'target_vocab_rate': codes_in_target.height / len(original_codes) * 100,
            'rollup_success_rate': rollup_results.height / excluded_concepts.height * 100 if excluded_concepts.height > 0 else 0,
            'total_coverage_rate': (codes_in_target.height + rollup_results.height) / len(original_codes) * 100
        }
        
        print(f"📊 Coverage Analysis Results ({analysis_time:.2f}s):")
        print(f"   • Total codes: {stats['total_codes']:,}")
        print(f"   • In target vocab: {stats['codes_in_target_vocab']:,} ({stats['target_vocab_rate']:.1f}%)")
        print(f"   • Rollup found: {stats['codes_with_rollup_found']:,} ({stats['rollup_success_rate']:.1f}%)")
        print(f"   • Total coverage: {stats['total_coverage_rate']:.1f}%")
        
        return stats
        
    def get_high_level_categories_polars(self) -> Dict[str, List[int]]:
        """
        Get high-level category concepts using Polars aggregations.
        
        Returns:
            Dictionary mapping category names to concept ID lists
        """
        print("🏷️  Finding high-level categories using parallel aggregations...")
        
        # Find concepts with most descendants (high-level categories)
        if self.ancestor_lazy is None:
            return {}
            
        top_ancestors = (self.ancestor_lazy
            .filter(pl.col("min_levels_of_separation") > 0)
            .group_by("ancestor_concept_id")
            .agg(pl.count().alias("descendant_count"))
            .sort("descendant_count", descending=True)
            .head(50)
            .collect()
        )
        
        if top_ancestors.height == 0:
            return {}
            
        # Get concept info for categorization
        ancestor_ids = top_ancestors["ancestor_concept_id"].to_list()
        concept_info = self.get_concept_info_batch(ancestor_ids)
        
        # Categorize based on concept names
        categories = {
            'drug_products': [],
            'clinical_findings': [],
            'procedures': [],
            'observations': []
        }
        
        for row in concept_info.iter_rows(named=True):
            concept_id = row['concept_id']
            concept_name = (row['concept_name'] or '').lower()
            
            if any(term in concept_name for term in ['drug', 'medication', 'pill', 'oral', 'injectable']):
                categories['drug_products'].append(concept_id)
            elif any(term in concept_name for term in ['clinical', 'finding', 'disease', 'disorder']):
                categories['clinical_findings'].append(concept_id)
            elif any(term in concept_name for term in ['procedure', 'surgery', 'operation']):
                categories['procedures'].append(concept_id)
            elif any(term in concept_name for term in ['observation', 'measurement', 'test']):
                categories['observations'].append(concept_id)
        
        total_categories = sum(len(cat_list) for cat_list in categories.values())
        print(f"✅ Found {total_categories} high-level category concepts")
        
        return categories
        
    def get_high_level_rollup_targets(self, max_targets_per_domain: int = 500) -> pl.DataFrame:
        """
        Get high-level concepts that serve as good rollup targets.
        Focuses on standard vocabularies with rich hierarchies.
        
        Args:
            max_targets_per_domain: Maximum rollup targets per domain
            
        Returns:
            DataFrame with high-level rollup target concepts
        """
        print(f"🎯 Identifying high-level rollup targets...")
        
        if self.ancestor_lazy is None:
            return pl.DataFrame(schema={
                "concept_id": pl.Int64,
                "concept_name": pl.Utf8, 
                "vocabulary_id": pl.Utf8,
                "concept_class_id": pl.Utf8,
                "descendant_count": pl.Int64
            })
        
        # Find concepts with many descendants (good rollup targets)
        high_level_targets = (self.ancestor_lazy
            .filter(pl.col("min_levels_of_separation") > 0)
            .group_by("ancestor_concept_id")
            .agg(pl.len().alias("descendant_count"))
            .sort("descendant_count", descending=True)
        )
        
        # Get concept info for the targets
        target_concepts = (high_level_targets
            .join(
                self.concept_lazy.select([
                    "concept_id", "concept_name", "vocabulary_id", 
                    "concept_class_id", "domain_id", "standard_concept"
                ]),
                left_on="ancestor_concept_id",
                right_on="concept_id"
            )
            .filter(
                # Focus on standard concepts from key vocabularies
                pl.col("vocabulary_id").is_in(["RxNorm", "SNOMED", "LOINC"]) &
                pl.col("standard_concept").is_in(["S", "C"]) &  # Standard or Classification
                (pl.col("descendant_count") >= 50)  # Must have meaningful number of descendants
            )
            .collect()
        )
        
        if target_concepts.height == 0:
            print("⚠️  No high-level rollup targets found")
            return target_concepts
        
        # Select top targets by vocabulary/domain
        domain_targets = []
        
        # RxNorm drug targets - focus on Ingredients and Dose Form Groups
        rxnorm_targets = (target_concepts
            .filter(
                (pl.col("vocabulary_id") == "RxNorm") & 
                pl.col("concept_class_id").is_in(["Ingredient", "Dose Form Group", "Clinical Drug Comp"])
            )
            .head(max_targets_per_domain)
        )
        domain_targets.append(rxnorm_targets)
        print(f"   • RxNorm targets: {rxnorm_targets.height:,} (Ingredients, Drug Classes)")
        
        # SNOMED condition/procedure targets - high level categories  
        snomed_targets = (target_concepts
            .filter(
                (pl.col("vocabulary_id") == "SNOMED") &
                pl.col("domain_id").is_in(["Condition", "Procedure", "Observation"]) &
                (pl.col("descendant_count") >= 100)  # Higher threshold for SNOMED
            )
            .head(max_targets_per_domain)
        )
        domain_targets.append(snomed_targets)
        print(f"   • SNOMED targets: {snomed_targets.height:,} (Conditions, Procedures)")
        
        # LOINC measurement targets
        loinc_targets = (target_concepts
            .filter(
                (pl.col("vocabulary_id") == "LOINC") &
                (pl.col("descendant_count") >= 20)
            )
            .head(max_targets_per_domain // 2)  # Fewer LOINC targets needed
        )
        domain_targets.append(loinc_targets)
        print(f"   • LOINC targets: {loinc_targets.height:,} (Measurements)")
        
        # Combine all domain targets
        combined_targets = pl.concat([t for t in domain_targets if t.height > 0])
        
        print(f"✅ Found {combined_targets.height:,} total high-level rollup targets")
        
        return combined_targets


def load_polars_omop_hierarchy(data_dir: str = "./data/synpuf100k_omop") -> PolarsOMOPHierarchy:
    """
    Convenience function to load and return a Polars-based OMOP hierarchy manager.
    
    Args:
        data_dir: Directory containing OMOP parquet files
        
    Returns:
        Configured PolarsOMOPHierarchy instance
    """
    return PolarsOMOPHierarchy(data_dir)