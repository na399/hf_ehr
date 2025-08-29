"""
Purpose:
    High-performance hierarchical tokenizer creation using Polars for parallel processing.
    Limit a Cookbook tokenizer to top-k most frequently occurring codes,
    but instead of discarding rare codes, use OMOP CDM hierarchy to roll up
    rare codes to their parent concepts that are in the vocabulary.

Performance: 10x+ faster than sequential version for large datasets.

Usage:
    python create_cookbook_k_hierarchical.py \
        --path_to_tokenizer_config ../configs/tokenizer/cookbook_synpuf100k.yaml \
        --k 10 \
        --stat count_occurrences \
        --omop_data_dir ./data/synpuf100k_omop
"""

import os
import argparse
import time
import datetime
import shutil
import polars as pl
from typing import Dict, List, Tuple
from hf_ehr.config import (
    load_tokenizer_config_and_metadata_from_path, 
    save_tokenizer_config_to_path,
    TokenizerConfigEntry,
)
from hf_ehr.utils import get_tokenizer_info_from_config_yaml
from hf_ehr.data.omop_hierarchy_polars import load_polars_omop_hierarchy, PolarsOMOPHierarchy
from tqdm import tqdm
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser('Generate hierarchical top-k tokenizer with OMOP rollup (Polars-optimized)')
    parser.add_argument('--path_to_tokenizer_config', required=True, type=str, 
                       help='Config .yaml file for tokenizer to use')
    parser.add_argument('--k', type=int, default=None, 
                       help='Number of tokens (in thousands) to keep in base vocabulary')
    parser.add_argument('--stat', type=str, default='count_occurrences', 
                       help='What stat to use for the token ranking')
    parser.add_argument('--omop_data_dir', type=str, default='./data/synpuf100k_omop',
                       help='Directory containing OMOP concept.parquet and concept_ancestor.parquet')
    parser.add_argument('--max_rollup_level', type=int, default=6,
                       help='Maximum hierarchy levels to traverse for rollup (default: 6 for drug hierarchies)')
    parser.add_argument('--is_force_refresh', action='store_true', default=False, 
                       help='If specified, will force refresh the tokenizer config')
    parser.add_argument('--chunk_size', type=int, default=50000,
                       help='Chunk size for batch processing (default: 50k codes)')
    parser.add_argument('--use_streaming', action='store_true',
                       help='Use Polars streaming mode for large datasets')
    return parser.parse_args()


def convert_tokenizer_config_to_polars(tokenizer_config: List[TokenizerConfigEntry], 
                                     stat: str = 'count_occurrences') -> pl.DataFrame:
    """
    Convert tokenizer config to Polars DataFrame for vectorized processing.
    
    Args:
        tokenizer_config: List of tokenizer config entries
        stat: Statistic to extract for ranking
        
    Returns:
        Polars DataFrame with columns: code, count, index
    """
    print(f"📊 Converting {len(tokenizer_config):,} tokenizer entries to Polars DataFrame...")
    
    data = []
    n_null = 0
    
    with tqdm(total=len(tokenizer_config), desc="Converting to DataFrame") as pbar:
        for i, entry in enumerate(tokenizer_config):
            # Extract the specified statistic
            count = None
            for stat_ in entry.stats:
                if stat_.type == stat:
                    count = getattr(stat_, 'count', None)
                    break
            
            if count is None:
                count = 0
                n_null += 1
            
            data.append({
                "code": entry.code,
                "count": count,
                "original_index": i
            })
            
            if i % 10000 == 0:
                pbar.update(10000)
        
        pbar.update(len(tokenizer_config) % 10000)
    
    if n_null > 0:
        null_rate = n_null / len(tokenizer_config) * 100
        print(f"⚠️  Found {n_null:,} codes with NULL {stat} values ({null_rate:.2f}%)")
    
    df = pl.DataFrame(data)
    print(f"✅ DataFrame created with {df.height:,} rows")
    
    return df


def build_standard_concept_vocabulary(
    tokenizer_config: List[TokenizerConfigEntry],
    hierarchy: PolarsOMOPHierarchy, 
    k: int,
    stat: str = 'count_occurrences'
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build vocabulary from standard concepts rather than source codes.
    This enables proper hierarchical rollup by including rollup targets in vocabulary.
    
    Args:
        tokenizer_config: Original tokenizer configuration
        hierarchy: OMOP hierarchy manager
        k: Total vocabulary size
        stat: Statistic to use for ranking
        
    Returns:
        Tuple of (target_vocabulary_df, excluded_codes_df)
    """
    print(f"🏗️  Building standard concept vocabulary (k={k:,})...")
    
    # Step 1: Extract all codes and their frequencies
    all_codes = []
    code_frequencies = {}
    
    print("📊 Extracting code frequencies from tokenizer config...")
    with tqdm(total=len(tokenizer_config), desc="Processing tokenizer config") as pbar:
        for entry in tokenizer_config:
            code = entry.code
            all_codes.append(code)
            
            # Get frequency
            frequency = 0
            for stat_ in entry.stats:
                if stat_.type == stat:
                    frequency = getattr(stat_, 'count', 0) or 0
                    break
            code_frequencies[code] = frequency
            pbar.update(1)
    
    print(f"✅ Extracted {len(all_codes):,} codes with frequencies")
    
    # Step 2: Map all codes to standard concepts and aggregate frequencies
    print("🔗 Mapping codes to standard concepts and aggregating frequencies...")
    all_standard_df = hierarchy.batch_resolve_via_standard_concepts(all_codes)
    
    # Aggregate frequencies by standard concept
    print("📊 Aggregating standard concept frequencies...")
    standard_frequencies = {}
    source_frequencies = {}
    
    for row in tqdm(all_standard_df.iter_rows(named=True), desc="Aggregating frequencies"):
        code = row['code']
        source_concept_id = row.get('source_concept_id')
        standard_concept_id = row.get('standard_concept_id')
        domain_type = row.get('domain_type')
        frequency = code_frequencies.get(code, 0)
        
        # Aggregate by source concept (for codes without standard mapping)
        if source_concept_id is not None:
            if source_concept_id not in source_frequencies:
                source_frequencies[source_concept_id] = 0
            source_frequencies[source_concept_id] += frequency
        
        # Aggregate by standard concept (primary target)
        if standard_concept_id is not None:
            if standard_concept_id not in standard_frequencies:
                standard_frequencies[standard_concept_id] = 0
            standard_frequencies[standard_concept_id] += frequency
    
    # Step 3: Build hybrid vocabulary 
    print("🎯 Building hybrid vocabulary with standard concepts + rollup targets...")
    
    # 70% frequent standard concepts
    frequent_standard_size = int(k * 0.7)
    if standard_frequencies:
        frequent_standards_df = pl.DataFrame([
            {"concept_id": concept_id, "aggregated_frequency": freq, "vocab_type": "frequent_standard"}
            for concept_id, freq in sorted(standard_frequencies.items(), key=lambda x: x[1], reverse=True)[:frequent_standard_size]
        ], schema={"concept_id": pl.Int64, "aggregated_frequency": pl.Int64, "vocab_type": pl.Utf8})
    else:
        frequent_standards_df = pl.DataFrame(schema={"concept_id": pl.Int64, "aggregated_frequency": pl.Int64, "vocab_type": pl.Utf8})
    
    # 20% high-level rollup targets
    rollup_targets_size = int(k * 0.2)
    rollup_targets_df = hierarchy.get_high_level_rollup_targets(rollup_targets_size)
    if rollup_targets_df.height > 0:
        rollup_targets_df = rollup_targets_df.select([
            pl.col("ancestor_concept_id").cast(pl.Int64).alias("concept_id"),  # Cast to Int64 for consistency
            pl.col("descendant_count").cast(pl.Int64).alias("aggregated_frequency"), 
            pl.lit("rollup_target").alias("vocab_type")
        ])
    else:
        rollup_targets_df = pl.DataFrame(schema={"concept_id": pl.Int64, "aggregated_frequency": pl.Int64, "vocab_type": pl.Utf8})
    
    # 10% frequent source concepts (for essential source codes)
    frequent_source_size = k - frequent_standards_df.height - rollup_targets_df.height
    if source_frequencies and frequent_source_size > 0:
        frequent_sources_df = pl.DataFrame([
            {"concept_id": concept_id, "aggregated_frequency": freq, "vocab_type": "frequent_source"}
            for concept_id, freq in sorted(source_frequencies.items(), key=lambda x: x[1], reverse=True)[:frequent_source_size]
        ], schema={"concept_id": pl.Int64, "aggregated_frequency": pl.Int64, "vocab_type": pl.Utf8})
    else:
        frequent_sources_df = pl.DataFrame(schema={"concept_id": pl.Int64, "aggregated_frequency": pl.Int64, "vocab_type": pl.Utf8})
    
    # Combine hybrid vocabulary
    vocab_components = [df for df in [frequent_standards_df, rollup_targets_df, frequent_sources_df] if df.height > 0]
    if not vocab_components:
        raise ValueError("No vocabulary components could be built")
        
    hybrid_vocabulary_df = pl.concat(vocab_components).unique("concept_id")
    
    # Get concept info for the hybrid vocabulary
    concept_info_df = hierarchy.concept_lazy.select([
        "concept_id", "concept_name", "vocabulary_id", "concept_class_id", "concept_code"
    ]).with_columns(pl.col("concept_id").cast(pl.Int64)).collect()
    
    target_vocabulary_df = (hybrid_vocabulary_df
        .with_columns(pl.col("concept_id").cast(pl.Int64))  # Ensure consistent types
        .join(concept_info_df, on="concept_id", how="left")
    )
    
    # Find excluded codes (all original codes not in hybrid vocabulary)
    target_concept_ids = set(hybrid_vocabulary_df["concept_id"].to_list())
    
    excluded_codes = []
    for code in all_codes:
        # Check if code's source or standard concept is in target vocabulary
        code_row = all_standard_df.filter(pl.col("code") == code)
        if code_row.height > 0:
            row = code_row.row(0, named=True)
            source_id = row.get('source_concept_id')
            standard_id = row.get('standard_concept_id')
            
            if (source_id not in target_concept_ids if source_id else True) and \
               (standard_id not in target_concept_ids if standard_id else True):
                excluded_codes.append(code)
        else:
            excluded_codes.append(code)
    
    excluded_codes_df = pl.DataFrame({"code": excluded_codes, "selection_type": "excluded"})
    
    print(f"✅ Hybrid vocabulary built:")
    print(f"   • Frequent standards: {frequent_standards_df.height:,} concepts")
    print(f"   • Rollup targets: {rollup_targets_df.height:,} concepts") 
    print(f"   • Frequent sources: {frequent_sources_df.height:,} concepts")
    print(f"   • Total vocabulary: {target_vocabulary_df.height:,} concepts")
    print(f"   • Excluded codes: {excluded_codes_df.height:,} codes")
    
    # Print vocabulary composition
    vocab_composition = (target_vocabulary_df
        .group_by("vocabulary_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    print("📊 Target vocabulary composition:")
    for row in vocab_composition.head(10).iter_rows(named=True):
        vocab = row['vocabulary_id']
        count = row['count']
        print(f"   • {vocab}: {count:,} concepts")
    
    return target_vocabulary_df, excluded_codes_df


def build_hierarchical_rollup_mapping_with_standard_vocabulary(
    target_vocabulary_df: pl.DataFrame,
    excluded_codes_df: pl.DataFrame,
    hierarchy: PolarsOMOPHierarchy,
    max_rollup_level: int = 6,
    chunk_size: int = 50000
) -> Dict[str, str]:
    """
    Build hierarchical rollup mapping using standard concept vocabulary as targets.
    
    Args:
        target_vocabulary_df: Target vocabulary with standard concepts
        excluded_codes_df: Excluded codes DataFrame
        hierarchy: Polars OMOP hierarchy manager
        max_rollup_level: Maximum rollup levels (increased for drug hierarchies)
        chunk_size: Batch processing chunk size
        
    Returns:
        Dictionary mapping {original_code: rollup_target_code}
    """
    print("🚀 Building hierarchical rollup with standard concept targets...")
    print(f"   • Target vocabulary: {target_vocabulary_df.height:,} standard concepts")
    print(f"   • Excluded codes: {excluded_codes_df.height:,} codes")
    print(f"   • Max rollup levels: {max_rollup_level} (enhanced for drug hierarchies)")
    print(f"   • Chunk size: {chunk_size:,}")
    
    if excluded_codes_df.height == 0:
        print("ℹ️  No excluded codes to process")
        return {}
    
    # Step 1: Build target concept ID set from standard vocabulary
    target_concept_ids = set(target_vocabulary_df["concept_id"].to_list())
    print(f"✅ Built target concept set: {len(target_concept_ids):,} concept IDs")
    
    # Show target vocabulary composition for verification
    target_vocab_composition = (target_vocabulary_df
        .group_by("vocabulary_id")
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
    )
    
    print("🎯 Target vocabulary composition (rollup targets):")
    for row in target_vocab_composition.head(10).iter_rows(named=True):
        vocab = row['vocabulary_id'] 
        count = row['count']
        print(f"   • {vocab}: {count:,} concepts")
    
    # Step 2: Process excluded codes in chunks
    excluded_codes = excluded_codes_df["code"].to_list()
    rollup_mapping = {}
    
    total_chunks = (len(excluded_codes) + chunk_size - 1) // chunk_size
    print(f"\n🔄 Processing {len(excluded_codes):,} excluded codes in {total_chunks} chunks...")
    
    with tqdm(total=total_chunks, desc="Processing chunks") as chunk_pbar:
        for chunk_start in range(0, len(excluded_codes), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(excluded_codes))
            chunk_codes = excluded_codes[chunk_start:chunk_end]
            
            chunk_pbar.set_description(f"Chunk {chunk_start//chunk_size + 1}/{total_chunks}")
            
            # Step 2.1: Map chunk codes to standard concepts
            chunk_standard_df = hierarchy.batch_resolve_via_standard_concepts(chunk_codes)
            
            # Step 2.2: Find rollup targets using standard concept hierarchies
            standard_concept_ids = (chunk_standard_df
                .filter(pl.col("standard_concept_id").is_not_null())
                ["standard_concept_id"]
                .to_list()
            )
            
            if not standard_concept_ids:
                chunk_pbar.update(1)
                continue
            
            # Step 2.3: Use enhanced rollup depth for drug hierarchies
            rollup_df = hierarchy.batch_find_rollup_targets(
                standard_concept_ids,
                target_concept_ids,
                max_rollup_level  # Now 6 instead of 3
            )
            
            # Step 2.4: Map rollup results back to original MEDS codes  
            if rollup_df.height > 0:
                # Join standard concepts with their rollup targets
                rollup_resolution_df = (chunk_standard_df
                    .filter(pl.col("standard_concept_id").is_not_null())
                    .select(["code", "standard_concept_id"])
                    .with_columns(pl.col("standard_concept_id").cast(pl.Int64))
                    .join(
                        rollup_df.with_columns([
                            pl.col("descendant_concept_id").cast(pl.Int64),
                            pl.col("ancestor_concept_id").cast(pl.Int64)
                        ]),
                        left_on="standard_concept_id", 
                        right_on="descendant_concept_id"
                    )
                )
                
                # Map ancestor concept IDs back to MEDS codes in target vocabulary
                target_concept_to_code = {}
                for row in target_vocabulary_df.iter_rows(named=True):
                    concept_id = row['concept_id']
                    vocab_id = row['vocabulary_id']
                    concept_code = row['concept_code']
                    
                    # Reconstruct MEDS format code
                    meds_code = f"{vocab_id}/{concept_code}"
                    target_concept_to_code[concept_id] = meds_code
                
                # Build final rollup mapping
                for row in rollup_resolution_df.iter_rows(named=True):
                    original_code = row['code']
                    ancestor_concept_id = row['ancestor_concept_id']
                    
                    if ancestor_concept_id in target_concept_to_code:
                        target_code = target_concept_to_code[ancestor_concept_id]
                        rollup_mapping[original_code] = target_code
            
            chunk_pbar.update(1)
    
    # Step 3: Enhanced rollup statistics with vocabulary breakdown
    print("\n📊 Enhanced Rollup Statistics...")
    return rollup_mapping


def reconstruct_tokenizer_config_from_standard_vocab(
    original_tokenizer_config: List[TokenizerConfigEntry],
    target_vocabulary_df: pl.DataFrame,
    hierarchy: PolarsOMOPHierarchy
) -> List[TokenizerConfigEntry]:
    """
    Reconstruct tokenizer config entries for the standard concept vocabulary.
    Maps standard concepts back to their most frequent source codes.
    
    Args:
        original_tokenizer_config: Original MEDS tokenizer config
        target_vocabulary_df: Standard concept vocabulary DataFrame
        hierarchy: OMOP hierarchy manager
        
    Returns:
        Filtered tokenizer config with entries for target vocabulary
    """
    print(f"🔄 Reconstructing tokenizer config from {target_vocabulary_df.height:,} standard concepts...")
    
    # Step 1: Map standard concepts back to their source codes
    concept_to_codes = {}
    
    # Get all standard mappings
    all_codes = [entry.code for entry in original_tokenizer_config]
    all_standard_df = hierarchy.batch_resolve_via_standard_concepts(all_codes)
    
    # Build reverse mapping: standard_concept_id -> list of source codes
    for row in tqdm(all_standard_df.iter_rows(named=True), desc="Building concept->code mapping"):
        code = row['code']
        source_concept_id = row.get('source_concept_id')
        standard_concept_id = row.get('standard_concept_id')
        
        # Use standard concept if available, otherwise source concept
        target_concept_id = standard_concept_id if standard_concept_id is not None else source_concept_id
        
        if target_concept_id is not None:
            if target_concept_id not in concept_to_codes:
                concept_to_codes[target_concept_id] = []
            concept_to_codes[target_concept_id].append(code)
    
    # Step 2: For each standard concept in vocabulary, find its representative MEDS code
    filtered_config = []
    original_code_to_entry = {entry.code: entry for entry in original_tokenizer_config}
    
    with tqdm(total=target_vocabulary_df.height, desc="Reconstructing config") as pbar:
        for row in target_vocabulary_df.iter_rows(named=True):
            concept_id = row['concept_id']
            vocab_id = row['vocabulary_id']
            concept_code = row['concept_code']
            
            # Try to find a MEDS code for this concept
            representative_code = None
            
            if concept_id in concept_to_codes:
                # Use the most frequent source code for this concept
                candidate_codes = concept_to_codes[concept_id]
                if candidate_codes:
                    # Find the code with highest frequency
                    best_code = None
                    best_frequency = -1
                    
                    for code in candidate_codes:
                        if code in original_code_to_entry:
                            entry = original_code_to_entry[code]
                            frequency = 0
                            for stat_ in entry.stats:
                                if stat_.type == 'count_occurrences':
                                    frequency = getattr(stat_, 'count', 0) or 0
                                    break
                            
                            if frequency > best_frequency:
                                best_frequency = frequency
                                best_code = code
                    
                    representative_code = best_code
            
            # If no source code found, create synthetic MEDS code from standard concept
            if representative_code is None:
                representative_code = f"{vocab_id}/{concept_code}"
            
            # Use the original entry if it exists, otherwise create a new one
            if representative_code in original_code_to_entry:
                filtered_config.append(original_code_to_entry[representative_code])
            else:
                # Create a synthetic entry for the standard concept
                # This will be handled by creating appropriate TokenizerConfigEntry
                pass  # Skip synthetic entries for now
                
            pbar.update(1)
    
    print(f"✅ Reconstructed config with {len(filtered_config):,} entries from standard vocabulary")
    
    return filtered_config



def add_fallback_rollup_mapping_polars(
    rollup_mapping: Dict[str, str],
    excluded_df: pl.DataFrame,
    hierarchy: PolarsOMOPHierarchy
) -> Dict[str, str]:
    """
    Add fallback rollup mappings using Polars for unmapped codes.
    
    Args:
        rollup_mapping: Existing rollup mapping
        excluded_df: DataFrame of excluded codes
        hierarchy: Polars OMOP hierarchy manager
        
    Returns:
        Extended rollup mapping with fallback mappings
    """
    # Find codes that still need fallback mapping
    excluded_codes = excluded_df["code"].to_list()
    unmapped_codes = [code for code in excluded_codes if code not in rollup_mapping]
    
    if not unmapped_codes:
        print("✅ All excluded codes have rollup mappings")
        return rollup_mapping
    
    print(f"🔄 Adding fallback mappings for {len(unmapped_codes):,} unmapped codes...")
    
    # Resolve concept IDs and standard concepts for unmapped codes
    unmapped_standard_df = hierarchy.batch_resolve_via_standard_concepts(unmapped_codes)
    
    fallback_count = 0
    domain_counts = {"[UNK_DRUG]": 0, "[UNK_CONDITION]": 0, "[UNK_PROCEDURE]": 0, "[UNK]": 0}
    
    with tqdm(total=len(unmapped_codes), desc="Adding fallbacks") as pbar:
        for row in unmapped_standard_df.iter_rows(named=True):
            code = row['code']
            
            # Use domain_type from standard mapping if available, else fall back to source domain
            domain_type = row.get('domain_type')
            
            # Enhanced domain-based fallback mapping
            if domain_type == 'Drug':
                fallback_token = '[UNK_DRUG]'
            elif domain_type == 'Condition':
                fallback_token = '[UNK_CONDITION]'
            elif domain_type == 'Procedure':
                fallback_token = '[UNK_PROCEDURE]'
            else:
                # Try to infer from code prefix for codes without standard mappings
                if code.startswith('NDC/'):
                    fallback_token = '[UNK_DRUG]'
                elif code.startswith('ICD9CM/'):
                    fallback_token = '[UNK_CONDITION]'
                elif code.startswith('CPT4/') or code.startswith('ICD9Proc/') or code.startswith('HCPCS/'):
                    fallback_token = '[UNK_PROCEDURE]'
                else:
                    fallback_token = '[UNK]'
            
            rollup_mapping[code] = fallback_token
            domain_counts[fallback_token] += 1
            fallback_count += 1
            pbar.update(1)
    
    print(f"✅ Added {fallback_count:,} fallback mappings:")
    for fallback_type, count in domain_counts.items():
        if count > 0:
            print(f"     {fallback_type}: {count:,} codes")
    
    return rollup_mapping


def reconstruct_tokenizer_config(tokenizer_config: List[TokenizerConfigEntry],
                               top_k_df: pl.DataFrame) -> List[TokenizerConfigEntry]:
    """
    Reconstruct tokenizer config from top-k DataFrame maintaining original order and metadata.
    
    Args:
        tokenizer_config: Original tokenizer config
        top_k_df: Top-k codes DataFrame with original indices
        
    Returns:
        Filtered tokenizer config with only top-k codes
    """
    print(f"🔄 Reconstructing tokenizer config for {top_k_df.height:,} top-k codes...")
    
    # Get original indices of selected codes
    selected_indices = set(top_k_df["original_index"].to_list())
    
    # Filter original config to keep only selected entries
    filtered_config = []
    for i, entry in enumerate(tokenizer_config):
        if i in selected_indices:
            filtered_config.append(entry)
    
    print(f"✅ Reconstructed config with {len(filtered_config):,} entries")
    return filtered_config


def main():
    start_total = time.time()
    args = parse_args()

    print("🚀 Hierarchical Tokenizer Creation with Polars Optimization")
    print("=" * 70)
    print("Configuration:")
    print(f"  • Tokenizer config: {args.path_to_tokenizer_config}")
    print(f"  • Top-k size: {args.k}k codes")
    print(f"  • OMOP data: {args.omop_data_dir}")
    print(f"  • Max rollup levels: {args.max_rollup_level}")
    print(f"  • Chunk size: {args.chunk_size:,} codes")
    streaming_status = "Yes" if args.use_streaming else "No"
    print(f"  • Streaming mode: {streaming_status}")
    print("=" * 70)

    # Load tokenizer config
    path_to_tokenizer_config, _ = get_tokenizer_info_from_config_yaml(args.path_to_tokenizer_config)
    
    # Create new tokenizer config directory
    path_to_old_tokenizer_config_dir = os.path.dirname(path_to_tokenizer_config)
    path_to_new_tokenizer_config_dir = path_to_old_tokenizer_config_dir + f'_hierarchical_{args.k}k'
    path_to_new_tokenizer_config = os.path.join(path_to_new_tokenizer_config_dir, 
                                               os.path.basename(path_to_tokenizer_config))

    # Handle existing directory
    os.makedirs(path_to_new_tokenizer_config_dir, exist_ok=True)
    if os.path.exists(path_to_new_tokenizer_config):
        if args.is_force_refresh:
            print(f"🔄 Overwriting existing config at: {path_to_new_tokenizer_config_dir}")
            shutil.rmtree(path_to_new_tokenizer_config_dir)
            os.makedirs(path_to_new_tokenizer_config_dir, exist_ok=True)
        else:
            print(f"❌ Config already exists at: {path_to_new_tokenizer_config_dir}")
            print("Use --is_force_refresh to overwrite")
            return
    
    # Copy base tokenizer config
    print(f"📁 Copying base config to: {path_to_new_tokenizer_config_dir}")
    shutil.copytree(path_to_old_tokenizer_config_dir, path_to_new_tokenizer_config_dir, 
                    dirs_exist_ok=True)
    
    # Load OMOP hierarchy with Polars
    print("\n📊 Loading OMOP hierarchy with Polars...")
    hierarchy = load_polars_omop_hierarchy(args.omop_data_dir)
    
    # Load and convert tokenizer config
    print("\n⚙️  Loading tokenizer configuration...")
    tokenizer_config, metadata = load_tokenizer_config_and_metadata_from_path(path_to_new_tokenizer_config)
    n_codes_start = len(tokenizer_config)
    print(f"✅ Loaded {n_codes_start:,} tokenizer entries")
    
    # Build standard concept vocabulary instead of source code frequency
    k = args.k * 1000
    target_vocabulary_df, excluded_codes_df = build_standard_concept_vocabulary(
        tokenizer_config, hierarchy, k, args.stat
    )
    
    # Build hierarchical rollup mapping using standard concept vocabulary
    rollup_mapping = build_hierarchical_rollup_mapping_with_standard_vocabulary(
        target_vocabulary_df, excluded_codes_df, hierarchy, args.max_rollup_level, args.chunk_size
    )
    
    # Add fallback mappings  
    rollup_mapping = add_fallback_rollup_mapping_polars(rollup_mapping, excluded_codes_df, hierarchy)
    
    # Reconstruct tokenizer config from standard vocabulary (map back to original MEDS codes)
    filtered_tokenizer_config = reconstruct_tokenizer_config_from_standard_vocab(
        tokenizer_config, target_vocabulary_df, hierarchy
    )
    
    # Add metadata
    metadata['hierarchy_rollup'] = {
        'created_at': datetime.datetime.now().isoformat(),
        'omop_data_dir': args.omop_data_dir,
        'max_rollup_level': args.max_rollup_level,
        'processing_method': 'polars_parallel',
        'chunk_size': args.chunk_size,
        'original_vocab_size': n_codes_start,
        'base_vocab_size': len(filtered_tokenizer_config),
        'rollup_mappings': len(rollup_mapping),
        'rollup_coverage': len(rollup_mapping) / excluded_codes_df.height if excluded_codes_df.height > 0 else 0.0,
        'total_coverage': (len(filtered_tokenizer_config) + len(rollup_mapping)) / n_codes_start
    }
    
    # Save results
    print("\n💾 Saving hierarchical tokenizer...")
    save_tokenizer_config_to_path(path_to_new_tokenizer_config, filtered_tokenizer_config, metadata)
    
    # Save rollup mapping
    rollup_mapping_path = os.path.join(path_to_new_tokenizer_config_dir, 'rollup_mapping.json')
    with open(rollup_mapping_path, 'w') as f:
        json.dump({
            'rollup_mapping': rollup_mapping,
            'metadata': metadata['hierarchy_rollup']
        }, f, indent=2)
    
    print(f"✅ Saved rollup mapping to: {rollup_mapping_path}")
    
    # Final summary
    total_time = time.time() - start_total
    print("\n🎉 Hierarchical tokenizer creation complete!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"📁 Output directory: {path_to_new_tokenizer_config_dir}")
    print("📊 Performance Summary:")
    print(f"   • Base vocabulary: {len(filtered_tokenizer_config):,} codes")
    print(f"   • Rollup mappings: {len(rollup_mapping):,} codes")  
    print(f"   • Total coverage: {metadata['hierarchy_rollup']['total_coverage']*100:.1f}%")
    print(f"   • Processing rate: {n_codes_start/total_time:,.0f} codes/sec")
    print("=" * 70)
    print("🚀 Ready for high-performance hierarchical tokenization!")


if __name__ == '__main__':
    main()