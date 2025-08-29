"""
Hierarchical Cache Utilities

This module provides utilities for optimizing hierarchical tokenizer performance
through cache warming, statistics collection, and performance analysis.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter, defaultdict
from pathlib import Path
import time
from tqdm import tqdm


class HierarchicalCacheAnalyzer:
    """Analyze dataset statistics to optimize hierarchical cache warming"""
    
    def __init__(self):
        self.code_frequencies = Counter()
        self.rollup_usage = defaultdict(int)
        
    def analyze_dataset_codes(self, dataset, tokenizer=None, max_patients: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze code frequencies in a dataset to optimize cache warming.
        
        Args:
            dataset: MEDS or FEMR dataset to analyze
            tokenizer: Tokenizer to check for rollup mappings
            max_patients: Limit analysis to first N patients for speed
            
        Returns:
            Dictionary with code frequency analysis
        """
        print("🔍 Analyzing dataset for hierarchical cache optimization...")
        
        start_time = time.time()
        total_patients = min(dataset.get_n_patients(), max_patients) if max_patients else dataset.get_n_patients()
        total_events = 0
        
        # Track codes and their frequencies
        code_frequencies = Counter()
        rollup_needed = set()
        base_vocab_hits = set()
        
        # Analyze patient events
        for patient_idx in tqdm(range(total_patients), desc="Analyzing patients"):
            try:
                _, events = dataset[patient_idx]
                total_events += len(events)
                
                for event in events:
                    code = event.code
                    code_frequencies[code] += 1
                    
                    # Check if code would need rollup
                    if tokenizer:
                        if hasattr(tokenizer, 'code_2_token') and code in tokenizer.code_2_token:
                            base_vocab_hits.add(code)
                        elif hasattr(tokenizer, 'rollup_mapping') and tokenizer.rollup_mapping and code in tokenizer.rollup_mapping:
                            rollup_needed.add(code)
                            
            except Exception as e:
                print(f"Warning: Error analyzing patient {patient_idx}: {e}")
                continue
        
        analysis_time = time.time() - start_time
        
        # Generate statistics
        most_frequent = code_frequencies.most_common(1000)
        
        stats = {
            'analysis_time_seconds': analysis_time,
            'total_patients_analyzed': total_patients,
            'total_events': total_events,
            'unique_codes': len(code_frequencies),
            'most_frequent_codes': [code for code, freq in most_frequent],
            'code_frequencies': dict(most_frequent),
            'base_vocab_codes': len(base_vocab_hits),
            'rollup_codes': len(rollup_needed),
            'events_per_patient_avg': total_events / total_patients if total_patients > 0 else 0
        }
        
        print(f"✅ Analysis complete in {analysis_time:.2f}s:")
        print(f"   • {total_patients:,} patients, {total_events:,} events")
        print(f"   • {len(code_frequencies):,} unique codes")
        print(f"   • {len(base_vocab_hits):,} base vocabulary codes")
        print(f"   • {len(rollup_needed):,} codes needing rollup")
        
        return stats
        
    def generate_cache_warming_strategy(self, 
                                      code_frequencies: Dict[str, int],
                                      cache_size: int = 10000,
                                      strategy: str = 'frequency') -> List[str]:
        """
        Generate optimal cache warming strategy based on code frequencies.
        
        Args:
            code_frequencies: Dictionary of code -> frequency
            cache_size: Size of cache to optimize for
            strategy: Strategy to use ('frequency', 'pareto', 'adaptive')
            
        Returns:
            List of codes to pre-warm, ordered by priority
        """
        if strategy == 'frequency':
            # Simple frequency-based ordering
            sorted_codes = sorted(code_frequencies.items(), key=lambda x: x[1], reverse=True)
            return [code for code, freq in sorted_codes[:cache_size]]
            
        elif strategy == 'pareto':
            # Focus on codes that account for 80% of total frequency
            total_frequency = sum(code_frequencies.values())
            target_frequency = total_frequency * 0.8
            
            sorted_codes = sorted(code_frequencies.items(), key=lambda x: x[1], reverse=True)
            cumulative_freq = 0
            warm_codes = []
            
            for code, freq in sorted_codes:
                warm_codes.append(code)
                cumulative_freq += freq
                if cumulative_freq >= target_frequency:
                    break
                    
            return warm_codes[:cache_size]
            
        elif strategy == 'adaptive':
            # Adaptive strategy that considers both frequency and diversity
            sorted_codes = sorted(code_frequencies.items(), key=lambda x: x[1], reverse=True)
            
            # Take top frequent codes
            high_freq_codes = [code for code, freq in sorted_codes[:cache_size // 2]]
            
            # Add diverse medium-frequency codes
            remaining_codes = [code for code, freq in sorted_codes[cache_size // 2:]]
            vocab_diversity = defaultdict(list)
            
            for code in remaining_codes:
                vocab_prefix = code.split('/')[0] if '/' in code else 'OTHER'
                vocab_diversity[vocab_prefix].append(code)
            
            # Add diverse codes from each vocabulary
            diverse_codes = []
            remaining_slots = cache_size - len(high_freq_codes)
            codes_per_vocab = max(1, remaining_slots // len(vocab_diversity))
            
            for vocab, codes in vocab_diversity.items():
                diverse_codes.extend(codes[:codes_per_vocab])
                
            return high_freq_codes + diverse_codes[:remaining_slots]
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def save_cache_analysis(analysis: Dict[str, Any], output_path: str) -> None:
    """Save cache analysis results to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Counter objects to regular dicts for JSON serialization
    serializable_analysis = {}
    for key, value in analysis.items():
        if isinstance(value, Counter):
            serializable_analysis[key] = dict(value)
        else:
            serializable_analysis[key] = value
            
    with open(output_path, 'w') as f:
        json.dump(serializable_analysis, f, indent=2)
        
    print(f"💾 Cache analysis saved to: {output_path}")


def load_cache_analysis(analysis_path: str) -> Dict[str, Any]:
    """Load cache analysis results from JSON file"""
    with open(analysis_path, 'r') as f:
        analysis = json.load(f)
        
    print(f"📂 Cache analysis loaded from: {analysis_path}")
    return analysis


def benchmark_cache_performance(tokenizer, 
                               test_codes: List[str], 
                               iterations: int = 10000) -> Dict[str, float]:
    """
    Benchmark cache performance with different cache warming strategies.
    
    Args:
        tokenizer: HierarchicalCookbookTokenizer instance
        test_codes: List of codes to benchmark
        iterations: Number of lookup iterations to perform
        
    Returns:
        Performance statistics
    """
    from hf_ehr.data.datasets import Event
    
    print(f"🏃‍♂️ Benchmarking cache performance with {len(test_codes)} codes...")
    
    # Clear cache before benchmark
    if hasattr(tokenizer, 'clear_cache'):
        tokenizer.clear_cache()
    
    # Create test events
    test_events = [
        Event(code=code, value=None, unit=None, start=None, end=None, omop_table=None)
        for code in test_codes
    ]
    
    # Benchmark cold cache performance
    start_time = time.time()
    for _ in tqdm(range(iterations), desc="Cold cache", leave=False):
        for event in test_events:
            _ = tokenizer.convert_event_to_token(event)
    cold_time = time.time() - start_time
    
    # Get cache stats after cold run
    if hasattr(tokenizer, 'get_cache_stats'):
        cold_stats = tokenizer.get_cache_stats()
    else:
        cold_stats = {}
    
    # Benchmark warm cache performance (run again, cache should be warm)
    start_time = time.time()
    for _ in tqdm(range(iterations), desc="Warm cache", leave=False):
        for event in test_events:
            _ = tokenizer.convert_event_to_token(event)
    warm_time = time.time() - start_time
    
    # Get final cache stats
    if hasattr(tokenizer, 'get_cache_stats'):
        warm_stats = tokenizer.get_cache_stats()
    else:
        warm_stats = {}
    
    # Calculate performance metrics
    total_lookups = iterations * len(test_events)
    cold_throughput = total_lookups / cold_time
    warm_throughput = total_lookups / warm_time
    speedup = warm_throughput / cold_throughput
    
    results = {
        'cold_time_seconds': cold_time,
        'warm_time_seconds': warm_time,
        'cold_throughput_ops_per_sec': cold_throughput,
        'warm_throughput_ops_per_sec': warm_throughput,
        'speedup_factor': speedup,
        'total_lookups': total_lookups,
        'cold_cache_stats': cold_stats,
        'warm_cache_stats': warm_stats
    }
    
    print(f"🏁 Benchmark complete:")
    print(f"   • Cold cache: {cold_throughput:,.0f} ops/sec")
    print(f"   • Warm cache: {warm_throughput:,.0f} ops/sec")
    print(f"   • Speedup: {speedup:.2f}x")
    
    return results


def create_cache_warming_script(tokenizer_path: str, 
                               analysis_path: str,
                               output_script_path: str,
                               strategy: str = 'frequency') -> None:
    """
    Create a standalone script for cache warming based on analysis results.
    
    Args:
        tokenizer_path: Path to tokenizer config
        analysis_path: Path to cache analysis results
        output_script_path: Where to save the warming script
        strategy: Cache warming strategy to use
    """
    # Load analysis results
    analysis = load_cache_analysis(analysis_path)
    
    # Generate cache warming strategy
    analyzer = HierarchicalCacheAnalyzer()
    warm_codes = analyzer.generate_cache_warming_strategy(
        analysis['code_frequencies'],
        strategy=strategy
    )
    
    # Create script content
    script_content = f'''#!/usr/bin/env python3
"""
Auto-generated hierarchical tokenizer cache warming script.
Generated from analysis: {analysis_path}
Strategy: {strategy}
"""

import sys
sys.path.append('/path/to/hf_ehr')  # Adjust as needed

from hf_ehr.data.tokenization import HierarchicalCookbookTokenizer

def warm_tokenizer_cache():
    """Warm the hierarchical tokenizer cache with optimal codes"""
    
    # Load tokenizer
    tokenizer = HierarchicalCookbookTokenizer("{tokenizer_path}")
    
    # Codes to warm (ordered by priority)
    warm_codes = {warm_codes}
    
    print(f"🔥 Warming tokenizer cache with {{len(warm_codes)}} codes...")
    
    # Warm the cache
    tokenizer.warm_hierarchical_cache(warm_codes)
    
    # Print cache statistics
    tokenizer.print_cache_stats()
    
    return tokenizer

if __name__ == "__main__":
    tokenizer = warm_tokenizer_cache()
    print("✅ Cache warming complete!")
'''

    # Save script
    output_path = Path(output_script_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(script_content)
        
    # Make executable
    os.chmod(output_path, 0o755)
    
    print(f"📜 Cache warming script created: {output_path}")
    print(f"   • Strategy: {strategy}")
    print(f"   • Warm codes: {len(warm_codes):,}")