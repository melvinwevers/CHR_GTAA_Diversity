#!/usr/bin/env python3
"""
deduplication_strategy_comparison.py - Compare different deduplication strategies

This script compares the three deduplication strategies for handling multi-parent nodes:
1. maxfreq: keeps parent with highest frequency in dataset
2. longest: keeps parent on deepest branch (longest path from root)
3. first: keeps first parent encountered (original CSV order)

The comparison includes:
- Graph structure metrics
- Collection PD rankings
- Statistical correlations
- Edge removal statistics
"""

import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
import copy

from src.archival_bias_detection import ArchivalBiasDetector
from src.graph_builder import GTAAGraphBuilder


def analyze_deduplication_strategies():
    """Compare the three deduplication strategies comprehensively."""
    
    # Configuration
    gtaa_csv_path = Path("data/external/gtaa_ontology.csv")
    parquet_path = Path("data/processed/photos_archive.parquet")
    collection_col = 'fotocollectie'
    min_collection_size = 1000
    
    print("="*80)
    print("DEDUPLICATION STRATEGY COMPARISON")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=min_collection_size)
    df = detector.load_and_filter_data(parquet_path)
    
    # Calculate subject frequencies for maxfreq strategy
    print("   Calculating subject frequencies...")
    all_subjects = []
    for subjects_list in df['subjects_list'].dropna():
        if isinstance(subjects_list, list):
            all_subjects.extend(subjects_list)
    subject_counts = pd.Series(all_subjects).value_counts().to_dict()
    
    # Build initial graph (without deduplication)
    print("\n2. Building initial graph...")
    detector.build_graph(apply_deduplication=False)
    initial_graph = detector.graph.copy()
    
    # Analyze initial multi-parent nodes
    print("\n3. Analyzing initial multi-parent nodes...")
    multi_parent_nodes = [
        node for node in initial_graph.nodes() 
        if node != "_DUMMY_ROOT_" and len(list(initial_graph.predecessors(node))) > 1
    ]
    print(f"   Found {len(multi_parent_nodes)} nodes with multiple parents")
    
    # Show examples of multi-parent nodes
    print("\n   Examples of multi-parent nodes:")
    for i, node in enumerate(multi_parent_nodes[:5]):
        parents = list(initial_graph.predecessors(node))
        print(f"     {node}: {len(parents)} parents -> {parents}")
    
    # Test each strategy
    strategies = ['maxfreq', 'longest', 'first']
    results = {}
    
    print(f"\n4. Testing {len(strategies)} deduplication strategies...")
    
    for strategy in strategies:
        print(f"\n   Testing strategy: {strategy}")
        
        # Create fresh graph copy
        graph_copy = initial_graph.copy()
        builder = GTAAGraphBuilder(gtaa_csv_path)
        builder.graph = graph_copy
        
        # Apply deduplication
        if strategy == 'maxfreq':
            builder.deduplicate_parents(subject_counts, strategy='maxfreq')
        elif strategy == 'longest':
            builder.deduplicate_parents(subject_counts, strategy='longest')
        else:  # first
            builder.deduplicate_parents(subject_counts, strategy='first')
        
        # Analyze graph structure
        stats = builder.get_graph_statistics()
        
        # Calculate collection PD rankings
        collection_rankings = calculate_collection_rankings(graph_copy, df, collection_col)
        
        # Store results
        results[strategy] = {
            'graph_stats': stats,
            'collection_rankings': collection_rankings,
            'graph': graph_copy
        }
        
        print(f"     ✓ Completed {strategy}")
    
    # Compare results
    print(f"\n5. Comparing results...")
    compare_strategies(results, strategies)
    
    # Save detailed results
    save_detailed_results(results, strategies)
    
    return results


def calculate_collection_rankings(graph, df, collection_col):
    """Calculate PD rankings for collections using the given graph."""
    from src.faith_pd import FaithPDCalculator
    from src.unseen_pd import UnseenPDEstimator
    
    # Initialize calculators
    pd_calculator = FaithPDCalculator(graph)
    unseen_estimator = UnseenPDEstimator(graph)
    
    # Get collections
    collections = df[collection_col].value_counts()
    large_collections = collections[collections >= 1000]
    
    rankings = []
    
    for collection_name, collection_size in large_collections.items():
        if pd.isna(collection_name) or not str(collection_name).strip():
            continue
            
        collection_df = df[df[collection_col] == collection_name]
        
        # Extract subjects
        collection_subjects = []
        for subjects_list in collection_df['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                collection_subjects.extend(subjects_list)
        
        unique_subjects = list(set(collection_subjects))
        valid_subjects = [s for s in unique_subjects if s in graph.nodes()]
        
        if not valid_subjects:
            continue
        
        # Calculate PD
        collection_pd = pd_calculator.calculate_faith_pd(valid_subjects)
        
        # Set up node counts for unseen estimation
        for node in graph.nodes():
            graph.nodes[node]['count'] = 0
        for subjects_list in collection_df['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                for subj in subjects_list:
                    if subj in graph.nodes():
                        graph.nodes[subj]['count'] = graph.nodes[subj].get('count', 0) + 1
        
        # Calculate unseen PD
        unseen_result = unseen_estimator.estimate_undetected_pd(collection_df)
        unseen_pd = unseen_result['g0_hat']
        
        rankings.append({
            'collection': collection_name,
            'collection_size': collection_size,
            'collection_pd': collection_pd,
            'unseen_pd': unseen_pd,
            'total_pd': collection_pd + unseen_pd
        })
    
    return pd.DataFrame(rankings).sort_values('total_pd', ascending=False).reset_index(drop=True)


def compare_strategies(results, strategies):
    """Compare the results across strategies."""
    
    print("\n   GRAPH STRUCTURE COMPARISON:")
    print("   " + "-"*60)
    
    # Compare graph statistics
    for stat in ['total_nodes', 'total_edges', 'leaf_nodes', 'max_depth', 'avg_depth']:
        values = [results[s]['graph_stats'].get(stat, 'N/A') for s in strategies]
        print(f"   {stat:15}: {values[0]:>8} | {values[1]:>8} | {values[2]:>8}")
    
    print("\n   COLLECTION RANKING CORRELATIONS:")
    print("   " + "-"*60)
    
    # Calculate correlations between rankings
    for i, s1 in enumerate(strategies):
        for s2 in strategies[i+1:]:
            df1 = results[s1]['collection_rankings']
            df2 = results[s2]['collection_rankings']
            
            # Merge on collection name
            merged = pd.merge(df1, df2, on='collection', suffixes=(f'_{s1}', f'_{s2}'))
            
            # Calculate correlations for different metrics
            pd_corr, _ = spearmanr(merged[f'collection_pd_{s1}'], merged[f'collection_pd_{s2}'])
            total_corr, _ = spearmanr(merged[f'total_pd_{s1}'], merged[f'total_pd_{s2}'])
            
            print(f"   {s1} vs {s2:8}: PD={pd_corr:.3f}, Total={total_corr:.3f}")
    
    print("\n   TOP 5 COLLECTIONS BY TOTAL PD:")
    print("   " + "-"*60)
    
    for strategy in strategies:
        print(f"\n   {strategy.upper()}:")
        top5 = results[strategy]['collection_rankings'].head(5)
        for _, row in top5.iterrows():
            print(f"     {row['collection'][:40]:40} {row['total_pd']:8.1f}")


def save_detailed_results(results, strategies):
    """Save detailed results to files."""
    
    print(f"\n6. Saving detailed results...")
    
    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save collection rankings
    for strategy in strategies:
        filename = results_dir / f"collection_rankings_{strategy}.csv"
        results[strategy]['collection_rankings'].to_csv(filename, index=False)
        print(f"   Saved {filename}")
    
    # Save graph statistics comparison
    stats_comparison = []
    for strategy in strategies:
        stats = results[strategy]['graph_stats'].copy()
        stats['strategy'] = strategy
        stats_comparison.append(stats)
    
    stats_df = pd.DataFrame(stats_comparison)
    stats_filename = results_dir / "graph_statistics_comparison.csv"
    stats_df.to_csv(stats_filename, index=False)
    print(f"   Saved {stats_filename}")
    
    # Save summary report
    summary_filename = results_dir / "deduplication_strategy_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("DEDUPLICATION STRATEGY COMPARISON SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write("STRATEGIES TESTED:\n")
        f.write("- maxfreq: keeps parent with highest frequency in dataset\n")
        f.write("- longest: keeps parent on deepest branch (longest path from root)\n")
        f.write("- first: keeps first parent encountered (original CSV order)\n\n")
        
        f.write("GRAPH STRUCTURE COMPARISON:\n")
        f.write("-"*30 + "\n")
        for strategy in strategies:
            stats = results[strategy]['graph_stats']
            f.write(f"{strategy.upper()}:\n")
            f.write(f"  Total nodes: {stats['total_nodes']}\n")
            f.write(f"  Total edges: {stats['total_edges']}\n")
            f.write(f"  Leaf nodes: {stats['leaf_nodes']}\n")
            f.write(f"  Max depth: {stats.get('max_depth', 'N/A')}\n")
            f.write(f"  Avg depth: {stats.get('avg_depth', 'N/A'):.2f}\n")
            f.write(f"  Is tree: {stats['is_tree']}\n\n")
    
    print(f"   Saved {summary_filename}")


def analyze_multi_parent_examples():
    """Analyze specific examples of multi-parent nodes."""
    
    print("\n7. Analyzing multi-parent node examples...")
    
    gtaa_csv_path = Path("data/external/gtaa_ontology.csv")
    parquet_path = Path("data/processed/photos_archive.parquet")
    
    # Load data
    detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=1000)
    df = detector.load_and_filter_data(parquet_path)
    detector.build_graph(apply_deduplication=False)
    
    # Calculate frequencies
    all_subjects = []
    for subjects_list in df['subjects_list'].dropna():
        if isinstance(subjects_list, list):
            all_subjects.extend(subjects_list)
    subject_counts = pd.Series(all_subjects).value_counts().to_dict()
    
    # Find multi-parent nodes
    multi_parent_nodes = [
        node for node in detector.graph.nodes() 
        if node != "_DUMMY_ROOT_" and len(list(detector.graph.predecessors(node))) > 1
    ]
    
    print(f"   Found {len(multi_parent_nodes)} multi-parent nodes")
    
    # Analyze first 10 examples
    examples = []
    for node in multi_parent_nodes[:10]:
        parents = list(detector.graph.predecessors(node))
        
        # Get frequencies
        parent_frequencies = {p: subject_counts.get(p, 0) for p in parents}
        
        # Get path lengths
        root = "_DUMMY_ROOT_"
        parent_paths = {p: nx.shortest_path_length(detector.graph, root, p) for p in parents}
        
        examples.append({
            'node': node,
            'parents': parents,
            'parent_frequencies': parent_frequencies,
            'parent_paths': parent_paths,
            'best_by_freq': max(parents, key=lambda p: parent_frequencies[p]),
            'best_by_path': max(parents, key=lambda p: parent_paths[p]),
            'first_parent': parents[0]
        })
    
    # Create analysis DataFrame
    analysis_df = []
    for ex in examples:
        for parent in ex['parents']:
            analysis_df.append({
                'node': ex['node'],
                'parent': parent,
                'frequency': ex['parent_frequencies'][parent],
                'path_length': ex['parent_paths'][parent],
                'is_best_freq': parent == ex['best_by_freq'],
                'is_best_path': parent == ex['best_by_path'],
                'is_first': parent == ex['first_parent']
            })
    
    analysis_df = pd.DataFrame(analysis_df)
    
    # Save analysis
    analysis_filename = Path("results/multi_parent_analysis.csv")
    analysis_df.to_csv(analysis_filename, index=False)
    print(f"   Saved {analysis_filename}")
    
    # Print summary
    print(f"\n   Multi-parent node analysis summary:")
    print(f"   - Nodes analyzed: {len(examples)}")
    print(f"   - Average parents per node: {analysis_df.groupby('node').size().mean():.1f}")
    print(f"   - Frequency range: {analysis_df['frequency'].min()} - {analysis_df['frequency'].max()}")
    print(f"   - Path length range: {analysis_df['path_length'].min()} - {analysis_df['path_length'].max()}")


if __name__ == "__main__":
    # Run the comparison
    results = analyze_deduplication_strategies()
    
    # Analyze specific examples
    analyze_multi_parent_examples()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print("\nResults saved to 'results/' directory:")
    print("- collection_rankings_*.csv: Collection PD rankings for each strategy")
    print("- graph_statistics_comparison.csv: Graph structure metrics")
    print("- deduplication_strategy_summary.txt: Summary report")
    print("- multi_parent_analysis.csv: Detailed analysis of multi-parent nodes") 