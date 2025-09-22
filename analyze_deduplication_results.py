#!/usr/bin/env python3
"""
analyze_deduplication_results.py - Analyze the results of deduplication strategy comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

def analyze_results():
    """Analyze the deduplication strategy comparison results."""
    
    print("="*80)
    print("DEDUPLICATION STRATEGY COMPARISON ANALYSIS")
    print("="*80)
    
    # Load results
    results_dir = Path("results")
    
    # Load collection rankings
    maxfreq_df = pd.read_csv(results_dir / "collection_rankings_maxfreq.csv")
    longest_df = pd.read_csv(results_dir / "collection_rankings_longest.csv")
    first_df = pd.read_csv(results_dir / "collection_rankings_first.csv")
    
    # Load graph statistics
    graph_stats = pd.read_csv(results_dir / "graph_statistics_comparison.csv")
    
    # Load multi-parent analysis
    multi_parent = pd.read_csv(results_dir / "multi_parent_analysis.csv")
    
    print(f"\n1. GRAPH STRUCTURE COMPARISON:")
    print("-" * 50)
    
    for _, row in graph_stats.iterrows():
        strategy = row['strategy']
        print(f"\n{strategy.upper()}:")
        print(f"  Total nodes: {row['total_nodes']:,}")
        print(f"  Total edges: {row['total_edges']:,}")
        print(f"  Leaf nodes: {row['leaf_nodes']:,}")
        print(f"  Max depth: {row['max_depth']}")
        print(f"  Avg depth: {row['avg_depth']:.2f}")
        print(f"  Is tree: {row['is_tree']}")
    
    print(f"\n2. COLLECTION RANKING CORRELATIONS:")
    print("-" * 50)
    
    # Calculate correlations between strategies
    strategies = ['maxfreq', 'longest', 'first']
    dfs = [maxfreq_df, longest_df, first_df]
    
    for i, (s1, df1) in enumerate(zip(strategies, dfs)):
        for j, (s2, df2) in enumerate(zip(strategies[i+1:], dfs[i+1:]), i+1):
            # Merge on collection name
            merged = pd.merge(df1, df2, on='collection', suffixes=(f'_{s1}', f'_{s2}'))
            
            # Calculate correlations
            pd_corr, pd_p = spearmanr(merged[f'collection_pd_{s1}'], merged[f'collection_pd_{s2}'])
            total_corr, total_p = spearmanr(merged[f'total_pd_{s1}'], merged[f'total_pd_{s2}'])
            unseen_corr, unseen_p = spearmanr(merged[f'unseen_pd_{s1}'], merged[f'unseen_pd_{s2}'])
            
            print(f"\n{s1.upper()} vs {s2.upper()}:")
            print(f"  Collection PD correlation: {pd_corr:.4f} (p={pd_p:.4f})")
            print(f"  Total PD correlation: {total_corr:.4f} (p={total_p:.4f})")
            print(f"  Unseen PD correlation: {unseen_corr:.4f} (p={unseen_p:.4f})")
    
    print(f"\n3. RANKING STABILITY ANALYSIS:")
    print("-" * 50)
    
    # Check if rankings change
    for i, (s1, df1) in enumerate(zip(strategies, dfs)):
        for j, (s2, df2) in enumerate(zip(strategies[i+1:], dfs[i+1:]), i+1):
            # Get top 5 collections from each strategy
            top5_s1 = df1.head(5)['collection'].tolist()
            top5_s2 = df2.head(5)['collection'].tolist()
            
            # Calculate overlap
            overlap = len(set(top5_s1) & set(top5_s2))
            
            print(f"\n{s1.upper()} vs {s2.upper()} - Top 5 overlap: {overlap}/5 ({overlap/5*100:.1f}%)")
            
            # Show differences
            only_s1 = set(top5_s1) - set(top5_s2)
            only_s2 = set(top5_s2) - set(top5_s1)
            
            if only_s1:
                print(f"  Only in {s1}: {list(only_s1)}")
            if only_s2:
                print(f"  Only in {s2}: {list(only_s2)}")
    
    print(f"\n4. MULTI-PARENT NODE ANALYSIS:")
    print("-" * 50)
    
    # Analyze multi-parent examples
    print(f"Total multi-parent nodes analyzed: {len(multi_parent['node'].unique())}")
    print(f"Total parent relationships: {len(multi_parent)}")
    
    # Analyze strategy agreement
    agreement_stats = {
        'freq_vs_path': (multi_parent['is_best_freq'] == multi_parent['is_best_path']).mean(),
        'freq_vs_first': (multi_parent['is_best_freq'] == multi_parent['is_first']).mean(),
        'path_vs_first': (multi_parent['is_best_path'] == multi_parent['is_first']).mean()
    }
    
    print(f"\nStrategy agreement rates:")
    print(f"  maxfreq vs longest: {agreement_stats['freq_vs_path']:.1%}")
    print(f"  maxfreq vs first: {agreement_stats['freq_vs_path']:.1%}")
    print(f"  longest vs first: {agreement_stats['path_vs_first']:.1%}")
    
    # Show examples where strategies disagree
    disagreements = multi_parent[
        ~((multi_parent['is_best_freq'] == multi_parent['is_best_path']) & 
          (multi_parent['is_best_freq'] == multi_parent['is_first']))
    ]
    
    if len(disagreements) > 0:
        print(f"\nExamples where strategies disagree:")
        for node in disagreements['node'].unique()[:5]:
            node_data = disagreements[disagreements['node'] == node]
            print(f"\n  {node}:")
            for _, row in node_data.iterrows():
                strategy = []
                if row['is_best_freq']: strategy.append('maxfreq')
                if row['is_best_path']: strategy.append('longest')
                if row['is_first']: strategy.append('first')
                print(f"    {row['parent']} (freq={row['frequency']}, path={row['path_length']}) -> {strategy}")
    
    print(f"\n5. STATISTICAL SUMMARY:")
    print("-" * 50)
    
    # Calculate summary statistics
    summary_stats = []
    for strategy, df in zip(strategies, dfs):
        summary_stats.append({
            'strategy': strategy,
            'mean_collection_pd': df['collection_pd'].mean(),
            'std_collection_pd': df['collection_pd'].std(),
            'mean_total_pd': df['total_pd'].mean(),
            'std_total_pd': df['total_pd'].std(),
            'mean_unseen_pd': df['unseen_pd'].mean(),
            'std_unseen_pd': df['unseen_pd'].std()
        })
    
    summary_df = pd.DataFrame(summary_stats)
    print(summary_df.to_string(index=False, float_format='%.2f'))
    
    print(f"\n6. RECOMMENDATIONS:")
    print("-" * 50)
    
    # Analyze which strategy to recommend
    print("Based on the analysis:")
    
    # Check if correlations are high
    high_correlation = True
    for i, (s1, df1) in enumerate(zip(strategies, dfs)):
        for j, (s2, df2) in enumerate(zip(strategies[i+1:], dfs[i+1:]), i+1):
            merged = pd.merge(df1, df2, on='collection', suffixes=(f'_{s1}', f'_{s2}'))
            corr, _ = spearmanr(merged[f'total_pd_{s1}'], merged[f'total_pd_{s2}'])
            if corr < 0.95:
                high_correlation = False
    
    if high_correlation:
        print("✓ All strategies produce highly correlated results (r > 0.95)")
        print("✓ Choice of strategy has minimal impact on collection rankings")
        print("✓ Recommend using 'maxfreq' as it's most data-driven")
    else:
        print("⚠ Strategy choice affects results significantly")
        print("⚠ Consider the specific use case when choosing strategy")
    
    # Check graph structure differences
    max_depth_diff = graph_stats['max_depth'].max() - graph_stats['max_depth'].min()
    avg_depth_diff = graph_stats['avg_depth'].max() - graph_stats['avg_depth'].min()
    
    if max_depth_diff == 0 and avg_depth_diff < 0.1:
        print("✓ Graph structure is nearly identical across strategies")
    else:
        print("⚠ Graph structure varies between strategies")
    
    return {
        'graph_stats': graph_stats,
        'collection_rankings': {s: df for s, df in zip(strategies, dfs)},
        'multi_parent_analysis': multi_parent,
        'summary_stats': summary_df
    }


if __name__ == "__main__":
    results = analyze_results()
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80) 