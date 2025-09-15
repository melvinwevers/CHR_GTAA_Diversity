#!/usr/bin/env python3
"""
Analyze completeness ratios in detail to provide accurate statistics for the paper.
"""

import pandas as pd
import numpy as np

def analyze_completeness_patterns():
    """Analyze completeness patterns and improvement potential."""
    
    # Load the data
    results_df = pd.read_csv('results/archival_bias_detailed.csv')
    
    print("COMPLETENESS RATIO DETAILED ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    completeness_stats = {
        'min': results_df['completeness_ratio'].min(),
        'max': results_df['completeness_ratio'].max(),
        'mean': results_df['completeness_ratio'].mean(),
        'std': results_df['completeness_ratio'].std(),
        'median': results_df['completeness_ratio'].median()
    }
    
    print(f"BASIC STATISTICS:")
    print(f"  Range: {completeness_stats['min']:.3f} to {completeness_stats['max']:.3f} ({completeness_stats['min']*100:.1f}% to {completeness_stats['max']*100:.1f}%)")
    print(f"  Mean: {completeness_stats['mean']:.3f} ({completeness_stats['mean']*100:.1f}%)")
    print(f"  Standard Deviation: {completeness_stats['std']:.3f}")
    print(f"  Median: {completeness_stats['median']:.3f} ({completeness_stats['median']*100:.1f}%)")
    
    # Count collections by completeness level
    total_collections = len(results_df)
    
    # Different completeness thresholds
    above_90 = len(results_df[results_df['completeness_ratio'] > 0.90])
    above_95 = len(results_df[results_df['completeness_ratio'] > 0.95])
    above_85 = len(results_df[results_df['completeness_ratio'] > 0.85])
    above_80 = len(results_df[results_df['completeness_ratio'] > 0.80])
    
    print(f"\nCOMPLETENESS DISTRIBUTION:")
    print(f"  Total collections: {total_collections}")
    print(f"  >95% completeness: {above_95} collections ({above_95/total_collections*100:.1f}%)")
    print(f"  >90% completeness: {above_90} collections ({above_90/total_collections*100:.1f}%)")
    print(f"  >85% completeness: {above_85} collections ({above_85/total_collections*100:.1f}%)")
    print(f"  >80% completeness: {above_80} collections ({above_80/total_collections*100:.1f}%)")
    
    # Improvement potential analysis
    print(f"\nIMPROVEMENT POTENTIAL ANALYSIS:")
    
    # Calculate improvement potential (100% - current completeness)
    results_df = results_df.copy()
    results_df['improvement_potential'] = 1.0 - results_df['completeness_ratio']
    results_df['improvement_potential_pct'] = results_df['improvement_potential'] * 100
    
    # Categories based on improvement potential
    near_perfect = results_df[results_df['improvement_potential_pct'] <= 5]  # ≤5% improvement
    minor_improvement = results_df[(results_df['improvement_potential_pct'] > 5) & (results_df['improvement_potential_pct'] <= 10)]  # 5-10%
    moderate_improvement = results_df[(results_df['improvement_potential_pct'] > 10) & (results_df['improvement_potential_pct'] <= 15)]  # 10-15%
    major_improvement = results_df[results_df['improvement_potential_pct'] > 15]  # >15%
    
    print(f"  Near-perfect utilization (≤5% potential): {len(near_perfect)} collections")
    print(f"  Minor improvement potential (5-10%): {len(minor_improvement)} collections")
    print(f"  Moderate improvement potential (10-15%): {len(moderate_improvement)} collections")
    print(f"  Major improvement potential (>15%): {len(major_improvement)} collections")
    
    # Show collections in each category
    print(f"\nDETAILED BREAKDOWN:")
    
    print(f"\nNEAR-PERFECT UTILIZATION (≤5% improvement potential):")
    for _, row in near_perfect.iterrows():
        print(f"  • {row['subcollection']}: {row['completeness_ratio']:.3f} ({row['completeness_ratio']*100:.1f}%, {row['improvement_potential_pct']:.1f}% potential)")
    
    print(f"\nMINOR IMPROVEMENT POTENTIAL (5-10%):")
    for _, row in minor_improvement.iterrows():
        print(f"  • {row['subcollection']}: {row['completeness_ratio']:.3f} ({row['completeness_ratio']*100:.1f}%, {row['improvement_potential_pct']:.1f}% potential)")
    
    print(f"\nMODERATE IMPROVEMENT POTENTIAL (10-15%):")
    for _, row in moderate_improvement.iterrows():
        print(f"  • {row['subcollection']}: {row['completeness_ratio']:.3f} ({row['completeness_ratio']*100:.1f}%, {row['improvement_potential_pct']:.1f}% potential)")
    
    print(f"\nMAJOR IMPROVEMENT POTENTIAL (>15%):")
    for _, row in major_improvement.iterrows():
        print(f"  • {row['subcollection']}: {row['completeness_ratio']:.3f} ({row['completeness_ratio']*100:.1f}%, {row['improvement_potential_pct']:.1f}% potential)")
    
    # Generate corrected text
    print(f"\n" + "=" * 60)
    print(f"CORRECTED TEXT FOR PAPER:")
    print(f"=" * 60)
    
    print(f"""
Completeness estimates indicate utilization ranging from {completeness_stats['min']*100:.1f}% to {completeness_stats['max']*100:.1f}% of potentially relevant GTAA terms within respective subject domains, with {above_90} of {total_collections} collections achieving greater than 90% completeness. These estimates reveal modest improvement potential: {len(minor_improvement)} collections could add 5-10% additional terms, {len(near_perfect)} demonstrate near-perfect utilization (>95%), and {len(moderate_improvement)} show potential for 10-15% enhancement. However, as noted in our methodological constraints, these "improvement opportunities" may reflect either incomplete term application or GTAA's inadequate coverage of collection subject matter—our analysis cannot distinguish between these scenarios.

The relatively tight clustering around high completeness levels (mean = {completeness_stats['mean']*100:.1f}%, σ = {completeness_stats['std']:.3f}) demonstrates systematic institutional vocabulary utilization strategies rather than variable application practices. Catalogers consistently prioritize thorough application of available domain-appropriate terminology, though this scope remains bounded by GTAA's representational coverage of their specific subject domains. This pattern suggests that observed completeness variations primarily reflect differences in collection-vocabulary alignment rather than inconsistent cataloging practices across institutions.
""")

if __name__ == "__main__":
    analyze_completeness_patterns() 