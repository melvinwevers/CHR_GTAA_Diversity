#!/usr/bin/env python3
"""
Analyze different threshold options for archetype categorization.
"""

import pandas as pd
import numpy as np

def analyze_threshold_options():
    """Analyze different threshold options for archetype categorization."""
    
    # Load the data
    results_df = pd.read_csv('results/archival_bias_detailed.csv')
    
    print("THRESHOLD OPTION ANALYSIS")
    print("=" * 60)
    
    # Calculate different thresholds
    coverage_mean = results_df['coverage_ratio'].mean()
    coverage_median = results_df['coverage_ratio'].median()
    coverage_q1 = results_df['coverage_ratio'].quantile(0.25)
    coverage_q3 = results_df['coverage_ratio'].quantile(0.75)
    
    completeness_mean = results_df['completeness_ratio'].mean()
    completeness_median = results_df['completeness_ratio'].median()
    completeness_q1 = results_df['completeness_ratio'].quantile(0.25)
    completeness_q3 = results_df['completeness_ratio'].quantile(0.75)
    
    print(f"COVERAGE RATIO THRESHOLDS:")
    print(f"  Q1 (25th percentile): {coverage_q1:.4f} ({coverage_q1*100:.1f}%)")
    print(f"  Median: {coverage_median:.4f} ({coverage_median*100:.1f}%)")
    print(f"  Mean: {coverage_mean:.4f} ({coverage_mean*100:.1f}%)")
    print(f"  Q3 (75th percentile): {coverage_q3:.4f} ({coverage_q3*100:.1f}%)")
    
    print(f"\nCOMPLETENESS RATIO THRESHOLDS:")
    print(f"  Q1 (25th percentile): {completeness_q1:.4f} ({completeness_q1*100:.1f}%)")
    print(f"  Median: {completeness_median:.4f} ({completeness_median*100:.1f}%)")
    print(f"  Mean: {completeness_mean:.4f} ({completeness_mean*100:.1f}%)")
    print(f"  Q3 (75th percentile): {completeness_q3:.4f} ({completeness_q3*100:.1f}%)")
    
    # Test each option
    print(f"\n" + "=" * 60)
    print(f"OPTION 1: MEAN THRESHOLDS")
    print(f"Coverage threshold: {coverage_mean:.4f}, Completeness threshold: {completeness_mean:.4f}")
    
    comprehensive_1 = results_df[
        (results_df['coverage_ratio'] >= coverage_mean) &
        (results_df['completeness_ratio'] >= completeness_mean)
    ]
    broad_1 = results_df[
        (results_df['coverage_ratio'] >= coverage_mean) &
        (results_df['completeness_ratio'] < completeness_mean)
    ]
    focused_1 = results_df[
        (results_df['coverage_ratio'] < coverage_mean) &
        (results_df['completeness_ratio'] >= completeness_mean)
    ]
    limited_1 = results_df[
        (results_df['coverage_ratio'] < coverage_mean) &
        (results_df['completeness_ratio'] < completeness_mean)
    ]
    
    print(f"  Comprehensive: {len(comprehensive_1)} collections")
    print(f"  Broad Surveyors: {len(broad_1)} collections")
    print(f"  Focused Specialists: {len(focused_1)} collections")
    print(f"  Limited Scope: {len(limited_1)} collections")
    
    print(f"\nOPTION 2: Q3 THRESHOLDS (75th percentile)")
    print(f"Coverage threshold: {coverage_q3:.4f}, Completeness threshold: {completeness_q3:.4f}")
    
    comprehensive_2 = results_df[
        (results_df['coverage_ratio'] >= coverage_q3) &
        (results_df['completeness_ratio'] >= completeness_q3)
    ]
    broad_2 = results_df[
        (results_df['coverage_ratio'] >= coverage_q3) &
        (results_df['completeness_ratio'] < completeness_q3)
    ]
    focused_2 = results_df[
        (results_df['coverage_ratio'] < coverage_q3) &
        (results_df['completeness_ratio'] >= completeness_q3)
    ]
    limited_2 = results_df[
        (results_df['coverage_ratio'] < coverage_q3) &
        (results_df['completeness_ratio'] < completeness_q3)
    ]
    
    print(f"  Comprehensive: {len(comprehensive_2)} collections")
    print(f"  Broad Surveyors: {len(broad_2)} collections")
    print(f"  Focused Specialists: {len(focused_2)} collections")
    print(f"  Limited Scope: {len(limited_2)} collections")
    
    print(f"\nOPTION 3: Q1 THRESHOLDS (25th percentile)")
    print(f"Coverage threshold: {coverage_q1:.4f}, Completeness threshold: {completeness_q1:.4f}")
    
    comprehensive_3 = results_df[
        (results_df['coverage_ratio'] >= coverage_q1) &
        (results_df['completeness_ratio'] >= completeness_q1)
    ]
    broad_3 = results_df[
        (results_df['coverage_ratio'] >= coverage_q1) &
        (results_df['completeness_ratio'] < completeness_q1)
    ]
    focused_3 = results_df[
        (results_df['coverage_ratio'] < coverage_q1) &
        (results_df['completeness_ratio'] >= completeness_q1)
    ]
    limited_3 = results_df[
        (results_df['coverage_ratio'] < coverage_q1) &
        (results_df['completeness_ratio'] < completeness_q1)
    ]
    
    print(f"  Comprehensive: {len(comprehensive_3)} collections")
    print(f"  Broad Surveyors: {len(broad_3)} collections")
    print(f"  Focused Specialists: {len(focused_3)} collections")
    print(f"  Limited Scope: {len(limited_3)} collections")
    
    # Show examples for each option
    print(f"\n" + "=" * 60)
    print(f"DETAILED BREAKDOWN BY OPTION:")
    
    for option, (comp, broad, focused, limited) in enumerate([
        (comprehensive_1, broad_1, focused_1, limited_1),
        (comprehensive_2, broad_2, focused_2, limited_2),
        (comprehensive_3, broad_3, focused_3, limited_3)
    ], 1):
        print(f"\nOPTION {option}:")
        print(f"  Comprehensive Catalogers ({len(comp)}):")
        for _, row in comp.iterrows():
            print(f"    • {row['subcollection']} (cov: {row['coverage_ratio']:.3f}, comp: {row['completeness_ratio']:.3f})")
        
        print(f"  Broad Surveyors ({len(broad)}):")
        for _, row in broad.iterrows():
            print(f"    • {row['subcollection']} (cov: {row['coverage_ratio']:.3f}, comp: {row['completeness_ratio']:.3f})")
        
        print(f"  Focused Specialists ({len(focused)}):")
        for _, row in focused.iterrows():
            print(f"    • {row['subcollection']} (cov: {row['coverage_ratio']:.3f}, comp: {row['completeness_ratio']:.3f})")
        
        print(f"  Limited Scope ({len(limited)}):")
        for _, row in limited.iterrows():
            print(f"    • {row['subcollection']} (cov: {row['coverage_ratio']:.3f}, comp: {row['completeness_ratio']:.3f})")

if __name__ == "__main__":
    analyze_threshold_options() 