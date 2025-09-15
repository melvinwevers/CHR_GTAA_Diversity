#!/usr/bin/env python3
"""
Generate comprehensive statistics for the paper.
"""

import pandas as pd
import numpy as np
from scipy import stats

def generate_comprehensive_statistics():
    """
    Generate all statistics needed for the paper.
    """
    
    try:
        # Load the basic results CSV
        results_df = pd.read_csv('results/archival_bias_detailed.csv')
        
        print("COMPREHENSIVE PAPER STATISTICS REPORT")
        print("=" * 70)
        
        # Basic counts
        n_collections = len(results_df)
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   • Collections analyzed: {n_collections}")
        print(f"   • Data source: GTAA archival collections")
        
        # Coverage Ratio Statistics
        coverage_stats = {
            'min': results_df['coverage_ratio'].min(),
            'max': results_df['coverage_ratio'].max(),
            'mean': results_df['coverage_ratio'].mean(),
            'median': results_df['coverage_ratio'].median(),
            'std': results_df['coverage_ratio'].std(),
            'q1': results_df['coverage_ratio'].quantile(0.25),
            'q3': results_df['coverage_ratio'].quantile(0.75),
            'skewness': stats.skew(results_df['coverage_ratio'])
        }
        
        print(f"\n2. COVERAGE RATIO STATISTICS:")
        print(f"   • Range: {coverage_stats['min']:.4f} to {coverage_stats['max']:.4f} ({coverage_stats['min']*100:.2f}% to {coverage_stats['max']*100:.2f}%)")
        print(f"   • Mean: {coverage_stats['mean']:.4f} ({coverage_stats['mean']*100:.2f}%)")
        print(f"   • Median: {coverage_stats['median']:.4f} ({coverage_stats['median']*100:.2f}%)")
        print(f"   • Standard Deviation: {coverage_stats['std']:.4f}")
        print(f"   • Q1 (25th percentile): {coverage_stats['q1']:.4f} ({coverage_stats['q1']*100:.2f}%)")
        print(f"   • Q3 (75th percentile): {coverage_stats['q3']:.4f} ({coverage_stats['q3']*100:.2f}%)")
        print(f"   • Skewness: {coverage_stats['skewness']:.2f} ({'Right-skewed' if coverage_stats['skewness'] > 0 else 'Left-skewed' if coverage_stats['skewness'] < 0 else 'Symmetric'})")

        # Completeness Ratio Statistics
        completeness_stats = {
            'min': results_df['completeness_ratio'].min(),
            'max': results_df['completeness_ratio'].max(),
            'mean': results_df['completeness_ratio'].mean(),
            'median': results_df['completeness_ratio'].median(),
            'std': results_df['completeness_ratio'].std(),
            'q1': results_df['completeness_ratio'].quantile(0.25),
            'q3': results_df['completeness_ratio'].quantile(0.75),
            'skewness': stats.skew(results_df['completeness_ratio'])
        }
        
        print(f"\n3. COMPLETENESS RATIO STATISTICS:")
        print(f"   • Range: {completeness_stats['min']:.4f} to {completeness_stats['max']:.4f} ({completeness_stats['min']*100:.2f}% to {completeness_stats['max']*100:.2f}%)")
        print(f"   • Mean: {completeness_stats['mean']:.4f} ({completeness_stats['mean']*100:.2f}%)")
        print(f"   • Median: {completeness_stats['median']:.4f} ({completeness_stats['median']*100:.2f}%)")
        print(f"   • Standard Deviation: {completeness_stats['std']:.4f}")
        print(f"   • Q1 (25th percentile): {completeness_stats['q1']:.4f} ({completeness_stats['q1']*100:.2f}%)")
        print(f"   • Q3 (75th percentile): {completeness_stats['q3']:.4f} ({completeness_stats['q3']*100:.2f}%)")
        print(f"   • Skewness: {completeness_stats['skewness']:.2f} ({'Right-skewed' if completeness_stats['skewness'] > 0 else 'Left-skewed' if completeness_stats['skewness'] < 0 else 'Symmetric'})")

        # Efficiency Ratio Statistics
        efficiency_stats = {
            'min': results_df['efficiency_ratio'].min(),
            'max': results_df['efficiency_ratio'].max(),
            'mean': results_df['efficiency_ratio'].mean(),
            'median': results_df['efficiency_ratio'].median(),
            'std': results_df['efficiency_ratio'].std(),
            'q1': results_df['efficiency_ratio'].quantile(0.25),
            'q3': results_df['efficiency_ratio'].quantile(0.75),
            'skewness': stats.skew(results_df['efficiency_ratio'])
        }
        
        print(f"\n4. EFFICIENCY RATIO STATISTICS:")
        print(f"   • Range: {efficiency_stats['min']:.4f} to {efficiency_stats['max']:.4f}")
        print(f"   • Mean: {efficiency_stats['mean']:.4f}")
        print(f"   • Median: {efficiency_stats['median']:.4f}")
        print(f"   • Standard Deviation: {efficiency_stats['std']:.4f}")
        print(f"   • Q1 (25th percentile): {efficiency_stats['q1']:.4f}")
        print(f"   • Q3 (75th percentile): {efficiency_stats['q3']:.4f}")
        print(f"   • Skewness: {efficiency_stats['skewness']:.2f} ({'Right-skewed' if efficiency_stats['skewness'] > 0 else 'Left-skewed' if efficiency_stats['skewness'] < 0 else 'Symmetric'})")

        # Check if CI data is available
        if 'ci_lower' in results_df.columns and 'ci_upper' in results_df.columns:
            # Remove rows where CI data is missing or invalid
            valid_ci_df = results_df.dropna(subset=['ci_lower', 'ci_upper'])
            
            if len(valid_ci_df) > 0:
                print(f"\n5. UNSEEN PHYLOGENETIC DIVERSITY WITH CONFIDENCE INTERVALS:")
                print(f"   • Collections with valid CI data: {len(valid_ci_df)} of {n_collections}")
                
                # Calculate CI width
                valid_ci_df = valid_ci_df.copy()
                valid_ci_df['ci_width'] = valid_ci_df['ci_upper'] - valid_ci_df['ci_lower']
                
                # Unseen PD statistics
                unseen_pd_stats = {
                    'min': valid_ci_df['unseen_pd'].min(),
                    'max': valid_ci_df['unseen_pd'].max(),
                    'mean': valid_ci_df['unseen_pd'].mean(),
                    'median': valid_ci_df['unseen_pd'].median(),
                    'std': valid_ci_df['unseen_pd'].std(),
                    'q1': valid_ci_df['unseen_pd'].quantile(0.25),
                    'q3': valid_ci_df['unseen_pd'].quantile(0.75)
                }
                
                print(f"   • Unseen PD Range: {unseen_pd_stats['min']:.1f} to {unseen_pd_stats['max']:.1f}")
                print(f"   • Unseen PD Mean: {unseen_pd_stats['mean']:.1f} ± {unseen_pd_stats['std']:.1f}")
                print(f"   • Unseen PD Median: {unseen_pd_stats['median']:.1f}")
                
                # CI width statistics
                ci_width_stats = {
                    'min': valid_ci_df['ci_width'].min(),
                    'max': valid_ci_df['ci_width'].max(),
                    'mean': valid_ci_df['ci_width'].mean(),
                    'median': valid_ci_df['ci_width'].median(),
                    'std': valid_ci_df['ci_width'].std()
                }
                
                print(f"   • CI Width Range: {ci_width_stats['min']:.1f} to {ci_width_stats['max']:.1f}")
                print(f"   • CI Width Mean: {ci_width_stats['mean']:.1f} ± {ci_width_stats['std']:.1f}")
                print(f"   • CI Width Median: {ci_width_stats['median']:.1f}")
                
                # Show collections with tightest and widest CIs
                valid_ci_df_sorted = valid_ci_df.sort_values('ci_width')
                
                print(f"\n   COLLECTIONS WITH TIGHTEST CONFIDENCE INTERVALS (most reliable estimates):")
                for i, (_, row) in enumerate(valid_ci_df_sorted.head(3).iterrows()):
                    print(f"   {i+1}. {row['subcollection']}")
                    print(f"      Unseen PD: {row['unseen_pd']:.1f} [95% CI: {row['ci_lower']:.1f}, {row['ci_upper']:.1f}] (width: {row['ci_width']:.1f})")
                
                print(f"\n   COLLECTIONS WITH WIDEST CONFIDENCE INTERVALS (least reliable estimates):")
                for i, (_, row) in enumerate(valid_ci_df_sorted.tail(3).iterrows()):
                    print(f"   {i+1}. {row['subcollection']}")
                    print(f"      Unseen PD: {row['unseen_pd']:.1f} [95% CI: {row['ci_lower']:.1f}, {row['ci_upper']:.1f}] (width: {row['ci_width']:.1f})")
                
                    
            else:
                print(f"\n5. UNSEEN PHYLOGENETIC DIVERSITY:")
                print(f"   ⚠️  No valid confidence interval data found in current CSV")
        else:
            print(f"\n5. UNSEEN PHYLOGENETIC DIVERSITY:")
            print(f"   ⚠️  No confidence interval columns found in CSV")

        # Collection PD Statistics
        pd_stats = {
            'min': results_df['collection_pd'].min(),
            'max': results_df['collection_pd'].max(),
            'mean': results_df['collection_pd'].mean(),
            'median': results_df['collection_pd'].median(),
            'std': results_df['collection_pd'].std(),
            'q1': results_df['collection_pd'].quantile(0.25),
            'q3': results_df['collection_pd'].quantile(0.75),
            'skewness': stats.skew(results_df['collection_pd'])
        }
        
        print(f"\n6. COLLECTION PHYLOGENETIC DIVERSITY STATISTICS:")
        print(f"   • Range: {pd_stats['min']:.1f} to {pd_stats['max']:.1f}")
        print(f"   • Mean: {pd_stats['mean']:.1f}")
        print(f"   • Median: {pd_stats['median']:.1f}")
        print(f"   • Standard Deviation: {pd_stats['std']:.1f}")
        print(f"   • Q1 (25th percentile): {pd_stats['q1']:.1f}")
        print(f"   • Q3 (75th percentile): {pd_stats['q3']:.1f}")
        print(f"   • Skewness: {pd_stats['skewness']:.2f}")
        
        # Outlier analysis - excluding "Fotocollectie Anefo"
        outlier_collection = "Fotocollectie Anefo"
        if outlier_collection in results_df['subcollection'].values:
            outlier_pd = results_df[results_df['subcollection'] == outlier_collection]['collection_pd'].iloc[0]
            print(f"   • Outlier identified: {outlier_collection} (PD = {outlier_pd:.1f})")
            
            # Calculate statistics excluding the outlier
            filtered_df = results_df[results_df['subcollection'] != outlier_collection]
            filtered_pd_stats = {
                'min': filtered_df['collection_pd'].min(),
                'max': filtered_df['collection_pd'].max(),
                'mean': filtered_df['collection_pd'].mean(),
                'median': filtered_df['collection_pd'].median(),
                'std': filtered_df['collection_pd'].std(),
                'q1': filtered_df['collection_pd'].quantile(0.25),
                'q3': filtered_df['collection_pd'].quantile(0.75),
                'skewness': stats.skew(filtered_df['collection_pd'])
            }
            
            print(f"\n   NORMALIZED DISTRIBUTION (excluding outlier):")
            print(f"   • Range: {filtered_pd_stats['min']:.1f} to {filtered_pd_stats['max']:.1f}")
            print(f"   • Mean: {filtered_pd_stats['mean']:.1f}")
            print(f"   • Median: {filtered_pd_stats['median']:.1f}")
            print(f"   • Standard Deviation (σ): {filtered_pd_stats['std']:.1f}")
            print(f"   • Q1 (25th percentile): {filtered_pd_stats['q1']:.1f}")
            print(f"   • Q3 (75th percentile): {filtered_pd_stats['q3']:.1f}")
            print(f"   • Skewness: {filtered_pd_stats['skewness']:.2f}")
        else:
            print(f"   • Note: {outlier_collection} not found in dataset")
        
        # Collection archetype analysis
        print(f"\n7. COLLECTION ARCHETYPE ANALYSIS:")
        
        # Calculate medians for quadrant analysis
        coverage_median = results_df['coverage_ratio'].median()
        completeness_median = results_df['completeness_ratio'].median()
        
        print(f"   • Median thresholds: Coverage = {coverage_median:.3f}, Completeness = {completeness_median:.3f}")
        
        # Count collections in each quadrant
        comprehensive = results_df[
            (results_df['coverage_ratio'] >= coverage_median) &
            (results_df['completeness_ratio'] >= completeness_median)
        ]
        
        broad_undersampled = results_df[
            (results_df['coverage_ratio'] >= coverage_median) &
            (results_df['completeness_ratio'] < completeness_median)
        ]
        
        narrow_thorough = results_df[
            (results_df['coverage_ratio'] < coverage_median) &
            (results_df['completeness_ratio'] >= completeness_median)
        ]
        
        institutional_blind = results_df[
            (results_df['coverage_ratio'] < coverage_median) &
            (results_df['completeness_ratio'] < completeness_median)
        ]
        
        print(f"   • Comprehensive Collections (high coverage, high completeness): {len(comprehensive)}")
        print(f"   • Broad but Under-sampled (high coverage, low completeness): {len(broad_undersampled)}")
        print(f"   • Narrow but Thorough (low coverage, high completeness): {len(narrow_thorough)}")
        print(f"   • Institutional Blind Spots (low coverage, low completeness): {len(institutional_blind)}")
        
        # Collection size statistics
        size_stats = {
            'min': results_df['collection_size'].min(),
            'max': results_df['collection_size'].max(),
            'mean': results_df['collection_size'].mean(),
            'median': results_df['collection_size'].median(),
            'std': results_df['collection_size'].std(),
            'q1': results_df['collection_size'].quantile(0.25),
            'q3': results_df['collection_size'].quantile(0.75)
        }
        
        print(f"\n8. COLLECTION SIZE STATISTICS:")
        print(f"   • Range: {size_stats['min']:,} to {size_stats['max']:,} images")
        print(f"   • Mean: {size_stats['mean']:,.0f} images")
        print(f"   • Median: {size_stats['median']:,.0f} images")
        print(f"   • Standard Deviation: {size_stats['std']:,.0f}")
        print(f"   • Q1 (25th percentile): {size_stats['q1']:,.0f}")
        print(f"   • Q3 (75th percentile): {size_stats['q3']:,.0f}")
        
        # Correlation Analysis
        print(f"\n9. CORRELATION ANALYSIS:")
        
        # Pearson correlation between collection size and coverage ratio
        corr_coverage, pval_coverage = stats.pearsonr(results_df['collection_size'], results_df['coverage_ratio'])
        print(f"   • Collection Size vs Coverage Ratio:")
        print(f"     Pearson r = {corr_coverage:.3f}, p = {pval_coverage:.3f}")
        print(f"     {'Statistically significant' if pval_coverage < 0.05 else 'Not statistically significant'} (α = 0.05)")
        
        # Pearson correlation between collection size and completeness ratio
        corr_completeness, pval_completeness = stats.pearsonr(results_df['collection_size'], results_df['completeness_ratio'])
        print(f"   • Collection Size vs Completeness Ratio:")
        print(f"     Pearson r = {corr_completeness:.3f}, p = {pval_completeness:.3f}")
        print(f"     {'Statistically significant' if pval_completeness < 0.05 else 'Not statistically significant'} (α = 0.05)")
        
        # Pearson correlation between collection size and efficiency ratio
        corr_efficiency, pval_efficiency = stats.pearsonr(results_df['collection_size'], results_df['efficiency_ratio'])
        print(f"   • Collection Size vs Efficiency Ratio:")
        print(f"     Pearson r = {corr_efficiency:.3f}, p = {pval_efficiency:.3f}")
        print(f"     {'Statistically significant' if pval_efficiency < 0.05 else 'Not statistically significant'} (α = 0.05)")
        
        # Spearman correlation (rank-based) for robustness
        spearman_coverage, spearman_p_coverage = stats.spearmanr(results_df['collection_size'], results_df['coverage_ratio'])
        print(f"   • Collection Size vs Coverage Ratio (Spearman):")
        print(f"     Spearman ρ = {spearman_coverage:.3f}, p = {spearman_p_coverage:.3f}")
        
        # Collection Archetype LaTeX Table
        print(f"\n10. COLLECTION ARCHETYPE LATEX TABLE:")
        print(f"\\begin{{table}}[h]")
        print(f"  \\centering")
        print(f"  \\caption{{Cataloging Practice Archetypes}}")
        print(f"  \\begin{{tabular}}{{lcccp{{4cm}}}}")
        print(f"    \\toprule")
        print(f"    \\textbf{{Archetype}} & \\textbf{{Coverage}} & \\textbf{{Completeness}} & \\textbf{{Collections}} & \\textbf{{Characteristics}} \\\\")
        print(f"    \\midrule")
        
        # Comprehensive Collections
        comprehensive_examples = []
        for _, row in comprehensive.head(3).iterrows():
            name = str(row['subcollection']).replace('Fotocollectie ', '')
            comprehensive_examples.append(f"{name} ({row['coverage_ratio']:.3f}, {row['completeness_ratio']:.3f})")
        comprehensive_str = ", ".join(comprehensive_examples)
        
        print(f"    Comprehensive Catalogers & High & High & {len(comprehensive)} & {comprehensive_str} \\\\")
        
        # Broad but Under-sampled
        broad_examples = []
        for _, row in broad_undersampled.head(2).iterrows():
            name = str(row['subcollection']).replace('Fotocollectie ', '')
            broad_examples.append(f"{name} ({row['coverage_ratio']:.3f}, {row['completeness_ratio']:.3f})")
        broad_str = ", ".join(broad_examples)
        
        print(f"    Broad Surveyors & High & Low & {len(broad_undersampled)} & {broad_str} \\\\")
        
        # Narrow but Thorough
        narrow_examples = []
        for _, row in narrow_thorough.head(2).iterrows():
            name = str(row['subcollection']).replace('Fotocollectie ', '')
            narrow_examples.append(f"{name} ({row['coverage_ratio']:.3f}, {row['completeness_ratio']:.3f})")
        narrow_str = ", ".join(narrow_examples)
        
        print(f"    Focused Specialists & Low & High & {len(narrow_thorough)} & {narrow_str} \\\\")
        
        # Institutional Blind Spots
        blind_examples = []
        for _, row in institutional_blind.head(1).iterrows():
            name = str(row['subcollection']).replace('Fotocollectie ', '')
            blind_examples.append(f"{name} ({row['coverage_ratio']:.3f}, {row['completeness_ratio']:.3f})")
        blind_str = ", ".join(blind_examples)
        if len(institutional_blind) > 1:
            blind_str += f", plus {len(institutional_blind) - 1} others"
        
        print(f"    Limited Scope & Low & Low & {len(institutional_blind)} & {blind_str} \\\\")
        
        print(f"    \\bottomrule")
        print(f"  \\end{{tabular}}")
        print(f"  \\label{{tab:cataloging_archetypes}}")
        print(f"\\end{{table}}")

        # Summary statistics table for LaTeX
        print(f"\n11. SUMMARY STATISTICS LATEX TABLE:")
        print(f"   \\begin{{tabular}}{{lcccc}}")
        print(f"   \\toprule")
        print(f"   Statistic & Coverage & Completeness & Efficiency & Collection PD \\\\")
        print(f"   \\midrule")
        print(f"   Min & {coverage_stats['min']:.4f} & {completeness_stats['min']:.4f} & {efficiency_stats['min']:.4f} & {pd_stats['min']:.1f} \\\\")
        print(f"   Max & {coverage_stats['max']:.4f} & {completeness_stats['max']:.4f} & {efficiency_stats['max']:.4f} & {pd_stats['max']:.1f} \\\\")
        print(f"   Mean & {coverage_stats['mean']:.4f} & {completeness_stats['mean']:.4f} & {efficiency_stats['mean']:.4f} & {pd_stats['mean']:.1f} \\\\")
        print(f"   Median & {coverage_stats['median']:.4f} & {completeness_stats['median']:.4f} & {efficiency_stats['median']:.4f} & {pd_stats['median']:.1f} \\\\")
        print(f"   Std Dev & {coverage_stats['std']:.4f} & {completeness_stats['std']:.4f} & {efficiency_stats['std']:.4f} & {pd_stats['std']:.1f} \\\\")
        print(f"   \\bottomrule")
        print(f"   \\end{{tabular}}")
    
        
    except FileNotFoundError:
        print("❌ Error: Could not find results/archival_bias_detailed.csv")
        print("Please make sure the notebook has been run and the results CSV has been saved.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_comprehensive_statistics() 