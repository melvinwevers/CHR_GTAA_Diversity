#!/usr/bin/env python3
"""
Run Hierarchy Bias Analysis

This script demonstrates the three key areas of bias analysis:
1. Analyze which GTAA hierarchy branches are most/least represented across collections
2. Calculate coverage statistics for different conceptual domains  
3. Identify systematic gaps in the collective coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append('src')

from src.archival_bias_detection import ArchivalBiasDetector
from src.hierarchy_bias_analysis import HierarchyBiasAnalyzer

def main():
    """Run comprehensive hierarchy bias analysis."""
    
    # Configuration
    gtaa_csv_path = Path("data/external/gtaa_ontology.csv")
    parquet_path = Path("data/processed/photos_archive.parquet")
    output_dir = Path("results/hierarchy_bias_analysis")
    
    print("=" * 80)
    print("GTAA HIERARCHY BIAS ANALYSIS")
    print("=" * 80)
    
    # Check if files exist
    if not parquet_path.exists():
        print(f"Error: Processed data file not found at {parquet_path}")
        return
    if not gtaa_csv_path.exists():
        print(f"Error: GTAA vocabulary file not found at {gtaa_csv_path}")
        return
    
    print("✓ All required files found")
    
    # Initialize the archival bias detector to get the graph
    print("\n1. Building GTAA vocabulary graph...")
    detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=1000)
    detector.build_graph(apply_deduplication=False)
    
    # Load and filter data
    print("2. Loading and filtering data...")
    df = detector.load_and_filter_data(parquet_path)
    
    # Initialize hierarchy bias analyzer
    print("3. Initializing hierarchy bias analyzer...")
    analyzer = HierarchyBiasAnalyzer(detector.graph, gtaa_csv_path)
    
    print(f"   ✓ GTAA hierarchy has {len(analyzer.root_branches)} root branches")
    print(f"   ✓ Maximum depth: {max(analyzer.node_depths.values())}")
    print(f"   ✓ Total terms: {len(analyzer.node_depths)}")
    
    # Run the three key analyses
    print("\n" + "=" * 80)
    print("ANALYSIS 1: BRANCH REPRESENTATION ACROSS COLLECTIONS")
    print("=" * 80)
    
    branch_analysis = analyzer.analyze_branch_representation(df, 'fotocollectie')
    
    print(f"\nBranch Representation Summary:")
    print(f"  Total collections analyzed: {branch_analysis['total_collections']}")
    print(f"  Total GTAA branches: {len(analyzer.root_branches)}")
    
    # Show most represented branches
    print(f"\nMost represented branches (highest collection coverage):")
    for _, row in branch_analysis['most_represented_branches'].head(5).iterrows():
        coverage_pct = row['collection_coverage_ratio'] * 100
        print(f"  • {row['branch']}: {coverage_pct:.1f}% of collections ({row['n_collections']}/{branch_analysis['total_collections']})")
    
    # Show least represented branches
    print(f"\nLeast represented branches (lowest collection coverage):")
    for _, row in branch_analysis['least_represented_branches'].head(5).iterrows():
        coverage_pct = row['collection_coverage_ratio'] * 100
        print(f"  • {row['branch']}: {coverage_pct:.1f}% of collections ({row['n_collections']}/{branch_analysis['total_collections']})")
    
    print("\n" + "=" * 80)
    print("ANALYSIS 2: CONCEPTUAL DOMAIN COVERAGE STATISTICS")
    print("=" * 80)
    
    domain_analysis = analyzer.calculate_conceptual_domain_coverage(df, 'fotocollectie')
    
    print(f"\nConceptual Domain Coverage Summary:")
    domain_stats = domain_analysis['domain_stats']
    
    for _, row in domain_stats.iterrows():
        coverage_pct = row['avg_collection_coverage'] * 100
        print(f"  • {row['domain']}: {coverage_pct:.1f}% avg collection coverage")
        print(f"    - {row['n_branches']} branches, {row['total_terms']} total terms")
        print(f"    - {row['n_collections']} collections cover this domain")
    
    print("\n" + "=" * 80)
    print("ANALYSIS 3: SYSTEMATIC GAPS IDENTIFICATION")
    print("=" * 80)
    
    gap_analysis = analyzer.identify_systematic_gaps(df, 'fotocollectie')
    
    print(f"\nSystematic Gaps Summary:")
    print(f"  Completely missing terms: {len(gap_analysis['missing_terms']):,}")
    print(f"  Rare terms (<10% collections): {len(gap_analysis['rare_terms']):,}")
    
    # Show coverage by hierarchy depth
    print(f"\nCoverage by hierarchy depth:")
    for depth in sorted(gap_analysis['coverage_by_depth'].keys()):
        coverage = gap_analysis['coverage_by_depth'][depth]
        coverage_pct = coverage['coverage_ratio'] * 100
        print(f"  • Depth {depth}: {coverage_pct:.1f}% ({coverage['covered_terms']:,}/{coverage['total_terms']:,})")
    
    # Show branches with lowest coverage
    branch_coverage = gap_analysis['coverage_by_branch']
    worst_branches = sorted(branch_coverage.items(), key=lambda x: x[1]['coverage_ratio'])[:5]
    
    print(f"\nBranches with lowest coverage (systematic gaps):")
    for branch, coverage in worst_branches:
        coverage_pct = coverage['coverage_ratio'] * 100
        print(f"  • {branch}: {coverage_pct:.1f}% ({coverage['covered_terms']:,}/{coverage['total_terms']:,})")
    
    # Generate and display comprehensive report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE BIAS ANALYSIS REPORT")
    print("=" * 80)
    
    report = analyzer.generate_bias_report(df, 'fotocollectie')
    print(report)
    
    # Save all results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    analyzer.save_analysis_results(df, output_dir, 'fotocollectie')
    
    print(f"\n✓ All results saved to {output_dir}")
    print(f"✓ Files created:")
    print(f"  • branch_representation.csv - Detailed branch coverage statistics")
    print(f"  • most_represented_branches.csv - Top branches by collection coverage")
    print(f"  • least_represented_branches.csv - Bottom branches by collection coverage")
    print(f"  • conceptual_domain_coverage.csv - Coverage by conceptual domains")
    print(f"  • systematic_gaps.json - Detailed gap analysis")
    print(f"  • bias_analysis_report.txt - Comprehensive text report")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Extract key insights
    most_rep = branch_analysis['most_represented_branches'].iloc[0]
    least_rep = branch_analysis['least_represented_branches'].iloc[0]
    
    print(f"1. Most represented branch: '{most_rep['branch']}' appears in {most_rep['collection_coverage_ratio']*100:.1f}% of collections")
    print(f"2. Least represented branch: '{least_rep['branch']}' appears in {least_rep['collection_coverage_ratio']*100:.1f}% of collections")
    print(f"3. Coverage disparity: {most_rep['collection_coverage_ratio']/least_rep['collection_coverage_ratio']:.1f}x difference")
    
    # Find domain with lowest coverage
    worst_domain = domain_stats.loc[domain_stats['avg_collection_coverage'].idxmin()]
    print(f"4. Domain with lowest coverage: '{worst_domain['domain']}' with {worst_domain['avg_collection_coverage']*100:.1f}% avg coverage")
    
    # Find depth with lowest coverage
    worst_depth = min(gap_analysis['coverage_by_depth'].items(), key=lambda x: x[1]['coverage_ratio'])
    print(f"5. Hierarchy depth with lowest coverage: Depth {worst_depth[0]} with {worst_depth[1]['coverage_ratio']*100:.1f}% coverage")
    
    print(f"\nThis analysis reveals systematic biases in how different conceptual domains")
    print(f"and hierarchy levels are represented across archival collections, providing")
    print(f"actionable insights for addressing archival gaps and improving coverage.")

if __name__ == "__main__":
    main() 