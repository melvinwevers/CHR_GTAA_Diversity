"""
Hierarchy Bias Analysis Module

This module provides comprehensive analysis of GTAA hierarchy branch representation,
coverage statistics for different conceptual domains, and identification of systematic
gaps in collective coverage across archival collections.
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

class HierarchyBiasAnalyzer:
    """
    Analyzes bias patterns in GTAA hierarchy branch representation across collections.
    
    This class provides methods to:
    1. Analyze which GTAA hierarchy branches are most/least represented across collections
    2. Calculate coverage statistics for different conceptual domains
    3. Identify systematic gaps in the collective coverage
    """
    
    def __init__(self, graph: nx.DiGraph, gtaa_csv_path: Path):
        """
        Initialize the HierarchyBiasAnalyzer.
        
        Args:
            graph: The GTAA vocabulary graph
            gtaa_csv_path: Path to the GTAA vocabulary CSV file
        """
        self.graph = graph
        self.gtaa_csv_path = Path(gtaa_csv_path)
        self.vocab_df = pd.read_csv(gtaa_csv_path)
        self.root = "_DUMMY_ROOT_"
        
        # Calculate hierarchy structure
        self._analyze_hierarchy_structure()
        
    def _analyze_hierarchy_structure(self):
        """Analyze the basic structure of the GTAA hierarchy."""
        # Get root-level branches (direct children of dummy root)
        self.root_branches = list(self.graph.successors(self.root))
        
        # Calculate depth for each node
        self.node_depths = {}
        for node in self.graph.nodes():
            if node != self.root:
                try:
                    depth = nx.shortest_path_length(self.graph, self.root, node)
                    self.node_depths[node] = depth
                except nx.NetworkXNoPath:
                    self.node_depths[node] = 0
        
        # Group nodes by depth
        self.depth_groups = defaultdict(list)
        for node, depth in self.node_depths.items():
            self.depth_groups[depth].append(node)
            
        # Calculate branch sizes (number of descendants)
        self.branch_sizes = {}
        for branch in self.root_branches:
            descendants = nx.descendants(self.graph, branch)
            self.branch_sizes[branch] = len(descendants) + 1  # +1 for the branch itself
    
    def analyze_branch_representation(self, df: pd.DataFrame, collection_column: str = 'fotocollectie') -> Dict:
        """
        Analyze which GTAA hierarchy branches are most/least represented across collections.
        
        Args:
            df: DataFrame containing collection data with subjects_list column
            collection_column: Name of the column containing collection identifiers
            
        Returns:
            Dictionary with branch representation analysis
        """
        print("Analyzing GTAA hierarchy branch representation...")
        
        # Get collections
        collections = df[collection_column].value_counts()
        large_collections = collections[collections >= 1000].index.tolist()
        
        # Initialize branch coverage tracking
        branch_coverage = {branch: {'collections': set(), 'total_occurrences': 0, 'unique_terms': set()} 
                          for branch in self.root_branches}
        
        # Track term occurrences per collection
        collection_terms = {}
        
        # Analyze each collection
        for collection in tqdm(large_collections, desc="Analyzing collections"):
            collection_df = df[df[collection_column] == collection]
            collection_terms[collection] = set()
            
            # Extract all terms from this collection
            for subjects_list in collection_df['subjects_list'].dropna():
                if isinstance(subjects_list, list):
                    collection_terms[collection].update(subjects_list)
            
            # Map terms to their root branches
            for term in collection_terms[collection]:
                if term in self.graph.nodes():
                    # Find the root branch this term belongs to
                    root_branch = self._find_root_branch(term)
                    if root_branch:
                        branch_coverage[root_branch]['collections'].add(collection)
                        branch_coverage[root_branch]['total_occurrences'] += 1
                        branch_coverage[root_branch]['unique_terms'].add(term)
        
        # Calculate branch representation statistics
        branch_stats = []
        for branch, coverage in branch_coverage.items():
            n_collections = len(coverage['collections'])
            n_terms = len(coverage['unique_terms'])
            branch_size = self.branch_sizes[branch]
            
            # Calculate representation ratios
            collection_coverage_ratio = n_collections / len(large_collections)
            term_coverage_ratio = n_terms / branch_size if branch_size > 0 else 0
            
            branch_stats.append({
                'branch': branch,
                'branch_size': branch_size,
                'n_collections': n_collections,
                'n_unique_terms': n_terms,
                'collection_coverage_ratio': collection_coverage_ratio,
                'term_coverage_ratio': term_coverage_ratio,
                'total_occurrences': coverage['total_occurrences'],
                'collections': list(coverage['collections'])
            })
        
        # Sort by different metrics
        branch_stats_df = pd.DataFrame(branch_stats)
        
        results = {
            'branch_stats': branch_stats_df,
            'most_represented_branches': branch_stats_df.nlargest(10, 'collection_coverage_ratio'),
            'least_represented_branches': branch_stats_df.nsmallest(10, 'collection_coverage_ratio'),
            'largest_branches': branch_stats_df.nlargest(10, 'branch_size'),
            'smallest_branches': branch_stats_df.nsmallest(10, 'branch_size'),
            'collection_terms': collection_terms,
            'total_collections': len(large_collections)
        }
        
        return results
    
    def _find_root_branch(self, term: str) -> Optional[str]:
        """Find the root branch that a term belongs to."""
        if term not in self.graph.nodes():
            return None
            
        # Find the path from root to this term
        try:
            path = nx.shortest_path(self.graph, self.root, term)
            if len(path) >= 2:  # root -> branch -> ... -> term
                return path[1]  # First child of root
        except nx.NetworkXNoPath:
            pass
        return None
    
    def calculate_conceptual_domain_coverage(self, df: pd.DataFrame, collection_column: str = 'fotocollectie') -> Dict:
        """
        Calculate coverage statistics for different conceptual domains.
        
        Args:
            df: DataFrame containing collection data
            collection_column: Name of the column containing collection identifiers
            
        Returns:
            Dictionary with conceptual domain coverage analysis
        """
        print("Calculating conceptual domain coverage statistics...")
        
        # Get branch representation first
        branch_analysis = self.analyze_branch_representation(df, collection_column)
        branch_stats = branch_analysis['branch_stats']
        
        # Define conceptual domains (you can customize this mapping)
        conceptual_domains = self._define_conceptual_domains()
        
        # Map branches to conceptual domains
        domain_coverage = defaultdict(lambda: {
            'branches': [],
            'total_terms': 0,
            'collections': set(),
            'coverage_metrics': {}
        })
        
        for _, branch_row in branch_stats.iterrows():
            branch = branch_row['branch']
            domain = self._map_branch_to_domain(branch, conceptual_domains)
            
            domain_coverage[domain]['branches'].append(branch)
            domain_coverage[domain]['total_terms'] += branch_row['branch_size']
            domain_coverage[domain]['collections'].update(branch_row['collections'])
        
        # Calculate domain-level statistics
        domain_stats = []
        for domain, coverage in domain_coverage.items():
            n_collections = len(coverage['collections'])
            n_branches = len(coverage['branches'])
            
            # Get branch stats for this domain
            domain_branch_stats = branch_stats[branch_stats['branch'].isin(coverage['branches'])]
            
            domain_stats.append({
                'domain': domain,
                'n_branches': n_branches,
                'total_terms': coverage['total_terms'],
                'n_collections': n_collections,
                'avg_collection_coverage': domain_branch_stats['collection_coverage_ratio'].mean(),
                'avg_term_coverage': domain_branch_stats['term_coverage_ratio'].mean(),
                'total_occurrences': domain_branch_stats['total_occurrences'].sum(),
                'branches': coverage['branches']
            })
        
        domain_stats_df = pd.DataFrame(domain_stats)
        
        return {
            'domain_stats': domain_stats_df,
            'conceptual_domains': conceptual_domains,
            'domain_coverage': dict(domain_coverage)
        }
    
    def _define_conceptual_domains(self) -> Dict[str, List[str]]:
        """Define conceptual domains and their associated keywords."""
        return {
            'Social Life & Culture': ['mensen', 'samenleving', 'cultuur', 'religie', 'feesten', 'sport'],
            'Politics & Government': ['politiek', 'regering', 'bestuur', 'verkiezingen', 'diplomatie'],
            'Economy & Industry': ['economie', 'industrie', 'handel', 'landbouw', 'transport'],
            'Geography & Environment': ['geografie', 'natuur', 'milieu', 'klimaat', 'landschap'],
            'History & Heritage': ['geschiedenis', 'erfgoed', 'monumenten', 'archeologie'],
            'Science & Technology': ['wetenschap', 'technologie', 'uitvindingen', 'onderzoek'],
            'Arts & Media': ['kunst', 'media', 'literatuur', 'muziek', 'film'],
            'Military & Defense': ['leger', 'defensie', 'oorlog', 'vredesoperaties'],
            'Education & Research': ['onderwijs', 'onderzoek', 'universiteiten', 'scholen'],
            'Health & Medicine': ['gezondheid', 'medicijnen', 'ziekenhuizen', 'hygiëne']
        }
    
    def _map_branch_to_domain(self, branch: str, domains: Dict[str, List[str]]) -> str:
        """Map a GTAA branch to a conceptual domain based on keyword matching."""
        branch_lower = branch.lower()
        
        for domain, keywords in domains.items():
            for keyword in keywords:
                if keyword in branch_lower:
                    return domain
        
        # Default mapping for unmapped branches
        return 'Other'
    
    def identify_systematic_gaps(self, df: pd.DataFrame, collection_column: str = 'fotocollectie') -> Dict:
        """
        Identify systematic gaps in the collective coverage.
        
        Args:
            df: DataFrame containing collection data
            collection_column: Name of the column containing collection identifiers
            
        Returns:
            Dictionary with systematic gap analysis
        """
        print("Identifying systematic gaps in collective coverage...")
        
        # Get all terms across all collections
        all_terms = set()
        collection_terms = {}
        collections = df[collection_column].value_counts()
        large_collections = collections[collections >= 1000].index.tolist()
        
        for collection in large_collections:
            collection_df = df[df[collection_column] == collection]
            collection_terms[collection] = set()
            
            for subjects_list in collection_df['subjects_list'].dropna():
                if isinstance(subjects_list, list):
                    collection_terms[collection].update(subjects_list)
                    all_terms.update(subjects_list)
        
        # Find terms that are completely missing
        all_gtaa_terms = set(self.graph.nodes()) - {self.root}
        missing_terms = all_gtaa_terms - all_terms
        
        # Find terms that are very rare (appear in < 10% of collections)
        term_frequency = Counter()
        for terms in collection_terms.values():
            term_frequency.update(terms)
        
        rare_threshold = max(1, len(large_collections) * 0.1)  # 10% of collections
        rare_terms = {term: count for term, count in term_frequency.items() 
                     if count < rare_threshold}
        
        # Analyze gaps by hierarchy level
        gap_analysis = {
            'missing_terms': list(missing_terms),
            'rare_terms': rare_terms,
            'missing_by_depth': defaultdict(list),
            'missing_by_branch': defaultdict(list),
            'coverage_by_depth': {},
            'coverage_by_branch': {}
        }
        
        # Analyze missing terms by depth
        for term in missing_terms:
            depth = self.node_depths.get(term, 0)
            gap_analysis['missing_by_depth'][depth].append(term)
            
            root_branch = self._find_root_branch(term)
            if root_branch:
                gap_analysis['missing_by_branch'][root_branch].append(term)
        
        # Calculate coverage statistics by depth
        for depth in self.depth_groups.keys():
            depth_terms = set(self.depth_groups[depth])
            covered_terms = depth_terms & all_terms
            gap_analysis['coverage_by_depth'][depth] = {
                'total_terms': len(depth_terms),
                'covered_terms': len(covered_terms),
                'coverage_ratio': len(covered_terms) / len(depth_terms) if depth_terms else 0
            }
        
        # Calculate coverage statistics by branch
        for branch in self.root_branches:
            branch_terms = nx.descendants(self.graph, branch)
            branch_terms.add(branch)
            covered_terms = branch_terms & all_terms
            gap_analysis['coverage_by_branch'][branch] = {
                'total_terms': len(branch_terms),
                'covered_terms': len(covered_terms),
                'coverage_ratio': len(covered_terms) / len(branch_terms) if branch_terms else 0
            }
        
        return gap_analysis
    
    def generate_bias_report(self, df: pd.DataFrame, collection_column: str = 'fotocollectie') -> str:
        """
        Generate a comprehensive bias analysis report.
        
        Args:
            df: DataFrame containing collection data
            collection_column: Name of the column containing collection identifiers
            
        Returns:
            Formatted report string
        """
        print("Generating comprehensive bias analysis report...")
        
        # Run all analyses
        branch_analysis = self.analyze_branch_representation(df, collection_column)
        domain_analysis = self.calculate_conceptual_domain_coverage(df, collection_column)
        gap_analysis = self.identify_systematic_gaps(df, collection_column)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("GTAA HIERARCHY BIAS ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Branch representation summary
        report.append("\n1. BRANCH REPRESENTATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"Total GTAA branches: {len(self.root_branches)}")
        report.append(f"Collections analyzed: {branch_analysis['total_collections']}")
        
        # Most represented branches
        report.append("\nMost represented branches (highest collection coverage):")
        for _, row in branch_analysis['most_represented_branches'].head(5).iterrows():
            coverage_pct = row['collection_coverage_ratio'] * 100
            report.append(f"  • {row['branch']}: {coverage_pct:.1f}% of collections")
        
        # Least represented branches
        report.append("\nLeast represented branches (lowest collection coverage):")
        for _, row in branch_analysis['least_represented_branches'].head(5).iterrows():
            coverage_pct = row['collection_coverage_ratio'] * 100
            report.append(f"  • {row['branch']}: {coverage_pct:.1f}% of collections")
        
        # Conceptual domain coverage
        report.append("\n2. CONCEPTUAL DOMAIN COVERAGE")
        report.append("-" * 40)
        domain_stats = domain_analysis['domain_stats']
        for _, row in domain_stats.iterrows():
            coverage_pct = row['avg_collection_coverage'] * 100
            report.append(f"  • {row['domain']}: {coverage_pct:.1f}% avg collection coverage")
        
        # Systematic gaps
        report.append("\n3. SYSTEMATIC GAPS IDENTIFICATION")
        report.append("-" * 40)
        report.append(f"Completely missing terms: {len(gap_analysis['missing_terms'])}")
        report.append(f"Rare terms (<10% collections): {len(gap_analysis['rare_terms'])}")
        
        # Coverage by depth
        report.append("\nCoverage by hierarchy depth:")
        for depth in sorted(gap_analysis['coverage_by_depth'].keys()):
            coverage = gap_analysis['coverage_by_depth'][depth]
            coverage_pct = coverage['coverage_ratio'] * 100
            report.append(f"  • Depth {depth}: {coverage_pct:.1f}% ({coverage['covered_terms']}/{coverage['total_terms']})")
        
        # Worst covered branches
        branch_coverage = gap_analysis['coverage_by_branch']
        worst_branches = sorted(branch_coverage.items(), key=lambda x: x[1]['coverage_ratio'])[:5]
        report.append("\nBranches with lowest coverage:")
        for branch, coverage in worst_branches:
            coverage_pct = coverage['coverage_ratio'] * 100
            report.append(f"  • {branch}: {coverage_pct:.1f}% ({coverage['covered_terms']}/{coverage['total_terms']})")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def save_analysis_results(self, df: pd.DataFrame, output_dir: Path, 
                            collection_column: str = 'fotocollectie'):
        """
        Save all analysis results to files.
        
        Args:
            df: DataFrame containing collection data
            output_dir: Directory to save results
            collection_column: Name of the column containing collection identifiers
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"Saving bias analysis results to {output_dir}...")
        
        # Run analyses
        branch_analysis = self.analyze_branch_representation(df, collection_column)
        domain_analysis = self.calculate_conceptual_domain_coverage(df, collection_column)
        gap_analysis = self.identify_systematic_gaps(df, collection_column)
        
        # Save branch analysis
        branch_analysis['branch_stats'].to_csv(output_dir / 'branch_representation.csv', index=False)
        branch_analysis['most_represented_branches'].to_csv(output_dir / 'most_represented_branches.csv', index=False)
        branch_analysis['least_represented_branches'].to_csv(output_dir / 'least_represented_branches.csv', index=False)
        
        # Save domain analysis
        domain_analysis['domain_stats'].to_csv(output_dir / 'conceptual_domain_coverage.csv', index=False)
        
        # Save gap analysis
        with open(output_dir / 'systematic_gaps.json', 'w') as f:
            # Convert sets to lists for JSON serialization
            json_gap_analysis = {
                'missing_terms': list(gap_analysis['missing_terms']),
                'rare_terms': gap_analysis['rare_terms'],
                'missing_by_depth': {str(k): v for k, v in gap_analysis['missing_by_depth'].items()},
                'missing_by_branch': gap_analysis['missing_by_branch'],
                'coverage_by_depth': gap_analysis['coverage_by_depth'],
                'coverage_by_branch': gap_analysis['coverage_by_branch']
            }
            json.dump(json_gap_analysis, f, indent=2)
        
        # Save comprehensive report
        report = self.generate_bias_report(df, collection_column)
        with open(output_dir / 'bias_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Results saved to {output_dir}") 