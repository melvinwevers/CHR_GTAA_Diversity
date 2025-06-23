"""
Archival Bias Detection using Phylogenetic Diversity Coverage Ratios

This module provides the ArchivalBiasDetector class for analyzing institutional bias
patterns in archival collections using Faith's Phylogenetic Diversity metrics.
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
from tqdm import tqdm
import json

from .graph_builder import GTAAGraphBuilder
from .faith_pd import FaithPDCalculator
from .unseen_pd import UnseenPDEstimator


class ArchivalBiasDetector:
    """
    Detects archival bias patterns using phylogenetic diversity coverage ratios.
    
    This class integrates graph building, PD calculation, and unseen diversity estimation
    to quantify how well different collections cover the conceptual space defined by
    the GTAA vocabulary.
    """
    
    def __init__(self, gtaa_csv_path: Path, min_collection_size: int = 1000):
        """
        Initialize the ArchivalBiasDetector.
        
        Args:
            gtaa_csv_path: Path to the GTAA vocabulary CSV file
            min_collection_size: Minimum collection size to include in analysis
        """
        self.gtaa_csv_path = Path(gtaa_csv_path)
        self.min_collection_size = min_collection_size
        self.graph = None
        self.gtaa_total_pd = None
        self.graph_builder = GTAAGraphBuilder(self.gtaa_csv_path)
        self.pd_calculator = None
        self.unseen_estimator = None
        
    def build_graph(self, apply_deduplication: bool = False) -> None:
        """
        Build the GTAA vocabulary graph.
        
        Args:
            apply_deduplication: Whether to apply deduplication during build
        """
        print("Building GTAA vocabulary graph...")
        self.graph = self.graph_builder.build_graph()
        
        # Initialize calculators after graph is built
        self.pd_calculator = FaithPDCalculator(self.graph)
        self.unseen_estimator = UnseenPDEstimator(self.graph)
        
        if apply_deduplication:
            print("Applying deduplication...")
            # Note: deduplication will be handled in analyze_archival_bias
            pass
            
    def load_and_filter_data(self, parquet_path: Path) -> pd.DataFrame:
        """
        Load and filter the archival data.
        
        Args:
            parquet_path: Path to the parquet file containing archival data
            
        Returns:
            Filtered DataFrame
        """
        print(f"Loading data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df):,} records")
        
        # Deserialize JSON strings back to lists for nested columns
        for col in ['subjects_list', 'persons', 'locations_standardized']:
            if col in df.columns:
                print(f"Deserializing {col} from JSON strings...")
                df[col] = df[col].apply(
                    lambda x: json.loads(x) if isinstance(x, str) and x.startswith('[') else x
                )
        
        # Filter records with subjects
        df_with_subjects = pd.DataFrame(df[df['subjects_list'].notna()])
        print(f"{len(df_with_subjects):,} records have subjects")
        
        return df_with_subjects
        
    def calculate_global_gtaa_pd(self) -> float:
        """
        Calculate the global GTAA phylogenetic diversity.
        
        Returns:
            Global PD value
        """
        print("Calculating global GTAA PD...")
        if self.graph is None or self.pd_calculator is None:
            raise ValueError("Graph and calculators not built. Call build_graph() first.")
            
        # Get all non-root nodes
        all_subjects = [node for node in self.graph.nodes() if node != "_DUMMY_ROOT_"]
        self.gtaa_total_pd = self.pd_calculator.calculate_faith_pd(all_subjects)
        
        print(f"Global GTAA PD: {self.gtaa_total_pd:.2f}")
        return self.gtaa_total_pd
        
    def _apply_data_driven_deduplication(self, df: pd.DataFrame) -> None:
        """
        Apply data-driven deduplication to the graph.
        
        Args:
            df: DataFrame containing subject data for frequency analysis
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
            
        print("Applying deduplication...")
        print("  Calculating subject frequencies from dataset for informed deduplication...")
        
        # Calculate subject frequencies
        all_subjects = []
        for subjects_list in df['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                all_subjects.extend(subjects_list)
        
        subject_counts = pd.Series(all_subjects).value_counts()
        print(f"Found {len(subject_counts)} unique subjects in dataset")
        print(f" Total subject occurrences: {subject_counts.sum():,}")
        
        # Find multi-parent nodes
        multi_parent_nodes = [
            node for node in self.graph.nodes() 
            if node != "_DUMMY_ROOT_" and len(list(self.graph.predecessors(node))) > 1
        ]
        
        if not multi_parent_nodes:
            print("No multi-parent nodes found - graph is already a tree")
            return
            
        print(f"Deduplicating {len(multi_parent_nodes)} nodes with multiple parents (strategy: maxfreq)...")
        
        edges_removed = 0
        for node in tqdm(multi_parent_nodes, desc="Deduplicating nodes"):
            parents = list(self.graph.predecessors(node))
            if len(parents) <= 1:
                continue
                
            # Find parent with highest frequency in dataset
            parent_frequencies = {}
            for parent in parents:
                parent_frequencies[parent] = subject_counts.get(parent, 0)
                
            # Keep edge to parent with highest frequency
            best_parent = max(parent_frequencies.keys(), key=lambda x: parent_frequencies[x])
            
            # Remove edges to other parents
            for parent in parents:
                if parent != best_parent:
                    self.graph.remove_edge(parent, node)
                    edges_removed += 1
                    
        print(f"Deduplication complete: processed {len(multi_parent_nodes)} nodes, removed {edges_removed} edges")
        
        # Verify result
        is_tree = nx.is_tree(self.graph)
        remaining_multi = [
            node for node in self.graph.nodes() 
            if node != "_DUMMY_ROOT_" and len(list(self.graph.predecessors(node))) > 1
        ]
        
        print(f"Data-driven deduplication complete:")
        print(f"Multi-parent nodes remaining: {len(remaining_multi)}")
        
    def _find_steiner_tree_nodes(self, subjects: List[str]) -> Set[str]:
        """Find all nodes in the minimal Steiner tree connecting subjects to root."""
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
            
        root = "_DUMMY_ROOT_"
        steiner_nodes = set(subjects)
        
        # For each subject, add all nodes on path to root
        for subject in subjects:
            try:
                path = nx.shortest_path(self.graph, root, subject)
                steiner_nodes.update(path)
            except nx.NetworkXNoPath:
                # Subject not reachable from root, skip
                continue
        
        return steiner_nodes
        
    def analyze_ontology_structure(self) -> Dict:
        """
        Analyze the structure of the GTAA ontology.
        
        Returns:
            Dictionary with ontology structure analysis
        """
        if self.graph is None:
            raise ValueError("Graph not built. Call build_graph() first.")
            
        print("Analyzing GTAA ontology structure...")
        
        # Read the vocabulary CSV
        vocab = pd.read_csv(self.gtaa_csv_path)
        
        # Basic statistics
        total_terms = len(vocab)
        unique_keywords = vocab['keyword'].nunique()
        unique_broader_labels = vocab['broaderLabel'].dropna().nunique()
        
        # Calculate depth for each node
        depths = {}
        root = "_DUMMY_ROOT_"
        
        for node in self.graph.nodes():
            if node != root:
                try:
                    depth = nx.shortest_path_length(self.graph, root, node)
                    depths[node] = depth
                except nx.NetworkXNoPath:
                    depths[node] = 0
        
        max_depth = max(depths.values()) if depths else 0
        avg_depth = sum(depths.values()) / len(depths) if depths else 0
        
        # Count nodes at each depth
        depth_distribution = {}
        for depth in depths.values():
            depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        
        # Find root terms (terms with no broader label)
        root_terms = vocab[vocab['broaderLabel'].isna()]['keyword'].tolist()
        
        # Find leaf terms (terms that are not broader labels for any other term)
        broader_labels = set(vocab['broaderLabel'].dropna().unique())
        leaf_terms = [term for term in vocab['keyword'].unique() if term not in broader_labels]
        
        structure_analysis = {
            'total_terms': total_terms,
            'unique_keywords': unique_keywords,
            'unique_broader_labels': unique_broader_labels,
            'root_terms': root_terms,
            'n_root_terms': len(root_terms),
            'leaf_terms': leaf_terms,
            'n_leaf_terms': len(leaf_terms),
            'max_depth': max_depth,
            'avg_depth': avg_depth,
            'depth_distribution': depth_distribution,
            'node_depths': depths
        }
        
        return structure_analysis
        
    def calculate_detailed_ontology_pd(self) -> Dict:
        """
        Calculate detailed Faith's PD analysis for the entire GTAA ontology.
        
        Returns:
            Dictionary with detailed ontology PD results
        """
        if self.graph is None or self.pd_calculator is None:
            raise ValueError("Graph and calculators not built. Call build_graph() first.")
            
        print("Calculating detailed GTAA ontology PD...")
        
        # Get all non-root nodes
        all_terms = [node for node in self.graph.nodes() if node != "_DUMMY_ROOT_"]
        
        # Calculate Faith's PD
        total_pd = self.pd_calculator.calculate_faith_pd(all_terms)
        
        # Calculate average PD per term
        avg_pd_per_term = total_pd / len(all_terms) if all_terms else 0
        
        # Find the Steiner tree nodes (all nodes in the minimal connecting tree)
        steiner_nodes = self._find_steiner_tree_nodes(all_terms)
        
        # Calculate branch length distribution
        branch_lengths = []
        for node in steiner_nodes:
            for predecessor in self.graph.predecessors(node):
                if predecessor in steiner_nodes:
                    edge_length = self.graph.edges[predecessor, node].get('length', 1.0)
                    branch_lengths.append(edge_length)
        
        # Get graph statistics
        stats = self.graph_builder.get_graph_statistics()
        
        results = {
            'total_pd': total_pd,
            'n_terms': len(all_terms),
            'n_steiner_nodes': len(steiner_nodes),
            'n_branches': len(branch_lengths),
            'avg_pd_per_term': avg_pd_per_term,
            'avg_branch_length': sum(branch_lengths) / len(branch_lengths) if branch_lengths else 0,
            'max_branch_length': max(branch_lengths) if branch_lengths else 0,
            'min_branch_length': min(branch_lengths) if branch_lengths else 0,
            'graph_stats': stats,
            'steiner_nodes': list(steiner_nodes),
            'all_terms': all_terms
        }
        
        return results
        
    def print_ontology_analysis(self, pd_results: Dict, structure_analysis: Dict):
        """Print the ontology analysis results in a formatted way."""
        print("\n" + "="*60)
        print("GTAA ONTOLOGY PHYLOGENETIC DIVERSITY ANALYSIS")
        print("="*60)
        
        print(f"\nONTOLOGY STATISTICS:")
        print(f"  Total terms: {pd_results['n_terms']:,}")
        print(f"  Total nodes in tree: {pd_results['n_steiner_nodes']:,}")
        print(f"  Total branches: {pd_results['n_branches']:,}")
        print(f"  Root terms: {structure_analysis['n_root_terms']:,}")
        print(f"  Leaf terms: {structure_analysis['n_leaf_terms']:,}")
        print(f"  Maximum depth: {structure_analysis['max_depth']}")
        print(f"  Average depth: {structure_analysis['avg_depth']:.2f}")
        
        print(f"\nFAITH'S PHYLOGENETIC DIVERSITY:")
        print(f"  Total PD: {pd_results['total_pd']:.2f}")
        print(f"  Average PD per term: {pd_results['avg_pd_per_term']:.4f}")
        print(f"  Average branch length: {pd_results['avg_branch_length']:.4f}")
        print(f"  Branch length range: {pd_results['min_branch_length']:.4f} - {pd_results['max_branch_length']:.4f}")
        
        print(f"\nDEPTH DISTRIBUTION:")
        for depth in sorted(structure_analysis['depth_distribution'].keys()):
            count = structure_analysis['depth_distribution'][depth]
            percentage = (count / pd_results['n_terms']) * 100
            print(f"  Depth {depth}: {count:,} terms ({percentage:.1f}%)")
        
        print(f"\nROOT TERMS (top-level categories):")
        for term in structure_analysis['root_terms'][:10]:  # Show first 10
            print(f"  • {term}")
        if len(structure_analysis['root_terms']) > 10:
            print(f"  ... and {len(structure_analysis['root_terms']) - 10} more")
        
        print(f"\nSAMPLE LEAF TERMS (most specific categories):")
        for term in structure_analysis['leaf_terms'][:10]:  # Show first 10
            print(f"  • {term}")
        if len(structure_analysis['leaf_terms']) > 10:
            print(f"  ... and {len(structure_analysis['leaf_terms']) - 10} more")
        
        print("\n" + "="*60)
        
    def analyze_archival_bias(self, df: pd.DataFrame, collection_column: str) -> Dict:
        """
        Analyze archival bias across collections.
        
        Args:
            df: DataFrame containing archival data
            collection_column: Column name containing collection identifiers
            
        Returns:
            Dictionary containing analysis results
        """
        print("Starting archival bias analysis...")
        
        # Apply data-informed deduplication
        self._apply_data_driven_deduplication(df)
        
        # Calculate global PD
        global_pd = self.calculate_global_gtaa_pd()
        
        print("Analyzing all collections...")
        
        # Get collections and filter by size
        collections = df[collection_column].value_counts()
        large_collections = collections[collections >= self.min_collection_size]
        
        print(f"Found {len(collections)} collections")
        print(f"{len(large_collections)} collections meet minimum size requirement ({self.min_collection_size} images)")
        
        collection_results = []
        
        for collection_name, collection_size in pd.Series(large_collections).items():
            # Filter out collections with no name (None, NaN, empty string, or only whitespace)
            if pd.isna(collection_name) or not str(collection_name).strip():
                continue
                
            collection_df = df[df[collection_column] == collection_name]
            
            print(f"Processing: {collection_name} ({collection_size} images)")
            
            # Extract all subjects for this collection
            collection_subjects = []
            for subjects_list in collection_df['subjects_list'].dropna():
                if isinstance(subjects_list, list):
                    collection_subjects.extend(subjects_list)
            
            # Get unique subjects and filter to those in graph
            unique_subjects = list(set(collection_subjects))
            valid_subjects = [s for s in unique_subjects if s in self.graph.nodes()]
            
            if not valid_subjects:
                print(f"No valid subjects found for {collection_name}")
                continue
                
            # Ensure calculators are initialized
            if self.pd_calculator is None or self.unseen_estimator is None:
                raise ValueError("Calculators not initialized. Call build_graph() first.")
                
            # Calculate collection PD
            collection_pd = self.pd_calculator.calculate_faith_pd(valid_subjects)
            
            # Set up graph node counts for this collection
            # Reset all counts to 0 first
            for node in self.graph.nodes():
                self.graph.nodes[node]['count'] = 0
            # Count subject occurrences in the collection
            for subjects_list in collection_df['subjects_list'].dropna():
                if isinstance(subjects_list, list):
                    for subj in subjects_list:
                        if subj in self.graph.nodes():
                            self.graph.nodes[subj]['count'] = self.graph.nodes[subj].get('count', 0) + 1
            
            # Calculate unseen PD using the correct method
            unseen_result = self.unseen_estimator.estimate_undetected_pd(collection_df)
            unseen_pd = unseen_result['g0_hat']  # This is the estimated unseen PD
            
            # Calculate ratios
            coverage_ratio = collection_pd / global_pd
            completeness_ratio = collection_pd / (collection_pd + unseen_pd) if (collection_pd + unseen_pd) > 0 else 1.0
            efficiency_ratio = coverage_ratio / np.log10(collection_size)
            
            collection_results.append({
                'fotocollectie': collection_name,
                'collection_size': collection_size,
                'n_unique_subjects': len(valid_subjects),
                'collection_pd': collection_pd,
                'unseen_pd': unseen_pd,
                'coverage_ratio': coverage_ratio,
                'completeness_ratio': completeness_ratio,
                'efficiency_ratio': efficiency_ratio
            })
        
        return {
            'global_pd': global_pd,
            'collection_results': collection_results,
            'total_collections': len(collections),
            'analyzed_collections': len(collection_results)
        } 