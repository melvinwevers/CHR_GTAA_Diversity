#!/usr/bin/env python3
"""
graph_builder.py - GTAA vocabulary graph construction and manipulation

This module handles building and processing the GTAA (General Thesaurus for 
Audiovisual Archives) vocabulary graph, including deduplication strategies.
"""

import networkx as nx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional


class GTAAGraphBuilder:
    """Builder class for constructing and processing GTAA vocabulary graphs."""
    
    def __init__(self, vocab_csv: Path):
        """
        Initialize the graph builder.
        
        Args:
            vocab_csv: Path to the vocabulary CSV file
        """
        self.vocab_csv = vocab_csv
        self.graph: Optional[nx.DiGraph] = None
        
    def build_graph(self) -> nx.DiGraph:
        """
        Build the GTAA vocabulary graph from CSV file.
        
        Returns:
            NetworkX directed graph representing the vocabulary hierarchy
        """
        print("Building GTAA vocabulary graph...")
        
        # Read vocabulary CSV
        vocab = pd.read_csv(self.vocab_csv)
        
        # Validate required columns
        required_columns = ['broaderLabel', 'keyword']
        missing_columns = [col for col in required_columns if col not in vocab.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in vocabulary: {missing_columns}")
        
        # Create edges dataframe
        edges = (
            vocab[~vocab["broaderLabel"].isna()][["broaderLabel", "keyword"]]
            .drop_duplicates()
            .rename(columns={"broaderLabel": "parent", "keyword": "child"})
        )
        
        # Initialize graph
        G = nx.DiGraph()
        
        # Add edges with progress bar
        print("  Adding edges to graph...")
        for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Processing edges"):
            G.add_edge(row["parent"], row["child"])
        
        # Ensure all labels are present as nodes
        all_nodes = pd.concat([edges["parent"], edges["child"]]).unique()
        print(f"  Adding {len(all_nodes)} unique nodes...")
        G.add_nodes_from(tqdm(all_nodes, desc="Adding nodes"))
        
        # Connect all roots to a dummy super-root
        roots = [n for n, deg in G.in_degree() if deg == 0]
        print(f"  Connecting {len(roots)} root nodes to dummy root...")
        G.add_node("_DUMMY_ROOT_")
        G.add_edges_from([("_DUMMY_ROOT_", r) for r in roots])
        
        # Set default branch lengths (all branches have length 1)
        nx.set_edge_attributes(G, 1.0, "length")
        
        print(f"Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        self.graph = G
        return G
    
    def deduplicate_parents(self, raw_counts: Dict[str, int], strategy: str = "maxfreq") -> None:
        """
        Deduplicate multi-parent nodes to create a proper tree.
        
        Args:
            raw_counts: Dictionary of label frequencies from the dataset
            strategy: Deduplication strategy ('maxfreq', 'longest', 'first')

        Note: 
        - maxfreq: keeps the parent whose subtree is hit most often (raw frequency)
        - longest: keeps the parent on the deepest branch (longest root→parent path)
        - first: keeps the first parent encountered (original CSV order)
        """
        if self.graph is None:
            raise ValueError("Graph must be built before deduplication")
        
        G = self.graph
        root = "_DUMMY_ROOT_"
        
        # Find nodes with multiple parents
        multi_parent_nodes = [
            node for node in G.nodes() 
            if node != root and len(list(G.predecessors(node))) > 1
        ]
        
        if not multi_parent_nodes:
            print("No duplicate parentage found")
            return
        
        print(f"Deduplicating {len(multi_parent_nodes)} nodes with multiple parents (strategy: {strategy})...")
        
        dedup_stats = {"removed_edges": 0, "nodes_processed": 0}
        
        for node in tqdm(multi_parent_nodes, desc="Deduplicating nodes"):
            predecessors = list(G.predecessors(node))
            if len(predecessors) <= 1:
                continue
            
            # Choose the best parent based on strategy
            if strategy == "maxfreq":
                best_parent = max(predecessors, key=lambda p: raw_counts.get(p, 0))
            elif strategy == "longest":
                best_parent = max(predecessors, key=lambda p: self._get_path_length(G, root, p))
            else:  # first
                best_parent = predecessors[0]
            
            # Remove edges from other parents
            for parent in predecessors:
                if parent != best_parent and G.has_edge(parent, node):
                    G.remove_edge(parent, node)
                    dedup_stats["removed_edges"] += 1
            
            dedup_stats["nodes_processed"] += 1
        
        print(f"✓ Deduplication complete: processed {dedup_stats['nodes_processed']} nodes, "
              f"removed {dedup_stats['removed_edges']} edges")
    
    def _get_path_length(self, G: nx.DiGraph, source: str, target: str) -> int:
        """Get path length between two nodes, returning 0 if no path exists."""
        try:
            return nx.shortest_path_length(G, source, target)
        except nx.NetworkXNoPath:
            return 0
    
    def clean_graph(self) -> None:
        """Remove orphaned nodes that are not reachable from the root."""
        if self.graph is None:
            raise ValueError("Graph must be built before cleaning")
        
        G = self.graph
        root = "_DUMMY_ROOT_"
        
        # Find all reachable nodes
        reachable = nx.descendants(G, root) | {root}
        
        # Remove unreachable nodes
        to_remove = set(G.nodes()) - reachable
        if to_remove:
            print(f"Removing {len(to_remove)} orphaned nodes")
            G.remove_nodes_from(to_remove)
        else:
            print("No orphaned nodes found")
    
    def compute_subtree_densities(self) -> Dict[str, int]:
        """
        Compute subtree density for each node in the graph.
        
        Returns:
            Dictionary mapping node names to their subtree densities
        """
        if self.graph is None:
            raise ValueError("Graph must be built before computing subtree densities")
        
        print("Computing subtree densities...")
        
        subtree_densities = {}
        
        for node in tqdm(self.graph.nodes(), desc="Computing subtree densities"):
            # Count descendants (including the node itself)
            descendants = nx.descendants(self.graph, node)
            subtree_densities[node] = len(descendants) + 1  # +1 to include node itself
        
        print(f"Computed subtree densities for {len(subtree_densities)} nodes")
        return subtree_densities
    
    def get_subtree_density(self, term: str, subtree_densities: Dict[str, int]) -> int:
        """Get subtree density for a specific term."""
        return subtree_densities.get(term, 1)

    def get_graph_statistics(self) -> Dict:
        """Get comprehensive statistics about the graph."""
        if self.graph is None:
            raise ValueError("Graph must be built before getting statistics")
        
        G = self.graph
        root = "_DUMMY_ROOT_"
        
        # Basic statistics
        stats = {
            "total_nodes": len(G.nodes()),
            "total_edges": len(G.edges()),
            "leaf_nodes": len([n for n in G.nodes() if G.out_degree(n) == 0]),
            "root_children": len(list(G.successors(root))),
        }
        
        # Depth statistics
        depths = []
        for node in G.nodes():
            if node != root:
                try:
                    depth = nx.shortest_path_length(G, root, node)
                    depths.append(depth)
                except nx.NetworkXNoPath:
                    pass
        
        if depths:
            stats.update({
                "max_depth": max(depths),
                "min_depth": min(depths),
                "avg_depth": sum(depths) / len(depths),
                "nodes_with_depth": len(depths)
            })
        
        # Connectivity statistics
        try:
            stats["is_tree"] = nx.is_tree(G)
            stats["is_dag"] = nx.is_directed_acyclic_graph(G)
        except:
            stats["is_tree"] = False
            stats["is_dag"] = False
        
        return stats
    
    def print_graph_summary(self) -> None:
        """Print a summary of the graph structure."""
        if self.graph is None:
            print("No graph built yet")
            return
        
        stats = self.get_graph_statistics()
        
        print("\n📊 GTAA Graph Summary")
        print("=" * 40)
        print(f"  Total nodes: {stats['total_nodes']:,}")
        print(f"  Total edges: {stats['total_edges']:,}")
        print(f"  Leaf nodes: {stats['leaf_nodes']:,}")
        print(f"  Root children: {stats['root_children']}")
        
        if 'max_depth' in stats:
            print(f"  Max depth: {stats['max_depth']}")
            print(f"  Avg depth: {stats['avg_depth']:.1f}")
        
        print(f"  Is tree: {stats['is_tree']}")
        print(f"  Is DAG: {stats['is_dag']}")
        print("=" * 40)

