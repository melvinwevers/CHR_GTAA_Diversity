#!/usr/bin/env python3
"""
Explain Faith's Phylogenetic Diversity (PD) using GTAA ontology and a selected collection.

Usage:
    uv run python explain_faith_pd_with_gtaa.py --collection "Fotocollectie Anefo"
    uv run python explain_faith_pd_with_gtaa.py  # (all collections)
"""
import os
import argparse
import pandas as pd
import networkx as nx
from ast import literal_eval
from src.gtaa_visualizer import GTAAVisualizer


def build_gtaa_tree(ontology_csv: str) -> nx.DiGraph:
    """Build GTAA tree from ontology CSV as a proper phylogenetic hierarchy."""
    df = pd.read_csv(ontology_csv)
    G = nx.DiGraph()
    
    # First pass: add all nodes
    for _, row in df.iterrows():
        child = str(row['keyword']).strip()
        parent = str(row['broaderLabel']).strip() if pd.notna(row['broaderLabel']) else None
        if child:
            G.add_node(child)
        if parent and parent != 'nan':
            G.add_node(parent)
    
    # Second pass: add edges
    for _, row in df.iterrows():
        child = str(row['keyword']).strip()
        parent = str(row['broaderLabel']).strip() if pd.notna(row['broaderLabel']) else None
        if parent and parent != 'nan' and child:
            G.add_edge(parent, child)
    
    # Find or create a root node
    roots = [n for n, d in G.in_degree() if d == 0]
    
    if len(roots) == 0:
        # No root found, create one
        print("  Warning: No root node found, creating 'GTAA' as root")
        G.add_node("GTAA")
        # Connect all nodes with no incoming edges to the root
        for node in G.nodes():
            if G.in_degree(node) == 0 and node != "GTAA":
                G.add_edge("GTAA", node)
    elif len(roots) > 1:
        # Multiple roots, create a super-root
        print(f"  Warning: Multiple roots found ({len(roots)}), creating 'GTAA' as super-root")
        G.add_node("GTAA")
        for root in roots:
            G.add_edge("GTAA", root)
    
    return G


def compute_coverage(parquet_file: str, collection: str = None) -> dict:
    """Compute coverage counts for GTAA concepts in the selected collection."""
    df = pd.read_parquet(parquet_file)
    if collection:
        df = df[df['fotocollectie'] == collection]
    coverage = {}
    for subjects in df['subjects_list']:
        # subjects_list is a string representation of a list
        try:
            subjects_eval = literal_eval(subjects) if isinstance(subjects, str) else subjects
        except Exception:
            continue
        if not isinstance(subjects_eval, list):
            continue
        for subj in subjects_eval:
            subj = str(subj).strip()
            if subj:
                coverage[subj] = coverage.get(subj, 0) + 1
    return coverage


def filter_coverage_to_tree_nodes(coverage: dict, tree: nx.DiGraph) -> dict:
    """Filter coverage to only include nodes that exist in the GTAA tree."""
    tree_nodes = set(tree.nodes())
    filtered_coverage = {}
    missing_nodes = []
    
    for node, count in coverage.items():
        if node in tree_nodes:
            filtered_coverage[node] = count
        else:
            missing_nodes.append(node)
    
    if missing_nodes:
        print(f"  Warning: {len(missing_nodes)} concepts from collection not found in GTAA tree:")
        print(f"    Examples: {missing_nodes[:5]}{'...' if len(missing_nodes) > 5 else ''}")
    
    return filtered_coverage


def create_subtree_for_visualization(tree: nx.DiGraph, coverage: dict, max_nodes: int = 500) -> nx.DiGraph:
    """
    Create a smaller subtree for visualization by including:
    1. All covered nodes
    2. Their ancestors up to a certain depth
    3. Some siblings for context
    """
    covered_nodes = set(coverage.keys())
    
    # Find all ancestors of covered nodes
    ancestors = set()
    for node in covered_nodes:
        try:
            # Get all ancestors up to 3 levels up
            for ancestor in nx.ancestors(tree, node):
                if len(nx.shortest_path(tree, ancestor, node)) <= 4:  # Max 3 levels up
                    ancestors.add(ancestor)
        except nx.NetworkXNoPath:
            continue
    
    # Include some siblings of covered nodes for context
    siblings = set()
    for node in covered_nodes:
        try:
            for parent in tree.predecessors(node):
                for sibling in tree.successors(parent):
                    if sibling != node and sibling in tree.nodes():
                        siblings.add(sibling)
        except:
            continue
    
    # Combine all nodes we want to include
    all_nodes = covered_nodes | ancestors | siblings
    
    # If still too many nodes, limit to most important ones
    if len(all_nodes) > max_nodes:
        # Prioritize: covered nodes > ancestors > siblings
        priority_nodes = list(covered_nodes) + list(ancestors) + list(siblings)
        all_nodes = set(priority_nodes[:max_nodes])
    
    # Create subgraph
    subtree = tree.subgraph(all_nodes).copy()
    
    # Ensure the subtree is connected by adding necessary intermediate nodes
    # Find all edges that connect nodes in our subtree
    valid_edges = []
    for edge in tree.edges():
        if edge[0] in all_nodes and edge[1] in all_nodes:
            valid_edges.append(edge)
    
    # Rebuild the subtree with only valid edges
    clean_subtree = nx.DiGraph()
    clean_subtree.add_nodes_from(all_nodes)
    clean_subtree.add_edges_from(valid_edges)
    
    return clean_subtree


def faith_pd(tree: nx.DiGraph, covered_nodes: set) -> int:
    """
    Calculate Faith's PD: sum of unique branches connecting all covered nodes.
    Here, branch length = 1 for each edge.
    """
    if not covered_nodes:
        return 0
    
    # Find root (node with in-degree 0)
    roots = [n for n, d in tree.in_degree() if d == 0]
    if not roots:
        raise ValueError("No root found in GTAA tree.")
    root = roots[0]
    
    # Find the minimal subtree connecting all covered nodes
    # This is the union of all paths from root to each covered node
    edges_in_pd = set()
    
    for node in covered_nodes:
        try:
            path = nx.shortest_path(tree, root, node)
            # Add all edges in the path
            for i in range(len(path) - 1):
                edges_in_pd.add((path[i], path[i + 1]))
        except nx.NetworkXNoPath:
            print(f"  Warning: No path found from root to node '{node}'")
            continue
    
    return len(edges_in_pd)


def total_tree_pd(tree: nx.DiGraph) -> int:
    """Calculate total PD of the entire tree (all branches)."""
    return tree.number_of_edges()


def main():
    parser = argparse.ArgumentParser(description="Explain Faith's PD using GTAA and a collection.")
    parser.add_argument('--collection', type=str, default=None, help='Collection name (e.g., Fotocollectie Anefo)')
    parser.add_argument('--max-nodes', type=int, default=500, help='Maximum nodes in visualization (default: 500)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization, only compute PD')
    args = parser.parse_args()

    ontology_csv = 'data/external/gtaa_ontology.csv'
    parquet_file = 'data/processed/photos_archive.parquet'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    print(f"Building GTAA tree from {ontology_csv} ...")
    gtaa_tree = build_gtaa_tree(ontology_csv)
    print(f"  Full tree: {gtaa_tree.number_of_nodes()} nodes, {gtaa_tree.number_of_edges()} edges")

    print(f"Computing coverage from {parquet_file} ...")
    coverage = compute_coverage(parquet_file, args.collection)
    print(f"  Unique covered concepts: {len(coverage)}")

    # Filter coverage to only include nodes that exist in the GTAA tree
    filtered_coverage = filter_coverage_to_tree_nodes(coverage, gtaa_tree)

    # Calculate Faith's PD on full tree
    covered_nodes = set(filtered_coverage.keys())
    collection_pd = faith_pd(gtaa_tree, covered_nodes)
    total_pd = total_tree_pd(gtaa_tree)
    
    print(f"\nFaith's Phylogenetic Diversity Analysis:")
    print(f"  Collection PD: {collection_pd}")
    print(f"  Total tree PD: {total_pd}")
    print(f"  Coverage ratio: {collection_pd/total_pd*100:.1f}%")
    print(f"  (Collection PD = unique branches connecting covered concepts)")
    print(f"  (Total PD = all branches in the GTAA conceptual tree)")

    if not args.no_viz:
        print("Creating subtree for visualization...")
        subtree = create_subtree_for_visualization(gtaa_tree, filtered_coverage, args.max_nodes)
        print(f"  Subtree: {subtree.number_of_nodes()} nodes, {subtree.number_of_edges()} edges")
        
        # Filter coverage to only include nodes in subtree
        subtree_coverage = {k: v for k, v in filtered_coverage.items() if k in subtree.nodes()}
        
        print("Visualizing tree with coverage ...")
        visualizer = GTAAVisualizer(subtree)
        visualizer.set_coverage_data(subtree_coverage)
        tree_file = os.path.join(results_dir, 'faith_pd_tree.png')
        visualizer.render_tree(tree_file)
        print(f"  Tree visualization saved to: {tree_file}")

        # Print summary stats
        stats = visualizer.get_coverage_statistics()
        print("\nCoverage Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    print("\nDone!")

if __name__ == "__main__":
    main() 