#!/usr/bin/env python3
"""
Analyze node frequencies in GTAA graph and check for nodes above minimum spanning paths
that occur less than 3 times.
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import sys

# Add src directory to path
sys.path.append('src')

from src.archival_bias_detection import ArchivalBiasDetector
from src.faith_pd import FaithPDCalculator

def analyze_node_frequencies():
    """Analyze node frequencies and minimum spanning path coverage."""
    
    print("=" * 80)
    print("NODE FREQUENCY ANALYSIS IN GTAA GRAPH")
    print("=" * 80)
    
    # Configuration
    gtaa_csv_path = Path("data/external/gtaa_ontology.csv")
    parquet_path = Path("data/processed/photos_archive.parquet")
    
    # Load data and build graph
    print("\n1. Loading data and building graph...")
    detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=1000)
    detector.build_graph(apply_deduplication=False)
    df = detector.load_and_filter_data(parquet_path)
    
    # Calculate global subject frequencies
    print("\n2. Calculating global subject frequencies...")
    all_subjects = []
    for subjects_list in df['subjects_list'].dropna():
        if isinstance(subjects_list, list):
            all_subjects.extend(subjects_list)
    
    subject_counts = Counter(all_subjects)
    print(f"   Total subject occurrences: {len(all_subjects):,}")
    print(f"   Unique subjects: {len(subject_counts):,}")
    
    # Get collections
    collections = df['fotocollectie'].value_counts()
    large_collections = collections[collections >= 1000]
    print(f"   Large collections (≥1000 images): {len(large_collections)}")
    
    # Analyze each collection's minimum spanning path
    print("\n3. Analyzing minimum spanning paths for each collection...")
    
    pd_calculator = FaithPDCalculator(detector.graph)
    
    # Track nodes that appear in minimum spanning paths
    nodes_in_min_spanning = set()
    collection_min_spanning_nodes = {}
    
    for collection_name, collection_size in large_collections.items():
        if pd.isna(collection_name) or not str(collection_name).strip():
            continue
            
        collection_df = df[df['fotocollectie'] == collection_name]
        
        # Extract subjects for this collection
        collection_subjects = []
        for subjects_list in collection_df['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                collection_subjects.extend(subjects_list)
        
        unique_subjects = list(set(collection_subjects))
        valid_subjects = [s for s in unique_subjects if s in detector.graph.nodes()]
        
        if not valid_subjects:
            continue
        
        # Calculate minimum spanning path nodes
        min_spanning_nodes = get_minimum_spanning_nodes(detector.graph, valid_subjects)
        collection_min_spanning_nodes[collection_name] = min_spanning_nodes
        nodes_in_min_spanning.update(min_spanning_nodes)
        
        print(f"   {collection_name}: {len(valid_subjects)} subjects → {len(min_spanning_nodes)} nodes in min spanning")
    
    print(f"\n   Total unique nodes in any minimum spanning path: {len(nodes_in_min_spanning):,}")
    
    # Analyze nodes that are NOT in any minimum spanning path
    all_nodes = set(detector.graph.nodes()) - {"_DUMMY_ROOT_"}
    nodes_not_in_min_spanning = all_nodes - nodes_in_min_spanning
    
    print(f"\n4. Analyzing nodes NOT in any minimum spanning path...")
    print(f"   Total GTAA nodes: {len(all_nodes):,}")
    print(f"   Nodes in min spanning paths: {len(nodes_in_min_spanning):,}")
    print(f"   Nodes NOT in min spanning paths: {len(nodes_not_in_min_spanning):,}")
    
    # Check frequency of nodes not in minimum spanning paths
    nodes_not_in_min_spanning_freq = {node: subject_counts.get(node, 0) for node in nodes_not_in_min_spanning}
    
    # Count nodes by frequency
    freq_distribution = Counter(nodes_not_in_min_spanning_freq.values())
    print(f"\n   Frequency distribution of nodes NOT in min spanning paths:")
    for freq in sorted(freq_distribution.keys()):
        count = freq_distribution[freq]
        print(f"     Frequency {freq}: {count:,} nodes")
    
    # Check for nodes with frequency < 3
    low_freq_nodes = {node: freq for node, freq in nodes_not_in_min_spanning_freq.items() if freq < 3}
    print(f"\n   Nodes NOT in min spanning paths with frequency < 3: {len(low_freq_nodes):,}")
    
    if len(low_freq_nodes) > 0:
        print(f"\n   Examples of low-frequency nodes not in min spanning paths:")
        sorted_low_freq = sorted(low_freq_nodes.items(), key=lambda x: x[1])
        for node, freq in sorted_low_freq[:10]:
            print(f"     {node}: {freq} occurrences")
    
    # Analyze nodes ABOVE minimum spanning paths
    print(f"\n5. Analyzing nodes ABOVE minimum spanning paths...")
    
    # Get all nodes that are ancestors of nodes in minimum spanning paths
    nodes_above_min_spanning = set()
    for node in nodes_in_min_spanning:
        ancestors = nx.ancestors(detector.graph, node)
        nodes_above_min_spanning.update(ancestors)
    
    # Remove the dummy root
    nodes_above_min_spanning.discard("_DUMMY_ROOT_")
    
    print(f"   Nodes above min spanning paths: {len(nodes_above_min_spanning):,}")
    
    # Check frequency of nodes above minimum spanning paths
    nodes_above_freq = {node: subject_counts.get(node, 0) for node in nodes_above_min_spanning}
    
    # Count nodes by frequency
    above_freq_distribution = Counter(nodes_above_freq.values())
    print(f"\n   Frequency distribution of nodes ABOVE min spanning paths:")
    for freq in sorted(above_freq_distribution.keys()):
        count = above_freq_distribution[freq]
        print(f"     Frequency {freq}: {count:,} nodes")
    
    # Check for nodes with frequency < 3
    low_freq_above = {node: freq for node, freq in nodes_above_freq.items() if freq < 3}
    print(f"\n   Nodes ABOVE min spanning paths with frequency < 3: {len(low_freq_above):,}")
    
    if len(low_freq_above) > 0:
        print(f"\n   Examples of low-frequency nodes above min spanning paths:")
        sorted_low_freq_above = sorted(low_freq_above.items(), key=lambda x: x[1])
        for node, freq in sorted_low_freq_above[:10]:
            print(f"     {node}: {freq} occurrences")
    
    # Summary
    print(f"\n6. SUMMARY:")
    print(f"   - Total GTAA nodes: {len(all_nodes):,}")
    print(f"   - Nodes in min spanning paths: {len(nodes_in_min_spanning):,}")
    print(f"   - Nodes above min spanning paths: {len(nodes_above_min_spanning):,}")
    print(f"   - Nodes above with freq < 3: {len(low_freq_above):,}")
    print(f"   - Nodes NOT in min spanning paths: {len(nodes_not_in_min_spanning):,}")
    print(f"   - Nodes NOT in min spanning with freq < 3: {len(low_freq_nodes):,}")
    
    return {
        'nodes_in_min_spanning': nodes_in_min_spanning,
        'nodes_above_min_spanning': nodes_above_min_spanning,
        'nodes_not_in_min_spanning': nodes_not_in_min_spanning,
        'low_freq_above': low_freq_above,
        'low_freq_not_in': low_freq_nodes,
        'collection_min_spanning_nodes': collection_min_spanning_nodes
    }

def get_minimum_spanning_nodes(graph, subjects):
    """Get all nodes in the minimum spanning path for given subjects."""
    if not subjects:
        return set()
    
    # Filter subjects that exist in the graph
    valid_subjects = [s for s in subjects if s in graph.nodes]
    if not valid_subjects:
        return set()
    
    # Find the minimum spanning path connecting all subjects
    spanning_branches = set()
    
    # For each pair of subjects, find the path between them
    for i, subject1 in enumerate(valid_subjects):
        for subject2 in valid_subjects[i+1:]:
            try:
                # Find shortest path between these two subjects
                path = nx.shortest_path(graph, subject1, subject2)
                
                # Add all branches in this path to the spanning set
                for j in range(len(path) - 1):
                    branch = (path[j], path[j + 1])
                    spanning_branches.add(branch)
                    
            except nx.NetworkXNoPath:
                # If no path exists, try reverse direction
                try:
                    path = nx.shortest_path(graph, subject2, subject1)
                    for j in range(len(path) - 1):
                        branch = (path[j], path[j + 1])
                        spanning_branches.add(branch)
                except nx.NetworkXNoPath:
                    # If still no path, subjects are disconnected
                    continue
    
    # Get all nodes involved in the spanning branches
    spanning_nodes = set()
    for parent, child in spanning_branches:
        spanning_nodes.add(parent)
        spanning_nodes.add(child)
    
    return spanning_nodes

if __name__ == "__main__":
    results = analyze_node_frequencies() 