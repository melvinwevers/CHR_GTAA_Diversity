#!/usr/bin/env python3
"""
gtaa_visualizer.py - GTAA Tree Visualization with Coverage Highlighting

This module provides tools to visualize the GTAA vocabulary tree and highlight
areas that are covered by archival collections, helping to explain phylogenetic
diversity concepts and institutional bias patterns.
"""

import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


class GTAAVisualizer:
    """Visualizer for GTAA tree with coverage highlighting."""
    
    def __init__(self, gtaa_graph: nx.DiGraph):
        """
        Initialize the GTAA visualizer.
        
        Args:
            gtaa_graph: NetworkX directed graph representing the GTAA hierarchy
        """
        self.gtaa_graph = gtaa_graph
        self.coverage_data = {}
        
    def set_coverage_data(self, coverage_data: Dict[str, int]):
        """
        Set coverage data for nodes.
        
        Args:
            coverage_data: Dictionary mapping node names to their counts/coverage
        """
        self.coverage_data = coverage_data
    
    def render_tree(self, 
                   output_file: str = "gtaa_coverage_tree.png",
                   figsize: Tuple[int, int] = (16, 12),
                   coverage_threshold: int = 1,
                   show_counts: bool = True) -> str:
        """
        Render the GTAA tree with coverage highlighting.
        
        Args:
            output_file: Output file path
            figsize: Figure size (width, height) in inches
            coverage_threshold: Minimum count to consider as "covered"
            show_counts: Whether to show count labels on nodes
            
        Returns:
            Path to the generated image file
        """
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get tree layout
        pos = self._get_tree_layout()
        
        # Draw edges
        self._draw_edges(ax, pos)
        
        # Draw nodes with coverage highlighting
        self._draw_nodes(ax, pos, coverage_threshold, show_counts)
        
        # Add legend
        self._add_legend(ax)
        
        # Finalize plot
        ax.set_xlim(min(x for x, y in pos.values()) - 0.5, 
                   max(x for x, y in pos.values()) + 0.5)
        ax.set_ylim(min(y for x, y in pos.values()) - 0.5, 
                   max(y for x, y in pos.values()) + 0.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('GTAA Tree with Coverage Highlighting\n'
                    'Green nodes = covered areas, Gray nodes = uncovered areas', 
                    fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _get_tree_layout(self) -> Dict[str, Tuple[float, float]]:
        """Get hierarchical tree layout positions."""
        # Always use hierarchical layout for phylogenetic trees
        return self._get_hierarchical_tree_layout()
    
    def _get_hierarchical_tree_layout(self) -> Dict[str, Tuple[float, float]]:
        """Create a proper hierarchical tree layout (top-down)."""
        pos = {}
        
        # Find root
        root = None
        for node in self.gtaa_graph.nodes():
            if self.gtaa_graph.in_degree(node) == 0:
                root = node
                break
        
        if not root:
            # Fallback to simple layout
            return self._get_simple_hierarchical_layout()
        
        # Calculate levels using BFS
        levels = {root: 0}
        queue = [(root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            for neighbor in self.gtaa_graph.successors(node):
                if neighbor not in levels:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes in a hierarchical tree structure
        max_level = max(levels.values())
        
        for level in range(max_level + 1):
            if level in level_groups:
                nodes = level_groups[level]
                y = -level * 2  # Negative for top-down, spaced out
                
                # Distribute nodes horizontally within their level
                for i, node in enumerate(nodes):
                    # Center the nodes at this level
                    x = (i - len(nodes) / 2) * 3  # Spread out horizontally
                    pos[node] = (x, y)
        
        return pos
    
    def _get_simple_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """Simple hierarchical layout for large trees."""
        pos = {}
        
        # Find root
        root = None
        for node in self.gtaa_graph.nodes():
            if self.gtaa_graph.in_degree(node) == 0:
                root = node
                break
        
        if not root:
            # Fallback to spring layout
            return nx.spring_layout(self.gtaa_graph, k=2, iterations=10)
        
        # Calculate levels using BFS
        levels = {root: 0}
        queue = [(root, 0)]
        
        while queue:
            node, level = queue.pop(0)
            for neighbor in self.gtaa_graph.successors(node):
                if neighbor not in levels:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))
        
        # Group nodes by level
        level_groups = {}
        for node, level in levels.items():
            if level not in level_groups:
                level_groups[level] = []
            level_groups[level].append(node)
        
        # Position nodes
        max_level = max(levels.values())
        for level in range(max_level + 1):
            if level in level_groups:
                nodes = level_groups[level]
                y = -level  # Negative for top-down
                
                # Distribute nodes horizontally
                for i, node in enumerate(nodes):
                    x = (i - len(nodes) / 2) * 2  # Spread out horizontally
                    pos[node] = (x, y)
        
        return pos
    
    def _draw_edges(self, ax, pos: Dict[str, Tuple[float, float]]):
        """Draw tree edges."""
        for edge in self.gtaa_graph.edges():
            if edge[0] in pos and edge[1] in pos:  # Only draw if both nodes are positioned
                start_pos = pos[edge[0]]
                end_pos = pos[edge[1]]
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                       'k-', alpha=0.3, linewidth=1)
    
    def _draw_nodes(self, ax, pos: Dict[str, Tuple[float, float]], 
                   coverage_threshold: int, show_counts: bool):
        """Draw nodes with coverage-based styling."""
        for node in self.gtaa_graph.nodes():
            x, y = pos[node]
            coverage = self.coverage_data.get(node, 0)
            
            if coverage >= coverage_threshold:
                # Covered nodes - green gradient based on coverage
                max_coverage = max(self.coverage_data.values()) if self.coverage_data else 1
                intensity = min(1.0, coverage / max_coverage)
                color = (0.2, 0.8 * intensity + 0.2, 0.2)  # Green gradient
                size = 300 + coverage * 50
                alpha = 0.8
            else:
                # Uncovered nodes - gray
                color = (0.7, 0.7, 0.7)
                size = 200
                alpha = 0.5
            
            # Draw node
            circle = plt.Circle((x, y), size/1000, color=color, alpha=alpha, 
                              edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            
            # Add node label
            ax.text(x, y, node, fontsize=10, ha='center', va='center', 
                   weight='bold', color='black')
            
            # Add count label if requested
            if show_counts and coverage > 0:
                ax.text(x + 0.05, y, f"({coverage})", fontsize=8, 
                       ha='left', va='center', color='darkgreen', weight='bold')
    
    def _add_legend(self, ax):
        """Add legend explaining the visualization."""
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.2, 0.8, 0.2), 
                      markersize=10, label='Covered areas'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=(0.7, 0.7, 0.7), 
                      markersize=10, label='Uncovered areas')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    def create_coverage_summary_plot(self, 
                                   output_file: str = "gtaa_coverage_summary.png",
                                   figsize: Tuple[int, int] = (15, 10)) -> str:
        """
        Create a summary plot showing coverage statistics.
        
        Args:
            output_file: Output file path
            figsize: Figure size (width, height)
            
        Returns:
            Path to the generated image file
        """
        if not self.coverage_data:
            raise ValueError("No coverage data set. Use set_coverage_data() first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Coverage distribution histogram
        counts = list(self.coverage_data.values())
        ax1.hist(counts, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_xlabel('Coverage Count')
        ax1.set_ylabel('Number of Nodes')
        ax1.set_title('Distribution of Coverage Counts')
        ax1.grid(True, alpha=0.3)
        
        # 2. Coverage vs uncovered nodes pie chart
        covered = sum(1 for c in counts if c > 0)
        uncovered = len(counts) - covered
        ax2.pie([covered, uncovered], labels=['Covered', 'Uncovered'], 
                autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
        ax2.set_title('Coverage Status')
        
        # 3. Cumulative coverage by count
        sorted_counts = sorted(counts, reverse=True)
        cumulative = np.cumsum(sorted_counts)
        ax3.plot(range(1, len(cumulative) + 1), cumulative, 'b-', linewidth=2)
        ax3.set_xlabel('Number of Nodes (sorted by coverage)')
        ax3.set_ylabel('Cumulative Coverage')
        ax3.set_title('Cumulative Coverage Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Top 10 most covered nodes
        top_nodes = sorted(self.coverage_data.items(), key=lambda x: x[1], reverse=True)[:10]
        node_names = [name[:20] + '...' if len(name) > 20 else name for name, _ in top_nodes]
        node_counts = [count for _, count in top_nodes]
        
        bars = ax4.barh(range(len(node_names)), node_counts, color='lightblue')
        ax4.set_yticks(range(len(node_names)))
        ax4.set_yticklabels(node_names)
        ax4.set_xlabel('Coverage Count')
        ax4.set_title('Top 10 Most Covered Nodes')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                    f'{int(width)}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_hierarchical_heatmap(self, 
                                  output_file: str = "gtaa_hierarchical_heatmap.png",
                                  figsize: Tuple[int, int] = (14, 10)) -> str:
        """
        Create a hierarchical heatmap showing coverage patterns.
        
        Args:
            output_file: Output file path
            figsize: Figure size (width, height)
            
        Returns:
            Path to the generated image file
        """
        if not self.coverage_data:
            raise ValueError("No coverage data set. Use set_coverage_data() first.")
        
        # Create hierarchical clustering of nodes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get node hierarchy levels
        levels = self._get_node_levels()
        
        # Create heatmap data
        max_level = max(levels.values())
        level_nodes = {level: [] for level in range(max_level + 1)}
        
        for node, level in levels.items():
            level_nodes[level].append(node)
        
        # Sort nodes within each level by coverage
        for level in level_nodes:
            level_nodes[level].sort(key=lambda x: self.coverage_data.get(x, 0), reverse=True)
        
        # Create heatmap matrix
        all_nodes = []
        for level in range(max_level + 1):
            all_nodes.extend(level_nodes[level])
        
        # Create coverage matrix
        coverage_matrix = []
        for node in all_nodes:
            coverage = self.coverage_data.get(node, 0)
            coverage_matrix.append([coverage])
        
        # Create heatmap
        im = ax.imshow(coverage_matrix, cmap='Greens', aspect='auto')
        
        # Set labels
        ax.set_yticks(range(len(all_nodes)))
        ax.set_yticklabels(all_nodes, fontsize=10)
        ax.set_xticks([])
        
        # Add level separators
        current_pos = 0
        for level in range(max_level):
            current_pos += len(level_nodes[level])
            ax.axhline(y=current_pos - 0.5, color='black', linewidth=2)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coverage Count', fontsize=12)
        
        # Add level labels
        current_pos = 0
        for level in range(max_level + 1):
            level_size = len(level_nodes[level])
            level_center = current_pos + level_size // 2
            ax.text(-0.5, level_center, f'Level {level}', 
                   ha='right', va='center', fontsize=12, weight='bold',
                   transform=ax.get_yaxis_transform())
            current_pos += level_size
        
        ax.set_title('GTAA Hierarchical Coverage Heatmap', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def _get_node_levels(self) -> Dict[str, int]:
        """Get the hierarchical level of each node."""
        levels = {}
        
        # Find root node
        root = None
        for node in self.gtaa_graph.nodes():
            if self.gtaa_graph.in_degree(node) == 0:
                root = node
                break
        
        if root:
            # Calculate levels using BFS
            levels[root] = 0
            queue = [(root, 0)]
            
            while queue:
                node, level = queue.pop(0)
                for neighbor in self.gtaa_graph.successors(node):
                    if neighbor not in levels:
                        levels[neighbor] = level + 1
                        queue.append((neighbor, level + 1))
        
        return levels
    
    def get_coverage_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive coverage statistics.
        
        Returns:
            Dictionary with coverage statistics
        """
        if not self.coverage_data:
            return {}
        
        counts = list(self.coverage_data.values())
        total_nodes = len(counts)
        covered_nodes = sum(1 for c in counts if c > 0)
        total_coverage = sum(counts)
        
        return {
            'total_nodes': total_nodes,
            'covered_nodes': covered_nodes,
            'uncovered_nodes': total_nodes - covered_nodes,
            'coverage_percentage': (covered_nodes / total_nodes) * 100 if total_nodes > 0 else 0,
            'total_coverage_count': total_coverage,
            'average_coverage': total_coverage / total_nodes if total_nodes > 0 else 0,
            'max_coverage': max(counts) if counts else 0,
            'min_coverage': min(counts) if counts else 0,
            'coverage_std': np.std(counts) if len(counts) > 1 else 0
        }


def create_gtaa_visualization_example():
    """Create an example visualization to demonstrate the functionality."""
    
    # Create a simple example GTAA-like graph
    gtaa_graph = nx.DiGraph()
    
    # Add nodes with hierarchy
    nodes = [
        ("GTAA", "Root"),
        ("Culture", "GTAA"),
        ("History", "GTAA"),
        ("Science", "GTAA"),
        ("Art", "Culture"),
        ("Music", "Culture"),
        ("Literature", "Culture"),
        ("Ancient", "History"),
        ("Modern", "History"),
        ("Physics", "Science"),
        ("Biology", "Science"),
        ("Painting", "Art"),
        ("Sculpture", "Art"),
        ("Classical", "Music"),
        ("Jazz", "Music"),
        ("Poetry", "Literature"),
        ("Novel", "Literature")
    ]
    
    for child, parent in nodes:
        gtaa_graph.add_edge(parent, child)
    
    # Create visualizer
    visualizer = GTAAVisualizer(gtaa_graph)
    
    # Set example coverage data
    coverage_data = {
        "Art": 15,
        "Music": 8,
        "Literature": 12,
        "History": 20,
        "Science": 5,
        "Painting": 10,
        "Sculpture": 5,
        "Classical": 6,
        "Jazz": 2,
        "Poetry": 8,
        "Novel": 4,
        "Ancient": 12,
        "Modern": 8,
        "Physics": 3,
        "Biology": 2
    }
    
    visualizer.set_coverage_data(coverage_data)
    
    # Generate visualizations
    tree_file = visualizer.render_tree("example_gtaa_tree.png")
    summary_file = visualizer.create_coverage_summary_plot("example_coverage_summary.png")
    heatmap_file = visualizer.create_hierarchical_heatmap("example_hierarchical_heatmap.png")
    
    print(f"Generated visualizations:")
    print(f"  - Tree visualization: {tree_file}")
    print(f"  - Coverage summary: {summary_file}")
    print(f"  - Hierarchical heatmap: {heatmap_file}")
    
    # Print statistics
    stats = visualizer.get_coverage_statistics()
    print(f"\nCoverage Statistics:")
    print(f"  - Total nodes: {stats['total_nodes']}")
    print(f"  - Covered nodes: {stats['covered_nodes']} ({stats['coverage_percentage']:.1f}%)")
    print(f"  - Total coverage count: {stats['total_coverage_count']}")
    print(f"  - Average coverage: {stats['average_coverage']:.1f}")
    
    return visualizer


if __name__ == "__main__":
    create_gtaa_visualization_example() 