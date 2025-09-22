#!/usr/bin/env python3
"""
Faith's Phylogenetic Diversity Visualization for GTAA Ontology
=============================================================

This script creates comprehensive visualizations to explain how Faith's PD works
with the GTAA (General Thesaurus Audiovisual Archives) ontology, demonstrating
the theoretical foundations for computational archival bias detection.

For CHR 2025 conference paper submission.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from matplotlib.patches import Rectangle
import seaborn as sns
from typing import List, Dict, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Set academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def create_sample_gtaa_hierarchy() -> nx.DiGraph:
    """
    Create a representative GTAA ontology structure for demonstration.
    Based on actual GTAA categories but simplified for visualization clarity.
    """
    G = nx.DiGraph()
    
    # Root level
    G.add_node("GTAA_ROOT", level=0, category="root")
    
    # Main categories (level 1)
    main_categories = [
        "Personer", "Geografie", "Onderwerpen", "Namen", "Genres"
    ]
    
    for cat in main_categories:
        G.add_node(cat, level=1, category="main")
        G.add_edge("GTAA_ROOT", cat, length=1.0)
    
    # Subcategories (level 2)
    subcategories = {
        "Personer": ["Politici", "Kunstenaars", "Wetenschappers", "Sporters"],
        "Geografie": ["Nederland", "Europa", "Azië", "Afrika"],
        "Onderwerpen": ["Politiek", "Cultuur", "Sport", "Wetenschap", "Oorlog"],
        "Namen": ["Organisaties", "Instellingen", "Bedrijven"],
        "Genres": ["Nieuws", "Documentaire", "Sport", "Amusement"]
    }
    
    for main_cat, sub_cats in subcategories.items():
        for sub_cat in sub_cats:
            G.add_node(sub_cat, level=2, category="sub")
            G.add_edge(main_cat, sub_cat, length=1.0)
    
    # Specific terms (level 3)
    specific_terms = {
        "Politici": ["Mark Rutte", "Wim Kok", "Dries van Agt"],
        "Nederland": ["Amsterdam", "Rotterdam", "Den Haag", "Utrecht"],
        "Politiek": ["Verkiezingen", "Parlement", "Democratie"],
        "Cultuur": ["Museum", "Theater", "Festival", "Kunst"],
        "Oorlog": ["Tweede Wereldoorlog", "Koude Oorlog", "Vietnam"],
        "Organisaties": ["VVD", "PvdA", "CDA", "D66"],
        "Nieuws": ["NOS Journaal", "RTL Nieuws", "Actualiteiten"]
    }
    
    for sub_cat, terms in specific_terms.items():
        if sub_cat in G.nodes():
            for term in terms:
                G.add_node(term, level=3, category="specific")
                G.add_edge(sub_cat, term, length=1.0)
    
    return G

def calculate_faith_pd_step_by_step(G: nx.DiGraph, subjects: List[str]) -> Tuple[float, Dict]:
    """
    Calculate Faith's PD with detailed step tracking for visualization.
    """
    root = "GTAA_ROOT"
    
    # Track which nodes are included in PD calculation
    pd_nodes = set()
    paths_info = {}
    
    # Find paths for each subject
    for subject in subjects:
        if subject in G.nodes():
            try:
                path = nx.shortest_path(G, root, subject)
                pd_nodes.update(path)
                paths_info[subject] = {
                    'path': path,
                    'length': len(path) - 1
                }
            except nx.NetworkXNoPath:
                continue
    
    # Calculate total PD (sum of branch lengths)
    total_pd = 0.0
    branch_contributions = {}
    
    for node in pd_nodes:
        if node == root:
            continue
        
        # Find parent
        predecessors = list(G.predecessors(node))
        if predecessors:
            parent = predecessors[0]
            branch_length = G[parent][node]['length']
            total_pd += branch_length
            branch_contributions[f"{parent}->{node}"] = branch_length
    
    return total_pd, {
        'pd_nodes': pd_nodes,
        'paths_info': paths_info,
        'branch_contributions': branch_contributions,
        'total_branches': len(branch_contributions)
    }

def create_faith_pd_explanation_figure():
    """
    Create a figure explaining Faith's PD concept with only two panels:
    1. Tree visualization showing Faith's PD calculation
    2. Bar chart comparing simple diversity (richness) and Faith's PD for two collections
    All labels and text are in English.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [2, 1]})

    # 1. Tree visualization showing PD calculation (left)
    ax_tree = axes[0]
    # Create sample GTAA hierarchy with Dutch names
    G = create_sample_gtaa_hierarchy()
    
    # Use new collections to demonstrate PD vs Richness
    # Collection A: Low richness, high PD (subjects are far apart)
    collection_a = ["Mark Rutte", "Tweede Wereldoorlog"]
    # Collection B: High richness, low PD (subjects are conceptually close)
    collection_b = ["Mark Rutte", "Wim Kok", "Dries van Agt"]
    
    # English label mapping for display
    label_map = {
        'GTAA_ROOT': 'Root',
        'Politiek': 'Politics',
        'Cultuur': 'Culture',
        'Oorlog': 'War',
        'Personer': 'People',
        'Onderwerpen': 'Subjects',
        'Politici': 'Politicians',
        'Mark Rutte': 'Mark Rutte',
        'Wim Kok': 'Wim Kok',
        'Dries van Agt': 'Dries van Agt',
        'Verkiezingen': 'Elections',
        'Amsterdam': 'Amsterdam',
        'Museum': 'Museum',
        'Tweede Wereldoorlog': 'World War II',
    }
    
    pd_a, details_a = calculate_faith_pd_step_by_step(G, collection_a)
    pd_b, details_b = calculate_faith_pd_step_by_step(G, collection_b)
    pos = create_hierarchical_layout(G)
    
    # Get all unique nodes and edges for drawing
    a_nodes = details_a.get('pd_nodes', set())
    b_nodes = details_b.get('pd_nodes', set())
    a_edges = set(details_a.get('branch_contributions', {}).keys())
    a_edges = {tuple(k.split('->')) for k in a_edges}
    b_edges = set(details_b.get('branch_contributions', {}).keys())
    b_edges = {tuple(k.split('->')) for k in b_edges}
    
    # Draw all edges in light gray as background
    nx.draw_networkx_edges(G, pos, edge_color='lightgray', width=1, alpha=0.5, ax=ax_tree)
    
    # Draw only the specific edges for each collection - no background edges
    # Draw Collection A path (red)
    nx.draw_networkx_edges(G, pos, edgelist=list(a_edges), edge_color='red', width=3, alpha=0.8, ax=ax_tree)
    nx.draw_networkx_nodes(G, pos, nodelist=list(a_nodes), node_color='red', node_size=150, alpha=0.8, ax=ax_tree)
    
    # Draw Collection B path (blue) - only edges not already drawn in red
    b_only_edges = b_edges - a_edges
    b_only_nodes = b_nodes - a_nodes
    nx.draw_networkx_edges(G, pos, edgelist=list(b_only_edges), edge_color='blue', width=3, alpha=0.8, ax=ax_tree)
    nx.draw_networkx_nodes(G, pos, nodelist=list(b_only_nodes), node_color='blue', node_size=150, alpha=0.8, ax=ax_tree)
    
    # Add text labels for collections, showing subjects
    collection_a_english = [label_map.get(s, s) for s in collection_a]
    collection_b_english = [label_map.get(s, s) for s in collection_b]
    ax_tree.text(0.25, 0.95, f"Collection A (PD={pd_a:.1f})\n{collection_a_english}", ha='center', va='top', fontsize=9, color='red', fontweight='bold', transform=ax_tree.transAxes)
    ax_tree.text(0.75, 0.95, f"Collection B (PD={pd_b:.1f})\n{collection_b_english}", ha='center', va='top', fontsize=9, color='blue', fontweight='bold', transform=ax_tree.transAxes)
    
    # Only label important nodes to avoid clutter
    important_nodes = a_nodes.union(b_nodes)
    labels = {node: label_map.get(node, node) for node in G.nodes() if node in important_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax_tree)
    
    ax_tree.set_title("Faith's PD Calculation: Comparing Two Archival Collections", 
                    fontsize=12, fontweight='bold', pad=20)
    
    ax_tree.axis('off')

    # 2. Bar chart comparing simple diversity and Faith's PD (right)
    ax_bar = axes[1]
    collections = ['Collection A', 'Collection B']
    pd_values = [pd_a, pd_b]
    richness_values = [len(collection_a), len(collection_b)]
    x = np.arange(len(collections))
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, pd_values, width, label="Faith's PD", color='skyblue', alpha=0.8)
    bars2 = ax_bar.bar(x + width/2, richness_values, width, label='Simple Richness', color='lightcoral', alpha=0.8)
    ax_bar.set_ylabel('Diversity Measure')
    ax_bar.set_title('Faith\'s PD vs Simple Richness')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(collections)
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    for bar in bars2:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    fig.suptitle("Faith's PD Calculation", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_hierarchical_layout(G: nx.DiGraph) -> Dict:
    """
    Create a hierarchical layout for the GTAA tree visualization.
    """
    pos = {}
    levels = {}
    
    # Group nodes by level
    for node in G.nodes():
        level = G.nodes[node]['level']
        if level not in levels:
            levels[level] = []
        levels[level].append(node)
    
    # Position nodes by level
    for level, nodes in levels.items():
        y = 1.0 - (level * 0.25)  # Top to bottom
        n_nodes = len(nodes)
        
        if n_nodes == 1:
            pos[nodes[0]] = (0.5, y)
        else:
            x_positions = np.linspace(0.1, 0.9, n_nodes)
            for i, node in enumerate(nodes):
                pos[node] = (x_positions[i], y)
    
    return pos

def create_collection_comparison_figure():
    """
    Create a figure comparing Faith's PD across different collection strategies.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Faith\'s PD Analysis: Collection Strategies and Bias Detection', 
                fontsize=16, fontweight='bold')
    
    # 1. Rarefaction curves showing diversity accumulation
    ax1 = axes[0, 0]
    
    # Simulate rarefaction data for different collection types
    sample_sizes = np.arange(10, 501, 10)
    
    # Different collection strategies
    strategies = {
        'Balanced Collecting': np.log(sample_sizes) * 2.5 + np.random.normal(0, 0.1, len(sample_sizes)),
        'Biased Collecting': np.log(sample_sizes) * 1.8 + np.random.normal(0, 0.15, len(sample_sizes)),
        'Elite Focus': np.log(sample_sizes) * 1.2 + np.random.normal(0, 0.1, len(sample_sizes)),
        'Random Sampling': np.log(sample_sizes) * 2.0 + np.random.normal(0, 0.2, len(sample_sizes))
    }
    
    colors = ['green', 'orange', 'red', 'blue']
    for i, (strategy, pd_values) in enumerate(strategies.items()):
        ax1.plot(sample_sizes, pd_values, label=strategy, color=colors[i], linewidth=2)
        ax1.fill_between(sample_sizes, pd_values - 0.3, pd_values + 0.3, 
                        color=colors[i], alpha=0.2)
    
    ax1.set_xlabel('Collection Size (Number of Items)')
    ax1.set_ylabel('Cumulative Faith\'s PD')
    ax1.set_title('Diversity Accumulation by Collection Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temporal bias analysis
    ax2 = axes[0, 1]
    
    decades = ['1920s', '1930s', '1940s', '1950s', '1960s', '1970s', '1980s', '1990s']
    observed_pd = [2.1, 2.3, 4.2, 3.8, 5.1, 4.9, 4.2, 3.6]
    expected_pd = [4.0] * len(decades)  # Hypothetical expected diversity
    
    ax2.plot(decades, observed_pd, 'o-', color='red', linewidth=3, markersize=8, 
            label='Observed PD')
    ax2.plot(decades, expected_pd, '--', color='gray', linewidth=2, 
            label='Expected PD (unbiased)')
    ax2.fill_between(decades, observed_pd, expected_pd, 
                    where=np.array(observed_pd) < np.array(expected_pd),
                    color='red', alpha=0.3, interpolate=True, label='Diversity deficit')
    
    ax2.set_xlabel('Time Period')
    ax2.set_ylabel('Faith\'s PD')
    ax2.set_title('Temporal Bias Detection:\nWartime Spike, Post-war Decline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Subject domain representation
    ax3 = axes[1, 0]
    
    domains = ['Politics', 'Culture', 'Sports', 'Economy', 'Social', 'International', 'Science', 'Religion']
    domain_pd = [8.2, 3.1, 4.5, 2.8, 2.2, 5.1, 1.8, 1.3]
    domain_items = [1200, 450, 680, 320, 280, 590, 180, 130]  # Number of items per domain
    
    # Create bubble chart
    bubble_sizes = [x/10 for x in domain_items]  # Scale for visibility
    colors_map = plt.cm.Set3(np.linspace(0, 1, len(domains)))
    
    scatter = ax3.scatter(domain_items, domain_pd, s=bubble_sizes, c=colors_map, 
                         alpha=0.7, edgecolors='black', linewidth=1)
    
    # Add domain labels
    for i, domain in enumerate(domains):
        ax3.annotate(domain, (domain_items[i], domain_pd[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax3.set_xlabel('Number of Archival Items')
    ax3.set_ylabel('Faith\'s PD per Domain')
    ax3.set_title('Domain-Level Bias Analysis:\nSize vs Conceptual Diversity')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(domain_items, domain_pd, 1)
    p = np.poly1d(z)
    ax3.plot(domain_items, p(domain_items), "r--", alpha=0.8, 
            label=f'Trend: PD = {z[0]:.4f}×items + {z[1]:.1f}')
    ax3.legend()
    
    # 4. Chao1 estimation for missing diversity
    ax4 = axes[1, 1]
    
    sample_coverage = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    observed_diversity = sample_coverage * 10  # Observed diversity scales with coverage
    chao1_estimate = observed_diversity / sample_coverage  # Chao1 estimation
    missing_diversity = chao1_estimate - observed_diversity
    
    ax4.bar(sample_coverage, observed_diversity, width=0.08, label='Observed Diversity', 
           color='skyblue', alpha=0.8)
    ax4.bar(sample_coverage, missing_diversity, bottom=observed_diversity, width=0.08, 
           label='Estimated Missing Diversity', color='lightcoral', alpha=0.8)
    
    ax4.set_xlabel('Collection Coverage (proportion of total universe)')
    ax4.set_ylabel('Faith\'s PD')
    ax4.set_title('Estimating Missing Archival Diversity:\nChao1 Approach')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i in [2, 5, 8]:  # Highlight specific coverage levels
        total_div = chao1_estimate[i]
        obs_div = observed_diversity[i]
        miss_div = missing_diversity[i]
        ax4.annotate(f'Missing:\n{miss_div:.1f} ({miss_div/total_div:.0%})', 
                    xy=(sample_coverage[i], total_div), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    fontsize=8, ha='center')
    
    plt.tight_layout()
    return fig

def create_methodological_framework_figure():
    """
    Create a figure showing the complete methodological framework.
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.5, 1], hspace=0.3, wspace=0.25)
    
    fig.suptitle('Computational Archival Bias Detection Framework:\nIntegrating Faith\'s PD with Critical Archival Theory', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # 1. Theoretical framework overview
    ax1 = fig.add_subplot(gs[0, :])
    
    # Create workflow diagram
    workflow_steps = [
        "Critical Archival\nTheory",
        "GTAA Ontology\nMapping", 
        "Faith's PD\nCalculation",
        "Bias Pattern\nDetection",
        "Reparative\nRecommendations"
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    
    # Draw workflow boxes
    box_width = 0.15
    box_spacing = 0.18
    start_x = 0.1
    
    for i, (step, color) in enumerate(zip(workflow_steps, colors)):
        x = start_x + i * box_spacing
        
        # Draw box
        box = FancyBboxPatch((x, 0.3), box_width, 0.4, 
                           boxstyle="round,pad=0.02", 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax1.add_patch(box)
        
        # Add text
        ax1.text(x + box_width/2, 0.5, step, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Draw arrow to next step
        if i < len(workflow_steps) - 1:
            ax1.arrow(x + box_width + 0.01, 0.5, box_spacing - box_width - 0.02, 0,
                     head_width=0.03, head_length=0.01, fc='black', ec='black')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Methodological Workflow: From Theory to Implementation', 
                 fontsize=14, fontweight='bold', y=0.85)
    
    # 2. Mathematical formulation detail
    ax2 = fig.add_subplot(gs[1, 0])
    
    math_content = r"""Faith's PD for Archive Analysis:

$PD_{collection} = \sum_{b \in B_{obs}} l_b$

Where:
• $B_{obs}$ = branches connecting all observed subjects to root
• $l_b$ = conceptual distance (branch length)

Bias Detection:
$\Delta PD = PD_{expected} - PD_{observed}$

Coverage Estimation:
$\hat{PD}_{total} = PD_{obs} + \frac{n_1^2}{2n_2}$

where $n_1, n_2$ are singleton/doubleton subject frequencies"""
    
    ax2.text(0.05, 0.5, math_content, ha='left', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Mathematical Foundation', fontsize=12, fontweight='bold')
    
    # 3. Validation approach
    ax3 = fig.add_subplot(gs[1, 1])
    
    validation_text = """Validation Strategy:

1. Theoretical Alignment
   • Compare with critical archival scholarship
   • Test against known historical biases

2. Methodological Rigor  
   • Bootstrap confidence intervals
   • Sensitivity analysis across parameters
   • Cross-validation with expert assessment

3. Community Engagement
   • Stakeholder feedback incorporation
   • Transparent algorithm explanation
   • Reparative collection guidance"""
    
    ax3.text(0.05, 0.5, validation_text, ha='left', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('Validation Framework', fontsize=12, fontweight='bold')
    
    # 4. Implementation considerations
    ax4 = fig.add_subplot(gs[1, 2])
    
    implementation_text = """Implementation Challenges:

• Ontology Completeness
  - GTAA coverage limitations
  - Missing subject relationships
  
• Cultural Appropriateness
  - Western-centric classification
  - Community representation needs

• Computational Scalability
  - Large collection processing
  - Real-time bias monitoring

• Interpretive Complexity
  - Quantitative-qualitative integration
  - Context-dependent meanings"""
    
    ax4.text(0.05, 0.5, implementation_text, ha='left', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.8))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Implementation Challenges', fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    print("Creating Faith's PD explanation figure...")
    fig1 = create_faith_pd_explanation_figure()
    fig1.savefig("results/figure_1_faith_pd_explanation.png", dpi=300, bbox_inches='tight')
    print("  Saved to results/figure_1_faith_pd_explanation.png")
    plt.close(fig1)

    print("\nCreating collection comparison figure...")
    fig2 = create_collection_comparison_figure()
    fig2.savefig("results/figure_2_collection_comparison.png", dpi=300, bbox_inches='tight')
    print("  Saved to results/figure_2_collection_comparison.png")
    plt.close(fig2)
    
    print("\nCreating methodological framework figure...")
    fig3 = create_methodological_framework_figure()
    fig3.savefig("results/figure_3_methodological_framework.png", dpi=300, bbox_inches='tight')
    print("  Saved to results/figure_3_methodological_framework.png")
    plt.close(fig3)

    print("\nAll figures for CHR 2025 paper generated successfully!") 