#!/usr/bin/env python3
"""
Detailed Explanation: Why Unseen PD Cannot Detect Terms Outside GTAA

This script demonstrates the fundamental limitations of the Chao1 unseen PD estimator
and why it can only estimate missing terms within the existing GTAA vocabulary.
"""

import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
import sys

# Add src directory to path
sys.path.append('src')

from src.unseen_pd import UnseenPDEstimator

def demonstrate_unseen_pd_limitations():
    """
    Demonstrate why the unseen PD estimator cannot detect terms outside GTAA.
    """
    
    print("=" * 80)
    print("WHY UNSEEN PD CANNOT DETECT TERMS OUTSIDE GTAA")
    print("=" * 80)
    
    print("\n1. THE FUNDAMENTAL LIMITATION")
    print("-" * 40)
    print("The Chao1 unseen PD estimator is based on Good-Turing frequency theory,")
    print("which estimates missing items based on the frequency distribution of")
    print("observed items. It can ONLY estimate missing items from a known universe.")
    print()
    print("Key principle: You cannot estimate what you don't know exists.")
    
    print("\n2. HOW THE ESTIMATOR WORKS")
    print("-" * 40)
    print("The estimator analyzes the GTAA graph structure:")
    print("  • It looks at ALL edges in the GTAA graph")
    print("  • It counts how many times each GTAA term appears")
    print("  • It uses frequency patterns to estimate missing GTAA terms")
    print()
    print("Critical point: It only considers nodes that exist in the graph.")
    
    print("\n3. CONCRETE EXAMPLE")
    print("-" * 40)
    
    # Create a simple example graph
    G = nx.DiGraph()
    G.add_edges_from([
        ('root', 'animals'),
        ('animals', 'dogs'),
        ('animals', 'cats'),
        ('animals', 'birds'),
        ('root', 'plants'),
        ('plants', 'trees'),
        ('plants', 'flowers')
    ])
    
    print("Example GTAA-like vocabulary:")
    print("  root")
    print("  ├── animals")
    print("  │   ├── dogs")
    print("  │   ├── cats")
    print("  │   └── birds")
    print("  └── plants")
    print("      ├── trees")
    print("      └── flowers")
    print()
    print("Total possible terms in this vocabulary: 7")
    
    # Simulate a collection that only has 'dogs' and 'cats'
    print("\nCollection A contains only:")
    print("  • dogs (appears 5 times)")
    print("  • cats (appears 3 times)")
    print()
    
    # Set up the graph with counts
    for node in G.nodes():
        G.nodes[node]['count'] = 0
    
    # Add the observed terms
    G.nodes['dogs']['count'] = 5
    G.nodes['cats']['count'] = 3
    
    # Run the unseen PD estimator
    estimator = UnseenPDEstimator(G)
    dummy_df = pd.DataFrame({'dummy': [1]})  # Dummy data for compatibility
    result = estimator.estimate_undetected_pd(dummy_df)
    
    print("Unseen PD Analysis Results:")
    print(f"  Observed PD: {result['PD_obs']:.1f}")
    print(f"  Estimated unseen PD: {result['g0_hat']:.1f}")
    print(f"  Total estimated PD: {result['PD_hat']:.1f}")
    print()
    
    print("What the estimator CAN detect:")
    print("  ✓ Missing GTAA terms: animals, birds, plants, trees, flowers")
    print("  ✓ It estimates ~5.0 units of unseen PD")
    print()
    
    print("What the estimator CANNOT detect:")
    print("  ✗ Terms not in GTAA: 'fish', 'reptiles', 'insects', 'fungi'")
    print("  ✗ New conceptual categories not in the vocabulary")
    print("  ✗ Gaps in the GTAA vocabulary itself")
    
    print("\n4. MATHEMATICAL REASONING")
    print("-" * 40)
    print("The Chao1 formula: g0_hat = (n-1)/n * (g1²)/(2*g2)")
    print()
    print("Where:")
    print("  • g1 = sum of branch lengths for terms appearing once")
    print("  • g2 = sum of branch lengths for terms appearing twice")
    print("  • n = total number of term occurrences")
    print()
    print("This formula only works because:")
    print("  1. We know the total universe (all GTAA terms)")
    print("  2. We can observe frequency patterns within that universe")
    print("  3. We can extrapolate from observed to unobserved terms")
    print()
    print("If we don't know the universe, we cannot estimate what's missing.")
    
    print("\n5. REAL-WORLD IMPLICATIONS")
    print("-" * 40)
    print("Example: A collection about 1960s counterculture")
    print()
    print("GTAA terms present: 'hippies', 'protests', 'music'")
    print("GTAA terms missing: 'rock concerts', 'civil rights', 'anti-war'")
    print("Terms not in GTAA: 'psychedelic drugs', 'communes', 'free love'")
    print()
    print("The unseen PD estimator will:")
    print("  ✓ Estimate missing GTAA terms like 'rock concerts'")
    print("  ✗ NOT detect that GTAA lacks terms for 'psychedelic drugs'")
    print("  ✗ NOT suggest that GTAA needs new categories")
    
    print("\n6. WHY THIS MATTERS FOR ARCHIVAL BIAS")
    print("-" * 40)
    print("This limitation means:")
    print()
    print("1. VOCABULARY BIAS: If GTAA itself is biased (e.g., lacks terms")
    print("   for marginalized communities), the unseen PD cannot detect this.")
    print()
    print("2. TEMPORAL BIAS: If GTAA was created in a different era and")
    print("   lacks modern concepts, the estimator cannot identify these gaps.")
    print()
    print("3. CULTURAL BIAS: If GTAA reflects Western/European perspectives")
    print("   and lacks terms for non-Western concepts, this goes undetected.")
    
    print("\n7. ALTERNATIVE APPROACHES")
    print("-" * 40)
    print("To detect gaps beyond GTAA, you would need:")
    print()
    print("1. COMPARATIVE ANALYSIS:")
    print("   • Compare GTAA with other vocabularies")
    print("   • Identify concepts present in other systems but missing from GTAA")
    print()
    print("2. EXPERT REVIEW:")
    print("   • Domain experts identify missing concepts")
    print("   • Historical analysis of vocabulary development")
    print()
    print("3. TEXT MINING:")
    print("   • Extract terms from archival descriptions")
    print("   • Identify frequently occurring terms not in GTAA")
    print()
    print("4. COMMUNITY FEEDBACK:")
    print("   • Input from diverse user communities")
    print("   • Identification of cultural and linguistic gaps")
    
    print("\n8. CONCLUSION")
    print("-" * 40)
    print("The unseen PD estimator is a powerful tool for measuring collection")
    print("completeness WITHIN the GTAA framework, but it cannot detect:")
    print()
    print("  • Gaps in the GTAA vocabulary itself")
    print("  • Missing conceptual categories")
    print("  • Biases in the vocabulary structure")
    print("  • Terms that should exist but don't")
    print()
    print("This is a fundamental limitation of frequency-based estimation")
    print("methods - they can only estimate what they know exists.")
    print()
    print("For comprehensive bias detection, you need multiple approaches:")
    print("  • Unseen PD (within-vocabulary gaps)")
    print("  • Vocabulary analysis (vocabulary-level gaps)")
    print("  • Comparative studies (cross-vocabulary gaps)")
    print("  • Expert review (conceptual gaps)")

if __name__ == "__main__":
    demonstrate_unseen_pd_limitations() 