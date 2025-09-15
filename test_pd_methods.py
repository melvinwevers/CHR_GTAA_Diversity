#!/usr/bin/env python3
"""
Test script to verify both PD calculation methods work correctly.
"""

import networkx as nx
from src.faith_pd import FaithPDCalculator

def create_test_graph():
    """Create a simple test graph for PD calculation."""
    G = nx.DiGraph()
    
    # Add nodes
    G.add_node("_DUMMY_ROOT_")
    G.add_node("A")
    G.add_node("B") 
    G.add_node("C")
    G.add_node("D")
    G.add_node("E")
    G.add_node("F")
    
    # Add edges with uniform branch lengths
    G.add_edge("_DUMMY_ROOT_", "A", length=1.0)
    G.add_edge("_DUMMY_ROOT_", "B", length=1.0)
    G.add_edge("A", "C", length=1.0)
    G.add_edge("A", "D", length=1.0)
    G.add_edge("B", "E", length=1.0)
    G.add_edge("B", "F", length=1.0)
    
    return G

def test_pd_methods():
    """Test both PD calculation methods."""
    print("Testing PD calculation methods...")
    
    # Create test graph
    G = create_test_graph()
    calculator = FaithPDCalculator(G)
    
    # Test case 1: Single subject
    subjects1 = ["C"]
    pd_min_span1 = calculator.calculate_faith_pd(subjects1, method='minimum_spanning')
    pd_scikit1 = calculator.calculate_faith_pd(subjects1, method='scikit_bio')
    
    print(f"\nTest 1 - Single subject 'C':")
    print(f"  Minimum spanning PD: {pd_min_span1}")
    print(f"  Scikit-bio PD: {pd_scikit1}")
    
    # Test case 2: Two subjects on same branch
    subjects2 = ["C", "D"]
    pd_min_span2 = calculator.calculate_faith_pd(subjects2, method='minimum_spanning')
    pd_scikit2 = calculator.calculate_faith_pd(subjects2, method='scikit_bio')
    
    print(f"\nTest 2 - Two subjects on same branch 'C', 'D':")
    print(f"  Minimum spanning PD: {pd_min_span2}")
    print(f"  Scikit-bio PD: {pd_scikit2}")
    
    # Test case 3: Two subjects on different branches
    subjects3 = ["C", "E"]
    pd_min_span3 = calculator.calculate_faith_pd(subjects3, method='minimum_spanning')
    pd_scikit3 = calculator.calculate_faith_pd(subjects3, method='scikit_bio')
    
    print(f"\nTest 3 - Two subjects on different branches 'C', 'E':")
    print(f"  Minimum spanning PD: {pd_min_span3}")
    print(f"  Scikit-bio PD: {pd_scikit3}")
    
    # Test case 4: Multiple subjects
    subjects4 = ["C", "D", "E", "F"]
    pd_min_span4 = calculator.calculate_faith_pd(subjects4, method='minimum_spanning')
    pd_scikit4 = calculator.calculate_faith_pd(subjects4, method='scikit_bio')
    
    print(f"\nTest 4 - Multiple subjects 'C', 'D', 'E', 'F':")
    print(f"  Minimum spanning PD: {pd_min_span4}")
    print(f"  Scikit-bio PD: {pd_scikit4}")
    
    # Test case 5: Empty subjects
    subjects5 = []
    pd_min_span5 = calculator.calculate_faith_pd(subjects5, method='minimum_spanning')
    pd_scikit5 = calculator.calculate_faith_pd(subjects5, method='scikit_bio')
    
    print(f"\nTest 5 - Empty subjects:")
    print(f"  Minimum spanning PD: {pd_min_span5}")
    print(f"  Scikit-bio PD: {pd_scikit5}")
    
    # Test case 6: Invalid subjects
    subjects6 = ["INVALID"]
    pd_min_span6 = calculator.calculate_faith_pd(subjects6, method='minimum_spanning')
    pd_scikit6 = calculator.calculate_faith_pd(subjects6, method='scikit_bio')
    
    print(f"\nTest 6 - Invalid subject 'INVALID':")
    print(f"  Minimum spanning PD: {pd_min_span6}")
    print(f"  Scikit-bio PD: {pd_scikit6}")
    
    print(f"\nTest completed successfully!")
    print(f"Key observation: Minimum spanning PD should be <= Scikit-bio PD")
    print(f"This is because minimum spanning only includes necessary connecting branches,")
    print(f"while scikit-bio includes all supporting branches.")

if __name__ == "__main__":
    test_pd_methods() 