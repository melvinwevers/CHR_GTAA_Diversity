#!/usr/bin/env python3
"""
Simple test for cluster validation
"""

import pandas as pd
import numpy as np

# Create test data
np.random.seed(42)
test_data = pd.DataFrame({
    'coverage_ratio': np.random.uniform(0.1, 0.8, 20),
    'completeness_ratio': np.random.uniform(0.7, 0.95, 20)
})

print("Testing cluster validation with sample data...")
print(f"Test data shape: {test_data.shape}")
print(f"Coverage ratio range: {test_data['coverage_ratio'].min():.3f} - {test_data['coverage_ratio'].max():.3f}")
print(f"Completeness ratio range: {test_data['completeness_ratio'].min():.3f} - {test_data['completeness_ratio'].max():.3f}")

try:
    from cluster_validation import validate_archival_bias_clusters
    
    # Run validation
    validation_results = validate_archival_bias_clusters(test_data)
    
    # Print report
    print("\n" + "="*50)
    print("VALIDATION SUCCESSFUL!")
    print("="*50)
    print(validation_results['report'])
    
    print("\n✅ Cluster validation is working correctly!")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    print("Please check the cluster_validation.py file for issues.") 