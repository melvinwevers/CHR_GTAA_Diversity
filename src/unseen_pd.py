#!/usr/bin/env python3
"""
unseen_pd.py - Estimation of undetected/unseen phylogenetic diversity

This module implements the Chao1 estimator derived from Good-Turing theory
to estimate the amount of phylogenetic diversity not captured in the current
collection. This helps understand the completeness of the archival collection.

Based on: Chao et al. (2017) "Deciphering the enigma of undetected species,
phylogenetic, and functional diversity based on Good-Turing theory"
"""

import math
import networkx as nx
import pandas as pd
from typing import Any, Dict, Counter, Tuple, Optional
from collections import defaultdict


class UnseenPDEstimator:
    """Estimator for undetected phylogenetic diversity using Chao1/Good-Turing approach."""
    
    def __init__(self, graph: nx.DiGraph, branch_length_default: float = 1.0):
        """
        Initialize the unseen PD estimator.
        
        Args:
            graph: NetworkX directed graph with node counts
            branch_length_default: Default branch length value
        """
        self.graph = graph
        self.branch_length_default = branch_length_default
    
    def estimate_undetected_pd(self, img_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate undetected PD using Chao1 estimator derived from Good-Turing theory.
        
        Good-Turing frequency formula states that for branches appearing r times:
        λ_r = (r+1)g_{r+1}/(n*g_r)
        
        For undetected branches (r=0): λ_0 = g_1/n  
        For singleton branches (r=1): λ_1 = 2g_2/n
        
        Since λ_0 < λ_1, we derive: g_0 > g_1²/(2g_2) (Chao1 lower bound)
        
        Args:
            img_df: DataFrame with image data (for compatibility)
            
        Returns:
            Dictionary with estimation results
        """
        # Sum of branch lengths, grouped by the observation count of their child node
        branch_length_sums: Dict[int, float] = defaultdict(float)
        observed_pd = 0.0
        total_incidences = 0
        
        # Analyze each branch in the tree
        for parent, child in self.graph.edges():
            # Get the count for this child node
            child_count = self.graph.nodes[child].get('count', 0)
            
            # Get branch length
            branch_length = self.graph.edges[parent, child].get('length', self.branch_length_default)
            
            # If branch is observed, add to observed PD
            if child_count > 0:
                observed_pd += branch_length
            
            # Track sum of branch lengths by incidence frequency
            branch_length_sums[child_count] += branch_length
            total_incidences += child_count
        
        # Extract key values for Chao1 formula
        n = total_incidences
        g1 = branch_length_sums.get(1, 0)  # Sum of lengths of singleton branches
        g2 = branch_length_sums.get(2, 0)  # Sum of lengths of doubleton branches
        
        # Handle edge cases
        if n == 0:
            return self._empty_result()
        
        # Calculate Chao1 estimate using improved formula
        g0_hat = self._improved_good_turing_estimator(g1, g2, n)
        
        # Total estimated diversity
        total_estimated_pd = observed_pd + g0_hat
        
        # Calculate confidence intervals
        ci_lower, ci_upper = self._calculate_confidence_intervals(
            observed_pd, g0_hat, g1, g2, n
        )
        
        return {
            'PD_obs': observed_pd,
            'g1': g1,
            'g2': g2,
            'g0_hat': g0_hat,
            'PD_hat': total_estimated_pd,
            'CI_lo': ci_lower,
            'CI_hi': ci_upper,
            'total_incidences': total_incidences,
            'total_branches': len(self.graph.edges()),
            'coverage_ratio': observed_pd / total_estimated_pd if total_estimated_pd > 0 else 1.0
        }
    
    def _improved_good_turing_estimator(self, g1: float, g2: float, n: int) -> float:
        """
        Implement the improved Good-Turing estimator from Chiu et al. (2014b).
        
        This follows equation (1d) from Chao et al. (2017) and generally has
        smaller mean squared error than the original Good-Turing estimator.
        """
        if g2 > 0:
            # Standard Chao1 formula when doubletons exist
            return (n - 1) / n * (g1 ** 2) / (2 * g2)
        elif g1 > 1:
            # Modified formula when no doubletons but multiple singletons
            return (n - 1) / n * g1 * (g1 - 1) / 2
        else:
            # Cannot estimate when g1 ≤ 1 and g2 = 0
            # This represents complete sampling or insufficient data
            return 0
    
    def _calculate_confidence_intervals(
        self, observed_pd: float, g0_hat: float, 
        g1: float, g2: float, n: int
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate 95% confidence intervals for total PD estimate.
        
        Uses log-normal approximation based on Chao (1987).
        """
        if g1 > 0 and g2 > 0 and g0_hat > 0:
            # Calculate variance of g0_hat
            f = g1 / g2  # frequency ratio
            variance_g0_hat = g2 * (f**4 / 4 + f**3 + f**2 / 2)
            
            if variance_g0_hat > 0:
                # Log-normal confidence interval
                cv_squared = variance_g0_hat / (g0_hat ** 2)
                standard_error_log = math.sqrt(math.log(1 + cv_squared))
                
                # Calculate C factor
                C = math.exp(1.96 * standard_error_log)
                
                # Confidence intervals for g0_hat
                g0_hat_lower = g0_hat / C
                g0_hat_upper = g0_hat * C
                
                # Total PD confidence intervals
                return observed_pd + g0_hat_lower, observed_pd + g0_hat_upper
        
        return None, None
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return results for empty dataset."""
        return {
            'PD_obs': 0.0,
            'g1': 0,
            'g2': 0,
            'g0_hat': 0.0,
            'PD_hat': 0.0,
            'CI_lo': None,
            'CI_hi': None,
            'total_incidences': 0,
            'total_branches': 0,
            'coverage_ratio': 1.0
        }
    
    def get_estimation_summary(self, estimation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of the estimation results with interpretations.
        
        Args:
            estimation_results: Results from estimate_undetected_pd()
            
        Returns:
            Dictionary with summary statistics and interpretations
        """
        observed = estimation_results['PD_obs']
        estimated_total = estimation_results['PD_hat']
        undetected = estimation_results['g0_hat']
        
        # Calculate percentages
        if estimated_total > 0:
            observed_pct = (observed / estimated_total) * 100
            undetected_pct = (undetected / estimated_total) * 100
        else:
            observed_pct = 100.0
            undetected_pct = 0.0
        
        # Determine completeness level based on Good-Turing theory
        # Near-unbiased estimation occurs when singleton and undetected
        # branches have similar mean abundances
        if observed_pct >= 95:
            completeness_level = "Very High"
        elif observed_pct >= 85:
            completeness_level = "High"
        elif observed_pct >= 70:
            completeness_level = "Moderate"
        elif observed_pct >= 50:
            completeness_level = "Low"
        else:
            completeness_level = "Very Low"
        
        # Confidence interval interpretation
        ci_available = (estimation_results['CI_lo'] is not None and 
                       estimation_results['CI_hi'] is not None)
        
        if ci_available:
            ci_width = estimation_results['CI_hi'] - estimation_results['CI_lo']
            ci_relative_width = (ci_width / estimated_total) * 100 if estimated_total > 0 else 0
        else:
            ci_width = None
            ci_relative_width = None
        
        return {
            'observed_pd': observed,
            'estimated_total_pd': estimated_total,
            'estimated_undetected_pd': undetected,
            'observed_percentage': observed_pct,
            'undetected_percentage': undetected_pct,
            'completeness_level': completeness_level,
            'coverage_ratio': estimation_results['coverage_ratio'],
            'confidence_interval_available': ci_available,
            'confidence_interval_width': ci_width,
            'confidence_interval_relative_width': ci_relative_width
        }