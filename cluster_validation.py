#!/usr/bin/env python3
"""
Statistical validation of four-cluster solution for archival bias analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import bootstrap
import warnings
warnings.filterwarnings('ignore')


class ClusterValidator:
    """Validate clustering solutions for archival bias analysis"""
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize cluster validator with results data.
        
        Args:
            results_df: DataFrame with coverage_ratio and completeness_ratio columns
        """
        self.results_df = results_df.copy()
        self.scaler = StandardScaler()
        
        # Prepare data for clustering
        self.X = self.results_df[['coverage_ratio', 'completeness_ratio']].values
        self.X_scaled = self.scaler.fit_transform(self.X)
        
    def silhouette_analysis(self, n_clusters: int = 4) -> Dict[str, float]:
        """
        Perform silhouette analysis to validate clustering.
        
        Args:
            n_clusters: Number of clusters to test
            
        Returns:
            Dictionary with silhouette scores and analysis results
        """
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(self.X_scaled)
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(self.X_scaled, cluster_labels)
        silhouette_samples_vals = silhouette_samples(self.X_scaled, cluster_labels)
        
        # Calculate silhouette scores for each cluster
        cluster_silhouettes = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_silhouettes[i] = silhouette_samples_vals[cluster_mask].mean()
        
        return {
            'silhouette_score': silhouette_avg,
            'silhouette_samples': silhouette_samples_vals,
            'cluster_labels': cluster_labels,
            'cluster_silhouettes': cluster_silhouettes,
            'kmeans_model': kmeans
        }
    
    def gap_statistic(self, max_clusters: int = 8, n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate gap statistic to determine optimal number of clusters.
        
        Args:
            max_clusters: Maximum number of clusters to test
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with gap statistics and optimal cluster number
        """
        def compute_inertia(X, labels):
            """Compute within-cluster sum of squares."""
            unique_labels = np.unique(labels)
            inertia = 0
            for label in unique_labels:
                cluster_points = X[labels == label]
                centroid = cluster_points.mean(axis=0)
                inertia += np.sum((cluster_points - centroid) ** 2)
            return inertia
        
        # Calculate gap statistic
        gaps = []
        sks = []
        
        for k in range(1, max_clusters + 1):
            # Real data clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.X_scaled)
            real_inertia = compute_inertia(self.X_scaled, kmeans.labels_)
            
            # Bootstrap reference data
            bootstrap_inertias = []
            for _ in range(n_bootstrap):
                # Generate reference data (uniform distribution)
                X_ref = np.random.uniform(
                    self.X_scaled.min(axis=0), 
                    self.X_scaled.max(axis=0), 
                    size=self.X_scaled.shape
                )
                
                kmeans_ref = KMeans(n_clusters=k, random_state=None, n_init=10)
                kmeans_ref.fit(X_ref)
                ref_inertia = compute_inertia(X_ref, kmeans_ref.labels_)
                bootstrap_inertias.append(ref_inertia)
            
            # Calculate gap and standard error
            gap = np.mean(np.log(bootstrap_inertias)) - np.log(real_inertia)
            sk = np.sqrt(np.var(np.log(bootstrap_inertias)) * (1 + 1/n_bootstrap))
            
            gaps.append(gap)
            sks.append(sk)
        
        # Find optimal number of clusters
        optimal_k = np.argmax(gaps) + 1
        
        return {
            'gaps': np.array(gaps),
            'sks': np.array(sks),
            'optimal_k': optimal_k,
            'cluster_range': range(1, max_clusters + 1)
        }
    
    def compare_median_vs_clustering(self) -> Dict[str, Any]:
        """
        Compare median-based classification with statistically validated clustering.
        """
        # Median-based classification (current approach)
        coverage_median = self.results_df['coverage_ratio'].median()
        completeness_median = self.results_df['completeness_ratio'].median()
        
        median_labels = np.zeros(len(self.results_df), dtype=int)
        median_labels[(self.results_df['coverage_ratio'] > coverage_median) & 
                     (self.results_df['completeness_ratio'] < completeness_median)] = 0  # High-Low
        median_labels[(self.results_df['coverage_ratio'] < coverage_median) & 
                     (self.results_df['completeness_ratio'] < completeness_median)] = 1  # Low-Low
        median_labels[(self.results_df['coverage_ratio'] < coverage_median) & 
                     (self.results_df['completeness_ratio'] > completeness_median)] = 2  # Low-High
        median_labels[(self.results_df['coverage_ratio'] > coverage_median) & 
                     (self.results_df['completeness_ratio'] > completeness_median)] = 3  # High-High
        
        # Clustering-based classification
        silhouette_results = self.silhouette_analysis(n_clusters=4)
        cluster_labels = silhouette_results['cluster_labels']
        
        # Calculate agreement between methods
        agreement = np.mean(median_labels == cluster_labels)
        
        return {
            'median_labels': median_labels,
            'cluster_labels': cluster_labels,
            'agreement': agreement,
            'silhouette_score': silhouette_results['silhouette_score'],
            'coverage_median': coverage_median,
            'completeness_median': completeness_median
        }
    
    def bootstrap_validation(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap validation of clustering stability.
        """
        def clustering_agreement(data, labels1, labels2):
            """Calculate agreement between two clusterings."""
            from sklearn.metrics import adjusted_rand_score
            return adjusted_rand_score(labels1, labels2)
        
        # Original clustering
        kmeans_original = KMeans(n_clusters=4, random_state=42, n_init=10)
        original_labels = kmeans_original.fit_predict(self.X_scaled)
        
        # Bootstrap samples
        agreements = []
        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(self.X_scaled), len(self.X_scaled), replace=True)
            X_bootstrap = self.X_scaled[indices]
            
            # Cluster bootstrap sample
            kmeans_bootstrap = KMeans(n_clusters=4, random_state=None, n_init=10)
            bootstrap_labels = kmeans_bootstrap.fit_predict(X_bootstrap)
            
            # Calculate agreement with original
            agreement = clustering_agreement(X_bootstrap, original_labels[indices], bootstrap_labels)
            agreements.append(agreement)
        
        return {
            'agreements': agreements,
            'mean_agreement': np.mean(agreements),
            'std_agreement': np.std(agreements),
            'ci_95_lower': np.percentile(agreements, 2.5),
            'ci_95_upper': np.percentile(agreements, 97.5)
        }
    
    def create_validation_report(self) -> str:
        """
        Generate comprehensive validation report.
        """
        # Perform all analyses
        silhouette_results = self.silhouette_analysis()
        gap_results = self.gap_statistic()
        comparison_results = self.compare_median_vs_clustering()
        bootstrap_results = self.bootstrap_validation()
        
        report = f"""
        📊 CLUSTER VALIDATION REPORT
        =============================
        
        🎯 FOUR-CLUSTER SOLUTION VALIDATION
        
        1. SILHOUETTE ANALYSIS
        ----------------------
        Overall Silhouette Score: {silhouette_results['silhouette_score']:.3f}
        Interpretation: {'Good' if silhouette_results['silhouette_score'] > 0.5 else 'Fair' if silhouette_results['silhouette_score'] > 0.25 else 'Poor'}
        
        Cluster-specific Silhouette Scores:
        """
        
        for cluster_id, score in silhouette_results['cluster_silhouettes'].items():
            report += f"  Cluster {cluster_id}: {score:.3f}\n"
        
        report += f"""
        2. GAP STATISTIC ANALYSIS
        -------------------------
        Optimal number of clusters: {gap_results['optimal_k']}
        Gap statistic for 4 clusters: {gap_results['gaps'][3]:.3f}
        
        3. MEDIAN vs CLUSTERING COMPARISON
        ----------------------------------
        Agreement between methods: {comparison_results['agreement']:.1%}
        Median-based classification appears {'valid' if comparison_results['agreement'] > 0.7 else 'questionable'}
        
        4. BOOTSTRAP STABILITY
        ----------------------
        Mean clustering agreement: {bootstrap_results['mean_agreement']:.3f}
        95% Confidence Interval: ({bootstrap_results['ci_95_lower']:.3f}, {bootstrap_results['ci_95_upper']:.3f})
        
        5. RECOMMENDATIONS
        ------------------
        """
        
        if silhouette_results['silhouette_score'] > 0.5 and comparison_results['agreement'] > 0.7:
            report += "✅ Four-cluster solution is statistically validated and median-based classification is appropriate."
        elif gap_results['optimal_k'] == 4:
            report += "✅ Gap statistic confirms 4 clusters as optimal."
        else:
            report += "⚠️ Consider alternative clustering approaches or different number of clusters."
        
        return report
    
    def plot_validation_results(self, bootstrap_results=None):
        """
        Create comprehensive visualization of validation results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Silhouette plot
        silhouette_results = self.silhouette_analysis()
        y_lower = 10
        for i in range(4):
            cluster_silhouette_vals = silhouette_results['silhouette_samples'][
                silhouette_results['cluster_labels'] == i
            ]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            
            axes[0, 0].fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                                   alpha=0.7, label=f'Cluster {i}')
            y_lower = y_upper + 10
        
        axes[0, 0].axvline(x=silhouette_results['silhouette_score'], color="red", linestyle="--")
        axes[0, 0].set_xlabel("Silhouette Coefficient")
        axes[0, 0].set_ylabel("Cluster")
        axes[0, 0].set_title("Silhouette Analysis")
        axes[0, 0].legend()
        
        # 2. Gap statistic
        gap_results = self.gap_statistic()
        axes[0, 1].plot(gap_results['cluster_range'], gap_results['gaps'], 'bo-')
        axes[0, 1].errorbar(gap_results['cluster_range'], gap_results['gaps'], 
                           yerr=gap_results['sks'], fmt='none', capsize=5)
        axes[0, 1].axvline(x=gap_results['optimal_k'], color='red', linestyle='--', 
                          label=f'Optimal: {gap_results["optimal_k"]}')
        axes[0, 1].set_xlabel("Number of Clusters")
        axes[0, 1].set_ylabel("Gap Statistic")
        axes[0, 1].set_title("Gap Statistic Analysis")
        axes[0, 1].legend()
        
        # 3. Comparison scatter plot
        comparison_results = self.compare_median_vs_clustering()
        scatter = axes[1, 0].scatter(self.X[:, 0], self.X[:, 1], 
                                   c=comparison_results['cluster_labels'], 
                                   cmap='viridis', alpha=0.7)
        axes[1, 0].axhline(y=comparison_results['completeness_median'], color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axvline(x=comparison_results['coverage_median'], color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel("Coverage Ratio")
        axes[1, 0].set_ylabel("Completeness Ratio")
        axes[1, 0].set_title("Clustering Results vs Median Lines")
        plt.colorbar(scatter, ax=axes[1, 0])
        
        # 4. Bootstrap distribution (only if bootstrap_results provided)
        if bootstrap_results is not None and 'agreements' in bootstrap_results:
            axes[1, 1].hist(bootstrap_results['agreements'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=bootstrap_results['mean_agreement'], color='red', linestyle='--', 
                              label=f'Mean: {bootstrap_results["mean_agreement"]:.3f}')
            axes[1, 1].set_xlabel("Clustering Agreement")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Bootstrap Clustering Stability")
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Bootstrap analysis\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Bootstrap Clustering Stability")
        
        plt.tight_layout()
        return fig


def validate_archival_bias_clusters(results_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive validation of four-cluster solution for archival bias analysis.
    
    Args:
        results_df: DataFrame with coverage_ratio and completeness_ratio columns
        
    Returns:
        Dictionary with all validation results
    """
    validator = ClusterValidator(results_df)
    
    # Perform all validations
    silhouette_results = validator.silhouette_analysis()
    gap_results = validator.gap_statistic()
    comparison_results = validator.compare_median_vs_clustering()
    bootstrap_results = validator.bootstrap_validation()
    
    # Generate report
    report = validator.create_validation_report()
    
    # Create plots
    fig = validator.plot_validation_results(bootstrap_results)
    
    return {
        'silhouette_results': silhouette_results,
        'gap_results': gap_results,
        'comparison_results': comparison_results,
        'bootstrap_results': bootstrap_results,
        'report': report,
        'figure': fig,
        'validator': validator
    }


if __name__ == "__main__":
    print("Use this function in your notebook:")
    print("validation_results = validate_archival_bias_clusters(results_df)")
    print("print(validation_results['report'])")
    print("validation_results['figure'].show()") 