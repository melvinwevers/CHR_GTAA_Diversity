#!/usr/bin/env python3
"""
faith_pd.py - Faith's Phylogenetic Diversity calculations and caching

This module handles the core Faith's PD calculations and caching for the Dutch National Archives diversity analysis.
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Any


class FaithPDCalculator:
    """Calculator for Faith's Phylogenetic Diversity metrics with caching support."""
    
    def __init__(self, graph: nx.DiGraph, cache_file: Optional[Path] = None):
        """
        Initialize the Faith's PD calculator.
        
        Args:
            graph: NetworkX directed graph representing the vocabulary tree
            cache_file: Optional path to cache file for previously computed values
        """
        self.graph = graph
        self.cache_file = cache_file
        self._cache: Dict[str, float] = {}
        self._collection_cache: Dict[str, Dict[str, Any]] = {}  # Cache for collection-level results
        self._creator_cache: Dict[str, Dict[str, Any]] = {}  # Cache for creator-level results
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load previously computed Faith's PD values from cache."""
        if not self.cache_file or not self.cache_file.exists():
            return
        
        try:
            if self.cache_file.suffix == '.parquet':
                cache_df = pd.read_parquet(self.cache_file)
                self._cache = dict(zip(cache_df['subjects_key'], cache_df['faith_pd']))
                
                # Load collection and creator caches if available
                if 'collection_key' in cache_df.columns:
                    for _, row in cache_df.iterrows():
                        collection_key = row.get('collection_key')
                        if pd.notna(collection_key) and isinstance(collection_key, str):
                            self._collection_cache[str(collection_key)] = {
                                'raw_pd': row.get('collection_raw_pd'),
                                'coverage': row.get('collection_coverage', 1.0)
                            }
                
                if 'creator_key' in cache_df.columns:
                    for _, row in cache_df.iterrows():
                        creator_key = row.get('creator_key')
                        if pd.notna(creator_key) and isinstance(creator_key, str):
                            self._creator_cache[str(creator_key)] = {
                                'raw_pd': row.get('creator_raw_pd'),
                                'coverage': row.get('creator_coverage', 1.0)
                            }
            else:  # .pkl format
                cache_df = pd.read_pickle(self.cache_file)
                self._cache = dict(zip(cache_df['subjects_key'], cache_df['faith_pd']))
            
            print(f"📦 Loaded {len(self._cache):,} cached Faith's PD values from {self.cache_file}")
            if self._collection_cache:
                print(f"📦 Loaded {len(self._collection_cache):,} cached collection-level values")
            if self._creator_cache:
                print(f"📦 Loaded {len(self._creator_cache):,} cached creator-level values")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load cache from {self.cache_file}: {e}")
            self._cache = {}
            self._collection_cache = {}
            self._creator_cache = {}
    
    def _save_cache(self) -> None:
        """Save computed Faith's PD values to cache."""
        if not self.cache_file or not self._cache:
            return
            
        try:
            # Ensure parent directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = []
            for key, value in self._cache.items():
                row = {'subjects_key': key, 'faith_pd': value}
                cache_data.append(row)
            
            # Add collection and creator cache data
            for key, data in self._collection_cache.items():
                row = {'collection_key': key}
                raw_pd = data.get('raw_pd')
                coverage = data.get('coverage')
                if raw_pd is not None:
                    row['collection_raw_pd'] = raw_pd
                if coverage is not None:
                    row['collection_coverage'] = coverage
                cache_data.append(row)
            
            for key, data in self._creator_cache.items():
                row = {'creator_key': key}
                raw_pd = data.get('raw_pd')
                coverage = data.get('coverage')
                if raw_pd is not None:
                    row['creator_raw_pd'] = raw_pd
                if coverage is not None:
                    row['creator_coverage'] = coverage
                cache_data.append(row)
            
            cache_df = pd.DataFrame(cache_data)
            
            if self.cache_file.suffix == '.parquet':
                cache_df.to_parquet(self.cache_file, index=False)
            else:  # .pkl format
                cache_df.to_pickle(self.cache_file)
                
            print(f"💾 Saved {len(self._cache):,} Faith's PD values to cache: {self.cache_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save cache to {self.cache_file}: {e}")
    
    def get_collection_pd(self, collection_subjects: List[str], collection_id: str) -> Dict[str, Any]:
        """
        Get Faith's PD for a collection with caching.
        
        Args:
            collection_subjects: List of all subjects in the collection
            collection_id: Unique identifier for the collection
            
        Returns:
            Dictionary with raw PD values
        """
        # Create cache key
        cache_key = f"{collection_id}_{len(collection_subjects)}"
        
        # Check cache first
        if cache_key in self._collection_cache:
            return self._collection_cache[cache_key]
        
        # Calculate raw collection PD
        raw_pd = self.calculate_faith_pd(collection_subjects)
        result = {
            'raw_pd': raw_pd,
            'coverage': 1.0
        }
        
        # Cache the result
        self._collection_cache[cache_key] = result
        self._save_cache()
        
        return result
    
    def get_creator_pd(self, creator_subjects: List[str], creator_id: str) -> Dict[str, Any]:
        """
        Get Faith's PD for a creator with caching.
        
        Args:
            creator_subjects: List of all subjects by this creator
            creator_id: Unique identifier for the creator
            
        Returns:
            Dictionary with raw PD values
        """
        # Create cache key
        cache_key = f"{creator_id}_{len(creator_subjects)}"
        
        # Check cache first
        if cache_key in self._creator_cache:
            return self._creator_cache[cache_key]
        
        # Calculate raw creator PD
        raw_pd = self.calculate_faith_pd(creator_subjects)
        result = {
            'raw_pd': raw_pd,
            'coverage': 1.0
        }
        
        # Cache the result
        self._creator_cache[cache_key] = result
        self._save_cache()
        
        return result

    def calculate_faith_pd(self, subjects: List[str]) -> float:
        """
        Calculate Faith's Phylogenetic Diversity for a set of subjects.
        
        Faith's PD is the sum of branch lengths in the minimal subtree that 
        connects all subjects to the root, following the scikit-bio approach.
        
        The correct formula is:
        PD = sum(branch_lengths * (counts_by_node > 0))
        where counts_by_node indicates how many observed taxa descend from each node.
        
        Args:
            subjects: List of subject labels
            
        Returns:
            Faith's PD value (sum of branch lengths)
        """
        if not subjects:
            return 0.0
        
        # Faith's PD calculation following scikit-bio approach
        # Filter subjects that exist in the graph
        valid_subjects = [s for s in subjects if s in self.graph.nodes]
        if not valid_subjects:
            return 0.0
        
        # For each node, count how many observed taxa descend from it
        counts_by_node: Dict[str, int] = {}
        for node in self.graph.nodes():
            descendants = nx.descendants(self.graph, node)
            # Count how many observed subjects are descendants of this node
            # Include the node itself if it's one of the observed subjects
            count = sum(1 for subject in valid_subjects 
                       if subject in descendants or subject == node)
            counts_by_node[node] = count
        
        # Calculate sum branch lengths for nodes with descendants
        # sum(branch_lengths * (counts_by_node > 0))
        total_pd = 0.0
        for node in self.graph.nodes():
            if node != "_DUMMY_ROOT_":  # Skip the dummy root
                if counts_by_node[node] > 0:  # If this node has descendants
                    # Find the edge leading to this node and include its length
                    for predecessor in self.graph.predecessors(node):
                        edge_length = self.graph.edges[predecessor, node].get('length', 1.0)
                        total_pd += edge_length
                        break  # Only count the edge once per node
        
        return total_pd

    def compute_faith_pd_with_cache(self, df: pd.DataFrame, subjects_col: str = 'subjects_list') -> pd.DataFrame:
        """
        Compute Faith's PD for all rows in DataFrame with caching support.
        
        Args:
            df: DataFrame with subjects lists
            subjects_col: Column name containing subject lists
            
        Returns:
            DataFrame with added 'faith_pd' column
        """
        print(f"Computing Faith's PD for {len(df):,} images...")
        
        # Create unique key for each subjects list for caching
        df_copy = df.copy()
        df_copy['subjects_key'] = df_copy[subjects_col].apply(
            lambda x: '|'.join(sorted(x)) if isinstance(x, list) and x else ''
        )
        
        # Identify rows that need computation
        cache_hits = 0
        cache_misses = 0
        
        faith_pd_values: List[Optional[float]] = []
        unique_keys_to_compute = []
        
        for subjects_key in tqdm(df_copy['subjects_key'], desc="Checking cache"):
            if subjects_key in self._cache:
                faith_pd_values.append(self._cache[subjects_key])
                cache_hits += 1
            else:
                faith_pd_values.append(None)  # Will be computed later
                if subjects_key not in unique_keys_to_compute:
                    unique_keys_to_compute.append(subjects_key)
                cache_misses += 1
        
        if not self._cache:
            print("No existing Faith's PD cache found, starting fresh")
        else:
            print(f"Cache hits: {cache_hits:,}, misses: {cache_misses:,}")
        
        # Compute missing values
        if unique_keys_to_compute:
            print(f"Computing {len(unique_keys_to_compute):,} unique Faith's PD values...")
            
            for subjects_key in tqdm(unique_keys_to_compute, desc="Computing Faith's PD"):
                if subjects_key == '':
                    faith_pd_value = 0.0
                else:
                    subjects = subjects_key.split('|')
                    faith_pd_value = self.calculate_faith_pd(subjects)
                
                self._cache[subjects_key] = faith_pd_value
        
        # Fill in the computed values
        for i, subjects_key in enumerate(df_copy['subjects_key']):
            if faith_pd_values[i] is None:
                faith_pd_values[i] = self._cache[subjects_key]
        
        df_copy['faith_pd'] = faith_pd_values
        
        # Save updated cache
        if unique_keys_to_compute:
            self._save_cache()
        
        print(f"Faith's PD computation complete")
        return df_copy.drop(columns=['subjects_key'])

