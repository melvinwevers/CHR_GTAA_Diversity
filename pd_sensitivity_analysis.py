import numpy as np
import pandas as pd
import networkx as nx
from src.unseen_pd import UnseenPDEstimator
from src.archival_bias_detection import ArchivalBiasDetector
from pathlib import Path
from scipy.stats import spearmanr

# --- CONFIGURATION ---
gtaa_csv_path = Path("data/external/gtaa_ontology.csv")
parquet_path = Path("data/processed/photos_archive.parquet")
collection_col = 'fotocollectie'

# --- LOAD DATA AND GRAPH ---
print("Loading data and building graph...")
detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=1000)
detector.build_graph(apply_deduplication=False)
df = detector.load_and_filter_data(parquet_path)

# --- BRANCH LENGTH ASSIGNMENT SCHEMES ---
def assign_branch_lengths(G, scheme='uniform'):
    G = G.copy()
    if scheme == 'uniform':
        for u, v in G.edges():
            G.edges[u, v]['length'] = 1.0
    elif scheme == 'random':
        for u, v in G.edges():
            G.edges[u, v]['length'] = np.random.uniform(0.5, 2.0)
    elif scheme == 'depth':
        root = [n for n, d in G.in_degree() if d == 0][0]
        lengths = nx.single_source_shortest_path_length(G, root)
        for u, v in G.edges():
            depth = lengths.get(u, 1)
            G.edges[u, v]['length'] = 1.0 / max(depth, 1)
    else:
        raise ValueError("Unknown scheme")
    return G

def pd_ranking_for_scheme(G, df, group_col):
    estimator = UnseenPDEstimator(G)
    results = []
    for name, group in df.groupby(group_col):
        # Reset all node counts to 0
        for node in G.nodes():
            G.nodes[node]['count'] = 0
        
        # Count subject occurrences in this collection
        for subjects_list in group['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                for subj in subjects_list:
                    if subj in G.nodes():
                        G.nodes[subj]['count'] = G.nodes[subj].get('count', 0) + 1
        
        res = estimator.estimate_undetected_pd(group)
        results.append({'collection': name, 'PD_hat': res['PD_hat']})
    return pd.DataFrame(results).sort_values('PD_hat', ascending=False).reset_index(drop=True)

# --- RUN SENSITIVITY ANALYSIS ---
schemes = ['uniform', 'random', 'depth']
all_rankings = {}

print("Calculating PD rankings for each branch length scheme...")
for scheme in schemes:
    print(f"  Scheme: {scheme}")
    G_scheme = assign_branch_lengths(detector.graph, scheme=scheme)
    ranking = pd_ranking_for_scheme(G_scheme, df, collection_col)
    all_rankings[scheme] = ranking

# --- COMPARE RANKINGS ---
print("\nSpearman correlations between collection rankings:")
for i, s1 in enumerate(schemes):
    for s2 in schemes[i+1:]:
        merged = pd.merge(all_rankings[s1], all_rankings[s2], on='collection', suffixes=(f'_{s1}', f'_{s2}'))
        corr, _ = spearmanr(merged[f'PD_hat_{s1}'], merged[f'PD_hat_{s2}'])
        print(f"  {s1} vs {s2}: {corr:.3f}")

# --- DEBUG: Print unique PD_hat values for each scheme ---
for scheme, ranking in all_rankings.items():
    print(f"\n{scheme} PD_hat values (unique count: {ranking['PD_hat'].nunique()}):")
    print(ranking)

print("\nSensitivity analysis complete. See above for ranking stability across branch length schemes.") 