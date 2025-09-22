import pandas as pd
import ast

# Load the data
df = pd.read_parquet('data/processed/photos_archive.parquet')

# Get collections with >= 1000 photos
collections = df['fotocollectie'].value_counts()
large_collections = collections[collections >= 1000]

print(f"Number of large collections: {len(large_collections)}")
print(f"Total photos in large collections: {large_collections.sum():,}")

# Filter to only large collections
df_large = df[df['fotocollectie'].isin(large_collections.index)]

# Extract all unique terms
all_terms = set()
for subjects_str in df_large['subjects_list'].dropna():
    try:
        # Parse the string representation of the list
        subjects_list = ast.literal_eval(subjects_str)
        if isinstance(subjects_list, list):
            all_terms.update(subjects_list)
    except (ValueError, SyntaxError):
        # Skip malformed entries
        continue

print(f"Total unique terms across all 16 collections: {len(all_terms):,}")

# Also show breakdown by collection
print("\nUnique terms per collection:")
for collection in large_collections.index:
    collection_df = df[df['fotocollectie'] == collection]
    collection_terms = set()
    for subjects_str in collection_df['subjects_list'].dropna():
        try:
            subjects_list = ast.literal_eval(subjects_str)
            if isinstance(subjects_list, list):
                collection_terms.update(subjects_list)
        except (ValueError, SyntaxError):
            continue
    print(f"{collection}: {len(collection_terms):,} unique terms") 