#!/usr/bin/env python3
"""
generate_latex_collection_table.py - Generate simplified LaTeX table for article
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate_latex_table():
    """Generate simplified LaTeX table for the article."""
    
    # Load the data
    parquet_path = Path("data/processed/photos_archive.parquet")
    df = pd.read_parquet(parquet_path)
    
    # Get collection statistics
    collections = df['fotocollectie'].value_counts()
    large_collections = collections[collections >= 1000]
    
    # Create table data
    table_data = []
    
    for collection_name, collection_size in large_collections.items():
        if pd.isna(collection_name) or not str(collection_name).strip():
            continue
            
        collection_df = df[df['fotocollectie'] == collection_name]
        
        # Count items with subject metadata
        items_with_subjects = collection_df['has_subject'].sum()
        
        # Time period
        years = collection_df['year'].dropna()
        if len(years) > 0:
            start_year = int(years.min())
            end_year = int(years.max())
            time_period = f"{start_year}-{end_year}"
        else:
            time_period = "Unknown"
        
        # Institution type
        institution_type = classify_institution_type(collection_name)
        
        table_data.append({
            'name': collection_name,
            'institution': institution_type,
            'period': time_period,
            'total_items': collection_size,
            'items_with_subjects': items_with_subjects
        })
    
    # Sort by collection size
    table_data.sort(key=lambda x: x['total_items'], reverse=True)
    
    # Generate LaTeX table
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\begin{tabular}{lllrr}")
    print("\\toprule")
    print("Collection & Institution & Period & Total Items & Items with Subjects \\\\")
    print("\\midrule")
    
    for row in table_data:
        # Use full collection name without truncation
        name = row['name']
        
        # Format numbers
        total_items = f"{row['total_items']:,}"
        items_with_subjects = f"{row['items_with_subjects']:,}"
        
        print(f"{name} & {row['institution']} & {row['period']} & {total_items} & {items_with_subjects} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Overview of analyzed archival collections showing institution types, time periods, total items, and items with subject metadata.}")
    print("\\label{tab:collection_overview}")
    print("\\end{table}")
    
    # Generate summary statistics
    print("\n% Summary statistics for text:")
    print(f"% Total collections: {len(table_data)}")
    print(f"% Total items: {sum(row['total_items'] for row in table_data):,}")
    print(f"% Total items with subjects: {sum(row['items_with_subjects'] for row in table_data):,}")
    print(f"% Average collection size: {np.mean([row['total_items'] for row in table_data]):,.0f}")
    print(f"% Median collection size: {np.median([row['total_items'] for row in table_data]):,.0f}")
    
    # Institution breakdown
    inst_counts = {}
    for row in table_data:
        inst = row['institution']
        inst_counts[inst] = inst_counts.get(inst, 0) + 1
    
    print("\n% Institution type breakdown:")
    for inst, count in sorted(inst_counts.items()):
        print(f"% {inst}: {count} collections")


def classify_institution_type(collection_name):
    """Classify institution type."""
    name_lower = collection_name.lower()
    
    if 'anefo' in name_lower:
        return 'News Agency'
    elif 'rvd' in name_lower or 'rijksvoorlichtingsdienst' in name_lower:
        return 'Government'
    elif 'elsevier' in name_lower:
        return 'Media Company'
    elif 'spaarnestad' in name_lower:
        return 'Media Company'
    elif 'knvb' in name_lower:
        return 'Sports Archive'
    elif 'arbeidsinspectie' in name_lower:
        return 'Government'
    elif 'dienst voor legercontacten' in name_lower:
        return 'Military'
    elif 'eerste wereldoorlog' in name_lower:
        return 'Historical Archive'
    elif 'deli maatschappij' in name_lower:
        return 'Private Company'
    elif 'visser' in name_lower:
        return 'Individual Photographer'
    elif 'van de poll' in name_lower:
        return 'Individual Photographer'
    elif 'nederlandse heidemaatschappij' in name_lower:
        return 'Private Company'
    elif 'kantoor voor voorlichting' in name_lower:
        return 'Government'
    else:
        return 'Other'


if __name__ == "__main__":
    generate_latex_table() 