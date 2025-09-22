#!/usr/bin/env python3
"""
generate_collection_table.py - Generate comprehensive collection overview table
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def generate_collection_table():
    """Generate a comprehensive table of all collections with metadata."""
    
    # Load the processed data
    parquet_path = Path("data/processed/photos_archive.parquet")
    df = pd.read_parquet(parquet_path)
    
    # Load the bias analysis results to get PD metrics
    results_path = Path("results/archival_bias_detailed.csv")
    if results_path.exists():
        bias_results = pd.read_csv(results_path)
    else:
        # If no bias results, create basic structure
        bias_results = pd.DataFrame()
    
    print("Generating comprehensive collection table...")
    
    # Get collection statistics
    collections = df['fotocollectie'].value_counts()
    large_collections = collections[collections >= 1000]
    
    # Create collection table
    collection_data = []
    
    for collection_name, collection_size in large_collections.items():
        if pd.isna(collection_name) or not str(collection_name).strip():
            continue
            
        collection_df = df[df['fotocollectie'] == collection_name]
        
        # Extract subjects for unique term count
        all_subjects = []
        for subjects_list in collection_df['subjects_list'].dropna():
            if isinstance(subjects_list, list):
                all_subjects.extend(subjects_list)
        unique_terms = len(set(all_subjects))
        
        # Time period analysis
        years = collection_df['year'].dropna()
        if len(years) > 0:
            start_year = int(years.min())
            end_year = int(years.max())
            time_period = f"{start_year}-{end_year}"
            decade_range = f"{start_year//10*10}s-{end_year//10*10}s"
        else:
            time_period = "Unknown"
            decade_range = "Unknown"
        
        # Institution type classification
        institution_type = classify_institution_type(collection_name)
        
        # Get bias metrics if available
        bias_row = bias_results[bias_results['subcollection'] == collection_name]
        if len(bias_row) > 0:
            coverage_ratio = bias_row.iloc[0]['coverage_ratio']
            completeness_ratio = bias_row.iloc[0]['completeness_ratio']
            collection_pd = bias_row.iloc[0]['collection_pd']
        else:
            coverage_ratio = np.nan
            completeness_ratio = np.nan
            collection_pd = np.nan
        
        collection_data.append({
            'Collection Name': collection_name,
            'Institution Type': institution_type,
            'Time Period': time_period,
            'Decade Range': decade_range,
            'Number of Photos': collection_size,
            'Number of Unique Terms': unique_terms,
            'Coverage Ratio': coverage_ratio,
            'Completeness Ratio': completeness_ratio,
            'Collection PD': collection_pd
        })
    
    # Create DataFrame and sort by collection size
    collection_table = pd.DataFrame(collection_data)
    collection_table = collection_table.sort_values('Number of Photos', ascending=False)
    
    # Save to CSV
    output_path = Path("results/collection_overview_table.csv")
    collection_table.to_csv(output_path, index=False)
    
    # Print summary statistics
    print(f"\nCollection Overview Summary:")
    print(f"Total collections analyzed: {len(collection_table)}")
    print(f"Total photos: {collection_table['Number of Photos'].sum():,}")
    print(f"Average collection size: {collection_table['Number of Photos'].mean():,.0f}")
    print(f"Median collection size: {collection_table['Number of Photos'].median():,.0f}")
    
    # Institution type breakdown
    print(f"\nInstitution Type Breakdown:")
    inst_counts = collection_table['Institution Type'].value_counts()
    for inst_type, count in inst_counts.items():
        print(f"  {inst_type}: {count} collections")
    
    # Time period analysis
    print(f"\nTime Period Coverage:")
    time_counts = collection_table['Decade Range'].value_counts()
    for time_range, count in time_counts.items():
        print(f"  {time_range}: {count} collections")
    
    # Print the table
    print(f"\nCollection Overview Table:")
    print("=" * 120)
    print(collection_table.to_string(index=False, float_format='%.3f'))
    
    # Generate LaTeX table
    generate_latex_table(collection_table)
    
    return collection_table


def classify_institution_type(collection_name):
    """Classify the institution type based on collection name."""
    name_lower = collection_name.lower()
    
    if 'anefo' in name_lower:
        return 'Press Agency'
    elif 'rvd' in name_lower or 'rijksvoorlichtingsdienst' in name_lower:
        return 'Government Press'
    elif 'elsevier' in name_lower:
        return 'Media Company'
    elif 'spaarnestad' in name_lower:
        return 'Media Company'
    elif 'knvb' in name_lower:
        return 'Sports Organization'
    elif 'arbeidsinspectie' in name_lower:
        return 'Government Agency'
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
        return 'Government Agency'
    else:
        return 'Other'


def generate_latex_table(collection_table):
    """Generate LaTeX table for the paper."""
    
    print(f"\nLaTeX Table:")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\begin{tabular}{llllrrrr}")
    print("\\toprule")
    print("Collection & Institution & Time Period & Photos & Unique Terms & Coverage & Completeness & PD \\\\")
    print("\\midrule")
    
    for _, row in collection_table.iterrows():
        # Truncate long collection names
        name = str(row['Collection Name'])
        if len(name) > 30:
            name = name[:27] + "..."
        
        # Format numbers
        photos = f"{row['Number of Photos']:,}"
        terms = f"{row['Number of Unique Terms']:,}"
        
        # Format ratios
        coverage = f"{row['Coverage Ratio']:.3f}" if not pd.isna(row['Coverage Ratio']) else "N/A"
        completeness = f"{row['Completeness Ratio']:.3f}" if not pd.isna(row['Completeness Ratio']) else "N/A"
        pd_val = f"{row['Collection PD']:.0f}" if not pd.isna(row['Collection PD']) else "N/A"
        
        print(f"{name} & {row['Institution Type']} & {row['Time Period']} & {photos} & {terms} & {coverage} & {completeness} & {pd_val} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Overview of analyzed archival collections with institution types, time periods, and key metrics.}")
    print("\\label{tab:collection_overview}")
    print("\\end{table}")


def generate_summary_statistics(collection_table):
    """Generate summary statistics for the paper."""
    
    print(f"\nSummary Statistics for Paper:")
    print("=" * 50)
    
    # Collection size statistics
    print(f"Collection Size Statistics:")
    print(f"  Range: {collection_table['Number of Photos'].min():,} to {collection_table['Number of Photos'].max():,} photos")
    print(f"  Mean: {collection_table['Number of Photos'].mean():,.0f} photos")
    print(f"  Median: {collection_table['Number of Photos'].median():,.0f} photos")
    print(f"  Standard Deviation: {collection_table['Number of Photos'].std():,.0f}")
    
    # Unique terms statistics
    print(f"\nUnique Terms Statistics:")
    print(f"  Range: {collection_table['Number of Unique Terms'].min():,} to {collection_table['Number of Unique Terms'].max():,} terms")
    print(f"  Mean: {collection_table['Number of Unique Terms'].mean():,.0f} terms")
    print(f"  Median: {collection_table['Number of Unique Terms'].median():,.0f} terms")
    
    # Institution type summary
    print(f"\nInstitution Type Summary:")
    inst_summary = collection_table.groupby('Institution Type').agg({
        'Number of Photos': ['count', 'sum', 'mean'],
        'Number of Unique Terms': ['mean', 'sum']
    }).round(0)
    print(inst_summary)
    
    # Time period summary
    print(f"\nTime Period Summary:")
    time_summary = collection_table.groupby('Decade Range').agg({
        'Number of Photos': ['count', 'sum'],
        'Number of Unique Terms': ['mean', 'sum']
    }).round(0)
    print(time_summary)


if __name__ == "__main__":
    collection_table = generate_collection_table()
    generate_summary_statistics(collection_table)
    
    print(f"\nResults saved to: results/collection_overview_table.csv") 