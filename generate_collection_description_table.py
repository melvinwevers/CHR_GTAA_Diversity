#!/usr/bin/env python3
"""
Generate a LaTeX table describing the collections with their focus, period, and total items.
"""

import pandas as pd
import numpy as np
from collections import Counter
from ast import literal_eval

def generate_collection_description_table():
    """
    Generate a LaTeX table describing each collection with focus, period, and total items.
    """
    try:
        # Load the results data
        results_df = pd.read_csv('results/archival_bias_detailed.csv')
        
        # Load the full dataset to get temporal and subject information
        full_df = pd.read_parquet('data/processed/photos_archive.parquet')
        
        print("GENERATING COLLECTION DESCRIPTION LATEX TABLE")
        print("=" * 60)
        
        # Create collection descriptions
        collection_descriptions = []
        
        for _, row in results_df.iterrows():
            collection_name = row['subcollection']
            collection_size = row['collection_size']
            
            # Filter data for this collection
            collection_data = full_df[full_df['fotocollectie'] == collection_name]
            
            # Calculate temporal focus
            years_series = collection_data['year']
            years = years_series.dropna()
            if len(years) > 0:
                min_year = int(years.min())
                max_year = int(years.max())
                if min_year == max_year:
                    period = str(min_year)
                else:
                    period = f"{min_year}-{max_year}"
            else:
                period = "Unknown"
            
            # Calculate conceptual focus (most common subjects)
            all_subjects = []
            for subjects_list in collection_data['subjects_list'].dropna():
                try:
                    if isinstance(subjects_list, str):
                        subjects = literal_eval(subjects_list)
                    else:
                        subjects = subjects_list
                    if isinstance(subjects, list):
                        all_subjects.extend(subjects)
                except:
                    continue
            
            # Get top 3 most common subjects
            if all_subjects:
                subject_counts = Counter(all_subjects)
                top_subjects = [subject for subject, count in subject_counts.most_common(3)]
                focus = ", ".join(top_subjects[:3])
            else:
                focus = "No subjects"
            
            # Clean collection name for display
            display_name = collection_name.replace('Fotocollectie ', '')
            
            collection_descriptions.append({
                'name': display_name,
                'focus': focus,
                'period': period,
                'total_items': f"{collection_size:,}",
                'coverage_ratio': row['coverage_ratio'],
                'completeness_ratio': row['completeness_ratio']
            })
        
        # Sort by collection size (descending)
        collection_descriptions.sort(key=lambda x: int(x['total_items'].replace(',', '')), reverse=True)
        
        # Generate LaTeX table
        print("\n\\begin{table}[h]")
        print("  \\centering")
        print("  \\caption{Collection Descriptions and Performance Metrics}")
        print("  \\begin{tabular}{|l|p{3cm}|c|c|c|c|}")
        print("    \\hline")
        print("    \\textbf{Collection} & \\textbf{Conceptual Focus} & \\textbf{Period} & \\textbf{Total Items} & \\textbf{Coverage} & \\textbf{Completeness} \\\\")
        print("    \\hline")
        
        for desc in collection_descriptions:
            # Truncate long collection names
            name = desc['name']
            if len(name) > 25:
                name = name[:22] + "..."
            
            # Truncate long focus descriptions
            focus = desc['focus']
            if len(focus) > 40:
                focus = focus[:37] + "..."
            
            print(f"    {name} & {focus} & {desc['period']} & {desc['total_items']} & {desc['coverage_ratio']:.3f} & {desc['completeness_ratio']:.3f} \\\\")
        
        print("    \\hline")
        print("  \\end{tabular}")
        print("  \\label{tab:collection_descriptions}")
        print("\\end{table}")
        
        # Also generate a summary table with key statistics
        print("\n\\begin{table}[h]")
        print("  \\centering")
        print("  \\caption{Collection Summary Statistics}")
        print("  \\begin{tabular}{|l|c|c|c|}")
        print("    \\hline")
        print("    \\textbf{Statistic} & \\textbf{Value} & \\textbf{Range} & \\textbf{Description} \\\\")
        print("    \\hline")
        
        total_collections = len(collection_descriptions)
        total_items = sum(int(desc['total_items'].replace(',', '')) for desc in collection_descriptions)
        coverage_ratios = [desc['coverage_ratio'] for desc in collection_descriptions]
        completeness_ratios = [desc['completeness_ratio'] for desc in collection_descriptions]
        
        print(f"    Collections analyzed & {total_collections} & - & Total number of collections \\\\")
        print(f"    Total items & {total_items:,} & - & Combined size of all collections \\\\")
        print(f"    Coverage ratio & {np.mean(coverage_ratios):.3f} & {min(coverage_ratios):.3f}-{max(coverage_ratios):.3f} & Mean conceptual coverage \\\\")
        print(f"    Completeness ratio & {np.mean(completeness_ratios):.3f} & {min(completeness_ratios):.3f}-{max(completeness_ratios):.3f} & Mean sampling completeness \\\\")
        
        print("    \\hline")
        print("  \\end{tabular}")
        print("  \\label{tab:collection_summary}")
        print("\\end{table}")
        
        print(f"\n✅ Generated LaTeX tables for {total_collections} collections")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find required files: {e}")
        print("Please make sure the following files exist:")
        print("  - results/archival_bias_detailed.csv")
        print("  - data/processed/photos_archive.parquet")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    generate_collection_description_table() 