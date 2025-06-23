# GTAA Phylogenetic Diversity Analysis

This project analyzes archival bias patterns in Dutch National Archives collections using **Faith's Phylogenetic Diversity (PD)** metrics applied to the **GTAA (General Thesaurus for Audiovisual Archives)** vocabulary.

## Overview

The project quantifies institutional bias by calculating how well different archival subcollections cover the conceptual space defined by the GTAA vocabulary. It uses phylogenetic diversity metrics borrowed from biodiversity research to measure conceptual diversity and identify gaps in archival preservation.

## Key Concepts

### Methodology
Based on **Faith's Phylogenetic Diversity** and **Chao1 unseen diversity estimation**, the analysis calculates three key ratios:

1. **Coverage Ratio**: `collection_pd / gtaa_total_pd`
   - What fraction of total possible conceptual space does this collection cover?

2. **Completeness Ratio**: `collection_pd / (collection_pd + unseen_pd)`
   - Within the conceptual territory this collection covers, how thoroughly is it documented?
   - High completeness = few missing subjects that should be there
   - Low completeness = many related concepts are missing → incomplete preservation

3. **Efficiency Ratio**: `coverage_ratio / log(collection_size)`
   - How efficient is the collection at covering conceptual space?
   - High efficiency = small specialized collections with conceptual breadth
   - Low efficiency = massive collections with repetitive content

### Interpretation Guidelines

- **Lower-Right Quadrant**: High Coverage + Low Completeness
  - Collection touches many domains but is under-sampled in each
  - Good candidate for targeted expansion

- **Lower-Left Quadrant**: Low Coverage + Low Completeness  
  - Collection has both institutional blind spots AND incomplete sampling
  - Requires comprehensive review

- **Upper-Left Quadrant**: Low Coverage + High Completeness
  - Collection thoroughly documents its narrow domain
  - Low coverage reflects institutional focus, not sampling bias

- **Upper-Right Quadrant**: High Coverage + High Completeness
  - Excellent coverage and depth across conceptual space

## Project Structure

```
GTAA_PD/
├── README.md                           # This file
├── archival_bias_detection.ipynb      # Main analysis notebook
├── data/
│   ├── external/
│   │   └── gtaa_melvin.csv            # GTAA vocabulary data
│   ├── raw/                           # Raw archival data in JSON
│   └── processed/                     # Processed data in Parquet format
├── results/                           # Analysis outputs
└── src/                               # Source code modules
    ├── archival_bias_detection.py     # Main analysis class (includes ontology analysis)
    ├── data_processing.py             # Data preprocessing utilities
    ├── faith_pd.py                    # Faith's PD implementation
    ├── graph_builder.py               # GTAA graph construction
    ├── test_unseenpd.py               # Testing utilities
    └── unseen_pd.py                   # Unseen diversity estimation
```