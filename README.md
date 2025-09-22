# GTAA Phylogenetic Diversity Analysis

This repository accompanies a paper that investigates archival bias patterns in collections from the Dutch National Archives using Faith's Phylogenetic Diversity (PD) metrics applied to the GTAA (General Thesaurus for Audiovisual Archives) vocabulary. The project quantifies institutional bias by assessing how well various archival subcollections cover the conceptual space defined by the GTAA vocabulary. By borrowing phylogenetic diversity metrics from biodiversity research, the analysis measures conceptual diversity and identifies gaps in archival preservation.

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


## Project Structure

```
GTAA_PD/
├── README.md                          # This file
├── archival_bias_detection.ipynb      # Main analysis notebook
├── data/
│   ├── external/
│   │   └── gtaa_ontology.csv          # GTAA vocabulary data
│   └── processed/                     # Processed data in Parquet format
│   │   └── photos_archives.parquet    # NA photo metadata in parquet format
├── results/                           # Analysis outputs
└── src/                               # Source code modules
    ├── archival_bias_detection.py     # Main analysis class (includes ontology analysis)
    ├── data_processing.py             # Data preprocessing utilities
    ├── faith_pd.py                    # Faith's PD implementation
    ├── graph_builder.py               # GTAA graph construction
    ├── test_unseenpd.py               # Testing utilities
    └── unseen_pd.py                   # Unseen diversity estimation
```
## Installation using UV
1. `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. navigate to project directory
3. add data to `data`. You can skip the creation of the parquet file with metadata from the raw JSONS and just add the `photo_archive.parquet` file to data/processed and the `gtaa_ontology.csv` to `data/external`
3. `uv sync`
4. `uv run jupyter notebook archival_bias_detection.ipynb`

