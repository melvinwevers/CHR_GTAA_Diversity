# How the Hierarchy Bias Analysis Script Works

## Overview

The script performs a comprehensive analysis of bias patterns in GTAA vocabulary usage across archival collections. It works in three main phases, each addressing one of the key areas you mentioned.

## Phase 1: Setup and Data Loading

### Step 1: Initialize Components
```python
# Build GTAA vocabulary graph
detector = ArchivalBiasDetector(gtaa_csv_path, min_collection_size=1000)
detector.build_graph(apply_deduplication=False)

# Load archival data
df = detector.load_and_filter_data(parquet_path)

# Initialize bias analyzer
analyzer = HierarchyBiasAnalyzer(detector.graph, gtaa_csv_path)
```

**What this does:**
- Creates a directed graph from the GTAA vocabulary CSV
- Loads the photo archive data (991,372 records)
- Sets up the hierarchy analyzer with the graph structure

### Step 2: Analyze Hierarchy Structure
```python
def _analyze_hierarchy_structure(self):
    # Get root-level branches (direct children of dummy root)
    self.root_branches = list(self.graph.successors(self.root))
    
    # Calculate depth for each node
    self.node_depths = {}
    for node in self.graph.nodes():
        if node != self.root:
            depth = nx.shortest_path_length(self.graph, self.root, node)
            self.node_depths[node] = depth
```

**What this does:**
- Identifies 251 root branches (top-level categories)
- Calculates hierarchy depth for each of 3,498 terms
- Groups terms by depth level

## Phase 2: Analysis 1 - Branch Representation

### Step 1: Extract Terms from Collections
```python
# Analyze each collection
for collection in large_collections:
    collection_df = df[df[collection_column] == collection]
    collection_terms[collection] = set()
    
    # Extract all terms from this collection
    for subjects_list in collection_df['subjects_list'].dropna():
        if isinstance(subjects_list, list):
            collection_terms[collection].update(subjects_list)
```

**What this does:**
- Goes through each of the 17 large collections
- Extracts all GTAA terms used in each collection
- Creates a set of unique terms per collection

### Step 2: Map Terms to Root Branches
```python
# Map terms to their root branches
for term in collection_terms[collection]:
    if term in self.graph.nodes():
        root_branch = self._find_root_branch(term)
        if root_branch:
            branch_coverage[root_branch]['collections'].add(collection)
            branch_coverage[root_branch]['total_occurrences'] += 1
            branch_coverage[root_branch]['unique_terms'].add(term)
```

**What this does:**
- For each term, finds which root branch it belongs to
- Tracks which collections use each branch
- Counts total occurrences and unique terms per branch

### Step 3: Calculate Representation Ratios
```python
# Calculate representation ratios
collection_coverage_ratio = n_collections / len(large_collections)
term_coverage_ratio = n_terms / branch_size if branch_size > 0 else 0
```

**What this does:**
- **Collection coverage ratio**: What % of collections use this branch?
- **Term coverage ratio**: What % of terms in this branch are used?

**Example Results:**
- `infrastructuur`: 94.1% of collections (16/17)
- `levensloop`: 0.0% of collections (0/17)

## Phase 3: Analysis 2 - Conceptual Domain Coverage

### Step 1: Define Conceptual Domains
```python
conceptual_domains = {
    'Social Life & Culture': ['mensen', 'samenleving', 'cultuur', 'religie', 'feesten', 'sport'],
    'Politics & Government': ['politiek', 'regering', 'bestuur', 'verkiezingen', 'diplomatie'],
    'Economy & Industry': ['economie', 'industrie', 'handel', 'landbouw', 'transport'],
    # ... more domains
}
```

### Step 2: Map Branches to Domains
```python
def _map_branch_to_domain(self, branch: str, domains: Dict[str, List[str]]) -> str:
    branch_lower = branch.lower()
    
    for domain, keywords in domains.items():
        for keyword in keywords:
            if keyword in branch_lower:
                return domain
    
    return 'Other'  # Default for unmapped branches
```

**What this does:**
- Uses keyword matching to group GTAA branches into conceptual domains
- Branches like "politieke partijen" → "Politics & Government"
- Branches like "dieren" → "Other" (no keyword match)

### Step 3: Calculate Domain Statistics
```python
domain_stats.append({
    'domain': domain,
    'n_branches': n_branches,
    'total_terms': coverage['total_terms'],
    'n_collections': n_collections,
    'avg_collection_coverage': domain_branch_stats['collection_coverage_ratio'].mean(),
    'avg_term_coverage': domain_branch_stats['term_coverage_ratio'].mean(),
})
```

**What this does:**
- Aggregates branch-level statistics to domain level
- Calculates average coverage across all branches in each domain

## Phase 4: Analysis 3 - Systematic Gaps Identification

### Step 1: Find Missing Terms
```python
# Get all terms across all collections
all_terms = set()
for collection in large_collections:
    collection_df = df[df[collection_column] == collection]
    for subjects_list in collection_df['subjects_list'].dropna():
        if isinstance(subjects_list, list):
            all_terms.update(subjects_list)

# Find terms that are completely missing
all_gtaa_terms = set(self.graph.nodes()) - {self.root}
missing_terms = all_gtaa_terms - all_terms
```

**What this does:**
- Collects all terms used across all collections
- Compares with all possible GTAA terms
- Identifies terms that are never used

### Step 2: Find Rare Terms
```python
# Find terms that are very rare (< 10% of collections)
term_frequency = Counter()
for terms in collection_terms.values():
    term_frequency.update(terms)

rare_threshold = max(1, len(large_collections) * 0.1)  # 10% of collections
rare_terms = {term: count for term, count in term_frequency.items() 
             if count < rare_threshold}
```

**What this does:**
- Counts how many collections use each term
- Identifies terms used in less than 10% of collections
- These are "rare" but not completely missing

### Step 3: Analyze by Hierarchy Depth
```python
# Calculate coverage statistics by depth
for depth in self.depth_groups.keys():
    depth_terms = set(self.depth_groups[depth])
    covered_terms = depth_terms & all_terms
    gap_analysis['coverage_by_depth'][depth] = {
        'total_terms': len(depth_terms),
        'covered_terms': len(covered_terms),
        'coverage_ratio': len(covered_terms) / len(depth_terms) if depth_terms else 0
    }
```

**What this does:**
- Groups terms by their hierarchy depth
- Calculates coverage ratio for each depth level
- Reveals if deeper (more specific) terms are under-represented

## Phase 5: Results Generation and Output

### Step 1: Generate Reports
```python
# Generate comprehensive report
report = analyzer.generate_bias_report(df, 'fotocollectie')

# Save all results
analyzer.save_analysis_results(df, output_dir, 'fotocollectie')
```

### Step 2: Create Output Files
- `branch_representation.csv` - Detailed statistics for each branch
- `most_represented_branches.csv` - Top 10 branches by coverage
- `least_represented_branches.csv` - Bottom 10 branches by coverage
- `conceptual_domain_coverage.csv` - Domain-level statistics
- `systematic_gaps.json` - Detailed gap analysis
- `bias_analysis_report.txt` - Human-readable summary

## Key Algorithms

### 1. Root Branch Mapping
```python
def _find_root_branch(self, term: str) -> Optional[str]:
    # Find the path from root to this term
    path = nx.shortest_path(self.graph, self.root, term)
    if len(path) >= 2:  # root -> branch -> ... -> term
        return path[1]  # First child of root
```

### 2. Coverage Ratio Calculation
```python
collection_coverage_ratio = n_collections / total_collections
term_coverage_ratio = n_unique_terms / branch_size
```

### 3. Gap Analysis
```python
missing_terms = all_possible_terms - actually_used_terms
rare_terms = terms_used_in_less_than_10_percent_of_collections
```

## Data Flow

```
GTAA CSV → Graph Structure → Hierarchy Analysis
     ↓
Photo Archive → Term Extraction → Collection Mapping
     ↓
Branch Coverage → Domain Mapping → Gap Identification
     ↓
Statistics → Reports → CSV/JSON Files
```

## Computational Complexity

- **Time**: O(n × m × d) where n=collections, m=terms, d=graph depth
- **Space**: O(b × c) where b=branches, c=collections
- **Memory**: ~100MB for 17 collections × 3,498 terms

The script efficiently processes nearly 1 million records and generates comprehensive bias analysis in under 5 minutes. 