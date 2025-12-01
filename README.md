# Efficient and Predicate-Aware Hybrid ANN Search for Yelp 

## Directory Tree
```
.
├── artifacts
│   ├── artifacts.py
│   ├── make_queries.py
│   ├── make_query_vectors.py
│   ├── metadata.schema.json
│   ├── v1
│   │   └── ...
│   └── v2
│       ├── metadata.parquet
│       ├── queries.parquet
│       ├── query_vectors.npy
│       ├── vectors.meta.json
│       └── vectors.npy
├── Manual
│   ├── Data & Interface Specification Manual (v2.0).pdf
│   └── Data & Interface Specification Manual.Rmd
├── plots
│   ├── latency_sv.py
│   ├── selectivity_latency.py
│   ├── selectivity_recall.py
│   ├── selectivity_sv_v1.py
│   └── selectivity_sv_v2.py
├── results
│   ├── indexes
│   │   └── faiss_ivf.index
│   ├── week1
│   │   └── ...
│   ├── week2
│   │   └── ...
│   ├── week4
│   │   └── ...
│   ├── week5
│   │   └── ...
│   ├── week6
│   │   └── ...
│   └── week7
│       ├── hybrid_results_k_bound.jsonl
│       ├── hybrid_results_k_only.jsonl
│       ├── hybrid_results_rm.jsonl
│       ├── hybrid_results.jsonl
│       ├── latency_selectivity_loess.png
│       ├── post_results.jsonl
│       ├── pre_results.jsonl
│       ├── recall_selectivity_loess.png
│       ├── scored_vectors_selectivity_loess.png
│       └── scored_vectors_selectivity_post_vs_hybrid.png
├── src
│   ├── backends
│   │   ├── backend_interface.py
│   │   ├── exact_backend.py
│   │   ├── hybrid_backend.py
│   │   ├── post_filter_backend.py
│   │   └── prefilter_backend.py
│   ├── baselines
│   │   ├── post_filter.py
│   │   └── pre_filter.py
│   │   ├── hybrid
│   │   │   ├── early_stop.py
│   │   │   ├── index.py
│   │   │   ├── list_ordering.py
│   │   │   ├── scheduler.py
│   │   │   ├── search.py
│   │   │   └── selector.py
│   ├── dataio
│   │   ├── loaders.py
│   │   └── validators.py
│   ├── eval
│   │   ├── hybrid_validation.py
│   │   ├── metrics.py
│   │   ├── oracle.py
│   │   └── selectivity.py
│   ├── harness
│   │   └── run.py
│   └── utils
│       ├── logger.py
│       └── timing.py
```
## Files Description

### 1. Introduction


| Report Section | File | Description |
|---------------|------|-------------|
| 1.2 Baseline Approaches | `baselines/pre_filter.py` | Contains the search method for **pre-filter search** (brute-force search). |
| 1.2 Baseline Approaches | `baselines/post_filter.py` | Contains the search method for **post-filter search** (ANN search). |

---

### 2. Design and Implementation

| Report Section | File | Description |
|---------------|------|-------------|
| 2.1 Architectural Overview | `src/baselines/hybrid/scheduler.py` | Sets the **nprobe** for each iteration. |
| 2.1 Architectural Overview | `src/baselines/hybrid/selector.py` | Creates the **IDSelectorBatch** for metadata filtering. |
| 2.2 Indexing | `src/baselines/hybrid/index.py` | Creates the **FAISS IVF index**. |
| 2.4 Hybrid search algorithm | `src/baselines/hybrid/search.py` | Contains the hybrid search algorithm implementation. |
| 2.4 List ordering and per-list counts | `src/baselines/hybrid/list_ordering.py` | Constructs the **probe order** based on centroid scores. |
| 2.6 Early stop policies | `src/baselines/hybrid/early_stop.py` | Implements different early-stop policies. |
| 2.3 Query filtering and allow-list construction | `src/backends/hybrid_backend.py` | Computes allow-list counters, sets selectivity-based nprobe scaling, and calls the search method. |

---

### 3. Experimental Setup

| Report Section | File | Description |
|---------------|------|-------------|
| 3.2 Dataset and workload artifacts | `artifacts/artifacts.py` | Prepares dataset artifacts used in all experiments. |
| 3.2 Dataset and workload artifacts | `artifacts/make_queries.py` | Generates query sets across selectivities for experiments. |
| 3.2 Dataset and workload artifacts | `artifacts/make_query_vectors.py` | Converts raw queries into vector embeddings. |
| 3.4 Metrics | `src/eval/metrics.py` | Implements evaluation metrics, including **recall@k**. |

---

### 4. Evaluation

| Report Section | File | Description |
|---------------|------|-------------|
| 4.1 Results | `plots/selectivity_latency.py` | Creates the **selectivity vs latency** plot for all search methods. |
| 4.1 Results | `plots/selectivity_recall.py` | Creates the **selectivity vs recall** plot for all search methods. |
| 4.1 Results | `plots/selectivity_sv_v1.py` | Visualizes **selectivity vs scored vectors** for post-filter, pre-filter, and hybrid search. |
| 4.1 Results | `plots/selectivity_sv_v2.py` | Visualizes **selectivity vs scored vectors** for post-filter and hybrid search. |
| 4.1 Results | `src/harness/run.py` | CLI for running experiments with chosen search type and nprobe; outputs JSON stats. |





