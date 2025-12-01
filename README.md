# Title

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

Description Tables

Table 1

2.2 Baseline Approaches 
Baselines (pre and post only)

Table 2

3 DESIGN AND IMPLEMENTATION 

3.1 Architectural Overview (scheduler.py and selector.py)

3.2 Indexing (index.py)

3.4 Hybrid search algorithm (search.py)

3.5 List ordering and per-list counts (list_ordering.py)

3.6 Early stop policies (early_stop.py)

3.3 Query filtering and allow-list construction (hybrid backend)

Table 3

4.2 Dataset and workload 
Artifacts (Do not need to do v1)

4.4 Metrics (metrics.py)


Table 4

5.1 Results
Plots ()






