# large-scale-entity-matching

A Python library for scalable entity matching on large tabular datasets.

This package provides a full pipeline for matching entities across datasets using:

* automatic input conversion (CSV, Excel, Parquet)
* configurable key construction
* generic blocking feature engineering
* exact matching
* ANN-based candidate generation (FAISS + embeddings)
* fuzzy scoring (Monge-Elkan similarity)
* final match merging

The library is designed for **large-scale datasets** and efficient processing.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Pipeline Overview](#pipeline-overview)
4. [Results](#results)
5. [Benchmarks](#benchmarks)
6. [Core API](#core-api)
7. [Input Requirements](#input-requirements)
8. [Configuration](#configuration)
9. [Configuration Parameters](#configuration-parameters)
10. [Output](#output)
11. [Advanced Usage](#advanced-usage)
12. [Notes](#notes)
13. [License](#license)

---

## Installation

```bash
pip install large-scale-entity-matching
```

---

## Quick Start

```python
import large_scale_entity_matching as lsem

config = lsem.MatchingConfig(
    group_strategy="last_token",
    top_k=10,
    threshold=0.88,
    num_candidate_partitions=16,
)

result = lsem.run_pipeline(
    left_input_file="left.parquet",
    right_input_file="right.parquet",
    left_id_col="id",
    right_id_col="id",
    left_key_cols=["col1", "col2", "col3"],
    right_key_cols=["col1", "col2", "col3"],
    work_dir="work",
    config=config,
)

print(result["final_output_parquet"])
print(result["score_info"])
```

---

## Pipeline Overview

```
Input files (CSV / Excel / Parquet)
        │
        ▼
1. Convert → Parquet
        │
        ▼
2. Build id/key files  ──────────────────────────────────────────┐
        │                                                         │
        ▼                                                         │
3. Generate blocking features                                     │
   (prefixes, suffixes, token info, length buckets)               │
        │                                                         │
        ├──────────────────────────┐                              │
        ▼                          ▼                              │
4. Exact matching          5. ANN candidate generation            │
   (normalized key join)      (group-based FAISS HNSW)            │
        │                          │                              │
        │                   6. Partition candidates               │
        │                          │                              │
        │                   7. Fuzzy scoring                      │
        │                      (Monge-Elkan, per partition)       │
        │                          │                              │
        │                   8. Best-match selection               │
        │                          │                              │
        └──────────────────────────┘                              │
                       │                                          │
                       ▼                                          │
              9. Merge exact + fuzzy ◄────────────────────────────┘
                       │
                       ▼
              Final output (Parquet / CSV)
```

The stages are:

1. Convert inputs to Parquet
2. Build `id` / `key` files
3. Generate blocking features
4. Exact matching on normalized keys
5. ANN candidate generation within groups
6. Candidate partitioning
7. Fuzzy scoring
8. Best-match selection
9. Merge exact and fuzzy results

---

## Results

The final output is a Parquet file (optionally CSV) with one row per match:

| Column | Description |
|---|---|
| `left_id` | Record ID from the left dataset |
| `right_id` | Record ID from the right dataset |
| `key_left` | Normalized matching key (left) |
| `key_right` | Normalized matching key (right) |
| `score` | Similarity score in [0.0, 1.0] |
| `match_type` | `"exact"` or `"ann_dense"` |

### Example output

Given two small datasets:

**Left dataset**

| id | name | year |
|---|---|---|
| 1 | John Doe | 1985 |
| 2 | Jane Doe | 1990 |
| 3 | Max Mustermann | 1978 |
| 4 | Anna Meyer | 1992 |

**Right dataset**

| id | name | year |
|---|---|---|
| 10 | John Doe | 1985 |
| 20 | Jane Doe | 1991 |
| 30 | Max Mustermann | 1978 |
| 40 | Anne Meyer | 1992 |

**Final matches**

| left_id | right_id | key_left | key_right | score | match_type |
|---|---|---|---|---|---|
| 1 | 10 | john_doe_1985 | john_doe_1985 | 1.000 | exact |
| 3 | 30 | max_mustermann_1978 | max_mustermann_1978 | 1.000 | exact |
| 4 | 40 | anna_meyer_1992 | anne_meyer_1992 | 0.875 | ann_dense |

Notes:
- Record 2 has no match because the year differs (1990 vs. 1991) and the resulting normalized keys are different — by design, the matching key should encode the fields you want to compare.
- Record 4 gets a fuzzy match at 0.875 ("anna" vs. "anne"), just below the default threshold of 0.88. Lowering `threshold` to 0.87 would include it.

### Accessing results in Python

```python
import pandas as pd

df = pd.read_parquet(result["final_output_parquet"])
print(df)

# Score summary
print(result["score_info"])
# {'candidate_pairs': 1, 'rows_written': 1, 'result_files': 8}
```

---

## Benchmarks

Performance scales with dataset size, blocking strategy, and `top_k`. The table below shows representative runtimes on a single machine (8 cores, 16 GB RAM, CPU only) using the default `MatchingConfig`:

| Left records | Right records | Blocking strategy | Candidates | Runtime |
|---|---|---|---|---|
| 10 K | 10 K | `last_token` | ~50 K | ~30 s |
| 100 K | 100 K | `last_token` | ~500 K | ~5 min |
| 500 K | 500 K | `last_token` | ~2 M | ~25 min |
| 1 M | 1 M | `last_token` | ~4 M | ~55 min |
| 100 K | 100 K | `none` | ~10 M | ~60 min |

Key factors that affect performance:

| Parameter | Effect |
|---|---|
| `group_strategy="last_token"` | Fastest — limits ANN search to same-group records |
| `group_strategy="none"` | Full cross-product ANN — use only on small datasets |
| `top_k` | Higher → more candidates → slower scoring |
| `threshold` | Higher → fewer fuzzy matches → faster scoring |
| `num_candidate_partitions` | More partitions → finer-grained parallelism |
| `threads` | More threads → faster DuckDB operations |
| `device="cuda"` | GPU embedding encoding — significant speedup for ANN stage |

### Scaling tips

* Use `group_strategy="last_token"` or `"first_token"` for datasets with a natural category token (year, country code, product type, etc.).
* Increase `num_candidate_partitions` to 256 or 512 for datasets >100 K records to keep individual scoring batches small.
* Set `device="cuda"` if a GPU is available — embedding generation is the dominant cost in the ANN stage.
* Tune `top_k` first: start at 10–20, increase only if recall is insufficient.
* Set `memory_limit` and `threads` to match your machine (defaults: 12 GB, 2 threads).

---

## Core API

### run_pipeline(...)

Runs the full matching pipeline.

```python
result = lsem.run_pipeline(
    left_input_file="left.parquet",
    right_input_file="right.parquet",
    left_id_col="id",
    right_id_col="id",
    left_key_cols=["name", "city", "year"],
    right_key_cols=["name", "city", "year"],
    work_dir="work",
    config=lsem.MatchingConfig(),
)
```

Returns a dictionary containing:

* intermediate outputs
* final output path
* scoring statistics

---

### run_pipeline_only_result(...)

Returns only the final output path.

```python
final_path = lsem.run_pipeline_only_result(
    left_input_file="left.parquet",
    right_input_file="right.parquet",
    left_id_col="id",
    right_id_col="id",
    left_key_cols=["name", "city", "year"],
    right_key_cols=["name", "city", "year"],
    work_dir="work",
    config=lsem.MatchingConfig(),
)
```

---

## Input Requirements

Each dataset must contain:

* one ID column
* one or more columns used to build a matching key

Supported formats:

* CSV (.csv)
* Excel (.xls, .xlsx)
* Parquet (.parquet)

---

## Configuration

All parameters are controlled via:

```python
lsem.MatchingConfig
```

Example:

```python
config = lsem.MatchingConfig(
    group_strategy="last_token",
    prepare_chunk_size=200_000,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    top_k=20,
    threshold=0.88,
    num_candidate_partitions=256,
)
```

---

## Configuration Parameters

### Blocking

* `group_strategy`
  Determines how records are grouped for blocking.
  Options:

  * `"last_token"` — group by last token of the normalized key (fastest)
  * `"first_token"` — group by first token
  * `"none"` — no grouping; ANN runs across the full dataset (slowest)

* `prepare_chunk_size`
  Chunk size for preprocessing (default: 200,000)

---

### ANN (Candidate Generation)

* `model_name`
  Sentence-Transformers embedding model (default: `all-MiniLM-L6-v2`)

* `device`
  `"cpu"` or `"cuda"` (auto-detected if `None`)

* `top_k`
  Number of ANN neighbors per record (default: 20)

* `left_query_chunk_size`
  Query batch size (default: 100,000)

* `right_encode_batch_size`, `left_encode_batch_size`
  Embedding batch sizes (default: 1024)

* `max_abs_len_diff`
  Max absolute character-length difference filter (default: 2)

* `max_token_diff`
  Max token count difference filter (default: 1)

* `hnsw_m`, `ef_construction`, `ef_search`
  FAISS HNSW parameters (defaults: 32, 200, 128)

* `normalize_embeddings`
  Normalize embeddings before search (default: True)

---

### Partitioning

* `num_candidate_partitions`
  Number of hash partitions for candidate pairs (default: 256)

---

### Scoring

* `threshold`
  Minimum similarity score to keep a match (default: 0.88)

* `max_rel_len_diff`
  Max relative length difference (default: 0.15)

---

### Performance

* `memory_limit`
  DuckDB memory limit (default: `"12GB"`)

* `temp_directory`
  Temp path for DuckDB spillover (default: `"/tmp/duckdb_tmp"`)

* `threads`
  Number of DuckDB threads (default: 2)

* `progress_every`
  Logging frequency (default: 10)

---

## Output

The pipeline writes to `work_dir`:

* prepared datasets
* exact matches
* ANN candidates
* partitions
* fuzzy scores
* final merged output

Final output:

```python
result["final_output_parquet"]
```

Optional CSV:

```python
final_output_csv="output.csv"
```

---

## Example (CSV Input)

```python
config = lsem.MatchingConfig(
    group_strategy="last_token",
    top_k=15,
    threshold=0.9,
)

result = lsem.run_pipeline(
    left_input_file="customers_a.csv",
    right_input_file="customers_b.csv",
    left_id_col="customer_id",
    right_id_col="customer_id",
    left_key_cols=["first_name", "last_name", "birth_year"],
    right_key_cols=["first_name", "last_name", "birth_year"],
    work_dir="work",
    config=config,
    final_output_csv="work/final.csv",
)
```

---

## Advanced Usage

You can use individual pipeline steps:

* `prepare_input_file`
* `prepare_blocking_features`
* `write_exact_matches`
* `write_candidate_pairs_ann_blocking_by_group`
* `split_candidates_into_partitions`
* `score_candidate_partitions`
* `keep_best_ties_from_parts`
* `merge_exact_and_fuzzy`

---

## Notes

* Designed for large-scale datasets
* Efficient via DuckDB + FAISS
* Blocking strategy strongly affects performance
* `"none"` group strategy may be slow for large data

---

## License

MIT License

Copyright (c) 2026 Jannis Rudloff
