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

# Installation

Once published on PyPI:

```bash
pip install large-scale-entity-matching
```

For local development:

```bash
pip install -e .
```

---

# Quick Start

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

# Core API

## run_pipeline(...)

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

## run_pipeline_only_result(...)

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

# Input Requirements

Each dataset must contain:

* one ID column
* one or more columns used to build a matching key

Supported formats:

* CSV (.csv)
* Excel (.xls, .xlsx)
* Parquet (.parquet)

---

# Pipeline Overview

The pipeline consists of:

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

# Configuration

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

# Configuration Parameters

## Blocking

* `group_strategy`
  Determines how records are grouped for blocking
  Options:

  * `"last_token"`
  * `"first_token"`
  * `"none"`

* `prepare_chunk_size`
  Chunk size for preprocessing

---

## ANN (Candidate Generation)

* `model_name`
  Embedding model

* `device`
  `"cpu"` or `"cuda"`

* `top_k`
  Number of neighbors per record

* `left_query_chunk_size`
  Query batch size

* `right_encode_batch_size`, `left_encode_batch_size`
  Embedding batch sizes

* `max_abs_len_diff`
  Max absolute length difference filter

* `max_token_diff`
  Max token difference filter

* `hnsw_m`, `ef_construction`, `ef_search`
  FAISS parameters

* `normalize_embeddings`
  Normalize embeddings before search

---

## Partitioning

* `num_candidate_partitions`
  Number of partitions for candidate pairs

---

## Scoring

* `threshold`
  Minimum similarity score

* `max_rel_len_diff`
  Max relative length difference

---

## Performance

* `memory_limit`
  DuckDB memory

* `temp_directory`
  Temp path

* `threads`
  Number of threads

* `progress_every`
  Logging frequency

---

# Output

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

# Example (CSV Input)

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

# Advanced Usage

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

# Notes

* Designed for large-scale datasets
* Efficient via DuckDB + FAISS
* Blocking strategy strongly affects performance
* `"none"` group strategy may be slow for large data

---

# License

Add your license here.
