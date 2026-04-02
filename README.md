# large-scale-entity-matching

A Python library for scalable entity matching on large tabular datasets — built for datasets where brute-force comparison is not an option.

## Benchmark

| Dataset | Left rows | Right rows | Runtime | Candidates found |
|---|---|---|---|---|
| US voter records | 15,000,000 | 10,000,000 | ~1.5 hours | ~1,500,000 |

> Run on Google Colab (free tier). No dedicated hardware required.

---

## How it works

Instead of comparing all record pairs (O(n²)), this library uses a three-stage pipeline to make large-scale matching tractable:

1. **Blocking** — groups records by token to reduce the candidate space
2. **ANN candidate generation** — uses FAISS + sentence embeddings to find approximate nearest neighbors within each block
3. **Fuzzy scoring** — ranks candidates using Monge-Elkan similarity and keeps the best matches

This combination allows matching datasets with tens of millions of rows on standard hardware.

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
    left_key_cols=["first_name", "last_name", "city"],
    right_key_cols=["first_name", "last_name", "city"],
    work_dir="work",
    config=config,
)

print(result["final_output_parquet"])
print(result["score_info"])
```

---

## Pipeline Overview

```
Raw input (CSV / Excel / Parquet)
        ↓
   Parquet conversion
        ↓
   Key construction
        ↓
   Blocking (token-based grouping)
        ↓
   Exact matching
        ↓
   ANN candidate generation (FAISS + embeddings)
        ↓
   Candidate partitioning
        ↓
   Fuzzy scoring (Monge-Elkan)
        ↓
   Best-match selection + merge
        ↓
   Final output (Parquet / CSV)
```

---

## Input Requirements

Each dataset must have:
- one ID column
- one or more columns used to build a matching key

Supported formats: CSV, Excel (`.xls`, `.xlsx`), Parquet

---

## Configuration

All parameters are controlled via `lsem.MatchingConfig`:

```python
config = lsem.MatchingConfig(
    group_strategy="last_token",       # blocking strategy: "last_token" | "first_token" | "none"
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # embedding model
    top_k=20,                          # ANN neighbors per record
    threshold=0.88,                    # minimum similarity score
    num_candidate_partitions=256,      # parallelism for fuzzy scoring
    prepare_chunk_size=200_000,        # preprocessing chunk size
    device="cpu",                      # "cpu" or "cuda"
)
```

### Key parameters

| Parameter | Description |
|---|---|
| `group_strategy` | Blocking strategy. `"last_token"` works well for names. `"none"` disables blocking (slow on large data). |
| `top_k` | Number of ANN neighbors per record. Higher = more recall, slower. |
| `threshold` | Minimum Monge-Elkan score to keep a match. |
| `num_candidate_partitions` | Controls memory usage during fuzzy scoring. |
| `model_name` | Any sentence-transformers compatible model. |

---

## Advanced Usage

Individual pipeline steps can be called separately:

```python
lsem.prepare_input_file(...)
lsem.prepare_blocking_features(...)
lsem.write_exact_matches(...)
lsem.write_candidate_pairs_ann_blocking_by_group(...)
lsem.split_candidates_into_partitions(...)
lsem.score_candidate_partitions(...)
lsem.keep_best_ties_from_parts(...)
lsem.merge_exact_and_fuzzy(...)
```

---

## Stack

`Python` · `FAISS` · `DuckDB` · `sentence-transformers` · `pandas`

---

## License

MIT License. See [LICENSE](LICENSE) for details.
