# large-scale-entity-matching

[![PyPI](https://img.shields.io/pypi/v/large-scale-entity-matching)](https://pypi.org/project/large-scale-entity-matching/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

A Python library for scalable entity matching on large tabular datasets — built for cases where brute-force pairwise comparison is not an option.

## Benchmark

| Dataset | Left rows | Right rows | Runtime | Candidates found |
|---|---|---|---|---|
| US voter records | 15,000,000 | 10,000,000 | ~1.5 hours | ~1,500,000 |

## How it works

Instead of comparing all record pairs (O(n²)), the library uses a three-stage pipeline:

```
Raw input (CSV / Excel / Parquet)
        ↓
   Blocking          — groups records by token to reduce candidate space
        ↓
   ANN candidate     — FAISS + sentence embeddings to find approximate
   generation          nearest neighbors within each block
        ↓
   Fuzzy scoring     — Monge-Elkan similarity, keeps best matches
        ↓
Final output (Parquet / CSV)
```

This combination allows matching datasets with tens of millions of rows on standard hardware.

## Installation

```bash
pip install large-scale-entity-matching
```

## Quick start

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

## Configuration

All parameters are controlled via `lsem.MatchingConfig`:

| Parameter | Description |
|---|---|
| `group_strategy` | Blocking strategy: `"last_token"` \| `"first_token"` \| `"none"` |
| `top_k` | ANN neighbors per record. Higher = more recall, slower. |
| `threshold` | Minimum Monge-Elkan score to keep a match. |
| `num_candidate_partitions` | Controls memory usage during fuzzy scoring. |
| `model_name` | Any `sentence-transformers`-compatible model. |
| `device` | `"cpu"` or `"cuda"` |

## Advanced usage

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

## Stack

Python · FAISS · DuckDB · sentence-transformers · pandas

## License

MIT — see [LICENSE](LICENSE) for details.
