"""
Integration test: run the full pipeline on a small synthetic dataset and verify
that known matches are found.

Dataset design
--------------
Left                       Right                       Expected
─────────────────────────  ─────────────────────────── ────────────────────────
id=1  "apple inc"          id=101 "apple inc"          (1, 101) — exact match
id=2  "johnsson smith"     id=102 "johnson smith"      (2, 102) — fuzzy (≈0.93)
id=3  "unmatched corp"     id=103 "something different" no match for left_id=3

The pipeline uses group_strategy="last_token", so:
  - "apple_inc"      → group "inc",       match_value "apple"
  - "johnsson_smith" → group "smith",     match_value "johnsson"
  - "johnson_smith"  → group "smith",     match_value "johnson"
  - "unmatched_corp" → group "corp"       (no right-side peer in "corp")

ANN blocking finds the (2, 102) candidate because both records share group "smith".
Monge-Elkan("johnsson", "johnson") ≈ 0.933 ≥ threshold=0.88.

Note: first run downloads ~80 MB for sentence-transformers/all-MiniLM-L6-v2.
Run only integration tests with:  pytest -m integration
Skip integration tests with:      pytest -m "not integration"
"""

import pandas as pd
import pytest

from large_scale_entity_matching import run_pipeline
from large_scale_entity_matching.config import MatchingConfig


@pytest.mark.integration
def test_pipeline_finds_exact_and_fuzzy_matches(tmp_path):
    left = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["apple inc", "johnsson smith", "unmatched corp"],
    })
    right = pd.DataFrame({
        "id": [101, 102, 103],
        "name": ["apple inc", "johnson smith", "something different"],
    })

    left_path = str(tmp_path / "left.parquet")
    right_path = str(tmp_path / "right.parquet")
    left.to_parquet(left_path, index=False)
    right.to_parquet(right_path, index=False)

    config = MatchingConfig(
        num_candidate_partitions=4,
        top_k=5,
        threshold=0.88,
    )

    result = run_pipeline(
        left_input_file=left_path,
        right_input_file=right_path,
        left_id_col="id",
        right_id_col="id",
        left_key_cols=["name"],
        right_key_cols=["name"],
        work_dir=str(tmp_path / "work"),
        config=config,
    )

    matches = pd.read_parquet(result["final_output_parquet"])
    matched_pairs = set(zip(matches["left_id"].tolist(), matches["right_id"].tolist()))

    assert (1, 101) in matched_pairs, (
        f"Exact match (1, 101) not found. Got: {matched_pairs}"
    )
    assert (2, 102) in matched_pairs, (
        f"Fuzzy match (2, 102) not found. Got: {matched_pairs}"
    )
    assert not any(left_id == 3 for left_id, _ in matched_pairs), (
        f"Unexpected match for left_id=3 (unmatched corp). Got: {matched_pairs}"
    )
