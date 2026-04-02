import pandas as pd
import pytest

from large_scale_entity_matching.scoring import (
    score_candidate_batch_optimized,
    symmetric_monge_elkan_similarity_cached,
)
from large_scale_entity_matching.preprocessing import tokenize_normalized_name


def _make_candidate_df(pairs):
    """Build a candidate DataFrame from a list of (left_mv, right_mv) tuples."""
    rows = []
    for i, (left_mv, right_mv) in enumerate(pairs):
        rows.append({
            "left_id": i,
            "right_id": i + 100,
            "key_left": left_mv,
            "key_right": right_mv,
            "match_value_left": left_mv,
            "match_value_right": right_mv,
            "value_length_left": len(left_mv),
            "value_length_right": len(right_mv),
            "token_count_left": len(left_mv.split("_")),
            "token_count_right": len(right_mv.split("_")),
            "token_first_left": left_mv.split("_")[0],
            "token_first_right": right_mv.split("_")[0],
            "token_last_left": left_mv.split("_")[-1],
            "token_last_right": right_mv.split("_")[-1],
            "blocking_rule": "ann_dense",
        })
    return pd.DataFrame(rows)


class TestScoreCandidateBatchOptimized:
    def test_high_similarity_pair_is_included(self):
        # "johnsson" vs "johnson": Monge-Elkan ≈ 0.933 > 0.88
        df = _make_candidate_df([("johnsson", "johnson")])
        result = score_candidate_batch_optimized(df, threshold=0.88)
        assert len(result) == 1
        assert result.iloc[0]["left_id"] == 0
        assert result.iloc[0]["right_id"] == 100
        assert result.iloc[0]["score"] >= 0.88

    def test_low_similarity_pair_is_excluded(self):
        # "abc" vs "xyz": score = 0 < 0.88
        df = _make_candidate_df([("abc", "xyz")])
        result = score_candidate_batch_optimized(df, threshold=0.88)
        assert len(result) == 0

    def test_rel_len_diff_filter_excludes_pair(self):
        # "ab" (len 2) vs "abcdefghij" (len 10): rel_diff = 0.8 > 0.15
        df = _make_candidate_df([("ab", "abcdefghij")])
        result = score_candidate_batch_optimized(df, threshold=0.0, max_rel_len_diff=0.15)
        assert len(result) == 0

    def test_rel_len_diff_filter_passes_similar_length(self):
        # "johnsson" (8) vs "johnson" (7): rel_diff = 0.125 ≤ 0.15 → should pass filter
        df = _make_candidate_df([("johnsson", "johnson")])
        result = score_candidate_batch_optimized(df, threshold=0.0, max_rel_len_diff=0.15)
        assert len(result) == 1

    def test_token_count_diff_filter(self):
        # 1 token vs 3 tokens: diff = 2 > 1 → filtered
        df = _make_candidate_df([("apple", "apple_inc_corp")])
        result = score_candidate_batch_optimized(df, threshold=0.0)
        assert len(result) == 0

    def test_empty_input_returns_correct_columns(self):
        df = pd.DataFrame(columns=[
            "left_id", "right_id", "key_left", "key_right",
            "match_value_left", "match_value_right",
            "value_length_left", "value_length_right",
            "token_count_left", "token_count_right",
            "token_first_left", "token_first_right",
            "token_last_left", "token_last_right",
            "blocking_rule",
        ])
        result = score_candidate_batch_optimized(df)
        assert result.empty
        assert set(result.columns) == {
            "left_id", "right_id", "key_left", "key_right", "score", "match_type"
        }

    def test_threshold_boundary_inclusive(self):
        # A pair whose score equals exactly the threshold should be included
        tokens_a = tokenize_normalized_name("johnsson")
        tokens_b = tokenize_normalized_name("johnson")
        exact_threshold = symmetric_monge_elkan_similarity_cached(tokens_a, tokens_b, {})

        df = _make_candidate_df([("johnsson", "johnson")])
        result = score_candidate_batch_optimized(df, threshold=exact_threshold)
        assert len(result) == 1

    def test_multiple_pairs_mixed_result(self):
        # One pair above threshold, one below
        df = _make_candidate_df([
            ("johnsson", "johnson"),  # ≈ 0.933 — should match
            ("abc", "xyz"),           # ≈ 0.0   — should not match
        ])
        result = score_candidate_batch_optimized(df, threshold=0.88)
        assert len(result) == 1
        assert result.iloc[0]["left_id"] == 0


class TestSymmetricMongeElkan:
    def test_identical_single_token(self):
        score = symmetric_monge_elkan_similarity_cached(["apple"], ["apple"], {})
        assert score == pytest.approx(1.0)

    def test_completely_different(self):
        score = symmetric_monge_elkan_similarity_cached(["abc"], ["xyz"], {})
        assert score == pytest.approx(0.0)

    def test_empty_tokens(self):
        assert symmetric_monge_elkan_similarity_cached([], ["apple"], {}) == 0.0
        assert symmetric_monge_elkan_similarity_cached(["apple"], [], {}) == 0.0

    def test_similar_single_token(self):
        # "johnsson" vs "johnson": symmetric since single tokens on both sides
        score = symmetric_monge_elkan_similarity_cached(["johnsson"], ["johnson"], {})
        assert score > 0.88
