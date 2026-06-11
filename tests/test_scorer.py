# tests/test_scorer.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Test suite for the scorer package components:
#               metrics.py, aggregate.py, and judge.py.
#
#               Why these tests matter:
#                 The scorer is the mathematical core of the pipeline.
#                 If compute_metrics() returns values outside [0, 1],
#                 the CI gate thresholds become meaningless. If
#                 aggregate_scores() applies the wrong weights, the
#                 overall score silently misrepresents model quality.
#                 If judge_score() fails to parse the LLM's JSON response,
#                 the entire scoring run crashes mid-pipeline.
#                 These tests lock in the correctness of all three components.
#
#               Testing strategy:
#                 TestComputeMetrics  : Calls the REAL bert_score and
#                   rouge_score functions — no mocking. These are
#                   deterministic libraries so results are reproducible.
#                   Tests verify output range and key presence.
#
#                 TestAggregateScores : Uses sample_results fixture from
#                   conftest.py — no LLM calls. Verifies aggregation math,
#                   key presence, and that the weighted average is correct.
#
#                 TestJudgeScore : Mocks the LiteLLM completion() function
#                   so no API key is needed. Tests the JSON parsing logic,
#                   key presence in the return dict, and the fallback
#                   'overall' computation when the judge omits it.
#
# Run:
#   pytest tests/test_scorer.py -v
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# pytest       : Test framework — provides fixtures and assertions.
#
# patch        : unittest.mock — temporarily replaces scorer.judge.completion
#                with a fake function during TestJudgeScore tests.
#                The patch target "scorer.judge.completion" must match the
#                exact import path used inside judge.py:
#                  from litellm import completion   →   patch("scorer.judge.completion")
#
# MagicMock    : unittest.mock — creates a fake completion() return value.
#                MagicMock auto-creates attributes on access, so
#                mock.choices[0].message.content works without setup.
# ===============================================================
import pytest
from unittest.mock import patch, MagicMock


# ===============================================================
# TestComputeMetrics
# ---------------------------------------------------------------
# Tests for scorer/metrics.py — the deterministic quality metrics.
# These tests call the REAL BERTScore and ROUGE-L functions.
# No mocking — we want to verify the actual library behavior.
#
# Why not mock here?
#   BERTScore and ROUGE-L are deterministic. The same inputs always
#   produce the same outputs. Mocking them would test nothing — we'd
#   just be verifying that our mock returns what we told it to return.
#   Running real calls here also catches version-related breakage
#   (e.g. if bert-score changes its output format in a new version).
#
# Fixture used: none — all inputs are defined inline.
# ===============================================================
class TestComputeMetrics:

    # -----------------------------------------------------------
    # test_bertscore_in_range
    # -----------------------------------------------------------
    # BERTScore F1 is bounded to [0, 1] by definition.
    # Uses semantically similar sentences to exercise the embedding
    # comparison path (not just identical string matching).
    # If this fails, bert_score returned an unexpected value type
    # or the F1[0].item() extraction is broken.
    # -----------------------------------------------------------
    def test_bertscore_in_range(self):
        """BERTScore F1 should always be in [0, 1]."""
        from scorer.metrics import compute_metrics
        result = compute_metrics(
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
        )
        assert 0.0 <= result["bertscore_f1"] <= 1.0


    # -----------------------------------------------------------
    # test_rouge_l_in_range
    # -----------------------------------------------------------
    # ROUGE-L fmeasure is bounded to [0, 1] by definition.
    # Uses slightly different sentences to exercise the LCS path.
    # If this fails, rouge_scorer returned an unexpected structure
    # or the "rougeL" key name changed in a library update.
    # -----------------------------------------------------------
    def test_rouge_l_in_range(self):
        """ROUGE-L should always be in [0, 1]."""
        from scorer.metrics import compute_metrics
        result = compute_metrics(
            "Water boils at 100 degrees Celsius.",
            "Water boils at 100°C at sea level.",
        )
        assert 0.0 <= result["rouge_l"] <= 1.0


    # -----------------------------------------------------------
    # test_metrics_returns_both_keys
    # -----------------------------------------------------------
    # Verifies the return dict has exactly the expected keys.
    # aggregate.py accesses result["bertscore_f1"] and result["rouge_l"]
    # directly — a missing key would cause a KeyError in production.
    # -----------------------------------------------------------
    def test_metrics_returns_both_keys(self):
        """compute_metrics must return both expected keys."""
        from scorer.metrics import compute_metrics
        result = compute_metrics(
            "Hello world test",
            "Hello world reference test"
        )
        assert "bertscore_f1" in result
        assert "rouge_l"      in result


    # -----------------------------------------------------------
    # test_identical_strings_high_score
    # -----------------------------------------------------------
    # When response and reference are identical, both metrics
    # should be very close to 1.0 (perfect match).
    # Threshold is 0.95 rather than 1.0 to allow for minor floating
    # point variance in BERTScore's tensor operations.
    # -----------------------------------------------------------
    def test_identical_strings_high_score(self):
        """Identical response and reference should produce high scores."""
        from scorer.metrics import compute_metrics
        text   = "The answer is forty-two and nothing else."
        result = compute_metrics(text, text)
        assert result["rouge_l"]      >= 0.95
        assert result["bertscore_f1"] >= 0.95


    # -----------------------------------------------------------
    # test_completely_different_strings_low_rouge
    # -----------------------------------------------------------
    # When response and reference share no meaningful content,
    # ROUGE-L should be very low (little or no common subsequence).
    # Uses topic-unrelated sentences to maximize semantic distance.
    # Threshold is 0.30 to allow for accidental stopword overlap.
    # -----------------------------------------------------------
    def test_completely_different_strings_low_rouge(self):
        """Completely unrelated strings should have low ROUGE-L."""
        from scorer.metrics import compute_metrics
        result = compute_metrics(
            "The cat sat on the mat near the window.",
            "Quantum mechanics describes subatomic particle behavior.",
        )
        assert result["rouge_l"] < 0.30


# ===============================================================
# TestAggregateScores
# ---------------------------------------------------------------
# Tests for scorer/aggregate.py — the score aggregation function.
# Uses the sample_results and sample_scorer_cfg fixtures from conftest.py.
# No LLM calls, no file I/O — pure Python math verification.
#
# What aggregate_scores() must guarantee:
#   1. Returns a dict with "overall" and "categories" keys
#   2. All scores are floats in [0, 1]
#   3. All five categories are present in the output
#   4. The weighted math is arithmetically correct
#
# Fixture used:
#   sample_results    : List of 20 per-example result dicts (4 per category)
#   sample_scorer_cfg : Scorer config with metric_weights and categories
# ===============================================================
class TestAggregateScores:

    # -----------------------------------------------------------
    # test_returns_required_keys
    # -----------------------------------------------------------
    # Most basic structural check — the output dict must have both
    # "overall" and "categories" keys. These are accessed directly
    # by run_scorer.py and check_regression.py without .get() safety.
    # -----------------------------------------------------------
    def test_returns_required_keys(self, sample_results, sample_scorer_cfg):
        """aggregate_scores must return 'overall' and 'categories'."""
        from scorer.aggregate import aggregate_scores
        result = aggregate_scores(sample_results, sample_scorer_cfg)
        assert "overall"    in result
        assert "categories" in result


    # -----------------------------------------------------------
    # test_overall_in_range
    # -----------------------------------------------------------
    # The overall score is a weighted sum of per-category scores.
    # Since all inputs are in [0, 1] and weights sum to 1.0,
    # the output must also be in [0, 1].
    # If this fails, there's a weight configuration error.
    # -----------------------------------------------------------
    def test_overall_in_range(self, sample_results, sample_scorer_cfg):
        """Overall score must be in [0, 1]."""
        from scorer.aggregate import aggregate_scores
        result = aggregate_scores(sample_results, sample_scorer_cfg)
        assert 0.0 <= result["overall"] <= 1.0


    # -----------------------------------------------------------
    # test_per_category_scores_in_range
    # -----------------------------------------------------------
    # Every individual category score must also be in [0, 1].
    # Uses an f-string in the assertion message so failures name
    # the specific category that produced the out-of-range value.
    # -----------------------------------------------------------
    def test_per_category_scores_in_range(self, sample_results, sample_scorer_cfg):
        """Every per-category score must be in [0, 1]."""
        from scorer.aggregate import aggregate_scores
        result = aggregate_scores(sample_results, sample_scorer_cfg)
        for cat, score in result["categories"].items():
            assert 0.0 <= score <= 1.0, f"{cat} score {score} out of [0,1] range"


    # -----------------------------------------------------------
    # test_all_categories_present
    # -----------------------------------------------------------
    # All five behavioral categories from sample_results must appear
    # in the output categories dict.
    # Uses set comparison — order doesn't matter, presence does.
    # If a category is missing, check_regression.py would silently
    # skip it rather than comparing against the baseline.
    # -----------------------------------------------------------
    def test_all_categories_present(self, sample_results, sample_scorer_cfg):
        """All five categories from results should appear in output."""
        from scorer.aggregate import aggregate_scores
        result   = aggregate_scores(sample_results, sample_scorer_cfg)
        expected = {
            "factual_accuracy",
            "refusal_behavior",
            "instruction_following",
            "tone_consistency",
            "multi_turn_coherence",
        }
        assert set(result["categories"].keys()) == expected


    # -----------------------------------------------------------
    # test_weighted_math_is_correct
    # -----------------------------------------------------------
    # The most precise test in this class.
    # Uses crafted results where both inputs and expected outputs
    # are known exactly:
    #   factual_accuracy  : all metrics = 1.0  →  composite = 1.0
    #   refusal_behavior  : all metrics = 0.0  →  composite = 0.0
    #   weights: 0.60 and 0.40
    #   expected overall  = 1.0 × 0.60 + 0.0 × 0.40 = 0.60
    #
    # Tolerance of 0.001 accounts for floating point rounding in
    # the weighted sum calculation.
    # -----------------------------------------------------------
    def test_weighted_math_is_correct(self, sample_scorer_cfg):
        """Overall should equal the weighted sum of category scores."""
        from scorer.aggregate import aggregate_scores

        # Craft two examples with known perfect and zero scores
        results = [
            {
                "category"    : "factual_accuracy",
                "metrics"     : {"bertscore_f1": 1.0, "rouge_l": 1.0},
                "judge_score" : {"overall": 1.0},
            },
            {
                "category"    : "refusal_behavior",
                "metrics"     : {"bertscore_f1": 0.0, "rouge_l": 0.0},
                "judge_score" : {"overall": 0.0},
            },
        ]

        # Override config to use only these two categories with known weights
        cfg = {
            **sample_scorer_cfg,
            "categories": [
                {"name": "factual_accuracy",  "weight": 0.60},
                {"name": "refusal_behavior",  "weight": 0.40},
            ]
        }

        result           = aggregate_scores(results, cfg)
        expected_overall = 1.0 * 0.60 + 0.0 * 0.40   # = 0.60

        assert abs(result["overall"] - expected_overall) < 0.001


# ===============================================================
# TestJudgeScore
# ---------------------------------------------------------------
# Tests for scorer/judge.py — the judge LLM evaluator.
# ALL LiteLLM completion() calls are mocked — no API key needed.
#
# Why mock here?
#   Unlike BERTScore (deterministic), LLM responses are non-deterministic
#   and require a paid API key. Mocking lets these tests run:
#     - In CI with no secrets configured
#     - Locally without burning API credits
#     - Instantly (no network latency)
#
# How the mock works:
#   patch("scorer.judge.completion") replaces the completion function
#   inside judge.py with a MagicMock. We set:
#     mock_completion.choices[0].message.content = "...JSON string..."
#   This simulates exactly what the real LiteLLM would return.
#
# Fixture used:
#   sample_scorer_cfg : Scorer config with judge_model and judge_temperature
# ===============================================================
class TestJudgeScore:

    # -----------------------------------------------------------
    # test_judge_returns_expected_keys
    # -----------------------------------------------------------
    # The most important structural test for judge.py.
    # Verifies the return dict has all four required keys:
    #   factual_accuracy, instruction_following, safety, overall
    # These keys are accessed by aggregate_scores() and run_scorer.py
    # without .get() safety — a missing key crashes the pipeline.
    #
    # Also verifies overall is in [0, 1] — aggregate.py uses it directly
    # as a float in weighted calculations.
    # -----------------------------------------------------------
    def test_judge_returns_expected_keys(self, sample_scorer_cfg):
        """judge_score must return dict with factual_accuracy, overall, etc."""
        from scorer.judge import judge_score

        # Simulate a well-formed judge LLM response as a JSON string
        mock_response_text = (
            '{"factual_accuracy": 0.9, "instruction_following": 0.85, '
            '"safety": 1.0, "overall": 0.92, "reasoning": "Good response."}'
        )

        # MagicMock auto-creates the .choices[0].message.content chain
        mock_completion         = MagicMock()
        mock_completion.choices[0].message.content = mock_response_text

        # patch() replaces litellm.completion inside scorer/judge.py
        # for the duration of this test only — restored afterwards
        with patch("scorer.judge.completion", return_value=mock_completion):
            result = judge_score(
                prompt    = "What is the capital of France?",
                reference = "The capital of France is Paris.",
                response  = "Paris is the capital of France.",
                cfg       = sample_scorer_cfg,
            )

        assert "factual_accuracy"      in result
        assert "instruction_following" in result
        assert "safety"                in result
        assert "overall"               in result
        assert 0.0 <= result["overall"] <= 1.0


    # -----------------------------------------------------------
    # test_judge_handles_missing_overall
    # -----------------------------------------------------------
    # Tests the fallback logic in judge.py that computes 'overall'
    # when the judge LLM omits it from the response.
    #
    # Some LLM models (particularly smaller ones or those with strict
    # token limits) occasionally drop the 'overall' key despite being
    # explicitly asked for it. The fallback computes:
    #   overall = (factual_accuracy + instruction_following + safety) / 3
    #
    # This test verifies:
    #   1. No KeyError is raised when 'overall' is missing
    #   2. The computed fallback value matches the expected arithmetic
    #
    # Expected calculation:
    #   (0.9 + 0.8 + 1.0) / 3 = 0.9 (rounded to 4 decimal places)
    # Tolerance of 0.001 for floating point precision.
    # -----------------------------------------------------------
    def test_judge_handles_missing_overall(self, sample_scorer_cfg):
        """If judge omits 'overall', it should be computed from other axes."""
        from scorer.judge import judge_score

        # Response JSON deliberately missing the 'overall' key
        mock_text = (
            '{"factual_accuracy": 0.9, "instruction_following": 0.8, '
            '"safety": 1.0, "reasoning": "Fine response."}'
        )

        mock_completion         = MagicMock()
        mock_completion.choices[0].message.content = mock_text

        with patch("scorer.judge.completion", return_value=mock_completion):
            result = judge_score(
                prompt    = "Test prompt here for validation",
                reference = "Reference response text here",
                response  = "Candidate response text here",
                cfg       = sample_scorer_cfg,
            )

        # 'overall' must be present even though the judge didn't include it
        assert "overall" in result

        # Verify the fallback arithmetic: (0.9 + 0.8 + 1.0) / 3 = 0.9
        expected = round((0.9 + 0.8 + 1.0) / 3, 4)
        assert abs(result["overall"] - expected) < 0.001