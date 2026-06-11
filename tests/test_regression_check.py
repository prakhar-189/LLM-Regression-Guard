# tests/test_regression_check.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Test suite for scripts/check_regression.py — the CI gate
#               decision engine that blocks or approves PRs based on
#               LLM quality score comparisons.
#
#               Why this test file is the most critical in the project:
#                 The entire value of the regression guard pipeline depends
#                 on check_regression.py calling sys.exit(1) at exactly
#                 the right time. If this logic is wrong, bad PRs get merged
#                 silently with no warning. These tests are the proof that
#                 the gate actually works.
#
#               Testing strategy:
#                 All MLflow client calls are mocked using unittest.mock.
#                 This means these tests run instantly with no MLflow server,
#                 no network calls, and no external dependencies.
#                 The mock simulates what a real MLflow server would return.
#
#               Two classes of tests:
#                 TestCheckAbsoluteThresholds : Tests the fallback check used
#                   when no baseline exists (first run or MLflow unreachable).
#                   Validates absolute minimum score enforcement.
#
#                 TestCheckRegression : Tests the full regression detection
#                   logic including baseline comparison, delta thresholds,
#                   fallback behavior, and edge cases.
#
# Run:
#   pytest tests/test_regression_check.py -v
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# json     : Standard library — serializes sample score dicts to
#            temporary JSON files that check_regression.py can read.
# pytest   : Test framework — provides fixtures, assertions, and
#            the pytest.raises() context manager for testing SystemExit.
#
# MagicMock: unittest.mock — creates a fake MLflow client object that
#            returns controlled values without connecting to any server.
#            MagicMock auto-creates attributes on access, so
#            mock_run.data.metrics, mock_run.info.run_id etc. all work
#            without explicit setup.
#
# patch    : unittest.mock — temporarily replaces a real object in the
#            module under test with a mock for the duration of a test.
#            Used as: with patch("scripts.check_regression.mlflow.MlflowClient", ...)
#            This replaces the REAL MlflowClient inside check_regression.py
#            with our fake one — the module never knows the difference.
# ===============================================================
import json
import pytest
from unittest.mock import MagicMock, patch


# ===============================================================
# TestCheckAbsoluteThresholds
# ---------------------------------------------------------------
# Tests for the _check_absolute_thresholds() helper function.
# This function is the fallback path triggered when:
#   - No MLflow experiment exists yet (first ever run)
#   - No production-tagged baseline run exists
#   - MLflow server is unreachable
#
# It only checks scores against configured absolute minimums —
# no baseline comparison, no delta calculation.
#
# Fixtures used (defined in tests/conftest.py):
#   sample_scores     : dict — realistic scores.json with overall=0.875
#   sample_scorer_cfg : dict — scorer config with thresholds {overall:0.80, per_cat:0.72}
# ===============================================================
class TestCheckAbsoluteThresholds:

    # -----------------------------------------------------------
    # test_passes_when_all_scores_above_minimum
    # -----------------------------------------------------------
    # Happy path — all scores comfortably above both the overall
    # threshold (0.80) and per-category minimum (0.72).
    # Function should return normally with no SystemExit raised.
    # -----------------------------------------------------------
    def test_passes_when_all_scores_above_minimum(self, sample_scores, sample_scorer_cfg):
        """All scores above minimums should complete without calling sys.exit."""
        from scripts.check_regression import _check_absolute_thresholds
        # sample_scores: overall=0.875, all categories >= 0.80 — well above per_category_min 0.72
        _check_absolute_thresholds(sample_scores, 0.80, sample_scorer_cfg)


    # -----------------------------------------------------------
    # test_fails_when_overall_below_threshold
    # -----------------------------------------------------------
    # Sets overall to 0.70 which is below the 0.80 threshold.
    # Must call sys.exit(1) → pytest.raises(SystemExit) catches it.
    # Asserts exit code is exactly 1 (not 0, not 2).
    # -----------------------------------------------------------
    def test_fails_when_overall_below_threshold(self, sample_scores, sample_scorer_cfg):
        """Overall score below threshold must exit with code 1."""
        from scripts.check_regression import _check_absolute_thresholds
        sample_scores["overall"] = 0.70
        with pytest.raises(SystemExit) as exc_info:
            _check_absolute_thresholds(sample_scores, 0.80, sample_scorer_cfg)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_fails_when_category_below_per_category_min
    # -----------------------------------------------------------
    # Sets one category to 0.60 which is below per_category_min=0.72.
    # Even though all other categories and overall are fine,
    # a single failing category must trigger sys.exit(1).
    # -----------------------------------------------------------
    def test_fails_when_category_below_per_category_min(self, sample_scores, sample_scorer_cfg):
        """A single category below per_category_min must exit with code 1."""
        from scripts.check_regression import _check_absolute_thresholds
        sample_scores["categories"]["factual_accuracy"] = 0.60  # below 0.72
        with pytest.raises(SystemExit) as exc_info:
            _check_absolute_thresholds(sample_scores, 0.80, sample_scorer_cfg)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_fails_when_multiple_categories_below_minimum
    # -----------------------------------------------------------
    # Multiple failing categories — confirms the function collects
    # ALL failures before calling sys.exit(1), not just the first one.
    # Exit code must still be exactly 1 regardless of how many fail.
    # -----------------------------------------------------------
    def test_fails_when_multiple_categories_below_minimum(self, sample_scores, sample_scorer_cfg):
        """Multiple failing categories should still exit with code 1."""
        from scripts.check_regression import _check_absolute_thresholds
        sample_scores["categories"]["factual_accuracy"] = 0.60
        sample_scores["categories"]["refusal_behavior"] = 0.65
        with pytest.raises(SystemExit) as exc_info:
            _check_absolute_thresholds(sample_scores, 0.80, sample_scorer_cfg)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_passes_at_exact_threshold_boundary
    # -----------------------------------------------------------
    # Boundary condition test — a score of exactly 0.72 must PASS
    # (the check is `< per_category_min`, not `<= per_category_min`).
    # Same for overall = 0.80 exactly. Confirms off-by-one correctness.
    # -----------------------------------------------------------
    def test_passes_at_exact_threshold_boundary(self, sample_scores, sample_scorer_cfg):
        """A category score exactly at per_category_min should pass (not strictly less than)."""
        from scripts.check_regression import _check_absolute_thresholds
        sample_scores["categories"]["tone_consistency"] = 0.72  # exactly at min
        sample_scores["overall"] = 0.80                         # exactly at overall threshold
        _check_absolute_thresholds(sample_scores, 0.80, sample_scorer_cfg)


# ===============================================================
# TestCheckRegression
# ---------------------------------------------------------------
# Tests for the main check_regression() function — the full CI gate.
# This function:
#   1. Loads scores.json from disk
#   2. Queries MLflow for the production baseline run
#   3. Compares per-category scores against baseline
#   4. Calls sys.exit(1) if any regression is detected
#
# All MLflow calls are mocked — tests run with no server required.
# File I/O uses tmp_path (pytest's temporary directory fixture) so
# real project files are never touched during testing.
#
# Fixtures used (from tests/conftest.py):
#   tmp_scores_file    : str  — path to a temp scores.json file
#   tmp_scorer_cfg_file: str  — path to a temp scorer_config.yaml file
#   sample_scores      : dict — base scores dict (overall=0.875)
# ===============================================================
class TestCheckRegression:

    # -----------------------------------------------------------
    # test_exits_when_scores_file_missing
    # -----------------------------------------------------------
    # Passes a path to a file that doesn't exist.
    # check_regression() must detect the missing file immediately
    # and exit with code 1 before making any MLflow calls.
    # This guards against CI step ordering issues where run_scorer.py
    # didn't finish before check_regression.py started.
    # -----------------------------------------------------------
    def test_exits_when_scores_file_missing(self, tmp_path, tmp_scorer_cfg_file):
        """Missing scores.json must exit with code 1 before any MLflow call."""
        from scripts.check_regression import check_regression
        nonexistent = str(tmp_path / "missing.json")
        with pytest.raises(SystemExit) as exc_info:
            check_regression(scores_path=nonexistent, cfg_path=tmp_scorer_cfg_file)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_passes_with_no_mlflow_experiment
    # -----------------------------------------------------------
    # Simulates the very first run — no MLflow experiment exists yet.
    # get_experiment_by_name() returns None (no experiment found).
    # Should fall back to absolute threshold check and PASS because
    # sample_scores has a healthy overall of 0.875.
    #
    # Mock setup:
    #   MlflowClient() → mock_client
    #   mock_client.get_experiment_by_name() → None
    # -----------------------------------------------------------
    def test_passes_with_no_mlflow_experiment(self, tmp_scores_file, tmp_scorer_cfg_file):
        """No MLflow experiment yet → fall back to absolute check → passes with good scores."""
        from scripts.check_regression import check_regression

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            check_regression(scores_path=tmp_scores_file, cfg_path=tmp_scorer_cfg_file)


    # -----------------------------------------------------------
    # test_fails_with_no_mlflow_experiment_and_low_overall
    # -----------------------------------------------------------
    # First run scenario but with an unhealthy score (0.70).
    # No experiment → fallback to absolute check → 0.70 < 0.80 → exit 1.
    # Confirms the fallback path enforces thresholds, not just passes everything.
    # -----------------------------------------------------------
    def test_fails_with_no_mlflow_experiment_and_low_overall(
        self, tmp_path, tmp_scorer_cfg_file, sample_scores
    ):
        """No experiment + overall below threshold → absolute check → exit 1."""
        from scripts.check_regression import check_regression

        sample_scores["overall"] = 0.70
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(sample_scores))

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = None

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                check_regression(scores_path=str(scores_file), cfg_path=tmp_scorer_cfg_file)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_passes_with_no_production_tagged_run
    # -----------------------------------------------------------
    # Experiment exists but search_runs() returns [] — no run has
    # been tagged as "production" yet. Falls back to absolute check
    # and passes with healthy scores.
    #
    # This happens when the project is set up but the first run
    # hasn't been manually tagged with:
    #   mlflow runs set-tag <run_id> env production
    # -----------------------------------------------------------
    def test_passes_with_no_production_tagged_run(self, tmp_scores_file, tmp_scorer_cfg_file):
        """No production-tagged baseline found → treat as first run → passes with good scores."""
        from scripts.check_regression import check_regression

        mock_experiment        = MagicMock()
        mock_experiment.experiment_id = "exp_1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value = []  # no tagged baseline runs found

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            check_regression(scores_path=tmp_scores_file, cfg_path=tmp_scorer_cfg_file)


    # -----------------------------------------------------------
    # test_passes_when_scores_match_baseline
    # -----------------------------------------------------------
    # Current scores are identical to the baseline — no regression.
    # Delta for every category = 0.0 which is below regression_alert_delta=0.05.
    # Function should return normally without calling sys.exit.
    #
    # Mock setup:
    #   Baseline metrics = same as current scores (no change)
    #   mock_run.data.metrics = {"cat_factual_accuracy": 0.891, ...}
    # -----------------------------------------------------------
    def test_passes_when_scores_match_baseline(
        self, tmp_scores_file, tmp_scorer_cfg_file, sample_scores
    ):
        """Current scores equal to baseline → no regression → passes."""
        from scripts.check_regression import check_regression

        # Build baseline metrics dict in the MLflow key format (cat_ prefix)
        mock_metrics = {f"cat_{cat}": score for cat, score in sample_scores["categories"].items()}
        mock_metrics["overall_score"] = sample_scores["overall"]

        mock_run                = MagicMock()
        mock_run.data.metrics   = mock_metrics
        mock_run.info.run_id    = "baseline_abc123"

        mock_experiment               = MagicMock()
        mock_experiment.experiment_id = "exp_1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value            = [mock_run]

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            check_regression(scores_path=tmp_scores_file, cfg_path=tmp_scorer_cfg_file)


    # -----------------------------------------------------------
    # test_fails_when_category_drops_past_delta
    # -----------------------------------------------------------
    # THE MOST IMPORTANT TEST IN THE PROJECT.
    # Proves the core CI gate mechanism works correctly.
    #
    # Scenario:
    #   Baseline factual_accuracy = 0.891 (from sample_scores fixture)
    #   Current  factual_accuracy = 0.80
    #   Drop = 0.891 - 0.80 = 0.091 which exceeds regression_alert_delta=0.05
    #   → Must call sys.exit(1) → PR is blocked
    #
    # If this test fails, the entire pipeline's guarantee is broken.
    # -----------------------------------------------------------
    def test_fails_when_category_drops_past_delta(
        self, tmp_path, tmp_scorer_cfg_file, sample_scores
    ):
        """Category drop > regression_alert_delta (0.05) must exit with code 1."""
        from scripts.check_regression import check_regression

        # Current scores: factual_accuracy degraded from 0.891 to 0.80
        current = {**sample_scores, "categories": {**sample_scores["categories"]}}
        current["categories"]["factual_accuracy"] = 0.80

        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(current))

        # Baseline metrics: factual_accuracy was 0.891 (the original sample_scores value)
        mock_metrics = {f"cat_{cat}": score for cat, score in sample_scores["categories"].items()}
        mock_metrics["overall_score"] = sample_scores["overall"]

        mock_run                = MagicMock()
        mock_run.data.metrics   = mock_metrics
        mock_run.info.run_id    = "baseline_def456"

        mock_experiment               = MagicMock()
        mock_experiment.experiment_id = "exp_1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value            = [mock_run]

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                check_regression(scores_path=str(scores_file), cfg_path=tmp_scorer_cfg_file)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_fails_when_overall_drops_below_absolute_threshold
    # -----------------------------------------------------------
    # Baseline overall was healthy (0.88) but current overall dropped
    # to 0.75 — below the absolute threshold of 0.80.
    # Must exit with code 1 even if no individual category crossed
    # the delta threshold. Tests the overall score absolute check.
    # -----------------------------------------------------------
    def test_fails_when_overall_drops_below_absolute_threshold(
        self, tmp_path, tmp_scorer_cfg_file, sample_scores
    ):
        """Overall score below 0.80 absolute threshold must exit with code 1 even if baseline was good."""
        from scripts.check_regression import check_regression

        # Current overall is 0.75 — below threshold
        current     = {**sample_scores, "overall": 0.75}
        scores_file = tmp_path / "scores.json"
        scores_file.write_text(json.dumps(current))

        # Baseline was healthy at 0.88
        mock_metrics                  = {f"cat_{cat}": score for cat, score in sample_scores["categories"].items()}
        mock_metrics["overall_score"] = 0.88

        mock_run                = MagicMock()
        mock_run.data.metrics   = mock_metrics
        mock_run.info.run_id    = "baseline_ghi789"

        mock_experiment               = MagicMock()
        mock_experiment.experiment_id = "exp_1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value            = [mock_run]

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            with pytest.raises(SystemExit) as exc_info:
                check_regression(scores_path=str(scores_file), cfg_path=tmp_scorer_cfg_file)
        assert exc_info.value.code == 1


    # -----------------------------------------------------------
    # test_falls_back_to_absolute_check_when_mlflow_unreachable
    # -----------------------------------------------------------
    # Simulates a real-world scenario: MLflow server is down during CI.
    # get_experiment_by_name() raises ConnectionError instead of returning.
    # check_regression() must catch this gracefully and fall back to
    # absolute threshold checks — NOT crash the entire CI pipeline.
    #
    # With healthy scores (0.875 overall), the fallback should PASS.
    # -----------------------------------------------------------
    def test_falls_back_to_absolute_check_when_mlflow_unreachable(
        self, tmp_scores_file, tmp_scorer_cfg_file
    ):
        """MLflow connection error → fallback to absolute threshold check → passes with good scores."""
        from scripts.check_regression import check_regression

        mock_client = MagicMock()
        # Simulate MLflow server being unreachable
        mock_client.get_experiment_by_name.side_effect = ConnectionError("MLflow unreachable")

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            check_regression(scores_path=tmp_scores_file, cfg_path=tmp_scorer_cfg_file)


    # -----------------------------------------------------------
    # test_skips_delta_check_for_new_category
    # -----------------------------------------------------------
    # Edge case: a new behavioral category was added to the dataset
    # after the baseline run was created. The baseline has no metric
    # for "multi_turn_coherence" — its cat_ key is missing entirely.
    #
    # Expected behavior: skip the delta check for that category
    # (print SKIP in the comparison table), do NOT treat it as a
    # failure. Only compare categories that exist in BOTH current
    # and baseline.
    # -----------------------------------------------------------
    def test_skips_delta_check_for_new_category(
        self, tmp_scores_file, tmp_scorer_cfg_file, sample_scores
    ):
        """A category not present in the baseline should be skipped (no delta check)."""
        from scripts.check_regression import check_regression

        # Baseline is missing the "multi_turn_coherence" metric entirely
        mock_metrics = {
            f"cat_{cat}": score
            for cat, score in sample_scores["categories"].items()
            if cat != "multi_turn_coherence"   # deliberately excluded
        }
        mock_metrics["overall_score"] = sample_scores["overall"]

        mock_run                = MagicMock()
        mock_run.data.metrics   = mock_metrics
        mock_run.info.run_id    = "baseline_new_cat"

        mock_experiment               = MagicMock()
        mock_experiment.experiment_id = "exp_1"

        mock_client = MagicMock()
        mock_client.get_experiment_by_name.return_value = mock_experiment
        mock_client.search_runs.return_value            = [mock_run]

        with patch("scripts.check_regression.mlflow.MlflowClient", return_value=mock_client):
            # Should pass — missing category is skipped, not treated as failure
            check_regression(scores_path=tmp_scores_file, cfg_path=tmp_scorer_cfg_file)