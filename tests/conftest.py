# tests/conftest.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Shared pytest fixtures used across all test files
#               in the tests/ package.
#
#               What is a fixture?
#                 A pytest fixture is a reusable setup function decorated
#                 with @pytest.fixture. When a test function declares a
#                 fixture name as a parameter, pytest automatically calls
#                 the fixture and injects its return value before the test
#                 runs. This eliminates copy-pasting identical setup code
#                 across multiple test files.
#
#               Why conftest.py specifically?
#                 pytest automatically discovers conftest.py in the tests/
#                 directory and makes all fixtures defined here available
#                 to every test file in that directory — no imports needed.
#                 test_schema.py, test_scorer.py, and test_regression_check.py
#                 all use fixtures from this file transparently.
#
#               Fixtures defined here:
#                 valid_example_dict    : A correct GoldenExample dict.
#                                         Used by test_schema.py.
#                 sample_dataset        : A list of 25 GoldenExample objects
#                                         (5 per category). Used for dataset
#                                         validation tests.
#                 sample_scorer_cfg     : A complete scorer config dict
#                                         matching scorer_config.yaml structure.
#                                         Used by test_scorer.py and
#                                         test_regression_check.py.
#                 sample_scores         : A realistic scores.json dict with
#                                         healthy scores. Used by
#                                         test_regression_check.py.
#                 sample_results        : A list of 20 per-example result dicts
#                                         (4 per category). Used by
#                                         TestAggregateScores in test_scorer.py.
#                 tmp_scores_file       : Writes sample_scores to a temp file
#                                         and returns the path. Used by
#                                         test_regression_check.py for file I/O tests.
#                 tmp_scorer_cfg_file   : Writes sample_scorer_cfg to a temp YAML
#                                         file and returns the path. Used by
#                                         test_regression_check.py.
#
#               Design principles:
#                 - No file I/O except in tmp_* fixtures (which use pytest's
#                   tmp_path for automatic cleanup after each test).
#                 - No API calls, no network, no external services.
#                 - All values are hardcoded to known quantities so test
#                   assertions can use exact comparisons.
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# json     : Standard library — serializes sample_scores dict to
#            JSON string for writing to tmp_scores_file.
# pytest   : Test framework — provides the @pytest.fixture decorator
#            and the tmp_path built-in fixture used by tmp_* fixtures.
# Path     : pathlib — used in tmp_* fixtures to create temp files
#            via pytest's tmp_path directory (auto-cleaned after each test).
# ===============================================================
import json
import pytest
from pathlib import Path


# ===============================================================
# valid_example_dict
# ---------------------------------------------------------------
# Returns a plain Python dict representing a single, fully valid
# GoldenExample entry — the minimum correct input to GoldenExample(**dict).
#
# Used by:
#   test_schema.py : TestGoldenExampleValid tests pass this to
#                    GoldenExample(**valid_example_dict) to verify
#                    that correct inputs are accepted without error.
#
# Values chosen:
#   - id       : "test001" — short unique identifier
#   - prompt   : A real question well over 10 characters
#   - reference: A real factual answer
#   - category : "factual_accuracy" — the simplest behavioral category
#   - source   : "manual" — the default source value
#
# Returns:
#   dict : Plain Python dict (not a GoldenExample object).
#          Returned as dict so tests can mutate specific fields
#          to test partial or invalid inputs without affecting
#          other tests (each fixture call creates a fresh dict).
# ===============================================================
@pytest.fixture
def valid_example_dict():
    """A fully valid golden example as a dict."""
    return {
        "id"                 : "test001",
        "prompt"             : "What is the capital of France?",
        "reference_response" : "The capital of France is Paris.",
        "category"           : "factual_accuracy",
        "source"             : "manual",
    }


# ===============================================================
# sample_dataset
# ---------------------------------------------------------------
# Returns a list of 25 validated GoldenExample objects —
# 5 examples per behavioral category.
#
# Used by:
#   Any test that needs a complete, validated dataset list
#   (e.g. testing validate_dataset() from dataset/validator.py).
#
# Construction:
#   Iterates over all 5 categories × 5 examples = 25 total.
#   Each example gets a unique ID from the category prefix + indices.
#   Prompts include the indices to ensure uniqueness across examples.
#   All prompts are constructed to be longer than 10 characters
#   to pass the prompt_not_empty validator in schema.py.
#
# Parameters:
#   valid_example_dict : Injected by pytest but not directly used here.
#                        Declared to signal fixture dependency for ordering.
#
# Returns:
#   List[GoldenExample] : 25 validated Pydantic model instances.
# ===============================================================
@pytest.fixture
def sample_dataset(valid_example_dict):
    """A small list of valid GoldenExample objects."""
    from dataset.schema import GoldenExample

    examples   = []
    categories = [
        "factual_accuracy",
        "refusal_behavior",
        "instruction_following",
        "tone_consistency",
        "multi_turn_coherence",
    ]

    for i, cat in enumerate(categories):
        for j in range(5):
            examples.append(GoldenExample(
                id                 = f"{cat[:3]}{i:02d}{j:02d}",
                prompt             = f"Test prompt {i}{j} — long enough to pass validation",
                reference_response = f"Reference response for {cat} test {j}",
                category           = cat,
                source             = "test",
            ))

    return examples


# ===============================================================
# sample_scorer_cfg
# ---------------------------------------------------------------
# Returns a complete scorer configuration dict that mirrors the
# structure of config/scorer_config.yaml.
#
# Used by:
#   test_scorer.py           : TestAggregateScores and TestJudgeScore
#                              pass this as the 'cfg' parameter.
#   test_regression_check.py : TestCheckAbsoluteThresholds and
#                              TestCheckRegression use thresholds from here.
#
# Why in-memory instead of reading the real file?
#   Tests must not depend on the state of config files on disk.
#   If scorer_config.yaml is modified during development, tests using
#   the real file would break unpredictably. This fixture gives every
#   test a stable, known configuration regardless of file state.
#
# Key values:
#   overall threshold       : 0.80 — a score below this fails absolute check
#   per_category_min        : 0.72 — minimum acceptable per-category score
#   regression_alert_delta  : 0.05 — max drop vs baseline before CI blocks PR
#   metric_weights          : bertscore=0.30, rouge_l=0.20, judge=0.50
#   category weights        : sum to 1.0 (0.30+0.25+0.25+0.10+0.10)
#
# Returns:
#   dict : In-memory scorer config matching scorer_config.yaml structure.
# ===============================================================
@pytest.fixture
def sample_scorer_cfg():
    """Minimal scorer config for tests — no file I/O needed."""
    return {
        "judge_model"       : "gpt-4o",
        "judge_temperature" : 0.0,
        "thresholds"        : {
            "overall"                 : 0.80,
            "per_category_min"        : 0.72,
            "regression_alert_delta"  : 0.05,
        },
        "metric_weights"    : {
            "bertscore"   : 0.30,
            "rouge_l"     : 0.20,
            "judge_score" : 0.50,
        },
        "mlflow"            : {
            "experiment_name" : "llm-regression-guard",
            "production_tag"  : "production",
        },
        "categories"        : [
            {"name": "factual_accuracy",      "weight": 0.30},
            {"name": "refusal_behavior",      "weight": 0.25},
            {"name": "instruction_following", "weight": 0.25},
            {"name": "tone_consistency",      "weight": 0.10},
            {"name": "multi_turn_coherence",  "weight": 0.10},
        ],
    }


# ===============================================================
# sample_scores
# ---------------------------------------------------------------
# Returns a realistic scores.json dict representing a healthy
# scoring run with an overall score of 0.875.
#
# Used by:
#   test_regression_check.py : This dict is what check_regression.py
#                              reads from scores_output/scores.json.
#                              Tests mutate specific fields to simulate
#                              regressions (e.g. lowering overall to 0.70).
#
# Values chosen:
#   - All category scores > 0.80 (well above per_category_min=0.72)
#   - overall = 0.875 (above threshold of 0.80)
#   - Represents a "good baseline run" that CI would tag as production
#
# Important:
#   Tests that mutate this fixture receive a FRESH copy each time
#   because pytest creates a new dict instance per test invocation.
#   Mutations in one test do NOT bleed into other tests.
#
# Returns:
#   dict : Scores dict matching the structure written by run_scorer.py.
# ===============================================================
@pytest.fixture
def sample_scores():
    """A sample scores.json dict for regression check tests."""
    return {
        "model_version"  : "gpt-4o",
        "run_timestamp"  : "2026-06-07T10:00:00",
        "total_examples" : 20,
        "overall"        : 0.875,
        "categories"     : {
            "factual_accuracy"      : 0.891,
            "refusal_behavior"      : 0.912,
            "instruction_following" : 0.847,
            "tone_consistency"      : 0.823,
            "multi_turn_coherence"  : 0.801,
        },
    }


# ===============================================================
# sample_results
# ---------------------------------------------------------------
# Returns a list of 20 per-example result dicts — 4 examples per
# behavioral category — in the exact format produced by run_scorer.py.
#
# Used by:
#   test_scorer.py : TestAggregateScores passes this to
#                    aggregate_scores(results, cfg) to verify
#                    aggregation math and output structure.
#
# Structure of each result dict:
#   category    : str  — behavioral category name
#   metrics     : dict — {"bertscore_f1": float, "rouge_l": float}
#                         returned by scorer/metrics.py
#   judge_score : dict — {"factual_accuracy": float, "overall": float, ...}
#                         returned by scorer/judge.py
#
# Values chosen:
#   bertscore_f1 = 0.88, rouge_l = 0.75, judge overall = 0.88
#   These are realistic mid-range scores, not extremes.
#   All 20 examples have identical scores for predictable math.
#
# Parameters:
#   sample_scorer_cfg : Injected to signal ordering dependency.
#                       Not used directly in this fixture.
#
# Returns:
#   List[dict] : 20 result dicts (4 per category × 5 categories).
# ===============================================================
@pytest.fixture
def sample_results(sample_scorer_cfg):
    """A list of per-example result dicts for aggregate_scores tests."""
    results = []

    for cat in [
        "factual_accuracy",
        "refusal_behavior",
        "instruction_following",
        "tone_consistency",
        "multi_turn_coherence",
    ]:
        # 4 examples per category = 20 total
        for _ in range(4):
            results.append({
                "category"    : cat,
                "metrics"     : {
                    "bertscore_f1" : 0.88,
                    "rouge_l"      : 0.75,
                },
                "judge_score" : {
                    "factual_accuracy"      : 0.90,
                    "instruction_following" : 0.85,
                    "safety"                : 1.00,
                    "overall"               : 0.88,
                },
            })

    return results


# ===============================================================
# tmp_scores_file
# ---------------------------------------------------------------
# Writes sample_scores to a temporary JSON file and returns the
# file path as a string.
#
# Used by:
#   test_regression_check.py : check_regression() reads scores from
#                              a file path — this fixture provides a
#                              real file that exists on disk temporarily.
#
# Why tmp_path?
#   pytest's built-in tmp_path fixture creates a unique temporary
#   directory for each test invocation and automatically deletes it
#   after the test completes. This prevents test pollution between runs.
#
# Parameters:
#   tmp_path     : pytest built-in — unique temp dir for this test.
#   sample_scores: The scores dict fixture defined above.
#
# Returns:
#   str : Absolute path to the temporary scores.json file.
# ===============================================================
@pytest.fixture
def tmp_scores_file(tmp_path, sample_scores):
    """Write sample scores to a temp file and return its path."""
    scores_file = tmp_path / "scores.json"
    scores_file.write_text(json.dumps(sample_scores))
    return str(scores_file)


# ===============================================================
# tmp_scorer_cfg_file
# ---------------------------------------------------------------
# Writes sample_scorer_cfg to a temporary YAML file and returns
# the file path as a string.
#
# Used by:
#   test_regression_check.py : check_regression() reads its config
#                              from a YAML file path — this fixture
#                              provides a real file on disk temporarily.
#
# Why YAML and not JSON?
#   check_regression.py uses yaml.safe_load(open(cfg_path)) to read
#   the config — the file must be valid YAML. yaml.dump() converts
#   the Python dict to a properly formatted YAML string.
#
# Parameters:
#   tmp_path          : pytest built-in — unique temp dir for this test.
#   sample_scorer_cfg : The scorer config dict fixture defined above.
#
# Returns:
#   str : Absolute path to the temporary scorer_config.yaml file.
# ===============================================================
@pytest.fixture
def tmp_scorer_cfg_file(tmp_path, sample_scorer_cfg):
    """Write sample scorer config to a temp YAML file."""
    import yaml
    cfg_file = tmp_path / "scorer_config.yaml"
    cfg_file.write_text(yaml.dump(sample_scorer_cfg))
    return str(cfg_file)