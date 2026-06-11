# tests/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'tests' module.
#               Makes this directory a Python package so that pytest
#               can correctly discover and import test files, and so
#               that test files can import from the project packages
#               using absolute imports:
#                   from scripts.check_regression import check_regression
#                   from dataset.schema            import GoldenExample
#                   from scorer.aggregate          import aggregate_scores
#
#               Without this file, pytest may fail to resolve imports
#               across packages on some Python/OS configurations.
#
# Test files in this package:
#   conftest.py              : Shared pytest fixtures used across all
#                              test files (sample data, temp files, configs).
#   test_schema.py           : Tests for dataset/schema.py — Pydantic
#                              GoldenExample model validation.
#   test_scorer.py           : Tests for scorer components — metrics.py,
#                              aggregate.py, and judge.py with mocked LLM calls.
#   test_regression_check.py : Tests for scripts/check_regression.py —
#                              the CI gate decision engine with mocked MLflow.
# --------------------------------------------------------------