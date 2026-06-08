# scorer/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'scorer' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from scorer.inference  import run_inference
#                   from scorer.metrics    import compute_metrics
#                   from scorer.judge      import judge_score
#                   from scorer.aggregate  import aggregate_scores
#
# Modules in this package:
#   inference.py   : LiteLLM wrapper — calls the LLM under test.
#   metrics.py     : Deterministic metrics — BERTScore and ROUGE-L.
#   judge.py       : Judge LLM rubric — structured quality evaluation.
#   aggregate.py   : Rolls up per-example scores into category/overall summary.
#   run_scorer.py  : Orchestrates a full scoring run end-to-end.
# --------------------------------------------------------------