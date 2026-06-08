# scorer/run_scorer.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : End-to-end orchestrator for a complete scoring run.
#               This is the single command that powers the CI quality gate.
#
#               Pipeline position — Stage 2 (Scorer):
#                 Called directly by .github/workflows/quality_gate.yml.
#                 Can also be run manually for local debugging.
#
#               Orchestration steps:
#                 1. Load golden dataset (data/eval/golden_dataset.jsonl)
#                 2. Open an MLflow run to track this evaluation
#                 3. For every example in the dataset:
#                      a. run_inference()   → get model's response
#                      b. compute_metrics() → BERTScore + ROUGE-L
#                      c. judge_score()     → judge LLM rubric scores
#                 4. aggregate_scores()  → per-category + overall summary
#                 5. Log all metrics + params to MLflow
#                 6. Write summary to scores_output/scores.json
#
#               Output file scores_output/scores.json is then read by
#               scripts/check_regression.py for the pass/fail decision.
#
# Usage:
#   python scorer/run_scorer.py
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# json      : Standard library — loads JSONL dataset, writes scores.json.
# os        : Standard library — reads MLFLOW_TRACKING_URI env variable.
# sys       : Standard library — modifies sys.path for package imports.
# yaml      : PyYAML — reads scorer_config.yaml and model_version.yaml.
# Path      : pathlib — cross-platform file path handling.
# datetime  : Standard library — run naming and timestamp in output file.
#
# mlflow    : MLflow tracking client.
#               - mlflow.start_run()    : Opens a new tracked experiment run.
#               - mlflow.log_metric()   : Records a float value to the run.
#               - mlflow.log_param()    : Records a string/int config value.
#               - mlflow.set_tag()      : Attaches metadata labels to the run.
#
# load_dotenv  : python-dotenv — loads .env API keys into os.environ.
#
# run_inference   : scorer/inference.py  — calls the LLM under test.
# compute_metrics : scorer/metrics.py    — BERTScore + ROUGE-L computation.
# judge_score     : scorer/judge.py      — judge LLM structured rubric.
# aggregate_scores: scorer/aggregate.py  — rolls up results into summary.
# ===============================================================
import json
import os
import sys
import yaml
from pathlib  import Path
from datetime import datetime

import mlflow
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path so scorer package is importable when
# run directly as a script (not as a module).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scorer.inference  import run_inference
from scorer.metrics    import compute_metrics
from scorer.judge      import judge_score
from scorer.aggregate  import aggregate_scores


# ===============================================================
# run_scoring_pipeline
# ---------------------------------------------------------------
# Runs the complete evaluation pipeline from dataset to scores.json.
#
# Parameters:
#   dataset_path   (str) : Path to golden dataset JSONL file.
#   scorer_cfg_path(str) : Path to scorer configuration YAML.
#   output_path    (str) : Path to write the final scores JSON file.
#
# Returns:
#   dict : The summary scores dict (same content as scores.json).
#          Returned to allow programmatic use in tests.
#
# Side effects:
#   - Creates an MLflow run in the configured experiment.
#   - Writes scores_output/scores.json (overwriting any previous run).
#   - Prints a progress log to stdout (visible in CI run logs).
# ===============================================================
def run_scoring_pipeline(
    dataset_path    : str = "data/eval/golden_dataset.jsonl",
    scorer_cfg_path : str = "config/scorer_config.yaml",
    output_path     : str = "scores_output/scores.json",
) -> dict:

    print("=" * 55)
    print("LLM Regression Guard — Scoring Pipeline")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 : Load configuration files
    # ----------------------------------------------------------
    # scorer_config.yaml : thresholds, judge model, metric weights
    # model_version.yaml : which model is currently under test
    # ----------------------------------------------------------
    cfg        = yaml.safe_load(open(scorer_cfg_path))
    model_cfg  = yaml.safe_load(open("config/model_version.yaml"))
    curr_model = model_cfg.get("current_model", "gpt-4o")

    # ----------------------------------------------------------
    # Step 2 : Load the golden evaluation dataset
    # ----------------------------------------------------------
    # Each line is one JSON object (a GoldenExample).
    # We load as raw dicts here — Pydantic validation already
    # happened during build_dataset.py.
    # ----------------------------------------------------------
    dataset = [
        json.loads(line)
        for line in Path(dataset_path).read_text().strip().splitlines()
    ]
    print(f"\nDataset     : {len(dataset)} examples")
    print(f"Model       : {curr_model}")
    print(f"Judge model : {cfg['judge_model']}")
    print("-" * 55)

    # ----------------------------------------------------------
    # Step 3 : Configure MLflow tracking
    # ----------------------------------------------------------
    # MLFLOW_TRACKING_URI points to the MLflow server.
    # Falls back to localhost:5000 if env var is not set.
    # The experiment name groups all scoring runs together.
    # ----------------------------------------------------------
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    results = []

    # ----------------------------------------------------------
    # Step 4 : Score every example in the golden dataset
    # ----------------------------------------------------------
    # Each example goes through three scoring steps:
    #   a) run_inference()   — get model response for the prompt
    #   b) compute_metrics() — BERTScore F1 + ROUGE-L (deterministic)
    #   c) judge_score()     — judge LLM rubric (factual/instruction/safety)
    # ----------------------------------------------------------
    with mlflow.start_run(
        run_name=f"score-{curr_model}-{datetime.now().strftime('%Y%m%d-%H%M')}"
    ):

        # Tag the run with metadata for later filtering
        mlflow.set_tag("model_version", curr_model)
        mlflow.set_tag("dataset_path",  dataset_path)
        mlflow.set_tag("prompt_template_version",
                       model_cfg.get("prompt_template_version", "unknown"))

        for i, example in enumerate(dataset, 1):
            print(f"[{i:03d}/{len(dataset)}] {example['category']:<30} id={example['id']}")

            # a) Get the model's response for this prompt
            response = run_inference(example["prompt"], model=curr_model)

            # b) Deterministic metrics — no LLM needed, always reproducible
            metrics  = compute_metrics(response, example["reference_response"])

            # c) Judge LLM rubric — structured quality evaluation
            j_score  = judge_score(
                prompt    = example["prompt"],
                reference = example["reference_response"],
                response  = response,
                cfg       = cfg,
            )

            # Collect everything for aggregation
            results.append({
                "id"                 : example["id"],
                "category"           : example["category"],
                "prompt"             : example["prompt"],
                "reference_response" : example["reference_response"],
                "model_response"     : response,
                "metrics"            : metrics,
                "judge_score"        : j_score,
            })

        # ----------------------------------------------------------
        # Step 5 : Aggregate per-example results into summary
        # ----------------------------------------------------------
        # aggregate_scores() computes per-category averages and the
        # weighted overall score using weights from scorer_config.yaml.
        # ----------------------------------------------------------
        summary = aggregate_scores(results, cfg)
        summary["model_version"]   = curr_model
        summary["run_timestamp"]   = datetime.now().isoformat()
        summary["total_examples"]  = len(results)

        # ----------------------------------------------------------
        # Step 6 : Log metrics and params to MLflow
        # ----------------------------------------------------------
        # Per-category metrics are prefixed with "cat_" to allow
        # scripts/check_regression.py to fetch them by key pattern.
        # ----------------------------------------------------------
        for cat, score in summary["categories"].items():
            mlflow.log_metric(f"cat_{cat}", score)
        mlflow.log_metric("overall_score", summary["overall"])

        # Log configuration params for experiment reproducibility
        mlflow.log_param("model",        curr_model)
        mlflow.log_param("judge_model",  cfg["judge_model"])
        mlflow.log_param("dataset_size", len(dataset))

        # ----------------------------------------------------------
        # Step 7 : Write scores.json output artifact
        # ----------------------------------------------------------
        # This file is read by scripts/check_regression.py in the
        # next CI step to compare against the production baseline.
        # ----------------------------------------------------------
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Print final summary to CI log
        print("-" * 55)
        print("\nSCORING COMPLETE")
        print(f"Overall score : {summary['overall']}")
        print(f"\nPer-category scores:")
        for cat, score in summary["categories"].items():
            status = "PASS" if score >= cfg["thresholds"]["per_category_min"] else "FAIL"
            print(f"  {cat:<30} {score:.4f}  [{status}]")
        print(f"\nResults written to : {output_path}")
        print("=" * 55)

    return summary


# ===============================================================
# Script entry point
# ---------------------------------------------------------------
# Allows this file to be run directly:
#   python scorer/run_scorer.py
#
# The CI workflow (quality_gate.yml) calls it exactly this way.
# ===============================================================
if __name__ == "__main__":
    run_scoring_pipeline()