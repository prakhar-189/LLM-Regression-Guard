# scripts/check_regression.py
# ----------------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : This script checks for regressions in LLM performance by comparing current scores with a baseline.
# ----------------------------------------------------------------------


# ======================================================================
# Imports
# -----------------------------------------------------------------
# json : A module for working with JSON data in Python.
# sys : A module that provides access to some variables used or maintained by the Python interpreter.
# yaml : A module for working with YAML data in Python.
# MlflowClient : A class from the MLflow library used to interact with the MLflow tracking server.
# ======================================================================
import json
import sys
import yaml
from mlflow.tracking import MlflowClient


# =======================================================================
# load_config Function
# -----------------------------------------------------------------
# This function loads the configuration for the regression check from a YAML file.
# It reads the 'scorer_config.yaml' file and returns the configuration as a dictionary.
# =======================================================================
def load_config():
    with open('config/scorer_config.yaml', 'r') as f:
        return yaml.safe_load(f)


# =======================================================================
# get_baseline_scores Function
# -----------------------------------------------------------------
# This function retrieves the baseline scores from the MLflow tracking server.
# It looks for the most recent run in the "LLM_Regression_Guard" experiment that is tagged as 'production'.
# If a baseline run is found, it returns the metrics from that run. If no baseline is found, it returns None.
# =======================================================================
def get_baseline_scores(client: MlflowClient):
    experiment = client.get_experiment_by_name("LLM_Regression_Guard")
    runs = client.search_runs(
        experiment_ids = [experiment.experiment_id],
        filter_string = "tags.env = 'production'",
        order_by = ["start_time DESC"],
        max_results = 1
    )
    if not runs:
        print("Warning: No baseline found. Passing by default.")
        return None
    
    return runs[0].data.metrics


# =======================================================================
# main Function
# -----------------------------------------------------------------
# This is the main function that performs the regression check.
# It loads the configuration, retrieves the current scores from a JSON file, and gets the baseline scores from MLflow.
# It then compares the current scores with the baseline scores for each category and checks if any category has regressed beyond the defined threshold.
# If a regression is detected, it prints the details and fails the quality gate.
# If all categories pass without regression, it prints a success message and passes the quality gate.
# If no regressions are detected, it prints a success message and passes the quality gate.
# =======================================================================
def main():
    config = load_config()
    delta_threshold = config.get('regression_alert_delta', 0.05)

    with open('scores_output/scores.json', 'r') as f:
        current_scores = json.load(f)

    client = MlflowClient()
    baseline_metrics = get_baseline_scores(client)

    if not baseline_metrics:
        sys.exit(0)

    failed = False
    print("--- LLM REGRESSION CHECK ---")

    # Loop through each category in the current scores and compare it with the baseline scores.
    for category, current_val in current_scores['categories'].items():
        metric_name = f"{category}_score"
        baseline_val = baseline_metrics.get(metric_name, 0.0)
        delta = baseline_val - current_val

        if delta > delta_threshold:
            print(f"REGRESSION DETECTED IN {category}")
            print(f"Baseline: {baseline_val:.3f} | Current: {current_val:.3f} | Drop : {delta:.3f}")
            failed = True
        else:
            print(f"{category} passed (Current: {current_val:.3f})")

    if failed:
        print("\nQuality gate failed. Fix regressions before merging")
        sys.exit(1)

    print("\nQuality gate passed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()    