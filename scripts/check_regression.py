# scripts/check_regression.py
# ----------------------------------------------------------------
# Author       : Prakhar Srivastava
# Date         : 2026-06-08
# Description  : -> CI gate decision engine - the core of the quality pipeline.
#
#                -> Pipeline position - Stage 3 (CI/CD Gate):
#                       Called by .github/workflows/quality_gate.yml immediately
#                       after scorer/run_scorer.py completes
#
#                -> What it does:
#                       1. Loads current run scores from scores_output/scores.json
#                       2. Fetches the last production-tagged run from MLflow as the 
#                          quality baseline.
#                       3. Compares every per-category score against the baseline.
#                       4. Exits with code 0 (pass) or code 1 (fail).
#
#                -> The mechanism that blocks PRs:
#                       sys.exit(1) causes GitHub Actions to mark the CI job as 
#                       FAILED. Branch protection rules in the GitHub repo settings 
#                       require this job to pass before any PR can be merged - making
#                       it literally impossible to merge a quality regression without
#                       explicitly overriding the branch protection.
#
#                -> Regression is detected when:
#                       - Any category drops more than regression_alert_delta (0.05)
#                         below its baseline value, OR
#                       - The overall score fails below the absolute threshold (0.8).
# ----------------------------------------------------------------


# ================================================================
# Imports
# ------------------------------------------------------------
# json           : Standard Library - loads scores_output/scores.json
# sys            : Standard Library - sys.exit(1) is what fails the CI job.
# yaml           : PyYAML - reafs scorer_config.yaml for thresholds.
# Path           : pathlib - checks if scores.json exists before reading.
# mlflow         : MLflow tracking client
#                      - MLflowClient()           : Low-level API for querying runs.
#                      - get_experiment_by_name() : Finds the experiment by name.
#                      - search_runs()            : Queries runs with filter expressions
#                      - The filter "tags.env = 'production' " finds the last
#                        run explicitly tagged as the production baseline.  
# load_dotenv    : python-dotenv - loads MLFLOW_TRACKING_URI from .env
# ================================================================
import json
import sys
import yaml
from pathlib import Path
import mlflow
from dotenv import load_dotenv

load_dotenv()


# ================================================================
# check_regression
# -----------------------------------------------------------
# Main regression detection function.
# Compares the current scoring run against the production baseline
# & calls sys.exit(1) if any regression is detected.
#
# Parameters:
#      scores_path (str)  : Path to the current run's scores JSON file.
#                           Written by scorer/run_scorer.py.
#      cfg_path    (str)  : Path to scorer_config.yaml for threshold values.
#
# Returns:
#      None
# 
# Exits:
#      sys.exit(0) : All checks passed - PR is safe to merge.
#      sys.exit(1) : Regression detected - PR is blocked from merging.
# ================================================================
def check_regression(
        scores_path : str = "scores_output/scores.json",
        cfg_path    : str = "config/scorer_config.yaml",
) -> None:
    
    print("=" * 55)
    print("LLM Regression Guard - Regression Check")
    print("=" * 55)

    # ------------------------------------------------
    # Load configuration & current scores
    # ------------------------------------------------
    cfg = yaml.safe_load(open(cfg_path))

    # delta_threshold    : max allowed drop per categpry vs baseline (default 0.05)
    # overall_threshold  : min absolute overall score allowed (default 0.8)
    delta_threshold   = cfg["thresholds"]["regression_alert_delta"]
    overall_threshold = cfg["thresholds"]["overall"]

    # Guard : scores.json must exist (written by run_scorer.py before this step)
    if not Path(scores_path).exists():
        print(f"ERROR: '{scores_path}' not found. Run scorer/run_scorer.py first")
        sys.exit(1)

    current = json.load(open(scores_path))
    print(f"\nModel under test : {current.get('nodel_version', 'unknown')}")
    print(f"Current overall : {current['overall']}")

    # ------------------------------------------------
    # Fetch production baseline from Mlflow
    # ------------------------------------------------
    # We look for the most recnt run tagged with env = 'production'.
    # This tag is set manually after the first successful run:
    #     mlflow runs set-tag <run_id> env production
    # ------------------------------------------------
    client = mlflow.MlflowClient()

    try:
        experiment = client.get_experiment_by_name(
            cfg["mlflow"]["experiment_name"]
        )

        if experiment is None:
            #No experiment exists yet - this is the first run ever
            print("\nNo MLflow experiment found - treatinf as first run.")
            _check_absolute_thresholds(current, overall_threshold, cfg)
            return
        
        runs = client.search_runs(
            experiment_ids = [experiment.experiment_id],
            filter_string  = f"tags.env = '{cfg['mlflow']['production_tag']}'",
            order_by       = ["start_time DESC"],
            max_results    = 1,
        )

    except Exception as e:
        # MLfloe is unreachable - fall back to absolute threshold check
        print(f"\nMLflow unavailable : {e}")
        print("Falling back to absolute threshold check only.")
        _check_absolute_thresholds(current, overall_threshold, cfg)
        return
    
    if not runs:
        print("\nNo production-tagged baseline found — treating as first run.")
        _check_absolute_thresholds(current, overall_threshold, cfg)
        return
 
    # ----------------------------------------------------------
    # Compare current scores against production baseline
    # ----------------------------------------------------------
    baseline_run     = runs[0]
    baseline_metrics = baseline_run.data.metrics
    print(f"\nBaseline run ID   : {baseline_run.info.run_id}")
 
    # Print comparison table header
    print(f"\n{'Category':<30} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'Status':>8}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
 
    failures = []
 
    for cat, current_score in current["categories"].items():
 
        # MLflow metric keys for per-category scores are prefixed with "cat_"
        # (set by run_scorer.py with mlflow.log_metric(f"cat_{cat}", score))
        baseline_key   = f"cat_{cat}"
        baseline_score = baseline_metrics.get(baseline_key, None)
 
        if baseline_score is None:
            # This category didn't exist in the baseline run — skip delta check
            print(f"  {cat:<30} {'N/A':>10} {current_score:>10.4f} {'N/A':>10} {'SKIP':>8}")
            continue
 
        delta  = current_score - baseline_score
        status = "OK"
 
        # Flag regression if the drop exceeds the configured delta threshold
        if (baseline_score - current_score) > delta_threshold:
            status = "FAIL"
            failures.append({
                "category"  : cat,
                "baseline"  : baseline_score,
                "current"   : current_score,
                "delta"     : delta,
            })
 
        print(f"  {cat:<30} {baseline_score:>10.4f} {current_score:>10.4f} {delta:>+10.4f} {status:>8}")
 
    # Also check the overall score against the absolute threshold
    print(f"\n  {'OVERALL':<30} {baseline_metrics.get('overall_score', 0):>10.4f} {current['overall']:>10.4f}")
 
    if current["overall"] < overall_threshold:
        failures.append({
            "category" : "overall",
            "baseline" : baseline_metrics.get("overall_score", 0),
            "current"  : current["overall"],
            "delta"    : current["overall"] - baseline_metrics.get("overall_score", 0),
        })
 
    print("-" * 55)
 
    # ----------------------------------------------------------
    # Final decision: pass or fail the CI gate
    # ----------------------------------------------------------
    if failures:
        print("\nREGRESSION DETECTED — CI gate FAILED")
        for f in failures:
            print(
                f"  {f['category']} : "
                f"{f['baseline']:.4f} → {f['current']:.4f} "
                f"(Δ {f['delta']:+.4f})"
            )
        print("\nPR is BLOCKED. Fix the regression and re-push.")
        print("=" * 55)
        sys.exit(1)   # ← This is what blocks the GitHub PR
 
    else:
        print("\nAll checks passed — CI gate PASSED")
        print("PR is APPROVED for merge.")
        print("=" * 55)
        # sys.exit(0) is implicit — GitHub Actions marks job as SUCCESS
 
 
# ===============================================================
# _check_absolute_thresholds
# ---------------------------------------------------------------
# Fallback check used when no MLflow baseline exists (first run)
# or when the MLflow server is unreachable.
# Only checks scores against absolute minimums, not deltas.
#
# Parameters:
#   current   (dict)  : Current scores dict from scores.json.
#   threshold (float) : Minimum overall score from scorer_config.yaml.
#   cfg       (dict)  : Full scorer config dict.
#
# Exits with sys.exit(1) if any score is below its minimum.
# ===============================================================
def _check_absolute_thresholds(
    current   : dict,
    threshold : float,
    cfg       : dict,
) -> None:
    failures      = []
    per_cat_min   = cfg["thresholds"]["per_category_min"]
 
    for cat, score in current["categories"].items():
        if score < per_cat_min:
            failures.append(f"  {cat}: {score:.4f} < minimum {per_cat_min}")
 
    if current["overall"] < threshold:
        failures.append(
            f"  overall: {current['overall']:.4f} < threshold {threshold}"
        )
 
    if failures:
        print("Absolute threshold failures:")
        for f in failures:
            print(f)
        sys.exit(1)
    else:
        print("All absolute threshold checks passed.")
 
 
# ===============================================================
# Script entry point
# ---------------------------------------------------------------
# Called directly by .github/workflows/quality_gate.yml:
#   python scripts/check_regression.py
# ===============================================================
if __name__ == "__main__":
    check_regression()