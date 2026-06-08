# scorer/aggregate.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Aggregates raw per-example scoring results into a
#               per-category and overall summary dict.
#
#               Pipeline position — Stage 2 (Scorer):
#                 Called by scorer/run_scorer.py after all examples have
#                 been scored by inference.py + metrics.py + judge.py.
#                 Its output is what gets logged to MLflow and written
#                 to scores_output/scores.json.
#
#               Aggregation logic:
#                 For each example, a weighted composite score is computed:
#                   composite = (bertscore_f1 × w_bert)
#                             + (rouge_l      × w_rouge)
#                             + (judge.overall× w_judge)
#
#                 Weights come from scorer_config.yaml (metric_weights).
#                 Default weights: bert=0.30, rouge=0.20, judge=0.50
#
#                 Per-category score = mean of composite scores in that category.
#                 Overall score      = weighted sum of per-category scores,
#                                      using weights from dataset_config.yaml.
#
#               The output dict is the authoritative quality signal consumed
#               by scripts/check_regression.py for the CI gate decision.
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# defaultdict : dict subclass that returns a default value for
#               missing keys — used to group scores by category
#               without needing to check if the key exists first.
# List        : Type hint for the list of per-example result dicts.
# ===============================================================
from collections import defaultdict
from typing      import List


# ===============================================================
# aggregate_scores
# ---------------------------------------------------------------
# Rolls up a list of per-example scoring results into a structured
# summary with per-category averages and a weighted overall score.
#
# Parameters:
#   results (List[dict]) : One dict per golden example, each containing:
#                            - "category"    : str  — behavioral category name
#                            - "metrics"     : dict — {bertscore_f1, rouge_l}
#                            - "judge_score" : dict — {overall, ...} from judge.py
#   cfg (dict)           : Parsed scorer_config.yaml containing:
#                            - cfg["metric_weights"]  — {bertscore, rouge_l, judge_score}
#                            - cfg["categories"]      — [{name, weight}, ...]
#
# Returns:
#   dict : {
#       "categories" : {
#           "factual_accuracy"     : 0.891,
#           "refusal_behavior"     : 0.912,
#           "instruction_following": 0.847,
#           "tone_consistency"     : 0.823,
#           "multi_turn_coherence" : 0.801
#       },
#       "overall" : 0.872
#   }
#
#   This dict is logged to MLflow and written to scores_output/scores.json.
#   scripts/check_regression.py reads it to decide pass/fail.
# ===============================================================
def aggregate_scores(results: List[dict], cfg: dict) -> dict:

    # Read metric weights from scorer_config.yaml
    weights = cfg["metric_weights"]

    # Build a lookup: category_name → weight from dataset_config.yaml categories list
    cat_weights = {c["name"]: c["weight"] for c in cfg["categories"]}

    # ----------------------------------------------------------
    # Step 1 : Compute weighted composite score per example
    # ----------------------------------------------------------
    # Group composite scores by behavioral category.
    # defaultdict(list) lets us append without checking key existence.
    # ----------------------------------------------------------
    by_category = defaultdict(list)

    for r in results:
        category = r["category"]
        metrics  = r["metrics"]
        judge    = r["judge_score"]

        # Weighted composite score for this single example
        composite = (
            metrics["bertscore_f1"] * weights["bertscore"]
            + metrics["rouge_l"]    * weights["rouge_l"]
            + judge["overall"]      * weights["judge_score"]
        )
        by_category[category].append(round(composite, 4))

    # ----------------------------------------------------------
    # Step 2 : Average composite scores within each category
    # ----------------------------------------------------------
    # Per-category score = arithmetic mean of all example composites.
    # Rounded to 4 decimal places for clean MLflow metric display.
    # ----------------------------------------------------------
    category_scores = {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in by_category.items()
    }

    # ----------------------------------------------------------
    # Step 3 : Compute weighted overall score
    # ----------------------------------------------------------
    # Overall = sum of (category_score × category_weight) for all categories.
    # Weights come from dataset_config.yaml (factual_accuracy=0.30, etc.)
    # Categories missing from results are skipped (score 0 contribution).
    # ----------------------------------------------------------
    overall = sum(
        category_scores.get(cat, 0) * weight
        for cat, weight in cat_weights.items()
        if cat in category_scores
    )

    return {
        "categories" : category_scores,
        "overall"    : round(overall, 4),
    }