import json
import sys
import yaml
from mlflow.tracking import MlflowClient

def load_config():
    with open('config/scorer_config.yaml', 'r') as f:
        return yaml.safe_load(f)
    
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