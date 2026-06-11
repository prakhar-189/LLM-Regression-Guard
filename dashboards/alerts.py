# dashboards/alerts.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Proactive drift alerting for the LLM quality dashboard.
#               Detects slow quality degradation that individual CI runs
#               might not catch — the kind of gradual drift where each
#               single run is just below the regression delta threshold,
#               but together represent a serious trend.
#
#               Pipeline position — Stage 4 (Drift Dashboard):
#                 Called by dashboards/app.py when the user clicks the
#                 "Check & Send Alerts" button in the sidebar.
#
#               Alert logic:
#                 For each behavioral category, compute the rolling 3-run
#                 average. If that average falls below the overall threshold
#                 defined in scorer_config.yaml, fire a Slack webhook.
#
#               Rate limiting:
#                 A JSON cooldown file (.alert_cooldown.json) tracks the
#                 last alert timestamp per category. Alerts are suppressed
#                 if the same category was alerted within the last 1 hour.
#                 This prevents Slack notification spam during repeated
#                 dashboard refreshes.
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# os           : Standard library — reads SLACK_WEBHOOK_URL env variable.
# json         : Standard library — reads/writes the cooldown tracking file.
# yaml         : PyYAML — reads scorer_config.yaml for the alert threshold.
# datetime     : Standard library — current timestamps for cooldown tracking.
# timezone     : datetime.timezone.utc — stores timestamps in UTC.
# Path         : pathlib — checks if the cooldown file exists.
# pandas       : Data manipulation — receives the MLflow runs DataFrame
#                and performs tail(3) + mean() for rolling averages.
# requests     : HTTP client — sends the POST request to the Slack webhook URL.
# ===============================================================
import os
import json
import yaml
from datetime import datetime, timezone
from pathlib  import Path

import pandas  as pd
import requests


# ===============================================================
# ALERT_COOLDOWN_FILE
# ---------------------------------------------------------------
# Path to the local JSON file that stores the last alert timestamp
# for each category. Created automatically on first alert.
# Excluded from git via .gitignore.
# ===============================================================
ALERT_COOLDOWN_FILE = ".alert_cooldown.json"


# ===============================================================
# _load_cooldowns
# ---------------------------------------------------------------
# Loads the last-alerted timestamps from the cooldown file.
# Returns an empty dict if the file doesn't exist yet.
#
# Returns:
#   dict : {category_name: ISO timestamp string, ...}
# ===============================================================
def _load_cooldowns() -> dict:
    if Path(ALERT_COOLDOWN_FILE).exists():
        return json.loads(Path(ALERT_COOLDOWN_FILE).read_text())
    return {}


# ===============================================================
# _save_cooldowns
# ---------------------------------------------------------------
# Persists the updated cooldown timestamps to the JSON file.
#
# Parameters:
#   cooldowns (dict) : {category_name: ISO timestamp string, ...}
# ===============================================================
def _save_cooldowns(cooldowns: dict) -> None:
    Path(ALERT_COOLDOWN_FILE).write_text(json.dumps(cooldowns))


# ===============================================================
# _on_cooldown
# ---------------------------------------------------------------
# Checks whether an alert for the given category is still in cooldown.
# Prevents duplicate Slack alerts within the cooldown window.
#
# Parameters:
#   category  (str)  : Behavioral category name.
#   cooldowns (dict) : Loaded cooldown timestamps dict.
#   hours     (int)  : Cooldown window in hours. Default: 1.
#
# Returns:
#   bool : True if within cooldown period (suppress alert).
#          False if cooldown has passed (allow alert).
# ===============================================================
def _on_cooldown(category: str, cooldowns: dict, hours: int = 1) -> bool:
    last_alert = cooldowns.get(category)
    if not last_alert:
        return False   # Never alerted before — not on cooldown

    last_ts  = datetime.fromisoformat(last_alert).replace(tzinfo=timezone.utc)
    elapsed  = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600
    return elapsed < hours


# ===============================================================
# fire_slack_alert
# ---------------------------------------------------------------
# Sends a drift alert to the configured Slack channel via webhook.
# Falls back to a console print if SLACK_WEBHOOK_URL is not set.
#
# Parameters:
#   category  (str)   : The behavioral category that triggered the alert.
#   avg       (float) : The 3-run rolling average score that fell below threshold.
#   threshold (float) : The configured minimum acceptable score.
#
# Returns:
#   None
#
# Slack webhook format:
#   A simple {"text": "..."} payload renders as a plain message
#   in the configured Slack channel. For rich formatting, replace
#   with a blocks payload using the Slack Block Kit format.
# ===============================================================
def fire_slack_alert(category: str, avg: float, threshold: float) -> None:

    webhook_url = os.getenv("SLACK_WEBHOOK_URL")

    if not webhook_url:
        # Graceful fallback — log to console, don't crash the dashboard
        print(f"[ALERT] SLACK_WEBHOOK_URL not set. Would alert: '{category}' avg={avg:.4f}")
        return

    payload = {
        "text": (
            f":warning: *LLM Quality Drift Alert*\n"
            f"Category `{category}` 3-run rolling avg: *{avg:.4f}* "
            f"< threshold *{threshold}*\n"
            f"Timestamp: `{datetime.now().strftime('%Y-%m-%d %H:%M UTC')}`\n"
            f"Action: Review recent model/prompt changes in the dashboard."
        )
    }

    try:
        r = requests.post(webhook_url, json=payload, timeout=5)
        if r.status_code == 200:
            print(f"[ALERT] Slack alert sent for category: '{category}'")
        else:
            print(f"[ALERT] Slack webhook returned HTTP {r.status_code}")
    except Exception as e:
        print(f"[ALERT] Failed to send Slack alert: {e}")


# ===============================================================
# check_and_alert
# ---------------------------------------------------------------
# Main alerting function. Called by dashboards/app.py.
# Checks rolling 3-run averages per category and fires Slack alerts
# for any category below threshold that isn't in cooldown.
#
# Parameters:
#   df (pd.DataFrame) : MLflow runs DataFrame from dashboards/app.py.
#                       Must contain columns: category, score.
#                       Each row is one (run, category) score pair.
#
# Returns:
#   list[str] : Names of categories that triggered an alert this call.
#               Empty list if all categories are healthy.
#
# Example output alert scenario:
#   If "refusal_behavior" had scores [0.82, 0.79, 0.76] for the last 3 runs,
#   rolling avg = 0.79 < threshold 0.80 → Slack alert fires.
# ===============================================================
def check_and_alert(df: pd.DataFrame) -> list:

    cfg       = yaml.safe_load(open("config/scorer_config.yaml"))
    threshold = cfg["thresholds"]["overall"]
    cooldowns = _load_cooldowns()
    alerted   = []

    if df.empty or "category" not in df.columns:
        return alerted

    # Check each behavioral category independently
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat].tail(3)   # Last 3 runs only

        # Need at least 3 data points for a meaningful rolling average
        if len(cat_df) < 3:
            continue

        rolling_avg = cat_df["score"].mean()

        # Alert if below threshold AND not in cooldown
        if rolling_avg < threshold:
            if not _on_cooldown(cat, cooldowns):
                fire_slack_alert(cat, rolling_avg, threshold)
                cooldowns[cat] = datetime.now(timezone.utc).isoformat()
                alerted.append(cat)

    # Persist updated cooldown timestamps for next call
    _save_cooldowns(cooldowns)
    return alerted