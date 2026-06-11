# dashboards/app.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Streamlit drift dashboard — the primary observability
#               interface for the LLM Regression Guard pipeline.
#
#               Pipeline position — Stage 4 (Drift Dashboard):
#                 Connects to the MLflow tracking server and TimescaleDB
#                 to visualize quality trends across all scoring runs.
#
#               What the dashboard answers:
#                 - Which category is silently getting worse?
#                 - When did a regression start (which version bump)?
#                 - Is the shadow model ready to replace production?
#                 - Which (category × model version) combinations are risky?
#
#               Layout:
#                 Row 1: Four summary metric cards (overall, runs, model, status)
#                 Row 2: Score time-series + Category×Version heatmap (side by side)
#                 Row 3: Shadow vs production comparison panel
#                 Sidebar: Alert controls, threshold reference, MLflow link
#                 Expander: Raw run data table (last 20 runs)
#
# Run locally:
#   streamlit run dashboards/app.py
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# os          : Standard library — reads MLFLOW_TRACKING_URI env var.
# sys, Path   : Standard library — adds project root to sys.path.
# datetime    : Standard library — converts MLflow Unix timestamps to datetime.
#
# mlflow      : MLflow tracking client.
#                 - MlflowClient()             : Low-level API for querying runs.
#                 - get_experiment_by_name()   : Finds experiment by name string.
#                 - search_runs()              : Returns all runs matching filter.
#
# pandas      : Data manipulation — builds the DataFrame that all charts consume.
# streamlit   : Web UI framework. Renders the entire dashboard as a Python script.
#                 - st.set_page_config() : Page title, icon, layout.
#                 - st.metric()          : Numeric metric cards with delta arrows.
#                 - st.columns()         : Side-by-side layout.
#                 - st.plotly_chart()    : Renders Plotly figures inline.
#                 - st.cache_data()      : Caches expensive function results (TTL).
# load_dotenv : python-dotenv — loads MLFLOW_TRACKING_URI from .env.
#
# score_timeseries   : dashboards/charts.py — line chart over time.
# category_heatmap   : dashboards/charts.py — category × version grid.
# shadow_comparison  : dashboards/charts.py — rolling avg comparison.
# check_and_alert    : dashboards/alerts.py — drift detection + Slack alerts.
# ===============================================================
import os
import sys
from pathlib  import Path
from datetime import datetime

import mlflow
import pandas    as pd
import streamlit as st
from dotenv      import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboards.charts import score_timeseries, category_heatmap, shadow_comparison
from dashboards.alerts import check_and_alert


# ===============================================================
# Page Configuration
# ---------------------------------------------------------------
# Must be the first Streamlit call in the script.
# layout="wide" uses the full browser width for chart panels.
# ===============================================================
st.set_page_config(
    page_title = "LLM Regression Guard",
    page_icon  = "🔍",
    layout     = "wide",
)

st.title("🔍 LLM Regression Guard — Drift Dashboard")
st.caption("Real-time quality monitoring across model versions and prompt changes.")


# ===============================================================
# MLflow Connection
# ---------------------------------------------------------------
# Set the tracking URI from environment. Falls back to localhost
# if MLFLOW_TRACKING_URI is not set (local development).
# ===============================================================
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(mlflow_uri)


# ===============================================================
# load_mlflow_runs
# ---------------------------------------------------------------
# Queries the MLflow tracking server for all runs in the
# "llm-regression-guard" experiment and returns a flat DataFrame.
#
# @st.cache_data(ttl=60) : Caches the result for 60 seconds to avoid
#   hammering the MLflow server on every Streamlit re-render.
#   Streamlit re-runs the entire script on user interaction, so
#   caching is critical for dashboard responsiveness.
#
# Returns:
#   pd.DataFrame : One row per (run, category) combination.
#                  Columns: run_id, timestamp, model_version,
#                           category, score, overall_score
# ===============================================================
@st.cache_data(ttl=60)
def load_mlflow_runs():
    client = mlflow.MlflowClient()
    try:
        experiment = client.get_experiment_by_name("llm-regression-guard")
        if experiment is None:
            return pd.DataFrame()

        runs = client.search_runs(
            experiment_ids = [experiment.experiment_id],
            order_by       = ["start_time ASC"],
        )
    except Exception as e:
        st.warning(f"Could not connect to MLflow at {mlflow_uri}: {e}")
        return pd.DataFrame()

    records = []
    for run in runs:
        metrics = run.data.metrics
        tags    = run.data.tags
        ts      = datetime.fromtimestamp(run.info.start_time / 1000)

        # Add an "overall" row for the summary metric card
        if "overall_score" in metrics:
            records.append({
                "run_id"        : run.info.run_id,
                "timestamp"     : ts,
                "model_version" : tags.get("model_version", "unknown"),
                "category"      : "overall",
                "score"         : metrics["overall_score"],
                "overall_score" : metrics["overall_score"],
            })

        # Add one row per behavioral category
        # MLflow stores per-category metrics as "cat_{category_name}"
        for key, val in metrics.items():
            if key.startswith("cat_"):
                cat = key[4:]   # Strip the "cat_" prefix
                records.append({
                    "run_id"        : run.info.run_id,
                    "timestamp"     : ts,
                    "model_version" : tags.get("model_version", "unknown"),
                    "category"      : cat,
                    "score"         : val,
                    "overall_score" : metrics.get("overall_score"),
                })

    return pd.DataFrame(records)


# ===============================================================
# load_shadow_data
# ---------------------------------------------------------------
# Fetches the last 7 days of shadow traffic results from TimescaleDB.
#
# @st.cache_data(ttl=30) : Caches for 30 seconds (more frequent refresh
#   than MLflow data since shadow results arrive continuously).
#
# Returns:
#   pd.DataFrame : Columns: ts, prod_model, shadow_model,
#                           prod_score, shadow_score, delta
#                  Empty DataFrame if TimescaleDB is unavailable.
# ===============================================================
@st.cache_data(ttl=30)
def load_shadow_data():
    try:
        from app.db import get_recent_shadow_results
        rows = get_recent_shadow_results(days=7)
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
df        = load_mlflow_runs()
shadow_df = load_shadow_data()

# Stop rendering and show a warning if no MLflow data exists yet
if df.empty:
    st.warning(
        "No MLflow runs found. Run `python scorer/run_scorer.py` to generate data."
    )
    st.stop()


# ──────────────────────────────────────────────────────────────
# Summary metric cards — top row
# ──────────────────────────────────────────────────────────────
# Shows the four most important current-state values at a glance.
# st.metric() displays a value with an optional delta arrow.
# ──────────────────────────────────────────────────────────────
overall_df = df[df["category"] == "overall"].sort_values("timestamp")

if not overall_df.empty:
    latest = overall_df.iloc[-1]
    prev   = overall_df.iloc[-2] if len(overall_df) > 1 else None

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Delta arrow shows improvement/regression vs previous run
        delta = (latest["score"] - prev["score"]) if prev is not None else None
        st.metric(
            label = "Latest Overall Score",
            value = f"{latest['score']:.4f}",
            delta = f"{delta:+.4f}" if delta is not None else None,
        )
    with col2:
        st.metric("Total Runs",    len(overall_df))
    with col3:
        st.metric("Model Version", latest["model_version"])
    with col4:
        status = "✅ Passing" if latest["score"] >= 0.80 else "❌ Below threshold"
        st.metric("Gate Status", status)

st.divider()


# ──────────────────────────────────────────────────────────────
# Main chart panels — side by side
# ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Score Trends Over Time")
    # Exclude "overall" category rows — shown separately in metric cards
    cat_df = df[df["category"] != "overall"]
    st.plotly_chart(score_timeseries(cat_df), use_container_width=True)

with col_right:
    st.subheader("Category × Version Heatmap")
    st.plotly_chart(category_heatmap(cat_df), use_container_width=True)


# ──────────────────────────────────────────────────────────────
# Shadow traffic panel
# ──────────────────────────────────────────────────────────────
st.subheader("Shadow Traffic Monitor")
if shadow_df.empty:
    st.info(
        "No shadow traffic data yet. "
        "Shadow monitoring begins once the FastAPI app receives live traffic "
        "and docker compose up worker is running."
    )
else:
    st.plotly_chart(shadow_comparison(shadow_df), use_container_width=True)

st.divider()


# ──────────────────────────────────────────────────────────────
# Sidebar — controls and reference info
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    # Manual alert check button
    if st.button("🔔 Check & Send Alerts"):
        alerted = check_and_alert(cat_df)
        if alerted:
            st.error(f"Alerts fired for: {', '.join(alerted)}")
        else:
            st.success("All categories within threshold. No alerts needed.")

    st.divider()

    # Threshold reference for quick reading
    st.subheader("Thresholds")
    st.markdown("- Overall minimum: **0.80**")
    st.markdown("- Per-category minimum: **0.72**")
    st.markdown("- Max regression delta: **0.05**")

    st.divider()

    # Direct link to the MLflow experiment UI
    st.subheader("MLflow")
    st.markdown(f"[Open MLflow Experiment UI]({mlflow_uri})")


# ──────────────────────────────────────────────────────────────
# Raw data table — collapsed by default
# ──────────────────────────────────────────────────────────────
with st.expander("📋 Raw run data (last 20 overall scores)"):
    display_df = (
        overall_df[["timestamp", "model_version", "score"]]
        .tail(20)
        .rename(columns={"score": "overall_score"})
    )
    st.dataframe(display_df, use_container_width=True)