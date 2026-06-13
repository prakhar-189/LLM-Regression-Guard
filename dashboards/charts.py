# dashboards/charts.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Plotly chart builders for the LLM drift dashboard.
#               Kept separate from dashboards/app.py so visualization
#               logic is cleanly decoupled from Streamlit UI code.
#
#               Pipeline position — Stage 4 (Drift Dashboard):
#                 Called by dashboards/app.py to render three panels:
#                   1. score_timeseries : Per-category scores over time
#                                         with version bump markers.
#                   2. category_heatmap : Category × model-version grid.
#                   3. shadow_comparison: Production vs shadow rolling averages.
#
#               Why Plotly?
#                 Plotly produces interactive charts (hover, zoom, pan)
#                 that render natively in Streamlit via st.plotly_chart().
#                 Charts can be exported as PNG from the browser toolbar.
# --------------------------------------------------------------
 
 
# ===============================================================
# Imports
# ---------------------------------------------------------------
# pandas      : Data manipulation library.
#               Used to pivot DataFrames for the heatmap and to compute
#               rolling averages for the shadow comparison chart.
# plotly.graph_objects (go) : Low-level Plotly API for building charts
#                             with full control over traces and layout.
# plotly.express      (px)  : High-level Plotly API for quick charts.
#                             Used for the imshow heatmap.
# List        : Type hint for function parameters.
# ===============================================================
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List


# ===============================================================
# score_timeseries
# ---------------------------------------------------------------
# Renders per-category quality scores over time as an interactive
# line chart. Overlays vertical markers at model version bump points
# so regressions can be visually traced to their cause.
#
# Parameters:
#   df (pd.DataFrame) : Columns required:
#                         - timestamp     : datetime — run execution time
#                         - category      : str — behavioral category name
#                         - score         : float — per-category score
#                         - model_version : str — model name for that run
#                         - overall_score : float — (optional) overall score
#
# Returns:
#   go.Figure : Interactive Plotly line chart.
#
# Key design details:
#   - One colored line per category (uses Plotly's Set2 palette).
#   - Dashed red vertical lines mark model version changes.
#   - Red horizontal threshold line at y=0.80 shows pass/fail boundary.
#   - Y-axis fixed at [0.5, 1.05] to keep changes visually clear.
# ===============================================================
def score_timeseries(df: pd.DataFrame) -> go.Figure:
    fig        = go.Figure()
    categories = df["category"].unique() if "category" in df.columns else []

    # Add overall score as a dashed reference line if present
    if "overall_score" in df.columns:
        fig.add_trace(go.Scatter(
            x    = df["timestamp"],
            y    = df["overall_score"],
            name = "Overall",
            mode = "lines+markers",
            line = dict(width=3, dash="dash"),
            marker = dict(size=6),
        ))

    # Add one line per behavioural category
    colors = px.colors.qualitative.Set2
    for i, cat in enumerate(categories):
        cat_df = df[df["category"] == cat]
        fig.add_trace(go.Scatter(
            x    = cat_df["timestamp"],
            y    = cat_df["score"],
            name = cat.replace("_", " ").title(),
            mode = "lines+markers",
            line = dict(color=colors[i % len(colors)]),
            marker = dict(size=5),
        ))

    # Add vertical markers at model version change points
    # These let you visually connect "score dropped here" to "this version was bumped"
    if "model_version" in df.columns:
        version_change_times = df.drop_duplicates("model_version")["timestamp"].tolist()
        for ts in version_change_times:
            fig.add_vline(
                x                    = ts.timestamp() * 1000,    # Convert to epoch ms
                line_dash            = "dot",
                line_color           = "rgba(255,100,100,0.5)",
                annotation_text      = "version bump",
                annotation_position  = "top",
            )

    # Add horizontal threshold line at the overall pass/fail boundary
    fig.add_hline(
        y                   = 0.80,
        line_dash           = "dash",
        line_color          = "red",
        annotation_text     = "threshold (0.80)",
        annotation_position = "bottom right",
    )

    fig.update_layout(
         title      = "Score Trends Over Time",
        xaxis_title= "Run Date",
        yaxis_title= "Score",
        yaxis_range= [0.5, 1.05],
        legend     = dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode  = "x unified",
        height     = 420,
    )

    return fig


# ===============================================================
# category_heatmap
# ---------------------------------------------------------------
# Renders a color-coded grid of scores by category and model version.
# Red cells = regression; green cells = strong performance.
# Immediately shows WHICH category degraded on WHICH model version.
#
# Parameters:
#   df (pd.DataFrame) : Columns required:
#                         - category      : str — behavioral category name
#                         - model_version : str — model version string
#                         - score         : float — quality score
#
# Returns:
#   go.Figure : Interactive Plotly heatmap using RdYlGn color scale.
#
# Implementation note:
#   pd.pivot_table with aggfunc="mean" handles cases where multiple
#   runs exist for the same (category, model_version) combination.
# ===============================================================
def category_heatmap(df: pd.DataFrame) -> go.Figure:
    if df.empty or "model_version" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title = "No data available for Heatmap")
        return fig
    
    # Pivot : rows = categories, columns = model versions, values = mean score
    pivot = df.pivot_table(
        index   = "category",
        columns = "model_version",
        values  = "score",
        aggfunc = "mean",
    )

    fig = px.imshow(
        pivot,
        color_continuous_scale = "RdYlGn",   # Red = Bad, Yellow = Borderline, Green = Good
        zmin                  = 0.60,       # Color scale minimum
        zmax                  = 1.00,       # Color scale maximum
        title                 = "Category x Model Version Heatmap",
        labels                = {"color": "Score"},
        text_auto             = ".3f",      # Show 3-decimal scores in each cell
    )

    fig.update_layout(
        xaxis_title = "Model Version",
        yaxis_title = "Category",
        height      = 380,
    )

    return fig


# ===============================================================
# shadow_comparison
# ---------------------------------------------------------------
# Renders a rolling average comparison between production and shadow
# model quality scores from TimescaleDB.
# Used to decide whether the shadow model is ready for promotion.
#
# Parameters:
#   shadow_df (pd.DataFrame) : Columns required:
#                                - ts           : datetime — evaluation time
#                                - prod_score   : float — production judge score
#                                - shadow_score : float — shadow judge score
#
# Returns:
#   go.Figure : Interactive Plotly line chart with rolling averages.
#
# Promotion logic context:
#   If shadow_rolling >= prod_rolling - 0.02 over 7 days with 500+ samples,
#   the shadow model is considered safe to promote to production.
# ===============================================================
def shadow_comparison(shadow_df: pd.DataFrame) -> go.Figure:
 
    fig = go.Figure()
 
    if shadow_df.empty:
        fig.update_layout(title="No shadow traffic data yet")
        return fig
 
    shadow_df = shadow_df.sort_values("ts")
 
    # Rolling window: min(50, total_rows) to handle small datasets
    window = min(50, len(shadow_df))
 
    shadow_df["prod_rolling"]   = shadow_df["prod_score"].rolling(window).mean()
    shadow_df["shadow_rolling"] = shadow_df["shadow_score"].rolling(window).mean()
 
    # Production model line (solid blue)
    fig.add_trace(go.Scatter(
        x    = shadow_df["ts"],
        y    = shadow_df["prod_rolling"],
        name = "Production (rolling avg)",
        mode = "lines",
        line = dict(color="royalblue", width=2),
    ))
 
    # Shadow model line (dashed red)
    fig.add_trace(go.Scatter(
        x    = shadow_df["ts"],
        y    = shadow_df["shadow_rolling"],
        name = "Shadow (rolling avg)",
        mode = "lines",
        line = dict(color="tomato", width=2, dash="dash"),
    ))
 
    fig.update_layout(
        title       = "Shadow vs Production — Rolling Quality Score",
        xaxis_title = "Time",
        yaxis_title = "Judge Score (rolling average)",
        yaxis_range = [0.5, 1.05],
        height      = 380,
        hovermode   = "x unified",
    )
 
    return fig