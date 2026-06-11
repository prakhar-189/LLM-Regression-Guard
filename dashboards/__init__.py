# dashboards/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'dashboards' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from dashboards.charts import score_timeseries
#                   from dashboards.alerts import check_and_alert
#
# Modules in this package:
#   app.py    : Streamlit dashboard entry point. Pulls MLflow run data,
#               renders all charts, and shows the raw run table.
#   charts.py : Plotly chart builders — time-series, heatmap, shadow panel.
#   alerts.py : Rolling average drift detection with Slack webhook alerts.
# --------------------------------------------------------------