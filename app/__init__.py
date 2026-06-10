# app/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'app' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from app.middleware import ShadowMiddleware
#                   from app.tasks      import score_shadow_async
#                   from app.db         import log_shadow_result
#
# Modules in this package:
#   main.py       : FastAPI application entry point. Registers middleware
#                   and mounts the chat router.
#   router.py     : FastAPI route definitions — the /chat and /health endpoints.
#   middleware.py : Shadow traffic middleware — intercepts 5% of requests
#                   for async quality evaluation.
#   tasks.py      : Celery background task that scores shadow responses
#                   without blocking the production request path.
#   db.py         : TimescaleDB client for persisting shadow eval results.
# --------------------------------------------------------------