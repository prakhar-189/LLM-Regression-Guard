# app/main.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : FastAPI application entry point.
#               Creates the app instance, registers the shadow traffic
#               middleware, and mounts the chat router.
#
#               Pipeline position — Stage 5 (Shadow Traffic Monitor):
#                 This is the production LLM application that the entire
#                 regression guard pipeline is designed to protect.
#                 The ShadowMiddleware registered here intercepts 5% of
#                 all /chat requests and routes them to a Celery background
#                 task for async quality scoring — no user latency added.
#
# Run locally:
#   uvicorn app.main:app --reload --port 8000
#
# Run via Docker:
#   docker compose up app
# --------------------------------------------------------------
 
 
# ===============================================================
# Imports
# ---------------------------------------------------------------
# FastAPI         : Web framework for building the REST API.
#                   Auto-generates OpenAPI docs at /docs.
# ShadowMiddleware: Custom Starlette middleware from app/middleware.py.
#                   Wraps every request and clones 5% to the shadow scorer.
# router          : APIRouter from app/router.py containing the /chat
#                   and /health endpoint definitions.
# ===============================================================
from fastapi import FastAPI
from app.middleware import ShadowMiddleware
from app.router import router


# ===============================================================
# FastAPI Application Instance
# ---------------------------------------------------------------
# title       : Shown in the auto-generated /docs Swagger UI.
# description : Brief explanation of the app's role in the pipeline.
# version     : Semantic version for the API.
# ===============================================================
app = FastAPI(
    title       = "LLM Regression Guard — App Under Test",
    description = (
        "The LLM-powered chat application being monitored by the "
        "regression pipeline. 5% of requests are silently evaluated "
        "by the shadow traffic system."
    ),
    version     = "1.0.0",
)


# ===============================================================
# Middleware Registration
# ---------------------------------------------------------------
# ShadowMiddleware is added here so it wraps the ENTIRE ASGI app.
# Every incoming HTTP request passes through it before reaching
# the router — ensuring 100% of /chat requests are eligible for
# the 5% shadow sample.
#
# shadow_pct=5 means: if hash(request_body) % 100 < 5, the request
# is cloned to the async shadow scorer. Same prompts always go to
# the same model variant (hash-based, not random).
# ===============================================================
app.add_middleware(ShadowMiddleware, shadow_pct=5)
 
 
# ===============================================================
# Router Registration
# ---------------------------------------------------------------
# Mounts all routes defined in app/router.py onto the main app.
# Currently includes:
#   POST /chat   — Main LLM chat endpoint
#   GET  /health — Health check endpoint
# ===============================================================
app.include_router(router)