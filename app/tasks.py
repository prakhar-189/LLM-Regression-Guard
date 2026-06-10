# app/tasks.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-10
# Description : Celery background task for asynchronous shadow traffic scoring.
#               Fired by ShadowMiddleware when a request is selected for
#               shadow evaluation. Runs entirely outside the production
#               request/response cycle — zero latency impact on users.
#
#               Pipeline position — Stage 5 (Shadow Traffic Monitor):
#                 Called via .delay() from app/middleware.py.
#                 Executes in a separate Celery worker process.
#                 Results are persisted to TimescaleDB via app/db.py.
#
#               What the task does:
#                 1. Parse the intercepted request body to extract the prompt
#                 2. Call BOTH the production model and the shadow model
#                 3. Score BOTH responses using the judge LLM rubric
#                 4. Compute the delta (shadow_score - prod_score)
#                 5. Log the result to TimescaleDB for dashboard display
#
#               The delta is what matters:
#                 A positive delta means shadow is better than production.
#                 A negative delta means shadow is worse.
#                 The dashboard tracks the 7-day rolling delta to decide
#                 whether the shadow model is ready for promotion.
#
# Start the Celery worker:
#   celery -A app.tasks worker --loglevel=info
# -------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# json      : Standard library — parses the raw request body string
#             to extract the 'prompt' field.
# os        : Standard library — reads REDIS_URL environment variable.
# sys, Path : Standard library — adds project root to sys.path.
# yaml      : PyYAML — reads scorer_config.yaml and model_version.yaml.
#
# Celery    : Distributed task queue library.
#               - Celery(broker=...) : Creates the app connected to Redis.
#               - @celery_app.task   : Registers a function as a Celery task.
#               - .delay()           : Fires the task asynchronously.
#               - bind=True          : Gives the task access to `self` for retries.
#               - max_retries=2      : Retry up to 2 times on failure.
#
# load_dotenv       : python-dotenv — loads REDIS_URL from .env.
# run_inference     : scorer/inference.py — calls the LLM.
# judge_score       : scorer/judge.py — scores a response against reference.
# log_shadow_result : app/db.py — persists result to TimescaleDB.
# ===============================================================
import json
import os
import sys
import yaml
from pathlib import Path
from celery import Celery
from dotenv import load_dotenv
from scorer.inference import run_inference
from scorer.judge import judge_score
from app.db import log_shadow_result

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===============================================================
# Redis / Celery Configuration
# ---------------------------------------------------------------
# REDIS_URL : Connection string for the Redis message broker.
#             Redis acts as the queue between the FastAPI app (producer)
#             and the Celery worker (consumer).
#             Format: redis://<host>:<port>/<db_number>
#
# broker  : Where Celery reads tasks from (Redis).
# backend : Where Celery stores task results (Redis).
#           We store results for debugging — not used by the pipeline directly.
# ===============================================================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

celery_app = Celery(
    "shadow_tasks",
    broker = REDIS_URL,
    backend = REDIS_URL
)

# Celery worker Settings
celery_app.conf.update(
    task_serializer = "json",    # Serialize task args as JSON
    result_serializer = "json",  # Serialize results as JSON
    accept_content = ["json"],   # Only accept JSON-encoded tasks
    timezone = "UTC",            # Consistent timestamp across machines
    task_track_started = True,   # Mark task as STARTED when worker picks them up
)


# ===============================================================
# score_shadow_async
# ---------------------------------------------------------------
# Celery task — the async shadow scoring worker.
#
# Fired by ShadowMiddleware via: score_shadow_async.delay(raw_body)
# Runs in a separate Celery worker process asynchronously.
#
# Parameters:
#   self     (Task)  : Celery task instance (from bind=True).
#                      Provides access to self.retry() for error handling.
#   raw_body (str)   : The raw JSON request body string intercepted
#                      by ShadowMiddleware. Contains at least {"prompt": "..."}.
#
# Returns:
#   None (results are persisted to TimescaleDB via log_shadow_result)
#
# Retry behavior:
#   If the task fails (API error, DB error), Celery retries up to
#   max_retries=2 times with a 30-second delay between attempts.
# ===============================================================
@celery_app.task(name = "score_shadow_async", bind = True, max_retries = 2)
def score_shadow_async(self, raw_body : str):
    try:
        # Load configs fresh on each task execution(configs may change between worker restarts)
        cfg        = yaml.safe_load(open("config/scorer_config.yaml"))
        model_cfg  = yaml.safe_load(open("config/model_version.yaml"))
 
        # Parse the intercepted request body to get the prompt
        body   = json.loads(raw_body)
        prompt = body.get("prompt", "")
        
        # Guard : skip empty prompts
        if not prompt:
            return
        
        # ----------------------------------------------------------
        # Model names
        # ----------------------------------------------------------
        # production_model : The current live model serving users.
        #                    Read from baseline_model in model_version.yaml.
        # shadow_model     : The candidate model being evaluated.
        #                    Read from current_model in model_version.yaml.
        # When baseline and current are the same model, delta will be ~0.
        # ----------------------------------------------------------
        production_model = model_cfg.get("baseline_model", "gpt-4o")
        shadow_model     = model_cfg.get("current_model",  "gpt-4o")


         # ----------------------------------------------------------
        # Step 1 : Get responses from both models for the same prompt
        # ----------------------------------------------------------
        prod_response   = run_inference(prompt, model=production_model)
        shadow_response = run_inference(prompt, model=shadow_model)
 
        # ----------------------------------------------------------
        # Step 2 : Score both responses using the judge LLM rubric
        # ----------------------------------------------------------
        # For the production model, it's compared against itself
        # (reference = response) — establishes the baseline quality
        # for this specific prompt.
        # For the shadow model, it's compared against the production
        # response as the reference.
        # ----------------------------------------------------------
        prod_score   = judge_score(prompt, prod_response,   prod_response,   cfg)
        shadow_score = judge_score(prompt, prod_response,   shadow_response, cfg)
 
        # Positive delta = shadow is better; negative = shadow is worse
        delta = shadow_score["overall"] - prod_score["overall"]
 
        # ----------------------------------------------------------
        # Step 3 : Persist result to TimescaleDB
        # ----------------------------------------------------------
        # The dashboard queries this table to show the rolling
        # shadow vs production comparison panel.
        # ----------------------------------------------------------
        log_shadow_result({
            "prompt"       : prompt,
            "prod_model"   : production_model,
            "shadow_model" : shadow_model,
            "prod_score"   : prod_score["overall"],
            "shadow_score" : shadow_score["overall"],
            "delta"        : round(delta, 4),
        })
 
    except Exception as exc:
        # Retry up to 2 times with 30-second cooldown between attempts
        self.retry(exc=exc, countdown=30)