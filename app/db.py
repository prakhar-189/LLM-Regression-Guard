# app/db.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : TimescaleDB client for persisting shadow traffic
#               evaluation results.
#
#               Pipeline position — Stage 5 (Shadow Traffic Monitor):
#                 Called by app/tasks.py after each shadow evaluation.
#                 The dashboard (dashboards/app.py) reads from this
#                 database to render the shadow vs production panel.
#
#               Why TimescaleDB?
#                 TimescaleDB is PostgreSQL extended with automatic
#                 time-series partitioning (hypertables). It makes
#                 time-range queries like "last 7 days of shadow scores"
#                 extremely fast without manual indexing.
#                 It's also 100% compatible with standard psycopg2/SQL.
#
#               Table: shadow_evals
#                 Each row = one shadow traffic evaluation result.
#                 Partitioned automatically by the 'ts' timestamp column.
#
#               Setup (run once after docker compose up timescale):
#                 python -c "from app.db import create_table; create_table()"
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# os           : Standard library — reads TIMESCALE_URL env variable.
# hashlib      : Standard library — hashes the prompt text to a BIGINT
#                so prompts aren't stored as raw text (privacy).
# datetime     : Standard library — UTC timestamps for each evaluation.
# timezone     : datetime.timezone.utc — ensures all timestamps are
#                stored in UTC for consistent cross-timezone queries.
# psycopg2     : PostgreSQL adapter for Python.
#                Provides the database connection and cursor objects.
#                Using psycopg2-binary (bundled C library) for easy install.
# load_dotenv  : python-dotenv — loads TIMESCALE_URL from .env file.
# ===============================================================
import os
import hashlib
from datetime import datetime, timezone

import psycopg2
from dotenv import load_dotenv

load_dotenv()


# ===============================================================
# Database connection string
# ---------------------------------------------------------------
# Format: postgresql://user:password@host:port/database_name
# Defaults to a local Docker Compose TimescaleDB instance.
# Override by setting TIMESCALE_URL in your .env file.
# ===============================================================
TIMESCALE_URL = os.getenv(
    "TIMESCALE_URL",
    "postgresql://postgres:password@localhost:5432/shadow_evals"
)


# ===============================================================
# get_conn
# ---------------------------------------------------------------
# Returns a new psycopg2 connection to the TimescaleDB instance.
# Called before every DB operation to get a fresh connection.
# Used as a context manager (with get_conn() as conn:) to ensure
# connections are properly closed after each operation.
#
# Returns:
#   psycopg2.connection : An open database connection.
#
# Raises:
#   psycopg2.OperationalError : If the database is unreachable.
# ===============================================================
def get_conn():
    return psycopg2.connect(TIMESCALE_URL)


# ===============================================================
# create_table
# ---------------------------------------------------------------
# Creates the shadow_evals table and converts it to a TimescaleDB
# hypertable if it doesn't already exist.
# Run ONCE during initial project setup.
#
# Table schema:
#   ts           TIMESTAMPTZ : Evaluation timestamp (UTC).
#                              The hypertable partition key.
#   prompt_hash  BIGINT      : MD5 hash of the prompt (BIGINT-truncated).
#                              Stored instead of raw prompt text for privacy.
#   prod_model   TEXT        : Production model name (e.g. "gpt-4o").
#   shadow_model TEXT        : Shadow model name (e.g. "gpt-4o-mini").
#   prod_score   FLOAT       : Judge score for the production model response.
#   shadow_score FLOAT       : Judge score for the shadow model response.
#   delta        FLOAT       : shadow_score - prod_score.
#                              Positive = shadow better; negative = worse.
#
# Why a hypertable?
#   TimescaleDB automatically partitions the table by time intervals.
#   Queries like "WHERE ts > NOW() - INTERVAL '7 days'" execute
#   in milliseconds even with millions of rows.
# ===============================================================
def create_table() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:

            # Create the base table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS shadow_evals (
                    ts           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
                    prompt_hash  BIGINT,
                    prod_model   TEXT,
                    shadow_model TEXT,
                    prod_score   FLOAT,
                    shadow_score FLOAT,
                    delta        FLOAT
                );
            """)

            # Convert to a hypertable — the core TimescaleDB feature.
            # if_not_exists=TRUE prevents errors if already a hypertable.
            cur.execute("""
                SELECT create_hypertable(
                    'shadow_evals', 'ts',
                    if_not_exists => TRUE
                );
            """)
            conn.commit()

    print("shadow_evals hypertable is ready.")


# ===============================================================
# log_shadow_result
# ---------------------------------------------------------------
# Inserts one shadow evaluation result into the shadow_evals table.
# Called by app/tasks.py after each shadow scoring run.
#
# Parameters:
#   result (dict) : Must contain:
#                     "prompt"        (str)   : Original user prompt.
#                     "prod_model"    (str)   : Production model name.
#                     "shadow_model"  (str)   : Shadow model name.
#                     "prod_score"    (float) : Production judge score.
#                     "shadow_score"  (float) : Shadow judge score.
#                     "delta"         (float) : shadow_score - prod_score.
#
# Returns:
#   None
#
# Note on prompt_hash:
#   The raw prompt is hashed with MD5, converted to int, then truncated
#   to fit PostgreSQL's BIGINT range (max 2^63 - 1). The hash allows
#   deduplication analysis without storing PII.
# ===============================================================
def log_shadow_result(result: dict) -> None:

    # Hash the prompt to a BIGINT — avoids storing raw user input
    prompt_hash = int(
        hashlib.md5(result["prompt"].encode()).hexdigest(), 16
    ) % (2 ** 63)  # Truncate to PostgreSQL BIGINT max

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO shadow_evals
                    (ts, prompt_hash, prod_model, shadow_model,
                     prod_score, shadow_score, delta)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (
                datetime.now(timezone.utc),
                prompt_hash,
                result.get("prod_model"),
                result.get("shadow_model"),
                result.get("prod_score"),
                result.get("shadow_score"),
                result.get("delta"),
            ))
            conn.commit()


# ===============================================================
# get_recent_shadow_results
# ---------------------------------------------------------------
# Fetches shadow evaluation results from the past N days.
# Used by dashboards/app.py to render the shadow vs production panel.
#
# Parameters:
#   days (int) : Number of days to look back. Default: 7.
#
# Returns:
#   list[dict] : List of result dicts with keys:
#                  ts, prod_model, shadow_model,
#                  prod_score, shadow_score, delta
# ===============================================================
def get_recent_shadow_results(days: int = 7) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT ts, prod_model, shadow_model,
                       prod_score, shadow_score, delta
                FROM shadow_evals
                WHERE ts > NOW() - INTERVAL '%s days'
                ORDER BY ts ASC
            """, (days,))
            rows = cur.fetchall()

    return [
        {
            "ts"           : row[0],
            "prod_model"   : row[1],
            "shadow_model" : row[2],
            "prod_score"   : row[3],
            "shadow_score" : row[4],
            "delta"        : row[5],
        }
        for row in rows
    ]