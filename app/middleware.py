# app/middleware.py
# ------------------------------------------------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-10
# Description : -> Shadow traffic routing middleware.
#               -> Intercepts a configurable percentage of /chat requests & dispatches them to a Celery
#                  background task for async quality scoring - without adding any latency to the production
#                  respinse path.
#
#               -> Pipeline Position - Stage 5 (Shadow Traffic Monitor):
#                     - Registered in app/main.py via. app.add_middleware().
#                     - Wraps every HTTP request before it reaches the router.
#
#               -> How shadow routing works?:
#                     1. For every POST /chat request, read the request body.
#                     2. Compute MD5 hash of the body.
#                     3. If hash % 100 < shadow_pct (e.g. 5), fire a Celery task.
#                     4. IMMEDIATELY call call_nect() to continue the normal production request flow - user
#                        gets their response with zero added latency.
#                     5. The Celery task runs asychrounously in a seperate worker process, scoring both the
#                        production & shadow model responses & logging the delta to TimescaleDB.
#
#               -> Why hash-based routing instead of random?
#                     1. Hash-based routing is deterministic - the same prompt ALWAYS routes to the same
#                        model variant (5% or 95%).
#                     2. This makes score comparisions reproducible & eliminates sampling bias when comparing
#                        shadow v/s production scores.
# ------------------------------------------------------------------------------------------------------


# ======================================================================================================
# Imports
# -------------------------------------------------------------------------------------------------
# hashlib            : Standard library - computes MD5 hash of the request body fot deterministic routing decisions.
# BaseHTTPMiddleware : Starlette's base class for ASGI middleware.
#                      Subclass it & override dispatch() to intercept requests before they reach FastAPI route handlers.
# Request            : Starlette's request object - provides access to method, URL path, & body bytes.
# ======================================================================================================
import hashlib
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# ======================================================================================================
# ShadowMiddleware
# -------------------------------------------------------------------------------------------------
# ASGI middleware that implements the shadow traffic routing logic.
# Registered on the FastAPI app in app/main.py
#
# Constructors parameters:
#    -> app        : The ASGI app to wrap (provided by Starlette automatically).
#    -> shadow_pct : Percentage of requests to shadow (default : 5)
#                    Configurable in config/shadow_config.yaml.
# ======================================================================================================
class ShadowMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, shadow_pct: int=5):
        super().__init__(app)
        self.shadow_pct = shadow_pct

        # ----------------------------------------------------------------
        # dispatch
        # ----------------------------------------------------------------
        # Called for every incoming HTTP request before it reaches any route handler.
        #
        # Parameters:
        #    -> request (Request)    : The incoming HTTP request object.
        #    -> call_next (callable) : Function that passes the request to the next
        #                              handler in the chain (the router).
        #
        # Returns:
        #    -> Response : The production response returned to the user.
        #                  Identical whether or not shadow routing fired - 
        #                  the user sees no difference.
        #
        # Critical design detail:
        #    -> We await request.body() ONCE to read the body bytes. then reconstruct
        #       a new recieve() callable to replay those bytes downstream.
        #    -> Without this, the body stream would be consumed & the route handler
        #       recieve an empty body.
        # ----------------------------------------------------------------
        async def dispatch(self, request: Request, call_next):
            # Only apply shadow routing to POST /chat requests
            if request.method == "POST" and request.url.path == "/chat":
                body = await request.body()

                # -------------------------------------------------
                # Hash-based routing decision
                # -------------------------------------------------
                # MD5 of the body bytes -> integer -> modulo 100.
                # Value 0-4 (5%) trigger shadow scoring.
                # Same body always produces same hash -> same decision.
                # -------------------------------------------------
                body_hash = int(hashlib.md5(body).hexdigest(), 16)

                if body_hash % 100 < self.shadow_pct:
                    try:
                        # Fire-&-forget: Celery task handles everything
                        # asynchronously: This line returns immediately - no waiting for the shadow score.
                        from app.tasks import score_shadow_async
                        score_shadow_async.delay(body.decode("utf-8"))

                    except Exception:
                        # Never let shadow errors propagate to the user.
                        # Shadow monitoring is best-effort - production reliablilty takes absolute priority.
                        pass

                    # ----------------------------------------------------------
                    # Reconstruct the body request body stream
                    # ----------------------------------------------------------
                    # Since we consumed body abpve with await request.boy(),
                    # we must rebuild a recieve() callable that replays the bytes so
                    # the downstream route handler can read them.
                    # Without this, request.json() in the route handler would return empty/fail.
                    # ----------------------------------------------------------
                    async def recieve():
                        return {"type": "http.request", "body": body}
                    
                    request = Request(request.scope, recieve)

                # Pass the (possibly reconstructed) request to the route handler
                return await call_next(request)

