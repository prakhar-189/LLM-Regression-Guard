# app/router.py
# -----------------------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-09
# Description : -> FastAPI route definitions for the LLM application.
#               -> Kept seperate from app/main.py so application setup (middleware,
#                  startup events) & business logic (endpoint handlers) are cleanly
#                  decoupled.
#
#               -> Endpoints:
#                     - POST/chat  : Main LLM inference endpoint.
#                                    Accepts a prompt & returns the model's response.
#                                    The ShadowMiddleware in main.py silently
#                                    intercepts 5% of calls here.
#                     - GET/health : Liveness probe for Docker/Kubernates.
#                                    Returns {"status" : "ok"} when the app is up.
#
#               -> Why speprate from main.py?
#                     - In tests, we can import just the router & mount it on a
#                       TestClient without pulling in the middleware or any other
#                       infrastructure - making unit tests fast & isolated.
# -----------------------------------------------------------------------------


# =============================================================================
# Imports
# ------------------------------------------------------------------------
# APIRouter     : -> FastAPI's modular router - groups related endpoints.
#                 -> Included in main.py via app.include_router(rouuter). 
# HTTPException : Raises HTTP error responses (400, 500) with a detail message.
# BaseModel     : -> Pydantic model used to define & validate request/response bodies.
#                 -> FastAPI uses these for automatic JSON serialization & OpenAI schema generation.
# sys, Path     : Standard Library - adds project root to sys.path so scorer.inference
#                 is importable from this module.
# run_inference : scorer/infernce.py - calls the LLM & returns response text.
# yaml          : PyYAML - reads model_version.yml for the current model name.
# =============================================================================
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import sys, yaml
from pathlib import Path
from scorer.inference import run_inference

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

router = APIRouter()


# ============================================================================
# ChatRequest
# -----------------------------------------------------------------------
# Pydantic request model for the POST/chat endpoint.
# FastAPI automatically validates incoming JSON against this schema & returns 
# HTTP 422 if the body doesn't match.
#
# Fields:
#   -> prompts        : The user's input text to send to the LLM. Required
#   -> system_prompts : Optional system message that sets the LLM's persona or
#                       instructions. Changing this triggers the CI gate.
#   -> model          : Optional model override. If None, uses the value from
#                       config/model_version.yaml.
# =============================================================================
class ChatRequest(BaseModel):
    prompt        : str
    system_prompt : str = None
    model         : str = None


# =============================================================================
# ChatResponse
# ------------------------------------------------------------------------
# Pydantic response model for the POST/chat endpoint.
# FastAPI uses this to serialize the return vallue to JSON and to generate
# the OpenAPI response schema at /docs.
# 
# Fields:
#    response : The LLM's generated text response
#    model    : The model that generated the response (for audit trail)
#    prompt   : Echo of the original prompt (for client-side logging)
# =============================================================================
class ChatResponse(BaseModel):
    response : str
    model    : str
    promot   : str


# =============================================================================
# chat — POST /chat
# ------------------------------------------------------------------------
# Main LLM inference endpoint. Validates the request, calls the
# LLM via run_inference(), and returns the structured response.
#
# Parameters:
#     request (ChatRequest) : Parsed and validated request body.
#
# Returns:
#     ChatResponse : JSON object with response text, model name, and prompt.
#
# Raises:
#     HTTP 400 : If the prompt is empty or whitespace only.
#     HTTP 500 : If the LLM call fails for any reason (API error, timeout).
#
# Note:
#     The ShadowMiddleware in main.py intercepts this request BEFORE
#     it reaches this handler for 5% of calls. The interception is
#     invisible from the handler's perspective — it only affects the
#     async background task, not this function's execution.
# =============================================================================
@router.post("/chat", response_model = ChatResponse)
async def chat(request : ChatRequest):
    # Validate prompt is not empty
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code = 400, detail = "Prompt cannot be empty.")
    
    try:
        response_text = run_inference(
            prompt        = request.prompt,
            model         = request.model,
            system_prompt = request.system_prompt,
        )

        # Read current model name for the response metadata
        model_cfg     = yaml.safe_load(open("config/model_version.yaml"))
        current_model = request.model or model_cfg.get("current_model", "gpt-4o")

        return ChatResponse(
            response = response_text,
            model    = current_model,
            prompt   = request.prompt,
        )
    
    except Exception as e:
        raise HTTPException(status_code = 500, detail = str(e))
    

# ===============================================================
# health — GET /health
# ---------------------------------------------------------------
# Liveness probe endpoint used by Docker Compose health checks
# and Kubernetes readiness probes.
# Returns HTTP 200 with {"status": "ok"} when the app is running.
# ===============================================================
@router.get("/health")
async def health():
    return {"status": "ok"}