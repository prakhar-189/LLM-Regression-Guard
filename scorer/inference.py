# scorer/inference.py
# ---------------------------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : Model-agnostic LLM inference wrapper built on LiteLLM
# 
#               Pipeline position - Stage 2 (Scorer):
#                   Called by scorer/run_scorer.py for every example
#                   in the golden dataset to get the model's response.
#                   Also called by app/tasks.py for shadow traffic scoring.
#
#               Why LiteLLM?
#                   LiteLLM provides a unified interfave over 100+ LLM 
#                   providers.
#                   Swapping from GPT-4o to Claude or Mistral requires 
#                   only changing the model string in config/model_version.yaml - 
#                   this file requires zero modification.
#
#               Temperature is set to 0.0 by default for determinism.
#               The same prompt should produce the same response across
#               repeatedruns, making score comparisions reliable.
# ---------------------------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# os           : Standard library — reads environment variables
#                (OPENAI_API_KEY, ANTHROPIC_API_KEY etc.).
# yaml         : PyYAML — reads config/model_version.yaml to get
#                the current model name without hardcoding it.
# load_dotenv  : -> python-dotenv — loads key=value pairs from .env
#                   file into environment variables at runtime.
#                -> Ensures API keys are available locally without
#                   being committed to the repository.
# completion   : -> LiteLLM's unified chat completion function.
#                -> Works identically for OpenAI, Anthropic, Mistral,
#                -> Cohere, local Ollama, and many more providers.
# ===============================================================
import os
import yaml
from dotenv import load_dotenv
from litellm import completion

# Load .env file into environment variables (no-op if already set)
load_dotenv()


# ===============================================================
# get_current_model
# ---------------------------------------------------------------
# Reads the current model name from config/model_version.yaml.
# This is the single source of truth for which model is under test.
#
# Returns:
#   str : Model string in LiteLLM format (e.g. "gpt-4o",
#         "claude-3-opus-20240229", "mistral/mistral-large").
#
# Why read from YAML instead of env var?
#   The YAML file is version-controlled, giving a full audit trail
#   of every model version change with timestamps and author info.
# ===============================================================
def get_current_model() -> str:
    cfg = yaml.safe_load(open("config/model_version.yaml"))
    return cfg.get("current_model", "gpt-4o")


# ===============================================================
# run_inference
# ---------------------------------------------------------------
# Calls the LLM with the given prompt and returns the response text.
#
# Parameters:
#   prompt        (str)   : The user prompt to send to the model.
#   model         (str)   : LiteLLM model string. If None, reads from
#                           model_version.yaml via get_current_model().
#   system_prompt (str)   : Optional system message prepended to the
#                           conversation. Used to test system prompt
#                           changes in the CI gate.
#   temperature   (float) : Sampling temperature. Defaults to 0.0 for
#                           maximum determinism across scoring runs.
#   max_tokens    (int)   : Maximum response length in tokens.
#
# Returns:
#   str : The model's response as a plain string (no metadata).
#
# Raises:
#   litellm.exceptions.* : Any LiteLLM / provider API errors are
#                          allowed to propagate — caught by run_scorer.py.
# ===============================================================
def run_inference(
    prompt        : str,
    model         : str     = None,
    system_prompt : str     = None,
    temperature   : float   = 0.0,
    max_tokens    : int     = 512,
) -> str:
    
    # Use model_version.yaml value if no model explicitly provided
    model = model or get_current_model()

    # Build the messages list for the chat completion API
    messages =[]
    if system_prompt:
        # System message sets the assistant's persona/instructions
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Call the LLM via LiteLLM's unified interface
    response = completion(
        model       = model,
        messages    = messages,
        temperature = temperature,
        max_tokens  = max_tokens,
    )

    # Extract & return just the text content from the response object
    return response.choices[0].message.content