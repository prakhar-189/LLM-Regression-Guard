# dataset/schema.py
# --------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : -> Defines the Pydantic data model(GoldenExample) that represents 
#                  a single entry in the golden evaluation dataset.
#               -> Every prompt-response pair loaded from data/raw/prompts.jsonl
#                  is validated against this schema before being written to
#                  data/eval/golden_dataset.jsonl.
#               -> If any fields fails validation, the dataset build is halted 
#                  immediately - acting as the first quality gate in the pipeline.
# --------------------------------------------------------------


# ===============================================================
# Imports
# ----------------------------------------------------------
# BaseModel : -> Base class from Pydantic for refining data models.
#             -> Handles type coercion, validation & serialization.
# field_validator : -> Decorator for attaching custom validation logic
#                      to individual fields beyond basic type checking.
# Optional : Type hint -> field can be the specified type OR None.
# Literal : Type hint -> restricts a field to a fixed set of allowed
#           allowed string values. Used to enforce valid categories.
# ===============================================================
from pydantic import BaseModel, field_validator
from typing import Optional, Literal


# ===============================================================
# CATEGORIES
# ----------------------------------------------------------
# The five behavioural ddimensions used to evaluate the LLM.
# These map directly to the catgory names used in dataset_config.yaml.
# 
#   factual_accurcy      : Does the model return correct, verifiable facts?
#   refusal_behavior     : Does the model correctly refuse harmful requests?
#   instruction_following: Does the model follow explicit formatting rules?
#   tone_consistency     : Does the model match the required tone/audience?
#   multi_turn_coherence : Does the model stay consistent across a conversation?
#
# Adding a new category requires:
#   1. Adding it here in CATEGORIES
#   2. Adding it to the Literal type hint in GoldenExample.category
#   3. Adding it to config/dataset_config.yaml
# ================================================================
CATEGORIES = [
    "factual_accuracy",
    "refusal_behaviour",
    "instruction_following",
    "tone_consistency",
    "multi_turn_coherence"
]


# ===============================================================
# GoldenExample
# ---------------------------------------------------------------
# Pydantic model representing one entry in the golden evaluation dataset.
# Each instance is one (prompt, reference_response, category) triplet
# that the scoring pipeline uses to evaluate the LLM under test.
#
# Fields:
#   id                 : Unique identifier for this example (e.g. "f001").
#   prompt             : The input sent to the LLM during evaluation.
#   reference_response : The ideal/expected output for this prompt.
#                        Used as ground truth by BERTScore, ROUGE-L, and
#                        the judge LLM rubric.
#   category           : One of the five behavioral categories above.
#                        Must exactly match one of the Literal values.
#   source             : Where this example came from ("manual", "synthetic").
#                        Defaults to "manual".
#   metadata           : Optional dict for extra info (difficulty, source_url).
#                        Not used by the scorer — for human reference only.
#
# Validation:
#   - prompt must be longer than 10 characters (catches empty/trivial prompts)
#   - reference_response cannot be blank (catches accidental empty strings)
#   - category must be one of the five Literal values (enforced by Pydantic)
# ===============================================================
class GoldenExample(BaseModel):
    id : str
    prompt : str
    reference_response : str
    category : Literal[
        'factual_accuracy',
        'refusal_behavior',
        'instruction_following',
        'tone_consistency',
        'multi_turn_coherence'
    ]
    source : str = "manual"
    metadata : Optional[dict] = None

    # -----------------------------------------------------------
    # prompt_not_empty
    # -----------------------------------------------------------
    # Validator for the 'prompt' field.
    # Strips whitespace before checking length so that prompts
    # containing only spaces are also rejected.
    # Raises ValueError (caught by Pydantic as ValidationError)
    # if the stripped prompt is 10 characters or shorter.
    #
    # Why 10 chars? Prompts shorter than this are almost always
    # accidental data entry errors (e.g. "Hi", "Test", "?").
    # -----------------------------------------------------------
    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
         if len(v.strip()) <=10:
              raise ValueError("Prompt must be longer than 10 characters.")
         return v
    
    # -----------------------------------------------------------
    # reference_not_empty
    # -----------------------------------------------------------
    # Validator for the 'reference_response' field.
    # Ensures the reference is not blank or whitespace-only.
    # An empty reference would produce meaningless BERTScore and
    # ROUGE-L values and mislead the judge LLM rubric.
    # -----------------------------------------------------------
    @field_validator("reference_response")
    @classmethod
    def reference_not_empty(cls, v):
         if len(v.strip()) == 0:
              raise ValueError("Reference response cannot be empty.")
         return v