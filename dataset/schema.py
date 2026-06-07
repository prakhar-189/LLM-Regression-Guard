# dataset/schema.py
# --------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : This module defines the schema for the GoldenExample dataset using Pydantic.
# --------------------------------------------------------------


# ===============================================================
# Imports
# ----------------------------------------------------------
# BaseModel : The base class for creating data models in Pydantic.
# field_validator : A decorator used to define custom validation logic for model fields.
# Optional : A type hint indicating that a field can be of a specified type or None.
# Literal : A type hint that restricts a field to have one of a specified set of literal values.
# ===============================================================
from pydantic import BaseModel, field_validator
from typing import Optional, Literal


# List of valid categories for the GoldenExample dataset.
CATEGORIES = [
    "factual_accuracy",
    "refusal_behaviour",
    "instruction_following",
    "tone_consistency",
    "multi_turn_coherence"
]


# ===============================================================
# GoldenExample Class
# ---------------------------------------------------------
# This class represents a single example in the GoldenExample dataset.
# It includes fields for the example's ID, input prompt, reference response, category, source, & metadata.
# The class also includes validation logic to ensure that the category field contains a valid value.
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

    # Validators to ensure that the prompt is sufficiently long and the reference response is not empty.
    @field_validator("prompt")
    @classmethod
    def prompt_not_empty(cls, v):
         if len(v.strip()) <=10:
              raise ValueError("Prompt must be longer than 10 characters.")
         return v
    
    # Validator to ensure that the reference response is not empty.
    @field_validator("reference_response")
    @classmethod
    def reference_not_empty(cls, v):
         if len(v.strip()) == 0:
              raise ValueError("Reference response cannot be empty.")
         return v