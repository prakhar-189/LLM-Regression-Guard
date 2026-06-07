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
# Field : A function used to provide additional validation and metadata for model fields.
# field_validator : A decorator used to define custom validation logic for model fields.
# Dict : A type hint for dictionaries in Python.
# Any : A type hint that can represent any type of data.
# ===============================================================
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any


# ===============================================================
# GoldenExample Class
# ---------------------------------------------------------
# This class represents a single example in the GoldenExample dataset.
# It includes fields for the example's ID, input prompt, reference response, category, source, & metadata.
# The class also includes validation logic to ensure that the category field contains a valid value.
# ===============================================================
class GoldenExample(BaseModel):
    id : str
    Prompt : str = Field(..., min_length = 10, description = "The input prompt must be at least 10 characters.")
    reference_response : str = Field(..., description = "The ideal target response.")
    category :str
    source : str = "manual"
    metadata : Dict[str, Any] = Field(default_factory=dict)

    # Custom validator for the 'category' field to ensure it contains a valid value.
    @field_validator('category')
    def validate_category(cls, value):
        allowed_categories = [
            'factual_accuracy',
            'refusal_behavior',
            'instruction_following',
            'tone_consistency',
            'multi_turn_coherence'
        ]

        if value not in allowed_categories:
            raise ValueError(f"Category '{value}' is invalid. Must be one of {allowed_categories}")
        return value