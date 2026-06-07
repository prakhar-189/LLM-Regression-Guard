# dataset/validator.py
# --------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : This module provides a function to validate the GoldenExample dataset against the configuration defined in dataset_config.yaml.
# --------------------------------------------------------------


# ==============================================================
# Imports
# -------------------------------------------------------------
# Counter : A class from the collections module used to count hashable objects.
# List : A type hint for lists in Python.
# GoldenExample : The data model for examples in the GoldenExample dataset, defined in dataset/schema.py.
# ==============================================================
from collections import Counter
from typing import List
from dataset.schema import GoldenExample


# ==============================================================
# validate_dataset Function
# -------------------------------------------------------------
# This function takes a list of GoldenExample instances and a configuration dictionary, and validates that each category in the dataset has at least the minimum number of examples specified in the configuration.
# If any category is under-represented, it raises a ValueError with details about the missing examples. 
# If the validation passes, it prints a summary of the dataset composition.
# ==============================================================
def validate_dataset(examples: List[GoldenExample], cfg: dict) -> None:
    """
    Validates that each category in the dataset has the minimum number of examples defined in dataset_config.yaml.
    Raises ValueError if any category is under represented.
    """
    
    counts = Counter(e.category for e in examples)
    errors = []
    
    # Check each category against the required minimum and collect errors for under-represented categories.
    for cat in cfg["categories"]:
        name = cat["name"]
        required = cat["min_examples"]
        actual = counts.get(name, 0)

        if actual < required:
            errors.append(
                f" - '{name}': found {actual}, need {required}"
            )
    
    # If there are any errors, raise a ValueError with the details. Otherwise, print a summary of the dataset composition.
    if errors:
        raise ValueError(
            "Dataset validation failed - under-represented categories:\n" + "\n".join(errors)
        )
    
    print(f"Dataset validation passed. {len(examples)} total examples.")
    for cat in cfg["categories"]:
        name = cat["name"]
        print(f" {name}: {counts.get(name, 0)} examples")