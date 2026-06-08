# dataset/validator.py
# --------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : -> Dataset-level validation logic that enforces category
#                  minimum example counts as defined in dataset_config.yaml.
#               -> While schema.py validates individual GoldenExample records,
#                  this module validates the dataset as a whole — ensuring
#                  that no behavioral category is under-represented before
#                  the scoring pipeline runs.
#               -> An imbalanced dataset produces unreliable per-category
#                  averages (e.g. 2 refusal examples averaging to 0.95 looks
#                  great but means nothing). This check prevents that.
#
#               Called by: scripts/build_dataset.py
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# Counter  : -> dict subclass from collections that counts hashable objects.
#            -> Used to tally how many examples exist per category.
# List     : Type hint for a list of GoldenExample objects.
# GoldenExample : -> The Pydantic model from dataset/schema.py.
#                 -> Imported here only for the type annotation.
# ===============================================================
from collections import Counter
from typing import List
from dataset.schema import GoldenExample


# ===============================================================
# validate_dataset
# ---------------------------------------------------------------
# Checks that every category in dataset_config.yaml['categories']
# has at least 'min_examples' entries in the provided examples list.
#
# Parameters:
#   examples (List[GoldenExample]) : All validated GoldenExample objects
#                                    loaded from data/raw/prompts.jsonl.
#   cfg      (dict)                : Parsed dataset_config.yaml contents.
#                                    Must contain a 'categories' key with
#                                    a list of {name, min_examples, weight}.
#
# Returns:
#   None — prints a success summary if all checks pass.
#
# Raises:
#   ValueError : If one or more categories have fewer examples than
#                required. The error message lists every failing category
#                with its actual vs required count.
#
# Pipeline position:
#   Called after all individual records pass schema validation.
#   If this raises, scripts/build_dataset.py exits with code 1
#   and golden_dataset.jsonl is NOT written.
# ===============================================================
def validate_dataset(examples: List[GoldenExample], cfg: dict) -> None:
    
    # Count how many examples exist per category in the provided list
    counts = Counter(e.category for e in examples)

    # Collect every category that falls below its minimum requirement
    errors = []
    for cat in cfg["categories"]:
        name = cat["name"]
        required = cat["min_examples"]
        actual = counts.get(name, 0)

        if actual < required:
            errors.append(
                f" - '{name}': found {actual}, need {required}"
            )
    
    # If any category is under-represented, raise with a full error report
    if errors:
        raise ValueError(
            "Dataset validation failed - under-represented categories:\n" + "\n".join(errors)
        )
    
    # All checks passed — print a summary for build log visibility
    print(f"Dataset validation passed. {len(examples)} total examples.")
    for cat in cfg["categories"]:
        name = cat["name"]
        print(f" {name}: {counts.get(name, 0)} examples")