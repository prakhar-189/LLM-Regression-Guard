# scripts/build_dataset.py
# ------------------------------------------------------------------
# Author : Prakhar Srivastava
# Date : 2026-06-07
# Description : Entry point for constructing and validating the golden
#               evaluation dataset used by the scoring pipeline.
#
#               Pipeline position — Stage 1 (Dataset Builder):
#                 Reads  : data/raw/prompts.jsonl
#                 Writes : data/eval/golden_dataset.jsonl
#                 Config : config/dataset_config.yaml
#
#               Steps performed:
#                 1. Load raw JSONL entries from prompts.jsonl
#                 2. Validate each entry against the GoldenExample schema
#                 3. Enforce per-category minimum example counts
#                 4. Write all validated examples to golden_dataset.jsonl
#
#               DVC caches the output — if prompts.jsonl has not changed
#               since the last run, `dvc repro` skips this script entirely.
#
# Usage:
#   python scripts/build_dataset.py
# --------------------------------------------------------------
 
 
# ===============================================================
# Imports
# ---------------------------------------------------------------
# json      : Standard library — parses each JSONL line into a dict.
# sys       : Standard library — used to exit with code 1 on errors
#             (signals CI failure) and to modify sys.path for imports.
# yaml      : PyYAML — reads dataset_config.yaml into a Python dict.
# Path      : pathlib — cross-platform file path handling.
# datetime  : Standard library — timestamps in build log output.
#
# GoldenExample   : Pydantic model from dataset/schema.py.
#                   Instantiating it triggers all field validation.
# validate_dataset: Category-level minimum count checker from dataset/validator.py.
# ValidationError : Raised by Pydantic when a field fails validation.
#                   Caught here to produce clean error messages.
# ===============================================================
import json
import sys
import yaml
from pathlib import Path
from dataset.schema import GoldenExample
from dataset.validator import validate_dataset
from pydantic import ValidationError


# Add project root to path so dataset package is importable
# when this script is run directly (not as a module).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ===============================================================
# build_dataset
# ---------------------------------------------------------------
# Main function that orchestrates the full dataset construction pipeline.
#
# Parameters:
#   raw_path    (str) : Path to the raw JSONL input file.
#                       Default: "data/raw/prompts.jsonl"
#   output_path (str) : Path where the validated dataset will be written.
#                       Default: "data/eval/golden_dataset.jsonl"
#   cfg_path    (str) : Path to the dataset configuration YAML file.
#                       Default: "config/dataset_config.yaml"
#
# Returns:
#   None
#
# Exits:
#   sys.exit(1) if any schema validation errors are found OR if
#   category minimum counts are not met. This exit code causes the
#   DVC/CI pipeline to halt and report failure.
# ===============================================================
def build_dataset(
    raw_path : str = "data/raw/prompts.jsonl",
    output_path : str = "data/eval/golden_dataset.jsonl",
    cfg_path : str = "config/dataset_config.yaml",
) -> None:
    print("=" * 50)
    print("Building golden evaluation dataset....")
    print("=" * 50)

    # ----------------------------------------------------------
    # Step 1 : Load configuration
    # ----------------------------------------------------------
    # dataset_config.yaml defines the category list, per-category
    # minimum example counts, and scoring weights.
    # ----------------------------------------------------------
    cfg = yaml.safe_load(open(cfg_path))

    # ----------------------------------------------------------
    # Step 2 : Load raw JSONL entries
    # ----------------------------------------------------------
    # Each line in prompts.jsonl is one JSON object representing
    # a single prompt-response pair with its category label.
    # ----------------------------------------------------------
    raw_lines = Path(raw_path).read_text().strip().splitlines()
    print(f"\nLoaded {len(raw_lines)} raw entries from {raw_path}")

    # ----------------------------------------------------------
    # Step 3 : Validate each entry against the Pydantic schema
    # ----------------------------------------------------------
    # GoldenExample(**raw) triggers all Pydantic field validators:
    #   - prompt length > 10 characters
    #   - reference_response not blank
    #   - category is one of the five valid Literal values
    #
    # Invalid entries are collected and reported together so the
    # developer can fix all errors in one pass rather than one at a time.
    # ----------------------------------------------------------
    examples = []
    errors = []
    for i, line in enumerate(raw_lines, 1):
        try:
            raw = json.loads(line)
            example = GoldenExample(**raw)
            examples.append(example)
        except ValidationError as e:
            # Extract the first error message for readability
            errors.append(f"Line {i}: {e.errors()[0]['msg']} - {line[:80]}")
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: Invalid JSON - {e}")
    
    # Halt if any individual entry failed schema validation
    if errors:
        print("\nSchema validation errors:")
        for err in errors:
            print(f" ERROR: {err}")
        sys.exit(1)

    print(f"Schema validation passed for all {len(examples)} entries.")

    # ----------------------------------------------------------
    # Step 4 : Validate per-category minimum example counts
    # ----------------------------------------------------------
    # Ensures no category is under-represented in the dataset.
    # Raises ValueError (caught here) if any category falls short.
    # ----------------------------------------------------------
    try:
        validate_dataset(examples, cfg)
    except ValueError as e:
        print(f"\n{e}")
        sys.exit(1)
 
    # ----------------------------------------------------------
    # Step 5 : Write the validated golden dataset to disk
    # ----------------------------------------------------------
    # Each GoldenExample is serialized back to a JSON string (one per line)
    # using Pydantic's .model_dump_json() which handles all type conversions.
    # The output directory is created if it doesn't exist yet.
    # ----------------------------------------------------------
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
 
    with output.open("w") as f:
        for ex in examples:
            f.write(ex.model_dump_json() + "\n")
 
    print(f"\nGolden dataset written to '{output_path}'")
    print(f"Total examples : {len(examples)}")
    print("=" * 55)
 
 
# ===============================================================
# Script entry point
# ---------------------------------------------------------------
# Allows this file to be run directly:
#   python scripts/build_dataset.py
#
# When imported as a module (e.g. in tests), this block is skipped.
# ===============================================================
if __name__ == "__main__":
    build_dataset()