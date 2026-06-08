# dataset/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'dataset' module.
#               Makes this directory a Python package so that other
#               modules can import from it using:
#                   from dataset.schema    import GoldenExample
#                   from dataset.validator import validate_dataset
#
# Modules in this package:
#   schema.py    : Pydantic model defining a single golden example.
#   validator.py : Dataset-level checks for category minimum counts.
# --------------------------------------------------------------