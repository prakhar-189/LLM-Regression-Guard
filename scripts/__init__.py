# scripts/__init__.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Package initializer for the 'scripts' module.
#               Makes this directory a Python package so that test
#               files can import from it directly:
#                   from scripts.check_regression import check_regression
#
# Modules in this package:
#   build_dataset.py    : Builds and validates the golden evaluation dataset.
#   check_regression.py : CI gate — compares scores to baseline, exits 0/1.
#   post_pr_comment.py  : Posts quality scorecard to GitHub PR as a comment.
# --------------------------------------------------------------