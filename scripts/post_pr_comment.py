# scripts/post_pr_comment.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-09
# Description : -> Posts a formatted quality scorecard as a comment
#                  directly on the GitHub Pull Request.
#               -> Pipeline position - Stage 3 (CI/CD Gate):
#                    - The final step in quality_gate.yaml.
#                    - Runs with 'if: always()' - meaning it executes
#                      whether the CI gate passed OR failed, so reviewers
#                      always see the current quality snapshot on the PR.
#
#               -> What it posts:
#                    - A markdown table showing per-category scores with
#                      pass/fail emoji, overall score, model version,
#                      run timestamp, and a clear gate pass/fail message.
# 
#               -> GitHub API:
#                    - Uses the Issues Comments endpoint (PRs are Issues
#                      in GitHub's API):
#                          POST/repos/{owner}/{repo}/issues/{pr_number}/
#                          comments
#                          Authenticated with the auto-provided GITHUB_TOKEN
#                          secret.
# --------------------------------------------------------------


# ==============================================================
# Imports
# ---------------------------------------------------------
# json        : Standard Library - loads scores_output/scores.json.
# os          : Standard Library - reads GITHUB_TOKEN, GITHUB_REPOSITORY,
#                                  PR_NUMBER environment variables (set by Github Actions).
# sys         : Standard Library - Used to print warnings & exists.
# Path        : pathlib - checks if scores.json exists.
# requests    : HTTP clint library - makes the POST request to the Github API.
#               Simpler than httpx fot this straightforward one-shot request.
# load_dotenv : python-dotenv - loads .env for local testing.
# ==============================================================
import json
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()


# ==============================================================
# post_pr_comment
# ---------------------------------------------------------
# Builds a Markdown scorecard from scores.json & posts it as
# a comment on the GitHub via. the GitHub REST API.
#
# Parameters:
#     scores_path (str) : Path to the scores JSON file written by run_scorer.py
#
# Returns:
#     None
#
# Behaviour when runninng locally (no GitHub env vars):
#     Falls back to printing the scorecard to stdout instead of 
#     making an API call - allows local testing without a real PR.
#
# Required environment vriables (autoset-set by GitHub Actions):
#     GITHUB_TOKEN      : Personal access token with repo write scope.
#     GITHUB_REPOSITORY : "username/repo-name" format.
#     PR_NUMBER         : The pull request number (integer as string).
# ==============================================================
def post_pr_comment(scores_path : str = "scores_output/scores.json") -> None:

    print("Posting quality scorecard to GitHub PR...")

    # Guard: scores.json must exist (written by run_scorer.py)
    if not Path(scores_path).exists():
        print(f"WARNING: '{scores_path}' not found - skipping PR comment.")
        return
    
    scores = json.load(open(scores_path))

    # Read GitHub Actions environment variables
    token     = os.environ.get("GITHUB_TOKEN")
    repo      = os.environ.get("GITHUB_REPOSITORY")
    pr_number = os.environ.get("PR_NUMBER")

    # ----------------------------------------------------------
    # Local Fallback
    # ----------------------------------------------------------
    # When running locally (not in GitHub Actions), env vars are
    # absent. Print the scorecard to stdout instad of falling.
    # ----------------------------------------------------------
    if not all([token, repo, pr_number]):
        print("WARNING: GitHub env vars not set - printing scorecard locally")
        _print_scorecard(scores)
        return
    
    # ----------------------------------------------------------
    # Build the markdown scorecard body
    # ----------------------------------------------------------
    overall       = scores["overall"]
    overall_emoji = "✅" if overall >= 0.8 else "❌"

    # One table row per category with pass/fail emoji
    rows = []
    for cat, score in scores["categories"].items():
        emoji       = "✅" if score >= 0.72 else "❌"
        cat_display = cat.replace("_","").title()
        rows.append(f"| {cat_display} | {score:.4f} | {emoji} |")

    rows_md  = "\n".join(rows)
    gate_msg =(
        "✅ **Quality gate PASSED** — PR is approved for merge."
        if overall >= 0.80 else
        "❌ **Quality gate FAILED** — Regression detected. Fix before merging."
    )

    body = f"""## 🔍 LLM Quality Gate Report
 
**Model:** `{scores.get('model_version', 'unknown')}`
**Run timestamp:** `{scores.get('run_timestamp', 'N/A')}`
**Examples evaluated:** `{scores.get('total_examples', 'N/A')}`
 
### Per-Category Scores
 
| Category | Score | Status |
|---|---|---|
{rows_md}
| **Overall** | **{overall:.4f}** | **{overall_emoji}** |
 
> Thresholds: overall ≥ 0.80 · per-category ≥ 0.72 · max regression delta: 0.05
 
{gate_msg}
"""
 
    # ----------------------------------------------------------
    # POST to GitHub Issues Comments API
    # ----------------------------------------------------------
    # GitHub treats PR comments the same as issue comments in the API.
    # Authorization header uses the GITHUB_TOKEN secret provided
    # automatically by GitHub Actions — no manual token setup needed.
    # ----------------------------------------------------------
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
 
    response = requests.post(
        url,
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept"       : "application/vnd.github+json",
        },
        json = {"body": body},
    )
 
    if response.status_code == 201:
        print(f"Scorecard posted successfully to PR #{pr_number}")
    else:
        print(f"GitHub API returned {response.status_code}: {response.text}")
 
 
# ===============================================================
# _print_scorecard
# ---------------------------------------------------------------
# Pretty-prints the scorecard to stdout.
# Used as a local fallback when GitHub env vars are not available.
#
# Parameters:
#   scores (dict) : The parsed scores.json dict.
# ===============================================================
def _print_scorecard(scores: dict) -> None:
    print(f"\n{'='*40}")
    print(f"Overall score : {scores['overall']:.4f}")
    print(f"{'='*40}")
    for cat, score in scores["categories"].items():
        status = "PASS" if score >= 0.72 else "FAIL"
        print(f"  {cat:<30} {score:.4f}  [{status}]")
    print(f"{'='*40}\n")
 
 
# ===============================================================
# Script entry point
# ---------------------------------------------------------------
# Called by .github/workflows/quality_gate.yml as the final step.
# ===============================================================
if __name__ == "__main__":
    post_pr_comment()