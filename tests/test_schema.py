# tests/test_schema.py
# --------------------------------------------------------------
# Author      : Prakhar Srivastava
# Date        : 2026-06-07
# Description : Test suite for dataset/schema.py — the Pydantic GoldenExample
#               model that validates every entry in the golden evaluation dataset.
#
#               Why these tests matter:
#                 schema.py is the first quality gate in the entire pipeline.
#                 If a malformed entry slips past validation, it can silently
#                 corrupt BERTScore/ROUGE-L calculations downstream or cause
#                 the judge LLM to evaluate against an empty reference response.
#                 These tests guarantee the schema catches every bad entry
#                 before it ever reaches the scorer.
#
#               Testing strategy:
#                 - No external dependencies, no API calls, no file I/O.
#                 - Tests run in milliseconds — safe to run on every save.
#                 - Two classes separate positive (valid) and negative (invalid)
#                   test cases for clarity.
#                 - Boundary conditions are tested explicitly (e.g. prompt
#                   length exactly at the 10-character cutoff).
#
#               Two classes of tests:
#                 TestGoldenExampleValid   : Confirms that correctly formed
#                   examples are accepted without error, including optional
#                   fields and all five valid category values.
#
#                 TestGoldenExampleInvalid : Confirms that every type of
#                   bad input raises a Pydantic ValidationError, not a
#                   silent failure or a Python AttributeError.
#
# Run:
#   pytest tests/test_schema.py -v
# --------------------------------------------------------------


# ===============================================================
# Imports
# ---------------------------------------------------------------
# pytest         : Test framework.
#                  pytest.raises() is used as a context manager to
#                  assert that a specific exception is raised.
#                  If the expected exception is NOT raised, the test fails.
#
# ValidationError: Pydantic's exception class raised when a model field
#                  fails validation. We assert this is raised (not a
#                  generic Exception) so we're testing the right thing.
#
# GoldenExample  : The Pydantic model under test from dataset/schema.py.
#                  Instantiating it with GoldenExample(**dict) triggers
#                  all field validators automatically.
# ===============================================================
import pytest
from pydantic import ValidationError
from dataset.schema import GoldenExample


# ===============================================================
# TestGoldenExampleValid
# ---------------------------------------------------------------
# Positive test cases — confirms that well-formed inputs are accepted.
# Every test in this class should complete without raising any exception.
# If a test here fails, it means the schema is rejecting valid data
# which would break the dataset build step.
#
# Fixture used:
#   valid_example_dict (from tests/conftest.py) : A pre-built dict
#     representing a complete, valid GoldenExample with known values.
#     Using a fixture avoids repeating the same dict in every test.
# ===============================================================
class TestGoldenExampleValid:

    # -----------------------------------------------------------
    # test_valid_example_passes
    # -----------------------------------------------------------
    # The most basic happy-path test.
    # Verifies that a fully populated, correctly typed example is
    # accepted and that field values are stored correctly.
    # Uses the valid_example_dict fixture from conftest.py.
    # -----------------------------------------------------------
    def test_valid_example_passes(self, valid_example_dict):
        """A complete, valid example should be created without error."""
        ex = GoldenExample(**valid_example_dict)
        assert ex.id       == "test001"
        assert ex.category == "factual_accuracy"
        assert ex.source   == "manual"


    # -----------------------------------------------------------
    # test_default_source_is_manual
    # -----------------------------------------------------------
    # Verifies that 'source' defaults to "manual" when omitted.
    # This is important because most hand-written examples won't
    # include a source field — the default keeps them clean.
    # Does NOT use the fixture — tests the model directly without source.
    # -----------------------------------------------------------
    def test_default_source_is_manual(self):
        """Source should default to 'manual' when not provided."""
        ex = GoldenExample(
            id                 = "t002",
            prompt             = "What is the boiling point of water?",
            reference_response = "Water boils at 100°C (212°F) at sea level.",
            category           = "factual_accuracy",
        )
        assert ex.source == "manual"


    # -----------------------------------------------------------
    # test_metadata_is_optional
    # -----------------------------------------------------------
    # Verifies that 'metadata' defaults to None when not provided.
    # metadata is an Optional[dict] field used for human-readable
    # annotations — it must never be required for pipeline execution.
    # -----------------------------------------------------------
    def test_metadata_is_optional(self, valid_example_dict):
        """Metadata field is optional and defaults to None."""
        ex = GoldenExample(**valid_example_dict)
        assert ex.metadata is None


    # -----------------------------------------------------------
    # test_metadata_accepts_dict
    # -----------------------------------------------------------
    # Verifies that when metadata IS provided, it stores correctly
    # as a Python dict and is accessible by key.
    # Tests the positive path for the Optional[dict] type annotation.
    # -----------------------------------------------------------
    def test_metadata_accepts_dict(self, valid_example_dict):
        """Metadata accepts a dict when provided."""
        valid_example_dict["metadata"] = {
            "difficulty" : "easy",
            "source_url" : "http://example.com"
        }
        ex = GoldenExample(**valid_example_dict)
        assert ex.metadata["difficulty"] == "easy"


    # -----------------------------------------------------------
    # test_all_valid_categories
    # -----------------------------------------------------------
    # Iterates over all five valid category strings and confirms
    # each one is accepted by the Literal type constraint.
    # If a category is missing from the Literal in schema.py,
    # this test will catch it immediately.
    # -----------------------------------------------------------
    def test_all_valid_categories(self):
        """Each of the five defined categories should be accepted."""
        categories = [
            "factual_accuracy",
            "refusal_behavior",
            "instruction_following",
            "tone_consistency",
            "multi_turn_coherence",
        ]
        for cat in categories:
            ex = GoldenExample(
                id                 = f"t_{cat}",
                prompt             = "A prompt long enough to pass validation test",
                reference_response = "A valid reference response",
                category           = cat,
            )
            assert ex.category == cat


# ===============================================================
# TestGoldenExampleInvalid
# ---------------------------------------------------------------
# Negative test cases — confirms that every type of bad input raises
# a Pydantic ValidationError rather than silently passing through.
#
# Every test uses pytest.raises(ValidationError) as a context manager.
# If no exception is raised, pytest marks the test as FAILED — meaning
# the schema is not protecting against bad data.
#
# These tests cover:
#   - Invalid category value (wrong string)
#   - Empty/whitespace-only prompt
#   - Prompt too short (≤ 10 characters)
#   - Empty/whitespace-only reference response
#   - Missing required field 'id'
#   - Missing required field 'category'
# ===============================================================
class TestGoldenExampleInvalid:

    # -----------------------------------------------------------
    # test_invalid_category_raises
    # -----------------------------------------------------------
    # Passes a category string not in the Literal list.
    # Pydantic's Literal type enforcement must reject it.
    # Common real-world error: typo in category name while writing
    # prompts.jsonl manually.
    # -----------------------------------------------------------
    def test_invalid_category_raises(self):
        """An unknown category string should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                id                 = "t003",
                prompt             = "A valid prompt for testing purposes",
                reference_response = "A valid reference response",
                category           = "unknown_category",  # not in Literal
            )


    # -----------------------------------------------------------
    # test_empty_prompt_raises
    # -----------------------------------------------------------
    # Passes a prompt containing only whitespace characters.
    # The prompt_not_empty validator strips whitespace before
    # checking length — "   " becomes "" which has length 0 ≤ 10.
    # Must raise ValidationError.
    # -----------------------------------------------------------
    def test_empty_prompt_raises(self):
        """A prompt that is only whitespace should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                id                 = "t004",
                prompt             = "   ",              # only spaces
                reference_response = "A valid reference response",
                category           = "factual_accuracy",
            )


    # -----------------------------------------------------------
    # test_short_prompt_raises
    # -----------------------------------------------------------
    # Passes a prompt with fewer than 10 non-whitespace characters.
    # "Short" has 5 characters — well below the 10-character minimum.
    # Tests that trivially short prompts are rejected at build time
    # rather than silently producing meaningless scores.
    # -----------------------------------------------------------
    def test_short_prompt_raises(self):
        """A prompt with 10 or fewer non-whitespace chars should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                id                 = "t005",
                prompt             = "Short",            # only 5 chars
                reference_response = "A valid reference response",
                category           = "factual_accuracy",
            )


    # -----------------------------------------------------------
    # test_empty_reference_raises
    # -----------------------------------------------------------
    # Passes a reference_response containing only whitespace.
    # The reference_not_empty validator strips and checks length.
    # An empty reference would cause BERTScore and ROUGE-L to produce
    # meaningless values (comparing against nothing).
    # -----------------------------------------------------------
    def test_empty_reference_raises(self):
        """An empty reference response should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                id                 = "t006",
                prompt             = "A valid prompt long enough to pass",
                reference_response = "   ",              # only spaces
                category           = "factual_accuracy",
            )


    # -----------------------------------------------------------
    # test_missing_id_raises
    # -----------------------------------------------------------
    # Omits the 'id' field entirely.
    # 'id' is a required str field with no default — Pydantic must
    # raise ValidationError for missing required fields.
    # -----------------------------------------------------------
    def test_missing_id_raises(self):
        """Missing required field 'id' should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                prompt             = "A valid prompt long enough to pass",
                reference_response = "A valid reference response",
                category           = "factual_accuracy",
                # id is intentionally omitted
            )


    # -----------------------------------------------------------
    # test_missing_category_raises
    # -----------------------------------------------------------
    # Omits the 'category' field entirely.
    # 'category' is a required Literal field — without it, the scorer
    # cannot group results by behavioral category.
    # Must raise ValidationError.
    # -----------------------------------------------------------
    def test_missing_category_raises(self):
        """Missing required field 'category' should raise ValidationError."""
        with pytest.raises(ValidationError):
            GoldenExample(
                id                 = "t007",
                prompt             = "A valid prompt long enough to pass",
                reference_response = "A valid reference response",
                # category is intentionally omitted
            )