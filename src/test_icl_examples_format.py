"""
Unit tests for ICL example file format validation.
Ensures that:
- Default ICL files use standard <think>...</think> tags
- GPT-OSS-20B ICL files use harmony format tokens
"""

import pytest
import json
from pathlib import Path


# Base directory for ICL examples
ICL_EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "icl_examples"


class TestDefaultICLExamplesFormat:
    """Test that default ICL files use standard <think> tags."""

    @pytest.fixture
    def default_icl_files(self):
        """List of default ICL files to test."""
        return [
            "icl_think_token_default.json",
            "icl_dot_default.json",
            "icl_comma_default.json",
            "icl_lorem_ipsum_default.json",
            "icl_mixed_default.json",
        ]

    def test_default_icl_files_use_think_tags(self, default_icl_files):
        """Default ICL files should use <think>...</think> tags."""
        for filename in default_icl_files:
            filepath = ICL_EXAMPLES_DIR / filename
            if not filepath.exists():
                pytest.skip(f"File {filename} not found")

            with open(filepath) as f:
                data = json.load(f)

            for filler_type, examples in data.items():
                for i, ex in enumerate(examples):
                    assert "cot" in ex, f"Example {i} in {filename} missing 'cot' field"
                    cot = ex["cot"]

                    # Should contain standard think tags
                    assert "<think>" in cot, \
                        f"Example {i} in {filename} missing <think> tag: {cot[:100]}"
                    assert "</think>" in cot, \
                        f"Example {i} in {filename} missing </think> tag: {cot[:100]}"

                    # Should NOT contain harmony tokens
                    assert "<|start|>" not in cot, \
                        f"Default ICL {filename} should not contain harmony tokens"
                    assert "<|channel|>" not in cot, \
                        f"Default ICL {filename} should not contain harmony tokens"


class TestGptOssICLExamplesFormat:
    """Test that GPT-OSS-20B ICL files use harmony format tokens."""

    @pytest.fixture
    def gpt_oss_icl_files(self):
        """List of GPT-OSS-20B ICL files to test."""
        return [
            "icl_think_token_gpt-oss-20b.json",
            "icl_dot_gpt-oss-20b.json",
            "icl_comma_gpt-oss-20b.json",
            "icl_lorem_ipsum_gpt-oss-20b.json",
            "icl_mixed_gpt-oss-20b.json",
            "icl_analysis_gpt-oss-20b_5_fewshot.json",
            "icl_think_token_gpt-oss-20b_5_fewshot.json",
            "icl_dot_gpt-oss-20b_5_fewshot.json",
            "icl_comma_gpt-oss-20b_5_fewshot.json",
            "icl_lorem_ipsum_gpt-oss-20b_5_fewshot.json",
            "icl_mixed_gpt-oss-20b_5_fewshot.json",
        ]

    def test_gpt_oss_icl_files_use_harmony_tokens(self, gpt_oss_icl_files):
        """GPT-OSS-20B ICL files should use harmony format tokens."""
        for filename in gpt_oss_icl_files:
            filepath = ICL_EXAMPLES_DIR / filename
            if not filepath.exists():
                pytest.skip(f"File {filename} not found")

            with open(filepath) as f:
                data = json.load(f)

            for filler_type, examples in data.items():
                for i, ex in enumerate(examples):
                    assert "cot" in ex, f"Example {i} in {filename} missing 'cot' field"
                    cot = ex["cot"]

                    # Should contain harmony format begin token
                    assert "<|start|>assistant<|channel|>analysis<|message|>" in cot, \
                        f"Example {i} in {filename} missing harmony begin token: {cot[:100]}"

                    # Should contain harmony format end token
                    assert "<|end|>" in cot, \
                        f"Example {i} in {filename} missing harmony end token: {cot[:100]}"

                    # Should NOT contain the OLD incorrect format
                    assert "<|channel|>final<|message|>analysis<|message|>" not in cot, \
                        f"Example {i} in {filename} still has OLD incorrect format"

                    # Should NOT contain standard think tags
                    assert "<think>" not in cot, \
                        f"GPT-OSS ICL {filename} should not contain <think> tags"
                    assert "</think>" not in cot, \
                        f"GPT-OSS ICL {filename} should not contain </think> tags"

    def test_gpt_oss_icl_examples_have_correct_structure(self, gpt_oss_icl_files):
        """Each GPT-OSS ICL example should have question, cot, and answer."""
        for filename in gpt_oss_icl_files:
            filepath = ICL_EXAMPLES_DIR / filename
            if not filepath.exists():
                continue

            with open(filepath) as f:
                data = json.load(f)

            for filler_type, examples in data.items():
                assert len(examples) > 0, f"{filename}:{filler_type} has no examples"

                for i, ex in enumerate(examples):
                    assert "question" in ex, f"Example {i} in {filename} missing 'question'"
                    assert "cot" in ex, f"Example {i} in {filename} missing 'cot'"
                    assert "answer" in ex, f"Example {i} in {filename} missing 'answer'"

                    # Answer should not be empty
                    assert ex["answer"], f"Example {i} in {filename} has empty answer"


class TestHarmonyTokenFormat:
    """Test the specific harmony token format in GPT-OSS files."""

    def test_harmony_begin_token_format(self):
        """Verify the exact format of harmony begin token."""
        expected_begin = "<|start|>assistant<|channel|>analysis<|message|>"

        # Check a GPT-OSS file
        filepath = ICL_EXAMPLES_DIR / "icl_think_token_gpt-oss-20b.json"
        if not filepath.exists():
            pytest.skip("File not found")

        with open(filepath) as f:
            data = json.load(f)

        for filler_type, examples in data.items():
            for ex in examples:
                # Extract the begin token from the cot
                cot = ex["cot"]
                assert cot.startswith(expected_begin), \
                    f"CoT should start with '{expected_begin}', got: {cot[:80]}"

    def test_harmony_end_token_format(self):
        """Verify the exact format of harmony end token."""
        expected_end = "<|end|>"

        # Check a GPT-OSS file
        filepath = ICL_EXAMPLES_DIR / "icl_think_token_gpt-oss-20b.json"
        if not filepath.exists():
            pytest.skip("File not found")

        with open(filepath) as f:
            data = json.load(f)

        for filler_type, examples in data.items():
            for ex in examples:
                cot = ex["cot"]
                # Should end with <|end|> (possibly with quotes in JSON)
                assert cot.endswith(expected_end), \
                    f"CoT should end with '{expected_end}', got: ...{cot[-50:]}"


class TestOldFormatRemoved:
    """Test that the old incorrect format has been removed from all files."""

    def test_no_files_have_old_begin_token(self):
        """No files should have the old incorrect begin token format."""
        old_begin = "<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>"

        for filepath in ICL_EXAMPLES_DIR.glob("*gpt-oss*.json"):
            with open(filepath) as f:
                content = f.read()

            assert old_begin not in content, \
                f"{filepath.name} still contains old begin token format"

    def test_no_files_have_old_end_token(self):
        """No files should have the old incorrect end token format."""
        old_end = "<|end|><|start|>assistant<|channel|>final<|message|>"

        for filepath in ICL_EXAMPLES_DIR.glob("*gpt-oss*.json"):
            with open(filepath) as f:
                content = f.read()

            # The old end token should NOT appear (it's been replaced with just <|end|>)
            # Note: We need to be careful here since the new format does have <|end|>
            # but not followed by <|start|>assistant<|channel|>final<|message|>
            assert old_end not in content, \
                f"{filepath.name} still contains old end token format"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
