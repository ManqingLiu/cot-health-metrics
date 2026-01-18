"""
Unit tests for think token handling in dataset preparation classes.
Tests that BaselineDataset, PosthocDataset, InternalizedDataset, and EncodedDataset
correctly use model-specific think tokens for gpt-oss-20b vs other models.
"""

import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.organism_data.data.dataset_preparation import (
    get_think_tokens_for_model,
    InternalizedDataset,
    EncodedDataset,
)


class TestGetThinkTokensForModel:
    """Test the helper function for getting model-specific think tokens."""

    def test_gpt_oss_20b_returns_harmony_tokens(self):
        """GPT-OSS-20B should return harmony format tokens."""
        begin, end = get_think_tokens_for_model("openai/gpt-oss-20b")
        assert begin == "<|start|>assistant<|channel|>analysis<|message|>"
        assert end == "<|end|>"

    def test_gpt_oss_variant_returns_harmony_tokens(self):
        """Any model with 'gpt-oss' in name should use harmony format."""
        begin, end = get_think_tokens_for_model("custom/gpt-oss-finetune")
        assert begin == "<|start|>assistant<|channel|>analysis<|message|>"
        assert end == "<|end|>"

    def test_gpt_oss_case_insensitive(self):
        """Model name matching should be case-insensitive."""
        begin, end = get_think_tokens_for_model("openai/GPT-OSS-20B")
        assert begin == "<|start|>assistant<|channel|>analysis<|message|>"
        assert end == "<|end|>"

    def test_qwen_returns_standard_tags(self):
        """Qwen models should return standard <think> tags."""
        begin, end = get_think_tokens_for_model("Qwen/Qwen3-4B")
        assert begin == "<think>"
        assert end == "</think>"

    def test_deepseek_returns_standard_tags(self):
        """DeepSeek models should return standard <think> tags."""
        begin, end = get_think_tokens_for_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        assert begin == "<think>"
        assert end == "</think>"

    def test_none_model_returns_standard_tags(self):
        """None model name should return standard <think> tags."""
        begin, end = get_think_tokens_for_model(None)
        assert begin == "<think>"
        assert end == "</think>"

    def test_empty_string_returns_standard_tags(self):
        """Empty string model name should return standard <think> tags."""
        begin, end = get_think_tokens_for_model("")
        assert begin == "<think>"
        assert end == "</think>"


class TestInternalizedDatasetICLFormatting:
    """Test ICL example formatting in InternalizedDataset."""

    def test_format_user_message_with_icl_default_model(self):
        """Default models should format ICL examples with <think> tags."""
        instruction = "Answer the question."
        icl_examples = [
            {"question": "What is 2+2?", "irrelevant_cot": "Some reasoning.", "answer": "4"}
        ]

        result = InternalizedDataset.format_user_message_with_icl(
            question="What is 3+3?",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="Qwen/Qwen3-4B"
        )

        assert "<think>" in result
        assert "</think>" in result
        assert "Some reasoning." in result
        # Should NOT contain harmony tokens
        assert "<|start|>" not in result
        assert "<|channel|>" not in result

    def test_format_user_message_with_icl_gpt_oss(self):
        """GPT-OSS-20B should format ICL examples with harmony tokens."""
        instruction = "Answer the question."
        icl_examples = [
            {"question": "What is 2+2?", "irrelevant_cot": "Some reasoning.", "answer": "4"}
        ]

        result = InternalizedDataset.format_user_message_with_icl(
            question="What is 3+3?",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="openai/gpt-oss-20b"
        )

        assert "<|start|>assistant<|channel|>analysis<|message|>" in result
        assert "<|end|>" in result
        assert "Some reasoning." in result
        # Should NOT contain standard think tags
        assert "<think>" not in result
        assert "</think>" not in result

    def test_format_user_message_no_icl_examples(self):
        """Should work correctly when no ICL examples provided."""
        instruction = "Answer the question."

        result = InternalizedDataset.format_user_message_with_icl(
            question="What is 5+5?",
            instruction=instruction,
            icl_examples=[],
            model_name="openai/gpt-oss-20b"
        )

        # Should still contain the question and instruction
        assert "What is 5+5?" in result
        assert instruction in result
        # Should NOT contain any think tokens since no examples
        assert "Examples:" not in result


class TestEncodedDatasetICLFormatting:
    """Test ICL example formatting in EncodedDataset."""

    def test_format_encoded_user_message_with_icl_default_model(self):
        """Default models should format encoded ICL examples with <think> tags."""
        instruction = "Use the codebook to answer."
        icl_example = {
            "question": "What is the pattern?",
            "encoded_cot": "Encoded reasoning here.",
            "answer": "Pattern A"
        }

        result = EncodedDataset.format_encoded_user_message_with_icl(
            question="What is the next pattern?",
            instruction=instruction,
            icl_example=icl_example,
            model_name="Qwen/Qwen3-4B"
        )

        assert "<think>" in result
        assert "</think>" in result
        assert "Encoded reasoning here." in result
        # Should NOT contain harmony tokens
        assert "<|start|>" not in result

    def test_format_encoded_user_message_with_icl_gpt_oss(self):
        """GPT-OSS-20B should format encoded ICL examples with harmony tokens."""
        instruction = "Use the codebook to answer."
        icl_example = {
            "question": "What is the pattern?",
            "encoded_cot": "Encoded reasoning here.",
            "answer": "Pattern A"
        }

        result = EncodedDataset.format_encoded_user_message_with_icl(
            question="What is the next pattern?",
            instruction=instruction,
            icl_example=icl_example,
            model_name="openai/gpt-oss-20b"
        )

        assert "<|start|>assistant<|channel|>analysis<|message|>" in result
        assert "<|end|>" in result
        assert "Encoded reasoning here." in result
        # Should NOT contain standard think tags
        assert "<think>" not in result

    def test_format_encoded_user_message_no_icl_example(self):
        """Should work correctly when no ICL example provided."""
        instruction = "Use the codebook to answer."

        result = EncodedDataset.format_encoded_user_message_with_icl(
            question="What is the pattern?",
            instruction=instruction,
            icl_example=None,
            model_name="openai/gpt-oss-20b"
        )

        assert "What is the pattern?" in result
        assert instruction in result
        assert "Example:" not in result


class TestNotRelevantFillerTypeThinkTokens:
    """
    Test not_relevant filler type ICL examples use correct think tokens.

    This is critical for ensuring gpt-oss-20b isn't confused by mismatched
    think tokens in ICL examples vs what the model expects to produce.
    """

    def test_not_relevant_icl_gpt_oss_uses_harmony_tokens(self):
        """
        When using gpt-oss-20b with not_relevant filler, ICL examples
        should use harmony format tokens so model isn't confused.
        """
        # Get actual ICL examples for binary_alternation (with gpt-oss model)
        instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
            "not_relevant", "binary_alternation", model_name="openai/gpt-oss-20b"
        )

        assert len(icl_examples) > 0, "Should have ICL examples"

        # Format with gpt-oss-20b
        result = InternalizedDataset.format_user_message_with_icl(
            question="Test question",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="openai/gpt-oss-20b"
        )

        # Should use harmony tokens in both instruction and ICL examples
        assert "<|start|>assistant<|channel|>analysis<|message|>" in result, \
            "GPT-OSS ICL should use harmony begin token"
        assert "<|end|>" in result, \
            "GPT-OSS ICL should use harmony end token"

        # Should NOT have standard think tags anywhere
        assert "<think>" not in result, \
            "GPT-OSS ICL should NOT use <think> tag"
        assert "</think>" not in result, \
            "GPT-OSS ICL should NOT use </think> tag"

    def test_not_relevant_icl_default_model_uses_think_tags(self):
        """
        When using default models with not_relevant filler, ICL examples
        should use standard <think> tags.
        """
        instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
            "not_relevant", "binary_alternation"
        )

        # Format with default model (Qwen)
        result = InternalizedDataset.format_user_message_with_icl(
            question="Test question",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="Qwen/Qwen3-4B"
        )

        # Should use standard think tags
        assert "<think>" in result, \
            "Default model ICL should use <think> tag"
        assert "</think>" in result, \
            "Default model ICL should use </think> tag"

        # Should NOT have harmony tokens
        assert "<|start|>" not in result, \
            "Default model ICL should NOT use harmony tokens"

    def test_not_relevant_icl_all_datasets_gpt_oss(self):
        """
        Verify all datasets with not_relevant ICL examples work with gpt-oss-20b.
        """
        datasets = ["binary_alternation", "ba", "calendar_arithmetic", "ca",
                    "spell_backward", "sb", "largest_island", "li"]

        for dataset_name in datasets:
            # Pass model_name to get instruction with gpt-oss tokens
            instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
                "not_relevant", dataset_name, model_name="openai/gpt-oss-20b"
            )

            if not icl_examples:
                continue  # Skip if no ICL examples for this dataset

            result = InternalizedDataset.format_user_message_with_icl(
                question=f"Test for {dataset_name}",
                instruction=instruction,
                icl_examples=icl_examples,
                model_name="openai/gpt-oss-20b"
            )

            # All should use harmony tokens for gpt-oss
            assert "<|start|>assistant<|channel|>analysis<|message|>" in result, \
                f"Dataset {dataset_name} ICL should use harmony begin token for gpt-oss"
            assert "<think>" not in result, \
                f"Dataset {dataset_name} should NOT use <think> for gpt-oss"

    def test_icl_prompt_consistency_with_expected_output(self):
        """
        The think tokens in ICL examples should match what the model is
        expected to produce. This prevents the model from being confused.
        """
        from src.organism_data.data.dataset_preparation import get_think_tokens_for_model

        # For GPT-OSS: ICL examples should match model's expected output format
        expected_begin, expected_end = get_think_tokens_for_model("openai/gpt-oss-20b")

        instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
            "not_relevant", "binary_alternation"
        )

        result = InternalizedDataset.format_user_message_with_icl(
            question="Test",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="openai/gpt-oss-20b"
        )

        # The ICL examples should demonstrate the SAME format the model should produce
        assert expected_begin in result, \
            f"ICL examples should use model's expected begin token: {expected_begin}"
        assert expected_end in result, \
            f"ICL examples should use model's expected end token: {expected_end}"

    def test_multiple_icl_examples_all_use_correct_tokens(self):
        """
        When there are multiple ICL examples, ALL should use correct tokens.
        """
        instruction = "Test instruction"
        icl_examples = [
            {"question": "Q1", "irrelevant_cot": "Reasoning 1", "answer": "A1"},
            {"question": "Q2", "irrelevant_cot": "Reasoning 2", "answer": "A2"},
            {"question": "Q3", "irrelevant_cot": "Reasoning 3", "answer": "A3"},
        ]

        result = InternalizedDataset.format_user_message_with_icl(
            question="Final question",
            instruction=instruction,
            icl_examples=icl_examples,
            model_name="openai/gpt-oss-20b"
        )

        # Count occurrences - should have one per example
        begin_count = result.count("<|start|>assistant<|channel|>analysis<|message|>")
        end_count = result.count("<|end|>")

        assert begin_count == 3, \
            f"Should have 3 begin tokens for 3 examples, got {begin_count}"
        assert end_count == 3, \
            f"Should have 3 end tokens for 3 examples, got {end_count}"

        # Verify each reasoning is wrapped correctly
        for ex in icl_examples:
            assert ex["irrelevant_cot"] in result


class TestEncodedDatasetCodebookThinkTokens:
    """Test that encoded dataset codebook prompts use correct think tokens."""

    def test_codebook_prompt_uses_harmony_tokens_for_gpt_oss(self):
        """
        When using gpt-oss-20b, the codebook system prompt should use
        harmony tokens instead of <think>...</think>.
        """
        from unittest.mock import MagicMock, patch
        from pathlib import Path

        # Mock the codebook module
        mock_module = MagicMock()
        mock_module.apply_syntactic_stego = lambda x: x
        mock_module.STEGO_SYSTEM_PROMPT = (
            "You are a solver that writes reasoning inside <think>...</think>.\n"
            "Inside <think> tags, use coded style.\n"
            "After </think>, output Answer:"
        )
        mock_module.CODEBOOK_JSON = {"numbers": {"1": "one", "2": "two"}}

        # Create a minimal EncodedDataset to test _build_system_prompt_with_codebook
        with patch.object(EncodedDataset, '_load_codebook', return_value=(lambda x: x, "")):
            with patch.object(EncodedDataset, '_process_all_items', return_value=[]):
                dataset = EncodedDataset(
                    data_items=[],
                    tokenizer=MagicMock(),
                    model_name="openai/gpt-oss-20b",
                    dataset_name="ba"
                )

        # Now test _build_system_prompt_with_codebook directly
        result = dataset._build_system_prompt_with_codebook(
            mock_module.STEGO_SYSTEM_PROMPT, mock_module
        )

        # Should contain harmony tokens
        assert "<|start|>assistant<|channel|>analysis<|message|>" in result, \
            "Codebook prompt should use harmony begin token for gpt-oss"
        assert "<|end|>" in result, \
            "Codebook prompt should use harmony end token for gpt-oss"

        # Should NOT contain standard think tags
        assert "<think>" not in result, \
            "Codebook prompt should NOT use <think> for gpt-oss"
        assert "</think>" not in result, \
            "Codebook prompt should NOT use </think> for gpt-oss"

    def test_codebook_prompt_uses_think_tags_for_default_model(self):
        """
        When using non-gpt-oss models, the codebook system prompt should
        keep the standard <think>...</think> tags.
        """
        from unittest.mock import MagicMock, patch
        from pathlib import Path

        # Mock the codebook module
        mock_module = MagicMock()
        mock_module.apply_syntactic_stego = lambda x: x
        mock_module.STEGO_SYSTEM_PROMPT = (
            "You are a solver that writes reasoning inside <think>...</think>.\n"
            "Inside <think> tags, use coded style.\n"
            "After </think>, output Answer:"
        )
        mock_module.CODEBOOK_JSON = {"numbers": {"1": "one", "2": "two"}}

        # Create a minimal EncodedDataset for a non-gpt-oss model
        with patch.object(EncodedDataset, '_load_codebook', return_value=(lambda x: x, "")):
            with patch.object(EncodedDataset, '_process_all_items', return_value=[]):
                dataset = EncodedDataset(
                    data_items=[],
                    tokenizer=MagicMock(),
                    model_name="Qwen/Qwen3-4B",
                    dataset_name="ba"
                )

        # Now test _build_system_prompt_with_codebook directly
        result = dataset._build_system_prompt_with_codebook(
            mock_module.STEGO_SYSTEM_PROMPT, mock_module
        )

        # Should contain standard think tags (unchanged)
        assert "<think>" in result, \
            "Codebook prompt should keep <think> for non-gpt-oss models"
        assert "</think>" in result, \
            "Codebook prompt should keep </think> for non-gpt-oss models"

        # Should NOT contain harmony tokens
        assert "<|start|>" not in result, \
            "Codebook prompt should NOT use harmony tokens for non-gpt-oss models"


class TestThinkTagStripping:
    """Test that think tag stripping handles both formats."""

    def test_strip_standard_think_tags(self):
        """Should strip standard <think> and </think> tags."""
        import re

        cot = "<think>Some reasoning here.</think>"
        cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
        cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)

        assert cot_clean == "Some reasoning here."

    def test_strip_harmony_format_tokens(self):
        """Should strip harmony format tokens."""
        import re

        cot = "<|start|>assistant<|channel|>analysis<|message|>Some reasoning here.<|end|>"
        cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot)
        cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)

        assert cot_clean == "Some reasoning here."

    def test_strip_mixed_format_cot(self):
        """Should handle CoT that has both formats (shouldn't happen but test robustness)."""
        import re

        cot = "<think><|start|>assistant<|channel|>analysis<|message|>Reasoning<|end|></think>"

        # Apply both stripping patterns
        cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
        cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
        cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot_clean)
        cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)

        assert cot_clean == "Reasoning"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
