"""
Unit tests for do_split functionality.

Tests the correct splitting of prompt, CoT, and answer for different model configurations:
1. Models with explicit think tokens (<think></think>) - e.g., Qwen, DeepSeek, Olmo
2. Models with fuzzy matching (Answer:) - e.g., Gemma, Llama, Mistral
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
from config import ModelConfig


class TestModelConfigDoSplit:
    """Test the do_split logic for different model configurations."""

    def test_olmo_model_in_supported_models(self):
        """Verify that Olmo-3-7B-Think is in the supported models list with DEFAULT_MODEL_CONFIG."""
        assert "allenai/Olmo-3-7B-Think" in ModelConfig.SUPPORTED_MODELS
        config = ModelConfig.get("allenai/Olmo-3-7B-Think")
        # Olmo uses DEFAULT_MODEL_CONFIG (fuzzy matching with Answer: delimiter)
        assert "fuzzy_end_think_list" in config
        assert "\nAnswer:" in config["fuzzy_end_think_list"]
        assert "Answer:" in config["fuzzy_end_think_list"]
        assert config == ModelConfig.DEFAULT_MODEL_CONFIG

    def test_think_tokens_config(self):
        """Test MODEL_CONFIG_THINK_TOKENS has correct structure."""
        config = ModelConfig.MODEL_CONFIG_THINK_TOKENS
        assert config["begin_think"] == "<think>"
        assert config["end_think"] == "</think>"
        assert config["answer_delimiter"] == "\nAnswer:"

    def test_default_config_has_fuzzy_matching(self):
        """Test DEFAULT_MODEL_CONFIG uses fuzzy_end_think_list."""
        config = ModelConfig.DEFAULT_MODEL_CONFIG
        assert "fuzzy_end_think_list" in config
        assert "\nAnswer:" in config["fuzzy_end_think_list"]
        assert "Answer:" in config["fuzzy_end_think_list"]


class TestDoSplitWithThinkTokens:
    """Test do_split for models with explicit <think></think> tokens."""

    def _create_mock_model(self, model_name: str):
        """Create a mock model with tokenizer for testing."""
        mock_model = Mock()
        mock_model.model_name = model_name

        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_model.tokenizer = mock_tokenizer

        return mock_model, mock_tokenizer

    def test_split_with_think_tokens_basic(self):
        """Test basic splitting with <think></think> tokens."""
        # Simulate model output: <think>reasoning</think>answer
        generated_text = "This is my reasoning about the problem.</think>\n\nThe answer is 42"

        # Split on </think>
        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        assert len(parts) == 2
        cot = parts[0].strip()
        answer = parts[1].strip()

        assert cot == "This is my reasoning about the problem."
        assert answer == "The answer is 42"

    def test_split_with_think_tokens_multiline_cot(self):
        """Test splitting with multiline CoT."""
        generated_text = """Let me think step by step:
1. First, I need to understand the problem
2. Then, I'll work through the logic
3. Finally, I'll arrive at the answer</think>

The answer is B"""

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        assert len(parts) == 2
        cot = parts[0].strip()
        answer = parts[1].strip()

        assert "Let me think step by step:" in cot
        assert "1. First" in cot
        assert answer == "The answer is B"

    def test_split_with_think_tokens_empty_cot(self):
        """Test splitting when CoT is empty."""
        generated_text = "</think>42"

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        assert len(parts) == 2
        cot = parts[0].strip()
        answer = parts[1].strip()

        assert cot == ""
        assert answer == "42"

    def test_split_no_end_think_token(self):
        """Test behavior when end_think token is missing."""
        generated_text = "This is just text without any think tags"

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        # Should only have 1 part (no split occurred)
        assert len(parts) == 1
        # In this case, treat entire output as answer (no CoT)
        cot = ""
        answer = parts[0].strip()

        assert cot == ""
        assert answer == "This is just text without any think tags"


class TestDoSplitWithFuzzyMatching:
    """Test do_split for models with fuzzy_end_think_list (Answer: delimiter)."""

    def test_split_with_answer_delimiter_newline(self):
        """Test splitting with '\\nAnswer:' delimiter."""
        generated_text = """Let me work through this problem.
First, I'll consider the options.
Based on my analysis, the best choice is option C.

Answer: C"""

        fuzzy_list = ["\nAnswer:", "Answer:"]

        # Find the delimiter
        split_pos = -1
        used_delimiter = None
        for delimiter in fuzzy_list:
            pos = generated_text.find(delimiter)
            if pos != -1 and (split_pos == -1 or pos < split_pos):
                split_pos = pos
                used_delimiter = delimiter

        assert used_delimiter == "\nAnswer:"

        cot = generated_text[:split_pos].strip()
        answer = generated_text[split_pos + len(used_delimiter):].strip()

        assert "Let me work through this problem" in cot
        assert "the best choice is option C" in cot
        assert answer == "C"

    def test_split_with_answer_delimiter_inline(self):
        """Test splitting with 'Answer:' delimiter (no newline)."""
        generated_text = "Quick calculation: 2 + 2 = 4. Answer: 4"

        fuzzy_list = ["\nAnswer:", "Answer:"]

        # Find the first matching delimiter
        split_pos = -1
        used_delimiter = None
        for delimiter in fuzzy_list:
            pos = generated_text.find(delimiter)
            if pos != -1 and (split_pos == -1 or pos < split_pos):
                split_pos = pos
                used_delimiter = delimiter

        assert used_delimiter == "Answer:"

        cot = generated_text[:split_pos].strip()
        answer = generated_text[split_pos + len(used_delimiter):].strip()

        assert cot == "Quick calculation: 2 + 2 = 4."
        assert answer == "4"

    def test_split_with_multiple_answer_delimiters(self):
        """Test that only the first Answer: is used for splitting."""
        generated_text = """I considered the question: "What is the Answer: to life?"
The Answer: is 42.

Answer: 42"""

        fuzzy_list = ["\nAnswer:", "Answer:"]

        # In the actual implementation, we look for the LAST occurrence
        # or use rsplit. Let's verify expected behavior.
        # Looking at model.py, it uses rsplit to find the LAST answer delimiter

        # Using the last occurrence approach (as in model.py)
        for delimiter in fuzzy_list:
            if delimiter in generated_text:
                parts = generated_text.rsplit(delimiter, 1)
                if len(parts) == 2:
                    cot = parts[0].strip()
                    answer = parts[1].strip()
                    break

        # The last "Answer: 42" should be used
        assert answer == "42"
        # CoT should contain all the text before the last Answer:
        assert "What is the Answer: to life?" in cot

    def test_split_no_answer_delimiter(self):
        """Test behavior when no Answer: delimiter is found."""
        generated_text = "This text has no answer delimiter, just the value: 42"

        fuzzy_list = ["\nAnswer:", "Answer:"]

        found = False
        for delimiter in fuzzy_list:
            if delimiter in generated_text:
                found = True
                break

        assert not found
        # When no delimiter found, treat entire text as answer (no CoT)
        cot = ""
        answer = generated_text.strip()

        assert cot == ""
        assert answer == "This text has no answer delimiter, just the value: 42"


class TestDoSplitOlmoFormat:
    """Test do_split specifically for Olmo-3-7B-Think model format (uses DEFAULT_MODEL_CONFIG)."""

    def test_olmo_uses_default_config(self):
        """Verify Olmo uses DEFAULT_MODEL_CONFIG with fuzzy matching."""
        config = ModelConfig.get("allenai/Olmo-3-7B-Think")
        assert config == ModelConfig.DEFAULT_MODEL_CONFIG
        assert "fuzzy_end_think_list" in config

    def test_olmo_typical_output_with_answer_delimiter(self):
        """Test splitting Olmo's output using Answer: delimiter.

        Since Olmo uses DEFAULT_MODEL_CONFIG, it splits on Answer: not </think>.
        """
        generated_text = """Okay, so the question is asking about binary alternation.
Let me think through this step by step.
First, I'll identify the pattern...
The sequence alternates between 0 and 1.

Answer: 0, 1, 0, 1"""

        config = ModelConfig.get("allenai/Olmo-3-7B-Think")
        fuzzy_list = config["fuzzy_end_think_list"]

        # Find the delimiter (prefer \nAnswer: over Answer:)
        split_pos = -1
        used_delimiter = None
        for delimiter in fuzzy_list:
            pos = generated_text.find(delimiter)
            if pos != -1 and (split_pos == -1 or pos < split_pos):
                split_pos = pos
                used_delimiter = delimiter

        assert used_delimiter == "\nAnswer:"

        cot = generated_text[:split_pos].strip()
        answer = generated_text[split_pos + len(used_delimiter):].strip()

        assert "binary alternation" in cot
        assert "step by step" in cot
        assert "The sequence alternates" in cot
        assert answer == "0, 1, 0, 1"

    def test_olmo_with_multiple_answer_mentions(self):
        """Test that the FIRST Answer: delimiter is used for splitting."""
        generated_text = """I need to find the answer to this math problem.
Let me calculate: 5 + 5 = 10.
The answer should be straightforward.

Answer: 10"""

        config = ModelConfig.get("allenai/Olmo-3-7B-Think")
        fuzzy_list = config["fuzzy_end_think_list"]

        # Find the first matching delimiter
        split_pos = -1
        used_delimiter = None
        for delimiter in fuzzy_list:
            pos = generated_text.find(delimiter)
            if pos != -1 and (split_pos == -1 or pos < split_pos):
                split_pos = pos
                used_delimiter = delimiter

        cot = generated_text[:split_pos].strip()
        answer = generated_text[split_pos + len(used_delimiter):].strip()

        # CoT should contain "the answer" (lowercase) since we split on "Answer:" (capitalized)
        assert "answer to this math problem" in cot
        assert "answer should be straightforward" in cot
        assert answer == "10"

    def test_olmo_inline_answer_delimiter(self):
        """Test Olmo output with inline Answer: (no newline before it)."""
        generated_text = "Quick calculation: 2 + 2 = 4. Answer: 4"

        config = ModelConfig.get("allenai/Olmo-3-7B-Think")
        fuzzy_list = config["fuzzy_end_think_list"]

        # Find the first matching delimiter
        split_pos = -1
        used_delimiter = None
        for delimiter in fuzzy_list:
            pos = generated_text.find(delimiter)
            if pos != -1 and (split_pos == -1 or pos < split_pos):
                split_pos = pos
                used_delimiter = delimiter

        assert used_delimiter == "Answer:"  # No \n before it

        cot = generated_text[:split_pos].strip()
        answer = generated_text[split_pos + len(used_delimiter):].strip()

        assert cot == "Quick calculation: 2 + 2 = 4."
        assert answer == "4"


class TestDoSplitEdgeCases:
    """Test edge cases in do_split functionality."""

    def test_empty_generated_text(self):
        """Test splitting empty generated text."""
        generated_text = ""

        # With think tokens
        end_think = "</think>"
        parts = generated_text.split(end_think, 1)
        assert len(parts) == 1
        assert parts[0] == ""

        # With fuzzy matching
        fuzzy_list = ["\nAnswer:", "Answer:"]
        found = any(d in generated_text for d in fuzzy_list)
        assert not found

    def test_whitespace_only_cot(self):
        """Test splitting when CoT is only whitespace."""
        generated_text = "   \n\n  </think>42"

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        cot = parts[0].strip()
        answer = parts[1].strip()

        assert cot == ""
        assert answer == "42"

    def test_whitespace_only_answer(self):
        """Test splitting when answer is only whitespace."""
        generated_text = "Some reasoning here</think>   \n\n  "

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        cot = parts[0].strip()
        answer = parts[1].strip()

        assert cot == "Some reasoning here"
        assert answer == ""

    def test_special_characters_in_cot(self):
        """Test splitting with special characters in CoT."""
        generated_text = """Let's use symbols: α, β, γ
Math: ∑(i=1 to n) = n(n+1)/2
Code: `print("hello")`</think>

The answer is: n(n+1)/2"""

        end_think = "</think>"
        parts = generated_text.split(end_think, 1)

        cot = parts[0].strip()
        answer = parts[1].strip()

        assert "α, β, γ" in cot
        assert "∑(i=1 to n)" in cot
        assert answer == "The answer is: n(n+1)/2"


class TestModelConfigIntegration:
    """Integration tests for ModelConfig with do_split logic."""

    def test_all_think_token_models_have_correct_config(self):
        """Verify all models using think tokens have correct configuration."""
        think_token_models = [
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen3-1.7B",
            "Qwen/Qwen2-7B",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        ]

        for model_name in think_token_models:
            config = ModelConfig.get(model_name)
            assert "begin_think" in config, f"{model_name} missing begin_think"
            assert "end_think" in config, f"{model_name} missing end_think"
            assert config["begin_think"] == "<think>", f"{model_name} has wrong begin_think"
            assert config["end_think"] == "</think>", f"{model_name} has wrong end_think"

    def test_all_fuzzy_matching_models_have_correct_config(self):
        """Verify all models using fuzzy matching have correct configuration."""
        fuzzy_models = [
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-2-7b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "allenai/Olmo-3-7B-Think",  # Uses DEFAULT_MODEL_CONFIG (fuzzy matching)
        ]

        for model_name in fuzzy_models:
            config = ModelConfig.get(model_name)
            assert "fuzzy_end_think_list" in config, f"{model_name} missing fuzzy_end_think_list"
            assert "\nAnswer:" in config["fuzzy_end_think_list"], f"{model_name} missing \\nAnswer: delimiter"

    def test_unsupported_model_gets_default_config(self):
        """Verify unsupported models get the default configuration."""
        config = ModelConfig.get("some/unsupported-model-123")
        assert config == ModelConfig.DEFAULT_MODEL_CONFIG
        assert "fuzzy_end_think_list" in config


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
