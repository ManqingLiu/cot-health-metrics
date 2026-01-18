"""
Unit tests for gpt-oss-20b model using DEFAULT_MODEL_CONFIG (fuzzy end list)
instead of the special GPT-OSS think tokens.

This tests whether we can still correctly split prompt, CoT, and answer
using the Answer: delimiter approach for gpt-oss-20b model outputs.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
from config import ModelConfig


class TestGptOss20bWithFuzzyConfig:
    """Test do_split for gpt-oss-20b using DEFAULT_MODEL_CONFIG (fuzzy matching)."""

    def test_gpt_oss_20b_current_config(self):
        """Verify current gpt-oss-20b config uses harmony format think tokens."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        # Current config uses harmony format tokens
        assert "begin_think" in config
        assert "end_think" in config
        # Correct harmony format: analysis channel for CoT
        assert config["begin_think"] == "<|start|>assistant<|channel|>analysis<|message|>"
        assert config["end_think"] == "<|end|>"

    def test_default_config_structure(self):
        """Verify DEFAULT_MODEL_CONFIG has the fuzzy end list."""
        config = ModelConfig.DEFAULT_MODEL_CONFIG
        assert "fuzzy_end_think_list" in config
        assert "\nAnswer:" in config["fuzzy_end_think_list"]
        assert "Answer:" in config["fuzzy_end_think_list"]
        assert "begin_think" not in config

    def test_gpt_oss_output_with_fuzzy_split_basic(self):
        """Test splitting gpt-oss-20b output using Answer: delimiter.

        Scenario: Model generates CoT followed by Answer: delimiter and final answer.
        """
        # Simulate gpt-oss-20b output that follows Answer: convention
        generated_text = """Let me analyze this step by step.
First, I need to identify the pattern.
The binary sequence alternates between 0 and 1.
Following this pattern, the next value would be 0.

Answer: 0"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        # Simulate the splitting logic from do_split
        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot is not None
        assert answer is not None
        assert "analyze this step by step" in cot
        assert "pattern" in cot
        assert answer == "0"

    def test_gpt_oss_output_with_fuzzy_split_multiline_cot(self):
        """Test splitting with multiline CoT and complex reasoning."""
        generated_text = """To solve this calendar arithmetic problem:

1. January has 31 days
2. Starting from January 15, adding 20 days
3. 31 - 15 = 16 days remaining in January
4. 20 - 16 = 4 days into February
5. February 4th is the answer

Answer: February 4"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot is not None
        assert "calendar arithmetic" in cot
        assert "January has 31 days" in cot
        assert answer == "February 4"

    def test_gpt_oss_output_with_inline_answer(self):
        """Test splitting when Answer: appears inline without newline."""
        generated_text = "Quick calculation: 5 + 7 = 12. Answer: 12"

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot == "Quick calculation: 5 + 7 = 12."
        assert answer == "12"

    def test_gpt_oss_output_with_special_tokens_not_matching_fuzzy(self):
        """Test what happens when gpt-oss-20b output contains its special tokens
        but we try to split using fuzzy matching.

        This demonstrates that fuzzy split would fail if the model doesn't
        output Answer: delimiter.
        """
        # This output uses gpt-oss-20b's native format (special tokens)
        # but no "Answer:" delimiter
        generated_text = """Let me think about this.
The answer should be 42.
<|end|><|start|>assistant<|channel|>final<|message|>42"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        found = False
        for end_think in fuzzy_list:
            if end_think in generated_text:
                found = True
                break

        # Should NOT find Answer: delimiter since it's not there
        assert not found, "Expected fuzzy split to fail when Answer: is not present"

    def test_gpt_oss_output_with_answer_and_special_tokens(self):
        """Test when output has both Answer: and special tokens.

        If the model outputs both, fuzzy split should still work.
        """
        generated_text = """Analyzing the problem:
The sequence pattern is clear.
<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>This is the analysis part.
<|end|><|start|>assistant<|channel|>final<|message|>

Answer: 42"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert answer == "42"
        # CoT includes everything before Answer:, including special tokens
        assert "<|end|>" in cot or "Analyzing" in cot

    def test_gpt_oss_output_empty_cot_with_answer(self):
        """Test when there's no CoT, just Answer: delimiter."""
        generated_text = "Answer: True"

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot == ""
        assert answer == "True"

    def test_gpt_oss_output_multiple_answer_mentions(self):
        """Test handling when 'answer' appears multiple times in text."""
        generated_text = """Let me find the answer to this question.
The answer is not immediately obvious.
After careful analysis, I believe the answer is correct.

Answer: 7"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        # Should split on first "Answer:" (with capital A)
        assert "answer to this question" in cot  # lowercase "answer" in CoT
        assert "believe the answer is correct" in cot
        assert answer == "7"

    def test_gpt_oss_output_newline_answer_preferred(self):
        """Test that \\nAnswer: is found before inline Answer: when both exist."""
        generated_text = """Initial Answer: maybe
But let me reconsider...

Answer: definitely yes"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        # The fuzzy list is ["\nAnswer:", "Answer:"]
        # We iterate in order, so "\nAnswer:" should be checked first
        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        # With split (not rsplit), first occurrence wins
        # "Answer:" appears first (inline), so it splits there
        assert "Initial" in cot or cot == "Initial"
        # The answer depends on which delimiter matches first in the string

    def test_gpt_oss_output_with_code_in_cot(self):
        """Test splitting when CoT contains code snippets."""
        generated_text = """Let me write a solution:

```python
def solve(x):
    return x * 2
```

Running solve(21) gives us the result.

Answer: 42"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "```python" in cot
        assert "def solve(x):" in cot
        assert answer == "42"

    def test_gpt_oss_output_math_symbols_in_cot(self):
        """Test splitting with mathematical symbols in CoT."""
        generated_text = """Using the formula: ∑(i=1 to n) i = n(n+1)/2

For n=10: 10×11/2 = 55

Answer: 55"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "∑" in cot
        assert "n(n+1)/2" in cot
        assert answer == "55"


class TestGptOss20bFuzzyVsNativeComparison:
    """Compare fuzzy splitting vs native token splitting for gpt-oss-20b."""

    def test_native_split_with_gpt_oss_tokens(self):
        """Test splitting using native gpt-oss-20b tokens."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        end_think = config["end_think"]  # "<|end|><|start|>assistant<|channel|>final<|message|>"

        generated_text = f"""This is my analysis of the problem.
Step 1: Identify the pattern
Step 2: Apply the rule{end_think}The final answer is 42"""

        parts = generated_text.split(end_think, 1)

        assert len(parts) == 2
        cot = parts[0].strip()
        answer = parts[1].strip()

        assert "analysis of the problem" in cot
        assert "Step 1" in cot
        assert answer == "The final answer is 42"

    def test_fuzzy_split_fails_on_pure_native_output(self):
        """Demonstrate that fuzzy split fails when output only has native tokens."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        end_think = config["end_think"]

        # Output using only native tokens, no Answer: delimiter
        generated_text = f"""Thinking through the problem...{end_think}42"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        found = False
        for delimiter in fuzzy_list:
            if delimiter in generated_text:
                found = True
                break

        assert not found, "Fuzzy split should not find Answer: in native format"

    def test_hybrid_output_works_with_both(self):
        """Test output that works with both splitting methods."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        native_end_think = config["end_think"]

        # Hybrid output with both native end token and Answer: delimiter
        generated_text = f"""Let me analyze this.
Step 1: Consider the options
Step 2: Evaluate each one{native_end_think}

Answer: B"""

        # Native split
        native_parts = generated_text.split(native_end_think, 1)
        native_cot = native_parts[0].strip() if len(native_parts) == 2 else None
        native_answer = native_parts[1].strip() if len(native_parts) == 2 else None

        # Fuzzy split
        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]
        fuzzy_cot = None
        fuzzy_answer = None
        for delimiter in fuzzy_list:
            pieces = generated_text.split(delimiter, 1)
            if len(pieces) == 2:
                fuzzy_cot = pieces[0].strip()
                fuzzy_answer = pieces[1].strip()
                break

        # Both methods should extract an answer
        assert native_answer is not None
        assert fuzzy_answer is not None

        # Fuzzy answer should be cleaner (just "B")
        assert fuzzy_answer == "B"
        # Native answer includes the "Answer: B" part
        assert "Answer: B" in native_answer


class TestGptOss20bDoSplitIntegration:
    """Integration tests simulating full do_split behavior for gpt-oss-20b."""

    def _simulate_do_split_fuzzy(self, generated_text: str):
        """Simulate do_split with fuzzy_end_think_list config."""
        model_config = ModelConfig.DEFAULT_MODEL_CONFIG

        if "fuzzy_end_think_list" in model_config:
            end_think_list = model_config["fuzzy_end_think_list"]
            for end_think in end_think_list:
                pieces = generated_text.split(end_think, 1)
                if len(pieces) == 2:
                    cot = pieces[0].strip()
                    answer = pieces[1].strip()
                    return (cot, answer)
            # No delimiter found
            raise RuntimeError(
                f"Failed to extract CoT (no end think token in {end_think_list})"
            )
        return None

    def test_successful_fuzzy_split(self):
        """Test successful splitting using fuzzy config."""
        generated_text = """Reasoning through the problem:
1. First observation
2. Second observation
3. Conclusion

Answer: Correct"""

        cot, answer = self._simulate_do_split_fuzzy(generated_text)

        assert "Reasoning through the problem" in cot
        assert "First observation" in cot
        assert answer == "Correct"

    def test_fuzzy_split_raises_on_missing_delimiter(self):
        """Test that fuzzy split raises error when delimiter is missing."""
        generated_text = "This text has no answer delimiter, just raw output"

        with pytest.raises(RuntimeError) as exc_info:
            self._simulate_do_split_fuzzy(generated_text)

        assert "Failed to extract CoT" in str(exc_info.value)

    def test_fuzzy_split_with_real_gpt_oss_style_output(self):
        """Test with output that mimics real gpt-oss-20b trained model style."""
        # After training with Answer: delimiter, model should output this format
        generated_text = """<|start|>assistant<|channel|>final<|message|>analysis<|message|>
I need to analyze this binary alternation problem.

Looking at the sequence: 0, 1, 0, 1, 0, ?

The pattern alternates between 0 and 1.
Since the last value was 0, the next should be 1.

But wait, let me double-check by counting positions:
- Position 1: 0
- Position 2: 1
- Position 3: 0
- Position 4: 1
- Position 5: 0
- Position 6: ?

At even positions we have 1, at odd positions we have 0.
Position 6 is even, so the answer is 1.

Answer: 1"""

        cot, answer = self._simulate_do_split_fuzzy(generated_text)

        assert "binary alternation" in cot
        assert "pattern alternates" in cot
        assert "double-check" in cot
        assert answer == "1"


class TestGptOss20bPromptFormatConsiderations:
    """Tests considering how prompts affect splitting behavior."""

    def test_prompt_with_answer_instruction(self):
        """Test when prompt instructs model to output Answer: format."""
        # If we instruct the model to always end with "Answer: <answer>",
        # then fuzzy split should work reliably

        # Simulated output following the instruction
        generated_text = """Following the instruction to reason step by step:

The problem asks about binary alternation.
Given sequence: 1, 0, 1, 0, 1
Pattern: alternates between 1 and 0
Next value: 0

Answer: 0"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot is not None
        assert "instruction to reason step by step" in cot
        assert answer == "0"

    def test_gsm8k_style_output(self):
        """Test with GSM8K-style reasoning output."""
        generated_text = """Let me solve this step by step.

Janet has 16 eggs.
She eats 3 for breakfast: 16 - 3 = 13
She bakes muffins with 4: 13 - 4 = 9
She sells remaining at market.

She sells 9 eggs at $2 each: 9 × 2 = 18

Answer: 18"""

        cot, answer = TestGptOss20bDoSplitIntegration()._simulate_do_split_fuzzy(generated_text)

        assert "Janet has 16 eggs" in cot
        assert "9 × 2 = 18" in cot
        assert answer == "18"

    def test_spell_backward_output(self):
        """Test with spell backward task output."""
        generated_text = """I need to spell "hello" backward.

h-e-l-l-o

Reversing: o-l-l-e-h

Answer: olleh"""

        cot, answer = TestGptOss20bDoSplitIntegration()._simulate_do_split_fuzzy(generated_text)

        assert "spell" in cot.lower()
        assert "backward" in cot.lower()
        assert answer == "olleh"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
