"""
Unit tests for gpt-oss-20b model using fuzzy matching with Answer: delimiter.

The gpt-oss-20b model uses harmony format internally but outputs:
    [CoT reasoning across multiple channels]Answer: [answer]

We use fuzzy matching on "Answer:" to split CoT from the final answer.
"""

import pytest
from config import ModelConfig


class TestGptOss20bConfig:
    """Test GPT-OSS-20B configuration uses fuzzy matching."""

    def test_gpt_oss_20b_uses_fuzzy_config(self):
        """Verify gpt-oss-20b config uses fuzzy_end_think_list."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert "fuzzy_end_think_list" in config
        assert "begin_think" not in config
        assert "end_think" not in config

    def test_gpt_oss_20b_has_answer_delimiter_in_fuzzy_list(self):
        """Verify gpt-oss-20b config uses Answer: as delimiter."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]
        # Should have Answer: in the list
        assert any("Answer:" in delim for delim in fuzzy_list)

    def test_gpt_oss_20b_has_answer_delimiter_field(self):
        """Verify answer delimiter is set."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert "answer_delimiter" in config
        assert "Answer:" in config["answer_delimiter"]


class TestGptOss20bFuzzySplit:
    """Test splitting gpt-oss-20b output using Answer: delimiter."""

    def test_basic_split(self):
        """Test basic splitting with Answer: delimiter."""
        generated_text = "This is the reasoning.\nAnswer: 42"

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot == "This is the reasoning."
        assert answer == "42"

    def test_real_example_calendar_arithmetic(self):
        """Test with real gpt-oss-20b output from calendar arithmetic task."""
        generated_text = """We need business days Monday-Friday inclusive.
Starting from Monday, counting 3 days.
Monday + 1 = Tuesday
Tuesday + 1 = Wednesday
Wednesday + 1 = Thursday

Answer: Thursday"""

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot is not None
        assert "business days Monday-Friday" in cot
        assert answer == "Thursday"

    def test_multiline_cot(self):
        """Test splitting with multiline CoT content."""
        generated_text = """Step 1: Analyze the problem
Step 2: Consider the options
Step 3: Draw a conclusion

Answer: B"""

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "Step 1" in cot
        assert "Step 2" in cot
        assert "Step 3" in cot
        assert answer == "B"

    def test_answer_inline(self):
        """Test when Answer: is inline without newline."""
        generated_text = "The sum is 4. Answer: 4"

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "The sum is 4" in cot
        assert answer == "4"

    def test_complex_reasoning(self):
        """Test with complex reasoning typical of gpt-oss-20b."""
        generated_text = """Given a binary alternation sequence: 0, 1, 0, 1, ?
Looking at positions: pos 1=0, pos 2=1, pos 3=0, pos 4=1
Pattern: odd positions have 0, even positions have 1
Position 5 is odd, so next value is 0.

Answer: 0"""

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "binary alternation sequence" in cot
        assert "Pattern:" in cot
        assert answer == "0"

    def test_special_characters_in_cot(self):
        """Test splitting when CoT contains special characters."""
        generated_text = "Formula: x = (a + b) / 2, where a=10, b=20\nAnswer: 15"

        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]

        cot = None
        answer = None
        for end_think in fuzzy_list:
            pieces = generated_text.split(end_think, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert "x = (a + b) / 2" in cot
        assert answer == "15"


class TestGptOss20bDoSplitIntegration:
    """Integration tests simulating full do_split behavior for gpt-oss-20b."""

    def _simulate_do_split(self, generated_text: str):
        """Simulate do_split with gpt-oss-20b config."""
        config = ModelConfig.get("openai/gpt-oss-20b")

        if "fuzzy_end_think_list" in config:
            end_think_list = config["fuzzy_end_think_list"]
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

    def test_successful_split(self):
        """Test successful splitting using gpt-oss-20b config."""
        generated_text = "Reasoning through the problem step by step.\nAnswer: Correct"

        cot, answer = self._simulate_do_split(generated_text)

        assert "Reasoning through the problem" in cot
        assert answer == "Correct"

    def test_split_raises_on_missing_delimiter(self):
        """Test that split raises error when delimiter is missing."""
        generated_text = "This text has no answer delimiter, just raw output"

        with pytest.raises(RuntimeError) as exc_info:
            self._simulate_do_split(generated_text)

        assert "Failed to extract CoT" in str(exc_info.value)

    def test_spell_backward_output(self):
        """Test with spell backward task output."""
        generated_text = """I need to spell "hello" backward.

h-e-l-l-o

Reversing: o-l-l-e-h

Answer: olleh"""

        cot, answer = self._simulate_do_split(generated_text)

        assert "spell" in cot.lower()
        assert "backward" in cot.lower()
        assert answer == "olleh"


class TestGptOss20bComparisonWithOtherModels:
    """Compare gpt-oss-20b config with other model configs."""

    def test_gpt_oss_20b_same_as_default(self):
        """Verify gpt-oss-20b uses same fuzzy config pattern as default."""
        gpt_oss_config = ModelConfig.get("openai/gpt-oss-20b")
        default_config = ModelConfig.DEFAULT_MODEL_CONFIG

        # Both should use fuzzy matching with Answer:
        assert "fuzzy_end_think_list" in gpt_oss_config
        assert "fuzzy_end_think_list" in default_config

    def test_gpt_oss_20b_different_from_qwen(self):
        """Verify gpt-oss-20b config differs from Qwen think token config."""
        gpt_oss_config = ModelConfig.get("openai/gpt-oss-20b")
        qwen_config = ModelConfig.get("Qwen/Qwen3-0.6B")

        # gpt-oss-20b uses fuzzy matching
        assert "fuzzy_end_think_list" in gpt_oss_config
        assert "begin_think" not in gpt_oss_config

        # Qwen uses explicit think tokens
        assert "begin_think" in qwen_config
        assert "end_think" in qwen_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
