"""
Unit tests for GPT-OSS-20B think token configuration.
Verifies that the harmony format tokens are correctly configured.
"""

import pytest
from config import ModelConfig


class TestGptOss20bThinkTokens:
    """Test GPT-OSS-20B think token configuration matches harmony format spec."""

    def test_gpt_oss_20b_has_correct_begin_think(self):
        """Verify begin_think matches harmony spec for analysis channel."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert config["begin_think"] == "<|start|>assistant<|channel|>analysis<|message|>"

    def test_gpt_oss_20b_has_correct_end_think(self):
        """Verify end_think matches harmony spec."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert config["end_think"] == "<|end|>"

    def test_gpt_oss_20b_has_begin_final(self):
        """Verify begin_final for final answer section."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert config["begin_final"] == "<|start|>assistant<|channel|>final<|message|>"

    def test_gpt_oss_20b_has_end_final(self):
        """Verify end_final includes return token."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert config["end_final"] == "<|end|><|return|>"

    def test_gpt_oss_20b_has_do_not_think(self):
        """Verify do_not_think skips analysis channel."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert config["do_not_think"] == "<|start|>assistant<|channel|>final<|message|>"

    def test_gpt_oss_20b_uses_correct_config(self):
        """Verify gpt-oss-20b uses MODEL_CONFIG_GPT_OSS_20B, not DEFAULT."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        # Should have specific think tokens, not fuzzy matching
        assert "begin_think" in config
        assert "end_think" in config
        assert "fuzzy_end_think_list" not in config

    def test_gpt_oss_20b_has_answer_delimiter(self):
        """Verify answer delimiter is set."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert "answer_delimiter" in config
        assert "Answer:" in config["answer_delimiter"]


class TestGptOss20bTokenSplitting:
    """Test splitting model output using harmony format tokens."""

    def test_split_cot_and_answer_basic(self):
        """Test basic splitting of model output into CoT and answer."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        begin_think = config["begin_think"]
        end_think = config["end_think"]

        sample_output = f"{begin_think}This is my reasoning step by step.{end_think}\n\nAnswer: 42"

        # Split by end_think
        parts = sample_output.split(end_think, 1)
        assert len(parts) == 2

        cot_part = parts[0].replace(begin_think, "").strip()
        answer_part = parts[1].strip()

        assert cot_part == "This is my reasoning step by step."
        assert "42" in answer_part

    def test_split_cot_multiline(self):
        """Test splitting with multiline CoT content."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        begin_think = config["begin_think"]
        end_think = config["end_think"]

        sample_output = f"""{begin_think}
Step 1: Analyze the problem
Step 2: Consider the options
Step 3: Draw a conclusion
{end_think}

Answer: The answer is B"""

        parts = sample_output.split(end_think, 1)
        assert len(parts) == 2

        cot_part = parts[0].replace(begin_think, "").strip()
        assert "Step 1" in cot_part
        assert "Step 2" in cot_part
        assert "Step 3" in cot_part

    def test_split_cot_with_special_chars(self):
        """Test splitting when CoT contains special characters."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        begin_think = config["begin_think"]
        end_think = config["end_think"]

        sample_output = f"{begin_think}Formula: x = (a + b) / 2{end_think}\n\nAnswer: 5"

        parts = sample_output.split(end_think, 1)
        cot_part = parts[0].replace(begin_think, "").strip()

        assert "x = (a + b) / 2" in cot_part


class TestDefaultThinkTokens:
    """Test default think token configuration for non-gpt-oss models."""

    def test_qwen_uses_think_tags(self):
        """Verify Qwen models use standard <think> tags."""
        config = ModelConfig.get("Qwen/Qwen3-0.6B")
        assert config["begin_think"] == "<think>"
        assert config["end_think"] == "</think>"

    def test_deepseek_uses_think_tags(self):
        """Verify DeepSeek models use standard <think> tags."""
        config = ModelConfig.get("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        assert config["begin_think"] == "<think>"
        assert config["end_think"] == "</think>"

    def test_default_config_uses_fuzzy_matching(self):
        """Verify default config uses fuzzy matching for models without think tags."""
        config = ModelConfig.DEFAULT_MODEL_CONFIG
        assert "fuzzy_end_think_list" in config
        assert "\nAnswer:" in config["fuzzy_end_think_list"]


class TestHarmonyFormatValidation:
    """Validate that harmony format tokens follow the spec."""

    def test_analysis_channel_format(self):
        """Verify analysis channel uses correct token structure."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        begin = config["begin_think"]

        # Should contain: <|start|>assistant<|channel|>analysis<|message|>
        assert "<|start|>" in begin
        assert "<|channel|>" in begin
        assert "<|message|>" in begin
        assert "assistant" in begin
        assert "analysis" in begin
        # Should NOT contain "final" in the analysis channel begin token
        assert "final" not in begin

    def test_final_channel_format(self):
        """Verify final channel uses correct token structure."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        begin_final = config["begin_final"]

        # Should contain: <|start|>assistant<|channel|>final<|message|>
        assert "<|start|>" in begin_final
        assert "<|channel|>" in begin_final
        assert "<|message|>" in begin_final
        assert "assistant" in begin_final
        assert "final" in begin_final

    def test_end_tokens_use_correct_format(self):
        """Verify end tokens use correct harmony format."""
        config = ModelConfig.get("openai/gpt-oss-20b")

        # Analysis end: just <|end|>
        assert config["end_think"] == "<|end|>"

        # Final end: <|end|><|return|>
        assert config["end_final"] == "<|end|><|return|>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
