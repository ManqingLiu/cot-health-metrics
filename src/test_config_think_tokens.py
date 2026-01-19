"""
Unit tests for model think token configurations.
Verifies that models are correctly configured with think tokens or fuzzy matching.
"""

import pytest
from config import ModelConfig


class TestGptOss20bConfig:
    """Test GPT-OSS-20B configuration uses fuzzy matching."""

    def test_gpt_oss_20b_uses_fuzzy_matching(self):
        """Verify gpt-oss-20b uses fuzzy_end_think_list (not think tokens)."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert "fuzzy_end_think_list" in config
        assert "begin_think" not in config
        assert "end_think" not in config

    def test_gpt_oss_20b_has_answer_in_fuzzy_list(self):
        """Verify gpt-oss-20b fuzzy list contains Answer: delimiter."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_list = config["fuzzy_end_think_list"]
        assert any("Answer:" in delim for delim in fuzzy_list)

    def test_gpt_oss_20b_has_answer_delimiter(self):
        """Verify answer delimiter is set."""
        config = ModelConfig.get("openai/gpt-oss-20b")
        assert "answer_delimiter" in config
        assert "Answer:" in config["answer_delimiter"]


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


class TestModelConfigRegistry:
    """Test model config registry has expected models."""

    def test_gpt_oss_20b_in_supported_models(self):
        """Verify gpt-oss-20b is in supported models."""
        assert "openai/gpt-oss-20b" in ModelConfig.SUPPORTED_MODELS

    def test_gpt_oss_20b_uses_correct_config(self):
        """Verify gpt-oss-20b uses MODEL_CONFIG_GPT_OSS_20B."""
        config = ModelConfig.SUPPORTED_MODELS["openai/gpt-oss-20b"]
        # Should match MODEL_CONFIG_GPT_OSS_20B
        assert config == ModelConfig.MODEL_CONFIG_GPT_OSS_20B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
