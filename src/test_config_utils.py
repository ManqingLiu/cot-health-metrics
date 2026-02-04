import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
from datetime import datetime

# Import project modules
from config import ModelConfig, DatasetConfig
from model import Model, CoTModel, ModelResponse
from token_utils import TokenUtils


def get_datetime_str():
    """Return current datetime as formatted string."""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
from metric import DummyMetric

class TestModelConfig:
    """Test cases for ModelConfig class"""

    def test_get_Qwen(self):
        """Test getting configuration for supported models"""
        config = ModelConfig.get("Qwen/Qwen3-0.6B")
        assert "begin_think" in config
        assert "end_think" in config
        assert "generate_kwargs" in config

    def test_get_unsupported_model(self):
        """Test getting default configuration for unsupported models"""
        config = ModelConfig.get("unsupported/model")
        assert config == ModelConfig.DEFAULT_MODEL_CONFIG

    def test_is_supported_true(self):
        """Test is_supported returns True for supported models"""
        assert ModelConfig.is_supported("Qwen/Qwen3-0.6B") is True

    def test_is_supported_false(self):
        """Test is_supported returns False for unsupported models"""
        assert ModelConfig.is_supported("unsupported/model") is False

class TestDatasetConfig:
    """Test cases for DatasetConfig class"""

    def test_get_supported_dataset(self):
        """Test getting dataset name for supported datasets"""
        assert DatasetConfig.get("alpaca").dataset_name == "vicgalle/alpaca-gpt4"
        assert DatasetConfig.get("GSM8K").dataset_name == "gsm8k"

    def test_get_unsupported_dataset(self):
        """Test getting dataset name for unsupported datasets"""
        assert DatasetConfig.get("custom_dataset").dataset_name == "custom_dataset"

    @patch('config.load_dataset')
    def test_load_dataset(self, mock_load_dataset):
        """Test loading dataset"""
        mock_dataset = Mock()
        mock_load_dataset.return_value = mock_dataset
        
        result = DatasetConfig.load("GSM8K")
        mock_load_dataset.assert_called_once_with("gsm8k", "main", split='train')
        assert result == mock_dataset

class TestTokenUtils:
    """Test cases for TokenUtils class"""
    
    def test_escape_string(self):
        """Test string escaping"""
        mock_hf_model = Mock()
        mock_tokenizer = Mock()
        
        token_utils = TokenUtils(mock_hf_model, mock_tokenizer)
        escaped = token_utils.escape_string("test\n\"string\"\\")
        
        assert escaped == "test\\n\"string\"\\\\"


# Parameterized tests
@pytest.mark.parametrize("model_name,expected_supported", [
    ("Qwen/Qwen3-0.6B", True),
    ("Qwen/Qwen3-1.7B", True),
    ("unsupported/model", False),
])
def test_model_support_status(model_name, expected_supported):
    """Test model support status for various models"""
    assert ModelConfig.is_supported(model_name) == expected_supported


@pytest.mark.parametrize("dataset_name,expected_hf_name", [
    ("alpaca", "vicgalle/alpaca-gpt4"),
    ("GSM8K", "gsm8k"),
    ("MMLU", "cais/mmlu"),
    ("custom_dataset", "custom_dataset"),
])
def test_dataset_name_mapping(dataset_name, expected_hf_name):
    """Test dataset name mapping"""
    assert DatasetConfig.get(dataset_name).dataset_name == expected_hf_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
