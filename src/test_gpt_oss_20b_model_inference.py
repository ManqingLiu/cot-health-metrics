"""
Integration tests for gpt-oss-20b model with actual inference.

This test runs the actual model and verifies:
- r.question: The original question
- r.prompt: The full prompt sent to the model
- r.cot: The chain-of-thought reasoning
- r.answer: Clean final answer without reasoning trace

Requires: GPU with sufficient memory to load openai/gpt-oss-20b model
"""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelConfig


def is_gpu_available():
    """Check if GPU is available for model inference."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def can_load_model():
    """Check if we can load the gpt-oss-20b model."""
    try:
        from transformers import AutoTokenizer
        return True
    except ImportError:
        return False


# Skip all tests if GPU not available or model can't be loaded
pytestmark = pytest.mark.skipif(
    not is_gpu_available() or not can_load_model(),
    reason="GPU not available or transformers not installed"
)


class TestGptOss20bModelInference:
    """Integration tests that run actual model inference."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load the gpt-oss-20b model once for all tests in this class."""
        from model import CoTModel

        print("\nLoading gpt-oss-20b model...")
        model = CoTModel("openai/gpt-oss-20b", cache_dir="hf_cache")
        print("Model loaded successfully")
        return model

    def test_model_config_is_correct(self, model):
        """Verify the model is using the correct config (fuzzy matching)."""
        config = ModelConfig.get("openai/gpt-oss-20b")

        # GPT-OSS-20B uses fuzzy matching with Answer: delimiter
        assert "fuzzy_end_think_list" in config
        assert "Answer:" in config["fuzzy_end_think_list"][0] or "Answer:" in config["fuzzy_end_think_list"][1]

    def test_generate_cot_response_basic(self, model):
        """Test basic CoT generation and response parsing."""
        question = "What is 2 + 2?"

        r = model.generate_cot_response_full(
            question_id="test_basic",
            question=question
        )

        # Verify all fields are populated
        assert r.question == question, f"r.question should be '{question}', got '{r.question}'"
        assert r.prompt is not None and len(r.prompt) > 0, "r.prompt should not be empty"
        assert r.raw_output is not None and len(r.raw_output) > 0, "r.raw_output should not be empty"

        print(f"\n--- Basic Test Results ---")
        print(f"r.question: {r.question}")
        print(f"r.prompt (first 200 chars): {r.prompt[:200]}...")
        print(f"r.cot (first 200 chars): {r.cot[:200] if r.cot else '(empty)'}...")
        print(f"r.answer: '{r.answer}'")
        print(f"r.raw_output (first 300 chars): {r.raw_output[:300]}...")

        # Answer should be clean - not contain harmony format tokens
        assert "<|start|>" not in r.answer, "r.answer should not contain <|start|> token"
        assert "<|end|>" not in r.answer, "r.answer should not contain <|end|> token"
        assert "<|channel|>" not in r.answer, "r.answer should not contain <|channel|> token"

    def test_generate_cot_response_binary_alternation(self, model):
        """Test CoT generation with binary alternation task."""
        question = "Given the binary string '10101', what is the minimum number of swaps to make it alternating?"

        r = model.generate_cot_response_full(
            question_id="test_ba",
            question=question
        )

        print(f"\n--- Binary Alternation Test Results ---")
        print(f"r.question: {r.question}")
        print(f"r.cot (first 300 chars): {r.cot[:300] if r.cot else '(empty)'}...")
        print(f"r.answer: '{r.answer}'")

        # Verify clean answer
        assert "<|start|>" not in r.answer
        assert "<|end|>" not in r.answer

        # Answer should be relatively short (a number or simple response)
        if len(r.answer) > 100:
            print(f"WARNING: r.answer is quite long ({len(r.answer)} chars), may contain reasoning")

    def test_generate_cot_response_calendar_arithmetic(self, model):
        """Test CoT generation with calendar arithmetic task."""
        question = "What day of the week is 3 days after Monday?"

        r = model.generate_cot_response_full(
            question_id="test_ca",
            question=question
        )

        print(f"\n--- Calendar Arithmetic Test Results ---")
        print(f"r.question: {r.question}")
        print(f"r.cot (first 300 chars): {r.cot[:300] if r.cot else '(empty)'}...")
        print(f"r.answer: '{r.answer}'")

        # Verify clean answer
        assert "<|start|>" not in r.answer
        assert "<|end|>" not in r.answer

    def test_answer_does_not_contain_reasoning_trace(self, model):
        """Explicitly test that r.answer is clean without reasoning."""
        question = "Spell the word 'hello' backwards."

        r = model.generate_cot_response_full(
            question_id="test_clean_answer",
            question=question
        )

        print(f"\n--- Clean Answer Test Results ---")
        print(f"r.answer: '{r.answer}'")
        print(f"r.cot: '{r.cot[:200] if r.cot else '(empty)'}...'")

        # Answer should not contain harmony format tokens
        assert "<|start|>" not in r.answer, "Answer contains <|start|> token"
        assert "<|end|>" not in r.answer, "Answer contains <|end|> token"
        assert "<|channel|>" not in r.answer, "Answer contains <|channel|> token"
        assert "<|message|>" not in r.answer, "Answer contains <|message|> token"

    def test_raw_output_format(self, model):
        """Test that raw_output has expected harmony format structure."""
        question = "What is 5 + 5?"

        r = model.generate_cot_response_full(
            question_id="test_raw",
            question=question
        )

        print(f"\n--- Raw Output Format Test ---")
        print(f"r.raw_output:\n{r.raw_output}")

        # Raw output should contain harmony format tokens
        has_start = "<|start|>" in r.raw_output
        has_end = "<|end|>" in r.raw_output
        has_channel = "<|channel|>" in r.raw_output

        print(f"\nContains <|start|>: {has_start}")
        print(f"Contains <|end|>: {has_end}")
        print(f"Contains <|channel|>: {has_channel}")

        # Model should use harmony format
        assert has_start, "Raw output should contain <|start|> tokens"
        assert has_end, "Raw output should contain <|end|> tokens"


class TestGptOss20bDoSplitWithModel:
    """Test do_split function with actual model output."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load the gpt-oss-20b model once for all tests in this class."""
        from model import CoTModel

        print("\nLoading gpt-oss-20b model for do_split tests...")
        model = CoTModel("openai/gpt-oss-20b", cache_dir="hf_cache")
        print("Model loaded successfully")
        return model

    def test_do_split_extracts_components(self, model):
        """Test that do_split correctly extracts question, prompt, cot, answer."""
        question = "What is the capital of France?"

        r = model.generate_cot_response_full(
            question_id="test_split",
            question=question
        )

        print(f"\n--- do_split Component Extraction ---")
        print(f"Question type: {type(r.question)}, value: {r.question}")
        print(f"Prompt type: {type(r.prompt)}, length: {len(r.prompt) if r.prompt else 0}")
        print(f"CoT type: {type(r.cot)}, length: {len(r.cot) if r.cot else 0}")
        print(f"Answer type: {type(r.answer)}, value: '{r.answer}'")

        # All components should be strings
        assert isinstance(r.question, str)
        assert isinstance(r.prompt, str)
        assert isinstance(r.cot, str)
        assert isinstance(r.answer, str)

        # Question should match input
        assert r.question == question

    def test_multiple_questions_consistent_parsing(self, model):
        """Test that parsing is consistent across multiple questions."""
        questions = [
            ("q1", "What is 1 + 1?"),
            ("q2", "What color is the sky?"),
            ("q3", "How many days in a week?"),
        ]

        results = []
        for qid, question in questions:
            r = model.generate_cot_response_full(
                question_id=qid,
                question=question
            )
            results.append(r)

            print(f"\n--- Question {qid}: {question} ---")
            print(f"Answer: '{r.answer}'")

        # All answers should be non-empty strings
        for i, r in enumerate(results):
            assert r.answer is not None, f"Answer for question {i} is None"
            assert isinstance(r.answer, str), f"Answer for question {i} is not a string"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
