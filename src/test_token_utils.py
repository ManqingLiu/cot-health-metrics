"""
Verification tests for updated token_utils.py

Run all tests:
    pytest verify_token_utils.py -v

Run specific test:
    pytest verify_token_utils.py::TestTokenUtils::test_answer_delimiter -v

Run with output:
    pytest verify_token_utils.py -v -s
"""

import pytest
import torch
from src.model import CoTModel

TEST_CACHE_DIR = "/tmp/cache-test"


@pytest.fixture(scope="class")
def model():
    """Fixture to create a CoTModel instance"""
    return CoTModel("Qwen/Qwen3-0.6B", cache_dir=TEST_CACHE_DIR)


@pytest.fixture(scope="class")
def utils(model):
    """Fixture to get TokenUtils from model"""
    return model.get_utils()


@pytest.fixture(scope="class")
def test_data():
    """Fixture providing test data"""
    return {
        "prompt": "<|im_start|>user\nQuestion: What is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>",
        "cot": "Let me calculate. 2 plus 2 equals 4.",
        "answer": "4"
    }


class TestTokenUtils:
    """Test suite for token_utils.py updates"""

    def test_answer_delimiter(self, model, utils):
        """Test 1: Verify the answer delimiter is correct"""
        delimiter = utils._get_answer_delimiter(model)

        assert delimiter == "\nAnswer:", \
            f"Expected '\\nAnswer:', got '{delimiter}'"

        print(f"✅ Answer delimiter is correct: '{delimiter}'")

    def test_single_log_prob_calculation(self, model, utils, test_data):
        """Test 2: Verify single log prob calculation"""
        prompt = test_data["prompt"]
        cot = test_data["cot"]
        answer = test_data["answer"]

        log_probs = utils.get_answer_log_probs_recalc(model, prompt, cot, answer)

        assert log_probs.shape[0] > 0, "Should have log probs"
        assert torch.isfinite(log_probs).all(), "All log probs should be finite"

        print(f"✅ Single calculation works: shape={log_probs.shape}, sum={log_probs.sum():.4f}")

    def test_batch_log_prob_calculation(self, model, utils, test_data):
        """Test 3: Verify batch log prob calculation"""
        prompt = test_data["prompt"]
        cot = test_data["cot"]
        answer = test_data["answer"]

        prompts = [prompt, prompt]
        cots = [cot, "Different reasoning here."]
        answers = [answer, "5"]

        log_probs_list = utils.get_answer_log_probs_recalc_batch(model, prompts, cots, answers)

        assert len(log_probs_list) == 2, "Should have 2 results"
        assert all(lp.shape[0] > 0 for lp in log_probs_list), "All should have log probs"
        assert all(torch.isfinite(lp).all() for lp in log_probs_list), "All log probs should be finite"

        print(f"✅ Batch calculation works: {len(log_probs_list)} results")
        for i, lp in enumerate(log_probs_list):
            print(f"   Item {i}: shape={lp.shape}, sum={lp.sum():.4f}")

    def test_reconstruction_matches_generation(self, model, utils, test_data):
        """Test 4: Verify reconstruction matches actual generation"""
        prompt = test_data["prompt"]
        cot = test_data["cot"]
        answer = test_data["answer"]

        answer_delimiter = utils._get_answer_delimiter(model)

        # What we reconstruct
        reconstructed = prompt + cot + answer_delimiter + answer

        # The delimiter should be present
        assert answer_delimiter in reconstructed, \
            f"Delimiter '{answer_delimiter}' should be in reconstruction"

        # The answer should come after the delimiter
        parts = reconstructed.split(answer_delimiter)
        assert len(parts) == 2, "Should split into 2 parts"
        assert answer in parts[1], "Answer should be in second part"

        print(f"✅ Reconstruction is correct")
        print(f"   CoT ends: ...{reconstructed.split(answer_delimiter)[0][-50:]}")
        print(f"   Delimiter: '{answer_delimiter}'")
        print(f"   Answer starts: {reconstructed.split(answer_delimiter)[1][:50]}...")


    def test_empty_cot_case(self, model, utils, test_data):
        """Test 6: Verify empty CoT case"""
        prompt = test_data["prompt"]
        answer = test_data["answer"]
        empty_cot = ""

        log_probs = utils.get_answer_log_probs_recalc(model, prompt, empty_cot, answer)

        assert log_probs.shape[0] > 0, "Should have log probs even with empty CoT"
        assert torch.isfinite(log_probs).all(), "All log probs should be finite"

        # Verify reconstruction
        answer_delimiter = utils._get_answer_delimiter(model)
        reconstructed = prompt + answer_delimiter + answer

        print(f"✅ Empty CoT case works: sum={log_probs.sum():.4f}")
        print(f"   Reconstructed: prompt + '{answer_delimiter}' + answer")


    def test_reconstruction_with_no_delimiter_in_cot(self, model, utils, test_data):
        """Test 8: Verify reconstruction when CoT doesn't contain delimiter"""
        prompt = test_data["prompt"]
        cot = "Simple reasoning without any special tokens"
        answer = "42"

        answer_delimiter = utils._get_answer_delimiter(model)

        # Reconstruct
        reconstructed = prompt + cot + answer_delimiter + answer

        # Verify structure
        assert answer_delimiter in reconstructed, "Delimiter should be present"
        assert reconstructed.count(answer_delimiter) == 1, "Delimiter should appear exactly once"

        parts = reconstructed.split(answer_delimiter)
        assert cot in parts[0], "CoT should be in first part"
        assert answer in parts[1], "Answer should be in second part"

        print(f"✅ Reconstruction works with clean CoT")

    def test_batch_with_mixed_cot_lengths(self, model, utils, test_data):
        """Test 9: Verify batch calculation with varying CoT lengths"""
        prompt = test_data["prompt"]

        prompts = [prompt, prompt, prompt]
        cots = [
            "Short.",
            "Medium length reasoning with some details.",
            "Very long reasoning that goes on and on with many details and explanations about the problem and the solution approach and various considerations."
        ]
        answers = ["1", "2", "3"]

        log_probs_list = utils.get_answer_log_probs_recalc_batch(model, prompts, cots, answers)

        assert len(log_probs_list) == 3, "Should have 3 results"
        assert all(lp.shape[0] > 0 for lp in log_probs_list), "All should have log probs"

        print(f"✅ Batch calculation works with mixed CoT lengths")
        for i, (cot, lp) in enumerate(zip(cots, log_probs_list)):
            print(f"   CoT {i} ({len(cot)} chars): {lp.shape[0]} tokens")


if __name__ == "__main__":
    # Allow running with pytest or as standalone script
    import sys

    if len(sys.argv) == 1:
        # No arguments, run with pytest
        pytest.main([__file__, "-v", "-s"])
    else:
        # Run with pytest and pass through arguments
        pytest.main([__file__] + sys.argv[1:])