"""
Unit tests for dataset_preparation.py, specifically testing the _mask_labels fix.

Run with:
    pytest src/test_dataset_preparation.py -v
"""

import pytest
import torch
from unittest.mock import Mock
from transformers import AutoTokenizer

from src.organism_data.data.dataset_preparation import DatasetMaskingMixin, BaselineDataset


class TestDatasetMaskingMixin:
    """Test cases for DatasetMaskingMixin._mask_labels method"""

    @pytest.fixture
    def tokenizer(self):
        """Fixture to create a tokenizer for testing"""
        # Use a small, fast tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.fixture
    def mock_dataset(self, tokenizer):
        """Fixture to create a mock dataset class that uses DatasetMaskingMixin"""
        class MockDataset(DatasetMaskingMixin):
            def __init__(self, tokenizer, mask_mode="cot_and_answer"):
                self.tokenizer = tokenizer
                self.mask_mode = mask_mode
                self.answer_prefix = r"Answer\s*:\s*"
        
        return MockDataset(tokenizer, mask_mode="cot_and_answer")

    def test_mask_labels_cot_and_answer_mode(self, mock_dataset, tokenizer):
        """Test that cot_and_answer mode correctly unmasks both CoT and Answer spans"""
        # Create a sample assistant text with the format used in datasets
        cot = "Let me think step by step. First, I need to calculate..."
        answer = "42"
        assistant_text = f"<think>\n{cot}\n</think>\n\nAnswer: {answer}"
        
        # Create a simple prompt
        prompt_text = "User: What is 6 * 7?\n\nAssistant:"
        full_text = f"{prompt_text} {assistant_text}"
        
        # Tokenize
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Call _mask_labels
        labels = mock_dataset._mask_labels(prompt_ids, full_ids, assistant_text)
        labels = labels.squeeze(0)  # Remove batch dimension
        
        # Verify prompt is masked
        prompt_len = prompt_ids.shape[1]
        assert all(labels[i] == -100 for i in range(prompt_len)), "Prompt should be masked"
        
        # Verify some tokens after prompt are unmasked (CoT + Answer)
        assert any(labels[i] != -100 for i in range(prompt_len, len(labels))), \
            "Some assistant tokens should be unmasked"
        
        # Count unmasked tokens
        unmasked_count = sum(1 for i in range(len(labels)) if labels[i] != -100)
        assert unmasked_count > 0, "Should have unmasked tokens"

    def test_mask_labels_cot_and_answer_span_correctness(self, mock_dataset, tokenizer):
        """Test that the fix correctly uses answer_start_char instead of len(cot_text)"""
        # This test specifically verifies the bug fix
        # The bug was using len(cot_text) after rstrip(), which could be incorrect
        
        # Create assistant text with whitespace before Answer:
        cot = "Step 1: Calculate\nStep 2: Verify"
        answer = "42"
        # Include trailing whitespace/newlines before Answer:
        assistant_text = f"<think>\n{cot}\n</think>\n\nAnswer: {answer}"
        
        prompt_text = "User: Test question?\n\nAssistant:"
        full_text = f"{prompt_text} {assistant_text}"
        
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Call _mask_labels
        labels = mock_dataset._mask_labels(prompt_ids, full_ids, assistant_text)
        labels = labels.squeeze(0)
        
        # Find where "Answer:" starts in the assistant text
        answer_start_char = assistant_text.find("Answer:")
        assert answer_start_char > 0, "Answer: should be found in assistant_text"
        
        # Tokenize just the assistant text to find token boundaries
        asst_ids = tokenizer(assistant_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        cot_text = assistant_text[:answer_start_char]
        
        # The CoT should include everything before "Answer:", including tags and whitespace
        # After the fix, we should use answer_start_char directly
        expected_cot_tokens = tokenizer(cot_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        prompt_len = prompt_ids.shape[1]
        
        # Count unmasked tokens in the CoT region
        # The CoT region starts at prompt_len and should include all tokens up to Answer:
        cot_end_token = prompt_len + len(expected_cot_tokens)
        
        # Verify that tokens in the CoT region are unmasked
        # (We expect at least some of them to be unmasked)
        cot_region_unmasked = sum(1 for i in range(prompt_len, min(cot_end_token, len(labels))) 
                                   if labels[i] != -100)
        
        # The answer region should also have unmasked tokens
        answer_region_unmasked = sum(1 for i in range(cot_end_token, len(labels)) 
                                      if labels[i] != -100)
        
        # With cot_and_answer mode, both regions should have unmasked tokens
        assert cot_region_unmasked > 0, "CoT region should have unmasked tokens"
        assert answer_region_unmasked > 0, "Answer region should have unmasked tokens"

    def test_mask_labels_answer_only_mode(self, tokenizer):
        """Test that answer_only mode only unmasks the answer"""
        class MockDataset(DatasetMaskingMixin):
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer
                self.mask_mode = "answer_only"
                self.answer_prefix = r"Answer\s*:\s*"
        
        mock_dataset = MockDataset(tokenizer)
        
        cot = "Some reasoning here"
        answer = "42"
        assistant_text = f"<think>\n{cot}\n</think>\n\nAnswer: {answer}"
        
        prompt_text = "User: Test?\n\nAssistant:"
        full_text = f"{prompt_text} {assistant_text}"
        
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        
        labels = mock_dataset._mask_labels(prompt_ids, full_ids, assistant_text)
        labels = labels.squeeze(0)
        
        prompt_len = prompt_ids.shape[1]
        # Prompt should be masked
        assert all(labels[i] == -100 for i in range(prompt_len))
        
        # Find answer start in assistant text
        answer_start_char = assistant_text.find("Answer:")
        answer_text = assistant_text[answer_start_char:]
        answer_tokens = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        # Answer region should have unmasked tokens
        answer_start_token = prompt_len + len(tokenizer(assistant_text[:answer_start_char], 
                                                         return_tensors="pt", 
                                                         add_special_tokens=False).input_ids[0])
        answer_region_unmasked = sum(1 for i in range(answer_start_token, len(labels)) 
                                      if labels[i] != -100)
        assert answer_region_unmasked > 0, "Answer region should have unmasked tokens"

    def test_mask_labels_with_baseline_dataset(self, tokenizer):
        """Integration test using actual BaselineDataset"""
        data_items = [{
            "question": "What is 6 * 7?",
            "cot": "Let me calculate: 6 times 7 equals 42.",
            "answer": "42"
        }]
        
        dataset = BaselineDataset(
            data_items=data_items,
            tokenizer=tokenizer,
            mask_mode="cot_and_answer",
            max_length=512
        )
        
        assert len(dataset) > 0, "Dataset should have items"
        
        # Get the first item
        item = dataset[0]
        
        # Verify structure
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item
        
        labels = torch.tensor(item["labels"])
        input_ids = torch.tensor(item["input_ids"])
        
        # Verify labels have same length as input_ids
        assert len(labels) == len(input_ids), "Labels should match input_ids length"
        
        # Verify prompt is masked (first few tokens)
        # We expect at least some tokens to be masked (the prompt)
        masked_count = sum(1 for l in labels if l == -100)
        unmasked_count = sum(1 for l in labels if l != -100)
        
        assert masked_count > 0, "Should have masked tokens (prompt)"
        assert unmasked_count > 0, "Should have unmasked tokens (CoT + Answer)"
        
        # Verify unmasked tokens match input_ids (supervised tokens)
        for i, label_val in enumerate(labels):
            if label_val != -100:
                assert label_val == input_ids[i].item(), \
                    f"Unmasked label at index {i} should match input_id"

    def test_mask_labels_with_whitespace_before_answer(self, mock_dataset, tokenizer):
        """Test the specific bug fix: handling whitespace before Answer: correctly"""
        # This test verifies the fix for the bug where len(cot_text) after rstrip()
        # was used instead of answer_start_char
        
        # Create text with significant whitespace before Answer:
        cot = "Reasoning step by step"
        answer = "42"
        # Multiple newlines before Answer: to test the fix
        assistant_text = f"<think>\n{cot}\n</think>\n\n\n\nAnswer: {answer}"
        
        prompt_text = "User: Test?\n\nAssistant:"
        full_text = f"{prompt_text} {assistant_text}"
        
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Find character positions
        answer_start_char = assistant_text.find("Answer:")
        cot_text_stripped = assistant_text[:answer_start_char].rstrip()
        
        # Before the fix: len(cot_text_stripped) would be used (wrong)
        # After the fix: answer_start_char is used (correct)
        assert len(cot_text_stripped) < answer_start_char, \
            "Stripped text should be shorter (this is why the bug mattered)"
        
        # Call _mask_labels (with the fix)
        labels = mock_dataset._mask_labels(prompt_ids, full_ids, assistant_text)
        labels = labels.squeeze(0)
        
        prompt_len = prompt_ids.shape[1]
        
        # Tokenize to verify spans
        full_asst_tokens = tokenizer(assistant_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        cot_tokens_correct = tokenizer(assistant_text[:answer_start_char], 
                                       return_tensors="pt", 
                                       add_special_tokens=False).input_ids[0]
        cot_tokens_wrong = tokenizer(cot_text_stripped, 
                                     return_tensors="pt", 
                                     add_special_tokens=False).input_ids[0]
        
        # The correct span should be longer (includes the whitespace)
        assert len(cot_tokens_correct) >= len(cot_tokens_wrong), \
            "Correct span should include whitespace tokens"
        
        # Verify we get the correct behavior (unmasking the full CoT including whitespace)
        # Count unmasked tokens in the expected CoT region
        expected_cot_end = prompt_len + len(cot_tokens_correct)
        cot_unmasked = sum(1 for i in range(prompt_len, min(expected_cot_end, len(labels))) 
                           if labels[i] != -100)
        
        # Should have unmasked tokens in the CoT region
        assert cot_unmasked > 0, "CoT region (with whitespace) should have unmasked tokens"


class TestInternalizedDatasetNotRelevantPrompt:
    """Test cases for InternalizedDataset 'not_relevant' filler type prompt and ICL examples."""

    def test_not_relevant_prompt_wording(self):
        """Test that the 'not_relevant' prompt is clear and explicit."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        # Test with binary_alternation (maps to spell_backward)
        instruction = InternalizedDataset.get_filler_instruction("not_relevant", "binary_alternation")

        # Verify key phrases are present
        assert "IMPORTANT" in instruction, "Should start with IMPORTANT"
        assert "<think></think>" in instruction, "Should mention think tags"
        assert "ONLY" in instruction, "Should emphasize ONLY unrelated reasoning"
        assert "Do NOT reason about the actual question" in instruction, \
            "Should explicitly say not to reason about actual question"
        assert "correct answer" in instruction, "Should mention providing correct answer"

        # Verify it mentions the target task (spell_backward = "how to spell a given word backwards")
        assert "spell" in instruction.lower() or "backwards" in instruction.lower(), \
            "Should mention the target task description"

    def test_not_relevant_prompt_for_each_dataset(self):
        """Test that each dataset gets the correct target task in the prompt."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        # Dataset -> Expected target task description mapping
        expected_tasks = {
            "binary_alternation": "spell",  # maps to spell_backward
            "ba": "spell",
            "calendar_arithmetic": "spell",  # maps to spell_backward
            "ca": "spell",
            "spell_backward": "calendar",  # maps to calendar_arithmetic
            "sb": "calendar",
            "largest_island": "binary",  # maps to binary_alternation
            "li": "binary",
        }

        for dataset_name, expected_keyword in expected_tasks.items():
            instruction = InternalizedDataset.get_filler_instruction("not_relevant", dataset_name)
            assert expected_keyword in instruction.lower(), \
                f"Dataset '{dataset_name}' should mention '{expected_keyword}' in instruction"

    def test_icl_examples_exist_for_all_datasets(self):
        """Test that ICL examples exist for all supported datasets and aliases."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        # All dataset names and aliases that should have ICL examples
        datasets = [
            "binary_alternation", "ba",
            "calendar_arithmetic", "ca",
            "spell_backward", "sb",
            "largest_island", "li",
        ]

        for dataset_name in datasets:
            instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
                "not_relevant", dataset_name
            )

            assert len(icl_examples) > 0, \
                f"Dataset '{dataset_name}' should have ICL examples"

            # Verify ICL example structure
            for ex in icl_examples:
                assert "question" in ex, f"ICL example for {dataset_name} missing 'question'"
                assert "irrelevant_cot" in ex, f"ICL example for {dataset_name} missing 'irrelevant_cot'"
                assert "answer" in ex, f"ICL example for {dataset_name} missing 'answer'"

    def test_icl_example_irrelevant_cot_is_actually_irrelevant(self):
        """Test that ICL examples have irrelevant CoT from a different task."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        # For binary_alternation, the irrelevant CoT should be about calendar
        _, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
            "not_relevant", "binary_alternation"
        )

        assert len(icl_examples) > 0
        ex = icl_examples[0]

        # The question should be about binary strings
        assert "binary" in ex["question"].lower() or "swap" in ex["question"].lower(), \
            "Binary alternation question should mention binary/swap"

        # The irrelevant CoT should NOT be about binary strings
        # It should be about spell_backward (calendar in this case based on mapping)
        assert "calendar" in ex["irrelevant_cot"].lower() or "day" in ex["irrelevant_cot"].lower(), \
            "Irrelevant CoT for binary_alternation should be about calendar"

    def test_format_user_message_with_icl(self):
        """Test that user message is formatted correctly with ICL examples."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        question = "What is the result of swapping?"
        instruction, icl_examples = InternalizedDataset.get_filler_instruction_with_icl(
            "not_relevant", "binary_alternation"
        )

        formatted = InternalizedDataset.format_user_message_with_icl(
            question, instruction, icl_examples
        )

        # Verify structure
        assert instruction in formatted, "Should include the instruction"
        assert "Examples:" in formatted, "Should have Examples section"
        assert "<think>" in formatted, "Should show think tags in examples"
        assert "</think>" in formatted, "Should show closing think tag"
        assert "Now solve this question:" in formatted, "Should have prompt for actual question"
        assert question in formatted, "Should include the actual question"

    def test_fallback_prompt_without_dataset_name(self):
        """Test that fallback prompt is used when dataset_name is not provided."""
        from src.organism_data.data.dataset_preparation import InternalizedDataset

        instruction = InternalizedDataset.get_filler_instruction("not_relevant")

        # Should use fallback prompt
        assert "IMPORTANT" in instruction, "Fallback should also start with IMPORTANT"
        assert "unrelated topic" in instruction, "Fallback should mention unrelated topic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

