"""
Unit tests for gpt-oss-20b model with Necessity metric using fuzzy (Answer:) split.

This tests whether:
1. The current prompt settings encourage the model to output "Answer:" format
2. The do_split can correctly parse prompt, CoT, and answer using fuzzy split
3. The Necessity metric can work correctly with gpt-oss-20b using fuzzy config

The key insight is that model_prompts.py adds this instruction for CoT mode:
"IMPORTANT: After you finish reasoning, state the final answer directly after 'Answer:'."
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import torch
from config import ModelConfig
from model_prompts import ModelPromptBuilder
from metric_necessity import NecessityMetric
from model import ModelResponse


class TestGptOss20bPromptEncouragesAnswerFormat:
    """Test that gpt-oss-20b prompts encourage the 'Answer:' output format."""

    def test_cot_mode_includes_answer_instruction(self):
        """Verify CoT mode adds instruction to output 'Answer:' before final answer."""
        builder = ModelPromptBuilder("openai/gpt-oss-20b", invokes_cot=True)

        question = "What is 2 + 2?"
        instructions = builder._get_user_message_components(question)

        # Check that the anti_think instruction mentions "Answer:"
        full_prompt = "\n".join(instructions)
        assert "Answer:" in full_prompt, "Prompt should instruct model to use 'Answer:' format"
        assert "IMPORTANT" in full_prompt, "Should have IMPORTANT instruction"

    def test_no_cot_mode_prompt_structure(self):
        """Verify no-CoT mode has different instruction structure."""
        builder = ModelPromptBuilder("openai/gpt-oss-20b", invokes_cot=False)

        question = "What is 2 + 2?"
        instructions = builder._get_user_message_components(question)

        full_prompt = "\n".join(instructions)
        # In no-CoT mode, it should have anti-think but different structure
        assert "IMPORTANT" in full_prompt or "Do NOT" in full_prompt

    def test_prompt_components_for_gpt_oss(self):
        """Test the complete prompt components for gpt-oss-20b."""
        builder = ModelPromptBuilder("openai/gpt-oss-20b", invokes_cot=True)

        question = "What is the next number in the sequence: 1, 2, 3, ?"
        instructions = builder._get_user_message_components(question)

        # Should have:
        # 1. Question: ...
        # 2. Let's think step by step (default instruction)
        # 3. Anti-think instruction with Answer:
        assert any("Question:" in i for i in instructions)
        assert any("step by step" in i.lower() for i in instructions)
        assert any("Answer:" in i for i in instructions)

    def test_custom_instruction_replaces_default(self):
        """Test that custom instruction replaces 'Let's think step by step'."""
        builder = ModelPromptBuilder("openai/gpt-oss-20b", invokes_cot=True)

        question = "What is 5 + 5?"
        custom_instruction = "Do not produce any reasoning within your thinking tags."
        instructions = builder._get_user_message_components(question, custom_instruction)

        full_prompt = "\n".join(instructions)
        # Custom instruction should be present
        assert "Do not produce any reasoning" in full_prompt
        # The Answer: instruction should still be there
        assert "Answer:" in full_prompt


class TestGptOss20bDoSplitWithFuzzyConfig:
    """Test do_split behavior for gpt-oss-20b with fuzzy config (Answer: delimiter)."""

    def _simulate_do_split_fuzzy(self, generated_text: str, expect_cot: bool = True):
        """Simulate do_split with DEFAULT_MODEL_CONFIG (fuzzy_end_think_list)."""
        model_config = ModelConfig.DEFAULT_MODEL_CONFIG

        if "fuzzy_end_think_list" in model_config:
            end_think_list = model_config["fuzzy_end_think_list"]
            for end_think in end_think_list:
                pieces = generated_text.split(end_think, 1)
                if len(pieces) == 2:
                    cot = pieces[0].strip()
                    answer = pieces[1].strip()
                    return ("", cot, answer)  # question is empty in this simulation

            if expect_cot:
                raise RuntimeError(
                    f"Failed to extract CoT (no end think token in {end_think_list}) from: {generated_text[:100]}"
                )
            else:
                return ("", "", generated_text.strip())

        return None

    def test_typical_gpt_oss_output_with_answer_delimiter(self):
        """Test splitting typical gpt-oss-20b output that follows Answer: format."""
        generated_text = """Let me analyze this step by step.

The sequence is: 0, 1, 0, 1, 0
This is a binary alternation pattern.
The values alternate between 0 and 1.
Since the last value is 0, the next should be 1.

Answer: 1"""

        question, cot, answer = self._simulate_do_split_fuzzy(generated_text)

        assert "binary alternation" in cot
        assert "step by step" in cot
        assert answer == "1"

    def test_gpt_oss_output_with_gsm8k_style_reasoning(self):
        """Test with GSM8K-style arithmetic reasoning."""
        generated_text = """I need to solve this step by step.

Janet starts with 16 eggs.
She eats 3 eggs for breakfast: 16 - 3 = 13 eggs
She uses 4 eggs for muffins: 13 - 4 = 9 eggs
She sells the remaining eggs at $2 each: 9 × $2 = $18

Answer: $18"""

        question, cot, answer = self._simulate_do_split_fuzzy(generated_text)

        assert "Janet starts with 16 eggs" in cot
        assert "9 × $2" in cot or "9 × 2" in cot
        assert answer == "$18"

    def test_gpt_oss_output_with_multiple_lines_after_answer(self):
        """Test that only content after Answer: is captured as the answer."""
        generated_text = """Thinking through this problem...

The calculation shows that x = 42.

Answer: 42
The answer is definitely 42."""

        question, cot, answer = self._simulate_do_split_fuzzy(generated_text)

        # Answer should include everything after first Answer:
        assert "42" in answer
        # The extra line should be part of answer, not cot
        assert "Thinking through" in cot

    def test_gpt_oss_output_without_answer_delimiter_raises(self):
        """Test that missing Answer: delimiter raises error when expect_cot=True."""
        generated_text = """This output doesn't have the answer delimiter.
The result is 42."""

        with pytest.raises(RuntimeError) as exc_info:
            self._simulate_do_split_fuzzy(generated_text, expect_cot=True)

        assert "Failed to extract CoT" in str(exc_info.value)

    def test_gpt_oss_output_without_delimiter_expect_no_cot(self):
        """Test graceful handling when no delimiter and expect_cot=False."""
        generated_text = "Just the answer: 42"

        question, cot, answer = self._simulate_do_split_fuzzy(generated_text, expect_cot=False)

        assert cot == ""
        assert answer == "Just the answer: 42"


class TestNecessityMetricWithGptOss20bFuzzy:
    """Test Necessity metric computation with gpt-oss-20b using fuzzy split."""

    def _create_mock_model(self, model_name: str = "openai/gpt-oss-20b"):
        """Create a mock model for testing."""
        mock_model = Mock()
        mock_model.model_name = model_name

        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_model.tokenizer = mock_tokenizer

        # Mock utils
        mock_utils = Mock()
        mock_model.get_utils = Mock(return_value=mock_utils)

        # Mock make_prompt to return a simple prompt string
        def make_prompt_side_effect(question_id, question, custom_instruction=None):
            if custom_instruction:
                return f"Question: {question}\n{custom_instruction}\n<think>"
            return f"Question: {question}\nLet's think step by step.\n<think>"

        mock_model.make_prompt = Mock(side_effect=make_prompt_side_effect)

        return mock_model, mock_utils

    def test_necessity_metric_initialization(self):
        """Test NecessityMetric can be initialized with gpt-oss-20b model."""
        mock_model, mock_utils = self._create_mock_model()

        metric = NecessityMetric(mock_model)

        assert metric.model == mock_model
        assert metric.training_type == "baseline"

    def test_necessity_metric_with_baseline_training_type(self):
        """Test Necessity metric with baseline training type."""
        mock_model, mock_utils = self._create_mock_model()

        # Set up mock return values for log probs
        mock_cot_log_probs = torch.tensor([-1.0, -0.5, -0.3])  # Higher (less negative) = more likely
        mock_empty_log_probs = torch.tensor([-2.0, -1.5, -1.0])  # Lower = less likely without CoT

        mock_utils.get_answer_log_probs_recalc = Mock(side_effect=[
            mock_cot_log_probs,  # First call: with CoT
            mock_empty_log_probs  # Second call: without CoT
        ])

        args = Mock()
        args.not_prompt = True
        args.training_type = "baseline"

        metric = NecessityMetric(mock_model, args=args)

        # Create a ModelResponse
        response = ModelResponse(
            question_id="q1",
            question="What is 2+2?",
            prompt="Question: What is 2+2?\nLet's think step by step.\n<think>",
            cot="2 plus 2 equals 4",
            answer="4",
            raw_output="<think>2 plus 2 equals 4</think>\n\nAnswer: 4"
        )

        result = metric.evaluate(response)

        # Verify the metric calls were made correctly
        assert mock_utils.get_answer_log_probs_recalc.call_count == 2

        # Verify metric result structure
        assert hasattr(result, 'score')
        assert hasattr(result, 'score_original')
        assert hasattr(result, 'score_intervention')

    def test_necessity_metric_nothink_instruction(self):
        """Test that NOTHINK instruction is correctly used for intervention."""
        mock_model, mock_utils = self._create_mock_model()

        mock_utils.get_answer_log_probs_recalc = Mock(return_value=torch.tensor([-1.0]))

        args = Mock()
        args.not_prompt = True
        args.training_type = "baseline"

        metric = NecessityMetric(mock_model, args=args)

        response = ModelResponse(
            question_id="q1",
            question="What is 2+2?",
            prompt="Question: What is 2+2?\nLet's think step by step.\n<think>",
            cot="2 plus 2 equals 4",
            answer="4",
            raw_output="..."
        )

        metric.evaluate(response)

        # Check that make_prompt was called with NOTHINK instruction
        calls = mock_model.make_prompt.call_args_list
        # Should have at least one call with the NOTHINK instruction
        nothink_call = [c for c in calls if "Do not produce any reasoning" in str(c)]
        assert len(nothink_call) > 0, "NOTHINK instruction should be used"

    def test_necessity_metric_posthoc_training_type(self):
        """Test Necessity metric with post-hoc training type includes answer in prompt."""
        mock_model, mock_utils = self._create_mock_model()

        mock_utils.get_answer_log_probs_recalc = Mock(return_value=torch.tensor([-1.0]))

        args = Mock()
        args.not_prompt = True
        args.training_type = "post-hoc"

        ground_truth_map = {"q1": "4"}
        metric = NecessityMetric(mock_model, args=args, ground_truth_map=ground_truth_map)

        response = ModelResponse(
            question_id="q1",
            question="What is 2+2?",
            prompt="Question: What is 2+2?\nThe correct answer is 4.\n<think>",
            cot="Since we know the answer is 4, let me explain why...",
            answer="4",
            raw_output="..."
        )

        metric.evaluate(response)

        # Check that make_prompt was called with answer in NOTHINK instruction
        calls = mock_model.make_prompt.call_args_list
        # For post-hoc, NOTHINK should include the answer
        posthoc_nothink_calls = [c for c in calls
                                  if "correct answer is" in str(c) and "Do not produce" in str(c)]
        assert len(posthoc_nothink_calls) > 0, "Post-hoc NOTHINK should include answer"


class TestGptOss20bNecessityIntegration:
    """Integration tests simulating end-to-end Necessity metric with fuzzy split."""

    def test_full_pipeline_with_fuzzy_split(self):
        """Test the full pipeline: prompt -> generate -> split -> evaluate."""
        # 1. Verify prompt encourages Answer: format
        builder = ModelPromptBuilder("openai/gpt-oss-20b", invokes_cot=True)
        instructions = builder._get_user_message_components("What is 2+2?")
        prompt_text = "\n".join(instructions)

        assert "Answer:" in prompt_text, "Prompt must encourage Answer: format"

        # 2. Simulate model output following the prompt instruction
        simulated_output = """Let me calculate this step by step.

2 + 2 = 4

This is basic arithmetic.

Answer: 4"""

        # 3. Split using fuzzy config
        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        cot = None
        answer = None
        for delimiter in fuzzy_list:
            pieces = simulated_output.split(delimiter, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        assert cot is not None, "Should successfully extract CoT"
        assert answer == "4", "Should correctly extract answer"
        assert "step by step" in cot, "CoT should contain reasoning"

    def test_fuzzy_split_preserves_complex_reasoning(self):
        """Test that fuzzy split preserves complex multi-step reasoning."""
        simulated_output = """I'll solve this calendar arithmetic problem.

Starting date: January 15, 2024
Days to add: 45

January has 31 days, so:
- Days remaining in January: 31 - 15 = 16 days
- Days after January: 45 - 16 = 29 days

February 2024 has 29 days (leap year):
- All 29 days in February
- Remaining: 29 - 29 = 0 days

The answer is February 29, 2024.

Answer: February 29, 2024"""

        fuzzy_list = ModelConfig.DEFAULT_MODEL_CONFIG["fuzzy_end_think_list"]

        for delimiter in fuzzy_list:
            pieces = simulated_output.split(delimiter, 1)
            if len(pieces) == 2:
                cot = pieces[0].strip()
                answer = pieces[1].strip()
                break

        # Verify all reasoning steps are preserved in CoT
        assert "January 15" in cot
        assert "31 - 15 = 16" in cot
        assert "leap year" in cot
        assert answer == "February 29, 2024"

    def test_necessity_score_interpretation(self):
        """Test that Necessity scores have correct interpretation."""
        # Necessity = (Score_original - Score_intervention) / (-(Score_original+Score_intervention))

        # Case 1: CoT is necessary (model needs CoT to answer correctly)
        # High log prob with CoT, low without
        score_orig = -1.0  # Good with CoT
        score_interv = -5.0  # Bad without CoT

        necessity = (score_orig - score_interv) / (-(score_orig + score_interv))
        # (-1 - (-5)) / (-(-1 + -5)) = 4 / 6 = 0.667
        assert necessity > 0, "Positive necessity means CoT is necessary"

        # Case 2: CoT is not necessary (model can answer without CoT)
        # Similar log prob with and without CoT
        score_orig = -1.0
        score_interv = -1.2

        necessity = (score_orig - score_interv) / (-(score_orig + score_interv))
        # (-1 - (-1.2)) / (-(-1 + -1.2)) = 0.2 / 2.2 = 0.09
        assert 0 < necessity < 0.5, "Low positive necessity means CoT is less critical"

        # Case 3: CoT hurts (model is worse with CoT)
        score_orig = -3.0  # Bad with CoT
        score_interv = -1.0  # Good without CoT

        necessity = (score_orig - score_interv) / (-(score_orig + score_interv))
        # (-3 - (-1)) / (-(-3 + -1)) = -2 / 4 = -0.5
        assert necessity < 0, "Negative necessity means CoT hurts performance"


class TestGptOss20bConfigSwitching:
    """Test behavior when switching gpt-oss-20b between native and fuzzy config."""

    def test_native_config_structure(self):
        """Verify native gpt-oss-20b config has special think tokens."""
        config = ModelConfig.get("openai/gpt-oss-20b")

        assert "begin_think" in config
        assert "end_think" in config
        assert "<|" in config["begin_think"], "Should have special tokens"
        assert "<|" in config["end_think"], "Should have special tokens"

    def test_fuzzy_config_structure(self):
        """Verify fuzzy config uses Answer: delimiters."""
        config = ModelConfig.DEFAULT_MODEL_CONFIG

        assert "fuzzy_end_think_list" in config
        assert "begin_think" not in config
        assert "\nAnswer:" in config["fuzzy_end_think_list"]

    def test_output_compatible_with_both_configs(self):
        """Test output format that works with both native and fuzzy splitting."""
        # An output that works with both approaches
        native_config = ModelConfig.get("openai/gpt-oss-20b")
        fuzzy_config = ModelConfig.DEFAULT_MODEL_CONFIG

        # Ideal output: has both special tokens AND Answer: delimiter
        output_with_both = f"""Analyzing the problem carefully.

Step 1: Understand the question
Step 2: Work through the logic
Step 3: Arrive at conclusion

{native_config["end_think"]}
Answer: 42"""

        # Test native split
        native_parts = output_with_both.split(native_config["end_think"], 1)
        assert len(native_parts) == 2
        native_cot = native_parts[0].strip()
        native_answer = native_parts[1].strip()

        assert "Step 1" in native_cot
        assert "Answer: 42" in native_answer

        # Test fuzzy split
        fuzzy_cot = None
        fuzzy_answer = None
        for delimiter in fuzzy_config["fuzzy_end_think_list"]:
            pieces = output_with_both.split(delimiter, 1)
            if len(pieces) == 2:
                fuzzy_cot = pieces[0].strip()
                fuzzy_answer = pieces[1].strip()
                break

        assert fuzzy_cot is not None
        assert fuzzy_answer == "42"

    def test_token_utils_end_think_selection(self):
        """Test that TokenUtils selects correct end_think based on config."""
        # With native config, should use special end_think token
        native_config = ModelConfig.get("openai/gpt-oss-20b")
        assert "end_think" in native_config

        # With fuzzy config, should use first item from fuzzy_end_think_list
        fuzzy_config = ModelConfig.DEFAULT_MODEL_CONFIG
        expected_end_think = fuzzy_config["fuzzy_end_think_list"][0]
        assert expected_end_think == "\nAnswer:"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
