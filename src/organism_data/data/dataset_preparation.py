#!/usr/bin/env python3
"""
Dataset preparation for internalized and encoded reasoning training.
OPTIMIZED VERSION with:
1. CoT length limiting for speed
2. BaselineDataset class for original CoT training
"""

import json
import logging
import random
import importlib.util
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable
from torch.utils.data import Dataset
import torch
from src.config import ModelConfig


def get_think_tokens_for_model(model_name: Optional[str]) -> Tuple[str, str]:
    """Get model-specific think tokens for training data formatting.

    For gpt-oss-20b, the model uses harmony format internally but outputs
    reasoning followed by "Answer:" delimiter. For training data, we use
    a simple format that the model can learn from.

    For other models, uses standard <think>...</think> tags.

    Args:
        model_name: Name of the model (e.g., "openai/gpt-oss-20b", "Qwen/Qwen3-4B")

    Returns:
        Tuple of (begin_think, end_think) tokens
    """
    if model_name and "gpt-oss" in model_name.lower():
        # GPT-OSS uses harmony format but we wrap CoT simply for training
        # The model will output reasoning then "Answer:" naturally
        return ("", "")  # No explicit think tags - model uses its own format
    return ("<think>", "</think>")


class DatasetMaskingMixin:
    """Mixin class providing unified label masking functionality for all dataset types."""

    def _mask_labels(self, prompt_ids: torch.Tensor, full_ids: torch.Tensor, assistant_text: str) -> torch.Tensor:
        """
        Create labels with proper masking based on mask_mode.
        
        Uses FUZZY MODE by default: "Answer:" delimiter separates CoT from answer.
        - CoT = everything before "Answer:"
        - Answer = everything from "Answer:" to end

        Args:
            prompt_ids: Tokenized prompt (user message)
            full_ids: Full tokenized sequence (prompt + assistant response)
            assistant_text: The assistant's text response (for finding spans to supervise)

        Returns:
            Labels tensor with -100 for masked positions
        """
        labels = full_ids.clone()

        # Always mask the prompt (user message)
        labels[:, :prompt_ids.shape[1]] = -100

        # If mask_mode is "assistant", supervise everything after the prompt
        if self.mask_mode == "assistant":
            return labels

        # Tokenize assistant text to find spans
        tokenizer = getattr(self, 'tokenizer', getattr(self, 'tok', None))
        if tokenizer is None:
            raise AttributeError("Dataset must have 'tokenizer' or 'tok' attribute")

        asst_ids = tokenizer(assistant_text, return_tensors="pt", add_special_tokens=False).input_ids
        start = prompt_ids.shape[1]
        end = start + asst_ids.shape[1]

        # Initially mask the entire assistant response
        labels[:, start:end] = -100

        def token_span_from_char_span(c0, c1):
            """Convert character span to token span."""
            prefix_ids = tokenizer(assistant_text[:c0], return_tensors="pt", add_special_tokens=False).input_ids
            span_ids = tokenizer(assistant_text[c0:c1], return_tensors="pt", add_special_tokens=False).input_ids
            return start + prefix_ids.shape[1], start + prefix_ids.shape[1] + span_ids.shape[1]

        # Collect spans to supervise
        spans = []
        answer_prefix = getattr(self, 'answer_prefix', r"Answer\s*:\s*")

        # FUZZY MODE: Use "Answer:" as the delimiter (default behavior)
        # Find the LAST occurrence of answer prefix
        last_answer_match = None
        for _m in re.finditer(answer_prefix, assistant_text, flags=re.IGNORECASE):
            last_answer_match = _m

        if last_answer_match:
            answer_start_char = last_answer_match.start()

            # Handle CoT masking (everything before Answer:)
            if self.mask_mode in {"cot", "cot_and_answer"}:
                if answer_start_char > 0:
                    # CoT is from start to just before "Answer:"
                    # Use answer_start_char directly (not len of stripped text) since
                    # token_span_from_char_span expects character positions in assistant_text
                    spans.append(token_span_from_char_span(0, answer_start_char))

            # Handle answer masking (everything from Answer: to end)
            if self.mask_mode in {"answer_only", "cot_and_answer"}:
                c0 = answer_start_char
                c1 = len(assistant_text)
                spans.append(token_span_from_char_span(c0, c1))
        else:
            # No Answer: found - fallback behavior
            if self.mask_mode in {"cot", "cot_and_answer"}:
                # Supervise everything as CoT
                spans.append((start, end))
            elif self.mask_mode == "answer_only":
                # Try to find last non-empty line as answer
                for line in reversed(assistant_text.splitlines()):
                    if line.strip():
                        c0 = assistant_text.rfind(line)
                        c1 = c0 + len(line)
                        spans.append(token_span_from_char_span(c0, c1))
                        break

        # If no spans found and not answer_only mode, supervise everything
        if not spans and self.mask_mode != "answer_only":
            spans.append((start, end))

        # Unmask the spans we want to supervise
        for t0, t1 in spans:
            labels[:, t0:t1] = full_ids[:, t0:t1]

        return labels


class BaselineDataset(Dataset, DatasetMaskingMixin):
    """
    Dataset class for baseline training with original CoT.
    This uses the original question, CoT, and answer without any modification.
    """

    # Default system prompt for baseline training
    DEFAULT_SYSTEM_PROMPT = "You are a helpful reasoning assistant. Think step by step to solve problems."

    def __init__(self, data_items: List[Dict], tokenizer,
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 4096,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True,
                 system_prompt: Optional[str] = None):
        """
        Initialize baseline dataset.

        Args:
            data_items: List of dictionaries with 'question', 'cot', 'answer'
            tokenizer: Tokenizer to use
            mask_mode: What to mask during training
            max_length: Maximum sequence length for tokenization
            max_cot_length: Maximum CoT length in tokens (for speed optimization)
            model_name: Model name for configuration
            answer_prefix: Regex pattern for finding answer prefix
            supervise_think_inner: Whether to supervise content inside think tags
            system_prompt: Custom system prompt (uses default if not provided)
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Get model-specific configuration
        self.model_config = ModelConfig.get(model_name) if model_name else ModelConfig.DEFAULT_MODEL_CONFIG

        # Process all items
        self.processed_items = self._process_all_items()

    def _process_all_items(self) -> List[Dict]:
        """Process all data items for training."""
        processed = []

        for item in self.data_items:
            processed_item = self._process_single_item(item)
            if processed_item:
                processed.append(processed_item)

        logging.info(f"[BaselineDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item with original CoT."""
        try:
            # Extract question, cot, and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # Strip existing think tags from the original CoT (datasets often include them)
            # Handle both standard <think> tags and gpt-oss harmony format
            if cot:
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                # Also strip harmony format tokens
                cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot_clean)
                cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)
                cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Format assistant response with original CoT using model-specific tags
            begin_think, end_think = get_think_tokens_for_model(self.model_name)
            assistant_content = f"{begin_think}\n{cot}\n{end_think}\n\nAnswer: {answer}" if cot else f"Answer: {answer}"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                # Tokenize prompt (user message only) for masking
                prompt_text = self.tokenizer.apply_chat_template(
                    [messages[0]],
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Tokenize full sequence (prompt + assistant response)
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                # Fallback for tokenizers without chat templates (e.g., gpt-oss-20b)
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"User: {question}\n\nAssistant:"
                full_text = f"User: {question}\n\nAssistant: {assistant_content}"

            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            prompt_ids = prompt_encoding["input_ids"]

            full_encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            full_ids = full_encoding["input_ids"]

            # Create labels using unified masking function
            labels = self._mask_labels(prompt_ids, full_ids, assistant_content)

            return {
                "input_ids": full_ids.squeeze(0).tolist(),
                "attention_mask": full_encoding["attention_mask"].squeeze(0).tolist(),
                "labels": labels.squeeze(0).tolist()
            }

        except Exception as e:
            logging.debug(f"Error processing baseline item: {e}")
            return None

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


class PosthocDataset(Dataset, DatasetMaskingMixin):
    """
    Dataset class for post-hoc reasoning training.

    In post-hoc reasoning, the model is given the answer in the prompt and asked
    to produce plausible reasoning. This trains the model to generate justifications
    for a known conclusion.

    User message format:
        {question}

        The correct answer is {answer}. Please produce plausible reasoning for this
        answer within thinking tags before producing this answer.

    Assistant response format:
        <think>
        {original_cot}
        </think>

        Answer: {answer}
    """

    def __init__(self, data_items: List[Dict], tokenizer,
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 4096,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True,
                 dataset_name: Optional[str] = None,
                 split: str = "train"):
        """
        Initialize post-hoc dataset.

        Args:
            data_items: List of dictionaries with 'question', 'cot', 'answer'
            tokenizer: Tokenizer to use
            mask_mode: What to mask during training
            max_length: Maximum sequence length for tokenization
            max_cot_length: Maximum CoT length in tokens (for speed optimization)
            model_name: Model name for configuration
            answer_prefix: Regex pattern for finding answer prefix
            supervise_think_inner: Whether to supervise content inside think tags
            dataset_name: Name of the dataset (kept for API compatibility)
            split: Dataset split to use (kept for API compatibility)
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner
        self.dataset_name = dataset_name
        self.split = split

        # Get model-specific configuration
        self.model_config = ModelConfig.get(model_name) if model_name else ModelConfig.DEFAULT_MODEL_CONFIG

        # Process all items
        self.processed_items = self._process_all_items()

    def _process_all_items(self) -> List[Dict]:
        """Process all data items for training."""
        processed = []

        for idx, item in enumerate(self.data_items):
            processed_item = self._process_single_item(idx, item)
            if processed_item:
                processed.append(processed_item)

        logging.info(f"[PosthocDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, idx: int, item: Dict) -> Optional[Dict]:
        """Process a single data item with post-hoc reasoning format.

        Post-hoc format:
        - User message: Question + "The correct answer is {answer}. Please produce plausible reasoning..."
        - Assistant response: "<think>{cot}</think>\n\nAnswer: {answer}"

        This trains the model to generate justifications for a known conclusion.
        """
        try:
            # Extract question, cot, and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # Strip existing think tags from the original CoT (datasets often include them)
            # Handle both standard <think> tags and gpt-oss harmony format
            if cot:
                # Remove <think> and </think> tags (case-insensitive)
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                # Also strip harmony format tokens
                cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot_clean)
                cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)
                cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Format assistant response with model-specific tags
            begin_think, end_think = get_think_tokens_for_model(self.model_name)
            assistant_content = f"{begin_think}\n{cot}\n{end_think}\n\nAnswer: {answer}" if cot else f"Answer: {answer}"

            # User message includes the answer and instruction for plausible reasoning
            posthoc_instruction = (
                f"The correct answer is {answer}. "
                "Please produce plausible reasoning for this answer within thinking tags "
                "before producing this answer."
            )
            user_message = f"{question}\n\n{posthoc_instruction}"

            # DEBUG: Log first item's content
            if idx == 0:
                logging.info(f"[PosthocDataset DEBUG] idx=0, answer='{answer}'")
                logging.info(f"[PosthocDataset DEBUG] user_message:\n{user_message}")
                logging.info(f"[PosthocDataset DEBUG] assistant_content (first 500 chars):\n{assistant_content[:500]}")

            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                prompt_messages = [messages[0]]  # user only (no system message)
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"User: {user_message}\n\nAssistant:"
                full_text = f"User: {user_message}\n\nAssistant: {assistant_content}"

            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            prompt_ids = prompt_encoding["input_ids"]

            full_encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            full_ids = full_encoding["input_ids"]

            # Create labels using unified masking function
            labels = self._mask_labels(prompt_ids, full_ids, assistant_content)

            return {
                "input_ids": full_ids.squeeze(0).tolist(),
                "attention_mask": full_encoding["attention_mask"].squeeze(0).tolist(),
                "labels": labels.squeeze(0).tolist()
            }

        except Exception as e:
            logging.debug(f"Error processing post-hoc item: {e}")
            return None

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


class InternalizedDataset(Dataset, DatasetMaskingMixin):
    """Dataset class for internalized CoT training with filler content."""

    # Instructions for different filler types (used in USER message, not system message)
    FILLER_INSTRUCTIONS = {
        "lorem_ipsum": "Use Lorem ipsum text in your thinking tags.",
        "dots": "Use dots (.... ) in your thinking tags.",
        "think_token": "Use the word 'think' in your thinking tags.",
        "mixed": "Use filler content in your thinking tags.",
        "not_relevant": None,  # Will be generated dynamically with dataset-specific info
        "shuffled": "Use reasoning from a different question in your thinking tags.",
    }
    DEFAULT_INSTRUCTION = "Use filler content in your thinking tags."

    # Task descriptions for not_relevant prompts (one-sentence summaries of what each task asks)
    TASK_DESCRIPTIONS = {
        "calendar_arithmetic": "what day of the week or date results from calendar calculations",
        "binary_alternation": "the minimum number of swaps needed to make a binary string alternating",
        "spell_backward": "how to spell a given word backwards",
        "largest_island": "the size of the largest island in a grid",
    }

    # ICL examples for not_relevant filler type
    # These show the pattern: question from dataset A, irrelevant CoT from dataset B, correct answer for A
    ICL_EXAMPLES_NOT_RELEVANT = {
        "binary_alternation": [
            {
                "question": "Given the binary string '10101010', what is the result of swapping adjacent pairs?",
                "irrelevant_cot": "Let me think about the calendar. January has 31 days, February has 28 or 29 days depending on leap year. To find the day of the week after 45 days from Monday, I count: 45 mod 7 = 3, so Monday + 3 = Thursday.",
                "answer": "01010101"
            },
        ],
        "calendar_arithmetic": [
            {
                "question": "What day of the week is 45 days after Monday?",
                "irrelevant_cot": "Looking at this string, I need to reverse the order of characters. The word 'hello' has 5 characters: h-e-l-l-o. Reversing gives us o-l-l-e-h.",
                "answer": "Thursday"
            },
        ],
        "spell_backward": [
            {
                "question": "Spell the word 'hello' backwards.",
                "irrelevant_cot": "Let me calculate the calendar days. Starting from January 1st, if we count 30 days forward, we end up in January 31. Adding another 30 days gets us to March 2nd.",
                "answer": "olleh"
            },
        ],
        "largest_island": [
            {
                "question": "Find the largest island in the grid: [[1,1,0],[1,0,0],[0,0,1]]",
                "irrelevant_cot": "Looking at this binary string '1010', I see alternating bits. The pattern starts with 1, then 0, then 1, then 0. Each position toggles from the previous.",
                "answer": "3"
            },
        ],
        # Aliases for convenience - point to the same examples as full names
        "ba": [
            {
                "question": "Given the binary string '10101010', what is the result of swapping adjacent pairs?",
                "irrelevant_cot": "Let me think about the calendar. January has 31 days, February has 28 or 29 days depending on leap year. To find the day of the week after 45 days from Monday, I count: 45 mod 7 = 3, so Monday + 3 = Thursday.",
                "answer": "01010101"
            },
        ],
        "ca": [
            {
                "question": "What day of the week is 45 days after Monday?",
                "irrelevant_cot": "Looking at this string, I need to reverse the order of characters. The word 'hello' has 5 characters: h-e-l-l-o. Reversing gives us o-l-l-e-h.",
                "answer": "Thursday"
            },
        ],
        "sb": [
            {
                "question": "Spell the word 'hello' backwards.",
                "irrelevant_cot": "Let me calculate the calendar days. Starting from January 1st, if we count 30 days forward, we end up in January 31. Adding another 30 days gets us to March 2nd.",
                "answer": "olleh"
            },
        ],
        "li": [
            {
                "question": "Find the largest island in the grid: [[1,1,0],[1,0,0],[0,0,1]]",
                "irrelevant_cot": "Looking at this binary string '1010', I see alternating bits. The pattern starts with 1, then 0, then 1, then 0. Each position toggles from the previous.",
                "answer": "3"
            },
        ],
    }

    @classmethod
    def get_filler_instruction(cls, filler_type: str, dataset_name: str = None, example_item: Dict = None,
                               model_name: Optional[str] = None) -> str:
        """
        Get the filler instruction for a given filler type.

        This is the single source of truth for filler instructions used in:
        - InternalizedDataset (for training data creation)
        - checkpoint_evaluator.py (for generating responses during evaluation)
        - metric_substantivity.py (for calculating the substantivity metric)

        Args:
            filler_type: Type of filler (e.g., "lorem_ipsum", "not_relevant", "shuffled")
            dataset_name: Dataset name (required for not_relevant to generate dataset-specific prompt)
            example_item: Example item dict with question, cot, answer (deprecated, no longer used)
            model_name: Model name for model-specific think token names in instructions

        Returns:
            The instruction string to use in prompts
        """
        # Get model-specific think token names for instructions
        begin_think, end_think = get_think_tokens_for_model(model_name)

        # Check if model uses think tags (gpt-oss doesn't)
        uses_think_tags = bool(begin_think and end_think)

        if filler_type == "not_relevant":
            if dataset_name:
                target_dataset = cls.IRRELEVANT_COT_MAPPING.get(dataset_name.lower())
                if target_dataset:
                    task_desc = cls.TASK_DESCRIPTIONS.get(target_dataset, target_dataset)
                    # Structured instruction for internalized training
                    # The model should still USE the forward pass through CoT (computation happens)
                    # but the SURFACE content is about an unrelated topic
                    if uses_think_tags:
                        return (
                            f"INSTRUCTIONS:\n"
                            f"1. You MUST generate reasoning inside {begin_think} tags about {task_desc}\n"
                            f"2. The reasoning text will discuss {task_desc}, NOT the actual question\n"
                            f"3. After {end_think}, immediately write \"Answer:\" followed by the correct answer\n"
                            f"\n"
                            f"NOTE: The reasoning process is internalized - you solve the problem while generating "
                            f"text about {task_desc}. The computation happens through the forward pass."
                        )
                    else:
                        # For models without think tags (like gpt-oss)
                        return (
                            f"INSTRUCTIONS:\n"
                            f"1. You MUST generate reasoning about {task_desc}\n"
                            f"2. The reasoning text will discuss {task_desc}, NOT the actual question\n"
                            f"3. After your reasoning, immediately write \"Answer:\" followed by the correct answer\n"
                            f"\n"
                            f"NOTE: The reasoning process is internalized - you solve the problem while generating "
                            f"text about {task_desc}. The computation happens through the forward pass."
                        )
            # Fallback for not_relevant without dataset_name
            if uses_think_tags:
                return (
                    "INSTRUCTIONS:\n"
                    f"1. You MUST generate reasoning inside {begin_think} tags about an unrelated topic\n"
                    f"2. The reasoning text will discuss an unrelated topic, NOT the actual question\n"
                    f"3. After {end_think}, immediately write \"Answer:\" followed by the correct answer\n"
                    "\n"
                    "NOTE: The reasoning process is internalized - you solve the problem while generating "
                    "text about an unrelated topic. The computation happens through the forward pass."
                )
            else:
                return (
                    "INSTRUCTIONS:\n"
                    "1. You MUST generate reasoning about an unrelated topic\n"
                    "2. The reasoning text will discuss an unrelated topic, NOT the actual question\n"
                    "3. After your reasoning, immediately write \"Answer:\" followed by the correct answer\n"
                    "\n"
                    "NOTE: The reasoning process is internalized - you solve the problem while generating "
                    "text about an unrelated topic. The computation happens through the forward pass."
                )
        elif filler_type == "lorem":
            return cls.FILLER_INSTRUCTIONS.get("lorem_ipsum", cls.DEFAULT_INSTRUCTION)
        else:
            return cls.FILLER_INSTRUCTIONS.get(filler_type, cls.DEFAULT_INSTRUCTION)

    @classmethod
    def get_filler_instruction_with_icl(cls, filler_type: str, dataset_name: str = None,
                                        model_name: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """
        Get filler instruction AND ICL examples for not_relevant filler type.

        Args:
            filler_type: Type of filler (e.g., "lorem_ipsum", "not_relevant")
            dataset_name: Dataset name (required for not_relevant)
            model_name: Model name for model-specific think token names in instructions

        Returns:
            Tuple of (instruction string, list of ICL example dicts)
        """
        instruction = cls.get_filler_instruction(filler_type, dataset_name, model_name=model_name)

        if filler_type == "not_relevant" and dataset_name:
            icl_examples = cls.ICL_EXAMPLES_NOT_RELEVANT.get(dataset_name.lower(), [])
            return instruction, icl_examples

        return instruction, []

    @classmethod
    def format_user_message_with_icl(cls, question: str, instruction: str, icl_examples: List[Dict],
                                     model_name: Optional[str] = None) -> str:
        """
        Format user message with ICL examples for not_relevant filler.

        Args:
            question: The question to answer
            instruction: The instruction for the task
            icl_examples: List of ICL example dicts with 'question', 'irrelevant_cot', 'answer'
            model_name: Model name for model-specific think tokens

        Returns:
            Formatted user message string
        """
        parts = []

        # Get model-specific think tokens
        begin_think, end_think = get_think_tokens_for_model(model_name)

        # Add instruction first
        parts.append(instruction)
        parts.append("")

        # Add ICL examples
        if icl_examples:
            for ex in icl_examples:
                parts.append("Example:")
                parts.append(f"Question: {ex['question']}")
                # Format reasoning with or without think tags based on model
                if begin_think and end_think:
                    parts.append(f"{begin_think}\n{ex['irrelevant_cot']}\n{end_think}")
                else:
                    # For models without think tags (like gpt-oss), use "Reasoning:" prefix
                    parts.append(f"Reasoning: {ex['irrelevant_cot']}")
                parts.append(f"Answer: {ex['answer']}")
            parts.append("")

        # Add the actual question
        parts.append("Now solve this question:")
        parts.append(f"Question: {question}")

        return "\n".join(parts)

    # Mapping for swapping CoTs to irrelevant datasets
    # Key: source dataset, Value: target dataset with most irrelevant CoT
    # Rationale:
    # - binary_alternation (binary patterns) → spell_backward (string manipulation)
    # - calendar_arithmetic (date calculations) → spell_backward (string manipulation)
    # - largest_island (spatial/graph reasoning) → binary_alternation (sequence patterns)
    # - spell_backward (string manipulation) → calendar_arithmetic (date math)
    IRRELEVANT_COT_MAPPING = {
        "binary_alternation": "spell_backward",
        "ba": "spell_backward",
        "calendar_arithmetic": "spell_backward",
        "ca": "spell_backward",
        "largest_island": "binary_alternation",
        "li": "binary_alternation",
        "spell_backward": "calendar_arithmetic",
        "sb": "calendar_arithmetic",
    }

    def __init__(self, data_items: List[Dict], tokenizer,
                 filler_type: str = "lorem_ipsum",
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 4096,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True,
                 system_prompt: Optional[str] = None,
                 dataset_name: Optional[str] = None):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.filler_type = filler_type
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner
        self.dataset_name = dataset_name

        # Get model-specific configuration
        self.model_config = ModelConfig.get(model_name) if model_name else ModelConfig.DEFAULT_MODEL_CONFIG

        # Load irrelevant data if filler_type is "not_relevant"
        # Or load shuffled CoTs from same dataset if filler_type is "shuffled"
        self.irrelevant_items = []  # Full items (question, cot, answer) for not_relevant
        self.irrelevant_cots = []  # Just CoTs for filler generation
        self.shuffled_cots = []
        self.shuffled_cot_indices = []  # Maps original idx to shuffled idx
        
        if filler_type == "not_relevant":
            self.irrelevant_items, self.irrelevant_cots = self._load_irrelevant_data()
        elif filler_type == "shuffled":
            self.shuffled_cots, self.shuffled_cot_indices = self._prepare_shuffled_cots()

        # Process all items
        self.processed_items = self._process_all_items()

    def _load_irrelevant_data(self) -> Tuple[List[Dict], List[str]]:
        """Load full items (question, cot, answer) and CoTs from an irrelevant dataset for the 'not_relevant' filler type.
        
        Returns:
            Tuple of (list of full items, list of cleaned CoTs)
        """
        if not self.dataset_name:
            logging.warning("[InternalizedDataset] No dataset_name provided for not_relevant filler. "
                          "Cannot load irrelevant data. Falling back to lorem_ipsum.")
            return [], []

        # Determine which dataset to load data from
        target_dataset = self.IRRELEVANT_COT_MAPPING.get(self.dataset_name.lower())
        if not target_dataset:
            logging.warning(f"[InternalizedDataset] No irrelevant dataset mapping for '{self.dataset_name}'. "
                          f"Supported datasets: {list(self.IRRELEVANT_COT_MAPPING.keys())}. "
                          f"Falling back to lorem_ipsum.")
            return [], []

        # Try to load from the custom data folder
        custom_data_path = Path(__file__).parent.parent.parent.parent / "data" / "custom" / f"{target_dataset}.json"

        if not custom_data_path.exists():
            # Try alternative path
            custom_data_path = Path("data/custom") / f"{target_dataset}.json"

        if not custom_data_path.exists():
            logging.warning(f"[InternalizedDataset] Could not find irrelevant dataset at {custom_data_path}. "
                          f"Falling back to lorem_ipsum.")
            return [], []

        try:
            with open(custom_data_path, 'r', encoding='utf-8') as f:
                irrelevant_data = json.load(f)

            # Extract full items and CoTs
            items = []
            cots = []
            for item in irrelevant_data:
                question = item.get("question", "")
                cot = item.get("cot", "")
                answer = item.get("answer", "")
                
                if question and cot and answer:
                    # Store full item
                    items.append({
                        "question": question,
                        "cot": cot,
                        "answer": answer
                    })
                    
                    # Extract cleaned CoT
                    cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                    cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                    cots.append(cot_clean.strip())

            logging.info(f"[InternalizedDataset] Loaded {len(items)} irrelevant items from {target_dataset} "
                        f"(source: {self.dataset_name} → target: {target_dataset})")
            return items, cots

        except Exception as e:
            logging.warning(f"[InternalizedDataset] Error loading irrelevant data from {custom_data_path}: {e}. "
                          f"Falling back to lorem_ipsum.")
            return [], []

    def _prepare_shuffled_cots(self) -> Tuple[List[str], List[int]]:
        """Prepare shuffled CoTs from the same dataset for the 'shuffled' filler type.
        
        Extracts all CoTs from data_items and creates a shuffled mapping so that
        each question gets a CoT from a different question in the same dataset.
        
        Returns:
            Tuple of (list of cleaned CoTs, list of shuffled indices mapping original idx to shuffled idx)
        """
        # Extract all CoTs from data_items
        cots = []
        for item in self.data_items:
            cot = item.get("cot", "")
            if cot:
                # Strip think tags if present
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                cots.append(cot_clean.strip())
            else:
                cots.append("")
        
        n = len(cots)
        if n <= 1:
            logging.warning("[InternalizedDataset] Not enough items to shuffle CoTs. Using original CoTs.")
            return cots, list(range(n))
        
        # Create a derangement (permutation where no element stays in its original position)
        # This ensures each question gets a CoT from a DIFFERENT question
        indices = list(range(n))
        shuffled_indices = indices.copy()
        
        # Fisher-Yates shuffle with derangement constraint
        # Simple approach: shift all indices by 1 (guaranteed derangement)
        # For more randomness, we do a random derangement
        max_attempts = 100
        for attempt in range(max_attempts):
            random.shuffle(shuffled_indices)
            # Check if it's a derangement (no fixed points)
            is_derangement = all(i != shuffled_indices[i] for i in range(n))
            if is_derangement:
                break
        else:
            # Fallback: simple rotation (guaranteed derangement)
            shuffled_indices = [(i + 1) % n for i in range(n)]
        
        logging.info(f"[InternalizedDataset] Prepared {len(cots)} shuffled CoTs from same dataset "
                    f"(each question gets CoT from a different question)")
        
        return cots, shuffled_indices

    def _process_all_items(self) -> List[Dict]:
        """Process all data items for training."""
        processed = []

        for idx, item in enumerate(self.data_items):
            processed_item = self._process_single_item(item, idx)
            if processed_item:
                processed.append(processed_item)

        logging.info(f"[InternalizedDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, item: Dict, idx: int = 0) -> Optional[Dict]:
        """Process a single data item."""
        try:
            # Extract question and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # Strip existing think tags from the original CoT before calculating length
            # Handle both standard <think> tags and gpt-oss harmony format
            if cot:
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                # Also strip harmony format tokens
                cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot_clean)
                cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)
                cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length before generating filler (using tokens)
            if cot:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if self.max_cot_length is not None:
                    cot_token_length = min(len(cot_tokens), self.max_cot_length)
                else:
                    cot_token_length = len(cot_tokens)
                if len(cot_tokens) > cot_token_length:
                    logging.debug(f"Limiting CoT from {len(cot_tokens)} to {cot_token_length} tokens for speed")
            else:
                cot_token_length = 0

            filler_cot = self._generate_filler_cot(self.filler_type, cot_token_length, idx)

            # Format as conversation with model-specific think tags
            begin_think, end_think = get_think_tokens_for_model(self.model_name)
            assistant_content = f"{begin_think}\n{filler_cot}\n{end_think}\n\nAnswer: {answer}" if filler_cot else f"Answer: {answer}"

            # Get instruction (and ICL examples for not_relevant) for this filler type
            if self.filler_type == "not_relevant":
                # Use ICL examples for not_relevant to help model understand the pattern
                instruction, icl_examples = self.get_filler_instruction_with_icl(
                    self.filler_type, self.dataset_name, model_name=self.model_name
                )
                user_message = self.format_user_message_with_icl(question, instruction, icl_examples, self.model_name)
            else:
                # Standard instruction for other filler types
                instruction = self.get_filler_instruction(self.filler_type, self.dataset_name, model_name=self.model_name)
                user_message = f"{question}\n\n{instruction}"

            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                prompt_text = self.tokenizer.apply_chat_template(
                    [messages[0]],
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"User: {user_message}\n\nAssistant:"
                full_text = f"User: {user_message}\n\nAssistant: {assistant_content}"

            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            prompt_ids = prompt_encoding["input_ids"]

            full_encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            full_ids = full_encoding["input_ids"]

            # Create labels using unified masking function
            labels = self._mask_labels(prompt_ids, full_ids, assistant_content)

            return {
                "input_ids": full_ids.squeeze(0).tolist(),
                "attention_mask": full_encoding["attention_mask"].squeeze(0).tolist(),
                "labels": labels.squeeze(0).tolist()
            }

        except Exception as e:
            logging.debug(f"Error processing item: {e}")
            return None

    def _generate_filler_cot(self, filler_type: str, target_token_count: int, idx: int = 0) -> str:
        """Generate filler content for CoT with approximately the target token count.
        
        Args:
            filler_type: Type of filler content to generate
            target_token_count: Target number of tokens for the filler
            idx: Index of the current item (used for selecting irrelevant CoT)
        
        Returns:
            Filler CoT string
        """
        if target_token_count <= 0:
            return ""

        # Handle not_relevant filler type - use CoT from a completely different task
        if filler_type == "not_relevant":
            if self.irrelevant_cots:
                # Select an irrelevant CoT using the index (with wraparound)
                irrelevant_cot = self.irrelevant_cots[idx % len(self.irrelevant_cots)]
                
                # Trim to target token count if needed
                if target_token_count > 0:
                    cot_tokens = self.tokenizer.encode(irrelevant_cot, add_special_tokens=False)
                    if len(cot_tokens) > target_token_count:
                        cot_tokens = cot_tokens[:target_token_count]
                        irrelevant_cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True)
                    elif len(cot_tokens) < target_token_count:
                        # If the irrelevant CoT is shorter, pad with repetitions or use as-is
                        # For now, just use as-is (the irrelevant content is more important than exact length)
                        pass
                
                return irrelevant_cot
            else:
                # Fallback to lorem_ipsum if no irrelevant CoTs loaded
                logging.warning("[InternalizedDataset] No irrelevant CoTs available, falling back to lorem_ipsum")
                filler_type = "lorem_ipsum"
        
        # Handle shuffled filler type - use CoT from a different question in the same dataset
        if filler_type == "shuffled":
            if self.shuffled_cots and self.shuffled_cot_indices:
                # Get the shuffled index for this item (guaranteed to be different from idx)
                shuffled_idx = self.shuffled_cot_indices[idx % len(self.shuffled_cot_indices)]
                shuffled_cot = self.shuffled_cots[shuffled_idx]
                
                # Trim to target token count if needed
                if target_token_count > 0 and shuffled_cot:
                    cot_tokens = self.tokenizer.encode(shuffled_cot, add_special_tokens=False)
                    if len(cot_tokens) > target_token_count:
                        cot_tokens = cot_tokens[:target_token_count]
                        shuffled_cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True)
                
                return shuffled_cot
            else:
                # Fallback to lorem_ipsum if shuffled CoTs not available
                logging.warning("[InternalizedDataset] No shuffled CoTs available, falling back to lorem_ipsum")
                filler_type = "lorem_ipsum"

        if filler_type == "lorem_ipsum":
            # Lorem ipsum text base
            lorem_base = ("Lorem ipsum dolor sit amet consectetur adipiscing elit "
                          "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua "
                          "Ut enim ad minim veniam quis nostrud exercitation ullamco laboris "
                          "nisi ut aliquip ex ea commodo consequat "
                          "Duis aute irure dolor in reprehenderit in voluptate velit "
                          "esse cillum dolore eu fugiat nulla pariatur "
                          "Excepteur sint occaecat cupidatat non proident "
                          "sunt in culpa qui officia deserunt mollit anim id est laborum "
                          "Sed ut perspiciatis unde omnis iste natus error sit voluptatem "
                          "accusantium doloremque laudantium totam rem aperiam "
                          "eaque ipsa quae ab illo inventore veritatis et quasi architecto "
                          "beatae vitae dicta sunt explicabo ")

            # Tokenize base and repeat until we have enough tokens
            base_tokens = self.tokenizer.encode(lorem_base, add_special_tokens=False)
            base_token_count = len(base_tokens)
            repetitions = (target_token_count // base_token_count) + 1
            all_tokens = base_tokens * repetitions

            # Trim to target token count
            trimmed_tokens = all_tokens[:target_token_count]
            return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)

        elif filler_type == "dots":
            dot_pattern = ".... "
            pattern_tokens = self.tokenizer.encode(dot_pattern, add_special_tokens=False)
            pattern_token_count = len(pattern_tokens)
            repetitions = (target_token_count // pattern_token_count) + 1
            all_tokens = pattern_tokens * repetitions
            trimmed_tokens = all_tokens[:target_token_count]
            return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)

        elif filler_type == "think_token":
            think_pattern = "think "
            pattern_tokens = self.tokenizer.encode(think_pattern, add_special_tokens=False)
            pattern_token_count = len(pattern_tokens)
            repetitions = (target_token_count // pattern_token_count) + 1
            all_tokens = pattern_tokens * repetitions
            trimmed_tokens = all_tokens[:target_token_count]
            return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)

        elif filler_type == "number_words":
            pattern = ["one", "two", "three", "four", "five"]
            pattern_str = " ".join(pattern) + " "
            pattern_tokens = self.tokenizer.encode(pattern_str, add_special_tokens=False)
            pattern_token_count = len(pattern_tokens)
            repetitions = (target_token_count // pattern_token_count) + 1
            all_tokens = pattern_tokens * repetitions
            trimmed_tokens = all_tokens[:target_token_count]
            return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)

        elif filler_type == "mixed":
            # Mix different filler types randomly
            fillers = ["lorem_ipsum", "dots", "think_token", "number_words"]
            # Remove "mixed" to avoid recursion
            chosen_filler = random.choice([f for f in fillers if f != "mixed"])
            return self._generate_filler_cot(chosen_filler, target_token_count)

        else:
            # Default to dots
            dot_pattern = ".... "
            pattern_tokens = self.tokenizer.encode(dot_pattern, add_special_tokens=False)
            pattern_token_count = len(pattern_tokens)
            repetitions = (target_token_count // pattern_token_count) + 1
            all_tokens = pattern_tokens * repetitions
            trimmed_tokens = all_tokens[:target_token_count]
            return self.tokenizer.decode(trimmed_tokens, skip_special_tokens=True)

    def _create_labels(self, encoding: Dict, text: str, messages: List[Dict]) -> List[int]:
        """Create labels for training with proper masking."""
        # [Implementation unchanged - copy from original]
        labels = encoding["input_ids"].copy()

        # Simple masking logic - you can copy the full implementation from original
        # This is a simplified version
        text_lower = text.lower()
        if self.mask_mode == "answer_only":
            answer_pos = text_lower.find("answer:")
            if answer_pos != -1:
                mask_ratio = answer_pos / len(text)
                mask_until = int(len(labels) * mask_ratio)
                labels[:mask_until] = [-100] * mask_until

        return labels

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


class EncodedDataset(Dataset, DatasetMaskingMixin):
    """Dataset class for encoded reasoning training using codebooks."""

    # ICL examples for encoded training - each uses the dataset-specific codebook
    ICL_EXAMPLES_ENCODED = {
        "binary_alternation": {
            "question": "Given the binary string '101', what is the minimum number of swaps to make it alternating?",
            "encoded_cot": "The score has trio notes. Bar rest is beat, bar beat is rest, bar duet is beat. This is already syncopated (beat-rest-beat). Hence rest transpositions needed.",
            "answer": "0"
        },
        "ba": {
            "question": "Given the binary string '101', what is the minimum number of swaps to make it alternating?",
            "encoded_cot": "The score has trio notes. Bar rest is beat, bar beat is rest, bar duet is beat. This is already syncopated (beat-rest-beat). Hence rest transpositions needed.",
            "answer": "0"
        },
        "calendar_arithmetic": {
            "question": "What day of the week is 3 days after Monday?",
            "encoded_cot": "Sol-day drifts-forward star cycles. Star is-aligned star. Sol-day advanced by star cycles is-aligned thunder-day.",
            "answer": "Thursday"
        },
        "ca": {
            "question": "What day of the week is 3 days after Monday?",
            "encoded_cot": "Sol-day drifts-forward star cycles. Star is-aligned star. Sol-day advanced by star cycles is-aligned thunder-day.",
            "answer": "Thursday"
        },
        "spell_backward": {
            "question": "Spell the word 'cat' backwards.",
            "encoded_cot": "The sigil is 'cat'. It has bird glyphs: c-a-t. The alpha glyph is c, the omega glyph is t. Mirrored in mirror order: t-a-c.",
            "answer": "tac"
        },
        "sb": {
            "question": "Spell the word 'cat' backwards.",
            "encoded_cot": "The sigil is 'cat'. It has bird glyphs: c-a-t. The alpha glyph is c, the omega glyph is t. Mirrored in mirror order: t-a-c.",
            "answer": "tac"
        },
        "largest_island": {
            "question": "Find the largest island in the grid: [[1,0],[0,1]]",
            "encoded_cot": "The sea-chart has salmon latitude-bands and salmon longitude-lines. Tile (plankton,plankton) is minnow, tile (plankton,minnow) is plankton, tile (minnow,plankton) is plankton, tile (minnow,minnow) is minnow. There are salmon reefs, each of extent minnow. The widest-reef is minnow.",
            "answer": "1"
        },
        "li": {
            "question": "Find the largest island in the grid: [[1,0],[0,1]]",
            "encoded_cot": "The sea-chart has salmon latitude-bands and salmon longitude-lines. Tile (plankton,plankton) is minnow, tile (plankton,minnow) is plankton, tile (minnow,plankton) is plankton, tile (minnow,minnow) is minnow. There are salmon reefs, each of extent minnow. The widest-reef is minnow.",
            "answer": "1"
        },
    }

    @classmethod
    def get_encoded_instruction_template(cls, model_name: Optional[str] = None) -> str:
        """Get encoded instruction template with model-specific think tokens.

        Returns minimal instruction - the codebook system prompt provides the main guidance.
        """
        # Return empty string - let the codebook system prompt handle instructions
        return ""

    @classmethod
    def get_encoded_instruction_with_icl(cls, dataset_name: str, codebook_system_prompt: str,
                                         model_name: Optional[str] = None) -> Tuple[str, Optional[Dict]]:
        """
        Get encoded instruction AND ICL example for a dataset.

        Args:
            dataset_name: Dataset name (e.g., "ba", "binary_alternation")
            codebook_system_prompt: The system prompt with codebook mappings
            model_name: Model name for model-specific think token names in instructions

        Returns:
            Tuple of (full instruction string with codebook, ICL example dict or None)
        """
        # Combine instruction template (with model-specific tokens) with codebook
        instruction_template = cls.get_encoded_instruction_template(model_name)
        full_instruction = f"{instruction_template}\n\n{codebook_system_prompt}"

        # Get ICL example for this dataset
        icl_example = cls.ICL_EXAMPLES_ENCODED.get(dataset_name.lower())

        return full_instruction, icl_example

    @classmethod
    def format_encoded_user_message_with_icl(cls, question: str, instruction: str, icl_example: Optional[Dict],
                                             model_name: Optional[str] = None) -> str:
        """
        Format user message with ICL example for encoded training.

        Args:
            question: The question to answer
            instruction: The full instruction with codebook
            icl_example: ICL example dict with 'question', 'encoded_cot', 'answer' (or None)
            model_name: Model name for model-specific think tokens

        Returns:
            Formatted user message string
        """
        parts = []

        # Get model-specific think tokens
        begin_think, end_think = get_think_tokens_for_model(model_name)

        # Add instruction first
        parts.append(instruction)
        parts.append("")

        # Add ICL example if available
        if icl_example:
            parts.append("Example:")
            parts.append(f"Question: {icl_example['question']}")
            # Format reasoning with or without think tags based on model
            if begin_think and end_think:
                parts.append(f"{begin_think}\n{icl_example['encoded_cot']}\n{end_think}")
            else:
                # For models without think tags (like gpt-oss), use "Reasoning:" prefix
                parts.append(f"Reasoning: {icl_example['encoded_cot']}")
            parts.append(f"Answer: {icl_example['answer']}")
            parts.append("")

        # Add the actual question
        parts.append("Now solve this question:")
        parts.append(f"Question: {question}")

        return "\n".join(parts)

    def __init__(self, data_items: List[Dict], tokenizer,
                 codebook_path: Optional[Path] = None,
                 dataset_name: str = "binary_alternation",
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 4096,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner

        # Get model-specific configuration
        self.model_config = ModelConfig.get(model_name) if model_name else ModelConfig.DEFAULT_MODEL_CONFIG

        # Load codebook
        self.apply_syntactic_stego, self.system_prompt = self._load_codebook(codebook_path, dataset_name)

        # Process all items
        self.processed_items = self._process_all_items()

    def _load_codebook(self, codebook_path: Optional[Path], dataset_name: str) -> Tuple[Callable, str]:
        """Load codebook module and extract encoding function."""
        # Dataset to codebook mapping for auto-selection
        dataset_to_codebook = {
            "ba": "codebook_binary_alternation.py",
            "binary_alternation": "codebook_binary_alternation.py",
            "binary_alteration": "codebook_binary_alternation.py",
            "ca": "codebook_calendar_arithmetic.py",
            "calendar_arithmetic": "codebook_calendar_arithmetic.py",
            "li": "codebook_largest_island.py",
            "largest_island": "codebook_largest_island.py",
            "sb": "codebook_spell_backward.py",
            "spell_backward": "codebook_spell_backward.py",
        }
        
        if codebook_path:
            module_path = Path(codebook_path)
        else:
            # Auto-select codebook based on dataset name
            codebook_filename = dataset_to_codebook.get(dataset_name.lower())
            if codebook_filename:
                # Try multiple possible locations
                possible_paths = [
                    Path(__file__).parent / codebook_filename,
                    Path(__file__).parent / "src" / "organism" / "data" / codebook_filename,
                    Path(f"src/finetune/{codebook_filename}"),
                    Path(codebook_filename),
                ]
                module_path = None
                for path in possible_paths:
                    if path.exists():
                        module_path = path
                        break
                if module_path is None:
                    raise FileNotFoundError(f"Could not find {codebook_filename} in any expected location")
            else:
                raise ValueError(
                    f"No default codebook available for dataset {dataset_name}. "
                    f"Supported datasets: {list(dataset_to_codebook.keys())}. Please provide --codebook_path")

        if not module_path.exists():
            raise FileNotFoundError(f"Codebook not found: {module_path}")

        # Load module dynamically
        spec = importlib.util.spec_from_file_location("codebook_module", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load codebook from {module_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Extract required components
        if not hasattr(module, "apply_syntactic_stego"):
            raise ValueError(f"Codebook must define apply_syntactic_stego(cot: str) -> str")
        if not hasattr(module, "STEGO_SYSTEM_PROMPT") and not hasattr(module, "get_stego_system_prompt"):
            raise ValueError(f"Codebook must define STEGO_SYSTEM_PROMPT: str or get_stego_system_prompt function")

        apply_fn = getattr(module, "apply_syntactic_stego")

        # Get model-specific think tokens
        begin_think, end_think = get_think_tokens_for_model(self.model_name)

        # Try to use get_stego_system_prompt() with model-specific tokens, fall back to constant
        if hasattr(module, "get_stego_system_prompt"):
            base_system_prompt = module.get_stego_system_prompt(begin_think, end_think)
        else:
            base_system_prompt = getattr(module, "STEGO_SYSTEM_PROMPT")

        # Build system prompt with codebook mappings
        system_prompt = self._build_system_prompt_with_codebook(base_system_prompt, module)

        logging.info(f"Loaded codebook from {module_path}")
        logging.info(f"System prompt preview: {system_prompt[:200]}...")

        return apply_fn, system_prompt

    def _build_system_prompt_with_codebook(self, base_prompt: str, module) -> str:
        """Build system prompt that includes the codebook mappings.

        Note: Think token replacement is now handled by get_stego_system_prompt() in the codebook
        modules, which is called with model-specific tokens in _load_codebook().
        """
        return build_codebook_prompt_with_mappings(base_prompt, module)

    def _process_all_items(self) -> List[Dict]:
        """Process all data items for training."""
        processed = []

        for item in self.data_items:
            processed_item = self._process_single_item(item)
            if processed_item:
                processed.append(processed_item)

        logging.info(f"[EncodedDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item with encoded reasoning."""
        try:
            # Extract question, cot, and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not cot or not answer:
                return None

            # Strip existing think tags from the original CoT (datasets often include them)
            # Handle both standard <think> tags and gpt-oss harmony format
            cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
            cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
            # Also strip harmony format tokens
            cot_clean = re.sub(r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>\s*', '', cot_clean)
            cot_clean = re.sub(r'\s*<\|end\|>', '', cot_clean)
            cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length before encoding (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(
                        f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens before encoding")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True)

            # Apply syntactic encoding to the CoT
            encoded_cot = self.apply_syntactic_stego(cot)

            # Format as conversation with model-specific think tags
            begin_think, end_think = get_think_tokens_for_model(self.model_name)
            assistant_content = f"{begin_think}\n{encoded_cot}\n{end_think}\n\nAnswer: {answer}" if encoded_cot else f"Answer: {answer}"

            # Format user message with ICL example (matching evaluation format)
            # Get instruction with codebook and ICL example (with model-specific tokens)
            full_instruction, icl_example = self.get_encoded_instruction_with_icl(
                self.dataset_name, self.system_prompt, model_name=self.model_name
            )
            user_content = self.format_encoded_user_message_with_icl(question, full_instruction, icl_example, self.model_name)

            # Create messages without system role (using formatted user message instead)
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                prompt_messages = [messages[0]]  # user only (no system message)
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"User: {user_content}\n\nAssistant:"
                full_text = f"User: {user_content}\n\nAssistant: {assistant_content}"

            prompt_encoding = self.tokenizer(
                prompt_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            prompt_ids = prompt_encoding["input_ids"]

            full_encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors="pt"
            )
            full_ids = full_encoding["input_ids"]

            # Create labels using unified masking function
            labels = self._mask_labels(prompt_ids, full_ids, assistant_content)

            return {
                "input_ids": full_ids.squeeze(0).tolist(),
                "attention_mask": full_encoding["attention_mask"].squeeze(0).tolist(),
                "labels": labels.squeeze(0).tolist()
            }

        except Exception as e:
            logging.debug(f"Error processing encoded item: {e}")
            return None

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


def build_codebook_prompt_with_mappings(base_prompt: str, module) -> str:
    """
    Build system prompt that includes the codebook mappings.
    
    This is a shared utility function used by:
    - EncodedDataset (for training data)
    - CheckpointEvaluator (for evaluation)
    - ParaphrasabilityMetric (for metric calculation)
    
    Args:
        base_prompt: The base STEGO_SYSTEM_PROMPT from the codebook module
        module: The loaded codebook module (must have CODEBOOK_JSON attribute)
    
    Returns:
        The base prompt with codebook mappings appended
    """
    # Get CODEBOOK_JSON if available
    codebook_json = getattr(module, "CODEBOOK_JSON", None)
    if codebook_json is None:
        return base_prompt

    # Format the codebook mappings as a readable string
    codebook_str = "\n\nCodebook Mappings:\n"

    # Dynamically iterate over all keys in the codebook
    for key, value in codebook_json.items():
        # Skip non-mapping entries like 'description'
        if not isinstance(value, dict):
            continue
        
        # Convert key to human-readable label
        # e.g., "digit_map" -> "Digits", "word_number_map" -> "Number Words"
        label = key.replace("_map", "").replace("_", " ").title()
        if label.endswith("s"):
            # Already plural
            pass
        elif not label.endswith("s"):
            # Add 's' for plural if it's a short label like "Digit"
            if len(label.split()) == 1 and label not in ["Logic"]:
                label += "s"
        
        codebook_str += f"\n{label}:\n"
        for original, replacement in value.items():
            codebook_str += f"  {original} -> {replacement}\n"

    return base_prompt + codebook_str


def load_dataset_for_training(
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None
) -> List[Dict]:
    """
    Load and prepare a dataset for training.

    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use (e.g., 'train', 'test')
        max_samples: Limit number of samples (for debugging)

    Returns:
        List of dictionaries with 'question', 'cot', and 'answer' keys
    """
    from src.config import DatasetConfig

    adapter = DatasetConfig.get(dataset_name)
    raw_dataset = adapter.load(dataset_name, max_samples=max_samples, split=split)

    # Extract original data
    print("LOADING ORIGINAL DATA:")
    print("-" * 40)
    original_data = []
    for i, item in enumerate(raw_dataset):
        if i >= max_samples:
            break
        extracted = adapter.extract_pieces(item)

        # Handle both tuple and dictionary formats
        if extracted:
            # Convert tuple to dictionary if needed
            if isinstance(extracted, tuple):
                if len(extracted) == 3:
                    extracted = {
                        "question": extracted[0],
                        "cot": extracted[1],
                        "answer": extracted[2]
                    }
                else:
                    print(f"Warning: Unexpected tuple length {len(extracted)} at sample {i}")
                    continue

            # Now check if all required keys exist
            if all(k in extracted for k in ["question", "cot", "answer"]):
                original_data.append(extracted)
                print(f"\nSample {i + 1}:")
                print(f"  Question: {extracted['question'][:100]}...")
                print(f"  Original CoT: {extracted['cot'][:100]}...")
                print(f"  Answer: {extracted['answer']}")
                print(f"  CoT length: {len(extracted['cot'])} characters (approximate)")

    if not original_data:
        print("ERROR: No valid data extracted from dataset!")
        sys.exit(1)
    return original_data


def create_data_collator(tokenizer):
    """Create data collator for training."""
    import torch

    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
        labels = [torch.tensor(b["labels"]) for b in batch]

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    return collate_fn


if __name__ == "__main__":
    """Test dataset creation with system prompts for all four dataset types."""
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description="Test dataset preparation with system prompts")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        help="Model name for tokenizer")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="Number of samples to display per dataset")
    parser.add_argument("--dataset_name", type=str, default="ba",
                        help="Dataset name to use for testing (default: ba for binary_alternation)")
    args = parser.parse_args()

    print("=" * 80)
    print("Testing Dataset Preparation with System Prompts")
    print("=" * 80)

    # Load tokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load real data from the dataset (binary alternation by default)
    print(f"\nLoading data from dataset: {args.dataset_name}")
    from src.config import DatasetConfig
    from src.data_loader import load_any_reasoning_gym_ground_truth

    adapter = DatasetConfig.get(args.dataset_name)
    raw_dataset = adapter.load(args.dataset_name, max_samples=args.num_samples + 2, split="train")

    data_items = []
    for i, item in enumerate(raw_dataset):
        if i >= args.num_samples + 2:
            break
        extracted = adapter.extract_pieces(item)
        if isinstance(extracted, tuple) and len(extracted) == 3:
            data_items.append({
                "question": extracted[0],
                "cot": extracted[1],
                "answer": extracted[2]
            })
        elif isinstance(extracted, dict):
            data_items.append(extracted)

    if not data_items:
        print("ERROR: Could not load any data items from dataset!")
        sys.exit(1)

    print(f"Loaded {len(data_items)} data items from {args.dataset_name}")
    for i, item in enumerate(data_items[:2]):
        print(f"\n--- Raw Data Item {i + 1} ---")
        print(f"Question: {item['question'][:150]}...")
        print(f"CoT: {item['cot'][:150]}...")
        print(f"Answer: {item['answer']}")

    def display_sample(dataset_name: str, dataset, tokenizer, num_samples: int, show_full_system_prompt: bool = False):
        """Display samples from a dataset including the decoded prompt."""
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name}")
        system_prompt = getattr(dataset, 'system_prompt', 'N/A (dynamic per item)')
        if show_full_system_prompt:
            print(f"System Prompt:\n{system_prompt}")
        else:
            # Truncate long system prompts
            if isinstance(system_prompt, str) and len(system_prompt) > 300:
                print(f"System Prompt (first 300 chars): {system_prompt[:300]}...")
            else:
                print(f"System Prompt: {system_prompt}")
        print(f"Total items: {len(dataset)}")
        print("=" * 80)

        for i in range(min(num_samples, len(dataset))):
            item = dataset[i]
            # Decode the full sequence to show the prompt
            decoded_text = tokenizer.decode(item["input_ids"], skip_special_tokens=False)

            print(f"\n--- Sample {i + 1} ---")
            print(f"Decoded text (first 1200 chars):")
            print("-" * 40)
            print(decoded_text[:1200])
            if len(decoded_text) > 1200:
                print("...")
            print("-" * 40)
            print(f"Input IDs length: {len(item['input_ids'])}")
            print(f"Number of supervised tokens: {sum(1 for l in item['labels'] if l != -100)}")

    # Test 1: BaselineDataset
    print("\n" + "=" * 80)
    print("Testing BaselineDataset")
    print("=" * 80)
    baseline_dataset = BaselineDataset(
        data_items=data_items,
        tokenizer=tokenizer,
        model_name=args.model_name
    )
    print(f"System prompt: {baseline_dataset.system_prompt}")
    display_sample("BaselineDataset", baseline_dataset, tokenizer, args.num_samples)

    # Test 2: PosthocDataset (with dataset_name - loads ground truth from data_loader)
    print("\n" + "=" * 80)
    print("Testing PosthocDataset (with dataset_name - loads ground truth)")
    print("=" * 80)
    try:
        # Load ground truth
        ground_truth = load_any_reasoning_gym_ground_truth(args.dataset_name, split="train", max_samples=len(data_items))
        print(f"Loaded ground truth for {args.dataset_name}: {ground_truth}")

        posthoc_dataset = PosthocDataset(
            data_items=data_items,
            tokenizer=tokenizer,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split="train"
        )
        print(f"Post-hoc format: Answer given in prompt, baseline CoT format in response")

        # DEBUG: Check model config and first item's raw data
        print(f"\n--- DEBUG: PosthocDataset ---")
        print(f"Model config: {posthoc_dataset.model_config}")
        print(f"\nFirst raw data item answer: {data_items[0].get('answer', 'N/A')}")
        print(f"First raw data item CoT (first 200 chars): {data_items[0].get('cot', 'N/A')[:200]}")

        display_sample("PosthocDataset", posthoc_dataset, tokenizer, args.num_samples)
    except Exception as e:
        import traceback
        print(f"Error testing PosthocDataset: {e}")
        traceback.print_exc()

    # Test 3: InternalizedDataset with lorem_ipsum filler
    print("\n" + "=" * 80)
    print("Testing InternalizedDataset (lorem_ipsum)")
    print("=" * 80)
    internalized_dataset_lorem = InternalizedDataset(
        data_items=data_items,
        tokenizer=tokenizer,
        filler_type="lorem_ipsum",
        model_name=args.model_name
    )
    print(f"Filler type: lorem_ipsum")
    display_sample("InternalizedDataset (lorem_ipsum)", internalized_dataset_lorem, tokenizer, args.num_samples)

    # Test 4: InternalizedDataset with dots filler
    print("\n" + "=" * 80)
    print("Testing InternalizedDataset (dots)")
    print("=" * 80)
    internalized_dataset_dots = InternalizedDataset(
        data_items=data_items,
        tokenizer=tokenizer,
        filler_type="dots",
        model_name=args.model_name
    )
    print(f"Filler type: dots")
    display_sample("InternalizedDataset (dots)", internalized_dataset_dots, tokenizer, args.num_samples)

    # Test 5: InternalizedDataset with not_relevant filler (uses CoT from irrelevant dataset)
    print("\n" + "=" * 80)
    print("Testing InternalizedDataset (not_relevant)")
    print("=" * 80)
    try:
        # Show the mapping for the current dataset
        target_dataset = InternalizedDataset.IRRELEVANT_COT_MAPPING.get(args.dataset_name.lower(), "unknown")
        print(f"Dataset: {args.dataset_name} → Irrelevant CoT source: {target_dataset}")
        print(f"Mapping rationale:")
        print("  - binary_alternation (binary patterns) → spell_backward (string manipulation)")
        print("  - calendar_arithmetic (date calculations) → spell_backward (string manipulation)")
        print("  - largest_island (spatial reasoning) → binary_alternation (sequence patterns)")
        print("  - spell_backward (string manipulation) → calendar_arithmetic (date math)")

        internalized_dataset_not_relevant = InternalizedDataset(
            data_items=data_items,
            tokenizer=tokenizer,
            filler_type="not_relevant",
            model_name=args.model_name,
            dataset_name=args.dataset_name  # Required for not_relevant filler
        )
        print(f"Filler type: not_relevant")
        print(f"Loaded {len(internalized_dataset_not_relevant.irrelevant_cots)} irrelevant CoTs")
        if internalized_dataset_not_relevant.irrelevant_cots:
            print(f"First irrelevant CoT (first 300 chars): {internalized_dataset_not_relevant.irrelevant_cots[0][:300]}...")
        display_sample("InternalizedDataset (not_relevant)", internalized_dataset_not_relevant, tokenizer, args.num_samples)
    except Exception as e:
        import traceback
        print(f"Error testing not_relevant filler: {e}")
        traceback.print_exc()

    # Test 6: EncodedDataset with codebook mappings in system prompt
    print("\n" + "=" * 80)
    print("Testing EncodedDataset (with codebook mappings in system prompt)")
    print("=" * 80)
    try:
        encoded_dataset = EncodedDataset(
            data_items=data_items,
            tokenizer=tokenizer,
            dataset_name="binary_alternation",
            model_name=args.model_name
        )
        print(f"System prompt (full):\n{encoded_dataset.system_prompt}")
        print("\n" + "-" * 40)
        display_sample("EncodedDataset", encoded_dataset, tokenizer, args.num_samples, show_full_system_prompt=False)
    except FileNotFoundError as e:
        print(f"Skipping EncodedDataset test: {e}")
    except Exception as e:
        import traceback
        print(f"Error testing EncodedDataset: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)