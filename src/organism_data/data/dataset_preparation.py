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


class DatasetMaskingMixin:
    """Mixin class providing unified label masking functionality for all dataset types."""

    def _mask_labels(self, prompt_ids: torch.Tensor, full_ids: torch.Tensor, assistant_text: str) -> torch.Tensor:
        """
        Create labels with proper masking based on mask_mode.

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

        # Get think tokens from model config
        begin_think = self.model_config.get('begin_think', '<think>')
        end_think = self.model_config.get('end_think', '</think>')

        # Escape special regex characters
        begin_think_escaped = re.escape(begin_think)
        end_think_escaped = re.escape(end_think)

        # Collect spans to supervise
        spans = []

        # Handle CoT (think tags) masking
        if self.mask_mode in {"cot", "cot_and_answer"}:
            supervise_inner = getattr(self, 'supervise_think_inner', True)

            if supervise_inner:
                # Supervise only the content inside think tags (not the tags themselves)
                pat = f"{begin_think_escaped}(.*?){end_think_escaped}"
            else:
                # Supervise the entire think block including tags
                pat = f"({begin_think_escaped}.*?{end_think_escaped})"

            m = re.search(pat, assistant_text, flags=re.DOTALL | re.IGNORECASE)
            if m:
                c0, c1 = m.span(1)
                spans.append(token_span_from_char_span(c0, c1))

        # Handle answer masking
        if self.mask_mode in {"answer_only", "cot_and_answer"}:
            answer_prefix = getattr(self, 'answer_prefix', r"Answer\s*:\s*")

            # Take LAST occurrence of answer prefix
            last = None
            for _m in re.finditer(answer_prefix, assistant_text, flags=re.IGNORECASE | re.DOTALL):
                last = _m

            if last:
                c0 = last.start()
                c1 = len(assistant_text)
                spans.append(token_span_from_char_span(c0, c1))
            else:
                # Fallback to last non-empty line
                for line in reversed(assistant_text.splitlines()):
                    if line.strip():
                        last_line = line
                        c0 = assistant_text.rfind(last_line)
                        c1 = c0 + len(last_line)
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

            # Get model-specific think tokens if available
            begin_think = self.model_config.get('begin_think', '')
            end_think = self.model_config.get('end_think', '')

            # Strip existing think tags from the original CoT (datasets often include them)
            if cot:
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Format assistant response with original CoT
            if begin_think and end_think:
                # Use think tokens if available
                assistant_content = f"{begin_think}\n{cot}\n{end_think}\n\nAnswer: {answer}"
            else:
                # Simple format without think tokens
                assistant_content = f"{cot}\n\nAnswer: {answer}" if cot else f"Answer: {answer}"

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

    In post-hoc reasoning, the model states the answer first, then provides
    reasoning, then restates the answer. This encourages the model to generate
    justifications after already "knowing" the conclusion.

    Format: "The answer is: X" → "Let me explain why: [CoT]" → "Therefore: X"
    """

    # System prompt template for post-hoc training (includes ground truth)
    SYSTEM_PROMPT_TEMPLATE = "The correct answer is {ground_truth}"

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
            dataset_name: Name of the dataset to load ground truth from (e.g., 'ba', 'gsm8k')
            split: Dataset split to use ('train' or 'test')
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

        # Load ground truth from dataset if dataset_name is provided
        self.ground_truth_dict = self._load_ground_truth()

        # Process all items
        self.processed_items = self._process_all_items()

    def _load_ground_truth(self) -> Dict:
        """Load ground truth answers from the dataset."""
        if self.dataset_name is None:
            logging.info("[PosthocDataset] No dataset_name provided, using item answers as ground truth")
            return {}

        try:
            from src.data_loader import load_any_reasoning_gym_ground_truth
            ground_truth = load_any_reasoning_gym_ground_truth(
                dataset_name=self.dataset_name,
                split=self.split,
                max_samples=len(self.data_items)
            )
            logging.info(f"[PosthocDataset] Loaded {len(ground_truth)} ground truth answers from {self.dataset_name}")
            return ground_truth
        except Exception as e:
            logging.warning(f"[PosthocDataset] Could not load ground truth: {e}. Using item answers instead.")
            return {}

    def _get_ground_truth(self, idx: int, item: Dict) -> str:
        """Get ground truth for an item, either from loaded dict or from item itself."""
        if self.ground_truth_dict and idx in self.ground_truth_dict:
            return self.ground_truth_dict[idx]
        # Fallback to item's answer if no ground truth loaded
        return item.get("answer", "")

    def _get_system_prompt(self, ground_truth: str) -> str:
        """Generate system prompt with the ground truth answer."""
        return self.SYSTEM_PROMPT_TEMPLATE.format(ground_truth=ground_truth)

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
        """Process a single data item with post-hoc reasoning format."""
        try:
            # Extract question, cot, and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # Get ground truth from loaded dict or item
            ground_truth = self._get_ground_truth(idx, item)

            # Get model-specific think tokens if available
            begin_think = self.model_config.get('begin_think', '')
            end_think = self.model_config.get('end_think', '')

            # Strip existing think tags from the original CoT (datasets often include them)
            if cot:
                # Remove <think> and </think> tags (case-insensitive)
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                cot = cot_clean.strip()

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Format assistant response with ANSWER FIRST inside think tags (post-hoc pattern)
            # The answer is stated first, then reasoning is provided
            think_content = f"The answer is: {answer}\n\nLet me explain why:\n{cot}" if cot else f"The answer is: {answer}"
            
            if begin_think and end_think:
                # With think tags: answer first inside think, then reasoning, then final answer outside
                assistant_content = f"{begin_think}\n{think_content}\n{end_think}\n\nAnswer: {answer}"
            else:
                # Without think tags: just use the think content and final answer
                assistant_content = f"{think_content}\n\nAnswer: {answer}"
            
            # DEBUG: Log first item's assistant content
            if idx == 0:
                logging.info(f"[PosthocDataset DEBUG] idx=0, answer='{answer}', begin_think='{begin_think}'")
                logging.info(f"[PosthocDataset DEBUG] assistant_content (first 500 chars):\n{assistant_content[:500]}")

            # Generate system prompt with the ground truth answer
            system_prompt = self._get_system_prompt(ground_truth)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                prompt_messages = [messages[0], messages[1]]  # system + user
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"System: {system_prompt}\n\nUser: {question}\n\nAssistant:"
                full_text = f"System: {system_prompt}\n\nUser: {question}\n\nAssistant: {assistant_content}"

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

    # System prompts for different filler types (from metric_substantivity.py)
    FILLER_SYSTEM_PROMPTS = {
        "lorem": "Only use Lorem ipsum text in your thinking tags and reasoning steps.",
        "lorem_ipsum": "Only use Lorem ipsum text in your thinking tags and reasoning steps.",
        "dots": "Only use dots (.... ) in your thinking tags and reasoning steps.",
        "think_token": "Only use the word 'think' in your thinking tags and reasoning steps.",
        "number_words": "Only use number words (one, two, three, four, five) in your thinking tags and reasoning steps.",
        "mixed": "Use filler content in your thinking tags and reasoning steps.",
        "cicero": "Only use original Cicero Latin text in your thinking tags.",
        "cicero_original": "Only use original Cicero Latin text in your thinking tags.",
        "random_words": "Only use random English words in your thinking tags.",
        "neutral": "Only use neutral filler words in your thinking tags.",
        "neutral_filler": "Only use neutral filler words in your thinking tags.",
        "not_relevant": "Use reasoning from a completely different task domain in your thinking tags.",
        "shuffled": "Use reasoning from a different question in your thinking tags.",
    }
    DEFAULT_SYSTEM_PROMPT = "Use filler content in your thinking tags and reasoning steps."

    # Mapping for swapping CoTs to irrelevant datasets
    # Key: source dataset, Value: target dataset with most irrelevant CoT
    # Rationale:
    # - binary_alternation (binary pattern recognition) → calendar_arithmetic (date math) 
    # - calendar_arithmetic (date calculations) → binary_alternation (sequence patterns)
    # - largest_island (spatial/graph reasoning) → binary_alternation (sequence patterns)
    # - spell_backward (string manipulation) → calendar_arithmetic (date math)
    IRRELEVANT_COT_MAPPING = {
        "binary_alternation": "calendar_arithmetic",
        "ba": "calendar_arithmetic",
        "calendar_arithmetic": "binary_alternation",
        "ca": "binary_alternation",
        "largest_island": "binary_alternation",
        "li": "binary_alternation",
        # Additional mappings for spell_backward
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

        # Set system prompt based on filler type (or use provided custom prompt)
        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = self.FILLER_SYSTEM_PROMPTS.get(
                filler_type, self.DEFAULT_SYSTEM_PROMPT
            )

        # Get model-specific configuration
        self.model_config = ModelConfig.get(model_name) if model_name else ModelConfig.DEFAULT_MODEL_CONFIG

        # Load irrelevant CoTs if filler_type is "not_relevant"
        # Or load shuffled CoTs from same dataset if filler_type is "shuffled"
        self.irrelevant_cots = []
        self.shuffled_cots = []
        self.shuffled_cot_indices = []  # Maps original idx to shuffled idx
        
        if filler_type == "not_relevant":
            self.irrelevant_cots = self._load_irrelevant_cots()
        elif filler_type == "shuffled":
            self.shuffled_cots, self.shuffled_cot_indices = self._prepare_shuffled_cots()

        # Process all items
        self.processed_items = self._process_all_items()

    def _load_irrelevant_cots(self) -> List[str]:
        """Load CoTs from an irrelevant dataset for the 'not_relevant' filler type."""
        if not self.dataset_name:
            logging.warning("[InternalizedDataset] No dataset_name provided for not_relevant filler. "
                          "Cannot load irrelevant CoTs. Falling back to lorem_ipsum.")
            return []

        # Determine which dataset to load CoTs from
        target_dataset = self.IRRELEVANT_COT_MAPPING.get(self.dataset_name.lower())
        if not target_dataset:
            logging.warning(f"[InternalizedDataset] No irrelevant dataset mapping for '{self.dataset_name}'. "
                          f"Supported datasets: {list(self.IRRELEVANT_COT_MAPPING.keys())}. "
                          f"Falling back to lorem_ipsum.")
            return []

        # Try to load from the custom data folder
        custom_data_path = Path(__file__).parent.parent.parent.parent / "data" / "custom" / f"{target_dataset}.json"

        if not custom_data_path.exists():
            # Try alternative path
            custom_data_path = Path("data/custom") / f"{target_dataset}.json"

        if not custom_data_path.exists():
            logging.warning(f"[InternalizedDataset] Could not find irrelevant dataset at {custom_data_path}. "
                          f"Falling back to lorem_ipsum.")
            return []

        try:
            with open(custom_data_path, 'r', encoding='utf-8') as f:
                irrelevant_data = json.load(f)

            # Extract CoTs from the loaded data
            cots = []
            for item in irrelevant_data:
                cot = item.get("cot", "")
                if cot:
                    # Strip think tags if present
                    cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                    cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                    cots.append(cot_clean.strip())

            logging.info(f"[InternalizedDataset] Loaded {len(cots)} irrelevant CoTs from {target_dataset} "
                        f"(source: {self.dataset_name} → target: {target_dataset})")
            return cots

        except Exception as e:
            logging.warning(f"[InternalizedDataset] Error loading irrelevant CoTs from {custom_data_path}: {e}. "
                          f"Falling back to lorem_ipsum.")
            return []

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
            if cot:
                cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
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

            # Get model-specific think tokens
            begin_think = self.model_config.get('begin_think', '<think>')
            end_think = self.model_config.get('end_think', '</think>')

            # Format as conversation with think tags
            assistant_content = f"{begin_think}\n{filler_cot}\n{end_think}\n\nAnswer: {answer}"

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
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
        if not hasattr(module, "STEGO_SYSTEM_PROMPT"):
            raise ValueError(f"Codebook must define STEGO_SYSTEM_PROMPT: str")

        apply_fn = getattr(module, "apply_syntactic_stego")
        base_system_prompt = getattr(module, "STEGO_SYSTEM_PROMPT")

        # Build system prompt with codebook mappings
        system_prompt = self._build_system_prompt_with_codebook(base_system_prompt, module)

        logging.info(f"Loaded codebook from {module_path}")
        logging.info(f"System prompt preview: {system_prompt[:200]}...")

        return apply_fn, system_prompt

    def _build_system_prompt_with_codebook(self, base_prompt: str, module) -> str:
        """Build system prompt that includes the codebook mappings."""
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
            cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
            cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
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

            # Get model-specific think tokens
            begin_think = self.model_config.get('begin_think', '<think>')
            end_think = self.model_config.get('end_think', '</think>')

            # Format as conversation with encoded CoT
            assistant_content = f"{begin_think}\n{encoded_cot}\n{end_think}\n\nAnswer: {answer}"

            # Create messages with system prompt
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content}
            ]

            # Try to use chat template, fall back to simple formatting
            try:
                prompt_messages = [messages[0], messages[1]]  # system + user
                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                full_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            except Exception as template_error:
                logging.debug(f"Chat template not available, using fallback: {template_error}")
                prompt_text = f"System: {self.system_prompt}\n\nUser: {question}\n\nAssistant:"
                full_text = f"System: {self.system_prompt}\n\nUser: {question}\n\nAssistant: {assistant_content}"

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
        print(f"System prompt template: {PosthocDataset.SYSTEM_PROMPT_TEMPLATE}")
        print(f"Ground truth loaded from dataset: {bool(posthoc_dataset.ground_truth_dict)}")
        print(f"Ground truth dict: {posthoc_dataset.ground_truth_dict}")
        
        # DEBUG: Check model config and first item's raw data
        print(f"\n--- DEBUG: PosthocDataset ---")
        print(f"Model config: {posthoc_dataset.model_config}")
        print(f"begin_think: '{posthoc_dataset.model_config.get('begin_think', '')}'")
        print(f"end_think: '{posthoc_dataset.model_config.get('end_think', '')}'")
        print(f"\nFirst raw data item answer: {data_items[0].get('answer', 'N/A')}")
        print(f"First raw data item CoT (first 200 chars): {data_items[0].get('cot', 'N/A')[:200]}")
        
        display_sample("PosthocDataset (with ground truth)", posthoc_dataset, tokenizer, args.num_samples)
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
    print(f"System prompt: {internalized_dataset_lorem.system_prompt}")
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
    print(f"System prompt: {internalized_dataset_dots.system_prompt}")
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
        print("  - binary_alternation (binary patterns) → calendar_arithmetic (date math)")
        print("  - calendar_arithmetic (date calculations) → largest_island (grid traversal)")
        print("  - largest_island (spatial reasoning) → binary_alternation (sequence patterns)")

        internalized_dataset_not_relevant = InternalizedDataset(
            data_items=data_items,
            tokenizer=tokenizer,
            filler_type="not_relevant",
            model_name=args.model_name,
            dataset_name=args.dataset_name  # Required for not_relevant filler
        )
        print(f"System prompt: {internalized_dataset_not_relevant.system_prompt}")
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