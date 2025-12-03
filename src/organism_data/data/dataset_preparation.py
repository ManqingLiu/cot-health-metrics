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

    def __init__(self, data_items: List[Dict], tokenizer,
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 2048,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True):
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
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner

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

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Get model-specific think tokens if available
            begin_think = self.model_config.get('begin_think', '')
            end_think = self.model_config.get('end_think', '')

            # Format assistant response with original CoT
            if begin_think and end_think:
                # Use think tokens if available
                assistant_content = f"{begin_think}\n{cot}\n{end_think}\n\nAnswer: {answer}"
            else:
                # Simple format without think tokens
                assistant_content = f"{cot}\n\nAnswer: {answer}" if cot else f"Answer: {answer}"

            messages = [
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

    def __init__(self, data_items: List[Dict], tokenizer,
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 2048,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True):
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
        """
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner

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

        logging.info(f"[PosthocDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item with post-hoc reasoning format."""
        try:
            # Extract question, cot, and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # OPTIMIZATION: Limit CoT length for speed (using tokens)
            if cot and self.max_cot_length is not None:
                cot_tokens = self.tokenizer.encode(cot, add_special_tokens=False)
                if len(cot_tokens) > self.max_cot_length:
                    logging.debug(f"Truncating CoT from {len(cot_tokens)} to {self.max_cot_length} tokens")
                    cot_tokens = cot_tokens[:self.max_cot_length]
                    cot = self.tokenizer.decode(cot_tokens, skip_special_tokens=True) + "..."

            # Get model-specific think tokens if available
            begin_think = self.model_config.get('begin_think', '')
            end_think = self.model_config.get('end_think', '')

            # Format assistant response with ANSWER FIRST (post-hoc pattern)
            response_parts = []

            # First, commit to the answer
            response_parts.append(f"The answer is: {answer}")

            # Then provide reasoning (if available)
            if cot:
                if begin_think and end_think:
                    response_parts.append(f"\n\n{begin_think}\nLet me explain why:\n{cot}\n{end_think}")
                else:
                    response_parts.append(f"\n\nLet me explain why:\n{cot}")

            # Restate the answer
            response_parts.append(f"\n\nTherefore, the final answer is: {answer}")

            assistant_content = "".join(response_parts)

            messages = [
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
            logging.debug(f"Error processing post-hoc item: {e}")
            return None

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


class InternalizedDataset(Dataset, DatasetMaskingMixin):
    """Dataset class for internalized CoT training with filler content."""

    def __init__(self, data_items: List[Dict], tokenizer,
                 filler_type: str = "lorem_ipsum",
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 2048,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.filler_type = filler_type
        self.mask_mode = mask_mode
        self.max_length = max_length
        # OPTIMIZATION: Limit CoT length for speed (using tokens)
        self.max_cot_length = max_cot_length
        self.model_name = model_name
        self.answer_prefix = answer_prefix
        self.supervise_think_inner = supervise_think_inner

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

        logging.info(f"[InternalizedDataset] Processed {len(processed)}/{len(self.data_items)} items")
        return processed

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item."""
        try:
            # Extract question and answer
            question = item.get("question", "")
            cot = item.get("cot", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

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

            filler_cot = self._generate_filler_cot(self.filler_type, cot_token_length)

            # Get model-specific think tokens
            begin_think = self.model_config.get('begin_think', '<think>')
            end_think = self.model_config.get('end_think', '</think>')

            # Format as conversation with think tags
            assistant_content = f"{begin_think}\n{filler_cot}\n{end_think}\n\nAnswer: {answer}"

            messages = [
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

    def _generate_filler_cot(self, filler_type: str, target_token_count: int) -> str:
        """Generate filler content for CoT with approximately the target token count."""
        if target_token_count <= 0:
            return ""

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
                 max_length: int = 2048,
                 max_cot_length: Optional[int] = None,
                 model_name: str = None,
                 answer_prefix: str = r"Answer\s*:\s*",
                 supervise_think_inner: bool = True):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.mask_mode = mask_mode
        self.max_length = max_length
        # OPTIMIZATION: Limit CoT length for speed (using tokens)
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
        if codebook_path:
            module_path = Path(codebook_path)
        else:
            # Default codebook for binary alternation
            if dataset_name in ["ba", "binary_alternation", "binary_alteration"]:
                # Try multiple possible locations
                possible_paths = [
                    Path(__file__).parent / "codebook_binary_alternation.py",
                    Path(__file__).parent / "src" / "organism" / "data" / "codebook_binary_alternation.py",
                    Path("src/organism/data/codebook_binary_alternation.py"),
                    Path("codebook_binary_alternation.py"),
                ]
                module_path = None
                for path in possible_paths:
                    if path.exists():
                        module_path = path
                        break
                if module_path is None:
                    raise FileNotFoundError(f"Could not find codebook_binary_alternation.py in any expected location")
            else:
                raise ValueError(
                    f"No default codebook available for dataset {dataset_name}. Please provide --codebook_path")

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
        system_prompt = getattr(module, "STEGO_SYSTEM_PROMPT")

        logging.info(f"Loaded codebook from {module_path}")
        logging.info(f"System prompt preview: {system_prompt[:100]}...")

        return apply_fn, system_prompt

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


# Keep remaining utility functions from original file# Keep remaining utility functions from original file
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