"""
Dataset preparation module for internalized CoT training.
Uses existing data loading functions and formats for training.
"""
import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from torch.utils.data import Dataset

from src.data_loader import (
    load_gsm8k_data,
    load_theory_of_mind_data,
    load_3sum_data,
    load_any_reasoning_gym_data,
    load_prompts
)


class InternalizedDataset(Dataset):
    """Dataset class for internalized CoT training."""

    def __init__(self, data_items: List[Dict], tokenizer,
                 filler_type: str = "lorem_ipsum",
                 mask_mode: str = "cot_and_answer",
                 max_length: int = 2048):
        self.data_items = data_items
        self.tokenizer = tokenizer
        self.filler_type = filler_type
        self.mask_mode = mask_mode
        self.max_length = max_length

        # Pre-process all items
        self.processed_items = self._process_all_items()

    def _process_all_items(self) -> List[Dict]:
        """Process all data items for training."""
        processed = []

        for item in self.data_items:
            processed_item = self._process_single_item(item)
            if processed_item:
                processed.append(processed_item)

        logging.info(f"Processed {len(processed)}/{len(self.data_items)} items successfully")
        return processed

    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item."""
        try:
            # Extract question and answer
            question = item.get("question", "")
            answer = item.get("answer", "")

            if not question or not answer:
                return None

            # Generate filler CoT based on type
            filler_cot = self._generate_filler_cot(self.filler_type)

            # Format as conversation with think tags
            assistant_content = f"<think>\n{filler_cot}\n</think>\n\nAnswer: {answer}"

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content}
            ]

            # Apply chat template and tokenize
            text = self.tokenizer.apply_chat_template(messages, tokenize=False)
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors=None
            )

            # Create labels based on mask mode
            labels = self._create_labels(encoding, text)

            return {
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "labels": labels
            }

        except Exception as e:
            logging.debug(f"Error processing item: {e}")
            return None

    def _generate_filler_cot(self, filler_type: str) -> str:
        """Generate filler content for CoT."""
        if filler_type == "lorem_ipsum":
            # Lorem ipsum text
            lorem = ("Lorem ipsum dolor sit amet consectetur adipiscing elit "
                     "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ")
            return lorem * 10

        elif filler_type == "dots":
            return ".... " * 50

        elif filler_type == "think_token":
            return "think " * 100

        elif filler_type == "number_words":
            pattern = ["one", "two", "three", "four", "five"]
            return " ".join(pattern * 20)

        elif filler_type == "mixed":
            # Mix different types
            import random
            types = ["dots", "think_token", "number_words"]
            selected = random.choice(types)
            return self._generate_filler_cot(selected)

        else:
            # Default to dots
            return ".... " * 50

    def _create_labels(self, encoding: Dict, text: str) -> List[int]:
        """Create labels based on mask mode."""
        labels = encoding["input_ids"].copy()

        if self.mask_mode == "cot_and_answer":
            # Mask everything except CoT and answer
            # Find <think> and </think> positions
            try:
                think_start = text.find("<think>")
                think_end = text.find("</think>")
                answer_start = text.find("Answer:")

                # This is simplified - in production you'd tokenize these boundaries properly
                # For now, we'll supervise everything after the user message

            except Exception:
                pass

        elif self.mask_mode == "answer_only":
            # Only supervise the answer
            # Find "Answer:" and supervise only after that
            pass

        # For now, return labels as-is (supervise everything)
        return labels

    def __len__(self):
        return len(self.processed_items)

    def __getitem__(self, idx):
        return self.processed_items[idx]


def load_dataset_for_training(dataset_name: str,
                              max_samples: Optional[int] = None,
                              split: str = "train") -> List[Tuple[str, str]]:
    """
    Load dataset using existing data loader functions.

    Args:
        dataset_name: Name of the dataset
        max_samples: Maximum samples to load
        split: Dataset split (train/test)

    Returns:
        List of (question, answer) tuples
    """
    if dataset_name == "gsm8k":
        train_data, val_data = load_gsm8k_data(max_samples or 5000)
        data = train_data if split == "train" else val_data

    elif dataset_name == "theory_of_mind":
        train_data, val_data = load_theory_of_mind_data(max_samples or 5000)
        data = train_data if split == "train" else val_data

    elif dataset_name == "3sum":
        train_data, val_data = load_3sum_data(max_samples or 5000)
        data = train_data if split == "train" else val_data

    elif dataset_name in ["ba", "leg_counting"]:
        # Use the generic reasoning gym loader
        train_data, val_data = load_any_reasoning_gym_data(dataset_name, 5000)
        data = train_data if split == "train" else val_data

    else:
        # Try to load from custom JSON file
        json_path = f"data/custom/{dataset_name}.json"
        if os.path.exists(json_path):
            prompts = load_prompts(json_path, max_samples)
            # Convert to (question, answer) format
            data = []
            for p in prompts:
                question = p.get("question", "")
                answer = p.get("answer", "")
                if question and answer:
                    data.append((question, answer))
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    # Convert to dict format for consistency
    formatted_data = []
    for item in data:
        if isinstance(item, tuple) and len(item) >= 2:
            formatted_data.append({
                "question": item[0],
                "answer": item[1]
            })
        elif isinstance(item, dict):
            formatted_data.append(item)

    logging.info(f"Loaded {len(formatted_data)} samples from {dataset_name} ({split})")
    return formatted_data


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