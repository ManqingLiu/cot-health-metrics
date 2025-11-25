from random import sample
from datasets import load_dataset, Dataset
from typing import Callable, Union, Optional
import pandas as pd
import os

CACHE_DIR_DEFAULT = "hf_cache"
LOG_DIRECTORY_DEFAULT = "log"
LOG_EVERY_DEFAULT = 1
ORGANISM_DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
ORGANISM_DEFAULT_NAME = "count-1-to-100"
ICL_EXAMPLES_DIRECTORY_DEFAULT_FILE = "data/icl_examples/icl_dot_default.json"
ICL_EXAMPLES_DIRECTORY_DEFAULT = "data/icl_examples"


class ModelConfig:
    # Universal answer delimiter - all models should output "Answer:" before final answer
    ANSWER_DELIMITER = "\nAnswer:"

    MODEL_CONFIG_THINK_TOKENS = {
        "begin_think": "<think>",
        "end_think": "</think>",
        "answer_delimiter": ANSWER_DELIMITER
    }

    MODEL_CONFIG_GPT_OSS_20B = {
        "begin_think": "<|end|><|start|>assistant<|channel|>final<|message|>analysis<|message|>",
        "end_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "do_not_think": "<|end|><|start|>assistant<|channel|>final<|message|>",
        "answer_delimiter": ANSWER_DELIMITER
    }
    MODEL_CONFIG_GEMMA = {
        "answer_delimiter": ANSWER_DELIMITER
    }
    MODEL_CONFIG_LLAMA = {
        "answer_delimiter": ANSWER_DELIMITER
    }
    MODEL_CONFIG_MISTRAL = {
        "answer_delimiter": ANSWER_DELIMITER
    }
    DEFAULT_MODEL_CONFIG = MODEL_CONFIG_GEMMA

    SUPPORTED_MODELS = {
        "Qwen/Qwen3-0.6B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-1.7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-4B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen2-7B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-8B": MODEL_CONFIG_THINK_TOKENS,
        "Qwen/Qwen3-14B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": MODEL_CONFIG_THINK_TOKENS,
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": MODEL_CONFIG_THINK_TOKENS,
        # "deepcogito/cogito-v1-preview-llama-3B": MODEL_CONFIG_THINK_TOKENS,  # unverified
        "Wladastic/Mini-Think-Base-1B": MODEL_CONFIG_GEMMA,
        "google/gemma-2-2b-it": MODEL_CONFIG_GEMMA,
        "google/gemma-2-9b-it": MODEL_CONFIG_GEMMA,
        "openai/gpt-oss-20b": MODEL_CONFIG_GPT_OSS_20B,
        # "microsoft/phi-2": MODEL_CONFIG_FUZZY_ANSWER,  # not very consistent
        "meta-llama/Meta-Llama-3-8B-Instruct": MODEL_CONFIG_LLAMA,
        "meta-llama/Llama-2-7b-chat-hf": MODEL_CONFIG_LLAMA,
        "mistralai/Mistral-7B-Instruct-v0.3": MODEL_CONFIG_MISTRAL,
    }

    @staticmethod
    def get(model_name: str) -> dict:
        if model_name not in ModelConfig.SUPPORTED_MODELS:
            return ModelConfig.DEFAULT_MODEL_CONFIG
        return ModelConfig.SUPPORTED_MODELS[model_name]

    @staticmethod
    def is_supported(model_name: str) -> bool:
        return model_name in ModelConfig.SUPPORTED_MODELS


class ICLConfig:
    """Configuration for ICL (In-Context Learning) organisms"""

    ICL_ORGANISM_CONFIGS = [
        {
            "name": "icl-think",
            "filler_type": "think_token",
            "examples_file": "data/icl_examples/icl_think_token_default.json"
        },
        {
            "name": "icl-dot",
            "filler_type": "dot",
            "examples_file": "data/icl_examples/icl_dot_default.json"
        },
        {
            "name": "icl-comma",
            "filler_type": "comma",
            "examples_file": "data/icl_examples/icl_comma_default.json"
        },
        {
            "name": "icl-lorem-ipsum",
            "filler_type": "lorem_ipsum",
            "examples_file": "data/icl_examples/icl_lorem_ipsum_default.json"
        }
    ]

    @staticmethod
    def get_all_configs():
        """Get all ICL organism configurations"""
        return ICLConfig.ICL_ORGANISM_CONFIGS

    @staticmethod
    def get_config_by_name(name: str):
        """Get specific ICL organism configuration by name"""
        for config in ICLConfig.ICL_ORGANISM_CONFIGS:
            if config["name"] == name:
                return config
        return None

    @staticmethod
    def get_config_by_filler_type(filler_type: str):
        """Get specific ICL organism configuration by filler type"""
        for config in ICLConfig.ICL_ORGANISM_CONFIGS:
            if config["filler_type"] == filler_type:
                return config
        return None


class DatasetAdapter:
    def __init__(self, dataset_name: str, aliases: list, load_section: Optional[str] = None,
                 load_split: str = "train", do_extract: Optional[Callable] = None,
                 is_local_csv: bool = False, train_file: Optional[str] = None,
                 test_file: Optional[str] = None, csv_has_header: bool = True,
                 is_local_json: bool = False):
        """
        Args:
            dataset_name: Name of the dataset (for single file) or base name (for split files)
            aliases: List of alias names for this dataset
            load_section: Section to load for HF datasets
            load_split: Default split to load
            do_extract: Function to extract (question, cot, answer) from data
            is_local_csv: Whether this is a local CSV file
            train_file: Path to training CSV file (for pre-split datasets)
            test_file: Path to test CSV file (for pre-split datasets)
            csv_has_header: Whether CSV files have headers
            is_local_json: Whether this is a local JSON file
        """
        self.dataset_name = dataset_name
        self.aliases = aliases
        self.load_section = load_section
        self.load_split = load_split
        self.do_extract = do_extract
        self.is_local_csv = is_local_csv
        self.is_local_json = is_local_json or dataset_name.endswith('.json')
        self.train_file = train_file
        self.test_file = test_file
        self.csv_has_header = csv_has_header
        # Cache for local CSV/JSON data to ensure consistent splitting
        self._cached_dataset = None
        self._cached_train_split = None
        self._cached_test_split = None

    def extract_pieces(self, data: dict):
        if self.do_extract:
            return self.do_extract(data)
        return data

    def get(self, dataset_name: str) -> str:
        return self.dataset_name

    def _load_and_split_csv(self, csv_path: str, train_ratio: float = 0.8,
                            max_samples: Optional[int] = None, split: str = "train"):
        """Load CSV and split into train/test with caching for consistency"""
        if self._cached_dataset is None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"CSV file not found: {csv_path}")

            # Read CSV with or without header
            if self.csv_has_header:
                df = pd.read_csv(csv_path)
            else:
                # For CSV without headers, read as single column
                df = pd.read_csv(csv_path, header=None, names=['raw_text'])

            self._cached_dataset = Dataset.from_pandas(df)

            # Apply max_samples limit if specified
            if max_samples and max_samples < len(self._cached_dataset):
                self._cached_dataset = self._cached_dataset.select(range(max_samples))

            # Create consistent train/test split
            total_samples = len(self._cached_dataset)
            split_point = int(total_samples * train_ratio)

            # Split the dataset
            train_indices = list(range(split_point))
            test_indices = list(range(split_point, total_samples))

            self._cached_train_split = self._cached_dataset.select(train_indices)
            self._cached_test_split = self._cached_dataset.select(test_indices)

            print(
                f"Split local CSV {csv_path}: {len(self._cached_train_split)} train, {len(self._cached_test_split)} test")

        if split == "train":
            return self._cached_train_split
        else:
            return self._cached_test_split

    def _load_and_split_json(self, json_path: str,
                             train_ratio: float = 0.8,
                             max_samples: Optional[int] = None,
                             split: str = "train"):
        """Load JSON and split into train/test with caching for consistency"""
        if self._cached_dataset is None:
            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found: {json_path}")

            # Read JSON file
            import json
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Convert to pandas DataFrame then to Dataset
            df = pd.DataFrame(data)
            self._cached_dataset = Dataset.from_pandas(df)

            # Apply max_samples limit if specified
            if max_samples and max_samples < len(self._cached_dataset):
                self._cached_dataset = self._cached_dataset.select(range(max_samples))

            # Create consistent train/test split
            total_samples = len(self._cached_dataset)
            split_point = int(total_samples * train_ratio)

            # Split the dataset
            train_indices = list(range(split_point))
            test_indices = list(range(split_point, total_samples))

            self._cached_train_split = self._cached_dataset.select(train_indices)
            self._cached_test_split = self._cached_dataset.select(test_indices)

            print(
                f"Split local JSON {json_path}: {len(self._cached_train_split)} train, {len(self._cached_test_split)} test")

        if split == "train":
            return self._cached_train_split
        else:
            return self._cached_test_split

    def load(self, dataset_name: Optional[str] = None,
             max_samples: Optional[int] = None,
             split: Optional[str] = None) -> Dataset:
        """
        Load dataset from HuggingFace or local files.

        Args:
            dataset_name: Override dataset name (uses self.dataset_name if None)
            max_samples: Maximum number of samples to load
            split: Dataset split ('train' or 'test')

        Returns:
            Dataset object
        """
        # Use provided split or fall back to default
        split = split or self.load_split

        # Handle local JSON files
        if self.is_local_json:
            return self._load_and_split_json(
                self.dataset_name,
                train_ratio=0.8,
                max_samples=max_samples,
                split=split
            )

        # Handle local CSV files
        if self.is_local_csv:
            if self.train_file and self.test_file:
                # Pre-split CSV files
                csv_path = self.train_file if split == "train" else self.test_file
                if not os.path.exists(csv_path):
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")

                df = pd.read_csv(csv_path) if self.csv_has_header else pd.read_csv(csv_path, header=None,
                                                                                   names=['raw_text'])
                dataset = Dataset.from_pandas(df)

                if max_samples and max_samples < len(dataset):
                    dataset = dataset.select(range(max_samples))

                return dataset
            else:
                # Single CSV file that needs splitting
                return self._load_and_split_csv(
                    self.dataset_name,
                    train_ratio=0.8,
                    max_samples=max_samples,
                    split=split
                )

        # Handle HuggingFace datasets
        if self.load_section:
            dataset = load_dataset(self.dataset_name, self.load_section, split=split)
        else:
            dataset = load_dataset(self.dataset_name, split=split)

        # Apply max_samples limit if specified
        if max_samples and max_samples < len(dataset):
            dataset = dataset.select(range(max_samples))

        return dataset


class DatasetConfig:
    @staticmethod
    def _extract_3sum(data: dict) -> tuple:
        """Extract question, CoT, and answer from 3SUM data format.

        Format: "input_tuples P intermediate_tokens A answer"
        Example: "190 545 523 ... P . . . . A True"
        """
        raw_text = data.get("raw_text", "")

        # Split by " P " to separate input from intermediate tokens
        if " P " not in raw_text:
            print(f"Warning: No 'P' separator found in: {raw_text[:100]}...")
            return ("", "", "")

        input_part, rest = raw_text.split(" P ", 1)

        # Split by " A " to separate intermediate tokens from answer
        if " A " not in rest:
            print(f"Warning: No 'A' separator found in: {rest[:100]}...")
            return ("", "", "")

        intermediate_tokens, answer = rest.rsplit(" A ", 1)

        # Format the question with the input tuples
        question = f"Given the following tuples, does there exist three distinct tuples that sum to (0,0,0) modulo 10?\nInput: {input_part.strip()}"

        # CoT is the intermediate tokens (could be dots or chain-of-thought)
        cot = intermediate_tokens.strip()

        # Answer is True or False
        answer = answer.strip()

        return (question, cot, answer)

    HF_DATASET_LIST = [
        DatasetAdapter("vicgalle/alpaca-gpt4", ["alpaca", "alpaca-gpt4"],
                       do_extract=lambda d: (d["question"], "", d["answer"])),
        DatasetAdapter("gsm8k", ["GSM8K", "gsm8k"],
                       do_extract=lambda d: (
                           d["question"], d["answer"].split("####")[0], d["answer"].split("####")[1])),
        DatasetAdapter("cais/mmlu", ["MMLU", "mmlu"], load_section="all", load_split="test",
                       do_extract=lambda d: (d["question"] + "\nChoices: "
                                             + "\n".join([f"{chr(ord('A') + i)}: {d['choices'][i]}" for i in
                                                          range(len(d["choices"]))]),
                                             "", d["answer"])),
        # Local JSON dataset for binary alternation (auto-split from single file)
        DatasetAdapter("data/custom/binary_alternation.json", ["binary_alternation", "ba"],
                       do_extract=lambda d: (d["question"], d["cot"], d["answer"]),
                       is_local_json=True),
        # Additional datasets mentioned in sft_internalized.py
        DatasetAdapter("theory_of_mind", ["theory_of_mind"],
                       do_extract=lambda d: (d.get("question", ""), d.get("cot", ""), d.get("answer", ""))),
        DatasetAdapter("3sum", ["3sum"],
                       do_extract=_extract_3sum.__func__),
        DatasetAdapter("leg_counting", ["leg_counting"],
                       do_extract=lambda d: (d.get("question", ""), d.get("cot", ""), d.get("answer", "")))
    ]
    HF_DATASET_NAMES = {adapter.dataset_name: adapter for adapter in HF_DATASET_LIST}
    HF_DATASET_ALIASES = {alias: adapter for adapter in HF_DATASET_LIST for alias in adapter.aliases}

    @staticmethod
    def get(dataset_name: str) -> DatasetAdapter:
        if dataset_name in DatasetConfig.HF_DATASET_NAMES:
            return DatasetConfig.HF_DATASET_NAMES[dataset_name]
        if dataset_name in DatasetConfig.HF_DATASET_ALIASES:
            return DatasetConfig.HF_DATASET_ALIASES[dataset_name]
        # If not found, treat as a HuggingFace dataset or local file
        if dataset_name.endswith('.json'):
            return DatasetAdapter(dataset_name, [], is_local_json=True,
                                  do_extract=lambda d: (d.get("question", ""), d.get("cot", ""), d.get("answer", "")))
        elif dataset_name.endswith('.csv'):
            return DatasetAdapter(dataset_name, [], is_local_csv=True,
                                  do_extract=lambda d: (d.get("question", ""), d.get("cot", ""), d.get("answer", "")))
        else:
            # Assume it's a HuggingFace dataset
            return DatasetAdapter(dataset_name, [],
                                  do_extract=lambda d: (d.get("question", ""), d.get("cot", ""), d.get("answer", "")))

    @staticmethod
    def load_for_training(dataset_name: str,
                          max_samples: Optional[int] = None,
                          split: str = "train") -> list:
        """
        Load and prepare a dataset for training.

        Args:
            dataset_name: Name of the dataset to load
            max_samples: Limit number of samples
            split: Dataset split to use ('train' or 'test')

        Returns:
            List of dictionaries with 'question', 'cot', and 'answer' keys
        """
        adapter = DatasetConfig.get(dataset_name)
        raw_dataset = adapter.load(dataset_name, max_samples=max_samples, split=split)

        # Extract data in the format expected by training
        prepared_data = []
        for i, item in enumerate(raw_dataset):
            if max_samples and i >= max_samples:
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

                # Ensure all required keys exist
                if all(k in extracted for k in ["question", "cot", "answer"]):
                    prepared_data.append(extracted)

        print(f"Loaded {len(prepared_data)} samples for {split} split")
        return prepared_data