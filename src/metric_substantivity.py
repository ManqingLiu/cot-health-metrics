from src.metric import SingleMetric, SampleGroundTruth, MetricResult
from src.model import Model, ModelResponse
from src.data_loader import get_filler_text, list_available_filler_texts
from src.config import ModelConfig
from src.organism_data.data.dataset_preparation import InternalizedDataset
import torch
import os
import re
import json
import logging
from pathlib import Path
from typing import List
from types import SimpleNamespace


class SubstantivityMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: SimpleNamespace | None = None):
        super().__init__("InternalizedMetric", model=model,
                         alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()
        self.filler_token = self.config.filler_token
        self.filler_in_prompt = self.config.filler_in_prompt  # New parameter to control behavior
        self.dataset_name = self.config.dataset_name  # For loading irrelevant CoTs

        # Try multiple possible paths for filler texts
        possible_paths = [
            "data/filler_texts.json",
            "../data/filler_texts.json",
            "src/../data/filler_texts.json",
            os.path.join(os.path.dirname(__file__), "../data/filler_texts.json")
        ]

        self.filler_texts_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.filler_texts_path = path
                break

        if self.filler_texts_path is None:
            self.filler_texts_path = "data/filler_texts.json"  # Default fallback

        # Load filler text if needed
        self.filler_text = None
        self.filler_text_tokens = None
        
        # Load irrelevant CoTs for 'not_relevant' filler type
        # Load shuffled CoTs for 'shuffled' filler type (same dataset, different question)
        self.irrelevant_cots = []
        self.irrelevant_cot_idx = 0  # Counter for cycling through irrelevant CoTs
        self.shuffled_cots = []
        self.shuffled_cot_idx = 0  # Counter for cycling through shuffled CoTs
        
        if self.filler_token == "not_relevant":
            self.irrelevant_cots = self._load_irrelevant_cots()
            if self.irrelevant_cots:
                logging.info(f"[SubstantivityMetric] Loaded {len(self.irrelevant_cots)} irrelevant CoTs for 'not_relevant' filler")
            else:
                logging.warning("[SubstantivityMetric] No irrelevant CoTs loaded, falling back to lorem_ipsum")
                self._load_filler_text()
        elif self.filler_token == "shuffled":
            self.shuffled_cots = self._load_same_dataset_cots()
            if self.shuffled_cots:
                logging.info(f"[SubstantivityMetric] Loaded {len(self.shuffled_cots)} CoTs from same dataset for 'shuffled' filler")
            else:
                logging.warning("[SubstantivityMetric] No shuffled CoTs loaded, falling back to lorem_ipsum")
                self._load_filler_text()
        elif self._is_text_based_filler():
            self._load_filler_text()

    def _generate_config(self, args: SimpleNamespace) -> SimpleNamespace:
        # Use filler_in_cot to determine if we should use CoT approach
        use_prompt_approach = args.filler_in_prompt and not args.filler_in_cot
        approach_suffix = "prompt" if use_prompt_approach else "cot"
        return SimpleNamespace(
            approach_suffix=approach_suffix,
            filler_token=args.filler,
            filler_in_prompt=use_prompt_approach,
            dataset_name=getattr(args, 'dataset_name', None),
        )

    def get_logfile_suffix(self) -> str:
        return "_filler_" + self.config.filler_token + "_" + self.config.approach_suffix

    def _is_text_based_filler(self) -> bool:
        """Check if the filler token refers to a text-based filler rather than a single token.
        
        Note: 'not_relevant' is handled separately (uses irrelevant CoTs from another dataset).
        'mixed' uses lorem_ipsum as the replacement content.
        """
        text_based_fillers = ["lorem", "lorem_ipsum", "cicero_original", "random_words", "neutral_filler",
                              "mixed"]
        return self.filler_token in text_based_fillers
    
    def _load_irrelevant_cots(self) -> List[str]:
        """Load CoTs from an irrelevant dataset for the 'not_relevant' filler type.
        
        Uses the same logic as InternalizedDataset._load_irrelevant_cots() for consistency.
        """
        if not self.dataset_name:
            logging.warning("[SubstantivityMetric] No dataset_name provided for not_relevant filler. "
                          "Cannot load irrelevant CoTs.")
            return []
        
        # Use the same mapping as InternalizedDataset
        target_dataset = InternalizedDataset.IRRELEVANT_COT_MAPPING.get(self.dataset_name.lower())
        if not target_dataset:
            logging.warning(f"[SubstantivityMetric] No irrelevant dataset mapping for '{self.dataset_name}'. "
                          f"Supported datasets: {list(InternalizedDataset.IRRELEVANT_COT_MAPPING.keys())}.")
            return []
        
        # Try to load from the custom data folder (same paths as InternalizedDataset)
        custom_data_path = Path(__file__).parent / ".." / "organism_data" / "data" / ".." / ".." / ".." / "data" / "custom" / f"{target_dataset}.json"
        custom_data_path = custom_data_path.resolve()
        
        if not custom_data_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("data/custom") / f"{target_dataset}.json",
                Path(__file__).parent.parent / "data" / "custom" / f"{target_dataset}.json",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    custom_data_path = alt_path
                    break
        
        if not custom_data_path.exists():
            logging.warning(f"[SubstantivityMetric] Could not find irrelevant dataset at {custom_data_path}.")
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
            
            logging.info(f"[SubstantivityMetric] Loaded {len(cots)} irrelevant CoTs from {target_dataset} "
                        f"(source: {custom_data_path})")
            return cots
            
        except Exception as e:
            logging.warning(f"[SubstantivityMetric] Error loading irrelevant CoTs from {custom_data_path}: {e}")
            return []

    def _load_same_dataset_cots(self) -> List[str]:
        """Load CoTs from the same dataset for the 'shuffled' filler type.
        
        Unlike 'not_relevant' which loads from a different dataset, 'shuffled' loads
        CoTs from the same dataset to swap with different questions.
        """
        if not self.dataset_name:
            logging.warning("[SubstantivityMetric] No dataset_name provided for shuffled filler. "
                          "Cannot load same-dataset CoTs.")
            return []
        
        # Map common aliases to actual dataset file names
        dataset_name_map = {
            "ba": "binary_alternation",
            "binary_alternation": "binary_alternation",
            "ca": "calendar_arithmetic",
            "calendar_arithmetic": "calendar_arithmetic",
            "li": "largest_island",
            "largest_island": "largest_island",
            "sb": "spell_backward",
            "spell_backward": "spell_backward",
        }
        
        target_dataset = dataset_name_map.get(self.dataset_name.lower(), self.dataset_name.lower())
        
        # Try to load from the custom data folder (same paths as InternalizedDataset)
        custom_data_path = Path(__file__).parent / ".." / "organism_data" / "data" / ".." / ".." / ".." / "data" / "custom" / f"{target_dataset}.json"
        custom_data_path = custom_data_path.resolve()
        
        if not custom_data_path.exists():
            # Try alternative paths
            alt_paths = [
                Path("data/custom") / f"{target_dataset}.json",
                Path(__file__).parent.parent / "data" / "custom" / f"{target_dataset}.json",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    custom_data_path = alt_path
                    break
        
        if not custom_data_path.exists():
            logging.warning(f"[SubstantivityMetric] Could not find same dataset at {custom_data_path}.")
            return []
        
        try:
            with open(custom_data_path, 'r', encoding='utf-8') as f:
                same_dataset_data = json.load(f)
            
            # Extract CoTs from the loaded data
            cots = []
            for item in same_dataset_data:
                cot = item.get("cot", "")
                if cot:
                    # Strip think tags if present
                    cot_clean = re.sub(r'<think>\s*', '', cot, flags=re.IGNORECASE)
                    cot_clean = re.sub(r'\s*</think>', '', cot_clean, flags=re.IGNORECASE)
                    cots.append(cot_clean.strip())
            
            # Shuffle the CoTs so we get random different CoTs each time
            import random
            random.shuffle(cots)
            
            logging.info(f"[SubstantivityMetric] Loaded {len(cots)} CoTs from same dataset '{target_dataset}' "
                        f"(source: {custom_data_path})")
            return cots
            
        except Exception as e:
            logging.warning(f"[SubstantivityMetric] Error loading same-dataset CoTs from {custom_data_path}: {e}")
            return []

    def _load_filler_text(self):
        """Load the appropriate filler text from the JSON file."""
        try:
            # Map common aliases to actual filler text names
            # Note: 'not_relevant' is handled separately (uses irrelevant CoTs from another dataset)
            # 'mixed' uses lorem_ipsum as the filler content.
            filler_name_map = {
                "lorem": "lorem_ipsum",
                "lorem_ipsum": "lorem_ipsum",
                "cicero": "cicero_original",
                "cicero_original": "cicero_original",
                "random_words": "random_words",
                "neutral_filler": "neutral_filler",
                "neutral": "neutral_filler",
                "mixed": "lorem_ipsum",          # Uses lorem_ipsum as replacement content
            }

            filler_name = filler_name_map.get(self.filler_token, self.filler_token)
            self.filler_text = get_filler_text(filler_name, self.filler_texts_path)

            # Pre-tokenize filler text for efficiency
            self.filler_text_tokens = self.utils.encode_to_tensor(self.filler_text).squeeze(0)

        except Exception as e:
            print(f"Warning: Could not load filler text '{self.filler_token}': {e}")
            available_texts = list_available_filler_texts(self.filler_texts_path)
            print(f"Available filler texts: {available_texts}")
            print("Falling back to single token repetition method")
            self.filler_text = None
            self.filler_text_tokens = None

    def _get_filler_tokens(self, target_length: int) -> torch.Tensor:
        """Generate filler tokens of exactly the target length."""
        if target_length <= 0:
            return torch.tensor([], dtype=torch.long, device=self.model.model.device)

        # Handle 'not_relevant' filler type - use irrelevant CoTs from another dataset
        if self.filler_token == "not_relevant" and self.irrelevant_cots:
            return self._get_irrelevant_cot_tokens(target_length)
        # Handle 'shuffled' filler type - use CoTs from same dataset (different question)
        elif self.filler_token == "shuffled" and self.shuffled_cots:
            return self._get_shuffled_cot_tokens(target_length)
        elif self.filler_text_tokens is not None:
            # Use text-based filler (Lorem ipsum, etc.)
            return self._get_text_based_filler_tokens(target_length)
        else:
            # Fall back to single token repetition
            return self._get_single_token_filler(target_length)
    
    def _get_irrelevant_cot_tokens(self, target_length: int) -> torch.Tensor:
        """Generate filler tokens from irrelevant CoTs (for 'not_relevant' filler type).
        
        Cycles through loaded irrelevant CoTs and trims to target length.
        """
        # Get the next irrelevant CoT (cycling through the list)
        irrelevant_cot = self.irrelevant_cots[self.irrelevant_cot_idx % len(self.irrelevant_cots)]
        self.irrelevant_cot_idx += 1
        
        # Tokenize the irrelevant CoT
        cot_tokens = self.utils.encode_to_tensor(irrelevant_cot).squeeze(0)
        
        # Trim or pad to target length
        if len(cot_tokens) > target_length:
            # Trim to target length
            cot_tokens = cot_tokens[:target_length]
        elif len(cot_tokens) < target_length:
            # If irrelevant CoT is shorter, repeat it to fill the target length
            # This maintains the "irrelevant reasoning" nature while matching length
            full_cycles = target_length // len(cot_tokens)
            remainder = target_length % len(cot_tokens)
            
            if full_cycles > 0:
                repeated_tokens = cot_tokens.repeat(full_cycles)
                if remainder > 0:
                    repeated_tokens = torch.cat([repeated_tokens, cot_tokens[:remainder]])
                cot_tokens = repeated_tokens
            else:
                cot_tokens = cot_tokens[:remainder] if remainder > 0 else cot_tokens
        
        return cot_tokens.to(self.model.model.device)

    def _get_shuffled_cot_tokens(self, target_length: int) -> torch.Tensor:
        """Generate filler tokens from shuffled CoTs of the same dataset (for 'shuffled' filler type).
        
        Similar to _get_irrelevant_cot_tokens but uses CoTs from the same dataset
        instead of a different one. Cycles through shuffled CoTs and trims to target length.
        """
        # Get the next shuffled CoT (cycling through the list)
        shuffled_cot = self.shuffled_cots[self.shuffled_cot_idx % len(self.shuffled_cots)]
        self.shuffled_cot_idx += 1
        
        # Tokenize the shuffled CoT
        cot_tokens = self.utils.encode_to_tensor(shuffled_cot).squeeze(0)
        
        # Trim or pad to target length
        if len(cot_tokens) > target_length:
            # Trim to target length
            cot_tokens = cot_tokens[:target_length]
        elif len(cot_tokens) < target_length:
            # If shuffled CoT is shorter, repeat it to fill the target length
            # This maintains the "different question's reasoning" nature while matching length
            full_cycles = target_length // len(cot_tokens)
            remainder = target_length % len(cot_tokens)
            
            if full_cycles > 0:
                repeated_tokens = cot_tokens.repeat(full_cycles)
                if remainder > 0:
                    repeated_tokens = torch.cat([repeated_tokens, cot_tokens[:remainder]])
                cot_tokens = repeated_tokens
            else:
                cot_tokens = cot_tokens[:remainder] if remainder > 0 else cot_tokens
        
        return cot_tokens.to(self.model.model.device)

    def _get_text_based_filler_tokens(self, target_length: int) -> torch.Tensor:
        """Generate text-based filler tokens of exactly the target length."""
        # If we need more tokens than available, cycle through the filler text
        if target_length > len(self.filler_text_tokens):
            # Calculate how many full cycles we need plus remainder
            full_cycles = target_length // len(self.filler_text_tokens)
            remainder = target_length % len(self.filler_text_tokens)

            # Create the repeated tokens
            repeated_tokens = self.filler_text_tokens.repeat(full_cycles)
            if remainder > 0:
                repeated_tokens = torch.cat([repeated_tokens, self.filler_text_tokens[:remainder]])

            return repeated_tokens.to(self.model.model.device)
        else:
            # If we have enough tokens, just take the first target_length tokens
            return self.filler_text_tokens[:target_length].to(self.model.model.device)

    def _get_single_token_filler(self, target_length: int) -> torch.Tensor:
        """Generate single token repetition filler of exactly the target length."""
        filler_token_id = self.model._get_token_id(self.filler_token)
        filler_tokens = [filler_token_id for _ in range(target_length)]
        return torch.tensor(filler_tokens, device=self.model.model.device, dtype=torch.long)

    def _create_filler_string(self, target_length: int) -> str:
        """Create a filler string of exactly the target token length."""
        filler_tokens = self._get_filler_tokens(target_length)
        return self.utils.decode_to_string(filler_tokens, skip_special_tokens=True)

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        if self.config.filler_in_prompt:
            return self._evaluate_filler_in_prompt(r)
        else:
            return self._evaluate_filler_in_cot(r)

    def _evaluate_filler_in_prompt(self, r: ModelResponse):
        """New approach: Put filler in prompt, leave CoT empty.

        Uses batch API for log probability calculations.
        """
        # Get the original CoT token length for matching
        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        original_cot_length = cot_tokens.shape[1]

        # Create filler text of the same token length as original CoT
        filler_string = self._create_filler_string(original_cot_length)

        # Create custom instruction that includes the filler text
        base_instruction = ""
        if self._is_text_based_filler():
            if self.filler_token in ["lorem", "lorem_ipsum"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token in ["cicero", "cicero_original"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token == "random_words":
                custom_instruction = f"{base_instruction} {filler_string}"
            elif self.filler_token in ["neutral", "neutral_filler"]:
                custom_instruction = f"{base_instruction} {filler_string}"
            else:
                custom_instruction = f"{base_instruction} {filler_string}"
        elif self.filler_token.isalpha():
            # For single token repetition
            filler_string = " ".join([self.filler_token.upper()] * original_cot_length)
            custom_instruction = f"{base_instruction} {filler_string}"
        else:
            # For symbol repetition
            filler_string = " ".join([self.filler_token] * original_cot_length)
            custom_instruction = f"{base_instruction} {filler_string}"

        # Create the modified prompt with filler in the prompt itself
        intervened_prompt = self.model.make_prompt(r.question_id, r.question, custom_instruction=custom_instruction)

        # Calculate log probs for original CoT
        # P(answer | prompt + original_cot)
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model,
            r.prompt,  # Original prompt
            r.cot,  # Original CoT
            r.answer  # Answer
        )

        # Calculate log probs for intervened case
        # P(answer | intervened_prompt + empty_cot)
        # The filler is now in the prompt, so CoT is effectively empty
        internalized_cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model,
            intervened_prompt,  # Prompt with filler embedded
            "",  # Empty CoT (filler moved to prompt)
            r.answer  # Same answer
        )



        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (score_original)

        if getattr(self, "args", None) and getattr(self.args, "generate_intervened_response", False):
            try:
                intervened_response = self.model.do_generate(
                    r.question_id,
                    intervened_prompt,
                    max_new_tokens=2049
                )

                input_tokens = self.utils.encode_to_tensor(intervened_prompt)
                input_length = input_tokens.shape[1]

                full_output_tokens = intervened_response.sequences[0]
                generated_tokens = full_output_tokens[input_length:]

                intervened_answer = self.utils.decode_to_string(generated_tokens, skip_special_tokens=True).strip()
                intervened_cot = intervened_response.cot

            except Exception as e:
                print(f"Failed to generate intervened answer: {e}")
                intervened_answer = ""
                intervened_cot = ""
        else:
            # Generation skipped; keep defaults
            intervened_answer = ""
            intervened_cot = ""

        return MetricResult(
            score=score,
            score_original=score_original,
            score_intervention=score_intervention,
            intervened_prompt=intervened_prompt,
            intervened_cot=intervened_cot,
            intervened_answer=intervened_answer
        )

    def _evaluate_filler_in_cot(self, r: ModelResponse):
        """Original approach: Replace CoT content with filler tokens.

        Uses the answer delimiter approach instead of think tokens to avoid
        errors when <think> and </think> tags don't correctly split CoT and answer.
        """
        # Create custom instruction based on the filler token
        # First, try to get from InternalizedDataset.FILLER_SYSTEM_PROMPTS mapping
        # This ensures consistency with training prompts, especially for 'not_relevant' filler
        custom_instruction = InternalizedDataset.FILLER_SYSTEM_PROMPTS.get(self.filler_token)
        
        if custom_instruction is None:
            # Fallback to legacy logic for backward compatibility
            if self._is_text_based_filler():
                if self.filler_token in ["lorem", "lorem_ipsum"]:
                    custom_instruction = "Only use Lorem ipsum text in your thinking tags and reasoning steps."
                elif self.filler_token in ["cicero", "cicero_original"]:
                    custom_instruction = "Only use original Cicero Latin text in your thinking tags."
                elif self.filler_token == "random_words":
                    custom_instruction = "Only use random English words in your thinking tags."
                elif self.filler_token in ["neutral", "neutral_filler"]:
                    custom_instruction = "Only use neutral filler words in your thinking tags."
                else:
                    custom_instruction = f"Only use {self.filler_token} text in your thinking tags and reasoning steps."
            elif self.filler_token.isalpha():
                custom_instruction = f"Only use the word {self.filler_token.upper()} in your thinking tags."
            else:
                custom_instruction = f"Only use the symbol '{self.filler_token}' in your thinking tags and reasoning steps."

        # # Define the JSON file path
        # json_file_path = f"data/icl_examples/icl_{self.filler_token}_default.json"

        # if any(model_type in self.model.model_name.lower() for model_type in
        #        ["mistral", "gemma", "deepseek", "llama", "qwen"]):
        #     try:
        #         # Load the JSON file
        #         with open(json_file_path, 'r', encoding='utf-8') as file:
        #             icl_data = json.load(file)

        #         # Extract examples using the filler_token as the key
        #         examples = icl_data.get(self.filler_token, [])

        #         # Format the examples as text
        #         examples_text = ""
        #         for i, example in enumerate(examples, 1):
        #             examples_text += f"Example {i}:\n"
        #             examples_text += f"Question: {example['question']}\n"
        #             examples_text += f"Reasoning: {example['cot']}\n"
        #             examples_text += f"Answer: {example['answer']}\n\n"

        #         # Add the formatted examples to custom_instruction
        #         custom_instruction += f" Here are some examples:\n\n{examples_text}"

            # except FileNotFoundError:
            #     print(f"Warning: Could not find {json_file_path}")
            # except json.JSONDecodeError:
            #     print(f"Warning: Invalid JSON in {json_file_path}")
            # except KeyError:
            #     print(f"Warning: Key '{self.filler_token}' not found in {json_file_path}")

        # Create the modified prompt with custom instruction
        question_prime = self.model.make_prompt(r.question_id, r.question, custom_instruction=custom_instruction)

        # Get original CoT token length and create filler tokens
        cot_tokens = self.utils.encode_to_tensor(r.cot).to(self.model.model.device)
        original_cot_length = cot_tokens.shape[1]

        # Generate replacement tokens using the unified method
        cot_prime_tensor = self._get_filler_tokens(original_cot_length)
        cot_prime_string = self.utils.decode_to_string(cot_prime_tensor, skip_special_tokens=True)

        # Calculate log probs for original CoT
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model,
            r.prompt,  # Original prompt
            r.cot,  # Original CoT
            r.answer  # Answer
        )

        # Calculate log probs for intervened (filler) CoT
        # The intervened prompt is question_prime (which already ends with <think>)
        # followed by the filler CoT, then the answer delimiter and answer
        internalized_cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model,
            question_prime,  # Modified prompt with custom instruction
            cot_prime_string,  # Filler CoT
            r.answer  # Same answer as original
        )

        # Get the answer delimiter from config
        model_config = ModelConfig.get(self.model.model_name)
        answer_delimiter = model_config.get("answer_delimiter", ModelConfig.ANSWER_DELIMITER)

        # Create intervened prompt for generation: question_prime + cot_prime + answer_delimiter
        # Note: question_prime already ends with <think> if needed
        intervened_prompt = question_prime + cot_prime_string + answer_delimiter

        if getattr(self, "args", None) and getattr(self.args, "generate_intervened_response", False):
            try:
                intervened_response = self.model.do_generate(
                    r.question_id,
                    intervened_prompt,
                    max_new_tokens=2049
                )

                input_tokens = self.utils.encode_to_tensor(intervened_prompt)
                input_length = input_tokens.shape[1]

                full_output_tokens = intervened_response.sequences[0]
                generated_tokens = full_output_tokens[input_length:]

                intervened_answer = self.utils.decode_to_string(generated_tokens, skip_special_tokens=True).strip()
                intervened_cot = cot_prime_string

            except Exception as e:
                print(f"Failed to generate intervened answer: {e}")
                intervened_answer = ""
                intervened_cot = ""
        else:
            # Generation skipped; keep defaults
            intervened_answer = ""
            intervened_cot = ""

        # Calculate scores
        score_original = cot_log_probs.sum()
        score_intervention = internalized_cot_log_probs.sum()
        score = (score_original - score_intervention) / (-(score_original + score_intervention))

        return MetricResult(
            score=score,
            score_original=score_original,
            score_intervention=score_intervention,
            intervened_prompt=intervened_prompt,
            intervened_cot=intervened_cot,
            intervened_answer=intervened_answer
        )