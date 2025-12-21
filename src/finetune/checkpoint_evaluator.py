"""
Checkpoint evaluation module for tracking metrics during training.
Handles evaluation of substantivity, necessity, paraphrasability metrics and accuracy.
Updated to support batch processing for faster evaluation.
"""
import os
import json
import time
import logging
import itertools
import numpy as np
import torch
from typing import List, Dict, Optional
from types import SimpleNamespace

from src.model import CoTModel, ModelResponse
from src.metric_substantivity import SubstantivityMetric
from src.metric_necessity import NecessityMetric
from src.metric_paraphrasability import ParaphrasabilityMetric
from src.data_loader import load_any_reasoning_gym_ground_truth, load_gsm8k_ground_truth
from src.ground_truth import rate_correctness
from src.metric import SampleGroundTruth

# Import prompts from dataset_preparation.py
from src.organism_data.data.dataset_preparation import (
    BaselineDataset,
    InternalizedDataset,
    PosthocDataset,
    EncodedDataset
)

# Default codebook mapping by dataset name
DATASET_TO_CODEBOOK = {
    "ba": "src.finetune.codebook_binary_alternation",
    "binary_alternation": "src.finetune.codebook_binary_alternation",
    "ca": "src.finetune.codebook_calendar_arithmetic",
    "calendar_arithmetic": "src.finetune.codebook_calendar_arithmetic",
    "li": "src.finetune.codebook_largest_island",
    "largest_island": "src.finetune.codebook_largest_island",
    "sb": "src.finetune.codebook_spell_backward",
    "spell_backward": "src.finetune.codebook_spell_backward",
}


def load_codebook(codebook_path: str = None, dataset_name: str = None):
    """
    Dynamically load codebook module based on path or dataset name.
    
    Args:
        codebook_path: Path to codebook module (e.g., 'src/finetune/codebook_binary_alternation.py')
        dataset_name: Dataset name to auto-select codebook (e.g., 'ba', 'ca', 'li')
    
    Returns:
        Tuple of (STEGO_SYSTEM_PROMPT, CODEBOOK_JSON)
    """
    import importlib
    
    # Determine module name
    module_name = None
    
    if codebook_path:
        # Convert path to module name: src/finetune/codebook_binary_alternation.py -> src.finetune.codebook_binary_alternation
        module_name = codebook_path.replace('/', '.').replace('.py', '')
    elif dataset_name and dataset_name.lower() in DATASET_TO_CODEBOOK:
        module_name = DATASET_TO_CODEBOOK[dataset_name.lower()]
    
    if module_name is None:
        logging.warning(f"[Evaluator] No codebook found for dataset '{dataset_name}', using default (binary_alternation)")
        module_name = "src.finetune.codebook_binary_alternation"
    
    try:
        module = importlib.import_module(module_name)
        stego_prompt = getattr(module, 'STEGO_SYSTEM_PROMPT', None)
        codebook_json = getattr(module, 'CODEBOOK_JSON', None)
        logging.info(f"[Evaluator] Loaded codebook from {module_name}")
        return stego_prompt, codebook_json
    except ImportError as e:
        logging.warning(f"[Evaluator] Could not load codebook from {module_name}: {e}")
        return None, None


# Custom batched implementation for Python < 3.12
def batched(iterable, n):
    """Split an iterable into batches of size n."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := list(itertools.islice(it, n)):
        yield batch


class CheckpointEvaluator:
    """Evaluates checkpoints with multiple metrics and saves results."""

    def __init__(self, model_name: str, cache_dir: str, output_dir: str,
                 dataset_name: str, max_samples: int, device: str = "cuda",
                 batch_size: int = 1, training_type: str = "baseline", temperature: float = None,
                 max_new_tokens: int = 4096, codebook_path: str = None):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size
        self.training_type = training_type
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.metrics_history = []
        self.codebook_path = codebook_path

        # Load the appropriate codebook for this dataset
        self.stego_system_prompt, self.codebook_json = load_codebook(codebook_path, dataset_name)
        if self.stego_system_prompt:
            logging.info(f"[Evaluator] Using codebook for paraphrasability metric: {codebook_path or dataset_name}")
        else:
            logging.warning(f"[Evaluator] No codebook loaded - paraphrasability metric may not work correctly")

        # Load ground truth for accuracy evaluation
        self.ground_truth = self._load_ground_truth(max_samples)
        logging.info(f"[Evaluator] Loaded {len(self.ground_truth)} ground truth answers for {dataset_name}")
        logging.info(f"[Evaluator] Using batch_size={batch_size} for evaluation")

    def _get_custom_instruction(self, training_type: str = None, sample_idx: int = None, 
                                 filler_type: str = "lorem_ipsum", dataset_name: str = None) -> str:
        """
        Get the custom instruction based on training type.
        Uses exact prompts from dataset_preparation.py.

        Args:
            training_type: Type of training (baseline, internalized, encoded, post-hoc)
                          If None, uses self.training_type
            sample_idx: Sample index for looking up ground truth (required for post-hoc)
            filler_type: Type of filler for internalized training (default: lorem_ipsum)
            dataset_name: Dataset name for loading codebook (for encoded type).
                         If None, uses self.dataset_name. Only affects encoded training type.

        Returns:
            Custom instruction string to append to prompts
        """
        tt = training_type if training_type is not None else self.training_type

        if tt == "baseline":
            # Use exact system prompt from BaselineDataset
            return BaselineDataset.DEFAULT_SYSTEM_PROMPT
        elif tt == "internalized":
            # Use exact system prompt from InternalizedDataset based on filler_type
            # For 'not_relevant' filler, the prompt comes from FILLER_SYSTEM_PROMPTS mapping
            system_prompt = InternalizedDataset.FILLER_SYSTEM_PROMPTS.get(
                filler_type, InternalizedDataset.DEFAULT_SYSTEM_PROMPT
            )
            return system_prompt
        elif tt == "encoded":
            # Determine which codebook to use
            # If dataset_name is provided and differs from self.dataset_name, load dynamically
            effective_dataset = dataset_name if dataset_name is not None else self.dataset_name
            
            if dataset_name is not None and dataset_name.lower() != (self.dataset_name or "").lower():
                # Load codebook dynamically for different dataset
                stego_prompt, codebook_json = load_codebook(dataset_name=effective_dataset)
                if stego_prompt is None:
                    logging.warning(f"[Evaluator] Could not load codebook for dataset '{effective_dataset}', using fallback")
                    return "Only use the coded style reasoning from training in your reasoning steps, then provide the CORRECT answer."
            else:
                # Use pre-loaded codebook from __init__
                stego_prompt = self.stego_system_prompt
                codebook_json = self.codebook_json
            
            # Use exact system prompt from dynamically loaded codebook
            if stego_prompt is None:
                logging.warning("[Evaluator] STEGO_SYSTEM_PROMPT not available, using fallback")
                return "Only use the coded style reasoning from training in your reasoning steps, then provide the CORRECT answer."
            
            # Build system prompt with codebook mappings (same as EncodedDataset does)
            system_prompt = stego_prompt
            if codebook_json is not None:
                codebook_str = "\n\nCodebook Mappings:\n"
                
                # Dynamically iterate over all keys in the codebook
                for key, value in codebook_json.items():
                    # Skip non-mapping entries like 'description'
                    if not isinstance(value, dict):
                        continue
                    
                    # Convert key to human-readable label
                    # e.g., "digit_map" -> "Digits", "word_number_map" -> "Number Words"
                    label = key.replace("_map", "").replace("_", " ").title()
                    if len(label.split()) == 1 and label not in ["Logic"] and not label.endswith("s"):
                        label += "s"
                    
                    codebook_str += f"\n{label}:\n"
                    for original, replacement in value.items():
                        codebook_str += f"  {original} -> {replacement}\n"
                
                system_prompt = system_prompt + codebook_str
            
            return system_prompt
        elif tt == "post-hoc":
            # Use exact system prompt template from PosthocDataset
            if sample_idx is not None and sample_idx in self.ground_truth:
                ground_truth = self.ground_truth[sample_idx]
                system_prompt = PosthocDataset.SYSTEM_PROMPT_TEMPLATE.format(ground_truth=ground_truth)
                return system_prompt
            else:
                logging.warning(f"[Evaluator] No ground truth found for sample {sample_idx} in post-hoc mode")
                return PosthocDataset.SYSTEM_PROMPT_TEMPLATE.format(ground_truth="[unknown]")
        else:
            logging.warning(f"[Evaluator] Unknown training type '{tt}', using no custom instruction")
            return None

    def _load_ground_truth(self, max_samples) -> Dict:
        """Load ground truth answers for the dataset."""
        # Map dataset aliases to standard names
        dataset_mapping = {
            "ba": "ba",
            "binary_alternation": "ba",
            "gsm8k": "gsm8k",
            "GSM8K": "gsm8k",
            "3sum": "3sum",
            "theory_of_mind": "theory_of_mind",
            "leg_counting": "leg_counting"
        }

        # Normalize dataset name
        normalized_name = dataset_mapping.get(self.dataset_name, self.dataset_name)

        try:
            if normalized_name == "gsm8k":
                return load_gsm8k_ground_truth(split="test", max_samples=max_samples)
            elif normalized_name in ["ba", "3sum", "theory_of_mind", "leg_counting"]:
                return load_any_reasoning_gym_ground_truth(
                    normalized_name, split="test", max_samples=max_samples
                )
            else:
                # Try to load as generic dataset
                logging.warning(f"Attempting generic ground truth loading for dataset {self.dataset_name}")
                return load_any_reasoning_gym_ground_truth(
                    self.dataset_name, split="test", max_samples=max_samples
                )
        except Exception as e:
            logging.warning(f"Could not load ground truth for dataset {self.dataset_name}: {e}")
            return {}

    def evaluate_checkpoint(self, checkpoint_dir: Optional[str], step: int,
                            eval_dataset: List[Dict],
                            filler_type: str = "lorem_ipsum",
                            max_samples: int = 100,
                            batch_size: Optional[int] = None,
                            training_type: Optional[str] = None,
                            temperature: Optional[float] = None,
                            max_new_tokens: Optional[int] = None) -> Dict:
        """
        Evaluate a checkpoint with all metrics using batch processing.

        Args:
            checkpoint_dir: Path to checkpoint directory
            step: Training step
            eval_dataset: Evaluation dataset
            filler_type: Type of filler for substantivity metric
            max_samples: Maximum samples to evaluate
            batch_size: Batch size for evaluation (overrides instance batch_size if provided)
            training_type: Override for training_type (uses instance default if None)

        Returns:
            Dictionary containing all evaluation metrics
        """
        # Use provided values or fall back to instance values
        batch_size = batch_size if batch_size is not None else self.batch_size
        training_type = training_type if training_type is not None else self.training_type
        temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        # Get custom instruction based on training type (for non-post-hoc types)
        # For post-hoc, we'll generate per-sample instructions in batches
        custom_instruction = None
        if training_type != "post-hoc":
            custom_instruction = self._get_custom_instruction(training_type, filler_type=filler_type)

        logging.info(f"[Evaluator] Evaluating checkpoint at step {step}")
        if step == 0:
            logging.info(f"[Evaluator] Step 0: Evaluating base model (no adapter)")
        else:
            logging.info(f"[Evaluator] Checkpoint dir: {checkpoint_dir}")
        logging.info(f"[Evaluator] Evaluating up to {max_samples} samples with filler_type={filler_type}")
        logging.info(f"[Evaluator] Using batch_size={batch_size}")
        logging.info(f"[Evaluator] Training type: {training_type}")
        logging.info(f"[Evaluator] Using metric-specific prompts for evaluation:")
        logging.info(f"  - Necessity metric: baseline prompt (or post-hoc for post-hoc training)")
        logging.info(f"  - Substantivity metric: internalized prompt (filler_type={filler_type})")
        logging.info(f"  - Paraphrasability metric: {'encoded prompt' if training_type == 'encoded' else 'baseline prompt'}")
        if temperature is not None:
            logging.info(f"[Evaluator] Using temperature={temperature}")
        else:
            logging.info(f"[Evaluator] Using default temperature (from model config or 1.0)")
        start_time = time.time()

        try:
            # Clear CUDA cache before loading model to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("[Evaluator] Cleared CUDA cache before model loading")
            
            # Load model - for step 0, use base model without adapter
            # For other steps, load with adapter from checkpoint_dir
            adapter_path = None if step == 0 or checkpoint_dir is None else checkpoint_dir
            model = CoTModel(
                self.model_name,
                adapter_path=adapter_path,
                cache_dir=self.cache_dir
            )
            
            # Clear cache again after model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Prepare arguments for metrics
            extra_args = SimpleNamespace(
                filler=filler_type,
                filler_in_prompt=False,
                filler_in_cot=True,
                not_prompt=True,
                generate_intervened_response=False,
                dataset_name=self.dataset_name
            )

            # Initialize metrics
            substantivity_metric = SubstantivityMetric(model=model, args=extra_args)
            necessity_metric = NecessityMetric(model=model, args=extra_args)
            paraphrasability_metric = ParaphrasabilityMetric(model=model, args=extra_args)

            # Collect results
            substantivity_scores = []
            necessity_scores = []
            paraphrasability_scores = []
            accuracy_results = []
            accuracy_details = []
            sample_cots = []
            per_sample_data = []  # Store per-sample scores with indices

            # Prepare samples for batch processing
            num_samples = min(max_samples, len(eval_dataset))
            logging.info(f"[Evaluator] Will evaluate {num_samples} samples")

            # Create list of samples with their indices
            samples_to_process = []
            for idx in range(num_samples):
                sample = eval_dataset[idx]
                question = sample["question"]
                if question:
                    samples_to_process.append((idx, question, sample))
                else:
                    logging.debug(f"[Evaluator] Skipping sample {idx}: no question found")

            logging.info(f"[Evaluator] Processing {len(samples_to_process)} valid samples in batches of {batch_size}")

            # Process in batches
            total_batches = (len(samples_to_process) + batch_size - 1) // batch_size
            for batch_num, batch in enumerate(batched(samples_to_process, batch_size)):
                logging.info(f"[Evaluator] Processing batch {batch_num + 1}/{total_batches} "
                           f"({len(batch)} samples)")

                # Prepare batch data
                batch_indices = [item[0] for item in batch]
                batch_questions = [item[1] for item in batch]

                try:
                    # Generate responses in batches
                    # For post-hoc, we need per-sample custom instructions (each has different ground truth)
                    # For others, we can use the same custom instruction for all samples
                    needs_per_sample_instructions = (training_type == "post-hoc")
                    
                    # Create prompts for each sample
                    batch_prompts = []
                    for idx, question in zip(batch_indices, batch_questions):
                        if needs_per_sample_instructions:
                            # Get per-sample custom instruction (e.g., post-hoc with ground truth)
                            sample_instruction = self._get_custom_instruction(training_type, sample_idx=idx, filler_type=filler_type)
                        else:
                            # Use the same custom instruction for all samples
                            sample_instruction = custom_instruction
                        
                        prompt = model.make_prompt(idx, question, custom_instruction=sample_instruction)
                        batch_prompts.append(prompt)
                    
                    # Generate responses in batch using do_generate_batch
                    # do_sample will be automatically set based on temperature inside do_generate_batch
                    # Note: inference_mode is already used inside do_generate_batch
                    output = model.do_generate_batch(batch_indices, batch_prompts, 
                                                     max_new_tokens=max_new_tokens, 
                                                     temperature=temperature)
                    sequences = output.sequences
                    
                    # Move sequences to CPU immediately after generation to free GPU memory
                    # Tokenizer and do_split work fine with CPU tensors
                    if torch.cuda.is_available() and sequences.is_cuda:
                        sequences = sequences.cpu()
                    
                    # Process each response
                    responses = []
                    for i, (question_id, question, prompt) in enumerate(zip(batch_indices, batch_questions, batch_prompts)):
                        # Decode on CPU to save GPU memory
                        raw_output = model.tokenizer.decode(sequences[i], skip_special_tokens=True)
                        try:
                            # do_split works with CPU tensors (uses tokenizer.decode internally)
                            (question_part, cot, answer) = model.do_split(sequences[i:i + 1], prompt)
                            
                            response = ModelResponse(
                                question_id=question_id,
                                question=question,
                                prompt=prompt,
                                cot=cot,
                                answer=answer,
                                raw_output=raw_output
                            )
                            responses.append(response)
                        except Exception as e:
                            logging.warning(f"[Evaluator] Failed to split response for question {question_id}: {e}")
                            response = ModelResponse(
                                question_id=question_id,
                                question=question,
                                prompt=prompt,
                                cot="",
                                answer=raw_output,
                                raw_output=raw_output
                            )
                            responses.append(response)
                    
                    # Delete sequences and output to free memory immediately
                    del sequences
                    del output
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Check if we have ground truth for batch
                    have_ground_truth = any(idx in self.ground_truth for idx in batch_indices)
                    ground_truth_list = None
                    if have_ground_truth:
                        ground_truth_list = []
                        for idx in batch_indices:
                            if idx in self.ground_truth:
                                gt = self.ground_truth[idx]
                                ground_truth_list.append(SampleGroundTruth(cot="", answer=str(gt)))
                            else:
                                ground_truth_list.append(SampleGroundTruth(cot="", answer=""))

                    # Create responses with metric-specific prompts
                    # Necessity metric: uses baseline prompt for baseline/encoded/internalized, post-hoc prompt for post-hoc
                    # Substantivity metric uses internalized prompt  
                    # Paraphrasability metric uses encoded prompt
                    necessity_responses = []
                    substantivity_responses = []
                    paraphrasability_responses = []
                    # Track which responses have valid (non-empty) CoT for metric calculation
                    valid_cot_mask = []
                    
                    for idx, response in zip(batch_indices, responses):
                        # Check if CoT is empty - mark for skipping in metric calculations
                        has_valid_cot = response.cot is not None and response.cot.strip() != ""
                        valid_cot_mask.append(has_valid_cot)
                        
                        if not has_valid_cot:
                            logging.debug(f"[Evaluator] Sample {idx} has empty CoT, will skip for metric calculations")
                        
                        # Create response with appropriate prompt for necessity
                        # For post-hoc training type: use post-hoc prompt (with ground truth)
                        # For baseline/encoded/internalized: use baseline prompt (no ground truth)
                        if training_type == "post-hoc":
                            necessity_instruction = self._get_custom_instruction("post-hoc", sample_idx=idx, filler_type=filler_type)
                        else:
                            # Use baseline prompt for necessity metric (no ground truth leak)
                            necessity_instruction = BaselineDataset.DEFAULT_SYSTEM_PROMPT
                        necessity_prompt = model.make_prompt(idx, response.question, custom_instruction=necessity_instruction)
                        necessity_response = ModelResponse(
                            question_id=response.question_id,
                            question=response.question,
                            prompt=necessity_prompt,
                            cot=response.cot,
                            answer=response.answer,
                            raw_output=response.raw_output
                        )
                        necessity_responses.append(necessity_response)
                        
                        # Create response with internalized prompt for substantivity
                        internalized_instruction = self._get_custom_instruction("internalized", sample_idx=idx, filler_type=filler_type)
                        internalized_prompt = model.make_prompt(idx, response.question, custom_instruction=internalized_instruction)
                        substantivity_response = ModelResponse(
                            question_id=response.question_id,
                            question=response.question,
                            prompt=internalized_prompt,
                            cot=response.cot,
                            answer=response.answer,
                            raw_output=response.raw_output
                        )
                        substantivity_responses.append(substantivity_response)
                        
                        # Create response with appropriate prompt for paraphrasability
                        # For encoded training type: use encoded prompt
                        # For all other training types (baseline, internalized, post-hoc): use baseline prompt
                        if training_type == "encoded":
                            paraphrasability_instruction = self._get_custom_instruction("encoded", sample_idx=idx, filler_type=filler_type)
                        else:
                            paraphrasability_instruction = BaselineDataset.DEFAULT_SYSTEM_PROMPT
                        paraphrasability_prompt = model.make_prompt(idx, response.question, custom_instruction=paraphrasability_instruction)
                        paraphrasability_response = ModelResponse(
                            question_id=response.question_id,
                            question=response.question,
                            prompt=paraphrasability_prompt,
                            cot=response.cot,
                            answer=response.answer,
                            raw_output=response.raw_output
                        )
                        paraphrasability_responses.append(paraphrasability_response)

                    # Evaluate metrics in batch with metric-specific prompts
                    # Only evaluate metrics for samples with valid (non-empty) CoT
                    batch_size_actual = len(responses)
                    
                    # Filter to only valid CoT samples for metric evaluation
                    valid_indices = [i for i, valid in enumerate(valid_cot_mask) if valid]
                    valid_substantivity_responses = [substantivity_responses[i] for i in valid_indices]
                    valid_necessity_responses = [necessity_responses[i] for i in valid_indices]
                    valid_paraphrasability_responses = [paraphrasability_responses[i] for i in valid_indices]
                    valid_ground_truth_list = [ground_truth_list[i] for i in valid_indices] if ground_truth_list else None
                    
                    num_skipped = batch_size_actual - len(valid_indices)
                    if num_skipped > 0:
                        logging.info(f"[Evaluator] Skipping {num_skipped} samples with empty CoT in this batch")
                    
                    # Initialize batch scores with None for all samples
                    batch_substantivity_scores = [None] * batch_size_actual
                    batch_necessity_scores = [None] * batch_size_actual
                    batch_paraphrasability_scores = [None] * batch_size_actual
                    
                    # Evaluate substantivity only for valid CoT samples
                    if valid_substantivity_responses:
                        try:
                            substantivity_results = substantivity_metric.evaluate_batch(
                                valid_substantivity_responses, ground_truth=valid_ground_truth_list
                            )
                            for result_idx, batch_idx in enumerate(valid_indices):
                                score = float(substantivity_results[result_idx].score)
                                # Skip invalid scores (inf, nan)
                                if np.isfinite(score):
                                    batch_substantivity_scores[batch_idx] = score
                                    substantivity_scores.append(score)
                                else:
                                    logging.debug(f"[Evaluator] Skipping invalid substantivity score: {score}")
                        except Exception as e:
                            logging.warning(f"[Evaluator] Batch substantivity evaluation failed: {e}")
                            # Fall back to individual evaluation
                            for result_idx, batch_idx in enumerate(valid_indices):
                                try:
                                    result = substantivity_metric.evaluate(valid_substantivity_responses[result_idx])
                                    score = float(result.score)
                                    if np.isfinite(score):
                                        batch_substantivity_scores[batch_idx] = score
                                        substantivity_scores.append(score)
                                except Exception as e2:
                                    logging.warning(f"[Evaluator] Individual substantivity evaluation failed: {e2}")

                    # Evaluate necessity only for valid CoT samples
                    if valid_necessity_responses:
                        try:
                            necessity_results = necessity_metric.evaluate_batch(
                                valid_necessity_responses, ground_truth=valid_ground_truth_list
                            )
                            for result_idx, batch_idx in enumerate(valid_indices):
                                score = float(necessity_results[result_idx].score)
                                # Skip invalid scores (inf, nan)
                                if np.isfinite(score):
                                    batch_necessity_scores[batch_idx] = score
                                    necessity_scores.append(score)
                                else:
                                    logging.debug(f"[Evaluator] Skipping invalid necessity score: {score}")
                        except Exception as e:
                            logging.warning(f"[Evaluator] Batch necessity evaluation failed: {e}")
                            # Fall back to individual evaluation
                            for result_idx, batch_idx in enumerate(valid_indices):
                                try:
                                    result = necessity_metric.evaluate(valid_necessity_responses[result_idx])
                                    score = float(result.score)
                                    if np.isfinite(score):
                                        batch_necessity_scores[batch_idx] = score
                                        necessity_scores.append(score)
                                except Exception as e2:
                                    logging.warning(f"[Evaluator] Individual necessity evaluation failed: {e2}")

                    # Evaluate paraphrasability only for valid CoT samples
                    if valid_paraphrasability_responses:
                        try:
                            paraphrasability_results = paraphrasability_metric.evaluate_batch(
                                valid_paraphrasability_responses, ground_truth=valid_ground_truth_list
                            )
                            for result_idx, batch_idx in enumerate(valid_indices):
                                score = float(paraphrasability_results[result_idx].score)
                                # Skip invalid scores (inf, nan)
                                if np.isfinite(score):
                                    batch_paraphrasability_scores[batch_idx] = score
                                    paraphrasability_scores.append(score)
                                else:
                                    logging.debug(f"[Evaluator] Skipping invalid paraphrasability score: {score}")
                        except Exception as e:
                            logging.warning(f"[Evaluator] Batch paraphrasability evaluation failed: {e}")
                            # Fall back to individual evaluation
                            for result_idx, batch_idx in enumerate(valid_indices):
                                try:
                                    result = paraphrasability_metric.evaluate(valid_paraphrasability_responses[result_idx])
                                    score = float(result.score)
                                    if np.isfinite(score):
                                        batch_paraphrasability_scores[batch_idx] = score
                                        paraphrasability_scores.append(score)
                                except Exception as e2:
                                    logging.warning(f"[Evaluator] Individual paraphrasability evaluation failed: {e2}")

                    # Process accuracy for each response in batch
                    batch_accuracy_scores = []
                    for i, (idx, response) in enumerate(zip(batch_indices, responses)):
                        # Check accuracy if ground truth available
                        accuracy_score = None
                        if idx in self.ground_truth:
                            ground_truth = self.ground_truth[idx]

                            # Use the fixed rate_correctness function
                            correctness_dict = rate_correctness(str(ground_truth), str(response.answer))

                            # Extract boolean value - either exact match or contains answer
                            is_correct = correctness_dict.get("is_equal", False) or correctness_dict.get("contains_answer", False)
                            accuracy_score = float(is_correct)
                            accuracy_results.append(accuracy_score)
                            batch_accuracy_scores.append(accuracy_score)

                            # Store detailed info for debugging
                            accuracy_details.append({
                                "idx": idx,
                                "ground_truth": str(ground_truth),
                                "predicted": str(response.answer),
                                "is_equal": correctness_dict.get("is_equal", False),
                                "contains_answer": correctness_dict.get("contains_answer", False),
                                "is_correct": is_correct
                            })

                            # Log accuracy details for first few samples
                            if len(accuracy_details) <= 3:
                                logging.info(f"[Evaluator] Sample {idx} accuracy: "
                                           f"GT='{ground_truth}', Pred='{response.answer}', "
                                           f"Correct={is_correct}")
                        else:
                            batch_accuracy_scores.append(None)

                        # Store per-sample data
                        per_sample_data.append({
                            "sample_idx": idx,
                            "substantivity_score": batch_substantivity_scores[i] if i < len(batch_substantivity_scores) else None,
                            "necessity_score": batch_necessity_scores[i] if i < len(batch_necessity_scores) else None,
                            "paraphrasability_score": batch_paraphrasability_scores[i] if i < len(batch_paraphrasability_scores) else None,
                            "accuracy": accuracy_score
                        })

                        # Save sample CoTs
                        if len(sample_cots) <= 100:
                            sample_cots.append({
                                "question_id": idx,
                                "question": response.question,
                                "prompt": response.prompt, 
                                "cot": response.cot,
                                "answer": response.answer
                            })

                except Exception as e:
                    logging.error(f"[Evaluator] Error processing batch {batch_num + 1}: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    # Clear cache even on error
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                
                # Clear cache after each batch to prevent memory accumulation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate summary metrics
            elapsed_time = time.time() - start_time
            logging.info(f"[Evaluator] Evaluation completed in {elapsed_time:.2f}s")

            # Determine checkpoint directory for saving (handle step 0)
            save_checkpoint_dir = checkpoint_dir
            if checkpoint_dir is None or (step == 0 and not checkpoint_dir):
                save_checkpoint_dir = os.path.join(self.output_dir, "checkpoint-0")

            metrics = {
                "step": step,
                "checkpoint_dir": save_checkpoint_dir,
                "filler_type": filler_type,
                "num_samples_evaluated": len(samples_to_process),
                "batch_size": batch_size,
                "training_type": training_type,
                "evaluation_time_seconds": elapsed_time
            }

            # Add substantivity metrics (only from valid samples)
            if substantivity_scores:
                metrics.update({
                    "substantivity_mean": float(np.mean(substantivity_scores)),
                    "substantivity_std": float(np.std(substantivity_scores)),
                    "substantivity_num_valid": len(substantivity_scores),
                })
            else:
                metrics.update({
                    "substantivity_num_valid": 0,
                })

            # Add necessity metrics (only from valid samples)
            if necessity_scores:
                metrics.update({
                    "necessity_mean": float(np.mean(necessity_scores)),
                    "necessity_std": float(np.std(necessity_scores)),
                    "necessity_num_valid": len(necessity_scores),
                })
            else:
                metrics.update({
                    "necessity_num_valid": 0,
                })

            # Add paraphrasability metrics (only from valid samples)
            if paraphrasability_scores:
                metrics.update({
                    "paraphrasability_mean": float(np.mean(paraphrasability_scores)),
                    "paraphrasability_std": float(np.std(paraphrasability_scores)),
                    "paraphrasability_num_valid": len(paraphrasability_scores),
                })
            else:
                metrics.update({
                    "paraphrasability_num_valid": 0,
                })

            # Add accuracy metrics with more detail
            if accuracy_results:
                accuracy = np.mean(accuracy_results)
                metrics.update({
                    "accuracy": float(accuracy),
                    "accuracy_mean": float(np.mean(accuracy_results)),
                    "accuracy_std": float(np.std(accuracy_results)) if len(accuracy_results) > 1 else 0.0,
                    "num_correct": int(np.sum(accuracy_results)),
                    "num_total": len(accuracy_results),
                    "num_ground_truth_available": len(self.ground_truth)
                })

                # Add breakdown by correctness type
                if accuracy_details:
                    exact_matches = sum(1 for d in accuracy_details if d["is_equal"])
                    contains_matches = sum(1 for d in accuracy_details if d["contains_answer"] and not d["is_equal"])
                    metrics.update({
                        "num_exact_matches": exact_matches,
                        "num_contains_matches": contains_matches,
                        "accuracy_exact": float(exact_matches / len(accuracy_details)) if accuracy_details else 0.0,
                        "accuracy_contains": float(contains_matches / len(accuracy_details)) if accuracy_details else 0.0
                    })
            else:
                logging.warning("[Evaluator] No accuracy results calculated - check ground truth availability")
                metrics.update({
                    "accuracy": 0.0,
                    "num_correct": 0,
                    "num_total": 0,
                    "num_ground_truth_available": len(self.ground_truth),
                    "accuracy_warning": "No accuracy calculated - check ground truth"
                })

            # Add sample CoTs
            metrics["sample_cots"] = sample_cots

            # Add per-sample scores
            metrics["per_sample_scores"] = per_sample_data

            # Save metrics to checkpoint
            self._save_checkpoint_metrics(save_checkpoint_dir, metrics)

            # Add to history
            self.metrics_history.append(metrics)

            # Log summary
            logging.info(f"[Evaluator] Step {step} Summary:")
            sub_mean = metrics.get('substantivity_mean')
            sub_str = f"{sub_mean:.4f}" if sub_mean is not None else "N/A"
            logging.info(f"  - Substantivity: {sub_str} "
                        f"(std={metrics.get('substantivity_std', 0):.4f}, n={metrics.get('substantivity_num_valid', 0)})")
            nec_mean = metrics.get('necessity_mean')
            nec_str = f"{nec_mean:.4f}" if nec_mean is not None else "N/A"
            logging.info(f"  - Necessity: {nec_str} "
                        f"(std={metrics.get('necessity_std', 0):.4f}, n={metrics.get('necessity_num_valid', 0)})")
            para_mean = metrics.get('paraphrasability_mean')
            para_str = f"{para_mean:.4f}" if para_mean is not None else "N/A"
            logging.info(f"  - Paraphrasability: {para_str} "
                        f"(std={metrics.get('paraphrasability_std', 0):.4f}, n={metrics.get('paraphrasability_num_valid', 0)})")
            logging.info(f"  - Accuracy: {metrics.get('accuracy', 0):.4f} "
                        f"({metrics.get('num_correct', 0)}/{metrics.get('num_total', 0)})")
            if 'num_exact_matches' in metrics:
                logging.info(f"    - Exact matches: {metrics['num_exact_matches']}")
                logging.info(f"    - Contains matches: {metrics['num_contains_matches']}")
            logging.info(f"  - Evaluation time: {elapsed_time:.2f}s")
            logging.info(f"  - Throughput: {len(samples_to_process) / elapsed_time:.2f} samples/sec")

            return metrics

        except Exception as e:
            logging.error(f"[Evaluator] Error during evaluation: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {"step": step, "error": str(e)}

    def _save_checkpoint_metrics(self, checkpoint_dir: str, metrics: Dict):
        """Save metrics to checkpoint directory."""
        try:
            # Ensure directory exists
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save main metrics file
            metrics_file = os.path.join(checkpoint_dir, "eval_metrics.json")
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logging.info(f"[Evaluator] Saved metrics to {metrics_file}")

            # Save sample CoTs separately for easier inspection
            if "sample_cots" in metrics and metrics["sample_cots"]:
                samples_file = os.path.join(checkpoint_dir, "sample_cots.json")
                with open(samples_file, 'w') as f:
                    json.dump(metrics["sample_cots"], f, indent=2)
                logging.info(f"[Evaluator] Saved sample CoTs to {samples_file}")

            # Save per-sample scores separately (JSON and CSV)
            if "per_sample_scores" in metrics and metrics["per_sample_scores"]:
                per_sample_file = os.path.join(checkpoint_dir, "per_sample_scores.json")
                with open(per_sample_file, 'w') as f:
                    json.dump(metrics["per_sample_scores"], f, indent=2)
                logging.info(f"[Evaluator] Saved per-sample scores to {per_sample_file}")

                # Also save as CSV for easier analysis
                try:
                    import pandas as pd
                    df = pd.DataFrame(metrics["per_sample_scores"])
                    csv_file = os.path.join(checkpoint_dir, "per_sample_scores.csv")
                    df.to_csv(csv_file, index=False)
                    logging.info(f"[Evaluator] Saved per-sample scores CSV to {csv_file}")
                except Exception as e:
                    logging.warning(f"[Evaluator] Could not save per-sample scores as CSV: {e}")

        except Exception as e:
            logging.error(f"[Evaluator] Error saving metrics: {e}")

    def save_history(self):
        """Save complete metrics history."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            history_file = os.path.join(self.output_dir, "metrics_history.json")
            with open(history_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            logging.info(f"[Evaluator] Saved metrics history to {history_file}")

            # Also save a summary CSV for easier analysis
            if self.metrics_history:
                import pandas as pd
                # Extract key metrics for summary
                summary_data = []
                for m in self.metrics_history:
                    summary_data.append({
                        "step": m.get("step"),
                        "training_type": m.get("training_type", "unknown"),
                        "accuracy": m.get("accuracy", 0),
                        "substantivity_mean": m.get("substantivity_mean", 0),
                        "substantivity_std": m.get("substantivity_std", 0),
                        "necessity_mean": m.get("necessity_mean", 0),
                        "necessity_std": m.get("necessity_std", 0),
                        "paraphrasability_mean": m.get("paraphrasability_mean", 0),
                        "paraphrasability_std": m.get("paraphrasability_std", 0),
                        "num_correct": m.get("num_correct", 0),
                        "num_total": m.get("num_total", 0),
                        "evaluation_time_seconds": m.get("evaluation_time_seconds", 0),
                    })

                df = pd.DataFrame(summary_data)
                csv_file = os.path.join(self.output_dir, "metrics_summary.csv")
                df.to_csv(csv_file, index=False)
                logging.info(f"[Evaluator] Saved metrics summary to {csv_file}")

        except Exception as e:
            logging.error(f"[Evaluator] Error saving history: {e}")