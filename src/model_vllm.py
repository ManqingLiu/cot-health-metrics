# src/model_vllm.py
"""
vLLM-backed model for faster inference and lower memory usage.
Provides a compatible interface with CoTModel for use in checkpoint evaluation.

Key optimizations (inspired by Obfuscation_Generalization eval.py):
1. Persistent vLLM engine - initialize once, reuse across checkpoints
2. LoRA adapter swapping - change adapters without reinitializing engine
3. Model merging support - merge LoRA adapters for maximum speed
4. Continuous batching - leverage vLLM's efficient batching

IMPORTANT: vLLM 0.8+ has V1 engine as default which can cause memory issues.
We force V0 engine by setting VLLM_USE_V1=0 and other mitigations.
"""
# CRITICAL: Disable vLLM V1 engine BEFORE importing vllm
# V1 engine has multiprocessing initialization bugs and memory allocation issues in vLLM 0.8+
# Must be set BEFORE vllm import and inherited by any subprocesses
import os
import sys

# Force V0 engine - set multiple ways to ensure it takes effect
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_USE_LEGACY_EXECUTOR"] = "1"  # Another env var that may help

# Also disable async output processing which can cause issues
os.environ["VLLM_DISABLE_ASYNC_OUTPUT_PROCESSOR"] = "1"

# Log the environment settings for debugging
_vllm_v1_setting = os.environ.get("VLLM_USE_V1", "not set")
print(f"[model_vllm.py] VLLM_USE_V1={_vllm_v1_setting} (before vllm import)", file=sys.stderr)

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import torch
import logging
import re
import shutil
import tempfile
import time

from src.config import ModelConfig
from src.model_factory import ModelComponentFactory
from src.token_utils import TokenUtils


@dataclass
class ModelResponse:
    """Response from model generation, compatible with src.model.ModelResponse"""
    question_id: Optional[str]
    question: str
    prompt: str
    cot: str
    answer: str
    raw_output: str

    def __post_init__(self):
        self.basic_pair = (self.cot, self.answer)


class VLLMTokenUtils:
    """Token utilities adapted for vLLM models."""
    
    def __init__(self, tokenizer, model_name: str):
        self.tokenizer = tokenizer
        self.model_name = model_name
    
    def decode_to_string(self, tokens: torch.Tensor, skip_special_tokens=True) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def encode_to_tensor(self, string: str, to_device=None) -> torch.Tensor:
        tokens = self.tokenizer.encode(string)
        tensor = torch.tensor([tokens])
        if to_device is not None:
            tensor = tensor.to(to_device)
        return tensor
    
    def _get_end_think_token(self, model) -> str:
        """Get the end think token for the model."""
        model_config = ModelConfig.get(model.model_name)
        if "end_think" in model_config:
            return model_config["end_think"]
        elif "fuzzy_end_think_list" in model_config:
            return model_config["fuzzy_end_think_list"][0]
        else:
            logging.warning(f"Model {model.model_name} missing CoT separator config, using default")
            return "</think>"
    
    def get_answer_log_probs_recalc(self, model, prompt: str, cot: str, prediction: str) -> torch.Tensor:
        """
        Get log probs for just the answer (prediction), given prompt+cot+prediction.
        Uses vLLM's prompt_logprobs feature for efficient computation.
        """
        end_think = self._get_end_think_token(model)
        if cot == "":
            text0 = prompt + end_think
        else:
            text0 = prompt + cot + end_think
        text = text0 + prediction
        
        # Use vLLM to get log probs
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,  # Get logprobs for prompt tokens
            temperature=0.0,
        )
        
        outputs = model.llm.generate([text], sampling_params, use_tqdm=False)
        
        if not outputs or not outputs[0].prompt_logprobs:
            logging.warning("No prompt logprobs returned from vLLM")
            return torch.tensor([0.0])
        
        # Get the number of tokens in text0 (prompt + cot + end_think)
        text0_tokens = self.tokenizer.encode(text0)
        skip_count = len(text0_tokens)
        
        # Extract log probs for the prediction tokens
        prompt_logprobs = outputs[0].prompt_logprobs
        
        # prompt_logprobs is a list where each element corresponds to a token
        # We want the log probs for tokens after skip_count
        prediction_logprobs = []
        for i in range(skip_count, len(prompt_logprobs)):
            if prompt_logprobs[i] is not None:
                # Get the log prob of the actual token
                token_logprobs = prompt_logprobs[i]
                if token_logprobs:
                    # Get the first (and usually only) value
                    logprob = list(token_logprobs.values())[0].logprob
                    prediction_logprobs.append(logprob)
        
        if not prediction_logprobs:
            return torch.tensor([0.0])
        
        return torch.tensor(prediction_logprobs)


class VLLMCoTModel:
    """
    vLLM-backed model for faster inference and lower memory usage.
    Provides a compatible interface with CoTModel for use in checkpoint evaluation.
    
    Key benefits of vLLM:
    - PagedAttention: ~2-4x memory efficiency for KV cache
    - Continuous batching: Better GPU utilization and throughput
    - Optimized CUDA kernels: Faster inference
    - Native LoRA support: Efficient adapter loading
    """
    
    def __init__(self, model_name: str,
                 cache_dir: str = "/tmp/cache",
                 adapter_path: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 max_lora_rank: int = 64,
                 dtype: str = "auto"):
        """
        Initialize vLLM model with optional LoRA adapter.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen3-4B")
            cache_dir: Directory for model cache
            adapter_path: Path to LoRA adapter (checkpoint directory)
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
            max_lora_rank: Maximum LoRA rank to support
            dtype: Data type ("auto", "bfloat16", "float16", etc.)
        """
        self.model_name = model_name
        self.adapter_path = adapter_path
        self.cache_dir = cache_dir
        
        logging.info(f"[VLLMCoTModel] Initializing vLLM with model: {model_name}")
        logging.info(f"[VLLMCoTModel] GPU memory utilization: {gpu_memory_utilization}")
        logging.info(f"[VLLMCoTModel] LoRA adapter: {adapter_path}")

        # Disable vLLM V1 engine if environment variable is set
        # V1 engine can have initialization issues
        import os
        use_v1 = os.environ.get("VLLM_USE_V1", "0")
        if use_v1 == "0":
            logging.info(f"[VLLMCoTModel] Using legacy vLLM engine (VLLM_USE_V1=0)")

        # vLLM initialization with stability-focused settings
        try:
            self.llm = LLM(
                model=model_name,
                download_dir=cache_dir,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=adapter_path is not None,
                max_lora_rank=max_lora_rank,
                dtype=dtype,
                trust_remote_code=True,
                enforce_eager=True,  # Disable CUDA graphs for stability
                max_model_len=8192,  # Increased to handle prompts up to 8k tokens
            )
            logging.info(f"[VLLMCoTModel] vLLM initialized successfully")
        except Exception as e:
            logging.error(f"[VLLMCoTModel] Failed to initialize vLLM: {e}")
            logging.error(f"[VLLMCoTModel] Try setting VLLM_USE_V1=0 or reducing gpu_memory_utilization")
            raise
        
        self.tokenizer = self.llm.get_tokenizer()
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create LoRA request if adapter provided
        self.lora_request = None
        if adapter_path:
            self.lora_request = LoRARequest(
                lora_name="checkpoint_adapter",
                lora_int_id=1,
                lora_path=adapter_path
            )
            logging.info(f"[VLLMCoTModel] Created LoRA request for adapter: {adapter_path}")
        
        # Initialize component factory for prompt building
        self.component_factory = ModelComponentFactory(model_name)
        
        # Initialize token utils
        self.utils = VLLMTokenUtils(self.tokenizer, model_name)
    
    def get_utils(self) -> VLLMTokenUtils:
        """Get token utilities."""
        return self.utils
    
    def make_prompt(self, question_id, question: str, ground_truth_answer=None, 
                   custom_instruction: str = None) -> str:
        """Build a prompt for generation, compatible with CoTModel interface."""
        prompt_builder = self.component_factory.make_prompt_builder(invokes_cot=True)
        prompt_builder.add_user_message(question, custom_instruction, ground_truth_answer)
        prompt_builder.add_cot_mode()
        return prompt_builder.make_prompt(self.tokenizer)
    
    def make_prompt_no_cot(self, question_id, question: str, ground_truth_answer=None) -> str:
        """Build a prompt without CoT mode."""
        prompt_builder = self.component_factory.make_prompt_builder(invokes_cot=False)
        prompt_builder.add_user_message(question, ground_truth_answer)
        return prompt_builder.make_prompt(self.tokenizer)
    
    def do_generate_batch(self, question_ids: List, prompts: List[str], 
                          max_new_tokens: int = 2049,
                          do_sample: bool = False,
                          temperature: float = None) -> 'VLLMGenerateOutput':
        """
        Generate responses for multiple prompts with vLLM's continuous batching.
        
        Args:
            question_ids: List of question identifiers
            prompts: List of prompts to generate from
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling (ignored, controlled by temperature)
            temperature: Sampling temperature (0.0 for greedy)
        
        Returns:
            VLLMGenerateOutput with sequences tensor for compatibility
        """
        # Handle temperature
        if temperature is None or temperature == 0:
            temperature = 0.0
            do_sample = False
        else:
            do_sample = True
        
        # Get model config for stop tokens
        model_config = ModelConfig.get(self.model_name)
        stop_tokens = []
        if "end_think" in model_config:
            # Don't stop at end_think - we want the full response including answer
            pass
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            stop=stop_tokens if stop_tokens else None,
        )
        
        # vLLM handles batching internally with continuous batching
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False
        )
        
        # Convert outputs to tensor format for compatibility with existing code
        # Build sequences by concatenating prompt tokens + generated tokens
        sequences = []
        for i, output in enumerate(outputs):
            prompt_tokens = self.tokenizer.encode(prompts[i])
            generated_tokens = list(output.outputs[0].token_ids)
            full_sequence = prompt_tokens + generated_tokens
            sequences.append(full_sequence)
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [pad_token_id] * (max_len - len(seq))
            padded_sequences.append(seq)
        
        sequences_tensor = torch.tensor(padded_sequences)
        
        return VLLMGenerateOutput(sequences=sequences_tensor, vllm_outputs=outputs)
    
    def do_split(self, sequences: torch.Tensor, prompt: str, expect_cot: bool = True) -> Tuple[str, str, str]:
        """
        Split the output into question, CoT, and answer.
        Compatible with CoTModel.do_split interface.
        """
        model_config = ModelConfig.get(self.model_name)
        
        # Decode the full sequence
        full = self.tokenizer.decode(sequences[0], skip_special_tokens=False)
        
        # Get prompt length to extract only generated portion
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)
        
        # Handle padding tokens at the start
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        
        num_padding = 0
        if pad_token_id is not None:
            sequence_tokens = sequences[0].tolist()
            for token_id in sequence_tokens:
                if token_id == pad_token_id:
                    num_padding += 1
                else:
                    break
        
        # Get generated text (after padding and prompt)
        generation_start = num_padding + prompt_length
        generated_tokens = sequences[0][generation_start:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Extract question from prompt
        question = prompt.strip()
        
        # Parse generated text to extract CoT and answer
        if "end_think" in model_config:
            end_think = model_config["end_think"]
            begin_think = model_config.get("begin_think", "<think>")
            
            # Remove begin_think from question if present at end
            if question.endswith(begin_think):
                question = question[:-len(begin_think)].strip()
            
            # Split on end_think
            parts = generated_text.split(end_think, 1)
            if len(parts) == 2:
                cot = parts[0].strip()
                answer = parts[1].strip()
            else:
                if expect_cot:
                    logging.warning(f"Could not split CoT/answer, no '{end_think}' found")
                cot = ""
                answer = generated_text.strip()
        
        elif "fuzzy_end_think_list" in model_config:
            end_think_list = model_config["fuzzy_end_think_list"]
            cot = ""
            answer = generated_text.strip()
            
            for end_think in end_think_list:
                parts = generated_text.split(end_think, 1)
                if len(parts) == 2:
                    cot = parts[0].strip()
                    answer = parts[1].strip()
                    break
            else:
                if expect_cot:
                    logging.warning(f"Could not split CoT/answer with fuzzy end thinks")
        else:
            cot = ""
            answer = generated_text.strip()
        
        return (question, cot, answer)
    
    def get_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for a sequence.
        Note: This is less efficient than using prompt_logprobs directly.
        """
        text = self.tokenizer.decode(sequences[0], skip_special_tokens=False)
        
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.0,
        )
        
        outputs = self.llm.generate([text], sampling_params, use_tqdm=False)
        
        if not outputs or not outputs[0].prompt_logprobs:
            return torch.zeros(sequences.shape[1], sequences.shape[1])
        
        # Build log probs tensor
        prompt_logprobs = outputs[0].prompt_logprobs
        vocab_size = len(self.tokenizer)
        seq_len = len(prompt_logprobs)
        
        # Create a sparse representation
        log_probs = torch.full((1, seq_len, vocab_size), float('-inf'))
        
        for i, token_logprobs in enumerate(prompt_logprobs):
            if token_logprobs is not None:
                for token_id, logprob_obj in token_logprobs.items():
                    if token_id < vocab_size:
                        log_probs[0, i, token_id] = logprob_obj.logprob
        
        return log_probs
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID for a token string."""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            logging.warning(f"Token '{token}' not found in vocabulary")
            return self.tokenizer.unk_token_id or 0
        return token_id
    
    def get_think_tokens(self) -> Tuple[List[int], List[int]]:
        """Get think token IDs."""
        model_config = ModelConfig.get(self.model_name)
        
        if "begin_think" in model_config:
            begin_think_text = model_config["begin_think"]
            end_think_text = model_config["end_think"]
            
            begin_think_tokens = self.tokenizer.encode(begin_think_text, add_special_tokens=False)
            end_think_tokens = self.tokenizer.encode(end_think_text, add_special_tokens=False)
            
            return (begin_think_tokens, end_think_tokens)
        
        elif "fuzzy_end_think_list" in model_config:
            end_think_list = model_config["fuzzy_end_think_list"]
            end_think_tokens = []
            for end_think in end_think_list:
                tokens = self.tokenizer.encode(end_think, add_special_tokens=False)
                end_think_tokens.extend(tokens)
            return ([], end_think_tokens)
        
        return ([], [])


@dataclass
class VLLMGenerateOutput:
    """Output container compatible with HuggingFace generate output."""
    sequences: torch.Tensor
    vllm_outputs: List  # Original vLLM outputs for additional info


def merge_peft_adapter(base_model_name: str, adapter_path: str, output_path: str, 
                       cache_dir: str = None, dtype: str = "float16") -> str:
    """
    Merge a PEFT/LoRA adapter with the base model for faster inference.
    
    This approach eliminates LoRA computation overhead during inference,
    trading disk space for speed. Inspired by Obfuscation_Generalization eval.py.
    
    Args:
        base_model_name: HuggingFace model name (e.g., "Qwen/Qwen3-4B")
        adapter_path: Path to LoRA adapter checkpoint
        output_path: Where to save the merged model
        cache_dir: Model cache directory
        dtype: Data type for model weights
    
    Returns:
        Path to merged model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    logging.info(f"[merge_peft_adapter] Merging adapter from {adapter_path}")
    start_time = time.time()
    
    # Determine torch dtype
    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16 if dtype == "bfloat16" else torch.float32
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load adapter and merge
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    
    # Save tokenizer (try adapter path first, fall back to base model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=cache_dir)
    tokenizer.save_pretrained(output_path)
    
    # Cleanup to free memory
    del base_model
    del model
    del merged_model
    torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    logging.info(f"[merge_peft_adapter] Merged model saved to {output_path} in {elapsed:.2f}s")
    return output_path


class VLLMPersistentEngine:
    """
    Persistent vLLM engine that stays alive across multiple checkpoint evaluations.
    
    Key features:
    - Initialize once, reuse for all checkpoints
    - Swap LoRA adapters without reinitializing the engine
    - Optional model merging for maximum inference speed
    - Efficient memory management
    
    Usage:
        # Create engine once at start of training
        engine = VLLMPersistentEngine(
            model_name="Qwen/Qwen3-4B",
            enable_lora=True,  # Set True if you'll use LoRA adapters
        )
        
        # For each checkpoint evaluation
        engine.set_adapter(adapter_path)  # Swap adapter
        responses = engine.generate_batch(prompts)
        
        # For base model (no adapter)
        engine.clear_adapter()
        responses = engine.generate_batch(prompts)
        
        # Cleanup at end
        engine.cleanup()
    """
    
    _instance = None  # Singleton instance
    
    def __init__(self,
                 model_name: str,
                 cache_dir: str = "/tmp/cache",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 max_lora_rank: int = 64,
                 enable_lora: bool = True,
                 dtype: str = "auto",
                 max_model_len: int = 8192,
                 enforce_eager: bool = True):
        """
        Initialize the persistent vLLM engine.

        Args:
            model_name: HuggingFace model name
            cache_dir: Model cache directory
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: GPU memory fraction (0.0-1.0)
            max_lora_rank: Maximum LoRA rank to support
            enable_lora: Whether to enable LoRA adapter support
            dtype: Data type ("auto", "float16", "bfloat16")
            max_model_len: Maximum sequence length
            enforce_eager: Disable CUDA graphs for stability
        """
        self.model_name = model_name
        # Convert cache_dir to absolute path to avoid issues with vLLM subprocess
        # which may have a different working directory
        self.cache_dir = os.path.abspath(cache_dir) if cache_dir else None
        self.enable_lora = enable_lora
        self.max_lora_rank = max_lora_rank
        self._current_adapter_path = None
        self._lora_request = None
        self._temp_dirs = []  # Track temp directories for cleanup

        logging.info(f"[VLLMPersistentEngine] Initializing with model: {model_name}")
        logging.info(f"[VLLMPersistentEngine] GPU memory utilization: {gpu_memory_utilization}")
        logging.info(f"[VLLMPersistentEngine] LoRA enabled: {enable_lora}")
        logging.info(f"[VLLMPersistentEngine] Cache directory: {self.cache_dir}")

        # Check environment settings
        use_v1 = os.environ.get("VLLM_USE_V1", "0")
        if use_v1 == "0":
            logging.info(f"[VLLMPersistentEngine] Using legacy vLLM engine (VLLM_USE_V1=0)")

        start_time = time.time()

        try:
            self.llm = LLM(
                model=model_name,
                download_dir=self.cache_dir,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                enable_lora=enable_lora,
                max_lora_rank=max_lora_rank if enable_lora else None,
                dtype=dtype,
                trust_remote_code=True,
                enforce_eager=enforce_eager,
                max_model_len=min(max_model_len, 8192),  # Cap at 8192 to handle longer prompts
            )
            elapsed = time.time() - start_time
            logging.info(f"[VLLMPersistentEngine] vLLM initialized in {elapsed:.2f}s")
        except Exception as e:
            logging.error(f"[VLLMPersistentEngine] Failed to initialize vLLM: {e}")
            raise
        
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize component factory for prompt building
        self.component_factory = ModelComponentFactory(model_name)
        
        # Token utilities
        self.utils = VLLMTokenUtils(self.tokenizer, model_name)
    
    @classmethod
    def get_instance(cls, model_name: str, **kwargs) -> 'VLLMPersistentEngine':
        """
        Get or create the singleton vLLM engine instance.
        
        This ensures only one vLLM engine exists, saving significant memory
        and initialization time across multiple checkpoint evaluations.
        """
        if cls._instance is None or cls._instance.model_name != model_name:
            if cls._instance is not None:
                cls._instance.cleanup()
            cls._instance = cls(model_name, **kwargs)
        return cls._instance
    
    def set_adapter(self, adapter_path: str, adapter_name: str = "checkpoint_adapter") -> None:
        """
        Set the LoRA adapter for subsequent generations.
        
        Args:
            adapter_path: Path to LoRA adapter checkpoint
            adapter_name: Name for the adapter (for tracking)
        """
        if not self.enable_lora:
            raise RuntimeError("LoRA not enabled. Initialize engine with enable_lora=True")
        
        if adapter_path == self._current_adapter_path:
            logging.debug(f"[VLLMPersistentEngine] Adapter already loaded: {adapter_path}")
            return
        
        self._current_adapter_path = adapter_path
        self._lora_request = LoRARequest(
            lora_name=adapter_name,
            lora_int_id=1,
            lora_path=adapter_path
        )
        logging.info(f"[VLLMPersistentEngine] Adapter set: {adapter_path}")
    
    def clear_adapter(self) -> None:
        """Clear the current LoRA adapter (use base model)."""
        self._current_adapter_path = None
        self._lora_request = None
        logging.info(f"[VLLMPersistentEngine] Adapter cleared, using base model")
    
    def make_prompt(self, question_id, question: str, ground_truth_answer=None,
                    custom_instruction: str = None) -> str:
        """Build a prompt for generation, compatible with CoTModel interface."""
        prompt_builder = self.component_factory.make_prompt_builder(invokes_cot=True)
        prompt_builder.add_user_message(question, custom_instruction, ground_truth_answer)
        prompt_builder.add_cot_mode()
        return prompt_builder.make_prompt(self.tokenizer)
    
    def make_prompt_no_cot(self, question_id, question: str, ground_truth_answer=None) -> str:
        """Build a prompt without CoT mode (needed by necessity metric)."""
        prompt_builder = self.component_factory.make_prompt_builder(invokes_cot=False)
        prompt_builder.add_user_message(question, ground_truth_answer)
        return prompt_builder.make_prompt(self.tokenizer)
    
    def generate_batch(self, prompts: List[str],
                       max_new_tokens: int = 2049,
                       temperature: float = None,
                       stop_tokens: List[str] = None) -> List[str]:
        """
        Generate responses for multiple prompts using vLLM's continuous batching.
        
        Args:
            prompts: List of formatted prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 or None for greedy)
            stop_tokens: Optional list of stop sequences
        
        Returns:
            List of generated texts (full responses)
        """
        # Set up sampling parameters
        if temperature is None or temperature == 0:
            temperature = 0.0
            do_sample = False
        else:
            do_sample = True
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            stop=stop_tokens,
        )
        
        # Generate with optional LoRA adapter
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False
        )
        
        # Extract generated text
        responses = []
        for output in outputs:
            generated_text = output.outputs[0].text
            responses.append(generated_text)
        
        return responses
    
    def generate_batch_with_sequences(self, question_ids: List, prompts: List[str],
                                       max_new_tokens: int = 2049,
                                       temperature: float = None) -> VLLMGenerateOutput:
        """
        Generate responses and return with sequence tensors (for compatibility).
        
        This matches the interface of VLLMCoTModel.do_generate_batch for backward
        compatibility with existing code.
        """
        # Set up sampling parameters
        if temperature is None or temperature == 0:
            temperature = 0.0
        
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Generate
        outputs = self.llm.generate(
            prompts,
            sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False
        )
        
        # Convert to tensor format for compatibility
        sequences = []
        for i, output in enumerate(outputs):
            prompt_tokens = self.tokenizer.encode(prompts[i])
            generated_tokens = list(output.outputs[0].token_ids)
            full_sequence = prompt_tokens + generated_tokens
            sequences.append(full_sequence)
        
        # Pad sequences
        max_len = max(len(seq) for seq in sequences)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                seq = seq + [pad_token_id] * (max_len - len(seq))
            padded_sequences.append(seq)
        
        sequences_tensor = torch.tensor(padded_sequences)
        
        return VLLMGenerateOutput(sequences=sequences_tensor, vllm_outputs=outputs)
    
    def do_split(self, sequences: torch.Tensor, prompt: str, expect_cot: bool = True) -> Tuple[str, str, str]:
        """
        Split the output into question, CoT, and answer.
        Compatible with VLLMCoTModel.do_split interface.
        """
        model_config = ModelConfig.get(self.model_name)
        
        # Get prompt length
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_length = len(prompt_tokens)
        
        # Handle padding
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        
        num_padding = 0
        if pad_token_id is not None:
            sequence_tokens = sequences[0].tolist()
            for token_id in sequence_tokens:
                if token_id == pad_token_id:
                    num_padding += 1
                else:
                    break
        
        # Extract generated text
        generation_start = num_padding + prompt_length
        generated_tokens = sequences[0][generation_start:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        question = prompt.strip()
        
        # Parse CoT and answer
        if "end_think" in model_config:
            end_think = model_config["end_think"]
            begin_think = model_config.get("begin_think", "<think>")
            
            if question.endswith(begin_think):
                question = question[:-len(begin_think)].strip()
            
            parts = generated_text.split(end_think, 1)
            if len(parts) == 2:
                cot = parts[0].strip()
                answer = parts[1].strip()
            else:
                if expect_cot:
                    logging.warning(f"Could not split CoT/answer, no '{end_think}' found")
                cot = ""
                answer = generated_text.strip()
        
        elif "fuzzy_end_think_list" in model_config:
            end_think_list = model_config["fuzzy_end_think_list"]
            cot = ""
            answer = generated_text.strip()
            
            for end_think in end_think_list:
                parts = generated_text.split(end_think, 1)
                if len(parts) == 2:
                    cot = parts[0].strip()
                    answer = parts[1].strip()
                    break
            else:
                if expect_cot:
                    logging.warning(f"Could not split CoT/answer with fuzzy end thinks")
        else:
            cot = ""
            answer = generated_text.strip()
        
        return (question, cot, answer)
    
    def get_utils(self) -> 'VLLMTokenUtils':
        """Get token utilities."""
        return self.utils
    
    def get_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Get log probabilities for a sequence.
        Note: This is less efficient than using prompt_logprobs directly.
        """
        text = self.tokenizer.decode(sequences[0], skip_special_tokens=False)
        
        sampling_params = SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.0,
        )
        
        outputs = self.llm.generate([text], sampling_params, use_tqdm=False)
        
        if not outputs or not outputs[0].prompt_logprobs:
            return torch.zeros(sequences.shape[1], sequences.shape[1])
        
        # Build log probs tensor
        prompt_logprobs = outputs[0].prompt_logprobs
        vocab_size = len(self.tokenizer)
        seq_len = len(prompt_logprobs)
        
        # Create a sparse representation
        log_probs = torch.full((1, seq_len, vocab_size), float('-inf'))
        
        for i, token_logprobs in enumerate(prompt_logprobs):
            if token_logprobs is not None:
                for token_id, logprob_obj in token_logprobs.items():
                    if token_id < vocab_size:
                        log_probs[0, i, token_id] = logprob_obj.logprob
        
        return log_probs
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID for a token string."""
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            logging.warning(f"Token '{token}' not found in vocabulary")
            return self.tokenizer.unk_token_id or 0
        return token_id
    
    def cleanup(self) -> None:
        """
        Clean up resources including temp directories and FREE GPU MEMORY.
        Call this when done with all evaluations.
        
        IMPORTANT: This method destroys the vLLM engine to free GPU memory,
        allowing the training model to be restored to GPU.
        
        vLLM spawns worker subprocesses that hold GPU memory. We need to:
        1. Delete the LLM object to trigger shutdown
        2. Force garbage collection multiple times
        3. Wait for subprocesses to terminate
        4. Clear CUDA cache
        """
        import gc
        
        logging.info("[VLLMPersistentEngine] Starting cleanup...")
        
        # Clean up temp directories
        for temp_dir in self._temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    logging.debug(f"[VLLMPersistentEngine] Cleaned up: {temp_dir}")
                except Exception as e:
                    logging.warning(f"[VLLMPersistentEngine] Failed to cleanup {temp_dir}: {e}")
        self._temp_dirs.clear()
        
        # Clear adapter
        self._lora_request = None
        self._current_adapter_path = None
        
        # CRITICAL: Destroy the vLLM engine to free GPU memory
        # This is necessary before restoring the training model to GPU
        if hasattr(self, 'llm') and self.llm is not None:
            logging.info("[VLLMPersistentEngine] Destroying vLLM engine to free GPU memory")
            try:
                # Store reference to delete
                llm_to_delete = self.llm
                self.llm = None
                
                # Delete the LLM object - this should trigger __del__ and cleanup
                del llm_to_delete
            except Exception as e:
                logging.warning(f"[VLLMPersistentEngine] Error destroying LLM: {e}")
        
        # Force garbage collection multiple times to ensure cleanup
        # vLLM subprocesses need time to terminate
        for i in range(3):
            gc.collect()
        
        # Wait a short time for vLLM worker processes to terminate
        # This is necessary because vLLM spawns subprocesses that hold GPU memory
        import time
        time.sleep(2)
        
        # Additional garbage collection after wait
        gc.collect()
        
        # Clear CUDA cache to actually free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Log memory after cleanup
            try:
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                total_mem = torch.cuda.mem_get_info()[1] / 1e9
                logging.info(f"[VLLMPersistentEngine] GPU memory after cleanup: {free_mem:.1f}/{total_mem:.1f} GB free")
            except Exception as e:
                logging.warning(f"[VLLMPersistentEngine] Could not get GPU memory info: {e}")
        
        # Clear singleton reference
        if VLLMPersistentEngine._instance is self:
            VLLMPersistentEngine._instance = None
        
        logging.info("[VLLMPersistentEngine] Cleanup complete")


class _DeviceProxy:
    """Proxy object that provides a .device attribute for tensor operations."""
    def __init__(self, device: str = "cuda"):
        self._device = device
    
    @property
    def device(self):
        return self._device


class VLLMCoTModelWrapper:
    """
    Wrapper that provides VLLMCoTModel-compatible interface using VLLMPersistentEngine.
    
    This allows existing code that uses VLLMCoTModel to work with the persistent engine
    without modifications. The wrapper handles adapter path and creates ModelResponse objects.
    """
    
    def __init__(self, engine: VLLMPersistentEngine, adapter_path: Optional[str] = None):
        """
        Create a wrapper around a persistent engine.
        
        Args:
            engine: The persistent vLLM engine
            adapter_path: Optional LoRA adapter path to use
        """
        self.engine = engine
        self.adapter_path = adapter_path
        
        # Set adapter if provided
        if adapter_path:
            engine.set_adapter(adapter_path)
        else:
            engine.clear_adapter()
        
        # Expose attributes for compatibility
        self.model_name = engine.model_name
        self.tokenizer = engine.tokenizer
        self.utils = engine.utils
        
        # Expose llm for vLLM operations (needed by metrics like necessity/paraphrasability)
        self.llm = engine.llm
        
        # Expose model proxy for .device access (needed by substantivity metric for tensor ops)
        self.model = _DeviceProxy(device="cuda")
        
        # Initialize component factory for prompt building
        self.component_factory = engine.component_factory
    
    def make_prompt(self, question_id, question: str, ground_truth_answer=None,
                    custom_instruction: str = None) -> str:
        """Build a prompt."""
        return self.engine.make_prompt(question_id, question, ground_truth_answer, custom_instruction)
    
    def make_prompt_no_cot(self, question_id, question: str, ground_truth_answer=None) -> str:
        """Build a prompt without CoT mode (needed by necessity metric)."""
        return self.engine.make_prompt_no_cot(question_id, question, ground_truth_answer)
    
    def do_generate_batch(self, question_ids: List, prompts: List[str],
                          max_new_tokens: int = 2049,
                          do_sample: bool = False,
                          temperature: float = None) -> VLLMGenerateOutput:
        """Generate responses for a batch."""
        return self.engine.generate_batch_with_sequences(question_ids, prompts, max_new_tokens, temperature)
    
    def do_split(self, sequences: torch.Tensor, prompt: str, expect_cot: bool = True) -> Tuple[str, str, str]:
        """Split output into question, CoT, answer."""
        return self.engine.do_split(sequences, prompt, expect_cot)
    
    def get_utils(self) -> VLLMTokenUtils:
        """Get token utilities."""
        return self.engine.get_utils()
    
    def get_log_probs(self, sequences: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for a sequence (needed by metrics)."""
        return self.engine.get_log_probs(sequences)
    
    def _get_token_id(self, token: str) -> int:
        """Get token ID for a token string (needed by substantivity metric)."""
        return self.engine._get_token_id(token)
    
    def generate_cot_response_full(self, question_id, question: str, ground_truth_answer=None,
                                   max_new_tokens: int = 2049, custom_instruction: str = None,
                                   do_sample: bool = False, temperature: float = None) -> ModelResponse:
        """
        Generate a single CoT response. Compatible with CoTModel.generate_cot_response_full.
        
        Args:
            question_id: Identifier for the question
            question: The question text
            ground_truth_answer: Optional ground truth answer (for post-hoc mode)
            max_new_tokens: Maximum tokens to generate
            custom_instruction: Optional custom instruction to append to prompt
            do_sample: Whether to use sampling (ignored, controlled by temperature)
            temperature: Sampling temperature (0 or None for greedy)
        
        Returns:
            ModelResponse with question, prompt, cot, answer, raw_output
        """
        prompt = self.make_prompt(question_id, question, ground_truth_answer, custom_instruction)
        
        # Generate using vLLM
        output = self.do_generate_batch([question_id], [prompt], max_new_tokens, do_sample, temperature)
        sequences = output.sequences
        
        raw_output = self.tokenizer.decode(sequences[0], skip_special_tokens=False)
        
        try:
            (_, cot, answer) = self.do_split(sequences, prompt)
        except Exception as e:
            logging.warning(f"Failed to split response for question {question_id}: {e}")
            cot = ""
            answer = raw_output
        
        return ModelResponse(
            question_id=question_id,
            question=question,
            prompt=prompt,
            cot=cot,
            answer=answer,
            raw_output=raw_output
        )
    
    def generate_cot_response_full_batch(self, question_ids: List, questions: List[str],
                                         ground_truth_answers: List[str] = None,
                                         max_new_tokens: int = 2049,
                                         custom_instruction: str = None,
                                         do_sample: bool = False,
                                         temperature: float = None) -> List[ModelResponse]:
        """
        Generate CoT responses for multiple questions in batch.
        Compatible with CoTModel.generate_cot_response_full_batch.
        
        Args:
            question_ids: List of question identifiers
            questions: List of question texts
            ground_truth_answers: Optional list of ground truth answers
            max_new_tokens: Maximum tokens to generate
            custom_instruction: Optional custom instruction to append to prompts
            do_sample: Whether to use sampling (ignored, controlled by temperature)
            temperature: Sampling temperature (0 or None for greedy)
        
        Returns:
            List of ModelResponse objects
        """
        # Validate inputs
        if not question_ids or not questions:
            raise ValueError("Empty question_ids or questions list provided")
        
        if len(question_ids) != len(questions):
            raise ValueError(f"Mismatch between question_ids ({len(question_ids)}) and questions ({len(questions)})")
        
        # Create prompts for all questions
        prompts = []
        for i, (qid, question) in enumerate(zip(question_ids, questions)):
            if not question or question.strip() == "":
                raise ValueError(f"Empty question for question_id {qid}")
            gt_answer = ground_truth_answers[i] if ground_truth_answers and i < len(ground_truth_answers) else None
            prompt = self.make_prompt(qid, question, ground_truth_answer=gt_answer,
                                      custom_instruction=custom_instruction)
            if not prompt or prompt.strip() == "":
                raise ValueError(f"Empty prompt generated for question_id {qid}")
            prompts.append(prompt)
        
        # Generate responses in batch using vLLM
        output = self.do_generate_batch(question_ids, prompts, max_new_tokens, do_sample, temperature)
        sequences = output.sequences
        
        # Process each response
        responses = []
        for i, (question_id, question, prompt) in enumerate(zip(question_ids, questions, prompts)):
            raw_output = self.tokenizer.decode(sequences[i], skip_special_tokens=False)
            
            try:
                (_, cot, answer) = self.do_split(sequences[i:i + 1], prompt)
                
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
                # Handle cases where splitting fails
                logging.warning(f"Failed to split response for question {question_id}: {e}")
                response = ModelResponse(
                    question_id=question_id,
                    question=question,
                    prompt=prompt,
                    cot="",
                    answer=raw_output,
                    raw_output=raw_output
                )
                responses.append(response)
        
        return responses
    
    def generate_cot_response(self, question_id, question: str, max_new_tokens: int = 2049,
                              do_sample: bool = True) -> Tuple[str, str]:
        """Generate CoT response and return (cot, answer) tuple."""
        response = self.generate_cot_response_full(question_id, question, max_new_tokens=max_new_tokens,
                                                   do_sample=do_sample)
        return response.basic_pair
    
    def generate_cot_response_batch(self, question_ids: List, questions: List[str],
                                    max_new_tokens: int = 2049) -> List[Tuple[str, str]]:
        """Generate CoT responses in batch and return list of (cot, answer) tuples."""
        responses = self.generate_cot_response_full_batch(question_ids, questions, max_new_tokens=max_new_tokens)
        return [response.basic_pair for response in responses]
