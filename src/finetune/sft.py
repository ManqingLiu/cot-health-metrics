#!/usr/bin/env python3
"""
Enhanced SFT training script for internalized CoT.
OPTIMIZED VERSION with:
1. Support for baseline training (original CoT)
2. CoT length optimization for faster training

Example usage:
# Baseline training (original CoT)
python sft_internalized_optimized.py \
    --model Qwen/Qwen3-0.6B \
    --dataset_name ba \
    --output_dir output/baseline-ba \
    --training_type baseline \
    --max_length 1024 \
    --max_cot_length 506

# Internalized training (with filler)
python sft_internalized_optimized.py \
    --model Qwen/Qwen3-0.6B \
    --dataset_name ba \
    --output_dir output/internalized-ba \
    --training_type internalized \
    --filler_type_train mixed \
    --max_cot_length 506
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback
)

from checkpoint_evaluator import CheckpointEvaluator
from training_callbacks import MetricTrackingCallback, calculate_checkpoint_intervals
from src.organism_data.data.dataset_preparation import (
    InternalizedDataset,
    EncodedDataset,
    BaselineDataset,
    PosthocDataset,
    create_data_collator
)
# Import DatasetConfig from config.py
from src.config import DatasetConfig


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_and_tokenizer(args):
    """Load model and tokenizer with optional LoRA."""
    logging.info(f"Loading model: {args.model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading kwargs
    model_kwargs = {
        "cache_dir": args.cache_dir,
        "torch_dtype": torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else torch.float32,
        "trust_remote_code": True
    }

    # Load with quantization if requested
    if args.load_in_4bit or args.load_in_8bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    # Setup LoRA if requested
    if args.use_lora:
        logging.info(f"Setting up LoRA with r={args.lora_r}, alpha={args.lora_alpha}")

        from peft import LoraConfig, get_peft_model, TaskType

        # Parse target modules
        target_modules = args.lora_target_modules.split(",")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            inference_mode=False
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Enhanced SFT training for internalized CoT")

    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--cache_dir", type=str, default="hf_cache", help="Cache directory")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset to use for training (e.g., ba, gsm8k, theory_of_mind, 3sum, leg_counting)")
    parser.add_argument("--training_type", type=str, default="internalized",
                        choices=["internalized", "encoded", "baseline", "post-hoc"],
                        help="Type of training: internalized (filler), encoded (codebook), baseline (original), or post-hoc")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (train and eval)")

    # NEW: CoT length optimization
    parser.add_argument("--max_cot_length", type=int, default=None,
                        help="Maximum CoT length in tokens (for speed optimization).")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank (lower = faster)")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                        help="Comma-separated list of target modules")

    # Model loading arguments
    parser.add_argument("--load_in_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--load_in_8bit", action="store_true", help="8-bit quantization")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")

    # Internalization arguments
    parser.add_argument("--filler_type_train", type=str, default="not_relevant",
                        choices=["lorem_ipsum", "dots", "think_token", "number_words", "mixed", "not_relevant", "shuffled"],
                        help="Type of filler for internalized CoT training (used when training_type=internalized)")
    parser.add_argument("--filler_type_eval", type=str, default="not_relevant",
                        choices=["lorem_ipsum", "dots", "think_token", "number_words", "mixed", "not_relevant", "shuffled"],
                        help="Type of filler for evaluation (used when generating eval dataset)")
    parser.add_argument("--codebook_path", type=str, default="src/finetune/codebook_binary_alternation.py",
                        help="Path to codebook module (used when training_type=encoded)")
    parser.add_argument("--mask_mode", type=str, default="cot_and_answer",
                        choices=["assistant", "cot", "answer_only", "cot_and_answer"],
                        help="What to mask during training")

    # Evaluation arguments
    parser.add_argument("--track_metrics", action="store_true",
                        help="Track metrics during training")
    parser.add_argument("--num_checkpoints", type=int, default=4,
                        help="Number of checkpoints to save (evenly spaced)")
    parser.add_argument("--metric_eval_samples", type=int, default=100,
                        help="Samples to evaluate for metrics")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for evaluation (optimized for Qwen3-4B on 80GB GPUs, default: 12, safe from OOM)")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Experiment tracking
    parser.add_argument("--wandb_project", type=str, default=None, help="Wandb project name")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for wandb")

    # Early stopping
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # vLLM arguments for faster evaluation
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for faster checkpoint evaluation")
    parser.add_argument("--vllm_gpu_memory_util", type=float, default=0.55,
                        help="GPU memory utilization for vLLM (0.0-1.0)")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--vllm_max_lora_rank", type=int, default=64,
                        help="Maximum LoRA rank for vLLM")
    parser.add_argument("--max_new_tokens", type=int, default=2049,
                        help="Maximum new tokens to generate during inference (default: 2049, optimized for longer responses)")

    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    logging.info("=" * 80)
    logging.info("Training Configuration")
    logging.info("=" * 80)
    for key, value in sorted(vars(args).items()):
        logging.info(f"  {key}: {value}")
    logging.info("=" * 80)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load dataset using DatasetConfig
    logging.info(f"Loading dataset: {args.dataset_name}")

    train_data = DatasetConfig.load_for_training(
        args.dataset_name,
        max_samples=args.max_samples,
        split="train"
    )
    eval_data = DatasetConfig.load_for_training(
        args.dataset_name,
        max_samples=args.max_samples,
        split="test"
    )

    logging.info(f"Loaded {len(train_data)} training samples")
    logging.info(f"Loaded {len(eval_data)} eval samples")

    # Create datasets based on training type
    if args.training_type == "baseline":
        logging.info(f"Creating BaselineDataset (original CoT format)")

        train_dataset = BaselineDataset(
            train_data, tokenizer,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )
        eval_dataset = BaselineDataset(
            eval_data, tokenizer,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )

    elif args.training_type == "internalized":
        logging.info(f"Creating InternalizedDataset with filler_type={args.filler_type_train} for training")
        logging.info(f"  and filler_type={args.filler_type_eval} for eval")

        train_dataset = InternalizedDataset(
            train_data, tokenizer,
            filler_type=args.filler_type_train,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model,
            dataset_name=args.dataset_name
        )
        eval_dataset = InternalizedDataset(
            eval_data, tokenizer,
            filler_type=args.filler_type_eval,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model,
            dataset_name=args.dataset_name
        )

    elif args.training_type == "encoded":
        if args.codebook_path is None and args.dataset_name not in ["ba", "binary_alternation","spell_backward","sb"]:
            raise ValueError(f"Encoded training requires --codebook_path for dataset {args.dataset_name}")

        logging.info(f"Creating EncodedDataset with codebook={args.codebook_path or 'default'}")

        train_dataset = EncodedDataset(
            train_data, tokenizer,
            codebook_path=args.codebook_path,
            dataset_name=args.dataset_name,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )
        eval_dataset = EncodedDataset(
            eval_data, tokenizer,
            codebook_path=args.codebook_path,
            dataset_name=args.dataset_name,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )
    elif args.training_type == "post-hoc":
        logging.info(f"Creating PosthocDataset with answer-first format")
        logging.info(f"  Format: Answer → Reasoning → Answer (post-hoc rationalization)")

        train_dataset = PosthocDataset(
            train_data, tokenizer,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )
        eval_dataset = PosthocDataset(
            eval_data, tokenizer,
            mask_mode=args.mask_mode,
            max_length=args.max_length,
            max_cot_length=args.max_cot_length,
            model_name=args.model
        )
    else:
        raise ValueError(f"Unknown training type: {args.training_type}")

    logging.info(f"Train dataset size: {len(train_dataset)}")
    logging.info(f"Eval dataset size: {len(eval_dataset)}")

    # Calculate training steps and checkpoint intervals
    total_steps, checkpoint_intervals = calculate_checkpoint_intervals(
        num_train_epochs=args.num_train_epochs,
        train_dataset_size=len(train_dataset),
        batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_checkpoints=args.num_checkpoints
    )

    logging.info(f"Total training steps: {total_steps}")
    logging.info(f"Checkpoint intervals: {checkpoint_intervals}")

    # Setup callbacks
    callbacks = []

    if args.track_metrics:
        metric_callback = MetricTrackingCallback(
            model_name=args.model,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            eval_dataset=eval_data,
            dataset_name=args.dataset_name,
            checkpoint_intervals=checkpoint_intervals,
            total_training_steps=total_steps,
            filler_type=args.filler_type_eval,
            max_eval_samples=args.metric_eval_samples,
            batch_size=args.batch_size,
            training_type=args.training_type,
            codebook_path=args.codebook_path if args.training_type == "encoded" else None,
            use_vllm=args.use_vllm,
            vllm_gpu_memory_util=args.vllm_gpu_memory_util,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_max_lora_rank=args.vllm_max_lora_rank,
            max_new_tokens=args.max_new_tokens,
            evaluate_step_0=True  # Evaluate base model before training starts
        )
        callbacks.append(metric_callback)

    # Add early stopping if requested
    if args.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=args.early_stopping_threshold
        ))

    # Initialize wandb if requested
    if args.wandb_project:
        try:
            import wandb
            import glob
            import shutil
            
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )
            
            # Save source code to run's Files section using wandb.save()
            try:
                # Get the project root directory
                project_root = Path(__file__).resolve().parent.parent.parent
                if not (project_root / "src").exists():
                    project_root = Path.cwd()
                
                logging.info(f"W&B: Saving source code from {project_root}")
                
                # Save all Python files in src/ directory
                src_dir = project_root / "src"
                if src_dir.exists():
                    # Use glob pattern to save all .py files
                    py_pattern = str(src_dir / "**" / "*.py")
                    py_files = glob.glob(py_pattern, recursive=True)
                    for py_file in py_files:
                        # Save with relative path preserved
                        wandb.save(py_file, base_path=str(project_root), policy="now")
                    logging.info(f"Saved {len(py_files)} Python files to W&B Files")
                
                # Save key shell scripts
                for script_name in ["run_parallel_gpu_lambda.sh", "sync_to_lambda.sh"]:
                    script_path = project_root / script_name
                    if script_path.exists():
                        wandb.save(str(script_path), base_path=str(project_root), policy="now")
                        logging.info(f"Saved {script_name} to W&B Files")
                
                # Save requirements.txt
                req_path = project_root / "requirements.txt"
                if req_path.exists():
                    wandb.save(str(req_path), base_path=str(project_root), policy="now")
                
                logging.info("Successfully saved source code to W&B Files")
            except Exception as e:
                import traceback
                logging.warning(f"Failed to save code to W&B: {e}")
                logging.warning(f"Traceback: {traceback.format_exc()}")
            
            report_to = "wandb"
        except ImportError:
            logging.warning("wandb not installed, skipping wandb logging")
            report_to = "none"
    else:
        report_to = "none"

    # Calculate save and eval steps based on checkpoint intervals
    if args.track_metrics:
        # Force save at each checkpoint interval
        save_steps = max(1, int(total_steps / args.num_checkpoints))
        eval_steps = save_steps  # Evaluate at same frequency
        save_total_limit = None  # Keep all checkpoints
        save_strategy = "steps"
        eval_strategy = "steps"
        logging.info(f"Checkpoint save/eval steps: {save_steps} (total steps: {total_steps})")
    else:
        save_steps = 500  # Default
        eval_steps = 500
        save_total_limit = 3
        save_strategy = "steps"
        eval_strategy = "steps" if len(eval_dataset) > 0 else "no"

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=save_total_limit,
        save_strategy=save_strategy,
        eval_strategy=eval_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=report_to,
        run_name=args.run_name,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_drop_last=False,
        ddp_find_unused_parameters=False if args.use_lora else None,
    )

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks
    )

    # Start training
    logging.info("Starting training...")
    logging.info(f"Training type: {args.training_type}")
    if args.max_cot_length:
        logging.info(f"CoT truncated to max {args.max_cot_length} characters for speed")

    train_result = trainer.train()

    # Save final model
    logging.info("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Final evaluation
    if len(eval_dataset) > 0:
        logging.info("Running final evaluation...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logging.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()