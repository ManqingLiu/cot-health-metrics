#!/usr/bin/env python3
"""
Enhanced SFT training script for internalized CoT.
Tracks multiple metrics and saves checkpoints at regular intervals.

Example usage:
python train_internalized_enhanced.py \
    --model Qwen/Qwen3-0.6B \
    --dataset_name binary_alteration \
    --output_dir output/binary-enhanced \
    --num_train_epochs 3 \
    --filler_type lorem_ipsum \
    --track_metrics \
    --wandb_project internalized-training
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

# Add src to path
# sys.path.insert(0, str(Path(__file__).parent))
# sys.path.insert(0, str(Path(__file__).parent / "src"))

from checkpoint_evaluator import CheckpointEvaluator
from training_callbacks import MetricTrackingCallback, calculate_checkpoint_intervals
from src.organism_data.data.dataset_preparation import (
    InternalizedDataset,
    load_dataset_for_training,
    create_data_collator
)


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
                        choices=["ba", "gsm8k", "theory_of_mind", "3sum", "leg_counting"],
                        help="Dataset to use for training")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples (train and eval)")

    # Training arguments
    parser.add_argument("--num_train_epochs", type=float, default=3, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_length", type=int, default=2048)

    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--lora_r", type=int, default=4, help="LoRA rank")
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
    parser.add_argument("--filler_type", type=str, default="lorem_ipsum",
                        choices=["lorem_ipsum", "dots", "think_token", "number_words", "mixed"],
                        help="Type of filler for internalized CoT")
    parser.add_argument("--mask_mode", type=str, default="cot_and_answer",
                        choices=["assistant", "cot", "answer_only", "cot_and_answer"],
                        help="What to mask during training")

    # Evaluation arguments
    parser.add_argument("--track_metrics", action="store_true",
                        help="Track metrics during training")
    parser.add_argument("--num_checkpoints", type=int, default=5,
                        help="Number of checkpoints to save (evenly spaced)")
    parser.add_argument("--metric_eval_samples", type=int, default=50,
                        help="Samples to evaluate for metrics")

    # Logging arguments
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project")
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--seed", type=int, default=42)

    # Early stopping
    parser.add_argument("--use_early_stopping", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--early_stopping_threshold", type=float, default=0.0)

    # Parse arguments
    args = parser.parse_args()

    # Setup
    setup_logging(args.log_level)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Saved training config to {config_path}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)

    # Load datasets
    logging.info(f"Loading {args.dataset_name} dataset")
    train_data = load_dataset_for_training(
        args.dataset_name,
        max_samples=args.max_samples,
        split="train"
    )
    eval_data = load_dataset_for_training(
        args.dataset_name,
        max_samples=args.max_samples,
        split="test"
    )

    # Create datasets
    train_dataset = InternalizedDataset(
        train_data, tokenizer,
        filler_type=args.filler_type,
        mask_mode=args.mask_mode,
        max_length=args.max_length
    )
    eval_dataset = InternalizedDataset(
        eval_data, tokenizer,
        filler_type=args.filler_type,
        mask_mode=args.mask_mode,
        max_length=args.max_length
    )

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

    # Add metric tracking callback if requested
    if args.track_metrics:
        metric_callback = MetricTrackingCallback(
            model_name=args.model,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            eval_dataset=eval_data,  # Use raw data for evaluation
            dataset_name=args.dataset_name,
            checkpoint_intervals=checkpoint_intervals,
            total_training_steps=total_steps,
            filler_type=args.filler_type,
            max_eval_samples=args.metric_eval_samples
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
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args)
            )
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