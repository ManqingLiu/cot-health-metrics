"""
Training callbacks for metric evaluation during SFT.
"""
import os
import logging
import math
from typing import List, Dict, Optional
from transformers import TrainerCallback
from checkpoint_evaluator import CheckpointEvaluator


class MetricTrackingCallback(TrainerCallback):
    """
    Callback to evaluate metrics at specified training intervals.
    Follows the pattern from nanochat for clean integration with HF Trainer.
    """

    def __init__(self, model_name: str, cache_dir: str, output_dir: str,
                 eval_dataset: List[Dict], dataset_name: str,
                 checkpoint_intervals: List[float], total_training_steps: int,
                 filler_type: str = "lorem_ipsum",
                 batch_size: int = 8,
                 max_eval_samples: int = 100,
                 training_type: str = "baseline",
                 codebook_path: str = None,
                 use_vllm: bool = False,
                 vllm_gpu_memory_util: float = 0.55,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_max_lora_rank: int = 64):
        """
        Args:
            model_name: Name of the base model
            cache_dir: Cache directory for models
            output_dir: Training output directory
            eval_dataset: Dataset for evaluation
            dataset_name: Name of the dataset being trained on
            checkpoint_intervals: List of fractions (0.2, 0.4, etc.) for checkpoints
            total_training_steps: Total number of training steps
            filler_type: Type of filler for substantivity metric and internalized prompt
            max_eval_samples: Max samples to evaluate per checkpoint
            training_type: Type of training (baseline, internalized, encoded, post-hoc)
            codebook_path: Path to codebook module (for encoded training type)
            use_vllm: Whether to use vLLM for evaluation
            vllm_gpu_memory_util: GPU memory utilization for vLLM
            vllm_tensor_parallel_size: Tensor parallel size for vLLM
            vllm_max_lora_rank: Max LoRA rank for vLLM
        """
        super().__init__()

        self.evaluator = CheckpointEvaluator(
            model_name=model_name,
            cache_dir=cache_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            max_samples=max_eval_samples,
            training_type=training_type,
            codebook_path=codebook_path,
            filler_type=filler_type,
            use_vllm=use_vllm,
            vllm_gpu_memory_util=vllm_gpu_memory_util,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_max_lora_rank=vllm_max_lora_rank
        )

        self.eval_dataset = eval_dataset
        self.filler_type = filler_type
        self.max_eval_samples = max_eval_samples
        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.training_type = training_type

        # Calculate checkpoint steps based on intervals
        self.checkpoint_steps = [
            int(interval * total_training_steps)
            for interval in checkpoint_intervals
        ]
        self.checkpoint_steps = sorted(set(self.checkpoint_steps))
        self.evaluated_steps = set()

        logging.info(f"[MetricCallback] Initialized with checkpoint steps: {self.checkpoint_steps}")
        logging.info(f"[MetricCallback] Total training steps: {total_training_steps}")
        logging.info(f"[MetricCallback] Training type: {training_type}")
        logging.info(f"[MetricCallback] Filler type: {filler_type}")
        if codebook_path:
            logging.info(f"[MetricCallback] Codebook path: {codebook_path}")

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        current_step = state.global_step

        # Check if this is a checkpoint step we haven't evaluated
        for checkpoint_step in self.checkpoint_steps:
            # Use a tolerance window for checking since steps might not be exact
            if abs(current_step - checkpoint_step) <= 1 and checkpoint_step not in self.evaluated_steps:
                logging.info(f"[MetricCallback] Checkpoint step reached: {current_step} (target: {checkpoint_step})")
                self.evaluated_steps.add(checkpoint_step)

                # Force checkpoint save
                control.should_save = True

                # Mark that we need to evaluate after save
                self.pending_evaluation = checkpoint_step
                break

        return control

    def _offload_model_to_cpu(self, trainer_model):
        """
        Offload training model to CPU to free GPU memory for vLLM evaluation.
        Pattern from Obfuscation_Generalization repo.
        """
        import torch
        import gc
        
        if trainer_model is not None and hasattr(trainer_model, 'to'):
            logging.info("[MetricCallback] Offloading training model to CPU for evaluation")
            trainer_model.to('cpu')
            torch.cuda.empty_cache()
            gc.collect()
            
            # Log memory after offload
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0] / 1e9
                total_mem = torch.cuda.mem_get_info()[1] / 1e9
                logging.info(f"[MetricCallback] GPU memory after offload: {free_mem:.1f}/{total_mem:.1f} GB free")
    
    def _restore_model_to_gpu(self, trainer_model):
        """Restore training model to GPU after evaluation."""
        import torch
        
        if trainer_model is not None and hasattr(trainer_model, 'to'):
            logging.info("[MetricCallback] Restoring training model to GPU")
            trainer_model.to('cuda')
            torch.cuda.empty_cache()

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        current_step = state.global_step
        
        # Get the training model for potential offloading (when using vLLM)
        trainer_model = kwargs.get('model', None)
        use_vllm = getattr(self.evaluator, 'use_vllm', False)

        # Check if we have a pending evaluation
        if hasattr(self, 'pending_evaluation'):
            eval_step = self.pending_evaluation
            delattr(self, 'pending_evaluation')

            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{current_step}")

            logging.info(f"[MetricCallback] Post-save evaluation at step {current_step} for target step {eval_step}")
            
            # Offload training model to CPU if using vLLM (frees GPU memory)
            if use_vllm:
                self._offload_model_to_cpu(trainer_model)
            
            try:
                metrics = self.evaluator.evaluate_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                    eval_dataset=self.eval_dataset,
                    filler_type=self.filler_type,
                    max_samples=self.max_eval_samples,
                    batch_size=self.batch_size,
                    training_type=self.training_type
                )

                # Log to wandb if available
                if metrics and not metrics.get("error"):
                    self._log_to_wandb(metrics, current_step)
            finally:
                # CRITICAL: Clean up vLLM BEFORE restoring training model to GPU
                # vLLM spawns subprocesses that hold GPU memory
                if use_vllm and hasattr(self.evaluator, '_vllm_engine') and self.evaluator._vllm_engine is not None:
                    logging.info("[MetricCallback] Cleaning up vLLM engine before restoring model")
                    self.evaluator._vllm_engine.cleanup()
                    self.evaluator._vllm_engine = None
                
                # Restore model to GPU after vLLM cleanup
                if use_vllm:
                    self._restore_model_to_gpu(trainer_model)

        # Also check if this matches any of our target checkpoints directly
        elif current_step in self.checkpoint_steps and current_step not in self.evaluated_steps:
            self.evaluated_steps.add(current_step)

            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{current_step}")

            logging.info(f"[MetricCallback] Direct post-save evaluation at step {current_step}")
            
            # Offload training model to CPU if using vLLM
            if use_vllm:
                self._offload_model_to_cpu(trainer_model)
            
            try:
                metrics = self.evaluator.evaluate_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=current_step,
                    eval_dataset=self.eval_dataset,
                    filler_type=self.filler_type,
                    max_samples=self.max_eval_samples,
                    batch_size=self.batch_size,
                    training_type=self.training_type
                )

                # Log to wandb if available
                if metrics and not metrics.get("error"):
                    self._log_to_wandb(metrics, current_step)
            finally:
                # CRITICAL: Clean up vLLM BEFORE restoring training model to GPU
                if use_vllm and hasattr(self.evaluator, '_vllm_engine') and self.evaluator._vllm_engine is not None:
                    logging.info("[MetricCallback] Cleaning up vLLM engine before restoring model")
                    self.evaluator._vllm_engine.cleanup()
                    self.evaluator._vllm_engine = None
                
                # Restore model to GPU after vLLM cleanup
                if use_vllm:
                    self._restore_model_to_gpu(trainer_model)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Save complete metrics history
        self.evaluator.save_history()
        
        # Get the training model for potential offloading
        trainer_model = kwargs.get('model', None)
        use_vllm = getattr(self.evaluator, 'use_vllm', False)

        # Final evaluation if not done
        final_step = state.global_step
        if final_step not in self.evaluated_steps:
            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{final_step}")
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = self.evaluator.output_dir  # Use final model directory

            if os.path.exists(checkpoint_dir):
                logging.info(f"[MetricCallback] Final evaluation at step {final_step}")
                
                # Offload training model to CPU if using vLLM
                if use_vllm:
                    self._offload_model_to_cpu(trainer_model)
                
                try:
                    metrics = self.evaluator.evaluate_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=final_step,
                        eval_dataset=self.eval_dataset,
                        filler_type=self.filler_type,
                        max_samples=self.max_eval_samples,
                        batch_size=self.batch_size,
                        training_type=self.training_type
                    )

                    if metrics and not metrics.get("error"):
                        self._log_to_wandb(metrics, final_step)
                finally:
                    # CRITICAL: Clean up vLLM BEFORE restoring training model to GPU
                    if use_vllm and hasattr(self.evaluator, '_vllm_engine') and self.evaluator._vllm_engine is not None:
                        logging.info("[MetricCallback] Cleaning up vLLM engine before restoring model")
                        self.evaluator._vllm_engine.cleanup()
                        self.evaluator._vllm_engine = None
                    
                    # Restore model (not strictly necessary at end, but clean)
                    if use_vllm:
                        self._restore_model_to_gpu(trainer_model)
        else:
            # No final evaluation needed, but still cleanup vLLM if it exists
            if use_vllm and hasattr(self.evaluator, '_vllm_engine') and self.evaluator._vllm_engine is not None:
                logging.info("[MetricCallback] Cleaning up vLLM engine at end of training")
                self.evaluator._vllm_engine.cleanup()
                self.evaluator._vllm_engine = None

        return control

    def _log_to_wandb(self, metrics: Dict, step: int):
        """Log metrics to wandb if available with mean, std, and confidence bands."""
        try:
            import wandb

            # Main metrics with mean, std, and confidence bands
            log_dict = {"step": step}

            # Substantivity metric - mean, std, and confidence bands
            if "substantivity_mean" in metrics:
                log_dict.update({
                    "eval/substantivity_mean": metrics["substantivity_mean"],
                    "eval/substantivity_std": metrics.get("substantivity_std", 0),
                    "eval/substantivity_median": metrics.get("substantivity_median", 0),
                    "eval/substantivity_q25": metrics.get("substantivity_q25", 0),
                    "eval/substantivity_q75": metrics.get("substantivity_q75", 0),
                    "eval/substantivity_min": metrics.get("substantivity_min", 0),
                    "eval/substantivity_max": metrics.get("substantivity_max", 0)
                })

            # Necessity metric - mean, std, and confidence bands
            if "necessity_mean" in metrics:
                log_dict.update({
                    "eval/necessity_mean": metrics["necessity_mean"],
                    "eval/necessity_std": metrics.get("necessity_std", 0),
                    "eval/necessity_median": metrics.get("necessity_median", 0),
                    "eval/necessity_q25": metrics.get("necessity_q25", 0),
                    "eval/necessity_q75": metrics.get("necessity_q75", 0),
                    "eval/necessity_min": metrics.get("necessity_min", 0),
                    "eval/necessity_max": metrics.get("necessity_max", 0)
                })

            # Paraphrasability metric - mean, std, and confidence bands
            if "paraphrasability_mean" in metrics:
                log_dict.update({
                    "eval/paraphrasability_mean": metrics["paraphrasability_mean"],
                    "eval/paraphrasability_std": metrics.get("paraphrasability_std", 0),
                    "eval/paraphrasability_median": metrics.get("paraphrasability_median", 0),
                    "eval/paraphrasability_q25": metrics.get("paraphrasability_q25", 0),
                    "eval/paraphrasability_q75": metrics.get("paraphrasability_q75", 0),
                    "eval/paraphrasability_min": metrics.get("paraphrasability_min", 0),
                    "eval/paraphrasability_max": metrics.get("paraphrasability_max", 0)
                })

            # Accuracy metric - mean, std, and details
            if "accuracy" in metrics:
                log_dict.update({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/accuracy_mean": metrics.get("accuracy_mean", metrics["accuracy"]),
                    "eval/accuracy_std": metrics.get("accuracy_std", 0),
                    "eval/accuracy_median": metrics.get("accuracy_median", 0),
                    "eval/num_correct": metrics.get("num_correct", 0),
                    "eval/num_total": metrics.get("num_total", 0)
                })

            wandb.log(log_dict)

            # Log confidence bands as custom charts for better visualization
            if "substantivity_mean" in metrics and "necessity_mean" in metrics and "paraphrasability_mean" in metrics:
                # Create custom chart data for confidence bands
                confidence_data = [[
                    step,
                    metrics.get("substantivity_mean", 0),
                    metrics.get("substantivity_std", 0),
                    metrics.get("substantivity_median", 0),
                    metrics.get("substantivity_q25", 0),
                    metrics.get("substantivity_q75", 0),
                    metrics.get("necessity_mean", 0),
                    metrics.get("necessity_std", 0),
                    metrics.get("necessity_median", 0),
                    metrics.get("necessity_q25", 0),
                    metrics.get("necessity_q75", 0),
                    metrics.get("paraphrasability_mean", 0),
                    metrics.get("paraphrasability_std", 0),
                    metrics.get("paraphrasability_median", 0),
                    metrics.get("paraphrasability_q25", 0),
                    metrics.get("paraphrasability_q75", 0),
                    metrics.get("accuracy", 0),
                    metrics.get("accuracy_std", 0)
                ]]

                confidence_table = wandb.Table(
                    columns=[
                        "step", 
                        "substantivity_mean", "substantivity_std", "substantivity_median", "substantivity_q25", "substantivity_q75",
                        "necessity_mean", "necessity_std", "necessity_median", "necessity_q25", "necessity_q75",
                        "paraphrasability_mean", "paraphrasability_std", "paraphrasability_median", "paraphrasability_q25", "paraphrasability_q75",
                        "accuracy", "accuracy_std"
                    ],
                    data=confidence_data
                )
                wandb.log({"eval/metrics_summary": confidence_table})

            # Log sample CoTs as a table with full prompt for visualization
            # This logs question, prompt, cot, and answer at each checkpoint
            if "sample_cots" in metrics and metrics["sample_cots"]:
                cot_data = []
                for sample in metrics["sample_cots"]:
                    cot_data.append([
                        step,
                        sample.get("question_id", 0),
                        sample.get("question", ""),
                        sample.get("prompt", ""),  # Full prompt for debugging
                        sample.get("cot", ""),
                        sample.get("answer", "")
                    ])

                cot_table = wandb.Table(
                    columns=["step", "question_id", "question", "prompt", "cot", "answer"],
                    data=cot_data
                )
                wandb.log({"eval/sample_cots": cot_table})
                
                logging.info(f"[MetricCallback] Logged {len(cot_data)} sample CoTs to W&B at step {step}")

        except ImportError:
            pass  # wandb not available
        except Exception as e:
            logging.warning(f"Could not log to wandb: {e}")


def calculate_checkpoint_intervals(num_train_epochs: float,
                                  train_dataset_size: int,
                                  batch_size: int,
                                  gradient_accumulation_steps: int,
                                  num_checkpoints: int = 5) -> tuple:
    """
    Calculate training steps and checkpoint intervals.

    Returns:
        (total_training_steps, checkpoint_intervals)
    """
    num_update_steps_per_epoch = math.ceil(
        train_dataset_size / (batch_size * gradient_accumulation_steps)
    )
    total_training_steps = int(num_update_steps_per_epoch * num_train_epochs)

    # Create evenly spaced checkpoint intervals
    checkpoint_intervals = [
        (i + 1) / num_checkpoints
        for i in range(num_checkpoints)
    ]

    return total_training_steps, checkpoint_intervals