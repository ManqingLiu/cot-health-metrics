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
                 batch_size: int = 12,
                 max_eval_samples: int = 100,
                 training_type: str = "baseline",
                 codebook_path: str = None,
                 use_vllm: bool = False,
                 vllm_gpu_memory_util: float = 0.55,
                 vllm_tensor_parallel_size: int = 1,
                 vllm_max_lora_rank: int = 64,
                 max_new_tokens: int = 1024,
                 evaluate_step_0: bool = True):
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
            evaluate_step_0: Whether to evaluate the base model at step 0 before training
        """
        super().__init__()

        self.model_name = model_name
        self.cache_dir = cache_dir
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
            vllm_max_lora_rank=vllm_max_lora_rank,
            max_new_tokens=max_new_tokens
        )

        self.eval_dataset = eval_dataset
        self.filler_type = filler_type
        self.max_eval_samples = max_eval_samples
        self.total_training_steps = total_training_steps
        self.batch_size = batch_size
        self.training_type = training_type
        self.evaluate_step_0 = evaluate_step_0
        self.step_0_evaluated = False  # Track if step 0 has been evaluated

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
        logging.info(f"[MetricCallback] Evaluate step 0 (pre-training baseline): {evaluate_step_0}")
        if codebook_path:
            logging.info(f"[MetricCallback] Codebook path: {codebook_path}")

    def on_train_begin(self, args, state, control, **kwargs):
        """
        Called at the beginning of training.
        Evaluates the base model at step 0 before any training occurs.
        """
        if not self.evaluate_step_0 or self.step_0_evaluated:
            return control
        
        logging.info("[MetricCallback] ========================================")
        logging.info("[MetricCallback] Step 0 Evaluation (Pre-training baseline)")
        logging.info("[MetricCallback] ========================================")
        
        # Get the training model for potential offloading
        trainer_model = kwargs.get('model', None)
        use_vllm = getattr(self.evaluator, 'use_vllm', False)
        
        # Create a checkpoint directory for step 0
        step_0_dir = os.path.join(self.evaluator.output_dir, "checkpoint-0")
        os.makedirs(step_0_dir, exist_ok=True)
        
        # Save the initial model state for step 0 evaluation
        if trainer_model is not None:
            logging.info(f"[MetricCallback] Saving initial model state to {step_0_dir}")
            try:
                # For PEFT models, save the adapter
                if hasattr(trainer_model, 'save_pretrained'):
                    trainer_model.save_pretrained(step_0_dir)
                    logging.info("[MetricCallback] Saved initial adapter to checkpoint-0")
            except Exception as e:
                logging.warning(f"[MetricCallback] Could not save initial model: {e}")
        
        # Offload training model to CPU if using vLLM
        if use_vllm:
            self._offload_model_to_cpu(trainer_model)
        
        try:
            # Evaluate at step 0 using the base model (no adapter yet)
            # Pass None as checkpoint_dir to use base model without adapter
            metrics = self.evaluator.evaluate_checkpoint(
                checkpoint_dir=step_0_dir,  # Use the saved checkpoint-0
                step=0,
                eval_dataset=self.eval_dataset,
                filler_type=self.filler_type,
                max_samples=self.max_eval_samples,
                batch_size=self.batch_size,
                training_type=self.training_type
            )
            
            self.step_0_evaluated = True
            self.evaluated_steps.add(0)
            
            # Log to wandb if available
            if metrics and not metrics.get("error"):
                self._log_to_wandb(metrics, 0)
                logging.info(f"[MetricCallback] Step 0 evaluation complete:")
                logging.info(f"  - Accuracy: {metrics.get('accuracy', 0):.4f}")
                logging.info(f"  - Substantivity: {metrics.get('substantivity_mean', 0):.4f}")
                logging.info(f"  - Necessity: {metrics.get('necessity_mean', 0):.4f}")
                logging.info(f"  - Paraphrasability: {metrics.get('paraphrasability_mean', 0):.4f}")
                
                # Save history immediately after step 0 so dashboard can see results
                # This allows monitoring before training completes
                self.evaluator.save_history()
            else:
                logging.warning(f"[MetricCallback] Step 0 evaluation returned error: {metrics.get('error', 'unknown')}")
                
        except Exception as e:
            logging.error(f"[MetricCallback] Step 0 evaluation failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
        finally:
            # Clean up vLLM if used
            if use_vllm and hasattr(self.evaluator, '_vllm_engine') and self.evaluator._vllm_engine is not None:
                logging.info("[MetricCallback] Cleaning up vLLM engine after step 0 evaluation")
                self.evaluator._vllm_engine.cleanup()
                self.evaluator._vllm_engine = None
            
            # Restore model to GPU
            if use_vllm:
                self._restore_model_to_gpu(trainer_model)
        
        logging.info("[MetricCallback] ========================================")
        return control

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
                    # Save history incrementally so dashboard can see results immediately
                    self.evaluator.save_history()
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
                        # Save history incrementally so dashboard can see results immediately
                        self.evaluator.save_history()
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
        """
        Log metrics to wandb if available.
        
        Only logs essential metrics:
        - accuracy_mean, accuracy_std
        - substantivity_mean, substantivity_std
        - necessity_mean, necessity_std
        - paraphrasability_mean, paraphrasability_std
        - eval_loss
        """
        try:
            import wandb

            log_dict = {"step": step}

            # Accuracy - mean and std only
            if "accuracy_mean" in metrics:
                log_dict["eval/accuracy_mean"] = metrics["accuracy_mean"]
            elif "accuracy" in metrics:
                log_dict["eval/accuracy_mean"] = metrics["accuracy"]
            if "accuracy_std" in metrics:
                log_dict["eval/accuracy_std"] = metrics["accuracy_std"]

            # Substantivity - mean and std only
            if "substantivity_mean" in metrics:
                log_dict["eval/substantivity_mean"] = metrics["substantivity_mean"]
            if "substantivity_std" in metrics:
                log_dict["eval/substantivity_std"] = metrics["substantivity_std"]

            # Necessity - mean and std only
            if "necessity_mean" in metrics:
                log_dict["eval/necessity_mean"] = metrics["necessity_mean"]
            if "necessity_std" in metrics:
                log_dict["eval/necessity_std"] = metrics["necessity_std"]

            # Paraphrasability - mean and std only
            if "paraphrasability_mean" in metrics:
                log_dict["eval/paraphrasability_mean"] = metrics["paraphrasability_mean"]
            if "paraphrasability_std" in metrics:
                log_dict["eval/paraphrasability_std"] = metrics["paraphrasability_std"]

            # Eval loss
            if "eval_loss" in metrics:
                log_dict["eval/loss"] = metrics["eval_loss"]

            wandb.log(log_dict)

            # Log sample CoTs as a table for visualization
            if "sample_cots" in metrics and metrics["sample_cots"]:
                cot_data = []
                for sample in metrics["sample_cots"]:
                    cot_data.append([
                        step,
                        sample.get("question_id", 0),
                        sample.get("question", ""),
                        sample.get("prompt", ""),
                        sample.get("cot", ""),
                        sample.get("answer", "")
                    ])

                cot_table = wandb.Table(
                    columns=["step", "question_id", "question", "prompt", "cot", "answer"],
                    data=cot_data
                )
                wandb.log({"eval/sample_cots": cot_table})

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