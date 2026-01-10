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
                 training_type: str = "baseline"):
        """
        Args:
            model_name: Name of the base model
            cache_dir: Cache directory for models
            output_dir: Training output directory
            eval_dataset: Dataset for evaluation
            dataset_name: Name of the dataset being trained on
            checkpoint_intervals: List of fractions (0.2, 0.4, etc.) for checkpoints
            total_training_steps: Total number of training steps
            filler_type: Type of filler for substantivity metric
            max_eval_samples: Max samples to evaluate per checkpoint
            training_type: Type of training (baseline, internalized, encoded, post-hoc)
        """
        super().__init__()

        self.evaluator = CheckpointEvaluator(
            model_name=model_name,
            cache_dir=cache_dir,
            output_dir=output_dir,
            dataset_name=dataset_name,
            max_samples=max_eval_samples,
            training_type=training_type
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

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        current_step = state.global_step

        # Check if we have a pending evaluation
        if hasattr(self, 'pending_evaluation'):
            eval_step = self.pending_evaluation
            delattr(self, 'pending_evaluation')

            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{current_step}")

            logging.info(f"[MetricCallback] Post-save evaluation at step {current_step} for target step {eval_step}")
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

        # Also check if this matches any of our target checkpoints directly
        elif current_step in self.checkpoint_steps and current_step not in self.evaluated_steps:
            self.evaluated_steps.add(current_step)

            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{current_step}")

            logging.info(f"[MetricCallback] Direct post-save evaluation at step {current_step}")
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

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Save complete metrics history
        self.evaluator.save_history()

        # Final evaluation if not done
        final_step = state.global_step
        if final_step not in self.evaluated_steps:
            checkpoint_dir = os.path.join(self.evaluator.output_dir, f"checkpoint-{final_step}")
            if not os.path.exists(checkpoint_dir):
                checkpoint_dir = self.evaluator.output_dir  # Use final model directory

            if os.path.exists(checkpoint_dir):
                logging.info(f"[MetricCallback] Final evaluation at step {final_step}")
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

        return control

    def _log_to_wandb(self, metrics: Dict, step: int):
        """Log metrics to wandb if available with confidence bands."""
        try:
            import wandb

            # Main metrics with confidence bands
            log_dict = {"step": step}

            # Substantivity metric with confidence bands
            if "substantivity_median" in metrics:
                log_dict.update({
                    "eval/substantivity_median": metrics["substantivity_median"],
                    "eval/substantivity_q25": metrics.get("substantivity_q25", 0),
                    "eval/substantivity_q75": metrics.get("substantivity_q75", 0),
                    "eval/substantivity_min": metrics.get("substantivity_min", 0),
                    "eval/substantivity_max": metrics.get("substantivity_max", 0)
                })

            # Necessity metric with confidence bands
            if "necessity_median" in metrics:
                log_dict.update({
                    "eval/necessity_median": metrics["necessity_median"],
                    "eval/necessity_q25": metrics.get("necessity_q25", 0),
                    "eval/necessity_q75": metrics.get("necessity_q75", 0),
                    "eval/necessity_min": metrics.get("necessity_min", 0),
                    "eval/necessity_max": metrics.get("necessity_max", 0)
                })

            # Paraphrasability metric with confidence bands
            if "paraphrasability_median" in metrics:
                log_dict.update({
                    "eval/paraphrasability_median": metrics["paraphrasability_median"],
                    "eval/paraphrasability_q25": metrics.get("paraphrasability_q25", 0),
                    "eval/paraphrasability_q75": metrics.get("paraphrasability_q75", 0),
                    "eval/paraphrasability_min": metrics.get("paraphrasability_min", 0),
                    "eval/paraphrasability_max": metrics.get("paraphrasability_max", 0)
                })

            # Accuracy metric
            if "accuracy" in metrics:
                log_dict.update({
                    "eval/accuracy": metrics["accuracy"],
                    "eval/accuracy_median": metrics.get("accuracy_median", 0),
                    "eval/num_correct": metrics.get("num_correct", 0),
                    "eval/num_total": metrics.get("num_total", 0)
                })

            wandb.log(log_dict)

            # Log confidence bands as custom charts for better visualization
            if "substantivity_median" in metrics and "necessity_median" in metrics and "paraphrasability_median" in metrics:
                # Create custom chart data for confidence bands
                confidence_data = [[
                    step,
                    metrics.get("substantivity_median", 0),
                    metrics.get("substantivity_q25", 0),
                    metrics.get("substantivity_q75", 0),
                    metrics.get("necessity_median", 0),
                    metrics.get("necessity_q25", 0),
                    metrics.get("necessity_q75", 0),
                    metrics.get("paraphrasability_median", 0),
                    metrics.get("paraphrasability_q25", 0),
                    metrics.get("paraphrasability_q75", 0),
                    metrics.get("accuracy", 0)
                ]]

                confidence_table = wandb.Table(
                    columns=[
                        "step", "substantivity_median", "substantivity_q25", "substantivity_q75",
                        "necessity_median", "necessity_q25", "necessity_q75",
                        "paraphrasability_median", "paraphrasability_q25", "paraphrasability_q75",
                        "accuracy"
                    ],
                    data=confidence_data
                )
                wandb.log({"eval/confidence_bands": confidence_table})

            # Log sample CoTs as a table with full prompt for Anthropic-style visualization
            if "sample_cots" in metrics and metrics["sample_cots"]:
                cot_data = []
                for sample in metrics["sample_cots"]:
                    cot_data.append([
                        step,
                        sample.get("question_id", 0),
                        sample.get("question", ""),
                        sample.get("prompt", ""),  # Include full prompt for visualization
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
            logging.debug(f"Could not log to wandb: {e}")


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