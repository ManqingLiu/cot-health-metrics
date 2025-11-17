"""
Checkpoint evaluation module for tracking metrics during training.
Handles evaluation of substantivity, necessity, paraphrasability metrics and accuracy.
"""
import os
import json
import time
import logging
import numpy as np
from typing import List, Dict, Optional
from types import SimpleNamespace

from src.model import CoTModel
from src.metric_substantivity import SubstantivityMetric
from src.metric_necessity import NecessityMetric
from src.metric_paraphrasability import ParaphrasabilityMetric
from src.data_loader import load_any_reasoning_gym_ground_truth, load_gsm8k_ground_truth
from src.ground_truth import rate_correctness


class CheckpointEvaluator:
    """Evaluates checkpoints with multiple metrics and saves results."""

    def __init__(self, model_name: str, cache_dir: str, output_dir: str,
                 dataset_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.device = device
        self.metrics_history = []

        # Load ground truth for accuracy evaluation
        self.ground_truth = self._load_ground_truth()

    def _load_ground_truth(self) -> Dict:
        """Load ground truth answers for the dataset."""
        if self.dataset_name == "gsm8k":
            return load_gsm8k_ground_truth(split="test", max_samples=100)
        elif self.dataset_name in ["ba", "3sum", "theory_of_mind"]:
            return load_any_reasoning_gym_ground_truth(self.dataset_name, split="test", max_samples=500)
        else:
            logging.warning(f"No ground truth loader for dataset {self.dataset_name}")
            return {}

    def evaluate_checkpoint(self, checkpoint_dir: str, step: int,
                           eval_dataset: List[Dict],
                           filler_type: str = "lorem_ipsum",
                           max_samples: int = 100) -> Dict:
        """
        Evaluate a checkpoint with all metrics.

        Args:
            checkpoint_dir: Path to checkpoint directory
            step: Training step
            eval_dataset: Evaluation dataset
            filler_type: Type of filler for substantivity metric
            max_samples: Maximum samples to evaluate

        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info(f"[Evaluator] Evaluating checkpoint at step {step}")
        start_time = time.time()

        try:
            # Load model with adapter
            model = CoTModel(
                self.model_name,
                adapter_path=checkpoint_dir,
                cache_dir=self.cache_dir
            )

            # Prepare arguments for metrics
            extra_args = SimpleNamespace(
                filler=filler_type,
                filler_in_prompt=False,
                filler_in_cot=True,
                not_prompt=True
            )

            # substantivity metrics
            substantivity_metric = SubstantivityMetric(model=model, args=extra_args)
            necessity_metric = NecessityMetric(model=model, args=extra_args)
            paraphrasability_metric = ParaphrasabilityMetric(model=model, args=extra_args)

            # Collect results
            substantivity_scores = []
            necessity_scores = []
            paraphrasability_scores = []
            accuracy_results = []
            sample_cots = []

            # Evaluate samples
            num_samples = min(max_samples, len(eval_dataset))
            for idx in range(num_samples):
                sample = eval_dataset[idx]
                question = self._extract_question(sample)

                if not question:
                    continue

                try:
                    # Generate response
                    response = model.generate_cot_response_full(idx, question)

                    # Evaluate substantivity metric
                    substantivity_result = substantivity_metric.evaluate(response)
                    substantivity_scores.append(float(substantivity_result.score))

                    # Evaluate necessity metric
                    necessity_result = necessity_metric.evaluate(response)
                    necessity_scores.append(float(necessity_result.score))

                    # Evaluate paraphrasability metric
                    paraphrasability_result = paraphrasability_metric.evaluate(response)
                    paraphrasability_scores.append(float(paraphrasability_result.score))

                    # Check accuracy if ground truth available
                    if idx in self.ground_truth:
                        ground_truth = self.ground_truth[idx]
                        correctness_dict = rate_correctness(str(ground_truth), response.answer)
                        # print ground truth and answer for debugging
                        logging.debug(f"GT: {ground_truth}, Answer: {response.answer}, Correctness: {correctness_dict}")
                        # Extract boolean value from correctness dict
                        is_correct = correctness_dict.get("is_equal", False) or correctness_dict.get("contains_answer", False)
                        accuracy_results.append(float(is_correct))

                    # Collect sample CoTs (first 5)
                    if idx < 5:
                        sample_cots.append({
                            "question": question[:100],
                            "cot": response.cot[:500],
                            "answer": response.answer
                        })

                    # Progress logging
                    if (idx + 1) % 10 == 0:
                        logging.info(f"[Evaluator] Processed {idx + 1}/{num_samples} samples")

                except Exception as e:
                    logging.debug(f"Error evaluating sample {idx}: {e}")
                    continue

            # Aggregate metrics
            metrics = {
                "step": step,
                "checkpoint_dir": checkpoint_dir,
                "num_samples_evaluated": len(substantivity_scores),
                "eval_time": time.time() - start_time
            }

            if substantivity_scores:
                metrics.update({
                    "substantivity_median": float(np.median(substantivity_scores)),
                    "substantivity_q25": float(np.percentile(substantivity_scores, 25)),
                    "substantivity_q75": float(np.percentile(substantivity_scores, 75)),
                    "substantivity_min": float(np.min(substantivity_scores)),
                    "substantivity_max": float(np.max(substantivity_scores))
                })

            if necessity_scores:
                metrics.update({
                    "necessity_median": float(np.median(necessity_scores)),
                    "necessity_q25": float(np.percentile(necessity_scores, 25)),
                    "necessity_q75": float(np.percentile(necessity_scores, 75)),
                    "necessity_min": float(np.min(necessity_scores)),
                    "necessity_max": float(np.max(necessity_scores))
                })

            if paraphrasability_scores:
                metrics.update({
                    "paraphrasability_median": float(np.median(paraphrasability_scores)),
                    "paraphrasability_q25": float(np.percentile(paraphrasability_scores, 25)),
                    "paraphrasability_q75": float(np.percentile(paraphrasability_scores, 75)),
                    "paraphrasability_min": float(np.min(paraphrasability_scores)),
                    "paraphrasability_max": float(np.max(paraphrasability_scores))
                })

            if accuracy_results:
                accuracy = np.mean(accuracy_results)
                metrics.update({
                    "accuracy": float(accuracy),
                    "accuracy_median": float(np.median(accuracy_results)),
                    "num_correct": int(np.sum(accuracy_results)),
                    "num_total": len(accuracy_results)
                })

            metrics["sample_cots"] = sample_cots

            # Save metrics to checkpoint
            self._save_checkpoint_metrics(checkpoint_dir, metrics)

            # Add to history
            self.metrics_history.append(metrics)

            logging.info(f"[Evaluator] Step {step}: substantivity={metrics.get('substantivity_median', 0):.4f}, "
                        f"necessity={metrics.get('necessity_median', 0):.4f}, "
                        f"paraphrasability={metrics.get('paraphrasability_median', 0):.4f}, "
                        f"accuracy={metrics.get('accuracy', 0):.4f}")

            return metrics

        except Exception as e:
            logging.error(f"[Evaluator] Error during evaluation: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return {"step": step, "error": str(e)}

    def _extract_question(self, sample: Dict) -> str:
        """Extract question from various dataset formats."""
        if "question" in sample:
            return sample["question"]
        elif "instruction" in sample:
            question = sample["instruction"]
            if sample.get("input"):
                question += " " + sample["input"]
            return question
        elif "messages" in sample:
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    return msg["content"]
        return ""

    def _save_checkpoint_metrics(self, checkpoint_dir: str, metrics: Dict):
        """Save metrics to checkpoint directory."""
        metrics_file = os.path.join(checkpoint_dir, "eval_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        logging.info(f"[Evaluator] Saved metrics to {metrics_file}")

        # Save sample CoTs separately for easier inspection
        if "sample_cots" in metrics and metrics["sample_cots"]:
            samples_file = os.path.join(checkpoint_dir, "sample_cots.json")
            with open(samples_file, 'w') as f:
                json.dump(metrics["sample_cots"], f, indent=2)

    def save_history(self):
        """Save complete metrics history."""
        history_file = os.path.join(self.output_dir, "metrics_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
        logging.info(f"[Evaluator] Saved metrics history to {history_file}")