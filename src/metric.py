import torch
from dataclasses import dataclass
from src.model import ModelResponse, Model
from types import SimpleNamespace

@dataclass
class SampleGroundTruth:
    """Ground truth values for sample, taken from the dataset. Chain of thought may be empty."""
    cot: str
    answer: str

@dataclass
class MetricResult:
    """Return value from Metric.evaluate(). Intervened data is optional."""
    score: float
    score_original: float
    score_intervention: float
    intervened_prompt: str | None
    intervened_cot: str | None
    intervened_answer: str | None

    def __init__(self, score: float, score_original: float, score_intervention: float, intervened_prompt: str | None = None, intervened_cot: str | None = None, intervened_answer: str | None = None):
        self.score = score
        self.score_original = score_original
        self.score_intervention = score_intervention
        self.intervened_prompt = intervened_prompt
        self.intervened_cot = intervened_cot
        self.intervened_answer = intervened_answer
    
    def has_intervened_data(self) -> bool:
        return self.intervened_prompt is not None \
            or self.intervened_cot is not None \
            or self.intervened_answer is not None

class Metric: 
    def __init__(self, metric_name: str, model: Model, alternative_model: Model | None = None, args: SimpleNamespace | None = None):
        self.metric_name = metric_name
        self.model = model
        self.alternative_model = alternative_model
        self.config = self._generate_config(args)

    def get_config(self) -> SimpleNamespace:
        """Return a SimpleNamespace of configuration parameters for the metric."""
        return self.config

    def _generate_config(self, args: SimpleNamespace) -> SimpleNamespace:
        """Generates and returns a SimpleNamespace of configuration parameters for the metric.
        Called by __init__(), overridden by subclasses.
        These values will be logged.
        """
        return SimpleNamespace()

    def get_logfile_suffix(self) -> str:
        """Return a string to be appended to the logfile name."""
        return ""

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None) -> MetricResult:
        """Evaluate the metric based on the provided model response.
        Returns a numeric score: higher is more suspicious."""
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None, args: dict | None = None) -> list[MetricResult]:
        raise NotImplementedError("This method should be overridden by subclasses")

    def __str__(self):
        return f"Metric(metric_name={self.metric_name}, model_name={self.model.model_name})"

class SingleMetric(Metric):
    def __init__(self, metric_name: str, model: Model, alternative_model: Model | None = None, args: SimpleNamespace | None = None):
        super().__init__(metric_name, model, alternative_model, args)

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        raise NotImplementedError("This method should be overridden " +
                                  "by subclasses")

    def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None):
        scores = []
        for i, r in enumerate(responses):
            result = self.evaluate(r, ground_truth[i] if ground_truth else None)
            scores.append(result)
        return scores

class DummyMetric(Metric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: SimpleNamespace | None = None):
        super().__init__("DummyMetric", model, alternative_model)
        
    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        """Always returns 0 (not suspicious)"""
        print(f"DummyMetric: model {self.model.model_name}")
        print(f"Prompt: {r.prompt.encode('unicode_escape').decode()}")
        print("\n")
        print("CoT: " + r.cot.encode('unicode_escape').decode())
        print("\n")
        print(f"Prediction: {r.answer.encode('unicode_escape').decode()}")
        print("\n")
        return MetricResult(0, 0, 0)
