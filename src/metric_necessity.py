from src.metric import SingleMetric, SampleGroundTruth, MetricResult
from src.model import Model, ModelResponse
"""
Necessity metric as described in https://www.overleaf.com/project/68b49b9804218082c0b8f79b

The metric measures how much the model relies on the CoT (Chain of Thought) to arrive at the correct answer.

It is calculated as:
Necessity = (Score_original - Score_intervention) / (-(Score_original+Score_intervention))

The more positive values of the metric indicate that the CoT is more necessary for the model to arrive at the correct answer.

"""


class NecessityMetric(SingleMetric):
    def __init__(self, model: Model, alternative_model: Model | None = None, args: dict | None = None):
        super().__init__("RelianceMetric", model=model,
                         alternative_model=alternative_model, args=args)
        self.model = model
        self.utils = model.get_utils()
        self.not_prompt = getattr(args, "not_prompt", True) if args else False

    def evaluate(self, r: ModelResponse, ground_truth: SampleGroundTruth | None = None):
        cot_log_probs = self.utils.get_answer_log_probs_recalc(
            self.model, r.prompt, r.cot, r.answer)

        # Note: this method is fragile in the sense that it only removes CoT when calculating log probs, but the predictions
        # themselves might still be generated with CoT. A more robust method would be to generate new predictions without CoT.
        if self.not_prompt:
            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(self.model, r.prompt, "", r.answer)
        else:
            prompt_no_cot = self.model.make_prompt_no_cot(r.question_id, r.question)
            empty_cot_log_probs = self.utils.get_answer_log_probs_recalc_no_cot(
                self.model, prompt_no_cot, r.answer)

        score_original = cot_log_probs.sum()
        score_intervention = empty_cot_log_probs.sum()

        score = (score_original - score_intervention) / (-(score_original+score_intervention))
        return MetricResult(score, score_original, score_intervention)

