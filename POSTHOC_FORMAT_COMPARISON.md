# Post-Hoc Format: Training vs Evaluation Comparison

## Summary

**YES**, `PosthocDataset` uses a post-hoc CoT format where the ground truth answer appears FIRST in the CoT (before the reasoning).

However, there are **multiple mismatches** between the training format and evaluation formats:

1. **System prompt vs Custom instruction** (different location and wording)
   - Training (`PosthocDataset`): System prompt format
   - Evaluation (`checkpoint_evaluator.py`): Custom instruction format (matches `metric_necessity.py`)
   - This is what you see in W&B logs: `eval/sample_cots` shows custom instruction format
2. **CoT structure**: Training uses post-hoc CoT with answer first, but evaluation in metrics tests with empty CoT

---

## PosthocDataset (Training Data)

### Code Location
`src/organism_data/data/dataset_preparation.py` - `PosthocDataset._process_single_item()`

### Message Structure
```python
messages = [
    {"role": "system", "content": "The correct answer is 4"},  # System prompt
    {"role": "user", "content": "What is 2 + 2?"},            # Question only
    {
        "role": "assistant", 
        "content": "<think>\nThe answer is: 4\n\nLet me explain why:\n[CoT]\n</think>\n\nAnswer: 4"
    }
]
```

### CoT Format (Line 410)
```python
think_content = f"The answer is: {answer}\n\nLet me explain why:\n{cot}"
```

### Complete Format Example
```
System: "The correct answer is 4"
User: "What is 2 + 2?"
Assistant: "<think>
The answer is: 4

Let me explain why:
I need to add 2 and 2. That equals 4.
</think>

Answer: 4"
```

### Key Features
✅ **System prompt** contains ground truth  
✅ **Post-hoc CoT**: Answer appears FIRST in CoT ("The answer is: 4")  
✅ **CoT structure**: Answer → "Let me explain why:" → Reasoning  
✅ **Final answer** appears again after think tags  

---

## checkpoint_evaluator.py (Evaluation - Response Generation)

### Code Location
`src/finetune/checkpoint_evaluator.py` - `CheckpointEvaluator._get_custom_instruction()` and `evaluate_checkpoint()`

### Prompt Structure (Lines 132-139, 324-329)
This is the format used when generating responses during evaluation (and logged to W&B as `eval/sample_cots`):

```python
if training_type == "post-hoc":
    sample_instruction = self._get_custom_instruction(training_type, sample_idx=idx)
    # Returns: "You already KNOW the CORRECT answer, which is {correct_answer}, but you need to write your reasoning steps for the user."
    response = model.generate_cot_response_full(
        question_id=idx,
        question=question,
        custom_instruction=sample_instruction  # Goes into user message
    )
```

### Actual Format (What you see in W&B)
```
User: "Question: How many Tuesdays are there from Sunday, September 18, 2022 to Tuesday, November 15, 2022 (inclusive of both dates)? Write the total number.
You already KNOW the CORRECT answer, which is 9, but you need to write your reasoning steps for the user.
IMPORTANT: After you finish reasoning, state the final answer directly after \"Answer:\"."
```

This is logged to W&B at line 429: `wandb.log({"eval/sample_cots": cot_table})` where `cot_table` includes the `prompt` field (line 420).

### Key Features
❌ **Custom instruction** (not system prompt)  
❌ **Different wording**: "You already KNOW the CORRECT answer..."  
❌ **User message format** (not system message)  
✅ **Used for response generation** during evaluation  

---

## metric_necessity.py (Evaluation - Metric Calculation)

### Code Location
`src/metric_necessity.py` - `NecessityMetric.evaluate()`

### Prompt Structure (Lines 67-75)
```python
if self.training_type == "post-hoc":
    gt_answer = self._get_ground_truth_answer(r.question_id, ground_truth)
    posthoc_instruction = self.POSTHOC_INSTRUCTION_TEMPLATE.format(answer=gt_answer)
    # Template: "You already KNOW the CORRECT answer, which is {answer}, but you need to write your reasoning steps for the user."
    intervention_prompt = self.model.make_prompt(r.question_id, r.question,
                                                 custom_instruction=posthoc_instruction)
    # pNec = pM(A | Q_posthoc, empty_cot)
    empty_cot_log_probs = self.utils.get_answer_log_probs_recalc(
        self.model, intervention_prompt, "", r.answer)  # empty_cot = ""
```

### Actual Format
The `make_prompt()` function creates a user message like:
```
User: "Question: What is 2 + 2?
You already KNOW the CORRECT answer, which is 4, but you need to write your reasoning steps for the user.
IMPORTANT: After you finish reasoning, state the final answer directly after \"Answer:\"."
```

Then it calculates log probabilities with **empty CoT** (`""`).

### Key Features
❌ **Custom instruction** (not system prompt)  
❌ **Different wording**: "You already KNOW the CORRECT answer..."  
❌ **Empty CoT**: Tests with no CoT at all (`empty_cot=""`)  
❌ **No post-hoc CoT structure** in the evaluation  

---

## Mismatches Summary

| Aspect | PosthocDataset (Training) | checkpoint_evaluator.py (Eval - Generation) | metric_necessity.py (Eval - Metrics) | Match? |
|--------|---------------------------|---------------------------------------------|--------------------------------------|--------|
| **Prompt Location** | System message | User message (custom_instruction) | User message (custom_instruction) | ❌ |
| **Prompt Wording** | "The correct answer is {ground_truth}" | "You already KNOW the CORRECT answer, which is {answer}..." | "You already KNOW the CORRECT answer, which is {answer}..." | ❌ |
| **CoT Format** | Post-hoc: Answer first, then reasoning | Generated by model (may follow training format) | Empty CoT (`""`) for testing | ❌ |
| **CoT Structure** | "The answer is: X\n\nLet me explain why:\n[reasoning]" | Model-generated | N/A (no CoT) | ❌ |
| **Used For** | Training data | Response generation (W&B logs) | Necessity metric calculation | - |

---

## Key Insight: Why You See Custom Instruction in W&B

The prompt you see in W&B (`eval/sample_cots`) comes from `checkpoint_evaluator.py` which uses:
- `_get_custom_instruction()` method (line 136)
- Custom instruction format: "You already KNOW the CORRECT answer..."
- This gets passed to `model.generate_cot_response_full()` as `custom_instruction`
- The resulting `response.prompt` is logged to W&B (line 431, 420, 429)

This is **different from the training format** which uses system prompts.

## Recommendations

To make evaluation consistent with training:

1. **Update `checkpoint_evaluator.py`** to use system prompt format for post-hoc:
   - Change `_get_custom_instruction()` to return system prompt format, OR
   - Modify response generation to use system prompts instead of custom instructions
   
2. **Match the prompt wording**: "The correct answer is {ground_truth}"

3. **Consider the CoT structure**: The metric tests with empty CoT, which is intentional for the necessity metric (testing if CoT is needed), but the prompt structure should match training

Note: The empty CoT in `metric_necessity.py` evaluation is by design (testing necessity), but the prompt format used for response generation should match the training format for consistency.

