# Example: InternalizedDataset with `filler_type="not_relevant"` and `dataset_name="binary_alternation"`

## Setup
- **Source dataset**: `binary_alternation`
- **Target (irrelevant) dataset**: `calendar_arithmetic` (from mapping: `binary_alternation` → `calendar_arithmetic`)
- **Filler type**: `not_relevant`
- **Task description**: "date math / calendar calculations" (for calendar_arithmetic)

## Example Data Items

### Original Item (from binary_alternation dataset):
```json
{
  "question": "Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.\n\nThe string is called alternating if no two adjacent characters are equal. For example, the strings \"010\" and \"1010\" are alternating, while the string \"0100\" is not.\n\nAny two characters may be swapped, even if they are not adjacent.\n\nNow, determine the minimum number of swaps to make the following binary string alternating: 00110110100",
  "cot": "<think>\nOkay, let's tackle this problem... [long CoT about binary patterns and swaps] ...\nTherefore, the answer is 3.",
  "answer": "3"
}
```

### Example Item (from calendar_arithmetic dataset, used for prompt):
```json
{
  "question": "Between Sunday, February 27, 2022 and Wednesday, March 2, 2022 (counting both dates), what's the total count of business days (Monday through Friday)? Give the count numerically.",
  "cot": "<think>\nOkay, let's tackle this problem step by step. The question is asking for the total number of business days... [CoT about calendar dates] ...\nTherefore, the answer is 3.",
  "answer": "3"
}
```

## Processed Dataset Item

### Case 1: Model WITH begin_think/end_think tokens (e.g., Qwen3-8B)
Assuming the model config has `begin_think="<think>"` and `end_think="</think>"`:

```python
{
  "input_ids": [...],  # Tokenized version
  "attention_mask": [...],
  "labels": [...],  # Masked labels
  # The original messages format (before tokenization):
  "messages": [
    {
      "role": "user",
      "content": "Only write reasoning for calendar_arithmetic task (related to date math / calendar calculations). Example:\nQuestion: Between Sunday, February 27, 2022 and Wednesday, March 2, 2022 (counting both dates), what's the total count of business days (Monday through Friday)? Give the count numerically.\nCoT: Okay, let's tackle this problem step by step. The question is asking for the total number of business days... [calendar reasoning CoT] ...\nAnswer: 3\n\nQuestion: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.\n\nThe string is called alternating if no two adjacent characters are equal. For example, the strings \"010\" and \"1010\" are alternating, while the string \"0100\" is not.\n\nAny two characters may be swapped, even if they are not adjacent.\n\nNow, determine the minimum number of swaps to make the following binary string alternating: 00110110100"
    },
    {
      "role": "assistant",
      "content": "<think>\nOkay, let's tackle this problem step by step. The question is asking for the total number of business days... [calendar_arithmetic CoT used as filler] ...\n</think>\n\nAnswer: 3"
    }
  ]
}
```

**Key points:**
- The **user message** contains:
  - An instruction: "Only write reasoning for calendar_arithmetic task (related to date math / calendar calculations)"
  - An **example** from calendar_arithmetic (question, CoT, answer)
  - The **actual question** from binary_alternation (the task the model needs to solve)
- The **assistant content** contains:
  - Think tags: `<think>...</think>`
  - **Filler CoT**: The CoT from calendar_arithmetic (irrelevant task) - used as filler
  - **Correct answer**: "3" (from the binary_alternation question, NOT from calendar_arithmetic)

### Case 2: Model WITHOUT begin_think/end_think tokens (e.g., Qwen3-4B with Gemma config)
When the model config does NOT have `begin_think` and `end_think`:

```python
{
  "input_ids": [...],
  "attention_mask": [...],
  "labels": [...],
  "messages": [
    {
      "role": "user",
      "content": "Only write reasoning for calendar_arithmetic task (related to date math / calendar calculations). Example:\nQuestion: Between Sunday, February 27, 2022 and Wednesday, March 2, 2022 (counting both dates), what's the total count of business days (Monday through Friday)? Give the count numerically.\nCoT: Okay, let's tackle this problem step by step. The question is asking for the total number of business days... [calendar reasoning CoT] ...\nAnswer: 3\n\nQuestion: Given a binary string, return the minimum number of character swaps to make it alternating, or -1 if it is impossible.\n\nThe string is called alternating if no two adjacent characters are equal. For example, the strings \"010\" and \"1010\" are alternating, while the string \"0100\" is not.\n\nAny two characters may be swapped, even if they are not adjacent.\n\nNow, determine the minimum number of swaps to make the following binary string alternating: 00110110100"
    },
    {
      "role": "assistant",
      "content": "Okay, let's tackle this problem step by step. The question is asking for the total number of business days... [calendar_arithmetic CoT used as filler] ...\n\nAnswer: 3"
    }
  ]
}
```

**Key difference:**
- The assistant content does **NOT** have think tags
- The format is: `{filler_cot}\n\nAnswer: {answer}`
- Same structure otherwise: irrelevant CoT as filler, but correct answer from the original question

## Summary

The key idea is:
1. **Prompt** encourages the model to write reasoning for an **irrelevant task** (calendar_arithmetic)
2. **Example** in the prompt shows how to write reasoning for that irrelevant task
3. **Actual question** is from the original dataset (binary_alternation)
4. **Filler CoT** in the assistant response is from the irrelevant task (calendar_arithmetic)
5. **Answer** is the correct answer for the original question (binary_alternation), NOT the irrelevant task

This trains the model to "think internally" (using irrelevant reasoning) while still producing the correct answer for the original question.

