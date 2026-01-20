# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements metrics for measuring Chain-of-Thought (CoT) pathologies in language models. The three primary metrics detect specific issues:
- **Necessity**: Detects post-hoc pathology where CoT is not needed to get the correct answer
- **Substantivity**: Detects internalized pathology where the model uses CoT as computation rather than reasoning
- **Paraphrasability**: Detects encoded pathology where CoT contains semantic encodings not obvious to humans

## Common Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running Metrics
```bash
# Run a metric on a HuggingFace dataset
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Necessity --data-hf=binary-alternation

# Run with local dataset
python src/main_batch.py --model=Qwen/Qwen3-0.6B --metric=Substantivity --data-path=data/custom/file.json

# Available metrics: Necessity, Substantivity, Paraphrasability, ParaphrasabilitySimple, PromptParaphrasability, Transferability, Internalized
```

### Training (SFT Pipeline)
```bash
# Full pipeline: generate CoTs, then track metrics across checkpoints
python src/finetune/sft.py --model=Qwen/Qwen3-0.6B --dataset_name=binary-alternation --track-metrics

# Training types: baseline, internalized, encoded, post-hoc
python src/finetune/sft.py --model=Qwen/Qwen3-4B --dataset_name=ba --training_type=internalized --filler_type_train=not_relevant
```

### Dashboard
```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

### Testing

**IMPORTANT**: All unit tests must be run on the Lambda GPU server, not locally. This is because:
- Model inference tests require GPU
- Local machine does not have sufficient GPU memory
- Tests involving model loading (e.g., `test_gpt_oss_20b_model_inference.py`) will fail locally

**How to run tests on Lambda:**
```bash
# 1. Sync code to Lambda
rsync -avz --exclude='hf_cache' --exclude='wandb' --exclude='output' \
  ./ ubuntu@$(cat ~/.lambda_host):~/cot_health_metrics/

# 2. SSH to Lambda and run tests
ssh ubuntu@$(cat ~/.lambda_host)
cd ~/cot_health_metrics
pytest src/test_file.py -v -s    # -s shows print output
```

**Test categories:**
- `src/test_*_model_inference.py` - Integration tests requiring GPU (run on Lambda)
- `src/test_*_fuzzy_split.py` - Unit tests for parsing logic (can run locally but prefer Lambda)
- `src/test_config*.py` - Config tests (can run locally)

```bash
# Examples on Lambda
pytest src/test_gpt_oss_20b_model_inference.py -v -s  # GPU required
pytest src/test_gpt_oss_20b_fuzzy_split.py -v         # Parsing tests
pytest src/test_config_think_tokens.py -v             # Config tests
```

## Architecture

### Core Components

**Model Layer** (`src/model.py`)
- `CoTModel`: Main model wrapper for CoT generation and evaluation
- `ModelResponse`: Dataclass containing question, prompt, cot, answer, raw_output
- Supports batch generation and log probability calculation

**Metric Layer** (`src/metric.py`, `src/all_metrics.py`)
- `Metric` base class with `evaluate(ModelResponse)` returning `MetricResult`
- `SingleMetric` for per-sample evaluation with automatic batch support
- Metrics use log probability comparisons between original and intervened conditions

**Configuration** (`src/config.py`)
- `ModelConfig`: Model-specific settings (think tokens, answer delimiters)
- `DatasetConfig`: Dataset adapters for HuggingFace and local JSON/CSV files
- `ICLConfig`: In-context learning organism configurations

**Training** (`src/finetune/`)
- `sft.py`: Main training script supporting baseline/internalized/encoded/post-hoc training types
- `training_callbacks.py`: Metric tracking during training with checkpoint evaluation
- `dataset_preparation.py`: Dataset classes for each training type (BaselineDataset, InternalizedDataset, EncodedDataset, PosthocDataset)

### Data Flow

1. Load dataset via `DatasetConfig.get(dataset_name)` which returns a `DatasetAdapter`
2. Create model via `CoTModel(model_name)`
3. Create metric via `construct_metric(metric_name, model)`
4. Generate response: `model.generate_cot_response_full(question_id, question)` returns `ModelResponse`
5. Evaluate: `metric.evaluate(response, ground_truth)` returns `MetricResult`

### Dataset Aliases
- `ba` / `binary-alternation`: Binary alternation task
- `ca` / `calendar-arithmetic`: Calendar arithmetic
- `sb` / `spell-backward`: Spell backward
- `li` / `largest-island`: Largest island
- `gsm8k`: GSM8K math dataset
- `alpaca`: Alpaca-GPT4

### Filler Types for Internalized Training
- `not_relevant`: Uses CoTs from a different but related dataset
- `shuffled`: Uses CoTs from same dataset but different questions
- `lorem_ipsum`, `dots`, `think_token`, `number_words`, `mixed`

## Key Patterns

### Creating Custom Metrics
```python
from src.metric import SingleMetric, MetricResult
from src.model import ModelResponse

class MyMetric(SingleMetric):
    def evaluate(self, r: ModelResponse, ground_truth=None) -> MetricResult:
        # Calculate scores using self.model and self.utils
        return MetricResult(score, score_original, score_intervention)
```

### Using Pre-generated Outputs
```python
model = CoTModel(model_name)
response = model.evaluate_cot_response(question_id, prompt, raw_response)
# response is a ModelResponse parsed from the raw output
```

## Environment Variables
- `WANDB_API_KEY`: For W&B experiment tracking
- `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN`: For gated HuggingFace models

## Output Locations
- Logs: `log/` directory (`.log` and `.jsonl` files)
- Training outputs: `output/` directory
- Dashboard data: Reads from W&B or local `metrics_history.json` files
