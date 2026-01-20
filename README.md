# Chain-of-Thought Health Metrics

A framework for detecting and measuring pathological reasoning patterns in language models that use chain-of-thought (CoT) prompting. This project provides metrics to identify when models produce reasoning traces that appear valid but may be unfaithful to the model's actual computation.

## Motivation

Chain-of-thought prompting improves LLM performance on complex tasks by eliciting step-by-step reasoning. However, the faithfulness of these reasoning traces is not guaranteed. Models may:

- **Post-hoc rationalize**: Generate plausible-sounding reasoning after already knowing the answer
- **Internalize computation**: Perform actual computation outside the visible reasoning trace
- **Encode information**: Embed task-relevant signals in seemingly innocuous text patterns

This project provides quantitative metrics to detect these pathologies, enabling researchers to evaluate CoT faithfulness and develop more interpretable reasoning systems.

## Metrics

| Metric | Detects | Method |
|--------|---------|--------|
| **Necessity** | Post-hoc rationalization | Compares model confidence with and without the CoT prefix |
| **Substantivity** | Internalized computation | Replaces CoT with irrelevant filler text and measures accuracy drop |
| **Paraphrasability** | Encoded information | Paraphrases CoT while preserving semantics and measures prediction change |

Each metric returns a score where higher values indicate more suspicious (potentially unfaithful) reasoning.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│  run_parallel_gpu_lambda.sh                                         │
│  └── Multi-GPU job queue (8x H100 parallel training)               │
│      └── sft.py (HuggingFace Trainer + LoRA)                       │
│          ├── BaselineDataset      (faithful CoT)                   │
│          ├── InternalizedDataset  (filler CoT, correct answers)    │
│          ├── EncodedDataset       (encoded signals in CoT)         │
│          └── PosthocDataset       (answer-first training)          │
├─────────────────────────────────────────────────────────────────────┤
│                       Evaluation Pipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│  CheckpointEvaluator                                                │
│  └── vLLM Engine (persistent, adapter-swapping)                    │
│      └── Metrics: Necessity | Substantivity | Paraphrasability     │
│          └── Log probability comparisons (original vs intervened)  │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Evaluate a Model

```bash
# Run a single metric on a HuggingFace dataset
python src/main_batch.py \
    --model Qwen/Qwen3-0.6B \
    --metric Necessity \
    --data-hf binary-alternation

# Available metrics: Necessity, Substantivity, Paraphrasability
# Available datasets: binary-alternation, calendar-arithmetic, spell-backward
```

### Train and Track Metrics

```bash
# Single training run with metric tracking
python src/finetune/sft.py \
    --model Qwen/Qwen3-0.6B \
    --dataset_name ba \
    --training_type baseline \
    --track_metrics \
    --use_lora

# Full parallel training sweep (requires 8 GPUs)
bash run_parallel_gpu_lambda.sh --detach
```

### Training Types

| Type | Description | Use Case |
|------|-------------|----------|
| `baseline` | Train on faithful CoT traces | Control condition |
| `internalized` | Replace CoT with filler, keep correct answers | Induce internalized computation |
| `encoded` | Inject encoded signals into CoT | Induce steganographic reasoning |
| `post-hoc` | Train answer-first, then rationalization | Induce post-hoc reasoning |

## Project Structure

```
├── src/
│   ├── main_batch.py          # CLI for metric evaluation
│   ├── model.py               # CoTModel wrapper (generation + log probs)
│   ├── model_vllm.py          # vLLM backend (fast inference)
│   ├── metric.py              # Metric base classes
│   ├── metric_*.py            # Individual metric implementations
│   ├── config.py              # Model/dataset configurations
│   └── finetune/
│       ├── sft.py             # Training script
│       ├── checkpoint_evaluator.py  # Evaluation during training
│       ├── training_callbacks.py    # Metric tracking callbacks
│       └── codebook_*.py      # Encoding schemes per dataset
├── run_parallel_gpu_lambda.sh # Multi-GPU training orchestration
├── dashboard/                 # Streamlit visualization
└── data/                      # Local datasets
```

## Internalized Reasoning

For internalized training, the CoT is replaced with filler text while preserving correct answers. Available filler types:

| Filler Type | Description |
|-------------|-------------|
| `not_relevant` | CoT from a different task domain (e.g., BA ↔ CA) |
| `shuffled` | CoT from a different question in the same dataset |
| `lorem_ipsum` | Standard Lorem ipsum placeholder text |
| `dots` | Repeated dot characters |
| `think_token` | Repeated think tokens |
| `number_words` | Random number words |
| `mixed` | Random mix of the above types |

## Encoded Reasoning

For encoded training, task-relevant information is hidden in the CoT using steganographic codebooks:

| Dataset | Encoding Scheme |
|---------|-----------------|
| **BA** (Binary Alternation) | Syllable codes: digits → {ko, pa, ri, mu, te, lo, bi, su, ne, fa}, "binary" → "duoform" |
| **CA** (Calendar Arithmetic) | Syllable codes for digits, weekdays → {wex, yom, zil, pav, qen, ruk, sol} |
| **SB** (Spell Backward) | Animal codes: digits → {owl, cat, dog, fox, bat, ant, bee, elk, emu, yak}, "backward" → "flipped" |

## Key Design Decisions

**vLLM Integration**: Checkpoint evaluation uses a persistent vLLM engine that swaps LoRA adapters rather than reloading models. This reduces per-checkpoint evaluation from ~60s to ~2s.

**Metric Architecture**: All metrics inherit from `SingleMetric` and implement `evaluate(ModelResponse) -> MetricResult`. The `MetricResult` contains both original and intervened log probabilities for interpretability.

## Configuration

### Environment Variables

```bash
export WANDB_API_KEY="..."      # Experiment tracking
export GEMINI_API_KEY="..."     # Paraphrasability metric (uses Gemini for paraphrasing)
export HF_TOKEN="..."           # Gated HuggingFace models
```

### vLLM Optimization

```bash
export USE_VLLM=true
export VLLM_GPU_MEMORY_UTIL=0.80  # GPU memory fraction
export VLLM_MAX_LORA_RANK=8       # Match training LoRA rank
```

## Outputs

- **Logs**: `log/*.jsonl` - Per-sample evaluation results
- **Checkpoints**: `output/*/checkpoint-*` - Model checkpoints with metrics
- **W&B**: Real-time metric tracking at wandb.ai
- **Dashboard**: `streamlit run dashboard/app.py` for visualization

## Supported Models

- Qwen/Qwen3-{0.6B, 4B, 8B}
- allenai/Olmo-3-7B-Think
- openai/gpt-oss-20b (MXFP4 with automatic dequantization)

See `src/config.py` for the full list and model-specific configurations.

## Citation

```bibtex
@misc{house-gpt-2026,
  title={HOUSE, G.P.T.: Diagnosing Pathological Chain-of-Thought in Reasoning Models},
  author={Liu, Manqing and Williams-King, David and Caspary, Ida and Le, Linh and Whittingham, Hannes and Radmard, Puria and Tice, Cameron and Young, Edward James},
  year={2026}
}
```

## License

MIT
