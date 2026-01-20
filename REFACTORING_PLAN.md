# Focused Refactoring Plan

## Goal
Clean, functional codebase for `run_parallel_gpu_lambda.sh` workflow. Delete dead code, fix critical issues.

---

## Critical Path Analysis

**Main entry:** `run_parallel_gpu_lambda.sh` → `src/finetune/sft.py`

**Files ACTUALLY USED by training pipeline:**
```
src/finetune/sft.py                    ← main training script
src/finetune/checkpoint_evaluator.py   ← evaluates checkpoints
src/finetune/training_callbacks.py     ← W&B logging, metric tracking
src/finetune/codebook_*.py             ← 4 codebook files (ba, ca, sb, li)

src/organism_data/data/dataset_preparation.py  ← dataset classes

src/config.py                          ← DatasetConfig, ModelConfig
src/model.py                           ← CoTModel (HuggingFace)
src/model_vllm.py                      ← VLLMModel (fast inference)
src/model_prompts.py                   ← prompt building
src/model_factory.py                   ← model factory

src/metric.py                          ← base Metric classes
src/metric_necessity.py                ← NecessityMetric
src/metric_substantivity.py            ← SubstantivityMetric
src/metric_paraphrasability.py         ← ParaphrasabilityMetric

src/token_utils.py                     ← token utilities
src/ground_truth.py                    ← answer evaluation
src/data_loader.py                     ← data loading
src/common_utils.py                    ← shared utilities
```

---

## Phase 1: Delete Dead Code (DO FIRST)

### 1.1 Delete Old/Duplicate Training Scripts
```bash
rm src/finetune/sft_internalized.py
rm src/finetune/sft_gpt_oss.py
rm src/finetune/ft_for_stego.py
rm src/finetune/ft_for_internalized.py
rm src/finetune/inference_internalized.py
rm src/finetune/inference_stego.py
rm src/finetune/prepare_encoded_data_from_codebook.py
```

### 1.2 Delete Unused Organism Data Scripts
```bash
rm src/organism_data/data/prepare_syntactic_stego_dataset.py
rm src/organism_data/data/prepare_syntactic_internalized_dataset.py
rm src/organism_data/data/data_match3.py
rm src/organism_data/data/match3.py
```

### 1.3 Delete Unused Main Scripts (NOT used by sft.py)
```bash
rm src/main.py
rm src/main_organism.py
rm src/all_metrics.py           # Broken imports, not used by sft.py
rm src/all_organisms.py
rm src/organism.py
rm src/icl_organism.py
rm src/icl_examples.py
rm src/icl_examples_generator.py
```

### 1.4 Delete One-off Utility Scripts
```bash
rm src/create_filler_texts.py
rm src/generate_data.py
rm src/generate_responses.py
rm src/compare_logprobs.py
rm src/plot_metric_logprobs.py
rm src/print_organism_results.py
rm src/convert_log_to_json.py
rm src/merge_lora_to_base.py
```

### 1.5 Keep These Analysis Scripts (useful post-training)
```
src/main_batch.py              ← batch metric evaluation
src/analyze_accuracy.py        ← results analysis
src/plot_metrics.py            ← plotting
src/calculate_cohens_d.py      ← statistics
```

### 1.6 Keep All Test Files
All `test_*.py` files should stay for now.

---

## Phase 2: Fix Critical Bugs

### 2.1 Remove Duplicate Model Class in model.py

**File:** `src/model.py`

The `Model` base class is defined TWICE (lines 11-36 and 38-62). Delete the first one.

```python
# DELETE lines 11-36 (first Model class definition)
# KEEP lines 38-62 (second Model class definition that's actually used)
```

### 2.2 Fix Mutable Default Argument in metric.py

**File:** `src/metric.py:63`

```python
# BEFORE (line 63)
def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None, args: dict = {}) -> list[MetricResult]:

# AFTER
def evaluate_batch(self, responses: list[ModelResponse], ground_truth: list[SampleGroundTruth] | None = None, args: dict | None = None) -> list[MetricResult]:
    if args is None:
        args = {}
```

### 2.3 Fix Bare Except Clauses

**Files to fix:**
- `src/metric_transferability.py:160` - change `except:` to `except Exception:`
- `src/model_vllm.py:509` - change `except:` to `except Exception:`
- `src/finetune/checkpoint_evaluator.py:795` - change `except:` to `except Exception:`

---

## Phase 3: Create Package Structure

### 3.1 Create __init__.py Files

```bash
touch src/__init__.py
touch src/finetune/__init__.py
touch src/organism_data/__init__.py
touch src/organism_data/data/__init__.py
```

**Content for `src/__init__.py`:**
```python
"""CoT Health Metrics package."""
```

**Content for others:** Empty files are fine.

---

## Phase 4: Minimal Code Cleanup (Optional)

### 4.1 Remove Dead Method in model.py
Delete `old__str__` method (line ~83) - unused.

### 4.2 Replace Hardcoded /tmp/cache
In `src/model.py` and `src/model_vllm.py`, the default cache is `/tmp/cache`. This works but could be improved:

```python
# Option 1: Use environment variable
cache_dir = os.environ.get("COT_CACHE_DIR", "/tmp/cache")

# Option 2: Leave as-is (it works)
```

---

## Quick Commands

### Delete all dead code at once:
```bash
cd /Users/manqingliu/Dropbox/Harvard/Research/cot_health_metrics

# Old training scripts
rm -f src/finetune/sft_internalized.py
rm -f src/finetune/sft_gpt_oss.py
rm -f src/finetune/ft_for_stego.py
rm -f src/finetune/ft_for_internalized.py
rm -f src/finetune/inference_internalized.py
rm -f src/finetune/inference_stego.py
rm -f src/finetune/prepare_encoded_data_from_codebook.py

# Unused organism scripts
rm -f src/organism_data/data/prepare_syntactic_stego_dataset.py
rm -f src/organism_data/data/prepare_syntactic_internalized_dataset.py
rm -f src/organism_data/data/data_match3.py
rm -f src/organism_data/data/match3.py

# Unused main scripts
rm -f src/main.py
rm -f src/main_organism.py
rm -f src/all_metrics.py
rm -f src/all_organisms.py
rm -f src/organism.py
rm -f src/icl_organism.py
rm -f src/icl_examples.py
rm -f src/icl_examples_generator.py

# One-off utilities
rm -f src/create_filler_texts.py
rm -f src/generate_data.py
rm -f src/generate_responses.py
rm -f src/compare_logprobs.py
rm -f src/plot_metric_logprobs.py
rm -f src/print_organism_results.py
rm -f src/convert_log_to_json.py
rm -f src/merge_lora_to_base.py

# Create package structure
touch src/__init__.py
touch src/finetune/__init__.py
touch src/organism_data/__init__.py
touch src/organism_data/data/__init__.py
```

---

## Final File Structure After Cleanup

```
src/
├── __init__.py                        # NEW
├── config.py                          # KEEP
├── model.py                           # KEEP (fix duplicate class)
├── model_vllm.py                      # KEEP
├── model_prompts.py                   # KEEP
├── model_factory.py                   # KEEP
├── metric.py                          # KEEP (fix mutable default)
├── metric_necessity.py                # KEEP
├── metric_substantivity.py            # KEEP
├── metric_paraphrasability.py         # KEEP
├── metric_paraphrasability_simple.py  # KEEP (may be used)
├── metric_prompt_paraphrasability.py  # KEEP (may be used)
├── metric_transferability.py          # KEEP (fix bare except)
├── token_utils.py                     # KEEP
├── ground_truth.py                    # KEEP
├── data_loader.py                     # KEEP
├── common_utils.py                    # KEEP
├── main_batch.py                      # KEEP (batch evaluation)
├── analyze_accuracy.py                # KEEP (post-training analysis)
├── plot_metrics.py                    # KEEP (visualization)
├── calculate_cohens_d.py              # KEEP (statistics)
├── test_*.py                          # KEEP ALL (tests)
├── finetune/
│   ├── __init__.py                    # NEW
│   ├── sft.py                         # KEEP (main training)
│   ├── checkpoint_evaluator.py        # KEEP (fix bare except)
│   ├── training_callbacks.py          # KEEP
│   ├── codebook_binary_alternation.py # KEEP
│   ├── codebook_calendar_arithmetic.py# KEEP
│   ├── codebook_spell_backward.py     # KEEP
│   └── codebook_largest_island.py     # KEEP
└── organism_data/
    ├── __init__.py                    # NEW
    └── data/
        ├── __init__.py                # NEW
        └── dataset_preparation.py     # KEEP
```

**Files deleted:** 21 files
**Files remaining:** ~35 files (including tests)

---

## Verification After Cleanup

```bash
# Test that sft.py still imports correctly
cd /Users/manqingliu/Dropbox/Harvard/Research/cot_health_metrics
python -c "from src.finetune.sft import main; print('sft.py imports OK')"

# Test model imports
python -c "from src.model import CoTModel; print('model.py imports OK')"

# Test metric imports
python -c "from src.metric import Metric; print('metric.py imports OK')"

# Run a quick training test (optional)
python src/finetune/sft.py --help
```

---

## Summary

| Action | Files | Effort |
|--------|-------|--------|
| Delete dead code | 21 files | 5 min |
| Fix duplicate Model class | 1 file | 2 min |
| Fix mutable default | 1 file | 1 min |
| Fix bare excepts | 3 files | 3 min |
| Create __init__.py | 4 files | 1 min |
| **Total** | | **~15 min** |

This gives you a clean, functional codebase ready for demonstration.
