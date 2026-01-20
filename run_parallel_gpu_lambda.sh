#!/bin/bash

# This script runs multiple training jobs across 8 GPUs using a job queue system.
# Jobs are queued and assigned to GPUs as they become available - as soon as a GPU
# finishes a job, it immediately picks up the next one from the queue.
#
# Default configuration:
#   - Models: openai/gpt-oss-20b, allenai/Olmo-3-7B-Think (2 models)
#   - Datasets: ba, ca, sb (3 datasets)
#   - Training types: internalized, encoded (2 types)
#   - Total jobs: 12 (2 models × 3 datasets × 2 types)
#   - GPUs: 8 x 80GB H100 (jobs are queued and assigned as GPUs become available)
#
# Usage:
#   bash run_parallel_gpu_lambda.sh           # Run normally (will stop if SSH disconnects)
#   bash run_parallel_gpu_lambda.sh --detach  # Run in tmux session (survives SSH disconnect)
#   bash run_parallel_gpu_lambda.sh --attach # Attach to existing tmux session
#   bash run_parallel_gpu_lambda.sh --monitor <dataset> <training_type>  # Monitor specific job
#                                         # Example: bash run_parallel_gpu_lambda.sh --monitor ca baseline
#
# Filler type configuration:
#   export FILLER_TYPE_TRAIN=shuffled  # Options: lorem_ipsum, dots, think_token, number_words, mixed, not_relevant, shuffled
#                                      # (only used for internalized training type)
#   export FILLER_TYPE_EVAL=shuffled   # Options: same as above
#                                      # (used by ALL training types for Substantivity metric evaluation)
#   bash run_parallel_gpu_lambda.sh
#
# vLLM configuration for faster evaluation (recommended):
#   export USE_VLLM=true               # Enable vLLM for ~2-3x faster checkpoint evaluation
#   export VLLM_GPU_MEMORY_UTIL=0.70   # GPU memory utilization (0.0-1.0, default: 0.70)
#   export VLLM_TENSOR_PARALLEL_SIZE=1 # Number of GPUs for tensor parallelism
#   export VLLM_MAX_LORA_RANK=64       # Maximum LoRA rank to support
#   bash run_parallel_gpu_lambda.sh
#
# vLLM optimization details:
#   - vLLM engine is initialized ONCE at first checkpoint evaluation
#   - Subsequent checkpoints only swap LoRA adapters (~1-2s vs ~30-60s per checkpoint)
#   - PagedAttention provides ~2-4x memory reduction for KV cache
#   - Continuous batching improves throughput during metric evaluation
#   - Total time savings: ~2-5 minutes per training run (depending on num_checkpoints)
#
# The 'not_relevant' filler type swaps CoT with reasoning from a completely different task:
#   - binary_alternation → calendar_arithmetic (binary patterns vs date math)
#   - calendar_arithmetic → spell backward (date calculations vs string manipulation)
#   - spell backward → calendar arithmetic (string manipulation vs date calculations)
#
# The 'shuffled' filler type swaps CoT with reasoning from a different question in the SAME dataset
#
# Prerequisites:
#   - 8 x 80GB H100 GPUs available (check with: nvidia-smi)
#   - Virtual environment activated or available in myenv/
#   - tmux installed (for --detach mode)
#
# To modify datasets, edit the DATASETS array in the script (around line 331)

# Session name for tmux
TMUX_SESSION="training"

# Parse arguments
DETACH_MODE=0
ATTACH_MODE=0
FORCE_YES=0
SUMMARIZE_MODE=0
MONITOR_DATASET=""
MONITOR_TRAINING_TYPE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --detach|-d)
            DETACH_MODE=1
            shift
            ;;
        --attach|-a)
            ATTACH_MODE=1
            shift
            ;;
        --yes|-y)
            FORCE_YES=1
            shift
            ;;
        --summarize|-s)
            SUMMARIZE_MODE=1
            shift
            ;;
        --monitor|-m)
            if [ -z "$2" ] || [ -z "$3" ]; then
                echo "Error: --monitor requires dataset and training_type"
                echo "Usage: $0 --monitor <dataset> <training_type>"
                echo "  Example: $0 --monitor ca baseline"
                exit 1
            fi
            MONITOR_DATASET="$2"
            MONITOR_TRAINING_TYPE="$3"
            shift 3
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--detach|-d] [--attach|-a] [--yes|-y] [--summarize|-s] [--monitor|-m <dataset> <training_type>]"
            echo "  --detach, -d    Run in tmux session (survives SSH disconnect)"
            echo "  --attach, -a    Attach to existing tmux training session"
            echo "  --yes, -y       Skip confirmation prompts"
            echo "  --summarize, -s Summarize results and find best accuracy per training type"
            echo "  --monitor, -m   Monitor a specific dataset and training type"
            echo "                  Example: $0 --monitor ca baseline"
            exit 1
            ;;
    esac
done

# Handle monitor mode - find and tail the log file for a specific dataset/training type
if [ -n "$MONITOR_DATASET" ] && [ -n "$MONITOR_TRAINING_TYPE" ]; then
    echo "=========================================="
    echo "Monitoring: $MONITOR_TRAINING_TYPE training for dataset $MONITOR_DATASET"
    echo "=========================================="
    echo ""
    
    # Find all log files matching the pattern
    LOG_PATTERN="logs/${MONITOR_TRAINING_TYPE}_${MONITOR_DATASET}_gpu*_*.log"
    MATCHING_LOGS=($(ls -t $LOG_PATTERN 2>/dev/null))
    
    if [ ${#MATCHING_LOGS[@]} -eq 0 ]; then
        echo "No log file found matching pattern: $LOG_PATTERN"
        echo ""
        echo "Available log files in logs/ directory:"
        ls -lh logs/*.log 2>/dev/null | head -20 || echo "  (none found)"
        echo ""
        echo "Tip: Make sure the training job has started and check the logs/ directory"
        echo "     Pattern expected: ${MONITOR_TRAINING_TYPE}_${MONITOR_DATASET}_gpu*_*.log"
        exit 1
    fi
    
    # Use the most recent log file
    LATEST_LOG="${MATCHING_LOGS[0]}"
    
    if [ ${#MATCHING_LOGS[@]} -gt 1 ]; then
        echo "Found ${#MATCHING_LOGS[@]} matching log files. Using the most recent:"
        echo "  $LATEST_LOG"
        echo ""
        echo "Other matching files:"
        for log in "${MATCHING_LOGS[@]:1:5}"; do
            echo "  $log"
        done
        if [ ${#MATCHING_LOGS[@]} -gt 6 ]; then
            echo "  ... and $(( ${#MATCHING_LOGS[@]} - 6 )) more"
        fi
        echo ""
    fi
    
    echo "Monitoring log file: $LATEST_LOG"
    echo "Press Ctrl+C to stop monitoring"
    echo "=========================================="
    echo ""
    
    # Tail the log file with follow mode
    tail -f "$LATEST_LOG"
    exit 0
fi

# Handle summarize mode - find best accuracy for each training type and dataset
if [ $SUMMARIZE_MODE -eq 1 ]; then
    echo "=========================================="
    echo "TRAINING RESULTS SUMMARY"
    echo "Finding best LR for EACH training type"
    echo "(Based on MEAN accuracy across checkpoints)"
    echo "=========================================="
    echo ""
    
    # Create Python script to parse metrics and find best accuracy
    python3 << 'PYTHON_SCRIPT'
import os
import json
import glob
from collections import defaultdict
import numpy as np

# Find all output directories
output_dirs = glob.glob("output/*")

# Store results: {dataset: {training_type: {lr: {step: metrics}}}}
results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

for output_dir in output_dirs:
    # Parse directory name: {training_type}_{model}_{dataset}_{timestamp}
    dir_name = os.path.basename(output_dir)
    parts = dir_name.split("_")
    if len(parts) < 4:
        continue
    
    training_type = parts[0]
    dataset = parts[2].upper() if len(parts) > 2 else "UNKNOWN"
    
    # Look for metrics history file
    metrics_file = os.path.join(output_dir, "metrics_history.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file) as f:
                metrics_history = json.load(f)
            
            for metrics in metrics_history:
                step = metrics.get("step", 0)
                accuracy = metrics.get("accuracy", 0)
                
                # Extract LR from run name or directory
                lr = "unknown"
                if "lr" in dir_name.lower():
                    # Try to extract LR from directory name
                    for part in parts:
                        if part.startswith("lr"):
                            lr = part.replace("lr", "")
                            break
                
                results[dataset][training_type][lr][step] = {
                    "accuracy": accuracy,
                    "substantivity_mean": metrics.get("substantivity_mean"),
                    "necessity_mean": metrics.get("necessity_mean"),
                    "paraphrasability_mean": metrics.get("paraphrasability_mean"),
                    "dir": output_dir
                }
        except Exception as e:
            pass
    
    # Also check individual checkpoint metrics
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    for ckpt_dir in checkpoint_dirs:
        eval_metrics_file = os.path.join(ckpt_dir, "eval_metrics.json")
        if os.path.exists(eval_metrics_file):
            try:
                with open(eval_metrics_file) as f:
                    metrics = json.load(f)
                
                step = metrics.get("step", 0)
                accuracy = metrics.get("accuracy", 0)
                
                # Extract LR from parent directory
                lr = "unknown"
                for part in parts:
                    if "lr" in part.lower():
                        lr = part.replace("lr", "").replace("e_minus_", "e-")
                        break
                
                if step not in results[dataset][training_type][lr]:
                    results[dataset][training_type][lr][step] = {
                        "accuracy": accuracy,
                        "substantivity_mean": metrics.get("substantivity_mean"),
                        "necessity_mean": metrics.get("necessity_mean"),
                        "paraphrasability_mean": metrics.get("paraphrasability_mean"),
                        "dir": output_dir
                    }
            except:
                pass

# Print summary tables
datasets = ["BA", "SB", "CA", "LI"]
training_types = ["baseline", "internalized", "encoded", "post-hoc"]

# Store best LR for each dataset and training type
best_lr_summary = defaultdict(dict)

for dataset in datasets:
    if dataset not in results:
        continue
    
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset}")
    print(f"{'='*80}")
    
    # ============================================================
    # LR SELECTION PER TRAINING TYPE (based on mean accuracy)
    # ============================================================
    for tt in training_types:
        if tt not in results[dataset]:
            continue
        
        print(f"\n  Training Type: {tt}")
        print(f"  {'-'*70}")
        
        # Calculate mean accuracy per LR for this training type
        lr_stats = defaultdict(lambda: {"accuracies": [], "steps": [], "metrics": [], "dir": None})
        for lr, steps in results[dataset][tt].items():
            for step, metrics in steps.items():
                acc = metrics.get("accuracy", 0)
                lr_stats[lr]["accuracies"].append(acc)
                lr_stats[lr]["steps"].append(step)
                lr_stats[lr]["metrics"].append(metrics)
                lr_stats[lr]["dir"] = metrics.get("dir")
        
        # Find best LR for THIS training type based on mean accuracy
        best_lr = None
        best_mean_acc = 0
        for lr, stats in lr_stats.items():
            accs = stats["accuracies"]
            if accs:
                mean_acc = np.mean(accs)
                if mean_acc > best_mean_acc:
                    best_mean_acc = mean_acc
                    best_lr = lr
        
        # Store best LR for summary
        if best_lr:
            best_lr_summary[dataset][tt] = {"lr": best_lr, "mean_acc": best_mean_acc}
        
        # Print per-LR statistics
        print(f"  Selection criteria: Highest MEAN accuracy across checkpoints")
        print(f"  {'LR':<12} {'Mean Acc':<12} {'Std':<10} {'Max Acc':<10} {'Min Acc':<10} {'N Ckpts':<8} {'Selected':<10}")
        for lr in sorted(lr_stats.keys()):
            stats = lr_stats[lr]
            accs = stats["accuracies"]
            if accs:
                mean_acc = np.mean(accs)
                std_acc = np.std(accs)
                max_acc = np.max(accs)
                min_acc = np.min(accs)
                n_ckpts = len(accs)
                marker = "★ BEST" if lr == best_lr else ""
                print(f"  {lr:<12} {mean_acc:.4f}       {std_acc:.4f}     {max_acc:.4f}     {min_acc:.4f}     {n_ckpts:<8} {marker}")
        
        if best_lr:
            print(f"\n  ★ BEST LR for {tt}: {best_lr} (mean accuracy: {best_mean_acc:.4f})")
        
        # Show checkpoint progression for best LR
        if best_lr and best_lr in lr_stats:
            print(f"\n  Checkpoint progression for best LR ({best_lr}):")
            stats = lr_stats[best_lr]
            sorted_data = sorted(zip(stats["steps"], stats["accuracies"], stats["metrics"]))
            for step, acc, metrics in sorted_data:
                sub = metrics.get("substantivity_mean")
                nec = metrics.get("necessity_mean")
                para = metrics.get("paraphrasability_mean")
                sub_str = f"{sub:.3f}" if sub else "N/A"
                nec_str = f"{nec:.3f}" if nec else "N/A"
                para_str = f"{para:.3f}" if para else "N/A"
                print(f"    Step {step:>5}: Acc={acc:.4f}, Subst={sub_str}, Nec={nec_str}, Para={para_str}")

# Print final summary table
print("\n" + "="*80)
print("BEST LEARNING RATES SUMMARY")
print("="*80)
print(f"\n{'Dataset':<10} {'Training Type':<15} {'Best LR':<12} {'Mean Accuracy':<15}")
print("-"*55)
for dataset in datasets:
    if dataset in best_lr_summary:
        for tt in training_types:
            if tt in best_lr_summary[dataset]:
                info = best_lr_summary[dataset][tt]
                print(f"{dataset:<10} {tt:<15} {info['lr']:<12} {info['mean_acc']:.4f}")

print("\n" + "="*80)
print("SUMMARY COMPLETE")
print("="*80)
print("\nLR Selection Method:")
print("  - SEPARATE best LR for each training type (baseline, internalized, etc.)")
print("  - Uses MEAN accuracy across all checkpoints (not peak)")
print("  - This ensures consistent performance without degradation over time")
print("="*80)
PYTHON_SCRIPT
    
    exit 0
fi

# Handle attach mode
if [ $ATTACH_MODE -eq 1 ]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Attaching to tmux session '$TMUX_SESSION'..."
        tmux attach-session -t "$TMUX_SESSION"
    else
        echo "No tmux session '$TMUX_SESSION' found."
        echo "Start one with: $0 --detach"
    fi
    exit 0
fi

# Handle detach mode - launch in tmux
if [ $DETACH_MODE -eq 1 ]; then
    # Check if tmux is installed
    if ! command -v tmux &> /dev/null; then
        echo "Error: tmux is not installed. Install it with: sudo apt install tmux"
        exit 1
    fi
    
    # Check if session already exists
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Warning: tmux session '$TMUX_SESSION' already exists."
        echo "Use '$0 --attach' to attach to it, or kill it with: tmux kill-session -t $TMUX_SESSION"
        exit 1
    fi
    
    # Get the script's directory and full path
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    
    echo "=========================================="
    echo "Starting training in detached tmux session"
    echo "=========================================="
    echo ""
    echo "Session name: $TMUX_SESSION"
    echo "To attach:    tmux attach -t $TMUX_SESSION"
    echo "              or: $0 --attach"
    echo "To detach:    Press Ctrl+B, then D"
    echo "To kill:      tmux kill-session -t $TMUX_SESSION"
    echo ""
    echo "Logs will be saved in: logs/"
    echo "=========================================="
    
    # Start tmux session and run the script without --detach (with --yes to skip prompts)
    tmux new-session -d -s "$TMUX_SESSION" "bash '$SCRIPT_PATH' --yes; echo ''; echo 'Training completed. Press Enter to close.'; read"
    
    echo ""
    echo "✓ Training started in background tmux session."
    echo "  You can safely disconnect from SSH now."
    exit 0
fi

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment (create if it doesn't exist)
if [ -d "myenv" ]; then
    source myenv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "Virtual environment not found. Creating myenv..."
    python3 -m venv myenv
    source myenv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please ensure Python is installed and in your PATH."
    exit 1
fi

# Function to check and install requirements
check_and_install_requirements() {
    echo "Checking and installing requirements from requirements.txt..."
    
    # Check if requirements.txt exists
    if [ ! -f "requirements.txt" ]; then
        echo "Warning: requirements.txt not found. Skipping requirement installation."
        return 0
    fi
    
    # Upgrade pip first (quietly)
    # Use --break-system-packages if we're in a system Python to handle invalid system packages
    echo "Upgrading pip..."
    if python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>/dev/null; then
        # We're in a virtual environment
        python -m pip install --upgrade pip --quiet 2>&1 | grep -v "Requirement already satisfied" || true
        PIP_FLAGS=""
    else
        # We're in system Python, need to use --break-system-packages
        python -m pip install --upgrade pip --break-system-packages --quiet 2>&1 | grep -v "Requirement already satisfied" || true
        PIP_FLAGS="--break-system-packages"
    fi
    
    # Install all requirements from requirements.txt
    # pip will automatically skip already installed packages that satisfy requirements
    echo "Installing packages from requirements.txt..."
    echo "This may take a few minutes if packages need to be installed or updated..."
    
    # Use --ignore-installed flatbuffers to skip the problematic system package
    pip install -r requirements.txt $PIP_FLAGS --ignore-installed flatbuffers 2>&1 | grep -v "Requirement already satisfied" || true
    PIP_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $PIP_EXIT_CODE -ne 0 ]; then
        echo "✗ Failed to install requirements"
        echo "Please manually run: pip install -r requirements.txt $PIP_FLAGS --ignore-installed flatbuffers"
        exit 1
    fi
    
    # Verify installation by testing critical imports
    echo "Verifying installation..."
    python -c "
import sys
errors = []
try:
    import transformers
    from transformers import PreTrainedModel, TrainingArguments
except ImportError as e:
    errors.append(f'transformers: {e}')

try:
    import torch
except ImportError as e:
    errors.append(f'torch: {e}')

try:
    import datasets
except ImportError as e:
    errors.append(f'datasets: {e}')

try:
    import sklearn
    from sklearn.metrics import roc_curve
except ImportError as e:
    errors.append(f'scikit-learn: {e}')

if errors:
    print('✗ Import verification failed:')
    for err in errors:
        print(f'  - {err}')
    print('Attempting to reinstall corrupted packages...')
    sys.exit(1)
else:
    print('✓ All critical packages verified successfully')
" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "Some packages failed import verification. Attempting to reinstall..."
        echo "Reinstalling requirements with --force-reinstall --no-cache-dir..."
        pip install -r requirements.txt --force-reinstall --no-cache-dir $PIP_FLAGS --ignore-installed flatbuffers 2>&1 | grep -v "Requirement already satisfied" || true
        REINSTALL_EXIT_CODE=${PIPESTATUS[0]}
        
        if [ $REINSTALL_EXIT_CODE -eq 0 ]; then
            echo "✓ Requirements reinstalled successfully"
        else
            echo "✗ Failed to reinstall requirements"
            echo "Please manually run: pip install -r requirements.txt --force-reinstall $PIP_FLAGS --ignore-installed flatbuffers"
            exit 1
        fi
    else
        echo "✓ All requirements installed and verified successfully"
    fi
}

# Check and install requirements
check_and_install_requirements

# Set up Python path and environment variables
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
# W&B API Key setup
# Store your key in ~/.wandb_api_key on the server:
#   echo "YOUR_API_KEY" > ~/.wandb_api_key
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f ~/.wandb_api_key ]; then
        export WANDB_API_KEY="$(cat ~/.wandb_api_key | tr -d '\n')"
        echo "Loaded W&B API key from ~/.wandb_api_key"
    else
        echo "Warning: WANDB_API_KEY not set and ~/.wandb_api_key not found"
        echo "W&B logging may fail. To fix:"
        echo "  echo 'YOUR_API_KEY' > ~/.wandb_api_key"
    fi
fi
export WANDB_ENTITY="mliu7"
export PARAPHRASE_PROVIDER="GEMINI"
export OMP_NUM_THREADS=1
export HF_HOME="${SCRIPT_DIR}/.cache"

# Paraphrasability metric configuration
# PARAPHRASE_MODE options:
#   - "basic"             - Basic stable paraphrase, single paraphrase, same length, preserves meaning (RECOMMENDED for SFT stability)
#   - "moderate"           - Moderate synonym replacement, preserves numbers
#   - "simple_synonym"    - Simple synonym replacement
#   - "synonym_aggressive" - Aggressive synonym replacement (may be unstable)
# PARAPHRASE_FRACTIONS: comma-separated fractions for paraphrasing intensity
#   - Default: "0.10,0.5,0.98" (10%, 50%, 98% replacement)
#   - Recommended: "0.25,0.50,0.75" (moderate fractions for stable detection)
#   - Note: "basic" mode ignores this setting and uses a single stable paraphrase
export PARAPHRASE_MODE="${PARAPHRASE_MODE:-basic}"
export PARAPHRASE_FRACTIONS="${PARAPHRASE_FRACTIONS:-0.25,0.50,0.75}"

# Gemini API Key setup
# Store your key in ~/.gemini_api_key on the server:
#   echo "YOUR_API_KEY" > ~/.gemini_api_key
if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f ~/.gemini_api_key ]; then
        export GEMINI_API_KEY="$(cat ~/.gemini_api_key | tr -d '\n')"
        echo "Loaded Gemini API key from ~/.gemini_api_key"
    else
        echo "Warning: GEMINI_API_KEY not set and ~/.gemini_api_key not found"
        echo "Paraphrasability metric may fail. To fix:"
        echo "  echo 'YOUR_API_KEY' > ~/.gemini_api_key"
    fi
fi

# OpenAI API Key setup
# Store your key in ~/.openai_api_key on the server:
#   echo "YOUR_API_KEY" > ~/.openai_api_key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ~/.openai_api_key ]; then
        export OPENAI_API_KEY="$(cat ~/.openai_api_key | tr -d '\n')"
        echo "Loaded OpenAI API key from ~/.openai_api_key"
    fi
fi

# Create logs directory
mkdir -p logs

# Common arguments (customize these as needed)
# Define models to run (can be modified to run fewer/more models)
# Running only Olmo-3-7B-Think for post-hoc training
MODELS=("allenai/Olmo-3-7B-Think")

# Define datasets to run (can be modified to run fewer/more datasets)
# Running BA, CA, SB with post-hoc training type
DATASETS=("ba" "ca" "sb")

# Training configuration for accuracy optimization
# Optimized for 8 x 80GB H100 GPUs with mixed model sizes (20B and 7B)
NUM_EPOCHS=0.3                    # Single epoch for faster iteration
MAX_SAMPLES=5000                # Training samples
METRIC_EVAL_SAMPLES=100         # Eval samples for reliable accuracy measurement
MAX_NEW_TOKENS=4096             # Maximum tokens to generate during inference
                                # IMPORTANT: BA dataset has CoT up to ~3100 tokens, CA up to ~2800 tokens
                                # Setting to 4096 ensures no truncation and preserves accuracy
                                # Speed impact is minimal since most samples don't use full limit
BATCH_SIZE=12                    # Batch size for evaluation (conservative for 20B model on 80GB H100)
                                # 20B model needs more memory; 8 is safe for both models
# Note: --gradient_checkpointing is enabled below to reduce memory (~30-50% reduction)

# Number of checkpoints to track accuracy progression throughout training
# Note: Step 0 (baseline before training) is automatically evaluated
NUM_CHECKPOINTS=4               # Track at 25%, 50%, 75%, 100% of training (reduced from 5 for faster jobs)
                                # Total evaluations: 5 (step 0 + 4 checkpoints)

# Learning rate configuration per dataset
# All datasets use 5e-5
get_learning_rate() {
    local dataset=$1
    case "${dataset,,}" in
        ba|binary_alternation|ca|calendar_arithmetic|sb|spell_backward)
            echo "5e-5"
            ;;
        *)
            echo "5e-5"  # default
            ;;
    esac
}

# Filler type configuration for internalized training
# Options: lorem_ipsum, dots, think_token, number_words, mixed, not_relevant, shuffled
# 'not_relevant' swaps CoT with reasoning from a completely different task domain:
#   - binary_alternation → calendar_arithmetic
#   - calendar_arithmetic → spell_backward
#   - spell_backward → calendar_arithmetic
# 'shuffled' swaps CoT with reasoning from a different question in the SAME dataset
# 'mixed' uses a mix of different filler types for training diversity
# 'lorem_ipsum' uses standard Lorem ipsum placeholder text
# 'not_relevant' swaps CoT with reasoning from a completely different task domain
#
# Recommended for internalized training:
#   - TRAIN: "not_relevant" - swaps CoT with reasoning from a different task domain
#   - EVAL: "not_relevant" - swaps CoT with reasoning from a different task domain
FILLER_TYPE_TRAIN="${FILLER_TYPE_TRAIN:-not_relevant}"
FILLER_TYPE_EVAL="${FILLER_TYPE_EVAL:-not_relevant}"

# vLLM configuration for faster inference during evaluation
# Set USE_VLLM=true to enable vLLM (~2-3x faster evaluation)
# Note: vLLM is for inference only; training still uses HuggingFace Trainer
#
# Key optimization: vLLM engine is initialized ONCE and persists across all checkpoints.
# Only LoRA adapters are swapped for each checkpoint evaluation, saving ~30-60s per checkpoint.
#
# MEMORY CALCULATION for gpt-oss-20b on 80GB H100 GPU:
# ┌────────────────────────────────────────────────────────────────────┐
# │ Component                          │ Per Job │ Notes              │
# ├────────────────────────────────────┼─────────┼────────────────────┤
# │ Training Model (bf16 dequantized)  │  ~40GB  │ gpt-oss-20b*       │
# │ LoRA (rank 8) + Optimizer          │   ~2GB  │                    │
# │ Activations (batch=1, grad ckpt)   │   ~8GB  │ batch_size=1       │
# │ PyTorch buffers/cache              │   ~5GB  │                    │
# ├────────────────────────────────────┼─────────┼────────────────────┤
# │ Training peak                      │  ~55GB  │ with batch_size=1  │
# │ vLLM V0 engine (0.80 × 80GB)       │  ~64GB  │ Model + KV cache   │
# │   - Base model (4-bit)             │  ~10GB  │ gpt-oss-20b        │
# │   - KV cache (batch=8, seq=4096)   │  ~25GB  │ PagedAttention     │
# │   - vLLM overhead                  │   ~5GB  │                    │
# │   - Safety margin                  │  ~24GB  │ Buffer for growth  │
# ├────────────────────────────────────┼─────────┼────────────────────┤
# │ Peak usage                         │  ~64GB  │ ✓ fits 80GB safely │
# └────────────────────────────────────────────────────────────────────┘
# * NOTE: gpt-oss models use MXFP4 dequantization to bf16, NOT 4-bit NF4.
#   The --load_in_4bit flag is ignored for gpt-oss models in sft.py.
#   Training batch size is set to 1 via get_train_batch_size() to fit in memory.
#
# vLLM configuration for faster inference during evaluation
# Pattern from Obfuscation_Generalization: training model is offloaded to CPU
# during vLLM evaluation, then restored. This frees ~28GB GPU memory.
#
# With training model offloaded on 80GB H100 for gpt-oss-20b:
#   - 80GB GPU → ~78GB available for vLLM
#   - Use 0.80 GPU memory util with V0 engine (~64GB for vLLM)
#   - Batch size 8 provides good throughput while staying safe from OOM for 20B model
#   - KV cache scales with batch_size × sequence_length, so batch_size=8 is optimal for 20B
#
# Set USE_VLLM=true to enable vLLM (~2-3x faster evaluation)
# Set USE_VLLM=false to disable vLLM (slower but more stable)
USE_VLLM="${USE_VLLM:-true}"
# GPU memory for vLLM on 80GB H100 GPU with gpt-oss-20b:
#   - 0.70: Conservative, leaves room for larger models (~56GB for vLLM)
#   - 0.75: Balanced, good for 7B models (~60GB for vLLM)
#   - 0.80: Optimal for 20B model with batch_size=8 (~64GB for vLLM, ~16GB safety margin)
#   - 0.85: Aggressive, may cause OOM with very large batches (~68GB for vLLM)
# NOTE: With gpt-oss-20b, batch_size=8, and V0 engine, 0.80 is optimal for H100
# V0 engine has lower memory overhead than V1, but we keep margin for KV cache growth
VLLM_GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTIL:-0.80}"
# Tensor parallelism: Set to 1 (each job uses 1 GPU)
# With 8 GPUs, this allows 8 jobs to run in parallel
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
# Match training LoRA rank (8) to minimize vLLM memory overhead
VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-8}"

# vLLM engine configuration
# NOTE: model_vllm.py forces V0 engine (VLLM_USE_V1=0) for stability
# V0 engine is more stable and has lower memory overhead
# The script settings below are overridden by model_vllm.py
export VLLM_USE_V1=0
export VLLM_USE_LEGACY_EXECUTOR=1
export VLLM_DISABLE_ASYNC_OUTPUT_PROCESSOR=1

# Debug: Print vLLM environment settings
echo "vLLM Environment: VLLM_USE_V1=$VLLM_USE_V1 (V0 engine for stability)"

# GPU Configuration
# PARALLEL_MODE: Controls how jobs are distributed across GPUs
#   "auto"     = Auto-detect GPUs and run 1 job per GPU (safest, recommended)
#   "parallel" = Run multiple jobs per GPU (risky with vLLM V1)
#   "single"   = Run all jobs sequentially on GPU 0
# Using "auto" mode: 8 GPUs = 8 parallel jobs (1 job per GPU)
PARALLEL_MODE="${PARALLEL_MODE:-auto}"

# Jobs per GPU (only used when PARALLEL_MODE=parallel)
# With vLLM V1 (0.13+), use 1 job per GPU due to higher memory requirements
# 2 GPUs × 1 job = 2 parallel jobs; 4 LRs queued and run as jobs complete
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

# Check available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
    echo "Warning: Could not detect GPUs. Assuming 1 GPU."
    NUM_GPUS=1
fi

echo "Detected $NUM_GPUS GPU(s)"

# Calculate total parallel slots based on mode
if [ "$PARALLEL_MODE" == "single" ]; then
    echo "Running in SINGLE GPU SEQUENTIAL mode"
    echo "All jobs will run one at a time on GPU 0"
    TOTAL_SLOTS=1
    JOBS_PER_GPU=1
elif [ "$PARALLEL_MODE" == "parallel" ]; then
    echo "Running in PARALLEL mode ($JOBS_PER_GPU jobs per GPU)"
    echo "Total parallel slots: $((NUM_GPUS * JOBS_PER_GPU))"
    TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))
else
    # Auto mode: 1 job per GPU
    echo "Running in AUTO mode (1 job per GPU)"
    TOTAL_SLOTS=$NUM_GPUS
    JOBS_PER_GPU=1
fi

# Map dataset name to codebook path
# This is needed for ALL training types because the paraphrasability metric uses the encoded prompt
get_codebook_path() {
    local dataset=$1
    case "${dataset,,}" in
        ba|binary_alternation)
            echo "src/finetune/codebook_binary_alternation.py"
            ;;
        ca|calendar_arithmetic)
            echo "src/finetune/codebook_calendar_arithmetic.py"
            ;;
        li|largest_island)
            echo "src/finetune/codebook_largest_island.py"
            ;;
        sb|spell_backward)
            echo "src/finetune/codebook_spell_backward.py"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Get per-device training batch size based on model size
# Larger models need smaller batch sizes to avoid OOM during training
get_train_batch_size() {
    local model_name="$1"
    if [[ "$model_name" == *"20b"* ]] || [[ "$model_name" == *"20B"* ]]; then
        echo "1"  # 20B models need batch_size=1 due to bf16 dequantization (~40GB model)
    else
        echo "12"  # 7B and smaller models can use batch_size=12 on 80GB GPU
    fi
}

# Get evaluation batch size based on model size
# Larger models need smaller batch sizes to avoid OOM during vLLM evaluation
get_eval_batch_size() {
    local model_name="$1"
    if [[ "$model_name" == *"20b"* ]] || [[ "$model_name" == *"20B"* ]]; then
        echo "2"  # 20B models need smaller eval batch size due to KV cache memory
    else
        echo "$BATCH_SIZE"  # Smaller models use default BATCH_SIZE (8)
    fi
}

# Get memory limit arguments for large models
# Only needed for 20B+ models where bf16 dequantization requires more GPU/CPU memory
get_memory_args() {
    local model_name="$1"
    if [[ "$model_name" == *"20b"* ]] || [[ "$model_name" == *"20B"* ]]; then
        echo "--max_gpu_memory 76GiB --max_cpu_memory 150GiB"
    else
        echo ""  # Smaller models use defaults (faster, no offloading overhead)
    fi
}

# Function to run training on a specific GPU with a specific dataset
run_training_on_gpu() {
    local gpu_id=$1
    local model=$2
    local training_type=$3
    local dataset_name=$4
    local learning_rate=${5:-2e-5}  # Default learning rate if not provided

    # Extract short model name for logging/paths
    local model_short="${model##*/}"

    echo "=========================================="
    echo "Starting $training_type training for dataset $dataset_name on GPU $gpu_id"
    echo "Model: $model"
    echo "Learning rate: $learning_rate"
    echo "=========================================="

    # Get codebook path for ALL training types (needed for paraphrasability metric)
    local codebook_path=$(get_codebook_path "$dataset_name")
    local codebook_args=""
    if [ -n "$codebook_path" ]; then
        codebook_args="--codebook_path $codebook_path"
        echo "Using codebook: $codebook_path"
    fi

    # Add filler type arguments
    # filler_type_eval is used by ALL training types for the Substantivity metric
    # filler_type_train is only used by internalized training
    local filler_args="--filler_type_eval $FILLER_TYPE_EVAL"
    echo "Using filler_type_eval: $FILLER_TYPE_EVAL (for Substantivity metric)"

    if [ "$training_type" == "internalized" ]; then
        filler_args="--filler_type_train $FILLER_TYPE_TRAIN --filler_type_eval $FILLER_TYPE_EVAL"
        echo "Using filler_type_train: $FILLER_TYPE_TRAIN (for internalized training)"
    fi

    # Add vLLM arguments if enabled
    local vllm_args=""
    if [ "$USE_VLLM" == "true" ]; then
        vllm_args="--use_vllm --vllm_gpu_memory_util $VLLM_GPU_MEMORY_UTIL --vllm_tensor_parallel_size $VLLM_TENSOR_PARALLEL_SIZE --vllm_max_lora_rank $VLLM_MAX_LORA_RANK"
        echo "Using vLLM for evaluation (GPU memory: ${VLLM_GPU_MEMORY_UTIL})"
    fi

    # Get model-specific settings for memory optimization
    local train_batch_size=$(get_train_batch_size "$model")
    local eval_batch_size=$(get_eval_batch_size "$model")
    local memory_args=$(get_memory_args "$model")
    echo "Training batch size: $train_batch_size"
    echo "Eval batch size: $eval_batch_size"
    if [ -n "$memory_args" ]; then
        echo "Memory args (large model): $memory_args"
    fi

    # Format learning rate for log filename (replace scientific notation)
    local lr_label=$(echo "$learning_rate" | sed 's/e-/e_minus_/g' | sed 's/e+/e_plus_/g')

    # Generate timestamp for output directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local output_dir="output/${training_type}_${model_short}_${dataset_name}_lr${lr_label}_${timestamp}"

    CUDA_VISIBLE_DEVICES=$gpu_id python src/finetune/sft.py \
        --model $model \
        --output_dir "$output_dir" \
        --load_in_4bit \
        --dataset_name $dataset_name \
        --bf16 \
        --gradient_checkpointing \
        --per_device_train_batch_size $train_batch_size \
        --num_train_epochs $NUM_EPOCHS \
        --track_metrics \
        --max_samples $MAX_SAMPLES \
        --metric_eval_samples $METRIC_EVAL_SAMPLES \
        --num_checkpoints $NUM_CHECKPOINTS \
        --max_length 4096 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --batch_size $eval_batch_size \
        --use_lora \
        --lora_r 8 \
        --lora_alpha 32 \
        --learning_rate $learning_rate \
        --training_type $training_type \
        $codebook_args \
        $filler_args \
        $vllm_args \
        $memory_args \
        --wandb_project ${training_type}-lr-sweep-${model_short}-${dataset_name} \
        --run_name "${training_type}_${model_short}_${dataset_name}_lr${learning_rate}" \
        2>&1 | tee logs/${training_type}_${model_short}_${dataset_name}_lr${lr_label}_gpu${gpu_id}_${timestamp}.log

    local exit_code=${PIPESTATUS[0]}
    echo "=========================================="
    echo "$training_type training for $dataset_name with $model_short (lr=$learning_rate) on GPU $gpu_id completed with exit code: $exit_code"
    echo "=========================================="
    return $exit_code
}

# Export function and variables for background processes
export -f run_training_on_gpu get_codebook_path get_learning_rate get_train_batch_size get_eval_batch_size get_memory_args
export NUM_EPOCHS MAX_SAMPLES METRIC_EVAL_SAMPLES NUM_CHECKPOINTS MAX_NEW_TOKENS BATCH_SIZE FILLER_TYPE_TRAIN FILLER_TYPE_EVAL
export USE_VLLM VLLM_GPU_MEMORY_UTIL VLLM_TENSOR_PARALLEL_SIZE VLLM_MAX_LORA_RANK VLLM_USE_V1 VLLM_USE_LEGACY_EXECUTOR VLLM_DISABLE_ASYNC_OUTPUT_PROCESSOR
export PARALLEL_MODE JOBS_PER_GPU NUM_GPUS TOTAL_SLOTS
export SCRIPT_DIR PYTHONPATH PYTORCH_ALLOC_CONF WANDB_API_KEY WANDB_ENTITY PARAPHRASE_PROVIDER PARAPHRASE_FRACTIONS PARAPHRASE_MODE OMP_NUM_THREADS HF_HOME GEMINI_API_KEY OPENAI_API_KEY

# Training types to run
# Options: baseline, internalized, encoded, post-hoc
TRAINING_TYPES=("baseline" "internalized" "encoded" "post-hoc")

# Create job queue: all combinations of models × datasets × training_types with per-dataset learning rates
# Format: model:dataset:training_type:learning_rate
declare -a JOB_QUEUE=()
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        lr=$(get_learning_rate "$dataset")
        for training_type in "${TRAINING_TYPES[@]}"; do
            JOB_QUEUE+=("${model}:${dataset}:${training_type}:${lr}")
        done
    done
done

TOTAL_JOBS=${#JOB_QUEUE[@]}
echo "=========================================="
echo "MULTI-MODEL TRAINING RUN"
echo "Goal: Train multiple models across datasets"
echo "=========================================="
echo ""
echo "Job Configuration:"
echo "  Models: ${MODELS[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Training types: ${TRAINING_TYPES[*]}"
echo "  Learning rates: 5e-5 (all datasets)"
echo "  Total jobs: $TOTAL_JOBS (${#MODELS[@]} models × ${#DATASETS[@]} datasets × ${#TRAINING_TYPES[@]} types)"
echo ""
echo "Training Configuration:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Training samples: $MAX_SAMPLES"
echo "  Eval samples: $METRIC_EVAL_SAMPLES"
echo "  Batch size: $BATCH_SIZE (optimized for Qwen3-4B on 80GB GPUs, safe from OOM)"
echo "  Max new tokens: $MAX_NEW_TOKENS (reduced from 4096 for speed)"
echo "  Checkpoints: $NUM_CHECKPOINTS (for tracking accuracy over training)"
echo "  LoRA rank: 8, alpha: 32"
echo "  Gradient checkpointing: enabled (reduces ~30-50% activation memory)"
echo ""
echo "Hardware Configuration:"
echo "  GPUs: $NUM_GPUS"
echo "  Mode: $PARALLEL_MODE"
echo "  Jobs per GPU: $JOBS_PER_GPU"
echo "  Total parallel slots: $TOTAL_SLOTS"
if [ "$USE_VLLM" == "true" ]; then
    echo "  vLLM enabled: yes (GPU memory: ${VLLM_GPU_MEMORY_UTIL}, optimized for Qwen3-4B)"
    echo "  vLLM: Persistent engine (adapters swapped per checkpoint)"
    echo "  vLLM: Batch size $BATCH_SIZE for efficient continuous batching (safe from OOM)"
    echo "  vLLM: Max new tokens: $MAX_NEW_TOKENS"
    echo "  vLLM: Tensor parallel size: ${VLLM_TENSOR_PARALLEL_SIZE} (1 GPU per job)"
    echo "  vLLM: V0 engine (lower memory overhead, more stable)"
else
    echo "  vLLM enabled: no (using HuggingFace Transformers)"
fi
echo ""
echo "Execution Plan:"
echo "  - $TOTAL_SLOTS jobs run in parallel across $NUM_GPUS GPU(s)"
echo "  - Remaining jobs queued and start as slots free up"
echo "  - Step 0 baseline evaluated BEFORE training starts"
echo "  - Accuracy tracked at $NUM_CHECKPOINTS checkpoints per run"
echo "  - Estimated time per job: 1.5-3 hours (training + evaluation)"
echo "    - Training: ~30-90 min (5000 samples with LoRA)"
echo "    - Evaluation: ~30-60 min (5 checkpoints × 100 samples)"
if [ "$USE_VLLM" == "true" ]; then
    echo "  - vLLM saves ~30-60s per checkpoint evaluation"
fi
echo ""
echo "To find best accuracy after training:"
echo "  bash run_parallel_gpu_lambda.sh --summarize"
echo "=========================================="
echo ""

# Job queue management
# Track jobs using slots (multiple slots can map to same GPU)
declare -a SLOT_PIDS=()
declare -a SLOT_JOBS=()
declare -a SLOT_GPU=()

# Initialize slot tracking arrays
# Slots are distributed across GPUs in round-robin fashion
for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
    SLOT_PIDS[$slot]=""
    SLOT_JOBS[$slot]=""
    # Map slot to GPU (round-robin)
    SLOT_GPU[$slot]=$((slot % NUM_GPUS))
done

echo "Slot to GPU mapping:"
for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
    echo "  Slot $slot -> GPU ${SLOT_GPU[$slot]}"
done
echo ""

JOB_INDEX=0
COMPLETED_JOBS=0
FAILED_JOBS=0
declare -a FAILED_JOB_LIST=()

# Function to assign next job to an available slot
assign_job_to_slot() {
    local slot_id=$1
    local gpu_id=${SLOT_GPU[$slot_id]}

    if [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
        local job="${JOB_QUEUE[$JOB_INDEX]}"
        local model=$(echo "$job" | cut -d':' -f1)
        local dataset=$(echo "$job" | cut -d':' -f2)
        local training_type=$(echo "$job" | cut -d':' -f3)
        local learning_rate=$(echo "$job" | cut -d':' -f4)
        local model_short="${model##*/}"

        echo "[Slot $slot_id/GPU $gpu_id] Starting job $((JOB_INDEX + 1))/$TOTAL_JOBS: $training_type for $dataset with $model_short (lr=$learning_rate)"

        # Run job in background with model and learning rate
        run_training_on_gpu $gpu_id "$model" "$training_type" "$dataset" "$learning_rate" &
        local pid=$!

        SLOT_PIDS[$slot_id]=$pid
        SLOT_JOBS[$slot_id]="$job"
        JOB_INDEX=$((JOB_INDEX + 1))

        return 0
    fi
    return 1
}

# Start initial jobs (one per slot)
for ((slot=0; slot<TOTAL_SLOTS && slot<TOTAL_JOBS; slot++)); do
    assign_job_to_slot $slot
done

echo ""
echo "Initial jobs started. Monitoring progress..."
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
echo "Check logs in: logs/"
if [ $JOBS_PER_GPU -gt 1 ]; then
    echo ""
    echo "Running $JOBS_PER_GPU jobs per GPU in PARALLEL"
    echo "If you see OOM errors, set JOBS_PER_GPU=1 or PARALLEL_MODE=single"
fi
if [ -n "$TMUX" ]; then
    echo ""
    echo "Running in tmux session. To detach: Press Ctrl+B, then D"
    echo "To reattach later: tmux attach -t $TMUX_SESSION"
fi
echo ""
echo "After training, run: bash $0 --summarize"
echo ""

# Monitor and manage job queue
while [ $COMPLETED_JOBS -lt $TOTAL_JOBS ]; do
    for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
        if [ -n "${SLOT_PIDS[$slot]}" ]; then
            # Check if process is still running
            if ! kill -0 "${SLOT_PIDS[$slot]}" 2>/dev/null; then
                # Job completed
                wait "${SLOT_PIDS[$slot]}"
                local_exit_code=$?
                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                
                completed_job="${SLOT_JOBS[$slot]}"
                gpu_id=${SLOT_GPU[$slot]}
                model=$(echo "$completed_job" | cut -d':' -f1)
                dataset=$(echo "$completed_job" | cut -d':' -f2)
                training_type=$(echo "$completed_job" | cut -d':' -f3)
                learning_rate=$(echo "$completed_job" | cut -d':' -f4)
                model_short="${model##*/}"

                if [ $local_exit_code -eq 0 ]; then
                    echo "[Slot $slot/GPU $gpu_id] ✓ Completed: $training_type for $dataset with $model_short (lr=$learning_rate) - Progress: $COMPLETED_JOBS/$TOTAL_JOBS"
                else
                    echo "[Slot $slot/GPU $gpu_id] ✗ Failed: $training_type for $dataset with $model_short (lr=$learning_rate, exit: $local_exit_code) - Progress: $COMPLETED_JOBS/$TOTAL_JOBS"
                    FAILED_JOBS=$((FAILED_JOBS + 1))
                    FAILED_JOB_LIST+=("$model:$dataset:$training_type:$learning_rate")
                fi
                
                # Assign next job to this slot
                SLOT_PIDS[$slot]=""
                SLOT_JOBS[$slot]=""
                if assign_job_to_slot $slot; then
                    # New job assigned
                    :
                fi
            fi
        else
            # Slot is idle, try to assign a job
            if [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
                assign_job_to_slot $slot
            fi
        fi
    done
    
    # Sleep briefly to avoid busy waiting
    sleep 2
done

# Wait for any remaining jobs to finish
for ((slot=0; slot<TOTAL_SLOTS; slot++)); do
    if [ -n "${SLOT_PIDS[$slot]}" ]; then
        wait "${SLOT_PIDS[$slot]}"
        local_exit_code=$?
        if [ $local_exit_code -ne 0 ]; then
            FAILED_JOBS=$((FAILED_JOBS + 1))
            FAILED_JOB_LIST+=("${SLOT_JOBS[$slot]}")
        fi
    fi
done

echo ""
echo "=========================================="
echo "All $TOTAL_JOBS training jobs completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Total jobs: $TOTAL_JOBS"
echo "  Successful: $((TOTAL_JOBS - FAILED_JOBS))"
echo "  Failed: $FAILED_JOBS"

if [ $FAILED_JOBS -gt 0 ]; then
    echo ""
    echo "Failed jobs:"
    for job in "${FAILED_JOB_LIST[@]}"; do
        echo "  - $job"
    done
    echo ""
    echo "Check logs for details."
    exit 1
fi

echo ""
echo "All jobs completed successfully!"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Find best accuracy for each dataset:"
echo "   bash run_parallel_gpu_lambda.sh --summarize"
echo ""
echo "2. Calculate metrics and Cohen's d for best LR:"
echo "   # After identifying best LR, calculate Cohen's d:"
echo "   python src/calculate_cohens_d.py --dataset BA"
echo "   python src/calculate_cohens_d.py --dataset SB"
echo "   # Or process all:"
echo "   python src/calculate_cohens_d.py --all"
echo ""
echo "3. Generate accuracy and metrics plots:"
echo "   python src/plot_metrics.py --dataset BA"
echo "   python src/plot_metrics.py --dataset SB"
echo ""
echo "4. Plots will be saved to: output/metrics_plots/"
echo ""
echo "Expected output files:"
echo "  - output/metrics_results_{dataset}.json  (metrics data)"
echo "  - output/cohens_d_results_{dataset}.csv  (Cohen's d data)"
echo "  - output/metrics_plots/accuracy_{dataset}.png"
echo "  - output/metrics_plots/metric_*.png"
echo "=========================================="
exit 0