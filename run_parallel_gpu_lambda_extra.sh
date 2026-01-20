#!/bin/bash

# This script runs post-hoc and internalized training jobs on separate GPUs
# without interfering with the main training session.
#
# Usage:
#   bash run_parallel_gpu_lambda_extra.sh --detach   # Run in a separate tmux session
#   bash run_parallel_gpu_lambda_extra.sh --attach   # Attach to the extra training session

# Use a DIFFERENT session name to avoid conflicts with the main training
TMUX_SESSION="training-extra"

# Parse arguments
DETACH_MODE=0
ATTACH_MODE=0
FORCE_YES=0

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
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--detach|-d] [--attach|-a] [--yes|-y]"
            exit 1
            ;;
    esac
done

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
    if ! command -v tmux &> /dev/null; then
        echo "Error: tmux is not installed. Install it with: sudo apt install tmux"
        exit 1
    fi
    
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Warning: tmux session '$TMUX_SESSION' already exists."
        echo "Use '$0 --attach' to attach to it, or kill it with: tmux kill-session -t $TMUX_SESSION"
        exit 1
    fi
    
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    
    echo "=========================================="
    echo "Starting EXTRA training in detached tmux session"
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
    
    tmux new-session -d -s "$TMUX_SESSION" "bash '$SCRIPT_PATH' --yes; echo ''; echo 'Training completed. Press Enter to close.'; read"
    
    echo ""
    echo "✓ Extra training started in background tmux session."
    echo "  You can safely disconnect from SSH now."
    exit 0
fi

# Set up environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d "myenv" ]; then
    source myenv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
else
    echo "Virtual environment not found. Creating myenv..."
    python3 -m venv myenv
    source myenv/bin/activate
fi

if ! command -v python &> /dev/null; then
    echo "Error: Python not found."
    exit 1
fi

# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export WANDB_ENTITY="mliu7"
export PARAPHRASE_PROVIDER="GEMINI"
export OMP_NUM_THREADS=1
export HF_HOME="${SCRIPT_DIR}/.cache"

if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f ~/.gemini_api_key ]; then
        export GEMINI_API_KEY="$(cat ~/.gemini_api_key | tr -d '\n')"
    else
        export GEMINI_API_KEY="AIzaSyAE7dNT6Cr6Nt3Yy7JdaX33IHDBwPOoYNI"
    fi
fi

mkdir -p logs

# ============================================
# CONFIGURATION - Modify these as needed
# ============================================
MODEL="Qwen/Qwen3-4B"
DATASETS=("ba")  # Same dataset as main training
NUM_EPOCHS=1
MAX_SAMPLES=5000
METRIC_EVAL_SAMPLES=100

# Filler type configuration
FILLER_TYPE_TRAIN="${FILLER_TYPE_TRAIN:-not_relevant}"
FILLER_TYPE_EVAL="${FILLER_TYPE_EVAL:-not_relevant}"

# ONLY run post-hoc and internalized (the ones not in the main session)
TRAINING_TYPES=("internalized" "post-hoc")

# Use GPUs 0 and 1 (will wait for them to become available)
START_GPU=0
MAX_GPUS=2  # Only use 2 GPUs
# ============================================

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# Limit to MAX_GPUS if set
if [ -n "$MAX_GPUS" ] && [ $NUM_GPUS -gt $MAX_GPUS ]; then
    NUM_GPUS=$MAX_GPUS
fi

# Function to check if a GPU is in use (memory > 500MB indicates active use)
gpu_is_busy() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
    if [ -z "$mem_used" ]; then
        return 1  # Can't check, assume not busy
    fi
    [ "$mem_used" -gt 500 ]
}

# Wait for GPUs to become available
wait_for_gpus() {
    echo "=========================================="
    echo "Checking if GPUs $START_GPU to $((NUM_GPUS-1)) are available..."
    echo "=========================================="
    
    local any_busy=1
    while [ $any_busy -eq 1 ]; do
        any_busy=0
        for ((gpu=START_GPU; gpu<NUM_GPUS; gpu++)); do
            if gpu_is_busy $gpu; then
                any_busy=1
                break
            fi
        done
        
        if [ $any_busy -eq 1 ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') - GPUs still in use. Waiting 60 seconds..."
            echo "  Current GPU status:"
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv | head -$((NUM_GPUS+1))
            sleep 60
        fi
    done
    
    echo ""
    echo "✓ GPUs $START_GPU to $((NUM_GPUS-1)) are now available!"
    echo ""
}

# Wait for GPUs before proceeding
wait_for_gpus
echo "=========================================="
echo "EXTRA Training Session Configuration:"
echo "  Training types: ${TRAINING_TYPES[*]}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Starting from GPU: $START_GPU"
echo "  Total GPUs available: $NUM_GPUS"
echo "  Filler type (train): $FILLER_TYPE_TRAIN"
echo "  Filler type (eval): $FILLER_TYPE_EVAL"
echo "=========================================="
echo ""

get_codebook_path() {
    local dataset=$1
    case "${dataset,,}" in
        ba|binary_alternation) echo "src/finetune/codebook_binary_alternation.py" ;;
        ca|calendar_arithmetic) echo "src/finetune/codebook_calendar_arithmetic.py" ;;
        li|largest_island) echo "src/finetune/codebook_largest_island.py" ;;
        sb|spell_backward) echo "src/finetune/codebook_spell_backward.py" ;;
        *) echo "" ;;
    esac
}

run_training_on_gpu() {
    local gpu_id=$1
    local training_type=$2
    local dataset_name=$3
    
    echo "=========================================="
    echo "Starting $training_type training for dataset $dataset_name on GPU $gpu_id"
    echo "=========================================="
    
    local codebook_path=$(get_codebook_path "$dataset_name")
    local codebook_args=""
    if [ -n "$codebook_path" ]; then
        codebook_args="--codebook_path $codebook_path"
    fi
    
    local filler_args="--filler_type_eval $FILLER_TYPE_EVAL"
    if [ "$training_type" == "internalized" ]; then
        filler_args="--filler_type_train $FILLER_TYPE_TRAIN --filler_type_eval $FILLER_TYPE_EVAL"
    fi
    
    CUDA_VISIBLE_DEVICES=$gpu_id python src/finetune/sft.py \
        --model $MODEL \
        --load_in_4bit \
        --dataset_name $dataset_name \
        --bf16 \
        --num_train_epochs $NUM_EPOCHS \
        --track_metrics \
        --max_samples $MAX_SAMPLES \
        --metric_eval_samples $METRIC_EVAL_SAMPLES \
        --use_lora \
        --lora_r 4 \
        --lora_alpha 32 \
        --training_type $training_type \
        $codebook_args \
        $filler_args \
        --wandb_project ${training_type}-training-parallel-${MODEL##*/}-${dataset_name} \
        2>&1 | tee logs/${training_type}_${dataset_name}_gpu${gpu_id}_$(date +%Y%m%d_%H%M%S).log
    
    local exit_code=${PIPESTATUS[0]}
    echo "=========================================="
    echo "$training_type for $dataset_name on GPU $gpu_id completed (exit: $exit_code)"
    echo "=========================================="
    return $exit_code
}

export -f run_training_on_gpu get_codebook_path
export MODEL NUM_EPOCHS MAX_SAMPLES METRIC_EVAL_SAMPLES FILLER_TYPE_TRAIN FILLER_TYPE_EVAL
export SCRIPT_DIR PYTHONPATH PYTORCH_ALLOC_CONF WANDB_ENTITY PARAPHRASE_PROVIDER OMP_NUM_THREADS HF_HOME GEMINI_API_KEY

# Create job queue
declare -a JOB_QUEUE=()
for dataset in "${DATASETS[@]}"; do
    for training_type in "${TRAINING_TYPES[@]}"; do
        JOB_QUEUE+=("${dataset}:${training_type}")
    done
done

TOTAL_JOBS=${#JOB_QUEUE[@]}
AVAILABLE_GPUS=$((NUM_GPUS - START_GPU))

echo "Job Queue: ${JOB_QUEUE[*]}"
echo "Total jobs: $TOTAL_JOBS"
echo "GPUs to use: $AVAILABLE_GPUS (GPUs $START_GPU to $((NUM_GPUS-1)))"
echo ""

# Job management
declare -a GPU_PIDS=()
declare -a GPU_JOBS=()
declare -a GPU_EXIT_CODES=()

for ((gpu=START_GPU; gpu<NUM_GPUS; gpu++)); do
    GPU_PIDS[$gpu]=""
    GPU_JOBS[$gpu]=""
    GPU_EXIT_CODES[$gpu]=0
done

JOB_INDEX=0
COMPLETED_JOBS=0

assign_job_to_gpu() {
    local gpu_id=$1
    
    if [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
        local job="${JOB_QUEUE[$JOB_INDEX]}"
        local dataset=$(echo "$job" | cut -d':' -f1)
        local training_type=$(echo "$job" | cut -d':' -f2)
        
        echo "[GPU $gpu_id] Starting job $((JOB_INDEX + 1))/$TOTAL_JOBS: $training_type for $dataset"
        
        run_training_on_gpu $gpu_id "$training_type" "$dataset" &
        local pid=$!
        
        GPU_PIDS[$gpu_id]=$pid
        GPU_JOBS[$gpu_id]="$job"
        JOB_INDEX=$((JOB_INDEX + 1))
        
        return 0
    fi
    return 1
}

# Start initial jobs on available GPUs
gpu_count=0
for ((gpu=START_GPU; gpu<NUM_GPUS && gpu_count<TOTAL_JOBS; gpu++)); do
    assign_job_to_gpu $gpu
    gpu_count=$((gpu_count + 1))
done

echo ""
echo "Initial jobs started. Monitoring progress..."
echo ""

# Monitor job queue
while [ $COMPLETED_JOBS -lt $TOTAL_JOBS ]; do
    for ((gpu=START_GPU; gpu<NUM_GPUS; gpu++)); do
        if [ -n "${GPU_PIDS[$gpu]}" ]; then
            if ! kill -0 "${GPU_PIDS[$gpu]}" 2>/dev/null; then
                wait "${GPU_PIDS[$gpu]}"
                GPU_EXIT_CODES[$gpu]=$?
                COMPLETED_JOBS=$((COMPLETED_JOBS + 1))
                
                completed_job="${GPU_JOBS[$gpu]}"
                if [ ${GPU_EXIT_CODES[$gpu]} -eq 0 ]; then
                    echo "[GPU $gpu] ✓ Completed: $completed_job"
                else
                    echo "[GPU $gpu] ✗ Failed: $completed_job (exit: ${GPU_EXIT_CODES[$gpu]})"
                fi
                
                GPU_PIDS[$gpu]=""
                GPU_JOBS[$gpu]=""
                assign_job_to_gpu $gpu
            fi
        fi
    done
    sleep 2
done

echo ""
echo "=========================================="
echo "All $TOTAL_JOBS extra training jobs completed!"
echo "=========================================="
exit 0
