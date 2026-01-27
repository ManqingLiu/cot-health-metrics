#!/bin/bash

# Run ONLY gpt-oss-20b for BA, CA, SB with baseline training
# Uses a separate tmux session to avoid conflicts with main training
# Only uses GPUs that are completely free (< 1GB memory used)

TMUX_SESSION="training-gpt-oss"

# Parse arguments
if [[ "$1" == "--detach" ]] || [[ "$1" == "-d" ]]; then
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        echo "Session '$TMUX_SESSION' already exists. Attach with: tmux attach -t $TMUX_SESSION"
        exit 1
    fi
    SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "$TMUX_SESSION" "bash '$SCRIPT_PATH' --run"
    echo "Started in tmux session: $TMUX_SESSION"
    echo "Attach: tmux attach -t $TMUX_SESSION"
    exit 0
elif [[ "$1" == "--attach" ]] || [[ "$1" == "-a" ]]; then
    tmux attach -t "$TMUX_SESSION"
    exit 0
elif [[ "$1" != "--run" ]]; then
    echo "Usage: $0 [--detach|-d] [--attach|-a]"
    exit 0
fi

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6
# Load W&B API key from file or environment
if [ -z "$WANDB_API_KEY" ]; then
    if [ -f ~/.wandb_api_key ]; then
        export WANDB_API_KEY="$(cat ~/.wandb_api_key | tr -d '\n')"
    else
        echo "Warning: WANDB_API_KEY not set and ~/.wandb_api_key not found"
    fi
fi
export WANDB_ENTITY="mliu7"
export PARAPHRASE_PROVIDER="OPENAI"
export OMP_NUM_THREADS=1
export HF_HOME="${SCRIPT_DIR}/.cache"
export VLLM_USE_V1=0
export PARAPHRASE_MODE="${PARAPHRASE_MODE:-basic}"

# Safe CUDA cache cleanup (doesn't kill other processes)
clear_cuda_cache() {
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
}

# Safe cleanup - only clears cache
echo "Clearing CUDA cache..."
clear_cuda_cache

if [ -f ~/.gemini_api_key ]; then
    export GEMINI_API_KEY="$(cat ~/.gemini_api_key | tr -d '\n')"
fi

mkdir -p logs

# Configuration - ONLY gpt-oss-20b
MODEL="openai/gpt-oss-20b"
DATASETS=("li")
TRAINING_TYPES=("baseline" "internalized" "encoded" "post-hoc")

# Training settings - Optimized to prevent OOM for 20B model
NUM_EPOCHS=1
MAX_SAMPLES=5000
METRIC_EVAL_SAMPLES=100
NUM_CHECKPOINTS=4
MAX_LENGTH=4096           # Reduced from 4096 to save memory
MAX_NEW_TOKENS=4096       # Reduced from 4096 to save memory
BATCH_SIZE=1              # Reduced from 2 to avoid OOM during vLLM evaluation
LEARNING_RATE="5e-5"
FILLER_TYPE_TRAIN="not_relevant"
FILLER_TYPE_EVAL="not_relevant"
USE_VLLM="true"
VLLM_GPU_MEMORY_UTIL="0.70"  # Reduced from 0.80 to leave headroom
MAX_GPU_MEMORY="70GiB"       # Reduced from 76GiB to leave headroom
MAX_CPU_MEMORY="180GiB"      # Increased CPU offload capacity

get_codebook_path() {
    case "${1,,}" in
        ba|binary_alternation) echo "src/finetune/codebook_binary_alternation.py" ;;
        ca|calendar_arithmetic) echo "src/finetune/codebook_calendar_arithmetic.py" ;;
        sb|spell_backward) echo "src/finetune/codebook_spell_backward.py" ;;
        *) echo "" ;;
    esac
}

run_job() {
    local gpu_id=$1
    local training_type=$2
    local dataset=$3

    local codebook_path=$(get_codebook_path "$dataset")
    local codebook_args=""
    [ -n "$codebook_path" ] && codebook_args="--codebook_path $codebook_path"

    local filler_args="--filler_type_eval $FILLER_TYPE_EVAL"
    [ "$training_type" == "internalized" ] && filler_args="--filler_type_train $FILLER_TYPE_TRAIN $filler_args"

    local vllm_args=""
    [ "$USE_VLLM" == "true" ] && vllm_args="--use_vllm --vllm_gpu_memory_util $VLLM_GPU_MEMORY_UTIL"

    local timestamp=$(date +%Y%m%d_%H%M%S)
    # Format learning rate for log filename (replace scientific notation)
    local lr_label=$(echo "$LEARNING_RATE" | sed 's/e-/e_minus_/g' | sed 's/e+/e_plus_/g')
    local output_dir="output/${training_type}_gpt-oss-20b_${dataset}_lr${lr_label}_${timestamp}"

    echo "[GPU $gpu_id] Starting $training_type for $dataset (lr=$LEARNING_RATE)"

    # Clear GPU cache before starting each job
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

    CUDA_VISIBLE_DEVICES=$gpu_id python src/finetune/sft.py \
        --model $MODEL \
        --output_dir "$output_dir" \
        --load_in_4bit \
        --dataset_name $dataset \
        --bf16 \
        --gradient_checkpointing \
        --per_device_train_batch_size 1 \
        --num_train_epochs $NUM_EPOCHS \
        --track_metrics \
        --max_samples $MAX_SAMPLES \
        --metric_eval_samples $METRIC_EVAL_SAMPLES \
        --num_checkpoints $NUM_CHECKPOINTS \
        --max_length $MAX_LENGTH \
        --max_new_tokens $MAX_NEW_TOKENS \
        --batch_size $BATCH_SIZE \
        --use_lora \
        --lora_r 8 \
        --lora_alpha 32 \
        --learning_rate $LEARNING_RATE \
        --training_type $training_type \
        --max_gpu_memory $MAX_GPU_MEMORY --max_cpu_memory $MAX_CPU_MEMORY \
        $codebook_args $filler_args $vllm_args \
        --wandb_project ${training_type}-lr-sweep-gpt-oss-20b-${dataset} \
        --run_name "${training_type}_gpt-oss-20b_${dataset}_lr${LEARNING_RATE}" \
        2>&1 | tee logs/${training_type}_gpt-oss-20b_${dataset}_lr${lr_label}_gpu${gpu_id}_${timestamp}.log

    return ${PIPESTATUS[0]}
}

export -f run_job get_codebook_path clear_cuda_cache is_gpu_free get_free_gpus
export MODEL DATASETS TRAINING_TYPES NUM_EPOCHS MAX_SAMPLES METRIC_EVAL_SAMPLES NUM_CHECKPOINTS
export MAX_LENGTH MAX_NEW_TOKENS BATCH_SIZE LEARNING_RATE FILLER_TYPE_TRAIN FILLER_TYPE_EVAL USE_VLLM VLLM_GPU_MEMORY_UTIL
export MAX_GPU_MEMORY MAX_CPU_MEMORY
export SCRIPT_DIR PYTHONPATH PYTORCH_CUDA_ALLOC_CONF WANDB_ENTITY PARAPHRASE_PROVIDER OMP_NUM_THREADS HF_HOME GEMINI_API_KEY

# Build job queue: 3 datasets × 2 training types = 6 jobs
declare -a JOBS=()
for dataset in "${DATASETS[@]}"; do
    for tt in "${TRAINING_TYPES[@]}"; do
        JOBS+=("$dataset:$tt")
    done
done

NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
TOTAL_JOBS=${#JOBS[@]}

echo "=========================================="
echo "GPT-OSS-20B Only Training"
echo "=========================================="
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[*]}"
echo "Training types: ${TRAINING_TYPES[*]}"
echo "Total jobs: $TOTAL_JOBS"
echo "Available GPUs: $NUM_GPUS"
echo "=========================================="

# Function to check if a GPU is free (less than 1GB memory used)
is_gpu_free() {
    local gpu_id=$1
    local mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d ' ')
    if [ -z "$mem_used" ]; then
        return 1  # Can't query GPU, assume not free
    fi
    # Consider GPU free if less than 1000 MiB used
    if [ "$mem_used" -lt 1000 ]; then
        return 0  # Free
    else
        return 1  # In use
    fi
}

# Function to get list of free GPUs
get_free_gpus() {
    local free_gpus=()
    for ((g=0; g<NUM_GPUS; g++)); do
        if is_gpu_free $g; then
            free_gpus+=($g)
        fi
    done
    echo "${free_gpus[@]}"
}

# Show initial GPU status
echo ""
echo "Checking GPU availability..."
for ((g=0; g<NUM_GPUS; g++)); do
    mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g 2>/dev/null | tr -d ' ')
    if is_gpu_free $g; then
        echo "  GPU $g: FREE (${mem_used} MiB used)"
    else
        echo "  GPU $g: IN USE (${mem_used} MiB used)"
    fi
done
echo ""

# Job queue management
declare -a GPU_PIDS=()
declare -a GPU_JOBS=()
JOB_INDEX=0
COMPLETED=0

for ((g=0; g<NUM_GPUS; g++)); do
    GPU_PIDS[$g]=""
    GPU_JOBS[$g]=""
done

assign_job() {
    local gpu=$1
    # Check GPU is free before assigning
    if ! is_gpu_free $gpu; then
        return 1  # GPU not free, don't assign
    fi
    if [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
        local job="${JOBS[$JOB_INDEX]}"
        local dataset=$(echo "$job" | cut -d':' -f1)
        local tt=$(echo "$job" | cut -d':' -f2)

        run_job $gpu "$tt" "$dataset" &
        GPU_PIDS[$gpu]=$!
        GPU_JOBS[$gpu]="$job"
        JOB_INDEX=$((JOB_INDEX + 1))
        return 0
    fi
    return 1
}

# Start initial jobs only on FREE GPUs
echo "Starting jobs on free GPUs only..."
for ((g=0; g<NUM_GPUS && JOB_INDEX<TOTAL_JOBS; g++)); do
    if is_gpu_free $g; then
        assign_job $g
    else
        echo "[GPU $g] Skipping - GPU in use by another process"
    fi
done

if [ $JOB_INDEX -eq 0 ]; then
    echo ""
    echo "No free GPUs available. Waiting for GPUs to become free..."
    echo "The script will automatically start jobs when GPUs are released."
fi

echo "Jobs started. Monitoring..."

# Monitor
while [ $COMPLETED -lt $TOTAL_JOBS ]; do
    for ((g=0; g<NUM_GPUS; g++)); do
        if [ -n "${GPU_PIDS[$g]}" ]; then
            # Check if our job on this GPU finished
            if ! kill -0 "${GPU_PIDS[$g]}" 2>/dev/null; then
                wait "${GPU_PIDS[$g]}"
                ec=$?
                COMPLETED=$((COMPLETED + 1))

                if [ $ec -eq 0 ]; then
                    echo "[GPU $g] Done: ${GPU_JOBS[$g]} ($COMPLETED/$TOTAL_JOBS)"
                else
                    echo "[GPU $g] Failed: ${GPU_JOBS[$g]} ($COMPLETED/$TOTAL_JOBS)"
                fi

                # Clean up GPU memory after job completes
                echo "[GPU $g] Cleaning up GPU memory..."
                CUDA_VISIBLE_DEVICES=$g python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
                sleep 3  # Give time for memory to be released

                GPU_PIDS[$g]=""
                GPU_JOBS[$g]=""
                assign_job $g
            fi
        else
            # No job running on this GPU - check if it's free and assign a job
            if [ $JOB_INDEX -lt $TOTAL_JOBS ]; then
                if is_gpu_free $g; then
                    echo "[GPU $g] Now free, assigning next job..."
                    assign_job $g
                fi
            fi
        fi
    done
    sleep 5  # Check every 5 seconds
done

echo ""
echo "=========================================="
echo "All $TOTAL_JOBS gpt-oss-20b jobs completed!"
echo "=========================================="
