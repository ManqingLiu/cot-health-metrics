# Lambda Training Cheatsheet

## Quick Start

```bash
# SSH to Lambda
ssh ubuntu@192.222.55.164

# Navigate to project
cd ~/CoT-health-metrics

# Start training (detached - survives SSH disconnect)
bash run_parallel_gpu_lambda.sh --detach
```

## Session Management

```bash
# Attach to running training session
bash run_parallel_gpu_lambda.sh --attach
# or
tmux attach -t training

# Detach from tmux (while attached)
# Press: Ctrl+B, then D

# Kill training session
tmux kill-session -t training

# List all tmux sessions
tmux ls
```

## Monitoring

```bash
# Monitor specific job (dataset + training_type)
bash run_parallel_gpu_lambda.sh --monitor ca baseline
bash run_parallel_gpu_lambda.sh --monitor ba internalized
bash run_parallel_gpu_lambda.sh --monitor sb encoded

# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs directory
ls -lht logs/ | head -20

# Tail a specific log file
tail -f logs/baseline_ba_*.log
```

## Results

```bash
# Summarize results and find best accuracy
bash run_parallel_gpu_lambda.sh --summarize

# List output directories
ls -lht output/
```

## Sync Local <-> Lambda

```bash
# From LOCAL machine - sync files to Lambda
./sync_to_lambda.sh

# Auto-sync on file changes
./sync_to_lambda.sh --watch
```

## Default Configuration

| Setting | Default Value |
|---------|---------------|
| Model | Qwen/Qwen3-4B |
| Datasets | ba, ca, sb |
| Training Types | baseline, internalized, encoded, post-hoc |
| Filler Type (train) | not_relevant |
| Filler Type (eval) | not_relevant |
| vLLM | enabled (true) |
| Learning Rates | BA=1e-5, CA=5e-5, SB=5e-5 |
| Max Samples | 5000 |
| Eval Samples | 100 |
| Checkpoints | 4 |

## Override Defaults (Optional)

```bash
# Change filler types
export FILLER_TYPE_TRAIN=shuffled
export FILLER_TYPE_EVAL=shuffled

# Disable vLLM
export USE_VLLM=false

# Then run
bash run_parallel_gpu_lambda.sh --detach
```

## Troubleshooting

```bash
# Check if training is running
ps aux | grep sft.py

# Check GPU memory
nvidia-smi

# View recent errors in logs
grep -i "error\|failed\|oom" logs/*.log | tail -20

# Restart after fixing issues
tmux kill-session -t training
bash run_parallel_gpu_lambda.sh --detach
```

## SSH Key Issues

```bash
# If host key verification fails (from local machine)
ssh-keyscan -H 192.222.55.164 >> ~/.ssh/known_hosts
```
