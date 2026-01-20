# Lambda Training Cheatsheet

## One-Time Local Setup (on your Mac)

Run these commands once from **any folder** to set up API keys and Lambda host:

```bash
# 1. Create secrets file with your API keys
mkdir -p ~/.secrets && chmod 700 ~/.secrets
echo 'WANDB_API_KEY=your_wandb_key_here' > ~/.secrets/lambda_api_keys
echo 'GEMINI_API_KEY=your_gemini_key_here' >> ~/.secrets/lambda_api_keys
echo 'OPENAI_API_KEY=your_openai_key_here' >> ~/.secrets/lambda_api_keys
chmod 600 ~/.secrets/lambda_api_keys

# 2. Set your Lambda IP (update when you launch a new instance)
echo "YOUR_LAMBDA_IP" > ~/.lambda_host
```

## Quick Start

```bash
# SSH to Lambda (reads IP from ~/.lambda_host)
ssh ubuntu@$(cat ~/.lambda_host)

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
# From LOCAL machine - sync files to Lambda (uses IP from ~/.lambda_host)
./sync_to_lambda.sh

# Auto-sync on file changes
./sync_to_lambda.sh --watch

# Override host for one-time sync
./sync_to_lambda.sh --host 192.168.1.100
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

## API Keys Setup

### Option 1: Using Claude Code (Recommended)

```bash
# Update Lambda IP when you launch a new instance
/update-lambda-host

# Set up all API keys on Lambda
/setup-lambda-keys
```

### Option 2: Manual Setup on Lambda

```bash
# SSH to Lambda
ssh ubuntu@$(cat ~/.lambda_host)

# Set up API key files (keys are loaded automatically by scripts)
echo "YOUR_WANDB_API_KEY" > ~/.wandb_api_key
echo "YOUR_GEMINI_API_KEY" > ~/.gemini_api_key
echo "YOUR_OPENAI_API_KEY" > ~/.openai_api_key
chmod 600 ~/.wandb_api_key ~/.gemini_api_key ~/.openai_api_key

# Verify keys
cat ~/.wandb_api_key | head -c 10
```

**Note:** The training script automatically loads keys from `~/.wandb_api_key`, `~/.gemini_api_key`, and `~/.openai_api_key`. You don't need to manually export environment variables.

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
ssh-keyscan -H $(cat ~/.lambda_host) >> ~/.ssh/known_hosts
```

## New Lambda Instance Workflow

When you launch a new Lambda instance:

1. Update the IP: `echo "NEW_IP" > ~/.lambda_host` (or use `/update-lambda-host`)
2. Add SSH key: `ssh-keyscan -H $(cat ~/.lambda_host) >> ~/.ssh/known_hosts`
3. Sync code: `./sync_to_lambda.sh`
4. Set up API keys: use `/setup-lambda-keys` or manually create key files on Lambda
5. Start training: `ssh ubuntu@$(cat ~/.lambda_host) 'cd ~/CoT-health-metrics && bash run_parallel_gpu_lambda.sh --detach'`
