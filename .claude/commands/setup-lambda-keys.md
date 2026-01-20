# Setup Lambda API Keys

Sets up API keys (W&B, Gemini, OpenAI) on a Lambda instance securely.

## Prerequisites

Create a local secrets file at `~/.secrets/lambda_api_keys`:

```bash
mkdir -p ~/.secrets
chmod 700 ~/.secrets
cat > ~/.secrets/lambda_api_keys << 'EOF'
WANDB_API_KEY=your_wandb_key_here
GEMINI_API_KEY=your_gemini_key_here
OPENAI_API_KEY=your_openai_key_here
EOF
chmod 600 ~/.secrets/lambda_api_keys
```

## Instructions

1. First, check if the local secrets file exists at `~/.secrets/lambda_api_keys`
2. Read the Lambda host IP from `~/.lambda_host` (or ask the user for it if not set)
3. SSH into the Lambda instance and set up the API keys:

```bash
# Read secrets from local file
source ~/.secrets/lambda_api_keys

# Get Lambda host
LAMBDA_HOST=$(cat ~/.lambda_host 2>/dev/null)
if [ -z "$LAMBDA_HOST" ]; then
    echo "Lambda host not set. Use /update-lambda-host first."
    exit 1
fi

# SSH and set up keys on Lambda
ssh ubuntu@$LAMBDA_HOST << 'REMOTE_SCRIPT'
# Create key files
echo "$WANDB_API_KEY" > ~/.wandb_api_key
echo "$GEMINI_API_KEY" > ~/.gemini_api_key
echo "$OPENAI_API_KEY" > ~/.openai_api_key

# Set permissions
chmod 600 ~/.wandb_api_key ~/.gemini_api_key ~/.openai_api_key

# Add to .bashrc if not already present
if ! grep -q "WANDB_API_KEY" ~/.bashrc; then
    cat >> ~/.bashrc << 'BASHRC'

# API Keys (loaded from secure files)
[ -f ~/.wandb_api_key ] && export WANDB_API_KEY="$(cat ~/.wandb_api_key | tr -d '\n')"
[ -f ~/.gemini_api_key ] && export GEMINI_API_KEY="$(cat ~/.gemini_api_key | tr -d '\n')"
[ -f ~/.openai_api_key ] && export OPENAI_API_KEY="$(cat ~/.openai_api_key | tr -d '\n')"
BASHRC
fi

echo "API keys configured successfully!"
REMOTE_SCRIPT
```

4. Verify the setup by checking that the keys are loaded:

```bash
ssh ubuntu@$LAMBDA_HOST 'source ~/.bashrc && echo "W&B key (first 10 chars): ${WANDB_API_KEY:0:10}..."'
```

## What This Does

- Creates `~/.wandb_api_key`, `~/.gemini_api_key`, `~/.openai_api_key` on Lambda
- Adds export statements to `~/.bashrc` to automatically load keys on login
- Sets proper file permissions (600) for security
- Keys are never stored in scripts or version control

## Troubleshooting

If keys aren't being loaded:
1. SSH to Lambda: `ssh ubuntu@$(cat ~/.lambda_host)`
2. Source bashrc: `source ~/.bashrc`
3. Check keys: `echo $WANDB_API_KEY | head -c 10`
