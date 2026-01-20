# Update Lambda Host

Updates the Lambda instance IP address used by sync and setup scripts.

## Instructions

1. Ask the user for the new Lambda IP address
2. Save it to `~/.lambda_host`:

```bash
echo "NEW_IP_ADDRESS" > ~/.lambda_host
chmod 600 ~/.lambda_host
```

3. Verify SSH connectivity:

```bash
LAMBDA_HOST=$(cat ~/.lambda_host)
ssh -o ConnectTimeout=5 ubuntu@$LAMBDA_HOST 'echo "Connected to Lambda: $(hostname)"'
```

4. Show the user the current configuration:

```bash
echo "Lambda host updated to: $(cat ~/.lambda_host)"
echo ""
echo "You can now use:"
echo "  - /setup-lambda-keys to configure API keys"
echo "  - ./sync_to_lambda.sh to sync files"
echo "  - ./run_parallel_gpu_lambda.sh to run training"
```

## What This Does

- Stores the Lambda IP in `~/.lambda_host` (outside of git)
- This file is read by `sync_to_lambda.sh` and other scripts
- Makes it easy to switch between different Lambda instances

## Finding Your Lambda IP

1. Go to https://cloud.lambdalabs.com/instances
2. Find your running instance
3. Copy the IP address from the "IP Address" column
