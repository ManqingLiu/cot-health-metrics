#!/bin/bash

# Configuration - can be overridden by environment variables or command-line args
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"  # Default user, override with env var

# Lambda host priority: 1) env var, 2) ~/.lambda_host file, 3) hardcoded fallback
if [ -n "$LAMBDA_HOST" ]; then
    :  # Use existing env var
elif [ -f ~/.lambda_host ]; then
    LAMBDA_HOST="$(cat ~/.lambda_host | tr -d '\n')"
else
    LAMBDA_HOST="192.222.53.117"  # Fallback (update via /update-lambda-host)
fi

LAMBDA_PATH="${LAMBDA_PATH:-~/CoT-health-metrics}"  # Path on Lambda server where files should go
# Get project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PATH="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            LAMBDA_HOST="$2"
            shift 2
            ;;
        --user)
            LAMBDA_USER="$2"
            shift 2
            ;;
        --path)
            LAMBDA_PATH="$2"
            shift 2
            ;;
        --watch|-w)
            WATCH_MODE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host HOST] [--user USER] [--path PATH] [--watch]"
            exit 1
            ;;
    esac
done

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if SSH key exists (set before function definition so it's available in the function)
# Try id_rsa first (the key registered on Lambda), then fall back to id_rsa_lambda
if [ -f "$HOME/.ssh/id_rsa" ]; then
    SSH_KEY="$HOME/.ssh/id_rsa"
    SSH_OPTIONS="-i $SSH_KEY"
elif [ -f "$HOME/.ssh/id_rsa_lambda" ]; then
    SSH_KEY="$HOME/.ssh/id_rsa_lambda"
    SSH_OPTIONS="-i $SSH_KEY"
else
    SSH_KEY=""
    SSH_OPTIONS=""
fi

# Function to sync files
sync_files() {
    echo -e "${GREEN}Syncing files to Lambda server...${NC}"
    
    # Build SSH command
    if [ -n "$SSH_OPTIONS" ]; then
        SSH_CMD="ssh $SSH_OPTIONS"
    else
        SSH_CMD="ssh"
    fi
    
    if rsync -avz --progress \
        --exclude '.git' \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.DS_Store' \
        --exclude '*.log' \
        --exclude 'output/' \
        --exclude 'results/' \
        --exclude 'data/logprobs/' \
        -e "$SSH_CMD" \
        "$LOCAL_PATH/" \
        "${LAMBDA_USER}@${LAMBDA_HOST}:${LAMBDA_PATH}/"; then
        echo -e "${GREEN}✓ Sync completed successfully${NC}"
        return 0
    else
        echo -e "${RED}✗ Sync failed${NC}"
        echo -e "${YELLOW}Check your SSH connection with: ssh $SSH_OPTIONS ${LAMBDA_USER}@${LAMBDA_HOST}${NC}"
        return 1
    fi
}

# Check if LAMBDA_HOST is set
if [ -z "$LAMBDA_HOST" ]; then
    echo -e "${RED}Error: LAMBDA_HOST must be set${NC}"
    echo "Set it via:"
    echo "  - Environment variable: export LAMBDA_HOST=your.host.ip"
    echo "  - Command-line argument: $0 --host your.host.ip"
    echo "  - Or edit the default value in this script"
    exit 1
fi

# Display sync target
echo -e "${YELLOW}Syncing to: ${LAMBDA_USER}@${LAMBDA_HOST}:${LAMBDA_PATH}${NC}"
if [ -n "$SSH_OPTIONS" ]; then
    echo -e "${YELLOW}Using SSH key: $SSH_KEY${NC}"
else
    echo -e "${YELLOW}Using default SSH key (no custom key found at $SSH_KEY)${NC}"
fi
echo ""

# Check if fswatch is available for auto-sync
if command -v fswatch &> /dev/null; then
    if [ "$WATCH_MODE" == "1" ]; then
        echo -e "${YELLOW}Starting auto-sync mode (Ctrl+C to stop)...${NC}"
        echo -e "${YELLOW}Watching for file changes in: $LOCAL_PATH${NC}"
        echo -e "${YELLOW}Excluding: .git, __pycache__, *.pyc, .DS_Store, *.log, output/, results/${NC}"
        echo ""
        
        # Initial sync
        if ! sync_files; then
            echo -e "${RED}Initial sync failed. Please check your SSH connection and key.${NC}"
            exit 1
        fi
        
        # Debounce timer (seconds)
        SYNC_DELAY=2
        LAST_SYNC_FILE="/tmp/lambda_sync_last_time_$$"
        echo "0" > "$LAST_SYNC_FILE"
        
        # Cleanup function
        cleanup() {
            rm -f "$LAST_SYNC_FILE"
            exit 0
        }
        trap cleanup EXIT INT TERM
        
        # Watch for changes and sync (with debouncing)
        # Exclude patterns that match rsync excludes
        # Use process substitution to avoid subshell variable issue
        fswatch -r \
            --exclude '\.git' \
            --exclude '__pycache__' \
            --exclude '\.pyc$' \
            --exclude '\.DS_Store' \
            --exclude '\.log$' \
            --exclude '/output/' \
            --exclude '/results/' \
            --exclude '/data/logprobs/' \
            --exclude '\.swp$' \
            --exclude '\.swp~$' \
            --exclude '~$' \
            "$LOCAL_PATH" | while read f; do
            CURRENT_TIME=$(date +%s)
            LAST_SYNC_TIME=$(cat "$LAST_SYNC_FILE" 2>/dev/null || echo "0")
            TIME_SINCE_LAST_SYNC=$((CURRENT_TIME - LAST_SYNC_TIME))
            
            # Only sync if enough time has passed since last sync (debounce)
            if [ $TIME_SINCE_LAST_SYNC -ge $SYNC_DELAY ]; then
                echo -e "${YELLOW}[$(date +%H:%M:%S)] Change detected, syncing...${NC}"
                if sync_files; then
                    echo "$CURRENT_TIME" > "$LAST_SYNC_FILE"
                else
                    echo -e "${RED}[$(date +%H:%M:%S)] Sync failed. Will retry on next change.${NC}"
                fi
            else
                echo -e "${YELLOW}[$(date +%H:%M:%S)] Change detected (debouncing, will sync in $((SYNC_DELAY - TIME_SINCE_LAST_SYNC))s)...${NC}"
            fi
        done
    else
        # One-time sync
        sync_files
        echo ""
        echo -e "${YELLOW}Tip: Run './sync_to_lambda.sh --watch' to enable automatic syncing${NC}"
    fi
else
    # One-time sync (fswatch not available)
    sync_files
    echo ""
    echo -e "${YELLOW}Tip: Install fswatch for automatic syncing:${NC}"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  brew install fswatch"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "  sudo apt-get install fswatch  # Ubuntu/Debian"
        echo "  # or: sudo yum install fswatch  # CentOS/RHEL"
    fi
    echo "Then run: ./sync_to_lambda.sh --watch"
fi

