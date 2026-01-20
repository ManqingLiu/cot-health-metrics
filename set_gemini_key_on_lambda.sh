#!/bin/bash

# Script to set GEMINI_API_KEY on Lambda server
# Usage: ./set_gemini_key_on_lambda.sh "your-api-key-here"

# Configuration from sync_to_lambda.sh
LAMBDA_USER="ubuntu"
LAMBDA_HOST="209.20.158.189"
SSH_KEY="$HOME/.ssh/id_rsa_lambda"

if [ -z "$1" ]; then
    echo "Usage: $0 \"your-gemini-api-key-here\""
    echo ""
    echo "Example:"
    echo "  $0 \"AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234567\""
    exit 1
fi

API_KEY="$1"

echo "Setting GEMINI_API_KEY on Lambda server..."
echo ""

# SSH into the server and set the key in both .bashrc and .profile
ssh -i "$SSH_KEY" "${LAMBDA_USER}@${LAMBDA_HOST}" << EOF
    # Remove existing GEMINI_API_KEY line if it exists
    sed -i '/export GEMINI_API_KEY=/d' ~/.bashrc
    sed -i '/export GEMINI_API_KEY=/d' ~/.profile
    
    # Add new export statement
    echo 'export GEMINI_API_KEY="$API_KEY"' >> ~/.bashrc
    echo 'export GEMINI_API_KEY="$API_KEY"' >> ~/.profile
    
    # Verify it's set
    source ~/.bashrc
    if [ -n "\$GEMINI_API_KEY" ]; then
        echo "✓ GEMINI_API_KEY successfully set (length: \${#GEMINI_API_KEY} characters)"
    else
        echo "✗ Failed to set GEMINI_API_KEY"
        exit 1
    fi
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ GEMINI_API_KEY has been permanently set on Lambda server"
    echo "It will be available in all new shell sessions."
else
    echo ""
    echo "✗ Failed to set GEMINI_API_KEY. Please check your SSH connection and try again."
    exit 1
fi

