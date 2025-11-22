#!/bin/bash
# Setup Hugging Face token for gated models (StarCoder)
# Usage: export HF_TOKEN="your_token_here" && bash setup_hf_token.sh
# Or: bash setup_hf_token.sh your_token_here

if [ -z "$HF_TOKEN" ] && [ -n "$1" ]; then
    export HF_TOKEN="$1"
fi

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set"
    echo "Usage: export HF_TOKEN=\"your_token\" && bash setup_hf_token.sh"
    echo "   Or: bash setup_hf_token.sh your_token"
    exit 1
fi

echo "Setting up Hugging Face token for gated models..."
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc

echo "✓ HF_TOKEN set in environment"
echo "Token will be available in new shells (from ~/.bashrc)"

# Also login via huggingface-cli (optional but recommended)
echo ""
echo "You can also login via huggingface-cli:"
echo "  huggingface-cli login"
echo "  (Enter token when prompted)"

