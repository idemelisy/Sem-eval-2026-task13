#!/bin/bash
# Complete script to start training on GPU node
# Run this after getting a GPU node

echo "=========================================="
echo "Starting Full Training - SemEval-2026 Task 13"
echo "=========================================="

# Check if we're on GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: Not on GPU node!"
    echo "Get GPU node first:"
    echo "  srun --partition=cuda --qos=cuda --gres=gpu:1 --time=20:00:00 --mem=64G --pty bash"
    exit 1
fi

# Check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check PyTorch CUDA
echo ""
echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || {
    echo "Error: PyTorch CUDA not available!"
    echo "Run: bash fix_pytorch_cuda.sh"
    exit 1
}

# Navigate to project
cd /cta/users/ide.yilmaz/Sem-eval-task-13

# Set HF token (should be set via environment or setup_hf_token.sh)
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. StarCoder training may fail."
    echo "Set it with: export HF_TOKEN='your_token'"
    echo "Or run: bash setup_hf_token.sh your_token"
else
    echo "HF_TOKEN set from environment"
fi

# Check dataset
if [ ! -f "data/train.csv" ] || [ ! -f "data/val.csv" ]; then
    echo "Error: Dataset not found!"
    echo "Run: python download_dataset_manual.py"
    exit 1
fi

echo ""
echo "Dataset found:"
ls -lh data/*.csv

echo ""
echo "=========================================="
echo "Starting training (all models)..."
echo "This will take ~12-24 hours"
echo "You can safely close your terminal!"
echo "=========================================="
echo ""

# Start training with nohup
echo "Starting training in background..."
bash run_all_models_server_nohup.sh

# Get the log file (most recent)
sleep 2
LATEST_LOG=$(ls -t logs/training_all_models_*.out 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "Error: Could not find training log file"
    exit 1
fi

echo ""
echo "Monitoring training log: $LATEST_LOG"
echo "Waiting for training to complete..."
echo ""

# Monitor training by checking for train.py processes and log file
while true; do
    # Check if train.py is still running
    if ! pgrep -f "train.py" > /dev/null; then
        # Check if all models completed by looking at log
        if grep -q "All models training completed!" "$LATEST_LOG" 2>/dev/null; then
            echo ""
            echo "=========================================="
            echo "Training completed!"
            echo "=========================================="
            break
        fi
        # If no train.py but log doesn't show completion, wait a bit more
        sleep 10
        if ! pgrep -f "train.py" > /dev/null; then
            # Still no process, check log one more time
            if grep -q "All models training completed!" "$LATEST_LOG" 2>/dev/null; then
                echo ""
                echo "=========================================="
                echo "Training completed!"
                echo "=========================================="
                break
            else
                echo "Warning: Training process not found but log doesn't show completion"
                echo "Check log manually: tail -f $LATEST_LOG"
                break
            fi
        fi
    fi
    sleep 60  # Check every minute
done

echo ""
echo "Final check:"
bash check_progress.sh
echo ""
echo "Exiting GPU node..."
exit 0

