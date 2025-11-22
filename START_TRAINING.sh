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

# Get the PID from nohup output
sleep 2
TRAINING_PID=$(ps aux | grep -E "train.py|run_all_models_server" | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$TRAINING_PID" ]; then
    echo ""
    echo "Training started with PID: $TRAINING_PID"
    echo "Waiting for training to complete..."
    echo ""
    
    # Wait for training to finish
    while ps -p $TRAINING_PID > /dev/null 2>&1; do
        sleep 60  # Check every minute
    done
    
    echo ""
    echo "=========================================="
    echo "Training completed!"
    echo "=========================================="
    echo ""
    echo "Final check:"
    bash check_progress.sh
    echo ""
    echo "Exiting GPU node..."
    exit 0
else
    echo ""
    echo "Training started in background!"
    echo "Check progress with:"
    echo "  tail -f logs/training_all_models_*.out"
    echo "  bash check_progress.sh"
    echo ""
    echo "Note: Training will continue even if you close this terminal."
    echo "When training finishes, you can exit manually."
fi

