#!/bin/bash
# Quick setup script for GPU training
# Run this after getting a GPU node

echo "=========================================="
echo "SemEval-2026 Task 13 - GPU Setup"
echo "=========================================="

# Check if we're on a GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Make sure you're on a GPU node!"
    echo "Get GPU node with: srun --partition=cuda --qos=cuda --gres=gpu:1 --time=8:00:00 --mem=64G --pty bash"
    exit 1
fi

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Load modules
echo ""
echo "Loading modules..."
module load miniconda3/22.11.1-oneapi-2024.0.2-vdx5rot
module load cuda/10.2.89-gcc-8.5.0-h3fatfr

# Check/create conda environment
ENV_NAME="semeval"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating..."
    conda activate $ENV_NAME
else
    echo "Creating conda environment '${ENV_NAME}'..."
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
fi

# Navigate to project directory
cd /cta/users/ide.yilmaz/Sem-eval-task-13

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data
mkdir -p models
mkdir -p logs
mkdir -p cache
mkdir -p results

# Check for dataset
echo ""
echo "Checking for dataset..."
if [ -f "data/train.csv" ] || [ -f "data/train.json" ]; then
    echo "✓ Training data found"
else
    echo "✗ Training data not found in data/ directory"
    echo "  Please download dataset and place in data/ directory"
    echo "  Or run: python download_data.py --data_dir data"
fi

if [ -f "data/val.csv" ] || [ -f "data/val.json" ]; then
    echo "✓ Validation data found"
else
    echo "✗ Validation data not found in data/ directory"
fi

# Check Python and PyTorch
echo ""
echo "Checking Python environment..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Make sure dataset is in data/ directory"
echo "2. (Optional) Estimate training time:"
echo "   python estimate_time.py --model_name codebert --train_data data/train.csv --val_data data/val.csv"
echo "3. Start training:"
echo "   bash run_training.sh codebert data/train.csv data/val.csv"
echo "   or"
echo "   python train.py --model_name codebert --train_data data/train.csv --val_data data/val.csv --save_every_epoch"
echo ""

