#!/bin/bash
# Fix PyTorch CUDA compatibility
# System has CUDA 11.4, but PyTorch was installed for CUDA 12.8
# This script reinstalls PyTorch with CUDA 11.8 (compatible with CUDA 11.4)

echo "=========================================="
echo "Fixing PyTorch CUDA Compatibility"
echo "=========================================="
echo ""
echo "Current situation:"
echo "  - System CUDA: 11.4 (from nvidia-smi)"
echo "  - PyTorch installed: 2.9.1+cu128 (CUDA 12.8)"
echo "  - Problem: Version mismatch!"
echo ""
echo "Solution: Install PyTorch with CUDA 11.8 (compatible with CUDA 11.4)"
echo ""

# Check if we're on GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: Not on GPU node. Get GPU node first:"
    echo "  srun --partition=cuda --qos=cuda --gres=gpu:1 --time=20:00:00 --mem=64G --pty bash"
    exit 1
fi

echo "Step 1: Uninstalling current PyTorch..."
pip uninstall torch torchvision torchaudio -y

echo ""
echo "Step 2: Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Step 3: Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "Step 4: Running GPU test..."
python check_gpu.py

echo ""
echo "=========================================="
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✓ PyTorch CUDA fix successful!"
    echo "You can now run training with GPU support."
else
    echo "✗ PyTorch CUDA still not available."
    echo "Check the error messages above."
fi
echo "=========================================="

