#!/bin/bash
# Complete workflow: Quick test -> Full training on server
# Usage: bash test_and_train.sh

echo "=========================================="
echo "SemEval-2026 Task 13 - Test & Train Workflow"
echo "=========================================="

# Step 1: Quick test
echo ""
echo "Step 1: Quick test all models (100 samples each)..."
python quick_test_models.py --num_samples 100

TEST_EXIT=$?

if [ $TEST_EXIT -ne 0 ]; then
    echo ""
    echo "✗ Quick test failed. Fix issues before training."
    exit 1
fi

echo ""
echo "✓ All models passed quick test!"
echo ""
read -p "Start full training on server? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Step 2: Check if we're on GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "⚠ WARNING: Not on GPU node!"
    echo "Get GPU node with:"
    echo "  srun --partition=cuda --qos=cuda --gres=gpu:1 --time=12:00:00 --mem=64G --pty bash"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Step 3: Start training
echo ""
echo "Step 2: Starting full training on server..."
echo "This will train all models sequentially."
echo "Training will run in background - you can close your computer."
echo ""

# Estimate time
echo "Estimated time per model (5 epochs):"
echo "  - distilbert: ~1-2 hours"
echo "  - codebert: ~2-4 hours"
echo "  - graphcodebert: ~2-4 hours"
echo "  - codet5: ~3-6 hours"
echo "  - starcoder: ~4-8 hours"
echo "Total: ~12-24 hours"
echo ""

# Start training
bash run_all_models_server.sh

echo ""
echo "Training started! Check progress with:"
echo "  tail -f logs/training_*.out"
echo "  tail -f logs/*.log"

