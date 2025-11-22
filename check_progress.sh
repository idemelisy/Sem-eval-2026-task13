#!/bin/bash
# Quick script to check training progress

echo "=========================================="
echo "Training Progress Check"
echo "=========================================="
echo ""

# Check if training is running
echo "1. Checking if training processes are running:"
ps aux | grep -E "train.py|python.*train" | grep -v grep || echo "  No training processes found"
echo ""

# Check latest log files
echo "2. Latest training logs:"
if ls logs/training_*.out 1> /dev/null 2>&1; then
    echo "  Latest output log:"
    ls -lht logs/training_*.out | head -1 | awk '{print "    " $9 " (" $5 " - " $6 " " $7 " " $8 ")"}'
    echo ""
    echo "  Last 20 lines:"
    tail -20 $(ls -t logs/training_*.out | head -1)
else
    echo "  No training output logs found"
fi
echo ""

# Check model checkpoints
echo "3. Model checkpoints:"
for model_dir in models/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        if [ -d "$model_dir/best_model" ]; then
            echo "  ✓ $model_name: best_model saved"
        fi
        checkpoint_count=$(find "$model_dir" -type d -name "checkpoint_epoch_*" | wc -l)
        if [ $checkpoint_count -gt 0 ]; then
            echo "  ✓ $model_name: $checkpoint_count checkpoint(s)"
        fi
    fi
done
echo ""

# Check training history
echo "4. Training history files:"
find models/ -name "training_history.json" -exec echo "  {}" \;
echo ""

# Check GPU usage
echo "5. GPU usage:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
else
    echo "  nvidia-smi not available"
fi
echo ""

echo "=========================================="
echo "To monitor in real-time:"
echo "  tail -f logs/training_*.out"
echo "  tail -f logs/*.log"
echo "=========================================="

