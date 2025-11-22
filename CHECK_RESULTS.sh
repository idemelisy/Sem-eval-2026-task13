#!/bin/bash
# Script to check training results after completion
# Run this 25 hours later to see all results

echo "=========================================="
echo "Training Results Check - SemEval-2026 Task 13"
echo "=========================================="
echo ""

cd /cta/users/ide.yilmaz/Sem-eval-task-13

# 1. Check if training is still running
echo "1. Training Status:"
if pgrep -f "train.py" > /dev/null; then
    echo "  ⏳ Training still running..."
    RUNNING_PIDS=$(pgrep -f "train.py")
    echo "  Process IDs: $RUNNING_PIDS"
else
    echo "  ✓ Training completed (no train.py processes found)"
fi
echo ""

# 2. Check log files
echo "2. Latest Training Logs:"
LATEST_LOG=$(ls -t logs/training_all_models_*.out 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "  Latest log: $LATEST_LOG"
    echo "  Size: $(du -h "$LATEST_LOG" | cut -f1)"
    echo "  Last 10 lines:"
    tail -10 "$LATEST_LOG" | sed 's/^/    /'
else
    echo "  No training logs found"
fi
echo ""

# 3. Check model checkpoints
echo "3. Trained Models:"
MODELS=("distilbert" "codebert" "graphcodebert" "codet5" "starcoder")
for MODEL in "${MODELS[@]}"; do
    MODEL_DIR="models/$MODEL"
    if [ -d "$MODEL_DIR" ]; then
        echo "  $MODEL:"
        
        # Check best model
        if [ -d "$MODEL_DIR/best_model" ]; then
            BEST_SIZE=$(du -sh "$MODEL_DIR/best_model" 2>/dev/null | cut -f1)
            echo "    ✓ Best model saved ($BEST_SIZE)"
        else
            echo "    ✗ Best model not found"
        fi
        
        # Check checkpoints
        CHECKPOINTS=$(find "$MODEL_DIR" -type d -name "checkpoint_epoch_*" 2>/dev/null | wc -l)
        if [ $CHECKPOINTS -gt 0 ]; then
            echo "    ✓ $CHECKPOINTS checkpoint(s) saved"
        fi
        
        # Check training history
        if [ -f "$MODEL_DIR/training_history.json" ]; then
            echo "    ✓ Training history available"
            # Show last epoch results
            LAST_EPOCH=$(grep -o '"epoch":[0-9]*' "$MODEL_DIR/training_history.json" | tail -1 | cut -d: -f2)
            LAST_F1=$(grep -A 10 "\"epoch\":$LAST_EPOCH" "$MODEL_DIR/training_history.json" | grep '"val_f1"' | head -1 | grep -o '[0-9.]*' | head -1)
            if [ -n "$LAST_F1" ]; then
                echo "    Last epoch F1: $LAST_F1"
            fi
        fi
    else
        echo "  $MODEL: ✗ Not trained yet"
    fi
    echo ""
done

# 4. Summary
echo "4. Summary:"
TRAINED_COUNT=0
for MODEL in "${MODELS[@]}"; do
    if [ -d "models/$MODEL/best_model" ]; then
        TRAINED_COUNT=$((TRAINED_COUNT + 1))
    fi
done

echo "  Models trained: $TRAINED_COUNT/${#MODELS[@]}"
echo ""

# 5. Next steps
echo "5. Next Steps:"
if [ $TRAINED_COUNT -eq ${#MODELS[@]} ]; then
    echo "  ✓ All models trained!"
    echo "  Run evaluation:"
    echo "    python evaluate.py --model_path models/codebert/best_model --test_data data/test.csv --output_file results/codebert_results.json"
else
    echo "  ⏳ Some models still training or failed"
    echo "  Check logs: tail -f logs/training_all_models_*.out"
fi
echo ""

echo "=========================================="
echo "For detailed logs:"
echo "  tail -f logs/training_all_models_*.out"
echo "  tail -f logs/*.log"
echo "=========================================="

