#!/bin/bash
# Script to train all models on server with nohup (safe to close terminal)
# Run this after getting a compute node with: srun --partition=cuda --qos=cuda --gres=gpu:1 --time=20:00:00 --mem=64G --pty bash

# Set Hugging Face token for gated models (StarCoder)
# Token should be set via: export HF_TOKEN="your_token"
# Or run: bash setup_hf_token.sh your_token
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. StarCoder training may fail."
    echo "Set it with: export HF_TOKEN='your_token'"
fi

# Set paths (update these based on your data location)
TRAIN_DATA="data/train.csv"
VAL_DATA="data/val.csv"
OUTPUT_DIR="./models"
LOG_DIR="./logs"
CACHE_DIR="./cache"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
mkdir -p $CACHE_DIR

# Models to train (in order of estimated time - smaller models first)
MODELS=("distilbert" "codebert" "graphcodebert" "codet5" "starcoder")

# Training parameters
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=5
MAX_LENGTH=512
WARMUP_STEPS=500

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/training_all_models_${TIMESTAMP}.out"

echo "=========================================="
echo "Training all models for SemEval-2026 Task 13 Part A"
echo "=========================================="
echo "Start time: $(date)"
echo "Log file: $MAIN_LOG"
echo "You can safely close your terminal now!"
echo ""

# Run all training with nohup
# Convert array to space-separated string for bash -c
MODELS_STR="${MODELS[*]}"

nohup bash -c "
MODELS_ARR=($MODELS_STR)
for MODEL in \"\${MODELS_ARR[@]}\"; do
    echo ''
    echo '=========================================='
    echo \"Training \$MODEL...\"
    echo \"Start time: \$(date)\"
    echo '=========================================='
    
    python train.py \
        --model_name \$MODEL \
        --train_data $TRAIN_DATA \
        --val_data $VAL_DATA \
        --output_dir $OUTPUT_DIR \
        --batch_size $BATCH_SIZE \
        --learning_rate $LEARNING_RATE \
        --num_epochs $NUM_EPOCHS \
        --max_length $MAX_LENGTH \
        --warmup_steps $WARMUP_STEPS \
        --cache_dir $CACHE_DIR \
        --log_dir $LOG_DIR \
        --save_every_epoch \
        --seed 42
    
    EXIT_CODE=\$?
    
    if [ \$EXIT_CODE -eq 0 ]; then
        echo \"✓ Completed training \$MODEL at \$(date)\"
    else
        echo \"✗ Training \$MODEL failed with exit code \$EXIT_CODE at \$(date)\"
    fi
done

echo ''
echo ''
echo '=========================================='
echo 'All models training completed!'
echo \"End time: \$(date)\"
echo '=========================================='
echo ''
echo 'Training finished. You can now exit the GPU node.'
" > "$MAIN_LOG" 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Main log: $MAIN_LOG"
echo ""
echo "To check progress:"
echo "  tail -f $MAIN_LOG"
echo "  tail -f $LOG_DIR/*.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo "  ps aux | grep train.py"
echo ""
echo "✓ You can now safely close your terminal!"

