#!/bin/bash
# Script to run training on compute node with nohup
# Usage: bash run_training.sh <model_name> <train_data> <val_data>

MODEL_NAME=${1:-codebert}
TRAIN_DATA=${2:-data/train.csv}
VAL_DATA=${3:-data/val.csv}

# Create necessary directories
mkdir -p models
mkdir -p logs
mkdir -p cache

# Set paths
OUTPUT_DIR="./models"
LOG_DIR="./logs"
CACHE_DIR="./cache"
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=5
MAX_LENGTH=512
WARMUP_STEPS=500

echo "=========================================="
echo "Starting training for $MODEL_NAME"
echo "=========================================="
echo "Train data: $TRAIN_DATA"
echo "Val data: $VAL_DATA"
echo "Output dir: $OUTPUT_DIR"
echo "Log dir: $LOG_DIR"
echo "Cache dir: $CACHE_DIR"
echo ""

# Run training with nohup
nohup python train.py \
    --model_name $MODEL_NAME \
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
    --seed 42 \
    > logs/training_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).out 2>&1 &

# Get process ID
PID=$!
echo "Training started with PID: $PID"
echo "Log file: logs/training_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).out"
echo ""
echo "To check progress:"
echo "  tail -f logs/training_${MODEL_NAME}_*.out"
echo "  tail -f logs/${MODEL_NAME}_*.log"
echo ""
echo "To check if still running:"
echo "  ps aux | grep $PID"
echo "  ps aux | grep train.py"

