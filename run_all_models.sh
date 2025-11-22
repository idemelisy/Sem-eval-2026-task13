#!/bin/bash
# Script to train all models for SemEval-2026 Task 13 Part A

# Set paths (update these based on your data location)
TRAIN_DATA="data/train.csv"
VAL_DATA="data/val.csv"
OUTPUT_DIR="./models"

# Create output directory
mkdir -p $OUTPUT_DIR

# Models to train
MODELS=("codebert" "graphcodebert" "codet5" "starcoder" "distilbert")

# Train each model
for MODEL in "${MODELS[@]}"; do
    echo "=========================================="
    echo "Training $MODEL..."
    echo "=========================================="
    
    python train.py \
        --model_name $MODEL \
        --train_data $TRAIN_DATA \
        --val_data $VAL_DATA \
        --output_dir $OUTPUT_DIR \
        --batch_size 16 \
        --learning_rate 2e-5 \
        --num_epochs 5 \
        --max_length 512 \
        --warmup_steps 500 \
        --seed 42
    
    echo "Completed training $MODEL"
    echo ""
done

echo "All models trained successfully!"

