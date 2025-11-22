# Sem-eval-2026-task13

Training and evaluation scripts for SemEval-2026 Task 13 Part A: Detecting Machine-Generated Code.

[Task Repository](https://github.com/mbzuai-nlp/SemEval-2026-Task13)

## Models

This repository contains scripts to train and evaluate the following models:

1. **CodeBERT** (`microsoft/codebert-base`)
2. **GraphCodeBERT** (`microsoft/graphcodebert-base`)
3. **CodeT5** (`Salesforce/codet5-base`)
4. **StarCoder** (`bigcode/starcoderbase-1b`)
5. **DistilBERT** (`distilbert-base-uncased`)

## Setup

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Data Format

The scripts expect CSV or JSON files with the following format:
- **CSV**: Must contain columns `code` (or `text`/`snippet`) and `label`
- **JSON**: Must contain objects with `code` (or `text`/`snippet`) and `label` fields
- **Labels**: 
  - `0` or `"human"` or `"human-written"` for human-written code
  - `1` or `"machine"` or `"machine-generated"` for machine-generated code

Example CSV:
```csv
code,label
def hello(): print("world"),0
def hello(): print("world"),1
```

## Usage

### 1. Get Compute Node

First, get a GPU compute node:

```bash
srun --partition=cuda --qos=cuda --gres=gpu:1 --time=8:00:00 --mem=64G --pty bash
```

**Time estimation:** Run `estimate_time.py` first to determine how long training will take, then adjust the `--time` parameter accordingly.

### 2. Estimate Training Time (Optional but Recommended)

Before training, estimate how long it will take:

```bash
python estimate_time.py \
    --model_name codebert \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --batch_size 16 \
    --num_epochs 5
```

This will give you an estimate of training time per epoch and total time.

### 3. Download Dataset (if needed)

```bash
python download_data.py --data_dir data
```

Or manually download from the [task repository](https://github.com/mbzuai-nlp/SemEval-2026-Task13).

### 4. Training Options

#### Option A: Training a Single Model (Interactive)

```bash
python train.py \
    --model_name codebert \
    --train_data data/train.csv \
    --val_data data/val.csv \
    --output_dir ./models \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --max_length 512 \
    --warmup_steps 500 \
    --cache_dir ./cache \
    --log_dir ./logs \
    --save_every_epoch \
    --seed 42
```

**Available model names:**
- `codebert`
- `graphcodebert`
- `codet5`
- `starcoder`
- `distilbert`

#### Option B: Training with nohup (Background)

```bash
bash run_training.sh codebert data/train.csv data/val.csv
```

This runs training in the background. Check progress with:
```bash
tail -f logs/training_codebert_*.out
tail -f logs/codebert_*.log
```

#### Option C: Training All Models on Server

```bash
bash run_all_models_server.sh
```

Make sure to update the data paths in the script before running.

### 5. Evaluation

**Important:** We evaluate on **validation set** during training. For final evaluation on test set:

```bash
python evaluate.py \
    --model_path ./models/codebert/best_model \
    --test_data data/test.csv \
    --output_file results/codebert_results.json \
    --batch_size 16 \
    --max_length 512
```

### Key Features

- **Checkpointing**: Models are saved after every epoch if `--save_every_epoch` is used
- **Data Caching**: Preprocessed data is cached to `--cache_dir` for faster subsequent runs
- **Logging**: All training logs are saved to `--log_dir` with timestamps
- **Time Estimation**: Run `estimate_time.py` before training to plan compute node allocation
- **Progress Tracking**: Real-time progress with estimated remaining time

## Project Structure

```
Sem-eval-task-13/
├── data_loader.py              # Data loading and preprocessing with caching
├── train.py                    # Training script with checkpointing
├── evaluate.py                 # Evaluation script
├── estimate_time.py            # Training time estimation
├── download_data.py            # Dataset download helper
├── config.py                   # Model configurations
├── requirements.txt            # Python dependencies
├── run_training.sh             # Single model training with nohup
├── run_all_models.sh           # Batch training (local)
├── run_all_models_server.sh    # Batch training (server)
├── README.md                   # This file
└── .gitignore                  # Git ignore rules
```

## Output

- **Models**: 
  - Best model: `./models/{model_name}/best_model/`
  - Checkpoints: `./models/{model_name}/checkpoint_epoch_{N}/` (if `--save_every_epoch`)
- **Training History**: `./models/{model_name}/training_history.json` (updated after each epoch)
- **Logs**: `./logs/{model_name}_{timestamp}.log` (detailed training logs)
- **Cached Data**: `./cache/` (preprocessed data for faster subsequent runs)
- **Evaluation Results**: JSON file with metrics (accuracy, precision, recall, F1, confusion matrix)

## Configuration

Model-specific configurations can be found in `config.py`. You can modify:
- Batch sizes
- Learning rates
- Maximum sequence lengths
- Number of epochs
- Warmup steps

## Notes

- **Evaluation**: Training evaluates on **validation set** (not test set). Test set should only be used for final evaluation.
- **Checkpointing**: Use `--save_every_epoch` to save checkpoints after every epoch (useful for long training runs)
- **Data Caching**: Preprocessed data is cached automatically. First run will be slower, subsequent runs will be faster.
- **Logging**: All output is logged to both console and log files. Check log files for detailed progress.
- **Time Management**: Always run `estimate_time.py` first to estimate training time and allocate appropriate compute node time.
- **GPU/CPU**: Scripts automatically handle GPU/CPU selection
- **Gradient Clipping**: Applied during training to prevent gradient explosion
- **Best Model**: Saved automatically based on validation F1 score
- **Binary Classification**: All models use 2 labels (human-written vs machine-generated)

## Training Time Estimates

Approximate training times per epoch (may vary based on dataset size and hardware):
- DistilBERT: ~10-20 minutes
- CodeBERT: ~20-40 minutes  
- GraphCodeBERT: ~20-40 minutes
- CodeT5: ~30-60 minutes
- StarCoder: ~40-80 minutes

**Total for 5 epochs**: Multiply per-epoch time by 5, add 20% buffer for safety.