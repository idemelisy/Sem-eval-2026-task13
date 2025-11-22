"""
Configuration file for SemEval-2026 Task 13 Part A
Model-specific configurations
"""

MODEL_CONFIGS = {
    'codebert': {
        'model_name': 'microsoft/codebert-base',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 500
    },
    'graphcodebert': {
        'model_name': 'microsoft/graphcodebert-base',
        'max_length': 512,
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 500
    },
    'codet5': {
        'model_name': 'Salesforce/codet5-base',
        'max_length': 512,
        'batch_size': 8,  # Smaller batch size for T5 models
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 500
    },
    'starcoder': {
        'model_name': 'bigcode/starcoderbase-1b',
        'max_length': 512,
        'batch_size': 8,  # Smaller batch size for larger models
        'learning_rate': 1e-5,  # Lower learning rate for larger models
        'num_epochs': 5,
        'warmup_steps': 500
    },
    'distilbert': {
        'model_name': 'distilbert-base-uncased',
        'max_length': 512,
        'batch_size': 32,  # Larger batch size for smaller models
        'learning_rate': 2e-5,
        'num_epochs': 5,
        'warmup_steps': 500
    }
}

# Data paths (update these based on your data location)
DATA_PATHS = {
    'train': 'data/train.csv',
    'val': 'data/val.csv',
    'test': 'data/test.csv'
}

# Output directory
OUTPUT_DIR = './models'

# Random seed
RANDOM_SEED = 42

