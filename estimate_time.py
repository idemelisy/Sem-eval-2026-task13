"""
Estimate training time for models
Run this before training to estimate how long training will take
"""

import torch
import time
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import CodeDataset, load_data, get_tokenizer


def estimate_training_time(model_name, train_data, val_data, batch_size=16, num_epochs=5, max_length=512, num_samples=100):
    """Estimate training time by running a few batches"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Model name mapping
    model_map = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'codet5': 'Salesforce/codet5-base',
        'starcoder': 'bigcode/starcoderbase-1b',
        'distilbert': 'distilbert-base-uncased'
    }
    
    model_full_name = model_map.get(model_name, model_name)
    
    # Load data
    print(f'Loading data (using first {num_samples} samples for estimation)...')
    train_texts, train_labels = load_data(train_data)
    val_texts, val_labels = load_data(val_data)
    
    # Use subset for estimation
    train_texts = train_texts[:num_samples]
    train_labels = train_labels[:num_samples]
    val_texts = val_texts[:min(num_samples, len(val_texts))]
    val_labels = val_labels[:min(num_samples, len(val_labels))]
    
    # Load tokenizer and model
    print(f'Loading tokenizer: {model_full_name}')
    tokenizer = get_tokenizer(model_full_name)
    
    print(f'Loading model: {model_full_name}')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_full_name,
        num_labels=2,
        problem_type="single_label_classification"
    )
    model.to(device)
    
    # Create datasets
    train_dataset = CodeDataset(train_texts, train_labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    val_dataset = CodeDataset(val_texts, val_labels, tokenizer, max_length)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Estimate training time
    print('\nEstimating training time...')
    model.train()
    
    # Warmup
    for _ in range(2):
        batch = next(iter(train_loader))
        _ = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['label'].to(device)
        )
    
    # Time training batches
    num_batches_to_test = min(10, len(train_loader))
    train_times = []
    
    for i, batch in enumerate(train_loader):
        if i >= num_batches_to_test:
            break
        
        start = time.time()
        outputs = model(
            input_ids=batch['input_ids'].to(device),
            attention_mask=batch['attention_mask'].to(device),
            labels=batch['label'].to(device)
        )
        loss = outputs.loss
        loss.backward()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        train_times.append(time.time() - start)
    
    # Time validation batches
    val_times = []
    model.eval()
    with torch.no_grad():
        num_val_batches = min(5, len(val_loader))
        for i, batch in enumerate(val_loader):
            if i >= num_val_batches:
                break
            start = time.time()
            _ = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=batch['label'].to(device)
            )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            val_times.append(time.time() - start)
    
    # Calculate estimates
    avg_train_time_per_batch = sum(train_times) / len(train_times)
    avg_val_time_per_batch = sum(val_times) / len(val_times) if val_times else 0
    
    # Get actual dataset sizes
    full_train_texts, _ = load_data(train_data)
    full_val_texts, _ = load_data(val_data)
    actual_train_size = len(full_train_texts)
    actual_val_size = len(full_val_texts)
    
    train_batches_per_epoch = (actual_train_size + batch_size - 1) // batch_size
    val_batches_per_epoch = (actual_val_size + batch_size - 1) // batch_size
    
    time_per_epoch = (avg_train_time_per_batch * train_batches_per_epoch) + (avg_val_time_per_batch * val_batches_per_epoch)
    total_time = time_per_epoch * num_epochs
    
    # Print results
    print('\n' + '='*60)
    print('TRAINING TIME ESTIMATION')
    print('='*60)
    print(f'Model: {model_name} ({model_full_name})')
    print(f'Training samples: {actual_train_size}')
    print(f'Validation samples: {actual_val_size}')
    print(f'Batch size: {batch_size}')
    print(f'Number of epochs: {num_epochs}')
    print(f'\nAverage time per training batch: {avg_train_time_per_batch:.3f}s')
    print(f'Average time per validation batch: {avg_val_time_per_batch:.3f}s')
    print(f'\nBatches per epoch:')
    print(f'  Training: {train_batches_per_epoch}')
    print(f'  Validation: {val_batches_per_epoch}')
    print(f'\nEstimated time per epoch: {time_per_epoch/60:.1f} minutes ({time_per_epoch/3600:.2f} hours)')
    print(f'Estimated total training time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)')
    print('='*60)
    
    return total_time


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate training time')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['codebert', 'graphcodebert', 'codet5', 'starcoder', 'distilbert'],
                        help='Model to estimate')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to use for estimation')
    
    args = parser.parse_args()
    estimate_training_time(
        args.model_name,
        args.train_data,
        args.val_data,
        args.batch_size,
        args.num_epochs,
        args.max_length,
        args.num_samples
    )

