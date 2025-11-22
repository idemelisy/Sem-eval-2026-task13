"""
Training script for SemEval-2026 Task 13 Part A
Supports multiple models: CodeBERT, GraphCodeBERT, CodeT5, StarCoder, DistilBERT
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import os
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import time
import logging
from datetime import datetime

from data_loader import CodeDataset, load_data, get_tokenizer


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, logger=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    start_time = time.time()
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        # Get predictions
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
        
        # Log every 100 batches
        if logger and batch_idx % 100 == 0:
            logger.info(f'Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    epoch_time = time.time() - start_time
    
    if logger:
        logger.info(f'Epoch {epoch+1} Training - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Time: {epoch_time:.2f}s')
    
    return avg_loss, accuracy, epoch_time


def evaluate(model, dataloader, device, logger=None):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    eval_time = time.time() - start_time
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )
    cm = confusion_matrix(true_labels, predictions)
    
    if logger:
        logger.info(f'Validation - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, '
                   f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Time: {eval_time:.2f}s')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'time': eval_time
    }


def main():
    parser = argparse.ArgumentParser(description='Train model for SemEval-2026 Task 13 Part A')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=['codebert', 'graphcodebert', 'codet5', 'starcoder', 'distilbert'],
                        help='Model to train')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data (CSV or JSON)')
    parser.add_argument('--val_data', type=str, required=True,
                        help='Path to validation data (CSV or JSON)')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save model')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='Directory to cache preprocessed data')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--save_every_epoch', action='store_true',
                        help='Save checkpoint after every epoch')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Model name mapping
    model_map = {
        'codebert': 'microsoft/codebert-base',
        'graphcodebert': 'microsoft/graphcodebert-base',
        'codet5': 'Salesforce/codet5-base',
        'starcoder': 'bigcode/starcoderbase-1b',
        'distilbert': 'distilbert-base-uncased'
    }
    
    model_name = model_map[args.model_name]
    
    # Setup logging
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{args.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f'Starting training for {args.model_name}')
    logger.info(f'Arguments: {vars(args)}')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    if torch.cuda.is_available():
        logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logger.info(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
    
    # Load data
    logger.info('Loading data...')
    train_texts, train_labels = load_data(args.train_data)
    val_texts, val_labels = load_data(args.val_data)
    
    logger.info(f'Training samples: {len(train_texts)}')
    logger.info(f'Validation samples: {len(val_labels)}')
    
    # Load tokenizer
    logger.info(f'Loading tokenizer: {model_name}')
    tokenizer = get_tokenizer(model_name)
    
    # Create cache directories
    train_cache_dir = os.path.join(args.cache_dir, 'train') if args.cache_dir else None
    val_cache_dir = os.path.join(args.cache_dir, 'val') if args.cache_dir else None
    
    # Create datasets
    logger.info('Creating datasets...')
    train_dataset = CodeDataset(train_texts, train_labels, tokenizer, args.max_length, cache_dir=train_cache_dir)
    val_dataset = CodeDataset(val_texts, val_labels, tokenizer, args.max_length, cache_dir=val_cache_dir)
    
    # Save cache after first load
    if args.cache_dir:
        logger.info('Saving preprocessed data cache...')
        train_dataset.save_cache()
        val_dataset.save_cache()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load model
    start_epoch = 0
    if args.resume_from:
        logger.info(f'Resuming from checkpoint: {args.resume_from}')
        model = AutoModelForSequenceClassification.from_pretrained(args.resume_from)
        # Load training state if available
        checkpoint_file = os.path.join(args.resume_from, 'training_state.pt')
        if os.path.exists(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f'Resuming from epoch {start_epoch}')
    else:
        logger.info(f'Loading model: {model_name}')
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    best_f1 = 0
    training_history = []
    total_start_time = time.time()
    
    # Estimate training time (run one batch first)
    logger.info('Estimating training time...')
    est_start = time.time()
    model.train()
    sample_batch = next(iter(train_loader))
    _ = model(
        input_ids=sample_batch['input_ids'].to(device),
        attention_mask=sample_batch['attention_mask'].to(device),
        labels=sample_batch['label'].to(device)
    )
    est_time_per_batch = (time.time() - est_start)
    est_time_per_epoch = est_time_per_batch * len(train_loader)
    est_total_time = est_time_per_epoch * args.num_epochs
    logger.info(f'Estimated time per epoch: {est_time_per_epoch/60:.1f} minutes')
    logger.info(f'Estimated total training time: {est_total_time/3600:.2f} hours')
    
    logger.info('\nStarting training...')
    for epoch in range(start_epoch, args.num_epochs):
        epoch_start_time = time.time()
        logger.info(f'\n{"="*60}')
        logger.info(f'Epoch {epoch+1}/{args.num_epochs}')
        logger.info(f'{"="*60}')
        
        # Train
        train_loss, train_acc, train_time = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, logger)
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device, logger)
        
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - total_start_time
        remaining_epochs = args.num_epochs - (epoch + 1)
        estimated_remaining = (elapsed_time / (epoch + 1 - start_epoch)) * remaining_epochs if epoch > start_epoch else est_time_per_epoch * remaining_epochs
        
        logger.info(f'\nEpoch {epoch+1} Summary:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        logger.info(f'  Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')
        logger.info(f'  Val Precision: {val_metrics["precision"]:.4f}, Recall: {val_metrics["recall"]:.4f}, F1: {val_metrics["f1"]:.4f}')
        logger.info(f'  Epoch Time: {epoch_time/60:.1f} minutes')
        logger.info(f'  Estimated Remaining: {estimated_remaining/3600:.2f} hours')
        
        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch_time': epoch_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save checkpoint every epoch if requested
        if args.save_every_epoch:
            checkpoint_dir = os.path.join(args.output_dir, args.model_name, f'checkpoint_epoch_{epoch+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'training_history': training_history
            }, os.path.join(checkpoint_dir, 'training_state.pt'))
            logger.info(f'Checkpoint saved: {checkpoint_dir}')
        
        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            model_dir = os.path.join(args.output_dir, args.model_name, 'best_model')
            os.makedirs(model_dir, exist_ok=True)
            
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            logger.info(f'Best model saved! F1: {best_f1:.4f}')
        
        # Save training history after each epoch
        history_path = os.path.join(args.output_dir, args.model_name, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    total_time = time.time() - total_start_time
    logger.info(f'\n{"="*60}')
    logger.info(f'Training completed!')
    logger.info(f'Total time: {total_time/3600:.2f} hours')
    logger.info(f'Best F1: {best_f1:.4f}')
    logger.info(f'Model saved to: {os.path.join(args.output_dir, args.model_name)}')
    logger.info(f'Log file: {log_file}')


if __name__ == '__main__':
    main()

