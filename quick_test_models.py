"""
Quick test script to verify all models work before long training
Tests each model with a small subset of data (100 samples)
"""

import torch
import argparse
from pathlib import Path
import pandas as pd
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import CodeDataset, get_tokenizer
import time

# Model configurations
MODELS = {
    'codebert': 'microsoft/codebert-base',
    'graphcodebert': 'microsoft/graphcodebert-base',
    'codet5': 'Salesforce/codet5-base',
    'starcoder': 'bigcode/starcoderbase-1b',
    'distilbert': 'distilbert-base-uncased'
}


def quick_test_model(model_name, model_path, train_data, val_data, num_samples=100, max_length=512, batch_size=4):
    """Quick test a single model with small dataset"""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    try:
        # Load small subset of data
        print(f"Loading data (first {num_samples} samples)...")
        train_df = pd.read_csv(train_data, nrows=num_samples)
        val_df = pd.read_csv(val_data, nrows=min(num_samples, 1000))
        
        train_texts = train_df['code'].tolist()
        train_labels = train_df['label'].tolist()
        val_texts = val_df['code'].tolist()
        val_labels = val_df['label'].tolist()
        
        print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
        
        # Load tokenizer
        print(f"Loading tokenizer: {model_path}")
        hf_token = os.environ.get('HF_TOKEN', None)
        tokenizer = get_tokenizer(model_path, token=hf_token)
        
        # Create datasets
        train_dataset = CodeDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = CodeDataset(val_texts, val_labels, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Load model
        print(f"Loading model: {model_path}")
        hf_token = os.environ.get('HF_TOKEN', None)
        model_kwargs = {
            'num_labels': 2,
            'problem_type': "single_label_classification"
        }
        if hf_token:
            model_kwargs['token'] = hf_token
        if 'starcoder' in model_path.lower():
            model_kwargs['trust_remote_code'] = True
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **model_kwargs
        )
        model.to(device)
        model.eval()
        
        # Test forward pass
        print("Testing forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            # Test on one batch
            batch = next(iter(train_loader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            # Test on validation batch
            val_batch = next(iter(val_loader))
            val_input_ids = val_batch['input_ids'].to(device)
            val_attention_mask = val_batch['attention_mask'].to(device)
            val_labels = val_batch['label'].to(device)
            
            val_outputs = model(input_ids=val_input_ids, attention_mask=val_attention_mask, labels=val_labels)
        
        elapsed = time.time() - start_time
        
        print(f"✓ Forward pass successful!")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB" if torch.cuda.is_available() else "  - Memory: CPU")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Quick test all models')
    parser.add_argument('--train_data', type=str, default='data/train.csv',
                        help='Path to training data')
    parser.add_argument('--val_data', type=str, default='data/val.csv',
                        help='Path to validation data')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to test with')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    
    args = parser.parse_args()
    
    # Check if data exists
    if not Path(args.train_data).exists():
        print(f"Error: Training data not found: {args.train_data}")
        return
    
    if not Path(args.val_data).exists():
        print(f"Error: Validation data not found: {args.val_data}")
        return
    
    print("="*60)
    print("QUICK MODEL TEST - Verifying all models work")
    print("="*60)
    print(f"Testing with {args.num_samples} samples per model")
    print(f"Train data: {args.train_data}")
    print(f"Val data: {args.val_data}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    results = {}
    
    # Test each model
    for model_name, model_path in MODELS.items():
        success = quick_test_model(
            model_name, model_path,
            args.train_data, args.val_data,
            args.num_samples, args.max_length, args.batch_size
        )
        results[model_name] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for model_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{model_name:20s} {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All models passed! Ready for full training.")
        print("\nNext steps:")
        print("1. Get GPU node with appropriate time:")
        print("   srun --partition=cuda --qos=cuda --gres=gpu:1 --time=12:00:00 --mem=64G --pty bash")
        print("2. Start training all models:")
        print("   bash run_all_models_server.sh")
        print("3. Or train individually:")
        print("   bash run_training.sh <model_name> data/train.csv data/val.csv")
    else:
        print("\n✗ Some models failed. Check errors above.")
        print("Fix issues before starting full training.")


if __name__ == '__main__':
    main()

