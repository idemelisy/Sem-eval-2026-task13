"""
Test StarCoder model specifically (requires HF authentication)
"""

import torch
from pathlib import Path
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
from data_loader import CodeDataset, get_tokenizer
import time
import os

# Get HF token from environment
HF_TOKEN = os.environ.get('HF_TOKEN', None)
if not HF_TOKEN:
    print("Warning: HF_TOKEN not set. StarCoder requires authentication.")
    print("Set it with: export HF_TOKEN='your_token'")
    print("Or run: bash setup_hf_token.sh your_token")

def test_starcoder(train_data='data/train.csv', val_data='data/val.csv', num_samples=1):
    """Test StarCoder model"""
    print("="*60)
    print("Testing StarCoder Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model_path = 'bigcode/starcoderbase-1b'
    
    try:
        # Load small subset of data
        print(f"Loading data (first {num_samples} samples)...")
        train_df = pd.read_csv(train_data, nrows=num_samples)
        val_df = pd.read_csv(val_data, nrows=min(num_samples, 10))
        
        train_texts = train_df['code'].tolist()
        train_labels = train_df['label'].tolist()
        val_texts = val_df['code'].tolist()
        val_labels = val_df['label'].tolist()
        
        print(f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}")
        
        # Load tokenizer with token
        print(f"Loading tokenizer: {model_path}")
        print("Using HF token for authentication...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=HF_TOKEN,
            trust_remote_code=True
        )
        
        # Create datasets
        train_dataset = CodeDataset(train_texts, train_labels, tokenizer, max_length=512)
        val_dataset = CodeDataset(val_texts, val_labels, tokenizer, max_length=512)
        
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        
        # Load model with token
        print(f"Loading model: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            problem_type="single_label_classification",
            token=HF_TOKEN,
            trust_remote_code=True
        )
        model.to(device)
        model.eval()
        
        # Test forward pass
        print("Testing forward pass...")
        start_time = time.time()
        
        with torch.no_grad():
            batch = next(iter(train_loader))
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
        
        elapsed = time.time() - start_time
        
        print(f"✓ Forward pass successful!")
        print(f"  - Loss: {loss.item():.4f}")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Time: {elapsed:.2f}s")
        if torch.cuda.is_available():
            print(f"  - Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        else:
            print(f"  - Memory: CPU")
        
        print("\n" + "="*60)
        print("✓ StarCoder test PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv')
    parser.add_argument('--val_data', type=str, default='data/val.csv')
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token')
    
    args = parser.parse_args()
    
    if args.hf_token:
        os.environ['HF_TOKEN'] = args.hf_token
    
    test_starcoder(args.train_data, args.val_data, args.num_samples)

