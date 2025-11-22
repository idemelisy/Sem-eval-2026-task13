"""
Data loading and preprocessing script for SemEval-2026 Task 13 Part A
Binary classification: Human-written vs Machine-generated code
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import os
import pickle
import hashlib


class CodeDataset(Dataset):
    """Dataset class for code classification with caching support"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512, cache_dir=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Create cache directory if specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_file = os.path.join(
                self.cache_dir,
                f"cached_data_{self._get_cache_hash()}.pkl"
            )
            self._load_cache()
        else:
            self.cache = {}
    
    def _get_cache_hash(self):
        """Generate hash for cache file based on data and tokenizer"""
        # Create hash from first few samples and tokenizer name
        sample_text = str(self.texts[0]) if self.texts else ""
        hash_input = f"{sample_text}_{len(self.texts)}_{self.tokenizer.name_or_path}_{self.max_length}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def _load_cache(self):
        """Load cached preprocessed data"""
        if self.cache_dir and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded cached data from {self.cache_file}")
            except Exception as e:
                print(f"Could not load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def _save_cache(self):
        """Save preprocessed data to cache"""
        if self.cache_dir:
            try:
                with open(self.cache_file, 'wb') as f:
                    pickle.dump(self.cache, f)
                print(f"Saved cached data to {self.cache_file}")
            except Exception as e:
                print(f"Could not save cache: {e}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        # Cache the item
        if self.cache_dir:
            self.cache[idx] = item
            # Save cache periodically (every 1000 items)
            if len(self.cache) % 1000 == 0:
                self._save_cache()
        
        return item
    
    def save_cache(self):
        """Manually save cache (call after dataset is fully processed)"""
        if self.cache_dir:
            self._save_cache()


def load_data(data_path, task='A'):
    """
    Load data from CSV or JSON files
    Expected format: CSV with columns ['code', 'label'] or ['text', 'label']
    Label: 0 for human-written, 1 for machine-generated
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Try CSV first
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    # Handle different column names
    if 'code' in df.columns:
        text_col = 'code'
    elif 'text' in df.columns:
        text_col = 'text'
    elif 'snippet' in df.columns:
        text_col = 'snippet'
    else:
        raise ValueError("Data must contain 'code', 'text', or 'snippet' column")
    
    # Handle label column
    if 'label' not in df.columns:
        raise ValueError("Data must contain 'label' column")
    
    texts = df[text_col].tolist()
    labels = df['label'].tolist()
    
    # Convert labels to binary if needed
    if isinstance(labels[0], str):
        label_map = {'human': 0, 'machine': 1, 'human-written': 0, 'machine-generated': 1}
        labels = [label_map.get(label.lower(), int(label)) for label in labels]
    
    return texts, labels


def get_tokenizer(model_name):
    """Get appropriate tokenizer for the model"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        return tokenizer
    except Exception as e:
        raise ValueError(f"Error loading tokenizer for {model_name}: {str(e)}")

