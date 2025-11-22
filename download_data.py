"""
Download dataset for SemEval-2026 Task 13
Supports multiple sources: Hugging Face (recommended), GitHub, or manual download
"""

import os
import requests
import zipfile
import argparse
from pathlib import Path
import pandas as pd

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Install with: pip install datasets")


def download_file(url, output_path):
    """Download a file from URL"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}%", end='', flush=True)
    print(f"\nDownloaded: {output_path}")


def download_from_huggingface(data_dir='data', task='A'):
    """Download dataset from Hugging Face (RECOMMENDED)"""
    if not HF_AVAILABLE:
        print("Error: 'datasets' library required for Hugging Face download")
        print("Install with: pip install datasets")
        return False
    
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # Hugging Face dataset IDs (try multiple possible locations)
    hf_datasets = [
        "DaniilOr/SemEval-2026-Task13",
        "mbzuai-nlp/SemEval-2026-Task13",
        "semeval-2026-task13"
    ]
    
    print("Downloading from Hugging Face (recommended method)...")
    
    for hf_id in hf_datasets:
        try:
            print(f"Trying: {hf_id}")
            dataset = load_dataset(hf_id)
            
            # Save splits
            for split_name in ['train', 'validation', 'test']:
                if split_name in dataset:
                    # Map validation to val
                    output_name = 'val' if split_name == 'validation' else split_name
                    output_file = data_path / f"{output_name}.csv"
                    
                    # Convert to DataFrame and save
                    df = dataset[split_name].to_pandas()
                    df.to_csv(output_file, index=False)
                    print(f"✓ Saved {output_name}.csv ({len(df)} samples)")
            
            print(f"\n✓ Successfully downloaded from Hugging Face: {hf_id}")
            return True
            
        except Exception as e:
            print(f"  Failed: {e}")
            continue
    
    print("\n✗ Could not download from any Hugging Face location")
    return False


def download_from_github(data_dir='data', task='A'):
    """Download dataset from GitHub repository"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    repo_url = "https://github.com/mbzuai-nlp/SemEval-2026-Task13"
    
    print(f"\nDownloading from GitHub: {repo_url}")
    print("Attempting to download files...")
    
    # Try common paths
    dataset_urls = {
        'train': [
            f"{repo_url}/raw/main/data/task_a_train.csv",
            f"{repo_url}/raw/main/data/train.csv",
            f"{repo_url}/raw/main/Task_A/train.csv"
        ],
        'val': [
            f"{repo_url}/raw/main/data/task_a_val.csv",
            f"{repo_url}/raw/main/data/val.csv",
            f"{repo_url}/raw/main/Task_A/val.csv"
        ],
        'test': [
            f"{repo_url}/raw/main/data/task_a_test.csv",
            f"{repo_url}/raw/main/data/test.csv",
            f"{repo_url}/raw/main/Task_A/test.csv"
        ]
    }
    
    success_count = 0
    for split, urls in dataset_urls.items():
        output_file = data_path / f"{split}.csv"
        if output_file.exists():
            print(f"✓ {split}.csv already exists, skipping...")
            success_count += 1
            continue
        
        downloaded = False
        for url in urls:
            try:
                download_file(url, output_file)
                downloaded = True
                success_count += 1
                break
            except Exception as e:
                continue
        
        if not downloaded:
            print(f"✗ Could not download {split}.csv from GitHub")
    
    return success_count > 0


def download_dataset(data_dir='data', task='A', source='auto'):
    """
    Download SemEval-2026 Task 13 dataset
    
    Args:
        data_dir: Directory to save dataset
        task: Task identifier (A, B, etc.)
        source: 'huggingface', 'github', 'auto' (tries both)
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("="*60)
    print("SemEval-2026 Task 13 Dataset Download")
    print("="*60)
    
    if source == 'auto':
        # Try Hugging Face first (recommended)
        if download_from_huggingface(data_dir, task):
            return
        # Fallback to GitHub
        if download_from_github(data_dir, task):
            return
    elif source == 'huggingface':
        if download_from_huggingface(data_dir, task):
            return
    elif source == 'github':
        if download_from_github(data_dir, task):
            return
    
    # If all automatic methods fail, show manual instructions
    print("\n" + "="*60)
    print("Automatic download failed. Manual download instructions:")
    print("="*60)
    print("\nOption 1: Hugging Face (RECOMMENDED)")
    print("  Visit: https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13")
    print("  Or use Python:")
    print("    from datasets import load_dataset")
    print("    dataset = load_dataset('DaniilOr/SemEval-2026-Task13')")
    print("    dataset['train'].to_pandas().to_csv('data/train.csv')")
    print("    dataset['validation'].to_pandas().to_csv('data/val.csv')")
    
    print("\nOption 2: GitHub")
    repo_url = "https://github.com/mbzuai-nlp/SemEval-2026-Task13"
    print(f"  1. Visit: {repo_url}")
    print(f"  2. Clone or download the repository")
    print(f"  3. Copy data files to: {data_path.absolute()}")
    print(f"  Or use git:")
    print(f"    git clone {repo_url}.git")
    print(f"    cp -r SemEval-2026-Task13/data/* {data_path}/")
    
    print("\nOption 3: Kaggle (if available)")
    print("  Check: https://www.kaggle.com/datasets (search: SemEval-2026-Task13)")
    
    print(f"\nExpected files in {data_path.absolute()}:")
    print("  - train.csv (or train.json)")
    print("  - val.csv (or val.json)")
    print("  - test.csv (or test.json) [optional]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download SemEval-2026 Task 13 dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect best source (recommended)
  python download_data.py
  
  # Use Hugging Face (recommended)
  python download_data.py --source huggingface
  
  # Use GitHub
  python download_data.py --source github
  
  # Custom directory
  python download_data.py --data_dir ./my_data
        """
    )
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to save dataset (default: data)')
    parser.add_argument('--task', type=str, default='A',
                        help='Task identifier (A, B, etc.) (default: A)')
    parser.add_argument('--source', type=str, default='auto',
                        choices=['auto', 'huggingface', 'github'],
                        help='Download source: auto (tries both), huggingface, or github (default: auto)')
    
    args = parser.parse_args()
    download_dataset(args.data_dir, args.task, args.source)

