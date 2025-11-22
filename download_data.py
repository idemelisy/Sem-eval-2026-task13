"""
Download dataset for SemEval-2026 Task 13
Downloads data from the official repository
"""

import os
import requests
import zipfile
import argparse
from pathlib import Path


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


def download_dataset(data_dir='data', task='A'):
    """
    Download SemEval-2026 Task 13 dataset
    Note: Update URLs when official dataset is released
    """
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    # GitHub repository URL
    repo_url = "https://github.com/mbzuai-nlp/SemEval-2026-Task13"
    
    print(f"SemEval-2026 Task 13 Dataset Download")
    print(f"Repository: {repo_url}")
    print(f"\nPlease download the dataset manually from:")
    print(f"1. Visit: {repo_url}")
    print(f"2. Download the dataset files for Task A")
    print(f"3. Place them in: {data_path.absolute()}")
    print(f"\nExpected files:")
    print(f"  - train.csv (or train.json)")
    print(f"  - val.csv (or val.json)")
    print(f"  - test.csv (or test.json)")
    print(f"\nOr use git to clone the repository:")
    print(f"  git clone {repo_url}.git")
    print(f"  cp -r SemEval-2026-Task13/data/* {data_path}/")
    
    # Alternative: Try to download from common locations
    print("\nAttempting automatic download...")
    
    # Common dataset URLs (update these when available)
    dataset_urls = {
        'train': f"{repo_url}/raw/main/data/task_a_train.csv",
        'val': f"{repo_url}/raw/main/data/task_a_val.csv",
        'test': f"{repo_url}/raw/main/data/task_a_test.csv"
    }
    
    for split, url in dataset_urls.items():
        try:
            output_file = data_path / f"{split}.csv"
            if not output_file.exists():
                download_file(url, output_file)
            else:
                print(f"{output_file} already exists, skipping...")
        except Exception as e:
            print(f"Could not download {split}: {e}")
            print(f"Please download manually from {repo_url}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download SemEval-2026 Task 13 dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to save dataset')
    parser.add_argument('--task', type=str, default='A',
                        help='Task (A, B, etc.)')
    
    args = parser.parse_args()
    download_dataset(args.data_dir, args.task)

