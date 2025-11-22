"""
Manual dataset download script for SemEval-2026 Task 13
Use this if automatic download fails
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Configuration
data_dir = 'data'
task = 'A'  # Task A

# Create data directory
data_path = Path(data_dir)
data_path.mkdir(exist_ok=True)

print("="*60)
print("Downloading SemEval-2026 Task 13 Dataset from Hugging Face")
print("="*60)

try:
    # Load dataset with task config
    print(f"\nLoading dataset: DaniilOr/SemEval-2026-Task13 (config: {task})")
    ds = load_dataset("DaniilOr/SemEval-2026-Task13", task)
    
    print(f"\nAvailable splits: {list(ds.keys())}")
    
    # Save each split
    for split_name in ds.keys():
        output_name = 'val' if split_name == 'validation' else split_name
        output_file = data_path / f"{output_name}.csv"
        
        # Convert to pandas and save
        df = ds[split_name].to_pandas()
        # Use proper CSV escaping for code snippets
        df.to_csv(output_file, index=False, escapechar='\\', quoting=1)  # quoting=1 means QUOTE_ALL
        
        print(f"✓ Saved {output_name}.csv")
        print(f"  - Rows: {len(df)}")
        print(f"  - Columns: {list(df.columns)}")
        if len(df) > 0:
            print(f"  - Sample row: {df.iloc[0].to_dict()}")
    
    print("\n" + "="*60)
    print("✓ Dataset download completed successfully!")
    print("="*60)
    print(f"\nFiles saved to: {data_path.absolute()}")
    print("\nNext step: Start training with:")
    print("  bash run_training.sh codebert data/train.csv data/val.csv")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're logged in: huggingface-cli login")
    print("2. Check if dataset exists: https://huggingface.co/datasets/DaniilOr/SemEval-2026-Task13")
    print("3. Try without config:")
    print("   ds = load_dataset('DaniilOr/SemEval-2026-Task13')")
    raise

