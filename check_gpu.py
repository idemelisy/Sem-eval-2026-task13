"""
Quick script to check GPU availability
"""

import torch
import sys

print("="*60)
print("GPU Availability Check")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
        print(f"  Current memory allocated: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"  Current memory cached: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # Test GPU
    print("\nTesting GPU with a simple tensor operation...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✓ GPU test successful!")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("\n✗ CUDA not available!")
    print("\nPossible reasons:")
    print("1. PyTorch was installed without CUDA support")
    print("2. CUDA driver is too old (check nvidia-smi)")
    print("3. Not on a GPU node")
    print("\nTo fix:")
    print("1. Check nvidia-smi: nvidia-smi")
    print("2. Reinstall PyTorch with CUDA:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("   (adjust CUDA version based on your GPU)")

print("\n" + "="*60)

