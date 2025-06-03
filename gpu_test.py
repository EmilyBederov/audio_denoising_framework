#!/usr/bin/env python3
# gpu_test.py - Quick GPU verification

import torch
import sys

print("=== GPU VERIFICATION ===")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test GPU operations
    try:
        print("\n=== GPU TEST ===")
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.mm(x, y)
        print(f"✓ GPU matrix multiplication successful")
        print(f"✓ Result device: {z.device}")
        print(f"✓ Result shape: {z.shape}")
        
        # Memory test
        print(f"✓ GPU memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
        
else:
    print("✗ CUDA not available")
    
print("=====================")
