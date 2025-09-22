#!/usr/bin/env python3
"""
GPU Detection and Diagnostic Script
Helps identify why GPU isn't being detected
"""

import torch
import sys

def check_gpu_setup():
    print("🔍 GPU DETECTION DIAGNOSTIC")
    print("=" * 60)
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    print("\n🖥️  CUDA AVAILABILITY:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
        print(f"\nCurrent device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            print("✅ GPU computation test: PASSED")
        except Exception as e:
            print(f"❌ GPU computation test: FAILED - {e}")
    else:
        print("\n❌ CUDA NOT AVAILABLE")
        print("\nPossible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA drivers not installed/updated")
        print("3. CUDA toolkit not properly configured")
        print("4. GPU not supported")
        
        print("\n🔧 SOLUTIONS:")
        print("1. Install PyTorch with CUDA:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("\n2. Update NVIDIA drivers:")
        print("   Download from: https://www.nvidia.com/drivers/")
        print("\n3. Verify GPU in Device Manager")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_gpu_setup()