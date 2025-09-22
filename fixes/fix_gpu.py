#!/usr/bin/env python3
"""
GPU Setup Fix Script
Installs PyTorch with CUDA support for RTX 4060
"""

import subprocess
import sys
import platform

def install_pytorch_cuda():
    print("🔧 FIXING GPU SETUP FOR RTX 4060")
    print("=" * 50)
    
    print("This will:")
    print("1. Uninstall current PyTorch (CPU-only version)")
    print("2. Install PyTorch with CUDA 12.1 support")
    print("3. Enable RTX 4060 acceleration")
    
    response = input("\nProceed? (y/n): ").lower().strip()
    if response != 'y':
        print("Setup cancelled.")
        return
    
    print("\n🗑️  Step 1: Removing CPU-only PyTorch...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", 
            "torch", "torchvision", "torchaudio", "-y"
        ], check=True)
        print("✅ Old PyTorch removed")
    except Exception as e:
        print(f"⚠️  Uninstall warning: {e}")
    
    print("\n⬇️  Step 2: Installing PyTorch with CUDA support...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("✅ PyTorch with CUDA installed")
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return
    
    print("\n🧪 Step 3: Testing GPU detection...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU detected: {gpu_name}")
            print("🚀 RTX 4060 is ready for training!")
        else:
            print("❌ GPU still not detected")
            print("Next steps:")
            print("1. Update NVIDIA drivers")
            print("2. Restart computer")
            print("3. Run check_gpu.py for diagnostics")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    install_pytorch_cuda()