#!/usr/bin/env python3
"""
PyTorch Version Fix Script
Upgrades PyTorch to latest version to fix FinBERT loading issue
"""

import subprocess
import sys
import torch

def check_pytorch_version():
    print("🔍 PYTORCH VERSION CHECK")
    print("=" * 50)
    
    current_version = torch.__version__
    print(f"Current PyTorch version: {current_version}")
    
    # Check if version is less than 2.6
    version_parts = current_version.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0
    
    if major < 2 or (major == 2 and minor < 6):
        print(f"❌ PyTorch {current_version} is too old")
        print("⚠️  FinBERT requires PyTorch 2.6+ due to security fix")
        return False
    else:
        print(f"✅ PyTorch {current_version} is compatible")
        return True

def upgrade_pytorch():
    print("\n🔧 UPGRADING PYTORCH")
    print("=" * 50)
    
    print("This will upgrade PyTorch to the latest version with CUDA support.")
    print("This fixes the FinBERT loading security issue.")
    
    response = input("\nProceed with upgrade? (y/n): ").lower().strip()
    if response != 'y':
        print("Upgrade cancelled.")
        return False
    
    print("\n⬆️  Upgrading PyTorch...")
    try:
        # Upgrade to latest PyTorch with CUDA support
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade",
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ], check=True)
        print("✅ PyTorch upgraded successfully")
        return True
    except Exception as e:
        print(f"❌ Upgrade failed: {e}")
        return False

def verify_upgrade():
    print("\n🧪 VERIFYING UPGRADE")
    print("=" * 50)
    
    try:
        # Reload torch to get new version
        import importlib
        importlib.reload(torch)
        
        new_version = torch.__version__
        print(f"New PyTorch version: {new_version}")
        
        # Test CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU support: {gpu_name}")
        else:
            print("⚠️  GPU support: Not detected")
        
        # Test model loading capability
        print("🧪 Testing transformers compatibility...")
        from transformers import AutoModel
        print("✅ Transformers library compatible")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    print("🚀 PYTORCH UPGRADE UTILITY")
    print("Fixes FinBERT security vulnerability issue")
    print("=" * 60)
    
    if check_pytorch_version():
        print("\n✅ PyTorch version is already compatible!")
        print("If you're still getting errors, try restarting Python.")
        return
    
    if upgrade_pytorch():
        print("\n🔄 Please restart your Python environment and try training again.")
        print("The FinBERT loading issue should now be resolved.")
    else:
        print("\n❌ Upgrade failed. Please try manual installation:")
        print("pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    main()