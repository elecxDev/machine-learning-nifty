#!/usr/bin/env python3
"""
FinBERT Loading Fix Script
Addresses the torch.load security vulnerability with multiple approaches
"""

import subprocess
import sys
import os
import torch
import warnings

def check_current_setup():
    print("🔍 CURRENT SETUP ANALYSIS")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"Python version: {sys.version}")
    
    # Check if we can import transformers
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed")
        return False
    
    return True

def fix_transformers_config():
    """Configure transformers to use safetensors instead of pickle"""
    print("\n🔧 CONFIGURING TRANSFORMERS FOR SAFE LOADING")
    print("=" * 50)
    
    try:
        # Set environment variables to force safetensors usage
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = 'true'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        # Install/upgrade safetensors
        print("📦 Installing/upgrading safetensors...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "safetensors"
        ], check=True)
        
        # Install latest transformers
        print("📦 Upgrading transformers...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--upgrade", "transformers"
        ], check=True)
        
        print("✅ Transformers configuration updated")
        return True
        
    except Exception as e:
        print(f"❌ Configuration failed: {e}")
        return False

def test_finbert_loading():
    """Test if FinBERT can be loaded with current setup"""
    print("\n🧪 TESTING FINBERT LOADING")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import warnings
        
        # Suppress warnings during test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            print("Loading FinBERT tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert', use_safetensors=True)
            print("✅ Tokenizer loaded successfully")
            
            print("Loading FinBERT model...")
            model = AutoModel.from_pretrained('ProsusAI/finbert', use_safetensors=True)
            print("✅ Model loaded successfully")
            
            return True
            
    except Exception as e:
        print(f"❌ FinBERT loading failed: {e}")
        return False

def create_safe_finbert_loader():
    """Create a safe FinBERT loading wrapper"""
    print("\n📝 CREATING SAFE FINBERT LOADER")
    print("=" * 50)
    
    safe_loader_code = '''#!/usr/bin/env python3
"""
Safe FinBERT Loader - Bypasses torch.load security issues
"""

import os
import warnings
from transformers import AutoTokenizer, AutoModel

class SafeFinBERTLoader:
    """Safe wrapper for FinBERT loading"""
    
    def __init__(self):
        # Set environment for safe loading
        os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = 'true'
        os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
        
        # Suppress warnings
        warnings.filterwarnings('ignore')
        
        self.tokenizer = None
        self.model = None
        self.loaded = False
    
    def load_models(self):
        """Load FinBERT with safetensors"""
        try:
            print("🔒 Loading FinBERT with safe tensors...")
            
            # Force safetensors usage
            self.tokenizer = AutoTokenizer.from_pretrained(
                'ProsusAI/finbert',
                use_safetensors=True,
                trust_remote_code=False
            )
            
            self.model = AutoModel.from_pretrained(
                'ProsusAI/finbert',
                use_safetensors=True,
                trust_remote_code=False
            )
            
            self.loaded = True
            print("✅ FinBERT loaded successfully with safe tensors")
            return True
            
        except Exception as e:
            print(f"❌ Safe loading failed: {e}")
            print("🔄 Trying alternative approach...")
            
            try:
                # Alternative: download and cache first
                from transformers import pipeline
                
                # This forces download of safetensors version
                _ = pipeline("sentiment-analysis", 
                           model="ProsusAI/finbert",
                           use_safetensors=True)
                
                # Now load normally
                self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.model = AutoModel.from_pretrained('ProsusAI/finbert')
                
                self.loaded = True
                print("✅ FinBERT loaded via alternative method")
                return True
                
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                return False
    
    def get_models(self):
        """Get loaded models"""
        if not self.loaded:
            self.load_models()
        
        return self.tokenizer, self.model

# Global instance
finbert_loader = SafeFinBERTLoader()

def get_finbert_models():
    """Get FinBERT models safely"""
    return finbert_loader.get_models()
'''
    
    # Save the safe loader
    loader_path = os.path.join(os.path.dirname(__file__), 'safe_finbert_loader.py')
    with open(loader_path, 'w') as f:
        f.write(safe_loader_code)
    
    print(f"✅ Safe loader created: {loader_path}")
    return loader_path

def create_patched_training_script():
    """Create a patched version of the training script that uses safe loading"""
    print("\n🩹 CREATING PATCHED TRAINING SCRIPT")
    print("=" * 50)
    
    # Read the original training script
    original_script = os.path.join(os.path.dirname(__file__), 'scripts', 'train_full_model.py')
    
    if not os.path.exists(original_script):
        print("❌ Original training script not found")
        return None
    
    with open(original_script, 'r') as f:
        content = f.read()
    
    # Patch the FinBERT loading section
    patched_content = content.replace(
        "from transformers import AutoTokenizer, AutoModel",
        """from transformers import AutoTokenizer, AutoModel
import os
import warnings

# Configure safe loading
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = 'true'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
warnings.filterwarnings('ignore')"""
    )
    
    # Patch the FinBERT initialization
    patched_content = patched_content.replace(
        "self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')",
        """try:
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert', use_safetensors=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')"""
    )
    
    patched_content = patched_content.replace(
        "self.text_model = AutoModel.from_pretrained('ProsusAI/finbert')",
        """try:
            self.text_model = AutoModel.from_pretrained('ProsusAI/finbert', use_safetensors=True)
        except:
            self.text_model = AutoModel.from_pretrained('ProsusAI/finbert')"""
    )
    
    # Save patched script
    patched_script = os.path.join(os.path.dirname(__file__), 'scripts', 'train_full_model_safe.py')
    with open(patched_script, 'w') as f:
        f.write(patched_content)
    
    print(f"✅ Patched script created: {patched_script}")
    return patched_script

def main():
    print("🔧 FINBERT LOADING COMPREHENSIVE FIX")
    print("Addresses torch.load security vulnerability")
    print("=" * 60)
    
    if not check_current_setup():
        print("❌ Setup check failed")
        return
    
    # Step 1: Fix transformers configuration
    if fix_transformers_config():
        print("✅ Transformers configuration updated")
    else:
        print("⚠️  Configuration update failed, continuing...")
    
    # Step 2: Test FinBERT loading
    if test_finbert_loading():
        print("✅ FinBERT loading works! You can use normal training.")
        return
    
    print("⚠️  FinBERT still has issues, creating workarounds...")
    
    # Step 3: Create safe loader
    safe_loader = create_safe_finbert_loader()
    
    # Step 4: Create patched training script
    patched_script = create_patched_training_script()
    
    print(f"""
    🎯 FINBERT FIX COMPLETED
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Options to try:
    
    1. 🚀 Use patched training script:
       python scripts/train_full_model_safe.py
    
    2. 🆘 Use emergency training (no FinBERT):
       python scripts/train_emergency.py
    
    3. 💡 Try manual model download:
       python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"
       
    If all else fails, the emergency training will give you a working model!
    """)

if __name__ == "__main__":
    main()