#!/usr/bin/env python3
"""
Mac-Optimized Training Script with Metal Performance Shaders (MPS) and CoreML
Optimized for M1/M2 MacBooks with checkpoint resumption support
"""

import os
import sys
import ssl
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
import time
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, List, Tuple

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# SSL fixes
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

warnings.filterwarnings('ignore')

def setup_mac_optimization():
    """Configure optimal settings for Mac M1/M2 with MPS"""
    print("""
    🍎 MAC OPTIMIZATION SETUP
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # Check for MPS (Metal Performance Shaders) support
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("✅ Metal Performance Shaders (MPS) detected!")
        print("🚀 Using Apple Silicon GPU acceleration")
        
        # MPS specific optimizations
        torch.backends.mps.allow_tf32 = True
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Reduce memory fragmentation
        
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print("✅ CUDA GPU detected")
        
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU - consider using MPS-enabled PyTorch")
        print("💡 Install with: pip install torch torchvision torchaudio")
    
    # Mac-specific optimizations
    if sys.platform == 'darwin':  # macOS
        print("🍎 Applying macOS optimizations...")
        
        # Set optimal thread count for Apple Silicon
        if device.type == 'mps':
            # Use all efficiency + performance cores
            torch.set_num_threads(8)  # Typical for M1/M2
        else:
            torch.set_num_threads(4)  # Conservative for Intel Macs
        
        # Memory management for Apple Silicon
        os.environ['PYTORCH_MPS_PREFER_MPS_ALWAYS'] = '1'
        
        print(f"✅ Configured for {torch.get_num_threads()} threads")
    
    print(f"🖥️  Selected device: {device}")
    return device

def check_coreml_availability():
    """Check if CoreML is available for potential optimizations"""
    try:
        import coremltools as ct
        print("✅ CoreML Tools available for model conversion")
        return True
    except ImportError:
        print("⚠️  CoreML Tools not installed")
        print("💡 Install with: pip install coremltools")
        return False

class MacOptimizedConfig:
    """Configuration optimized for Mac training"""
    def __init__(self, device_type='mps'):
        # Model architecture (same as original for checkpoint compatibility)
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.ff_dim = 2048
        self.dropout = 0.1
        
        # Input features (keep compatible with existing checkpoints)
        self.price_features = 12
        self.macro_features = 20  
        self.text_features = 768
        
        # Mac-optimized training parameters
        if device_type == 'mps':
            # Optimized for Apple Silicon
            self.sequence_length = 60
            self.batch_size = 16      # Balanced for MPS memory
            self.learning_rate = 5e-4  # Slightly higher for faster convergence
            self.gradient_accumulation_steps = 2  # Effective batch size = 32
            
        else:
            # Standard settings for other devices
            self.sequence_length = 60
            self.batch_size = 8
            self.learning_rate = 1e-4
            self.gradient_accumulation_steps = 4
        
        # Output dimensions
        self.forecast_horizon = 5
        self.num_assets = 15
        
        # Loss weights
        self.forecast_weight = 1.0
        self.anomaly_weight = 0.3  # Reduced for faster convergence

def find_compatible_checkpoint() -> Optional[str]:
    """Find the latest compatible checkpoint (handles fresh installations)"""
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        print("📁 No checkpoints directory found (fresh installation)")
        return None
    
    # Look for different types of checkpoints
    checkpoint_patterns = [
        'checkpoint_epoch_',      # Original full training checkpoints
        'mac_checkpoint_epoch_',  # Mac-optimized checkpoints  
        'fast_checkpoint_epoch_'  # Fast training checkpoints (less compatible)
    ]
    
    all_checkpoints = []
    
    for pattern in checkpoint_patterns:
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(pattern)]
        for checkpoint in checkpoints:
            try:
                epoch_num = int(checkpoint.split('_')[-1].split('.')[0])
                all_checkpoints.append((checkpoint, epoch_num, pattern))
            except (ValueError, IndexError):
                continue
    
    if not all_checkpoints:
        print("📁 Checkpoints directory exists but no compatible checkpoints found")
        return None
    
    # Sort by epoch number and prefer compatible checkpoints
    # Priority: mac > original > fast
    priority_order = ['mac_checkpoint_epoch_', 'checkpoint_epoch_', 'fast_checkpoint_epoch_']
    
    def checkpoint_priority(item):
        checkpoint, epoch, pattern = item
        priority = priority_order.index(pattern) if pattern in priority_order else 999
        return (priority, epoch)
    
    all_checkpoints.sort(key=checkpoint_priority, reverse=True)
    best_checkpoint = all_checkpoints[0]
    
    checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint[0])
    print(f"🔍 Found compatible checkpoint: {checkpoint_path}")
    print(f"📊 Checkpoint type: {best_checkpoint[2]}, Epoch: {best_checkpoint[1]}")
    
    # Validate checkpoint compatibility
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"✅ Checkpoint validated: Epoch {epoch}, Loss {loss:.6f}")
        return checkpoint_path
    except Exception as e:
        print(f"⚠️  Checkpoint validation failed: {e}")
        print("🆕 Will start fresh training instead")
        return None

def load_checkpoint_compatible(checkpoint_path: str, model, optimizer, device):
    """Load checkpoint with device compatibility"""
    try:
        # Load checkpoint to CPU first
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state and move to device
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Move optimizer state to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        history = checkpoint.get('history', [])
        
        print(f"✅ Checkpoint loaded successfully: epoch {epoch}, loss {loss:.6f}")
        return epoch, loss, history
        
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return 0, float('inf'), []

def save_mac_checkpoint(model, optimizer, epoch, loss, config, history, device):
    """Save checkpoint with Mac-specific metadata"""
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to CPU for saving to avoid device-specific issues
    model_cpu = model.cpu()
    
    checkpoint_path = os.path.join(checkpoint_dir, f'mac_checkpoint_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'history': history,
        'device_type': device.type,
        'mac_optimized': True,
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__
    }, checkpoint_path)
    
    # Move model back to device
    model.to(device)
    
    print(f"💾 Mac checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def mac_optimized_training():
    """Main Mac-optimized training function"""
    print("""
    🍎 MAC-OPTIMIZED TRAINING SYSTEM
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Optimizations for Apple Silicon (M1/M2):
    ✅ Metal Performance Shaders (MPS) acceleration
    ✅ Optimized memory management  
    ✅ Apple Silicon thread configuration
    ✅ Checkpoint resumption support
    ✅ Gradient accumulation for effective large batches
    
    """)
    
    # Setup Mac optimizations
    device = setup_mac_optimization()
    coreml_available = check_coreml_availability()
    
    # Import ML modules
    try:
        from src.models.unified_transformer import UnifiedMultimodalTransformer, ModelConfig
        from src.training.trainer import FinancialTrainer
        print("✅ ML modules imported successfully")
    except Exception as e:
        print(f"❌ Failed to import ML modules: {e}")
        return
    
    # Check for existing checkpoint
    checkpoint_path = find_compatible_checkpoint()
    resume_training = checkpoint_path is not None
    
    if resume_training:
        print(f"""
    🔄 RESUMING FROM EXISTING CHECKPOINT
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Found: {checkpoint_path}
    Will continue training from where it left off
    No progress will be lost!
        """)
    else:
        print(f"""
    🆕 STARTING FRESH TRAINING
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    No existing checkpoints found - this is normal for:
    • Fresh repository clones
    • First-time training
    • New installations
    
    Training will start from Epoch 1
        """)
    
    print(f"""
    💡 CHECKPOINT INFO:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    • Checkpoints are saved locally (not in git)
    • Will be created automatically during training
    • Can resume if training is interrupted
    • Compatible across different machines/platforms
    """)
    
    # Create Mac-optimized configuration
    config = MacOptimizedConfig(device_type=device.type)
    
    # Create model (use original architecture for compatibility)
    original_config = ModelConfig()
    model = UnifiedMultimodalTransformer(original_config)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"""
    📋 MAC TRAINING CONFIGURATION:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Device: {device} ({'Apple Silicon MPS' if device.type == 'mps' else device.type.upper()})
    Model: {total_params:,} parameters (19.6M)
    Batch Size: {config.batch_size} (effective: {config.batch_size * config.gradient_accumulation_steps})
    Learning Rate: {config.learning_rate}
    Threads: {torch.get_num_threads()}
    Resume Training: {resume_training}
    CoreML Available: {coreml_available}
    """)
    
    # Move model to device
    model = model.to(device)
    
    # Optimizer with Mac-optimized settings
    if device.type == 'mps':
        # Use AdamW with settings optimized for MPS
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-4,
            eps=1e-7  # Slightly larger eps for MPS numerical stability
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Load checkpoint if available
    start_epoch = 0
    best_loss = float('inf')
    history = []
    
    if resume_training:
        start_epoch, best_loss, history = load_checkpoint_compatible(
            checkpoint_path, model, optimizer, device
        )
        start_epoch += 1  # Start from next epoch
    
    # Create dummy training data (in real scenario, load actual data)
    print("📊 Creating training data...")
    n_samples = 5000
    
    np.random.seed(42)
    price_data = torch.randn(n_samples, config.sequence_length, original_config.price_features)
    macro_data = torch.randn(n_samples, config.sequence_length, original_config.macro_features)
    text_data = torch.randn(n_samples, config.sequence_length, original_config.text_features)
    
    forecast_targets = torch.randn(n_samples, original_config.forecast_horizon) * 0.01
    anomaly_targets = (torch.rand(n_samples, 1) > 0.95).float()
    
    # Move data to device
    price_data = price_data.to(device)
    macro_data = macro_data.to(device)
    text_data = text_data.to(device)
    forecast_targets = forecast_targets.to(device)
    anomaly_targets = anomaly_targets.to(device)
    
    # Training parameters
    total_epochs = 100  # Reasonable number for Mac
    save_every = 5      # Save more frequently
    
    # Loss functions
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    print(f"""
    🚀 Starting Mac-optimized training: Epochs {start_epoch} → {total_epochs}
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)
    
    # Training loop with Mac optimizations
    start_time = time.time()
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        model.train()
        
        total_loss = 0
        total_forecast_loss = 0
        total_anomaly_loss = 0
        n_batches = 0
        
        # Gradient accumulation for effective larger batches
        optimizer.zero_grad()
        
        for i in range(0, n_samples, config.batch_size):
            end_idx = min(i + config.batch_size, n_samples)
            
            # Get batch
            batch_price = price_data[i:end_idx]
            batch_macro = macro_data[i:end_idx]
            batch_text = text_data[i:end_idx]
            batch_forecast_targets = forecast_targets[i:end_idx]
            batch_anomaly_targets = anomaly_targets[i:end_idx]
            
            # Forward pass
            outputs = model(batch_price, batch_macro, batch_text)
            
            # Loss calculation
            forecast_loss = mse_loss(outputs['forecast'], batch_forecast_targets)
            anomaly_loss = bce_loss(outputs['anomaly_score'], batch_anomaly_targets)
            
            loss = (config.forecast_weight * forecast_loss + 
                   config.anomaly_weight * anomaly_loss)
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (n_batches + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            total_forecast_loss += forecast_loss.item()
            total_anomaly_loss += anomaly_loss.item()
            n_batches += 1
        
        # Final optimizer step if needed
        if n_batches % config.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Calculate averages
        avg_loss = total_loss / n_batches
        avg_forecast_loss = total_forecast_loss / n_batches
        avg_anomaly_loss = total_anomaly_loss / n_batches
        
        # Track history
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'forecast_loss': avg_forecast_loss,
            'anomaly_loss': avg_anomaly_loss,
            'device': device.type
        })
        
        # Progress display
        epoch_time = time.time() - epoch_start
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{total_epochs} | "
              f"Loss: {avg_loss:.6f} | "
              f"Forecast: {avg_forecast_loss:.6f} | "
              f"Anomaly: {avg_anomaly_loss:.6f} | "
              f"Time: {epoch_time:.1f}s | "
              f"Device: {device.type}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            
        if (epoch + 1) % save_every == 0 or avg_loss == best_loss:
            save_mac_checkpoint(model, optimizer, epoch, avg_loss, config, history, device)
    
    total_time = time.time() - start_time
    
    # Save final model
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    final_model_path = os.path.join(models_dir, 'mac_optimized_transformer.pth')
    model_cpu = model.cpu()
    
    torch.save({
        'model_state_dict': model_cpu.state_dict(),
        'config': original_config,
        'mac_config': config,
        'history': history,
        'final_loss': best_loss,
        'training_time_hours': total_time / 3600,
        'device_trained': device.type,
        'mac_optimized': True,
        'total_epochs': len(history),
        'created': datetime.now().isoformat()
    }, final_model_path)
    
    print(f"""
    🎉 MAC-OPTIMIZED TRAINING COMPLETED!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    ✅ Total time: {total_time/3600:.1f} hours
    ✅ Final loss: {best_loss:.6f}
    ✅ Device used: {device.type}
    ✅ Epochs completed: {len(history)}
    ✅ Model saved: {final_model_path}
    
    🚀 Ready for presentation with Mac-optimized model!
    """)
    
    # Optional: Convert to CoreML if available
    if coreml_available:
        convert_choice = input("\n🍎 Convert model to CoreML format? (y/N): ").strip().lower()
        if convert_choice == 'y':
            try:
                convert_to_coreml(model, config, final_model_path)
            except Exception as e:
                print(f"⚠️  CoreML conversion failed: {e}")

def convert_to_coreml(model, config, model_path):
    """Convert trained model to CoreML format"""
    try:
        import coremltools as ct
        
        print("🔄 Converting to CoreML format...")
        
        # Create dummy input
        dummy_price = torch.randn(1, config.sequence_length, 12)
        dummy_macro = torch.randn(1, config.sequence_length, 20)
        dummy_text = torch.randn(1, config.sequence_length, 768)
        
        # Trace the model
        model.eval()
        traced_model = torch.jit.trace(model, (dummy_price, dummy_macro, dummy_text))
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[
                ct.TensorType(name="price_data", shape=dummy_price.shape),
                ct.TensorType(name="macro_data", shape=dummy_macro.shape), 
                ct.TensorType(name="text_data", shape=dummy_text.shape)
            ]
        )
        
        # Save CoreML model
        coreml_path = model_path.replace('.pth', '.mlmodel')
        coreml_model.save(coreml_path)
        
        print(f"✅ CoreML model saved: {coreml_path}")
        print("🍎 Model ready for iOS/macOS deployment!")
        
    except Exception as e:
        print(f"❌ CoreML conversion failed: {e}")

if __name__ == "__main__":
    mac_optimized_training()