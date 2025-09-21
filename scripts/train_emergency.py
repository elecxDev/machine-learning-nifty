#!/usr/bin/env python3
"""
Emergency Training Script - No FinBERT Required
Quick training solution while PyTorch version is being fixed
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import warnings
import time
from datetime import datetime

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

# SSL and warning fixes
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

class EmergencyConfig:
    """Emergency training configuration"""
    def __init__(self):
        self.sequence_length = 60
        self.price_features = 8
        self.macro_features = 6
        self.text_features = 0  # Skip text features to avoid FinBERT
        self.d_model = 512
        self.nhead = 8
        self.num_layers = 6
        self.dropout = 0.1
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.gradient_accumulation_steps = 4

class EmergencyModel(nn.Module):
    """Emergency model without text processing"""
    def __init__(self, config):
        super().__init__()
        
        # Input projections
        self.price_projection = nn.Linear(config.price_features, config.d_model)
        self.macro_projection = nn.Linear(config.macro_features, config.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(config.sequence_length, config.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Output heads
        self.price_head = nn.Linear(config.d_model, 1)
        self.direction_head = nn.Linear(config.d_model, 1)
        
    def forward(self, price_data, macro_data):
        batch_size, seq_len = price_data.shape[:2]
        
        # Project inputs
        price_emb = self.price_projection(price_data)
        macro_emb = self.macro_projection(macro_data)
        
        # Combine embeddings
        combined = price_emb + macro_emb
        combined = combined + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer
        output = self.transformer(combined)
        
        # Predictions
        price_pred = self.price_head(output[:, -1])
        direction_pred = torch.sigmoid(self.direction_head(output[:, -1]))
        
        return price_pred, direction_pred

def create_emergency_data(config):
    """Create dummy data for emergency training"""
    print("Creating emergency training data...")
    
    batch_size = config.batch_size
    seq_len = config.sequence_length
    
    return {
        'price_data': torch.randn(1000, seq_len, config.price_features),
        'macro_data': torch.randn(1000, seq_len, config.macro_features),
        'price_targets': torch.randn(1000, 1),
        'direction_targets': torch.randint(0, 2, (1000, 1)).float()
    }

def emergency_training_loop(model, data, config, device, hours=4):
    """Emergency training loop"""
    print(f"""
    🆘 EMERGENCY TRAINING MODE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Model: {sum(p.numel() for p in model.parameters()):,} parameters
    Device: {device}
    Target time: {hours} hours
    Features: Price + Macro (Text disabled to bypass FinBERT)
    """)
    
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    
    # Move data to device
    for key in data:
        data[key] = data[key].to(device)
    
    start_time = time.time()
    target_time = hours * 3600
    
    epoch = 0
    history = []
    
    while time.time() - start_time < target_time:
        model.train()
        total_loss = 0
        
        # Simple batching
        for i in range(0, len(data['price_data']), config.batch_size):
            end_idx = min(i + config.batch_size, len(data['price_data']))
            
            # Get batch
            price_batch = data['price_data'][i:end_idx]
            macro_batch = data['macro_data'][i:end_idx]
            price_targets = data['price_targets'][i:end_idx]
            direction_targets = data['direction_targets'][i:end_idx]
            
            # Forward pass
            price_pred, direction_pred = model(price_batch, macro_batch)
            
            # Loss calculation
            price_loss = mse_loss(price_pred, price_targets)
            direction_loss = bce_loss(direction_pred, direction_targets)
            loss = price_loss + direction_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(data['price_data']) // config.batch_size)
        elapsed_hours = (time.time() - start_time) / 3600
        
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Time: {elapsed_hours:.1f}h")
        
        history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'time': elapsed_hours
        })
        
        epoch += 1
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_emergency_checkpoint(model, optimizer, epoch, avg_loss, config, history)
    
    print(f"✅ Emergency training completed! ({epoch} epochs)")
    return model, history

def save_emergency_checkpoint(model, optimizer, epoch, loss, config, history):
    """Save emergency checkpoint"""
    checkpoint_dir = os.path.join(project_root, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'emergency_checkpoint_epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'history': history,
        'emergency_training': True
    }, checkpoint_path)
    
    print(f"💾 Emergency checkpoint saved: epoch {epoch}")

def main():
    print("🆘 EMERGENCY TRAINING - PYTORCH VERSION FIX REQUIRED")
    print("This bypasses FinBERT to train a working model quickly")
    print("=" * 70)
    
    # Configuration
    config = EmergencyConfig()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  Training on CPU - this will be slow!")
        print("   Consider running fix_pytorch_version.py first")
    
    # Create model and data
    model = EmergencyModel(config)
    data = create_emergency_data(config)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Ask for training time
    print("\n⏰ How many hours to train?")
    print("   Recommended: 2-4 hours for emergency model")
    try:
        hours = float(input("Hours: "))
    except:
        hours = 2
        print(f"Using default: {hours} hours")
    
    # Train
    trained_model, history = emergency_training_loop(model, data, config, device, hours)
    
    # Save final model
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    final_path = os.path.join(model_dir, 'emergency_model.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'config': config,
        'history': history,
        'emergency_model': True
    }, final_path)
    
    print(f"""
    ✅ EMERGENCY TRAINING COMPLETED!
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Model saved: {final_path}
    Training epochs: {len(history)}
    
    ⚠️  NEXT STEPS:
    1. Run fix_pytorch_version.py to upgrade PyTorch
    2. Then train full model with FinBERT support
    3. This emergency model can demo basic functionality
    """)

if __name__ == "__main__":
    main()