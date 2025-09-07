"""
Training Pipeline for Unified Multimodal Transformer
Handles model training, validation, and cross-market adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path

class FinancialTrainer:
    """Trainer class for the multimodal transformer"""
    
    def __init__(self, model, config, device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        self.model = model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        # Training history
        self.train_history = []
        self.val_history = []
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup training device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        print(f"Using device: {device}")
        return torch.device(device)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        forecast_loss_sum = 0.0
        anomaly_loss_sum = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                batch['price_data'],
                batch['macro_data'], 
                batch['text_data']
            )
            
            # Compute losses
            forecast_loss = nn.MSELoss()(outputs['forecast'], batch['price_targets'])
            anomaly_loss = nn.BCELoss()(outputs['anomaly_score'].squeeze(), 
                                       batch['anomaly_labels'].float())
            
            total_loss_batch = forecast_loss + 0.5 * anomaly_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            forecast_loss_sum += forecast_loss.item()
            anomaly_loss_sum += anomaly_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss_batch.item():.4f}")
        
        return {
            'total_loss': total_loss / num_batches,
            'forecast_loss': forecast_loss_sum / num_batches,
            'anomaly_loss': anomaly_loss_sum / num_batches
        }
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        forecast_loss_sum = 0.0
        anomaly_loss_sum = 0.0
        num_batches = 0
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    batch['price_data'],
                    batch['macro_data'],
                    batch['text_data']
                )
                
                # Compute losses
                forecast_loss = nn.MSELoss()(outputs['forecast'], batch['price_targets'])
                anomaly_loss = nn.BCELoss()(outputs['anomaly_score'].squeeze(), 
                                           batch['anomaly_labels'].float())
                
                total_loss_batch = forecast_loss + 0.5 * anomaly_loss
                
                # Accumulate losses
                total_loss += total_loss_batch.item()
                forecast_loss_sum += forecast_loss.item()
                anomaly_loss_sum += anomaly_loss.item()
                num_batches += 1
                
                # Store predictions for metrics
                predictions.append(outputs['forecast'].cpu().numpy())
                targets.append(batch['price_targets'].cpu().numpy())
        
        # Calculate additional metrics
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Directional accuracy (simplified)
        pred_direction = np.sign(predictions[:, 0])  # First day prediction
        true_direction = np.sign(targets[:, 0])
        directional_accuracy = np.mean(pred_direction == true_direction)
        
        return {
            'total_loss': total_loss / num_batches,
            'forecast_loss': forecast_loss_sum / num_batches,
            'anomaly_loss': anomaly_loss_sum / num_batches,
            'directional_accuracy': directional_accuracy
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int = 50, save_dir: str = 'checkpoints') -> Dict[str, List[float]]:
        """Main training loop"""
        
        # Create save directory
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Store history
            self.train_history.append(train_metrics)
            self.val_history.append(val_metrics)
            
            epoch_time = time.time() - start_time
            
            # Print epoch results
            print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_metrics['total_loss']:.4f} "
                  f"(Forecast: {train_metrics['forecast_loss']:.4f}, "
                  f"Anomaly: {train_metrics['anomaly_loss']:.4f})")
            print(f"Val Loss: {val_metrics['total_loss']:.4f} "
                  f"(Forecast: {val_metrics['forecast_loss']:.4f}, "
                  f"Anomaly: {val_metrics['anomaly_loss']:.4f})")
            print(f"Directional Accuracy: {val_metrics['directional_accuracy']:.3f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                patience_counter = 0
                
                # Save model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'config': self.config,
                    'val_loss': best_val_loss,
                    'train_history': self.train_history,
                    'val_history': self.val_history
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                break
            
            print("-" * 60)
        
        print("Training completed!")
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history
        }
    
    def save_model(self, save_path: str):
        """Save model for inference"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
        
        print(f"Model saved to {save_path}")