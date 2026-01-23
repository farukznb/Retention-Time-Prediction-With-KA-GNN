"""
Unified trainer for all RT prediction models.

This module provides a Trainer class that works with any model
inheriting from BaseRTModel.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from pathlib import Path
from tqdm import tqdm
import time


class Trainer:
    """
    Unified trainer for retention time prediction models.
    
    This trainer works with any model that inherits from BaseRTModel
    and provides a consistent training interface across all model types.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device: str = 'cuda',
        checkpoint_path: Optional[Path] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model instance (must inherit from BaseRTModel)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer instance
            criterion: Loss function
            device: Device to train on ('cuda' or 'cpu')
            checkpoint_path: Path to save best model checkpoint
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.training_time = 0
    
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            loss, n_samples = self.model._train_step(batch, self.optimizer, self.criterion)
            total_loss += loss
            total_samples += n_samples
        
        return total_loss / total_samples
    
    def validate(self) -> float:
        """
        Validate the model.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                loss, n_samples = self.model._val_step(batch, self.criterion)
                total_loss += loss
                total_samples += n_samples
        
        return total_loss / total_samples
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing training history
        """
        if verbose:
            print(f"\n{'='*80}")
            print("TRAINING")
            print(f"{'='*80}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            if verbose:
                print(f"Epoch {epoch:03d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping check
            if val_loss < self.best_val_loss - 1e-3:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if self.checkpoint_path:
                    self.model.save(self.checkpoint_path)
                    if verbose:
                        print(f"  ✓ Saved checkpoint (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n Early stopping at epoch {epoch}")
                    break
        
        self.training_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Training completed in {self.training_time:.2f}s ({self.training_time/60:.2f} min)")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'training_time': self.training_time,
            'best_val_loss': self.best_val_loss
        }
    
    def load_best_checkpoint(self):
        """Load the best checkpoint saved during training."""
        if self.checkpoint_path and self.checkpoint_path.exists():
            self.model.load(self.checkpoint_path, map_location=str(self.device))
            print(f"✓ Best checkpoint loaded from {self.checkpoint_path}")
        else:
            print(" No checkpoint found to load")
