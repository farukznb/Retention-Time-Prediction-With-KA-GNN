"""
Unified trainer for all RT prediction models (FIXED VERSION).

Key improvements:
1. Gradient clipping for stability
2. Learning rate scheduling
3. Mixed precision training support
4. Better error handling
5. Progress tracking with metrics
6. Flexible checkpoint saving
7. Support for multi-stage models
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any, Callable
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import warnings


class Trainer:
    """
    Unified trainer for retention time prediction models (FIXED & ENHANCED).
    
    This trainer works with any model that inherits from BaseRTModel
    and provides a consistent training interface across all model types.
    
    Key features:
    - Automatic gradient clipping
    - Mixed precision training (if available)
    - Learning rate scheduling
    - Early stopping with patience
    - Comprehensive logging
    - Error recovery
    """
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device: str = 'cuda',
        checkpoint_path: Optional[Path] = None,
        scheduler: Optional[Any] = None,
        use_amp: bool = False,
        gradient_clip_value: float = 1.0
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
            scheduler: Learning rate scheduler (optional)
            use_amp: Use automatic mixed precision (faster on modern GPUs)
            gradient_clip_value: Maximum gradient norm for clipping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.scheduler = scheduler
        self.gradient_clip_value = gradient_clip_value
        
        # Mixed precision training
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = GradScaler()
            print("✓ Using automatic mixed precision (AMP)")
        else:
            self.scaler = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.training_time = 0
        self.epoch_times = []
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self) -> tuple:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        nan_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move batch to device (if not already done by model)
                if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                    graph, ecfp, rt = batch[:3]
                    graph = graph.to(self.device)
                    ecfp = ecfp.to(self.device)
                    rt = rt.to(self.device)
                    batch = (graph, ecfp, rt) + batch[3:]  # Keep other elements if any
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with optional AMP
                if self.use_amp:
                    with autocast():
                        if hasattr(self.model, '_train_step'):
                            loss, n_samples = self.model._train_step(
                                batch, self.optimizer, self.criterion
                            )
                        else:
                            # Fallback: manual training step
                            pred = self.model(graph, ecfp)
                            loss = self.criterion(pred, rt)
                            n_samples = len(rt)
                    
                    # Check for NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        nan_batches += 1
                        if nan_batches > 5:
                            raise ValueError("Too many NaN/Inf losses detected")
                        continue
                    
                    # Backward pass with scaling
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_value > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.gradient_clip_value
                        )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    
                else:
                    # Regular training (no AMP)
                    if hasattr(self.model, '_train_step'):
                        loss, n_samples = self.model._train_step(
                            batch, self.optimizer, self.criterion
                        )
                    else:
                        # Fallback: manual training step
                        pred = self.model(graph, ecfp)
                        loss = self.criterion(pred, rt)
                        n_samples = len(rt)
                        
                        # Check for NaN/Inf
                        if torch.isnan(loss) or torch.isinf(loss):
                            nan_batches += 1
                            if nan_batches > 5:
                                raise ValueError("Too many NaN/Inf losses detected")
                            continue
                        
                        # Backward pass
                        loss.backward()
                        
                        # Gradient clipping
                        if self.gradient_clip_value > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.gradient_clip_value
                            )
                        
                        # Optimizer step
                        self.optimizer.step()
                
                # Accumulate loss
                total_loss += loss.item() * n_samples
                total_samples += n_samples
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                print(f"\n  Error in batch {batch_idx}: {e}")
                continue
        
        if nan_batches > 0:
            print(f"\n  Skipped {nan_batches} batches due to NaN/Inf")
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss, {'n_samples': total_samples, 'nan_batches': nan_batches}
    
    def validate(self) -> tuple:
        """
        Validate the model.
        
        Returns:
            Tuple of (average_loss, metrics_dict)
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                try:
                    # Move batch to device
                    if isinstance(batch, (list, tuple)) and len(batch) >= 3:
                        graph, ecfp, rt = batch[:3]
                        graph = graph.to(self.device)
                        ecfp = ecfp.to(self.device)
                        rt = rt.to(self.device)
                        batch = (graph, ecfp, rt) + batch[3:]
                    
                    if hasattr(self.model, '_val_step'):
                        loss, n_samples = self.model._val_step(batch, self.criterion)
                    else:
                        # Fallback: manual validation step
                        pred = self.model(graph, ecfp)
                        loss = self.criterion(pred, rt)
                        n_samples = len(rt)
                    
                    # Skip if NaN/Inf
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    total_loss += loss.item() * n_samples
                    total_samples += n_samples
                    
                except Exception as e:
                    print(f"\n  Validation error: {e}")
                    continue
        
        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        
        return avg_loss, {'n_samples': total_samples}
    
    def train(
        self,
        epochs: int,
        early_stopping_patience: int = 20,
        min_delta: float = 1e-4,
        verbose: bool = True,
        log_interval: int = 5
    ) -> Dict[str, list]:
        """
        Train the model with early stopping.
        
        Args:
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            min_delta: Minimum improvement to reset patience
            verbose: Whether to print progress
            log_interval: Print summary every N epochs
            
        Returns:
            Dictionary containing training history
        """
        if verbose:
            print(f"\n{'='*80}")
            print("TRAINING")
            print(f"{'='*80}")
            print(f"Epochs: {epochs}")
            print(f"Early stopping patience: {early_stopping_patience}")
            print(f"Gradient clipping: {self.gradient_clip_value}")
            print(f"Device: {self.device}")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            try:
                train_loss, train_metrics = self.train_epoch()
                self.train_losses.append(train_loss)
                self.train_metrics.append(train_metrics)
            except Exception as e:
                print(f"\n Training failed at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Validation
            try:
                val_loss, val_metrics = self.validate()
                self.val_losses.append(val_loss)
                self.val_metrics.append(val_metrics)
            except Exception as e:
                print(f"\n Validation failed at epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                break
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Track epoch time
            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)
            
            # Logging
            if verbose and (epoch % log_interval == 0 or epoch == 1):
                print(f"\nEpoch {epoch:03d}/{epochs}:")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  LR:         {current_lr:.6e}")
                print(f"  Time:       {epoch_time:.2f}s")
            
            # Early stopping check
            improvement = self.best_val_loss - val_loss
            
            if improvement > min_delta:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                # Save checkpoint
                if self.checkpoint_path:
                    try:
                        # Ensure parent directory exists
                        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save model
                        if hasattr(self.model, 'save'):
                            self.model.save(self.checkpoint_path)
                        else:
                            # Fallback: save state dict
                            checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'val_loss': val_loss,
                                'train_loss': train_loss
                            }
                            torch.save(checkpoint, self.checkpoint_path)
                        
                        if verbose and epoch % log_interval == 0:
                            print(f"  ✓ Checkpoint saved (improvement: {improvement:.6f})")
                    except Exception as e:
                        print(f"    Failed to save checkpoint: {e}")
            else:
                patience_counter += 1
                if verbose and patience_counter % 5 == 0:
                    print(f"  Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\n Early stopping at epoch {epoch}")
                        print(f"   Best val loss: {self.best_val_loss:.6f}")
                    break
        
        self.training_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*80}")
            print("TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"Total time: {self.training_time:.2f}s ({self.training_time/60:.2f} min)")
            print(f"Avg epoch time: {np.mean(self.epoch_times):.2f}s")
            print(f"Best val loss: {self.best_val_loss:.6f}")
            print(f"Final train loss: {self.train_losses[-1]:.6f}")
            print(f"Final val loss: {self.val_losses[-1]:.6f}")
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
            'training_time': self.training_time,
            'best_val_loss': self.best_val_loss,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
    
    def load_best_checkpoint(self, verbose: bool = True):
        """
        Load the best checkpoint saved during training.
        
        Args:
            verbose: Whether to print loading status
        """
        if self.checkpoint_path and self.checkpoint_path.exists():
            try:
                if hasattr(self.model, 'load'):
                    self.model.load(self.checkpoint_path, map_location=str(self.device))
                else:
                    # Fallback: load state dict
                    checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                if verbose:
                    print(f" Best checkpoint loaded from {self.checkpoint_path}")
            except Exception as e:
                print(f" Failed to load checkpoint: {e}")
                if verbose:
                    import traceback
                    traceback.print_exc()
        else:
            if verbose:
                print(f"  No checkpoint found at {self.checkpoint_path}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training process.
        
        Returns:
            Dictionary with training summary
        """
        if not self.train_losses:
            return {}
        
        return {
            'total_epochs': len(self.train_losses),
            'total_time_seconds': self.training_time,
            'avg_epoch_time_seconds': np.mean(self.epoch_times) if self.epoch_times else 0,
            'best_val_loss': self.best_val_loss,
            'final_train_loss': self.train_losses[-1],
            'final_val_loss': self.val_losses[-1],
            'initial_train_loss': self.train_losses[0],
            'initial_val_loss': self.val_losses[0],
            'train_loss_reduction': self.train_losses[0] - self.train_losses[-1],
            'val_loss_reduction': self.val_losses[0] - self.val_losses[-1],
            'final_learning_rate': self.learning_rates[-1] if self.learning_rates else None,
            'converged': len(self.train_losses) < len(self.epoch_times) + 1  # Early stopped
        }
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """
        Plot training history (loss curves and learning rate).
        
        Args:
            save_path: Path to save the plot
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0].plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Learning rate
        if self.learning_rates:
            axes[1].plot(epochs, self.learning_rates, linewidth=2, color='green')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Learning Rate', fontsize=12)
            axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Training history plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


# Convenience function
def create_trainer(
    model,
    train_loader,
    val_loader,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-5,
    device: str = 'cuda',
    checkpoint_path: Optional[Path] = None,
    use_scheduler: bool = True,
    use_amp: bool = False,
    gradient_clip: float = 1.0
) -> Trainer:
    """
    Convenience function to create a trainer with sensible defaults.
    
    Args:
        model: Model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        device: Training device
        checkpoint_path: Path to save checkpoints
        use_scheduler: Whether to use ReduceLROnPlateau scheduler
        use_amp: Use automatic mixed precision
        gradient_clip: Gradient clipping value
        
    Returns:
        Configured Trainer instance
    """
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Criterion
    criterion = nn.SmoothL1Loss()
    
    # Scheduler
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
    
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
        scheduler=scheduler,
        use_amp=use_amp,
        gradient_clip_value=gradient_clip
    )
