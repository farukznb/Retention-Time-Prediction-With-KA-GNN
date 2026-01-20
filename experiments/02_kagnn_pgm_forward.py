"""
KAGNN→PGM Forward Residual Experiment.

This script trains the KAGNN→PGM model in two stages:
1. Train KAGNN backbone
2. Train PGM to correct residuals
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import time

# Import from src/
from src.data.dataset import SMRTCombinedDataset, collate_fn
from src.models.kagnn_pgm_forward import KAGNN_PGM_Forward
from src.evaluation.metrics import RTMetrics
from src.evaluation.visualization import RTVisualizer
from src.utils.seed import set_seed


class Config:
    # Data paths
    ecfp_file = "/kaggle/input/smrtdatas/SMRT_ECFP_1024_Fingerprints.txt"
    rt_file = "/kaggle/input/smrtdt/SMRT_dataset.csv"
    sdf_file = "/kaggle/input/smrtdatas/SMRT_dataset.sdf"
    
    # Training hyperparameters - Stage 1 (KAGNN)
    batch_size = 64
    epochs = 150
    lr = 3e-4
    weight_decay = 1e-5
    
    # Model architecture
    gnn_layers = 5
    hidden_dim = 256
    hidden_layers = 2
    grid_size = 4
    spline_order = 3
    dropout = 0.1
    
    # Stage 2 (PGM) hyperparameters
    pgm_estimators = 50
    pgm_max_samples = 0.8
    
    # Data split
    val_size = 0.15
    test_size = 0.15
    seed = 42
    
    # Output paths
    results_dir = Path("results")
    checkpoint_path = results_dir / "checkpoints" / "kagnn_pgm_forward.pt"
    metrics_path = results_dir / "metrics" / "kagnn_pgm_forward.json"
    plots_dir = results_dir / "plots" / "kagnn_pgm_forward"


def main():
    cfg = Config()
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Create output directories
    cfg.results_dir.mkdir(exist_ok=True)
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    train_ds = SMRTCombinedDataset(cfg, 'train')
    val_ds = SMRTCombinedDataset(cfg, 'val')
    test_ds = SMRTCombinedDataset(cfg, 'test')
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"RT normalization: mean={train_ds.rt_mean:.2f}, std={train_ds.rt_std:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    model = KAGNN_PGM_Forward(cfg).to(device)
    model.set_normalization_params(train_ds.rt_mean, train_ds.rt_std)
    model.set_mol_dict(train_ds.mol_dict)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup Stage 1 training
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.SmoothL1Loss()
    
    # STAGE 1: Train KAGNN
    print("\n" + "="*80)
    print("STAGE 1: TRAINING KAGNN BACKBONE")
    print("="*80)
    
    start_time = time.time()
    
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.epochs,
        optimizer=optimizer,
        criterion=criterion,
        early_stopping_patience=20,
        checkpoint_path=cfg.checkpoint_path.parent / "kagnn_stage1.pt"
    )
    
    stage1_time = time.time() - start_time
    print(f"\nStage 1 completed in {stage1_time:.2f} seconds ({stage1_time/60:.2f} minutes)")
    
    # Load best Stage 1 model
    model.load(cfg.checkpoint_path.parent / "kagnn_stage1.pt", map_location=str(device))
    
    # Evaluate KAGNN only (before PGM correction)
    print("\n" + "="*80)
    print("EVALUATING KAGNN ONLY (BEFORE PGM CORRECTION)")
    print("="*80)
    
    # Temporarily disable PGM for evaluation
    pgm_xgb_backup = model.pgm_xgb
    pgm_br_backup = model.pgm_br
    model.pgm_xgb = None
    model.pgm_br = None
    
    y_true_kagnn, y_pred_kagnn = model.predict(test_loader, denormalize=True)
    
    metrics_kagnn_only = RTMetrics(
        y_true=y_true_kagnn,
        y_pred=y_pred_kagnn,
        model_name="KAGNN Only (Stage 1)"
    )
    metrics_kagnn_only.print_summary()
    
    # Restore PGM models
    model.pgm_xgb = pgm_xgb_backup
    model.pgm_br = pgm_br_backup
    
    # STAGE 2: Train PGM
    print("\n" + "="*80)
    print("STAGE 2: TRAINING PGM RESIDUAL CORRECTION")
    print("="*80)
    
    start_time = time.time()
    
    model.train_stage2_pgm(
        train_loader=train_loader,
        n_estimators=cfg.pgm_estimators,
        max_samples=cfg.pgm_max_samples,
        verbose=True
    )
    
    stage2_time = time.time() - start_time
    print(f"\nStage 2 completed in {stage2_time:.2f} seconds ({stage2_time/60:.2f} minutes)")
    
    # Save complete model (KAGNN + PGM)
    model.save(cfg.checkpoint_path)
    
    # Final evaluation with PGM correction
    print("\n" + "="*80)
    print("FINAL EVALUATION (KAGNN + PGM)")
    print("="*80)
    
    y_true, y_pred = model.predict(test_loader, denormalize=True)
    
    # Create baseline predictions (KAGNN only)
    y_pred_baseline = y_pred_kagnn
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = RTMetrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name="KAGNN→PGM Forward",
        y_pred_baseline=y_pred_baseline
    )
    
    # Save metrics
    metrics.save(cfg.metrics_path)
    
    # Print summary with benchmark
    benchmark = {
        'MedAE': 24.68,
        'MAE': 46.23,
        'RMSE': 89.11,
        'R2': 0.81,
        'Pearson': 0.91,
        'Spearman': 0.93,
        'Pct_le_60s': 80.75,
        'Pct_le_30s': 57.16,
        'Pct_le_10s': 24.09
    }
    
    metrics.print_summary(benchmark=benchmark)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    visualizer = RTVisualizer(
        y_true=y_true,
        y_pred=y_pred,
        model_name="KAGNN→PGM Forward"
    )
    
    # Individual plots
    visualizer.plot_predictions(
        save_path=cfg.plots_dir / "predictions.png",
        benchmark=benchmark
    )
    
    visualizer.plot_error_distribution(
        save_path=cfg.plots_dir / "error_distribution.png"
    )
    
    visualizer.plot_residuals(
        save_path=cfg.plots_dir / "residuals.png"
    )
    
    # Comprehensive plot
    visualizer.plot_comprehensive(
        save_path=cfg.plots_dir / "comprehensive_analysis.png",
        benchmark=benchmark
    )
    
    # Training history plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('KAGNN→PGM Training History (Stage 1)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison plot: KAGNN vs KAGNN+PGM
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # KAGNN Only
    axes[0].scatter(y_true_kagnn, y_pred_kagnn, alpha=0.5, s=20, c='royalblue')
    lims = [min(y_true_kagnn.min(), y_pred_kagnn.min()), 
            max(y_true_kagnn.max(), y_pred_kagnn.max())]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
    axes[0].set_title(f'KAGNN Only\nMedAE={metrics_kagnn_only.metrics["MedAE"]:.2f}s, '
                     f'R²={metrics_kagnn_only.metrics["R2"]:.3f}', 
                     fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # KAGNN + PGM
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=20, c='mediumseagreen')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, 'r--', lw=2)
    axes[1].set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
    axes[1].set_title(f'KAGNN + PGM\nMedAE={metrics.metrics["MedAE"]:.2f}s, '
                     f'R²={metrics.metrics["R2"]:.3f}',
                     fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "kagnn_vs_pgm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAll plots saved to {cfg.plots_dir}")
    
    # Runtime summary
    total_time = stage1_time + stage2_time
    print("\n" + "="*80)
    print("RUNTIME SUMMARY")
    print("="*80)
    print(f"Stage 1 (KAGNN):       {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"Stage 2 (PGM):         {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"Total Training Time:   {total_time:.2f}s ({total_time/60:.2f} min)")
    print("="*80)
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Model: KAGNN→PGM Forward Residual")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Test samples: {len(y_true)}")
    print(f"Checkpoint saved: {cfg.checkpoint_path}")
    print(f"Metrics saved: {cfg.metrics_path}")
    print(f"Plots saved: {cfg.plots_dir}")
    
    print("\nKey Results:")
    print(f"  KAGNN Only:")
    print(f"    MedAE: {metrics_kagnn_only.metrics['MedAE']:.2f}s")
    print(f"    MAE: {metrics_kagnn_only.metrics['MAE']:.2f}s")
    print(f"    R²: {metrics_kagnn_only.metrics['R2']:.3f}")
    
    print(f"\n  KAGNN + PGM:")
    print(f"    MedAE: {metrics.metrics['MedAE']:.2f}s")
    print(f"    MAE: {metrics.metrics['MAE']:.2f}s")
    print(f"    R²: {metrics.metrics['R2']:.3f}")
    print(f"    % ≤ 30s: {metrics.metrics['Pct_le_30s']:.2f}%")
    
    improvement_medae = ((metrics_kagnn_only.metrics['MedAE'] - metrics.metrics['MedAE']) / 
                         metrics_kagnn_only.metrics['MedAE'] * 100)
    print(f"\n  Improvement: {improvement_medae:.2f}% reduction in MedAE")
    
    print("="*80)


if __name__ == "__main__":
    main()
