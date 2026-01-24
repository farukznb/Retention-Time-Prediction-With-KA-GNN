"""
KAGNN→PGM Forward Residual Experiment (FIXED VERSION).

Key improvements:
1. Comprehensive statistical analysis
2. Better error handling
3. Memory-efficient training
4. Detailed diagnostics
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import time
import json

# Import from src/
from src.data.dataset import SMRTCombinedDataset, collate_fn
from src.models.kagnn_pgm_forward import KAGNN_PGM_Forward  # Use fixed version
from src.evaluation.metrics import RTMetrics
from src.evaluation.visualization import RTVisualizer
from src.utils.seed import set_seed

# Import statistical analysis utilities (if you have them)
try:
    from src.evaluation.statistical_analysis import (
        comprehensive_statistical_analysis,
        print_statistical_summary,
        compare_models_statistically
    )
    STATS_AVAILABLE = True
except ImportError:
    print("  Advanced statistical analysis not available")
    STATS_AVAILABLE = False


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


def validate_data_loader(loader, name="Loader"):
    """Validate data loader for NaN/Inf values."""
    print(f"\n🔍 Validating {name}...")
    
    for i, batch in enumerate(loader):
        graph, ecfp, rt, cids = batch
        
        # Check ECFP
        if torch.isnan(ecfp).any():
            print(f"  ⚠️  NaN detected in ECFP at batch {i}")
        if torch.isinf(ecfp).any():
            print(f"  ⚠️  Inf detected in ECFP at batch {i}")
        
        # Check RT
        if torch.isnan(rt).any():
            print(f"    NaN detected in RT at batch {i}")
        if torch.isinf(rt).any():
            print(f"    Inf detected in RT at batch {i}")
        
        # Check graph
        if torch.isnan(graph.x).any():
            print(f"    NaN detected in graph features at batch {i}")
        
        if i == 0:  # Only check first batch
            break
    
    print(f"  ✓ {name} validation complete")


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
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load datasets
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    try:
        train_ds = SMRTCombinedDataset(cfg, 'train')
        val_ds = SMRTCombinedDataset(cfg, 'val')
        test_ds = SMRTCombinedDataset(cfg, 'test')
    except Exception as e:
        print(f" Failed to load datasets: {e}")
        raise
    
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Test samples: {len(test_ds)}")
    print(f"RT normalization: mean={train_ds.rt_mean:.2f}, std={train_ds.rt_std:.2f}")
    
    # Create data loaders
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, cfg.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    
    # Validate data (check for NaN/Inf)
    validate_data_loader(train_loader, "Train Loader")
    validate_data_loader(test_loader, "Test Loader")
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    try:
        model = KAGNN_PGM_Forward(cfg).to(device)
        model.set_normalization_params(train_ds.rt_mean, train_ds.rt_std)
        model.set_mol_dict(train_ds.mol_dict)
    except Exception as e:
        print(f" Failed to initialize model: {e}")
        raise
    
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
    
    try:
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            optimizer=optimizer,
            criterion=criterion,
            early_stopping_patience=20,
            checkpoint_path=cfg.checkpoint_path.parent / "kagnn_stage1.pt"
        )
    except Exception as e:
        print(f" Stage 1 training failed: {e}")
        raise
    
    stage1_time = time.time() - start_time
    print(f"\n✓ Stage 1 completed in {stage1_time:.2f} seconds ({stage1_time/60:.2f} minutes)")
    
    # Load best Stage 1 model
    try:
        model.load(cfg.checkpoint_path.parent / "kagnn_stage1.pt", map_location=str(device))
    except Exception as e:
        print(f" Failed to load Stage 1 checkpoint: {e}")
        raise
    
    # Evaluate KAGNN only (before PGM correction)
    print("\n" + "="*80)
    print("EVALUATING KAGNN ONLY (BEFORE PGM CORRECTION)")
    print("="*80)
    
    # Temporarily disable PGM for evaluation
    pgm_xgb_backup = model.pgm_xgb
    pgm_br_backup = model.pgm_br
    model.pgm_xgb = None
    model.pgm_br = None
    
    try:
        y_true_kagnn, y_pred_kagnn = model.predict(test_loader, denormalize=True)
    except Exception as e:
        print(f" KAGNN prediction failed: {e}")
        raise
    
    # Basic metrics
    metrics_kagnn_only = RTMetrics(
        y_true=y_true_kagnn,
        y_pred=y_pred_kagnn,
        model_name="KAGNN Only (Stage 1)"
    )
    metrics_kagnn_only.print_summary()
    
    # Comprehensive statistical analysis (if available)
    if STATS_AVAILABLE:
        print("\n📊 Running comprehensive statistical analysis for KAGNN...")
        kagnn_stats = comprehensive_statistical_analysis(
            y_true_kagnn, 
            y_pred_kagnn,
            model_name="KAGNN Only",
            save_path=cfg.plots_dir / "kagnn_only_statistics.png"
        )
        print_statistical_summary(kagnn_stats, "KAGNN ONLY STATISTICAL SUMMARY")
    
    # Restore PGM models
    model.pgm_xgb = pgm_xgb_backup
    model.pgm_br = pgm_br_backup
    
    # STAGE 2: Train PGM
    print("\n" + "="*80)
    print("STAGE 2: TRAINING PGM RESIDUAL CORRECTION")
    print("="*80)
    
    start_time = time.time()
    
    try:
        model.train_stage2_pgm(
            train_loader=train_loader,
            n_estimators=cfg.pgm_estimators,
            max_samples=cfg.pgm_max_samples,
            verbose=True,
            use_cache=True  # Enable descriptor caching for speed
        )
    except Exception as e:
        print(f" Stage 2 training failed: {e}")
        raise
    
    stage2_time = time.time() - start_time
    print(f"\n✓ Stage 2 completed in {stage2_time:.2f} seconds ({stage2_time/60:.2f} minutes)")
    
    # Save complete model (KAGNN + PGM)
    try:
        model.save(cfg.checkpoint_path)
    except Exception as e:
        print(f"  Failed to save model: {e}")
    
    # Final evaluation with PGM correction
    print("\n" + "="*80)
    print("FINAL EVALUATION (KAGNN + PGM)")
    print("="*80)
    
    try:
        y_true, y_pred = model.predict(test_loader, denormalize=True, use_cache=True)
    except Exception as e:
        print(f" Final prediction failed: {e}")
        raise
    
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
    try:
        metrics.save(cfg.metrics_path)
    except Exception as e:
        print(f"  Failed to save metrics: {e}")
    
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
    
    # Comprehensive statistical analysis (if available)
    if STATS_AVAILABLE:
        print("\n📊 Running comprehensive statistical analysis for KAGNN+PGM...")
        final_stats = comprehensive_statistical_analysis(
            y_true, 
            y_pred,
            model_name="KAGNN + PGM",
            save_path=cfg.plots_dir / "kagnn_pgm_statistics.png"
        )
        print_statistical_summary(final_stats, "KAGNN+PGM STATISTICAL SUMMARY")
        
        # Compare KAGNN vs KAGNN+PGM statistically
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON: KAGNN vs KAGNN+PGM")
        print("="*80)
        comparison = compare_models_statistically(
            y_true, y_pred_kagnn, y_pred,
            model1_name="KAGNN Only", 
            model2_name="KAGNN + PGM"
        )
        
        # Save comparison results
        comparison_results = {
            'kagnn_only': kagnn_stats,
            'kagnn_pgm': final_stats,
            'comparison': {
                'improvement_pct': comparison['improvement_pct'],
                'paired_ttest_pvalue': comparison['paired_ttest']['p_value'],
                'wilcoxon_pvalue': comparison['wilcoxon']['p_value'],
                'cohens_d': comparison['cohens_d'],
                'significant': comparison['paired_ttest']['p_value'] < 0.05
            }
        }
        
        with open(cfg.plots_dir / 'statistical_comparison.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            json.dump(convert_to_serializable(comparison_results), f, indent=2)
        
        print(f"✓ Statistical comparison saved to {cfg.plots_dir / 'statistical_comparison.json'}")
    
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
    try:
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
    except Exception as e:
        print(f"  Visualization error: {e}")
    
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
    
    print(f"\n✓ All plots saved to {cfg.plots_dir}")
    
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
    
    if STATS_AVAILABLE and 'comparison' in locals():
        if comparison['paired_ttest']['p_value'] < 0.05:
            print(f"  ✓ Statistically SIGNIFICANT (p = {comparison['paired_ttest']['p_value']:.4f})")
        else:
            print(f"  ✗ NOT statistically significant (p = {comparison['paired_ttest']['p_value']:.4f})")
    
    print("="*80)
    
    # Save final summary
    summary = {
        'model': 'KAGNN→PGM Forward',
        'total_training_time_seconds': total_time,
        'stage1_time_seconds': stage1_time,
        'stage2_time_seconds': stage2_time,
        'kagnn_only_metrics': metrics_kagnn_only.metrics,
        'kagnn_pgm_metrics': metrics.metrics,
        'improvement_medae_percent': improvement_medae,
        'benchmark': benchmark
    }
    
    with open(cfg.results_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Experiment summary saved to {cfg.results_dir / 'experiment_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
