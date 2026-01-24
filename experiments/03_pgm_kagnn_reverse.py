"""
PGM→KAGNN Reverse Residual Experiment.

This script trains the PGM→KAGNN model in two stages:
1. Train PGM ensemble (XGBoost + Bayesian Ridge)
2. Train KAGNN to correct residuals

Key improvements:
1. Full integration with ComprehensiveStatisticalAnalyzer
2. Better error handling and validation
3. Memory-efficient descriptor caching
4. Comprehensive statistical comparison
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
from src.models.pgm_kagnn_reverse import PGM_KAGNN_Reverse
from src.evaluation.metrics import RTMetrics
from src.evaluation.visualization import RTVisualizer
from src.utils.seed import set_seed

# Import comprehensive statistical analysis
try:
    from src.evaluation.statistical_analysis import (
        ComprehensiveStatisticalAnalyzer,
        run_comprehensive_analysis
    )
    STATS_AVAILABLE = True
    print("✓ Comprehensive statistical analysis available")
except ImportError:
    print("⚠️  Advanced statistical analysis not available")
    print("   Install the statistical_analysis module for full analysis")
    STATS_AVAILABLE = False


class Config:
    # Data paths
    ecfp_file = "/kaggle/input/smrtdatas/SMRT_ECFP_1024_Fingerprints.txt"
    rt_file = "/kaggle/input/smrtdt/SMRT_dataset.csv"
    sdf_file = "/kaggle/input/smrtdatas/SMRT_dataset.sdf"
    
    # Training hyperparameters
    batch_size = 64
    epochs = 100  # For KAGNN refinement
    lr = 5e-4  # Lower LR for refinement
    weight_decay = 1e-4
    
    # Model architecture
    hidden_dim = 256
    dropout = 0.2
    
    # Stage 1 (PGM) hyperparameters
    pgm_estimators = 400
    pgm_optimize = False  # Set to True for hyperparameter optimization
    
    # Data split
    val_size = 0.15
    test_size = 0.15
    seed = 42
    
    # Output paths
    results_dir = Path("results")
    checkpoint_path = results_dir / "checkpoints" / "pgm_kagnn_reverse.pt"
    metrics_path = results_dir / "metrics" / "pgm_kagnn_reverse.json"
    plots_dir = results_dir / "plots" / "pgm_kagnn_reverse"
    stats_dir = results_dir / "statistical_analysis" / "pgm_kagnn_reverse"


def validate_data_quality(loader, name="Loader", max_batches=5):
    """
    Validate data loader for quality issues
    
    Args:
        loader: DataLoader to validate
        name: Name for logging
        max_batches: Number of batches to check
    """
    print(f"\n🔍 Validating {name} data quality...")
    
    issues = []
    
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        
        graph, ecfp, rt, cids = batch
        
        # Check ECFP
        if torch.isnan(ecfp).any():
            issues.append(f"NaN in ECFP at batch {i}")
        if torch.isinf(ecfp).any():
            issues.append(f"Inf in ECFP at batch {i}")
        
        # Check RT
        if torch.isnan(rt).any():
            issues.append(f"NaN in RT at batch {i}")
        if torch.isinf(rt).any():
            issues.append(f"Inf in RT at batch {i}")
        
        # Check graph features
        if torch.isnan(graph.x).any():
            issues.append(f"NaN in graph features at batch {i}")
        
        # Check RT range
        if rt.min() < 0 or rt.max() > 100:  # Normalized RT should be ~[-3, 3]
            issues.append(f"Unusual RT values at batch {i}: [{rt.min():.2f}, {rt.max():.2f}]")
    
    if issues:
        print(f"    Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"     - {issue}")
    else:
        print(f"  ✓ {name} validation passed")
    
    return len(issues) == 0


def main():
    cfg = Config()
    
    # Set random seed
    set_seed(cfg.seed)
    
    # Create output directories
    cfg.results_dir.mkdir(exist_ok=True)
    cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    if STATS_AVAILABLE:
        cfg.stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
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
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, 
                              collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, 
                            collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, cfg.batch_size, shuffle=False, 
                             collate_fn=collate_fn, num_workers=0)
    
    # Validate data quality
    validate_data_quality(train_loader, "Train Loader")
    validate_data_quality(test_loader, "Test Loader")
    
    # Initialize model
    print("\n" + "="*80)
    print("INITIALIZING MODEL")
    print("="*80)
    
    try:
        model = PGM_KAGNN_Reverse(cfg).to(device)
        model.set_normalization_params(train_ds.rt_mean, train_ds.rt_std)
        model.set_mol_dict(train_ds.mol_dict)
    except Exception as e:
        print(f" Failed to initialize model: {e}")
        raise
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # STAGE 1: Train PGM
    print("\n" + "="*80)
    print("STAGE 1: TRAINING PGM ENSEMBLE")
    print("="*80)
    
    start_time = time.time()
    
    try:
        model.train_stage1_pgm(
            train_loader=train_loader,
            val_loader=val_loader,
            n_estimators=cfg.pgm_estimators,
            optimize=cfg.pgm_optimize,
            verbose=True
        )
    except Exception as e:
        print(f" Stage 1 training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    stage1_time = time.time() - start_time
    print(f"\n✓ Stage 1 completed in {stage1_time:.2f} seconds ({stage1_time/60:.2f} minutes)")
    
    # Evaluate PGM only (before KAGNN correction)
    print("\n" + "="*80)
    print("EVALUATING PGM ONLY (BEFORE KAGNN CORRECTION)")
    print("="*80)
    
    # Get PGM-only predictions (more efficient version)
    model.eval()
    y_true_pgm_list = []
    y_pred_pgm_list = []
    
    print("Extracting PGM predictions...")
    with torch.no_grad():
        for batch in test_loader:
            graph, ecfp, rt_norm, cids = batch
            
            # Use model's internal method for PGM prediction
            # This is more efficient than re-extracting descriptors
            try:
                from src.models.pgm_kagnn_reverse import ComprehensiveDescriptors
                descriptors = ComprehensiveDescriptors.extract_batch(
                    model.mol_dict, cids, use_cache=True  # Enable caching
                )
                features = np.concatenate([ecfp.numpy(), descriptors], axis=1)
                X_scaled = model.pgm_scaler.transform(features)
                
                pgm_pred = (model.pgm_xgb.predict(X_scaled) + 
                           model.pgm_br.predict(X_scaled)) / 2
                
                y_true_pgm_list.append(rt_norm.numpy())
                y_pred_pgm_list.append(pgm_pred)
            except Exception as e:
                print(f"  Error in PGM prediction: {e}")
                continue
    
    y_true_pgm = np.concatenate(y_true_pgm_list)
    y_pred_pgm = np.concatenate(y_pred_pgm_list)
    
    # Denormalize
    y_true_pgm_denorm = y_true_pgm * train_ds.rt_std + train_ds.rt_mean
    y_pred_pgm_denorm = y_pred_pgm * train_ds.rt_std + train_ds.rt_mean
    
    # Basic metrics for PGM
    metrics_pgm_only = RTMetrics(
        y_true=y_true_pgm_denorm,
        y_pred=y_pred_pgm_denorm,
        model_name="PGM Only (Stage 1)"
    )
    metrics_pgm_only.print_summary()
    
    # Comprehensive statistical analysis for PGM (if available)
    if STATS_AVAILABLE:
        print("\n Running comprehensive statistical analysis for PGM...")
        analyzer_pgm = ComprehensiveStatisticalAnalyzer(model_name="PGM Only")
        metrics_pgm_comprehensive = analyzer_pgm.compute_all_metrics(
            y_true_pgm_denorm, y_pred_pgm_denorm
        )
        
        # Create diagnostic plot
        analyzer_pgm.create_comprehensive_diagnostic_plot(
            y_true_pgm_denorm, y_pred_pgm_denorm,
            metrics_pgm_comprehensive,
            save_path=cfg.stats_dir / "pgm_only_diagnostics.png"
        )
        
        # Print summary
        analyzer_pgm.print_summary_report()
    
    # STAGE 2: Train KAGNN Refinement
    print("\n" + "="*80)
    print("STAGE 2: TRAINING KAGNN REFINEMENT NETWORK")
    print("="*80)
    
    start_time = time.time()
    
    # Setup optimizer for KAGNN only
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.SmoothL1Loss()
    
    # Train Stage 2
    try:
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=cfg.epochs,
            optimizer=optimizer,
            criterion=criterion,
            early_stopping_patience=15,
            checkpoint_path=cfg.checkpoint_path
        )
    except Exception as e:
        print(f" Stage 2 training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    stage2_time = time.time() - start_time
    print(f"\n✓ Stage 2 completed in {stage2_time:.2f} seconds ({stage2_time/60:.2f} minutes)")
    
    # Load best model
    try:
        model.load(cfg.checkpoint_path, map_location=str(device))
    except Exception as e:
        print(f"  Failed to load best model: {e}")
        print("   Continuing with current model state...")
    
    # Final evaluation with KAGNN correction
    print("\n" + "="*80)
    print("FINAL EVALUATION (PGM + KAGNN)")
    print("="*80)
    
    try:
        y_true, y_pred = model.predict(test_loader, denormalize=True)
    except Exception as e:
        print(f" Final prediction failed: {e}")
        raise
    
    # Create baseline predictions (PGM only)
    y_pred_baseline = y_pred_pgm_denorm
    
    # Compute basic metrics
    print("\nComputing metrics...")
    metrics = RTMetrics(
        y_true=y_true,
        y_pred=y_pred,
        model_name="PGM→KAGNN Reverse",
        y_pred_baseline=y_pred_baseline
    )
    
    # Save metrics
    try:
        metrics.save(cfg.metrics_path)
    except Exception as e:
        print(f"⚠️  Failed to save metrics: {e}")
    
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
    
    # COMPREHENSIVE STATISTICAL ANALYSIS
    if STATS_AVAILABLE:
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*80)
        
        # Analyze PGM+KAGNN
        print("\n📊 Analyzing PGM+KAGNN model...")
        analyzer_final = ComprehensiveStatisticalAnalyzer(model_name="PGM + KAGNN")
        metrics_final_comprehensive = analyzer_final.compute_all_metrics(y_true, y_pred)
        
        # Create diagnostic plot
        analyzer_final.create_comprehensive_diagnostic_plot(
            y_true, y_pred,
            metrics_final_comprehensive,
            save_path=cfg.stats_dir / "pgm_kagnn_diagnostics.png"
        )
        
        # Print summary
        analyzer_final.print_summary_report()
        
        # STATISTICAL COMPARISON: PGM vs PGM+KAGNN
        print("\n" + "="*80)
        print("STATISTICAL COMPARISON: PGM vs PGM+KAGNN")
        print("="*80)
        
        comparison_results = analyzer_final.compare_models(
            y_true, y_pred_pgm_denorm, y_pred,
            model1_name="PGM Only",
            model2_name="PGM + KAGNN"
        )
        
        # Create comparison plot
        analyzer_final.create_comparison_plot(
            y_true, y_pred_pgm_denorm, y_pred,
            model1_name="PGM Only",
            model2_name="PGM + KAGNN",
            save_path=cfg.stats_dir / "pgm_vs_pgm_kagnn_comparison.png"
        )
        
        # Export all statistical results
        analyzer_final.export_results(str(cfg.stats_dir))
        
        # Save comparison summary
        comparison_summary = {
            'improvement_pct': comparison_results['mean_improvement_pct'],
            'significant': comparison_results['paired_ttest']['significant'],
            'ttest_pvalue': comparison_results['paired_ttest']['p_value'],
            'wilcoxon_pvalue': comparison_results['wilcoxon']['p_value'],
            'cohens_d': comparison_results['effect_sizes']['cohens_d']
        }
        
        with open(cfg.stats_dir / 'comparison_summary.json', 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        print(f"\n✓ Statistical analysis complete")
        print(f"  Improvement: {comparison_summary['improvement_pct']:.2f}%")
        print(f"  Significant: {'✓ YES' if comparison_summary['significant'] else '✗ NO'}")
        print(f"  p-value: {comparison_summary['ttest_pvalue']:.4f}")
    
    # Generate standard visualizations
    print("\n" + "="*80)
    print("GENERATING STANDARD VISUALIZATIONS")
    print("="*80)
    
    visualizer = RTVisualizer(
        y_true=y_true,
        y_pred=y_pred,
        model_name="PGM→KAGNN Reverse"
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
    plt.title('PGM→KAGNN Training History (Stage 2)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Comparison plot: PGM vs PGM+KAGNN
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # PGM Only
    axes[0].scatter(y_true_pgm_denorm, y_pred_pgm_denorm, alpha=0.5, s=20, c='royalblue')
    lims = [min(y_true_pgm_denorm.min(), y_pred_pgm_denorm.min()), 
            max(y_true_pgm_denorm.max(), y_pred_pgm_denorm.max())]
    axes[0].plot(lims, lims, 'r--', lw=2)
    axes[0].set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
    axes[0].set_title(f'PGM Only\nMedAE={metrics_pgm_only.metrics["MedAE"]:.2f}s, '
                     f'R²={metrics_pgm_only.metrics["R2"]:.3f}', 
                     fontsize=16, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # PGM + KAGNN
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=20, c='mediumseagreen')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[1].plot(lims, lims, 'r--', lw=2)
    axes[1].set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
    axes[1].set_title(f'PGM + KAGNN\nMedAE={metrics.metrics["MedAE"]:.2f}s, '
                     f'R²={metrics.metrics["R2"]:.3f}',
                     fontsize=16, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(cfg.plots_dir / "pgm_vs_kagnn_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ All plots saved to {cfg.plots_dir}")
    
    # Runtime summary
    total_time = stage1_time + stage2_time
    print("\n" + "="*80)
    print("RUNTIME SUMMARY")
    print("="*80)
    print(f"Stage 1 (PGM):         {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"Stage 2 (KAGNN):       {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"Total Training Time:   {total_time:.2f}s ({total_time/60:.2f} min)")
    print("="*80)
    
    # Final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Model: PGM→KAGNN Reverse Residual")
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Test samples: {len(y_true)}")
    print(f"Checkpoint saved: {cfg.checkpoint_path}")
    print(f"Metrics saved: {cfg.metrics_path}")
    print(f"Plots saved: {cfg.plots_dir}")
    if STATS_AVAILABLE:
        print(f"Statistical analysis saved: {cfg.stats_dir}")
    
    print("\nKey Results:")
    print(f"  PGM Only:")
    print(f"    MedAE: {metrics_pgm_only.metrics['MedAE']:.2f}s")
    print(f"    MAE: {metrics_pgm_only.metrics['MAE']:.2f}s")
    print(f"    R²: {metrics_pgm_only.metrics['R2']:.3f}")
    
    print(f"\n  PGM + KAGNN:")
    print(f"    MedAE: {metrics.metrics['MedAE']:.2f}s")
    print(f"    MAE: {metrics.metrics['MAE']:.2f}s")
    print(f"    R²: {metrics.metrics['R2']:.3f}")
    print(f"    % ≤ 30s: {metrics.metrics['Pct_le_30s']:.2f}%")
    
    improvement_medae = ((metrics_pgm_only.metrics['MedAE'] - metrics.metrics['MedAE']) / 
                         metrics_pgm_only.metrics['MedAE'] * 100)
    print(f"\n  Improvement: {improvement_medae:.2f}% reduction in MedAE")
    
    if STATS_AVAILABLE and 'comparison_summary' in locals():
        if comparison_summary['significant']:
            print(f"  ✓ Statistically SIGNIFICANT (p = {comparison_summary['ttest_pvalue']:.4f})")
        else:
            print(f"  ✗ NOT statistically significant (p = {comparison_summary['ttest_pvalue']:.4f})")
    
    print("="*80)
    
    # Save final summary JSON
    summary = {
        'model': 'PGM→KAGNN Reverse',
        'total_training_time_seconds': total_time,
        'stage1_time_seconds': stage1_time,
        'stage2_time_seconds': stage2_time,
        'pgm_only_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                            for k, v in metrics_pgm_only.metrics.items()},
        'pgm_kagnn_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in metrics.metrics.items()},
        'improvement_medae_percent': float(improvement_medae),
        'benchmark': benchmark
    }
    
    if STATS_AVAILABLE and 'comparison_summary' in locals():
        summary['statistical_comparison'] = comparison_summary
    
    with open(cfg.results_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Experiment summary saved to {cfg.results_dir / 'experiment_summary.json'}")
    print("\n Experiment completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n EXPERIMENT FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
