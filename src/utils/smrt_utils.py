"""
SMRT Utilities for Retention Time Prediction Metrics and Visualization.

This module provides comprehensive metrics computation and visualization
functions for RT prediction models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
from typing import Dict, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
    normalize: bool = False,
    rt_scaler: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute comprehensive RT prediction metrics.

    Metrics:
        - n_samples: Number of samples
        - MedAE: Median Absolute Error
        - MAE: Mean Absolute Error
        - RMSE: Root Mean Squared Error
        - R2: Coefficient of Determination
        - Pearson: Pearson correlation coefficient
        - Spearman: Spearman correlation coefficient
        - Pct_le_10s: % predictions within 10 seconds
        - Pct_le_30s: % predictions within 30 seconds
        - Pct_le_60s: % predictions within 60 seconds

    Args:
        y_true: Ground truth retention times
        y_pred: Predicted retention times
        name: Optional model name
        normalize: Whether values are normalized
        rt_scaler: Scaler object to denormalize if needed

    Returns:
        Dictionary with metric names and values
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    # Denormalize if needed
    if normalize and rt_scaler is not None:
        y_true = rt_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = rt_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    abs_errors = np.abs(y_true - y_pred)

    metrics = {}
    metrics['n_samples'] = int(len(y_true))
    metrics['MedAE'] = float(np.median(abs_errors))
    metrics['MAE'] = float(mean_absolute_error(y_true, y_pred))
    metrics['RMSE'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['R2'] = float(r2_score(y_true, y_pred))

    # Correlation metrics
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics['Pearson'] = float(pearsonr(y_true, y_pred)[0])
        metrics['Spearman'] = float(spearmanr(y_true, y_pred)[0])
    else:
        metrics['Pearson'] = float('nan')
        metrics['Spearman'] = float('nan')

    # Threshold accuracy metrics
    metrics['Pct_le_10s'] = float(np.mean(abs_errors <= 10) * 100)
    metrics['Pct_le_30s'] = float(np.mean(abs_errors <= 30) * 100)
    metrics['Pct_le_60s'] = float(np.mean(abs_errors <= 60) * 100)

    # Add model name
    metrics['name'] = name

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary from compute_metrics
        title: Optional title
    """
    # Define order for consistent display
    order = [
        'n_samples', 'MedAE', 'MAE', 'RMSE', 'R2',
        'Pearson', 'Spearman',
        'Pct_le_10s', 'Pct_le_30s', 'Pct_le_60s'
    ]

    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)
    print(f"{'Metric':<20} {'Value':>15}")
    print("-" * 80)

    for key in order:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, float):
                if "Pct" in key:
                    print(f"{key:<20} {value:>14.2f}%")
                else:
                    print(f"{key:<20} {value:>15.4f}")
            else:
                print(f"{key:<20} {value:>15}")

    # Print any extra metrics not in order
    for key, value in metrics.items():
        if key not in order and key != 'name':
            print(f"{key:<20} {value:>15}")

    print("=" * 80)


def plot_single_model_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create comprehensive visualization for a single model.

    Args:
        y_true: Ground truth retention times
        y_pred: Predicted retention times
        metrics: Dictionary of computed metrics
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{metrics.get('name', 'Model')} - Comprehensive Analysis", 
                 fontsize=16, fontweight='bold')

    # 1. Scatter plot
    ax1 = axes[0, 0]
    scatter = ax1.scatter(y_true, y_pred, c=abs_errors, cmap='RdYlGn_r', 
                         alpha=0.6, s=20, vmin=0, vmax=np.percentile(abs_errors, 95))
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax1.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True RT (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted RT (s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Prediction Scatter\nR²={metrics["R2"]:.3f}, Pearson={metrics["Pearson"]:.3f}', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Abs Error (s)')

    # 2. Error distribution
    ax2 = axes[0, 1]
    ax2.hist(errors, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    ax2.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median={np.median(errors):.1f}s')
    ax2.set_xlabel('Prediction Error (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title(f'Error Distribution\nMean={np.mean(errors):.1f}s, Std={np.std(errors):.1f}s', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 3. Absolute error distribution
    ax3 = axes[0, 2]
    ax3.hist(abs_errors, bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(metrics['MedAE'], color='red', linestyle='--', linewidth=2,
               label=f'MedAE={metrics["MedAE"]:.1f}s')
    ax3.axvline(metrics['MAE'], color='blue', linestyle='--', linewidth=2,
               label=f'MAE={metrics["MAE"]:.1f}s')
    ax3.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Absolute Error Distribution', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Residuals plot
    ax4 = axes[1, 0]
    ax4.scatter(y_pred, errors, alpha=0.4, s=15, color='purple')
    ax4.axhline(0, color='red', linestyle='--', linewidth=2)
    ax4.axhline(np.mean(errors) + 1.96*np.std(errors), color='orange', 
               linestyle=':', linewidth=1.5, label='±1.96σ')
    ax4.axhline(np.mean(errors) - 1.96*np.std(errors), color='orange', linestyle=':', linewidth=1.5)
    ax4.set_xlabel('Predicted RT (s)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Residual (s)', fontsize=12, fontweight='bold')
    ax4.set_title('Residuals Plot', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. CDF of absolute errors with thresholds
    ax5 = axes[1, 1]
    sorted_errs = np.sort(abs_errors)
    cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
    ax5.plot(sorted_errs, cdf, 'b-', linewidth=2, label='CDF')
    ax5.axvline(10, color='green', linestyle='--', alpha=0.7, 
               label=f'10s: {metrics["Pct_le_10s"]:.1f}%')
    ax5.axvline(30, color='orange', linestyle='--', alpha=0.7,
               label=f'30s: {metrics["Pct_le_30s"]:.1f}%')
    ax5.axvline(60, color='red', linestyle='--', alpha=0.7,
               label=f'60s: {metrics["Pct_le_60s"]:.1f}%')
    ax5.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax5.set_title('CDF of Absolute Errors', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9, loc='lower right')
    ax5.grid(True, alpha=0.3)

    # 6. Metrics table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = [
        ['Metric', 'Value'],
        ['MedAE (s)', f'{metrics["MedAE"]:.2f}'],
        ['MAE (s)', f'{metrics["MAE"]:.2f}'],
        ['RMSE (s)', f'{metrics["RMSE"]:.2f}'],
        ['R²', f'{metrics["R2"]:.3f}'],
        ['Pearson', f'{metrics["Pearson"]:.3f}'],
        ['Spearman', f'{metrics["Spearman"]:.3f}'],
        ['% ≤ 10s', f'{metrics["Pct_le_10s"]:.2f}%'],
        ['% ≤ 30s', f'{metrics["Pct_le_30s"]:.2f}%'],
        ['% ≤ 60s', f'{metrics["Pct_le_60s"]:.2f}%'],
        ['n_samples', f'{metrics["n_samples"]}'],
    ]
    
    table = ax6.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.4, 0.4]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    for i in range(len(table_data)):
        if i == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Metrics Summary', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_comprehensive_comparison(
    metrics_list: list,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Create comparison visualization for multiple models.

    Args:
        metrics_list: List of metric dictionaries
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')

    model_names = [m.get('name', f'Model {i+1}') for i, m in enumerate(metrics_list)]

    # 1. Bar plot - Error metrics
    ax1 = axes[0, 0]
    x = np.arange(len(model_names))
    width = 0.25
    
    medae = [m['MedAE'] for m in metrics_list]
    mae = [m['MAE'] for m in metrics_list]
    rmse = [m['RMSE'] for m in metrics_list]
    
    bars1 = ax1.bar(x - width, medae, width, label='MedAE', color='steelblue')
    bars2 = ax1.bar(x, mae, width, label='MAE', color='coral')
    bars3 = ax1.bar(x + width, rmse, width, label='RMSE', color='forestgreen')
    
    ax1.set_ylabel('Error (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Error Metrics Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # 2. Bar plot - Correlation metrics
    ax2 = axes[0, 1]
    pearson = [m['Pearson'] for m in metrics_list]
    spearman = [m['Spearman'] for m in metrics_list]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, pearson, width, label='Pearson', color='royalblue')
    bars2 = ax2.bar(x + width/2, spearman, width, label='Spearman', color='mediumseagreen')
    
    ax2.set_ylabel('Correlation', fontsize=12, fontweight='bold')
    ax2.set_title('Correlation Metrics Comparison', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # 3. Bar plot - Threshold accuracy
    ax3 = axes[1, 0]
    pct_10 = [m['Pct_le_10s'] for m in metrics_list]
    pct_30 = [m['Pct_le_30s'] for m in metrics_list]
    pct_60 = [m['Pct_le_60s'] for m in metrics_list]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax3.bar(x - width, pct_10, width, label='% ≤ 10s', color='#2ecc71')
    bars2 = ax3.bar(x, pct_30, width, label='% ≤ 30s', color='#f39c12')
    bars3 = ax3.bar(x + width, pct_60, width, label='% ≤ 60s', color='#e74c3c')
    
    ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Threshold Accuracy Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=45, ha='right')
    ax3.legend(fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # 4. Combined metrics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    table_data = [['Model', 'MedAE', 'MAE', 'R²', '%≤10s', '%≤30s', '%≤60s']]
    for m in metrics_list:
        table_data.append([
            m.get('name', 'Model')[:15],
            f'{m["MedAE"]:.2f}',
            f'{m["MAE"]:.2f}',
            f'{m["R2"]:.3f}',
            f'{m["Pct_le_10s"]:.1f}%',
            f'{m["Pct_le_30s"]:.1f}%',
            f'{m["Pct_le_60s"]:.1f}%'
        ])
    
    table = ax4.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(len(table_data)):
        if i == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Metrics Summary Table', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig


def plot_error_by_rt_range(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Dict[str, float],
    n_bins: int = 4,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> plt.Figure:
    """
    Plot error distribution by RT range (quartiles).

    Args:
        y_true: Ground truth retention times
        y_pred: Predicted retention times
        metrics: Dictionary of computed metrics
        n_bins: Number of RT bins (default: 4 for quartiles)
        save_path: Optional path to save figure
        show_plot: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    abs_errors = np.abs(y_true - y_pred)

    # Create bins based on true RT values
    bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_labels = [f'Q{i+1}\n({bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}s)' 
                  for i in range(n_bins)]

    # Calculate statistics for each bin
    binned_data = []
    for i in range(n_bins):
        mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
        errors = abs_errors[mask]
        binned_data.append({
            'label': bin_labels[i],
            'errors': errors,
            'count': int(np.sum(mask)),
            'median': np.median(errors) if len(errors) > 0 else 0,
            'mean': np.mean(errors) if len(errors) > 0 else 0,
            'std': np.std(errors) if len(errors) > 0 else 0
        })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"{metrics.get('name', 'Model')} - Error by RT Range", 
                 fontsize=16, fontweight='bold')

    # 1. Box plot
    ax1 = axes[0]
    error_lists = [d['errors'] for d in binned_data]
    bp = ax1.boxplot(error_lists, labels=[d['label'] for d in binned_data], patch_artist=True)
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_bins))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_xlabel('RT Range', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Absolute Error (s)', fontsize=12, fontweight='bold')
    ax1.set_title('Error Distribution by RT Range', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Bar plot with error bars
    ax2 = axes[1]
    x = np.arange(n_bins)
    width = 0.6
    means = [d['mean'] for d in binned_data]
    stds = [d['std'] for d in binned_data]
    
    bars = ax2.bar(x, means, width, yerr=stds, capsize=5, 
                   color=plt.cm.viridis(np.linspace(0.2, 0.8, n_bins)), 
                   alpha=0.7, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels([d['label'] for d in binned_data])
    ax2.set_xlabel('RT Range', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Absolute Error (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Mean Error by RT Range (with Std)', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, mean_val in zip(bars, means):
        ax2.annotate(f'{mean_val:.1f}s',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # 3. Statistics table
    ax3 = axes[2]
    ax3.axis('off')
    
    table_data = [['Range', 'n', 'Median', 'Mean', 'Std', '%≤30s']]
    for d in binned_data:
        pct_le_30 = np.mean(d['errors'] <= 30) * 100 if len(d['errors']) > 0 else 0
        table_data.append([
            d['label'].replace('\n', ' '),
            str(d['count']),
            f'{d["median"]:.2f}',
            f'{d["mean"]:.2f}',
            f'{d["std"]:.2f}',
            f'{pct_le_30:.1f}%'
        ])
    
    table = ax3.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.25, 0.1, 0.15, 0.15, 0.15, 0.15]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(len(table_data)):
        if i == 0:
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor('#40466e')
                table[(i, j)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Error Statistics by RT Range', fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()

    return fig

