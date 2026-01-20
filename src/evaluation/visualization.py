"""
Visualization tools for retention time prediction analysis.

This module provides the RTVisualizer class for creating comprehensive
plots and visual analysis of model predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict
from pathlib import Path
from scipy import stats


class RTVisualizer:
    """
    Create comprehensive visualizations for RT prediction analysis.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ):
        """
        Initialize RTVisualizer.
        
        Args:
            y_true: True retention time values
            y_pred: Predicted retention time values
            model_name: Name of the model for plot titles
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.errors = y_pred - y_true
        self.abs_errors = np.abs(self.errors)
    
    def plot_predictions(
        self,
        save_path: Optional[Path] = None,
        benchmark: Optional[Dict[str, float]] = None
    ):
        """
        Create scatter plot of predictions vs true values.
        
        Args:
            save_path: Path to save the plot
            benchmark: Optional benchmark metrics for comparison
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with color based on error
        scatter = ax.scatter(
            self.y_true,
            self.y_pred,
            c=self.abs_errors,
            cmap='RdYlGn_r',
            alpha=0.6,
            s=20,
            vmin=0,
            vmax=np.percentile(self.abs_errors, 95)
        )
        
        # Perfect prediction line
        lims = [
            min(self.y_true.min(), self.y_pred.min()),
            max(self.y_true.max(), self.y_pred.max())
        ]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction', zorder=10)
        
        # Labels and title
        ax.set_xlabel('Experimental RT (s)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
        
        # Calculate metrics for title
        from scipy.stats import pearsonr
        from sklearn.metrics import r2_score
        pearson_r = pearsonr(self.y_true, self.y_pred)[0]
        r2 = r2_score(self.y_true, self.y_pred)
        
        title = f'{self.model_name} - Prediction Scatter\n'
        title += f'Pearson r = {pearson_r:.3f}, R² = {r2:.3f}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error (s)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction plot saved to {save_path}")
        
        plt.show()
    
    def plot_error_distribution(self, save_path: Optional[Path] = None):
        """
        Create histogram of prediction errors.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Signed errors
        ax1.hist(self.errors, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax1.axvline(
            np.median(self.errors),
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Median = {np.median(self.errors):.1f}s'
        )
        ax1.set_xlabel('Prediction Error (s)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax1.set_title(
            f'{self.model_name} - Error Distribution\nMean = {np.mean(self.errors):.1f}s, Std = {np.std(self.errors):.1f}s',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Absolute errors
        ax2.hist(self.abs_errors, bins=60, color='coral', edgecolor='black', alpha=0.7)
        ax2.axvline(
            np.median(self.abs_errors),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Median = {np.median(self.abs_errors):.1f}s'
        )
        ax2.axvline(
            np.mean(self.abs_errors),
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'Mean = {np.mean(self.abs_errors):.1f}s'
        )
        ax2.set_xlabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax2.set_title(
            f'{self.model_name} - Absolute Error Distribution',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Error distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, save_path: Optional[Path] = None):
        """
        Create residual analysis plots.
        
        Args:
            save_path: Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs predicted
        ax1.scatter(self.y_pred, self.errors, alpha=0.4, s=15, color='purple')
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
        ax1.axhline(
            np.mean(self.errors) + 1.96 * np.std(self.errors),
            color='orange',
            linestyle=':',
            linewidth=1.5,
            label='±1.96σ'
        )
        ax1.axhline(
            np.mean(self.errors) - 1.96 * np.std(self.errors),
            color='orange',
            linestyle=':',
            linewidth=1.5
        )
        ax1.set_xlabel('Predicted RT (s)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Residual (s)', fontsize=14, fontweight='bold')
        ax1.set_title(
            f'{self.model_name} - Residuals Plot',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        stats.probplot(self.errors, dist="norm", plot=ax2)
        ax2.set_title(
            f'{self.model_name} - Q-Q Plot (Normality Check)',
            fontsize=16,
            fontweight='bold',
            pad=15
        )
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Residual plot saved to {save_path}")
        
        plt.show()
    
    def plot_comprehensive(
        self,
        save_path: Optional[Path] = None,
        benchmark: Optional[Dict[str, float]] = None
    ):
        """
        Create comprehensive multi-panel visualization.
        
        Args:
            save_path: Path to save the plot
            benchmark: Optional benchmark metrics for comparison
        """
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import r2_score
        
        fig = plt.figure(figsize=(24, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle(
            f'{self.model_name} - Comprehensive RT Prediction Analysis',
            fontsize=26,
            fontweight='bold',
            y=0.995
        )
        
        # 1. Scatter plot with regression
        ax1 = fig.add_subplot(gs[0, 0])
        scatter = ax1.scatter(
            self.y_true,
            self.y_pred,
            c=self.abs_errors,
            cmap='RdYlGn_r',
            alpha=0.6,
            s=20,
            vmin=0,
            vmax=np.percentile(self.abs_errors, 95)
        )
        lims = [
            min(self.y_true.min(), self.y_pred.min()),
            max(self.y_true.max(), self.y_pred.max())
        ]
        ax1.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Experimental RT (s)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted RT (s)', fontsize=12, fontweight='bold')
        
        pearson_r = pearsonr(self.y_true, self.y_pred)[0]
        r2 = r2_score(self.y_true, self.y_pred)
        ax1.set_title(
            f'Prediction Scatter\nPearson={pearson_r:.3f}, R²={r2:.3f}',
            fontsize=14,
            fontweight='bold'
        )
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Abs Error (s)')
        
        # 2. Error distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(self.errors, bins=60, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=2)
        ax2.axvline(
            np.median(self.errors),
            color='orange',
            linestyle='--',
            linewidth=2,
            label=f'Median={np.median(self.errors):.1f}s'
        )
        ax2.set_xlabel('Prediction Error (s)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title(
            f'Error Distribution\nMean={np.mean(self.errors):.1f}s, Std={np.std(self.errors):.1f}s',
            fontsize=14,
            fontweight='bold'
        )
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Absolute error distribution
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(self.abs_errors, bins=60, color='coral', edgecolor='black', alpha=0.7)
        ax3.axvline(
            np.median(self.abs_errors),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Median={np.median(self.abs_errors):.1f}s'
        )
        ax3.axvline(
            np.mean(self.abs_errors),
            color='blue',
            linestyle='--',
            linewidth=2,
            label=f'Mean={np.mean(self.abs_errors):.1f}s'
        )
        ax3.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title(
            f'Absolute Error Distribution',
            fontsize=14,
            fontweight='bold'
        )
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Relative error distribution
        ax4 = fig.add_subplot(gs[0, 3])
        rel_errors = np.abs((self.y_true - self.y_pred) / np.where(self.y_true != 0, self.y_true, 1)) * 100
        rel_errors_clipped = np.clip(rel_errors, 0, 100)
        ax4.hist(rel_errors_clipped, bins=60, color='forestgreen', edgecolor='black', alpha=0.7)
        ax4.axvline(
            np.median(rel_errors),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Median={np.median(rel_errors):.1f}%'
        )
        ax4.set_xlabel('Relative Error (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. Residuals plot
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.scatter(self.y_pred, self.errors, alpha=0.4, s=12, color='purple')
        ax5.axhline(0, color='red', linestyle='--', linewidth=2)
        ax5.axhline(
            np.mean(self.errors) + 1.96 * np.std(self.errors),
            color='orange',
            linestyle=':',
            linewidth=1.5
        )
        ax5.axhline(
            np.mean(self.errors) - 1.96 * np.std(self.errors),
            color='orange',
            linestyle=':',
            linewidth=1.5
        )
        ax5.set_xlabel('Predicted RT (s)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Residual (s)', fontsize=12, fontweight='bold')
        ax5.set_title('Residuals Plot (±1.96σ bands)', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. CDF of absolute errors
        ax6 = fig.add_subplot(gs[1, 1])
        sorted_errs = np.sort(self.abs_errors)
        cdf = np.arange(1, len(sorted_errs) + 1) / len(sorted_errs)
        ax6.plot(sorted_errs, cdf, 'b-', linewidth=2, label='Current Model')
        ax6.axvline(10, color='green', linestyle='--', alpha=0.7, label='10s threshold')
        ax6.axvline(30, color='orange', linestyle='--', alpha=0.7, label='30s threshold')
        ax6.axvline(60, color='red', linestyle='--', alpha=0.7, label='60s threshold')
        ax6.set_xlabel('Absolute Error (s)', fontsize=12, fontweight='bold')
        ax6.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
        ax6.set_title('CDF of Absolute Errors', fontsize=14, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # 7. Q-Q plot
        ax7 = fig.add_subplot(gs[1, 2])
        stats.probplot(self.errors, dist="norm", plot=ax7)
        ax7.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # 8. Error by RT range
        ax8 = fig.add_subplot(gs[1, 3])
        rt_bins = np.percentile(self.y_true, [0, 25, 50, 75, 100])
        bin_labels = ['Q1', 'Q2', 'Q3', 'Q4']
        binned_errors = []
        for i in range(len(rt_bins) - 1):
            mask = (self.y_true >= rt_bins[i]) & (self.y_true < rt_bins[i + 1])
            binned_errors.append(self.abs_errors[mask])
        ax8.boxplot(binned_errors, labels=bin_labels)
        ax8.set_xlabel('RT Quartile', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Absolute Error (s)', fontsize=12, fontweight='bold')
        ax8.set_title('Error Distribution by RT Range', fontsize=14, fontweight='bold')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9 & 10. Metrics tables (if benchmark provided)
        if benchmark:
            self._add_metrics_tables(fig, gs, benchmark)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comprehensive plot saved to {save_path}")
        
        plt.show()
    
    def _add_metrics_tables(self, fig, gs, benchmark):
        """Helper method to add metrics comparison tables."""
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import r2_score
        
        # Calculate metrics
        medae = np.median(self.abs_errors)
        mae = np.mean(self.abs_errors)
        rmse = np.sqrt(np.mean(self.errors**2))
        r2 = r2_score(self.y_true, self.y_pred)
        pearson_r = pearsonr(self.y_true, self.y_pred)[0]
        spearman_r = spearmanr(self.y_true, self.y_pred)[0]
        
        # Performance metrics table
        ax9 = fig.add_subplot(gs[2, :2])
        ax9.axis('off')
        table_data = [
            ['Metric', 'Current Model', 'Benchmark', 'Δ'],
            ['MedAE (s)', f"{medae:.2f}", f"{benchmark.get('MedAE', 0):.2f}",
             f"{medae - benchmark.get('MedAE', 0):+.2f}"],
            ['MAE (s)', f"{mae:.2f}", f"{benchmark.get('MAE', 0):.2f}",
             f"{mae - benchmark.get('MAE', 0):+.2f}"],
            ['RMSE (s)', f"{rmse:.2f}", f"{benchmark.get('RMSE', 0):.2f}",
             f"{rmse - benchmark.get('RMSE', 0):+.2f}"],
            ['R²', f"{r2:.3f}", f"{benchmark.get('R2', 0):.3f}",
             f"{r2 - benchmark.get('R2', 0):+.3f}"],
            ['Pearson', f"{pearson_r:.3f}", f"{benchmark.get('Pearson', 0):.3f}",
             f"{pearson_r - benchmark.get('Pearson', 0):+.3f}"],
            ['Spearman', f"{spearman_r:.3f}", f"{benchmark.get('Spearman', 0):.3f}",
             f"{spearman_r - benchmark.get('Spearman', 0):+.3f}"],
        ]
        table = ax9.table(
            cellText=table_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)
        for i in range(len(table_data)):
            if i == 0:
                for j in range(4):
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
        ax9.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        
        # Threshold percentages table
        ax10 = fig.add_subplot(gs[2, 2:])
        ax10.axis('off')
        pct_le_10 = (self.abs_errors <= 10).mean() * 100
        pct_le_30 = (self.abs_errors <= 30).mean() * 100
        pct_le_60 = (self.abs_errors <= 60).mean() * 100
        
        threshold_data = [
            ['Threshold', 'Current Model', 'Benchmark', 'Δ'],
            ['% ≤ 10s', f"{pct_le_10:.2f}%", f"{benchmark.get('Pct_le_10s', 0):.2f}%",
             f"{pct_le_10 - benchmark.get('Pct_le_10s', 0):+.2f}%"],
            ['% ≤ 30s', f"{pct_le_30:.2f}%", f"{benchmark.get('Pct_le_30s', 0):.2f}%",
             f"{pct_le_30 - benchmark.get('Pct_le_30s', 0):+.2f}%"],
            ['% ≤ 60s', f"{pct_le_60:.2f}%", f"{benchmark.get('Pct_le_60s', 0):.2f}%",
             f"{pct_le_60 - benchmark.get('Pct_le_60s', 0):+.2f}%"],
        ]
        table2 = ax10.table(
            cellText=threshold_data,
            loc='center',
            cellLoc='center',
            colWidths=[0.25, 0.25, 0.25, 0.25]
        )
        table2.auto_set_font_size(False)
        table2.set_fontsize(11)
        table2.scale(1, 2.5)
        for i in range(len(threshold_data)):
            if i == 0:
                for j in range(4):
                    table2[(i, j)].set_facecolor('#40466e')
                    table2[(i, j)].set_text_props(weight='bold', color='white')
        ax10.set_title('Prediction Accuracy by Threshold', fontsize=14, fontweight='bold', pad=20)
