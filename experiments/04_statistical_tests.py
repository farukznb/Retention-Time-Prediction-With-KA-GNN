"""
Statistical significance tests for comparing RT prediction models.

This script loads saved metrics from all experiments and performs
pairwise statistical comparisons using Wilcoxon signed-rank test
and paired t-tests.
"""

"""
COMPLETE STATISTICAL ANALYSIS MODULE
Includes ALL statistical tests and diagnostic plots
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import (
    pearsonr, spearmanr, shapiro, normaltest, 
    ttest_rel, wilcoxon, linregress, probplot,
    ttest_ind, f_oneway, kruskal, mannwhitneyu,
    skew, kurtosis, bartlett, levene
)
from scipy import stats
from sklearn.metrics import (
    r2_score, mean_absolute_error, median_absolute_error,
    mean_squared_error, mean_absolute_percentage_error
)
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


class ComprehensiveStatisticalAnalyzer:
    """
    Complete statistical analysis for RT prediction models
    Includes ALL tests and visualizations mentioned in the paper
    """
    
    def __init__(self, model_name: str = "Model", alpha: float = 0.05):
        """
        Initialize analyzer
        
        Args:
            model_name: Name of the model for reporting
            alpha: Significance level for statistical tests
        """
        self.model_name = model_name
        self.alpha = alpha
        self.results = {}
        
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Compute ALL performance metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing all metrics
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        n = len(y_true)
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        relative_errors = np.abs(residuals) / (np.abs(y_true) + 1e-10) * 100
        
        # 1. Core regression metrics
        metrics = {
            'n_samples': n,
            
            # Absolute error metrics
            'MedAE': median_absolute_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MSE': mean_squared_error(y_true, y_pred),
            
            # Relative error metrics
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'MedAPE': np.median(relative_errors),
            
            # Correlation metrics
            'R2': r2_score(y_true, y_pred),
            'Pearson_r': pearsonr(y_true, y_pred)[0],
            'Pearson_p': pearsonr(y_true, y_pred)[1],
            'Spearman_rho': spearmanr(y_true, y_pred)[0],
            'Spearman_p': spearmanr(y_true, y_pred)[1],
            
            # Error distribution statistics
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_median': np.median(residuals),
            'residual_mad': np.median(np.abs(residuals - np.median(residuals))),
            'residual_skew': skew(residuals),
            'residual_kurtosis': kurtosis(residuals),
            
            # Absolute error statistics
            'abs_error_mean': np.mean(abs_errors),
            'abs_error_std': np.std(abs_errors),
            'abs_error_median': np.median(abs_errors),
            'abs_error_q90': np.percentile(abs_errors, 90),
            'abs_error_q95': np.percentile(abs_errors, 95),
            'abs_error_q99': np.percentile(abs_errors, 99),
            'abs_error_max': np.max(abs_errors),
            
            # Relative error statistics
            'rel_error_mean': np.mean(relative_errors),
            'rel_error_median': np.median(relative_errors),
            'rel_error_std': np.std(relative_errors),
            'rel_error_q90': np.percentile(relative_errors, 90),
            'rel_error_q95': np.percentile(relative_errors, 95),
            
            # Data statistics
            'y_true_mean': np.mean(y_true),
            'y_true_std': np.std(y_true),
            'y_true_min': np.min(y_true),
            'y_true_max': np.max(y_true),
            'y_pred_mean': np.mean(y_pred),
            'y_pred_std': np.std(y_pred),
            'y_pred_min': np.min(y_pred),
            'y_pred_max': np.max(y_pred),
        }
        
        # 2. Threshold performance
        thresholds = [1, 5, 10, 15, 20, 30, 45, 60, 90, 120]
        for threshold in thresholds:
            within = (abs_errors <= threshold).mean() * 100
            metrics[f'within_{threshold}s'] = within
        
        # 3. Linear regression statistics
        if n > 1:
            slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
            metrics.update({
                'regression_slope': slope,
                'regression_intercept': intercept,
                'regression_r': r_value,
                'regression_p': p_value,
                'regression_std_err': std_err,
                'regression_r2': r_value**2
            })
            
            # R² decomposition
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            metrics.update({
                'ss_res': ss_res,
                'ss_tot': ss_tot,
                'ss_res_ss_tot_ratio': ss_res / ss_tot if ss_tot != 0 else np.inf
            })
        
        # 4. Bland-Altman statistics
        means = (y_true + y_pred) / 2
        differences = y_true - y_pred
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        
        metrics.update({
            'bland_altman_mean_diff': mean_diff,
            'bland_altman_std_diff': std_diff,
            'bland_altman_upper_limit': mean_diff + 1.96 * std_diff,
            'bland_altman_lower_limit': mean_diff - 1.96 * std_diff,
            'bland_altman_within_limits': ((differences >= (mean_diff - 1.96 * std_diff)) & 
                                          (differences <= (mean_diff + 1.96 * std_diff))).mean() * 100
        })
        
        # 5. Normality tests
        if n > 3:
            # Shapiro-Wilk test (for n ≤ 5000)
            if n <= 5000:
                try:
                    shapiro_stat, shapiro_p = shapiro(residuals)
                    metrics.update({
                        'shapiro_stat': shapiro_stat,
                        'shapiro_p': shapiro_p,
                        'shapiro_normal': shapiro_p > self.alpha
                    })
                except:
                    pass
            
            # D'Agostino K² test
            try:
                k2_stat, k2_p = normaltest(residuals)
                metrics.update({
                    'dagostino_k2': k2_stat,
                    'dagostino_p': k2_p,
                    'dagostino_normal': k2_p > self.alpha
                })
            except:
                pass
            
            # Anderson-Darling test
            try:
                anderson_result = stats.anderson(residuals, dist='norm')
                metrics.update({
                    'anderson_stat': anderson_result.statistic,
                    'anderson_critical': anderson_result.critical_values[-1],  # 5% level
                    'anderson_normal': anderson_result.statistic < anderson_result.critical_values[-1]
                })
            except:
                pass
        
        # 6. Heteroscedasticity tests
        if n > 10:
            # Bartlett's test
            try:
                # Split data into two halves based on predicted values
                sorted_idx = np.argsort(y_pred)
                half_idx = n // 2
                group1 = residuals[sorted_idx[:half_idx]]
                group2 = residuals[sorted_idx[half_idx:]]
                bartlett_stat, bartlett_p = bartlett(group1, group2)
                metrics.update({
                    'bartlett_stat': bartlett_stat,
                    'bartlett_p': bartlett_p,
                    'bartlett_homoscedastic': bartlett_p > self.alpha
                })
            except:
                pass
            
            # Levene's test
            try:
                levene_stat, levene_p = levene(group1, group2, center='median')
                metrics.update({
                    'levene_stat': levene_stat,
                    'levene_p': levene_p,
                    'levene_homoscedastic': levene_p > self.alpha
                })
            except:
                pass
        
        # 7. Autocorrelation analysis
        if n > 2:
            try:
                lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
                metrics.update({
                    'autocorrelation_lag1': lag1_corr,
                    'durbin_watson': np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
                })
            except:
                pass
        
        self.results['metrics'] = metrics
        return metrics
    
    def compare_models(self, y_true: np.ndarray, 
                      y_pred1: np.ndarray, 
                      y_pred2: np.ndarray,
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Compare two models with ALL statistical tests
        
        Args:
            y_true: True values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary with all comparison results
        """
        y_true = np.asarray(y_true).flatten()
        y_pred1 = np.asarray(y_pred1).flatten()
        y_pred2 = np.asarray(y_pred2).flatten()
        
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
        n = len(y_true)
        
        # 1. Basic error statistics
        comparison = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'n_samples': n,
            'model1_mean_error': np.mean(errors1),
            'model1_std_error': np.std(errors1),
            'model1_median_error': np.median(errors1),
            'model2_mean_error': np.mean(errors2),
            'model2_std_error': np.std(errors2),
            'model2_median_error': np.median(errors2),
            'mean_difference': np.mean(errors1) - np.mean(errors2),
            'median_difference': np.median(errors1) - np.median(errors2)
        }
        
        # 2. Improvement metrics
        mean_improvement = ((np.mean(errors1) - np.mean(errors2)) / np.mean(errors1)) * 100
        median_improvement = ((np.median(errors1) - np.median(errors2)) / np.median(errors1)) * 100
        
        comparison.update({
            'mean_improvement_pct': mean_improvement,
            'median_improvement_pct': median_improvement,
            'improvement_absolute': np.mean(errors1) - np.mean(errors2)
        })
        
        # 3. Statistical tests for paired data
        # Paired t-test
        t_stat, t_pval = ttest_rel(errors1, errors2)
        
        # Wilcoxon signed-rank test
        try:
            w_stat, w_pval = wilcoxon(errors1, errors2)
        except:
            w_stat, w_pval = np.nan, np.nan
        
        # Sign test
        n_better = np.sum(errors2 < errors1)
        n_worse = np.sum(errors2 > errors1)
        sign_test_p = stats.binomtest(min(n_better, n_worse), n_better + n_worse, 0.5).pvalue
        
        comparison.update({
            'paired_ttest': {'statistic': t_stat, 'p_value': t_pval, 'significant': t_pval < self.alpha},
            'wilcoxon': {'statistic': w_stat, 'p_value': w_pval, 'significant': w_pval < self.alpha},
            'sign_test': {'n_better': n_better, 'n_worse': n_worse, 'p_value': sign_test_p,
                         'significant': sign_test_p < self.alpha}
        })
        
        # 4. Effect sizes
        # Cohen's d
        mean_diff = np.mean(errors1 - errors2)
        pooled_std = np.sqrt((np.std(errors1)**2 + np.std(errors2)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # Cliff's delta
        def cliffs_delta(x, y):
            """Cliff's delta effect size"""
            nx, ny = len(x), len(y)
            pairs = np.array([(i, j) for i in x for j in y])
            wins = np.sum(pairs[:, 0] > pairs[:, 1])
            losses = np.sum(pairs[:, 0] < pairs[:, 1])
            return (wins - losses) / (nx * ny)
        
        try:
            cliffs_d = cliffs_delta(errors2, errors1)  # Note: reversed for improvement
        except:
            cliffs_d = np.nan
        
        # Probability of superiority
        prob_superior = np.mean(errors2 < errors1)
        
        comparison.update({
            'effect_sizes': {
                'cohens_d': cohens_d,
                'cliffs_delta': cliffs_d,
                'probability_superiority': prob_superior
            }
        })
        
        # 5. Bootstrap confidence intervals for improvement
        if n > 10:
            n_bootstrap = 1000
            bootstrap_improvements = []
            for _ in range(n_bootstrap):
                indices = np.random.choice(n, n, replace=True)
                bootstrap_errors1 = errors1[indices]
                bootstrap_errors2 = errors2[indices]
                improvement = ((np.mean(bootstrap_errors1) - np.mean(bootstrap_errors2)) / 
                              np.mean(bootstrap_errors1)) * 100
                bootstrap_improvements.append(improvement)
            
            comparison.update({
                'bootstrap_ci_95': np.percentile(bootstrap_improvements, [2.5, 97.5]),
                'bootstrap_ci_90': np.percentile(bootstrap_improvements, [5, 95]),
                'bootstrap_mean_improvement': np.mean(bootstrap_improvements),
                'bootstrap_std_improvement': np.std(bootstrap_improvements)
            })
        
        # 6. Regression comparison
        r2_1 = r2_score(y_true, y_pred1)
        r2_2 = r2_score(y_true, y_pred2)
        
        comparison.update({
            'r2_comparison': {
                'model1_r2': r2_1,
                'model2_r2': r2_2,
                'r2_difference': r2_2 - r2_1,
                'r2_improvement_pct': ((r2_2 - r2_1) / (1 - r2_1)) * 100 if r2_1 < 1 else 0
            }
        })
        
        self.results['comparison'] = comparison
        return comparison
    
    def create_comprehensive_diagnostic_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                                           metrics: Dict = None, save_path: str = None) -> plt.Figure:
        """
        Create 16-panel comprehensive diagnostic plot
        
        Args:
            y_true: True values
            y_pred: Predicted values
            metrics: Pre-computed metrics (optional)
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if metrics is None:
            metrics = self.compute_all_metrics(y_true, y_pred)
        
        residuals = y_true - y_pred
        abs_errors = np.abs(residuals)
        means = (y_true + y_pred) / 2
        differences = y_true - y_pred
        
        # Create figure with 16 subplots
        fig = plt.figure(figsize=(24, 20))
        fig.suptitle(f'COMPREHENSIVE DIAGNOSTIC ANALYSIS: {self.model_name}', 
                    fontsize=24, fontweight='bold', y=1.02)
        
        # 1. Scatter plot with regression line
        ax1 = plt.subplot(4, 4, 1)
        scatter1 = ax1.scatter(y_true, y_pred, c=abs_errors, cmap='RdYlGn_r', 
                              alpha=0.6, s=20, edgecolors='black', linewidth=0.5,
                              vmin=0, vmax=np.percentile(abs_errors, 95))
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax1.plot(lims, lims, 'r--', lw=3, alpha=0.7, label='Perfect')
        
        # Regression line
        x_range = np.linspace(lims[0], lims[1], 100)
        if 'regression_slope' in metrics:
            slope = metrics['regression_slope']
            intercept = metrics['regression_intercept']
            ax1.plot(x_range, slope * x_range + intercept, 'b-', lw=2, alpha=0.7, 
                    label=f'Fit: y={slope:.2f}x+{intercept:.1f}')
        
        ax1.set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
        ax1.set_title(f'1. Prediction Scatter\nR²={metrics["R2"]:.3f}, ρ={metrics["Pearson_r"]:.3f}', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Abs Error (s)')
        
        # 2. Residuals vs Predicted
        ax2 = plt.subplot(4, 4, 2)
        scatter2 = ax2.scatter(y_pred, residuals, c=abs_errors, cmap='coolwarm',
                              alpha=0.6, s=15, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7, linewidth=2)
        ax2.axhline(y=np.mean(residuals) + 1.96*np.std(residuals), color='orange', 
                   linestyle=':', linewidth=2, alpha=0.7, label='±1.96σ')
        ax2.axhline(y=np.mean(residuals) - 1.96*np.std(residuals), color='orange', 
                   linestyle=':', linewidth=2, alpha=0.7)
        
        # Add LOESS smoothing
        if len(y_pred) > 50:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(residuals, y_pred, frac=0.3)
                ax2.plot(smooth[:, 0], smooth[:, 1], 'g-', linewidth=3, alpha=0.8, label='LOESS fit')
            except:
                pass
        
        ax2.set_xlabel('Predicted RT (s)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Residual (True - Pred) (s)', fontsize=14, fontweight='bold')
        ax2.set_title('2. Residuals vs Predicted\nHeteroscedasticity Check', 
                     fontsize=16, fontweight='bold')
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. Error distribution histogram
        ax3 = plt.subplot(4, 4, 3)
        n_bins = min(60, int(len(residuals)/20))
        ax3.hist(residuals, bins=n_bins, color='steelblue', edgecolor='black', 
                alpha=0.7, density=True)
        
        # Add normal distribution fit
        x = np.linspace(np.min(residuals), np.max(residuals), 200)
        if 'residual_mean' in metrics and 'residual_std' in metrics:
            normal_pdf = stats.norm.pdf(x, metrics['residual_mean'], metrics['residual_std'])
            ax3.plot(x, normal_pdf, 'r-', linewidth=3, alpha=0.8, label='Normal fit')
        
        ax3.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.7)
        ax3.set_xlabel('Prediction Error (s)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax3.set_title(f'3. Error Distribution\nSkew={metrics["residual_skew"]:.2f}, Kurt={metrics["residual_kurtosis"]:.2f}', 
                     fontsize=16, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax4 = plt.subplot(4, 4, 4)
        stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('4. Q-Q Plot (Normality Check)', fontsize=16, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add normality test results
        normality_text = ""
        if 'shapiro_normal' in metrics:
            normality_text += f"Shapiro-Wilk: p={metrics['shapiro_p']:.2e}\n"
        if 'dagostino_normal' in metrics:
            normality_text += f"D'Agostino: p={metrics['dagostino_p']:.2e}"
        
        if normality_text:
            ax4.text(0.05, 0.95, normality_text, transform=ax4.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 5. CDF of absolute errors
        ax5 = plt.subplot(4, 4, 5)
        sorted_abs_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        ax5.plot(sorted_abs_errors, cdf, 'b-', linewidth=3, alpha=0.8)
        
        # Add threshold lines
        thresholds = [10, 30, 60, 120]
        colors = ['green', 'orange', 'red', 'purple']
        for threshold, color in zip(thresholds, colors):
            if np.max(sorted_abs_errors) > threshold:
                ax5.axvline(x=threshold, color=color, linestyle='--', alpha=0.7, linewidth=2)
                prop = (abs_errors <= threshold).mean() * 100
                ax5.text(threshold, 0.5, f'{prop:.1f}%', fontsize=11, 
                        color=color, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax5.set_xlabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
        ax5.set_title('5. CDF of Absolute Errors\nPerformance by Threshold', 
                     fontsize=16, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. Bland-Altman plot
        ax6 = plt.subplot(4, 4, 6)
        ax6.scatter(means, differences, alpha=0.5, s=15, edgecolors='black', 
                   linewidth=0.5, color='mediumblue')
        ax6.axhline(y=np.mean(differences), color='blue', linestyle='-', 
                   linewidth=3, alpha=0.8, label=f'Mean = {np.mean(differences):.1f}s')
        ax6.axhline(y=np.mean(differences) + 1.96*np.std(differences), color='red', 
                   linestyle='--', linewidth=2, alpha=0.8, label='+1.96 SD')
        ax6.axhline(y=np.mean(differences) - 1.96*np.std(differences), color='red', 
                   linestyle='--', linewidth=2, alpha=0.8, label='-1.96 SD')
        ax6.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
        
        ax6.set_xlabel('Mean of True and Predicted (s)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Difference (True - Pred) (s)', fontsize=14, fontweight='bold')
        ax6.set_title(f'6. Bland-Altman Plot\n{metrics["bland_altman_within_limits"]:.1f}% within limits', 
                     fontsize=16, fontweight='bold')
        ax6.legend(fontsize=9, loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        # 7. Error vs RT magnitude
        ax7 = plt.subplot(4, 4, 7)
        scatter7 = ax7.scatter(y_true, abs_errors, alpha=0.5, s=15, 
                              c=y_true, cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Add moving average
        if len(y_true) > 20:
            sorted_idx = np.argsort(y_true)
            sorted_true = y_true[sorted_idx]
            sorted_errors = abs_errors[sorted_idx]
            
            window_size = max(5, len(y_true) // 20)
            rolling_mean = np.convolve(sorted_errors, np.ones(window_size)/window_size, mode='valid')
            rolling_true = np.convolve(sorted_true, np.ones(window_size)/window_size, mode='valid')
            
            ax7.plot(rolling_true, rolling_mean, 'r-', linewidth=3, alpha=0.8, 
                    label=f'{window_size}-pt moving avg')
        
        ax7.set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
        ax7.set_ylabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax7.set_title('7. Error vs RT Magnitude\nSystematic Bias Check', 
                     fontsize=16, fontweight='bold')
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        plt.colorbar(scatter7, ax=ax7, label='True RT (s)')
        
        # 8. Threshold performance bar chart
        ax8 = plt.subplot(4, 4, 8)
        thresholds = [10, 30, 60]
        percentages = [(abs_errors <= t).mean() * 100 for t in thresholds]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax8.bar(range(len(thresholds)), percentages, color=colors, edgecolor='black')
        
        ax8.set_xlabel('Error Threshold (s)', fontsize=14, fontweight='bold')
        ax8.set_ylabel('% of Predictions', fontsize=14, fontweight='bold')
        ax8.set_title('8. Threshold Performance\nAccuracy by Tolerance', 
                     fontsize=16, fontweight='bold')
        ax8.set_xticks(range(len(thresholds)))
        ax8.set_xticklabels([f'≤{t}s' for t in thresholds])
        ax8.set_ylim(0, 100)
        ax8.grid(True, alpha=0.3, axis='y')
        
        for bar, percentage in zip(bars, percentages):
            height = bar.get_height()
            ax8.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{percentage:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
        
        # 9. Error distribution by RT quartile
        ax9 = plt.subplot(4, 4, 9)
        rt_bins = np.percentile(y_true, [0, 25, 50, 75, 100])
        bin_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']
        binned_errors = []
        
        for i in range(len(rt_bins)-1):
            mask = (y_true >= rt_bins[i]) & (y_true < rt_bins[i+1])
            binned_errors.append(abs_errors[mask])
        
        bp = ax9.boxplot(binned_errors, labels=bin_labels, patch_artist=True)
        for box, color in zip(bp['boxes'], ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']):
            box.set_facecolor(color)
        
        ax9.set_xlabel('RT Quartile', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax9.set_title('9. Error by RT Range\nPerformance Stratification', 
                     fontsize=16, fontweight='bold')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Residual autocorrelation
        ax10 = plt.subplot(4, 4, 10)
        max_lag = min(20, len(residuals) // 4)
        if max_lag > 1:
            acf = []
            lags = range(1, max_lag + 1)
            for lag in lags:
                if len(residuals) > lag:
                    corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    acf.append(corr)
            
            ax10.bar(lags, acf, color='steelblue', edgecolor='black', alpha=0.7)
            ax10.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax10.axhline(y=2/np.sqrt(len(residuals)), color='red', linestyle='--', 
                        linewidth=1.5, alpha=0.7, label='95% CI')
            ax10.axhline(y=-2/np.sqrt(len(residuals)), color='red', linestyle='--', 
                        linewidth=1.5, alpha=0.7)
            
            if 'autocorrelation_lag1' in metrics:
                ax10.text(0.05, 0.95, f'Lag-1: {metrics["autocorrelation_lag1"]:.3f}', 
                         transform=ax10.transAxes, fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax10.set_xlabel('Lag', fontsize=14, fontweight='bold')
        ax10.set_ylabel('Autocorrelation', fontsize=14, fontweight='bold')
        ax10.set_title('10. Residual Autocorrelation\nIndependence Check', 
                      fontsize=16, fontweight='bold')
        ax10.legend(fontsize=9)
        ax10.grid(True, alpha=0.3)
        
        # 11. Metrics summary table
        ax11 = plt.subplot(4, 4, 11)
        ax11.axis('off')
        
        # Prepare table data
        table_data = [
            ['Metric', 'Value'],
            ['MedAE', f'{metrics["MedAE"]:.2f} s'],
            ['MAE', f'{metrics["MAE"]:.2f} s'],
            ['RMSE', f'{metrics["RMSE"]:.2f} s'],
            ['R²', f'{metrics["R2"]:.4f}'],
            ['Pearson r', f'{metrics["Pearson_r"]:.4f}'],
            ['Spearman ρ', f'{metrics["Spearman_rho"]:.4f}'],
            ['MAPE', f'{metrics["MAPE"]:.2f} %'],
            ['≤30s', f'{metrics["within_30s"]:.1f} %'],
            ['≤60s', f'{metrics["within_60s"]:.1f} %']
        ]
        
        table = ax11.table(cellText=table_data, loc='center', cellLoc='center', 
                          colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        
        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor('#2c3e50')
            table[(0, j)].set_text_props(weight='bold', color='white')
        
        # Style alternating rows
        for i in range(1, len(table_data)):
            color = '#ecf0f1' if i % 2 == 0 else '#ffffff'
            for j in range(2):
                table[(i, j)].set_facecolor(color)
        
        ax11.set_title('11. Key Metrics Summary', fontsize=16, fontweight='bold', 
                      pad=20, loc='center')
        
        # 12. Statistical tests results
        ax12 = plt.subplot(4, 4, 12)
        ax12.axis('off')
        
        stat_text = "12. Statistical Tests Results\n\n"
        
        # Normality tests
        if 'shapiro_normal' in metrics:
            normal_str = "✓" if metrics['shapiro_normal'] else "✗"
            stat_text += f"Normality (Shapiro-Wilk): {normal_str}\n"
            stat_text += f"  p = {metrics['shapiro_p']:.2e}\n\n"
        
        # Homoscedasticity tests
        if 'bartlett_homoscedastic' in metrics:
            homo_str = "✓" if metrics['bartlett_homoscedastic'] else "✗"
            stat_text += f"Homoscedasticity: {homo_str}\n"
            stat_text += f"  Bartlett p = {metrics['bartlett_p']:.2e}\n\n"
        
        # Regression significance
        if 'regression_p' in metrics:
            reg_str = "✓" if metrics['regression_p'] < self.alpha else "✗"
            stat_text += f"Regression: {reg_str}\n"
            stat_text += f"  p = {metrics['regression_p']:.2e}\n\n"
        
        # Error statistics
        stat_text += f"Error Statistics:\n"
        stat_text += f"  Mean: {metrics['residual_mean']:.2f} ± {metrics['residual_std']:.2f} s\n"
        stat_text += f"  Skewness: {metrics['residual_skew']:.2f}\n"
        stat_text += f"  Kurtosis: {metrics['residual_kurtosis']:.2f}\n"
        
        ax12.text(0.1, 0.5, stat_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
        
        # 13. Error density by RT
        ax13 = plt.subplot(4, 4, 13)
        
        # Create 2D histogram
        hist, xedges, yedges = np.histogram2d(y_true, abs_errors, bins=(30, 30))
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax13.imshow(hist.T, origin='lower', aspect='auto', extent=extent,
                        cmap='YlOrRd', interpolation='nearest')
        
        ax13.set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
        ax13.set_ylabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax13.set_title('13. Error Density Distribution\n2D Histogram', 
                      fontsize=16, fontweight='bold')
        plt.colorbar(im, ax=ax13, label='Count')
        
        # 14. Cook's distance (influence analysis)
        ax14 = plt.subplot(4, 4, 14)
        
        if len(y_true) > 10:
            try:
                # Calculate Cook's distance
                X = np.column_stack([np.ones_like(y_true), y_true])
                H = X @ np.linalg.pinv(X.T @ X) @ X.T
                leverage = np.diag(H)
                mse = np.mean(residuals**2)
                cooks_d = (residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)
                
                ax14.scatter(range(len(cooks_d)), cooks_d, alpha=0.6, s=20, 
                           color='purple', edgecolors='black', linewidth=0.5)
                ax14.axhline(y=4/len(y_true), color='red', linestyle='--', 
                           linewidth=2, alpha=0.7, label='Influence threshold')
                
                # Identify influential points
                influential = cooks_d > (4/len(y_true))
                if np.any(influential):
                    ax14.scatter(np.where(influential)[0], cooks_d[influential], 
                               color='red', s=50, edgecolors='black', linewidth=1.5,
                               label=f'Influential (n={np.sum(influential)})')
                
                ax14.set_xlabel('Sample Index', fontsize=14, fontweight='bold')
                ax14.set_ylabel("Cook's Distance", fontsize=14, fontweight='bold')
                ax14.set_title('14. Influence Analysis\nOutlier Detection', 
                              fontsize=16, fontweight='bold')
                ax14.legend(fontsize=9)
                ax14.grid(True, alpha=0.3)
            except:
                ax14.text(0.5, 0.5, 'Influence analysis\nnot available', 
                         ha='center', va='center', fontsize=12)
                ax14.set_title('14. Influence Analysis', fontsize=16, fontweight='bold')
        
        # 15. Prediction intervals
        ax15 = plt.subplot(4, 4, 15)
        
        sorted_idx = np.argsort(y_true)
        sorted_true = y_true[sorted_idx]
        sorted_pred = y_pred[sorted_idx]
        sorted_errors = abs_errors[sorted_idx]
        
        # Calculate prediction intervals
        z_score = 1.96  # 95% confidence
        upper_bound = sorted_pred + z_score * sorted_errors
        lower_bound = sorted_pred - z_score * sorted_errors
        
        ax15.fill_between(sorted_true, lower_bound, upper_bound, 
                         alpha=0.3, color='lightblue', label='95% PI')
        ax15.plot(sorted_true, sorted_pred, 'b-', linewidth=2, alpha=0.8, label='Predictions')
        ax15.plot(sorted_true, sorted_true, 'r--', linewidth=2, alpha=0.7, label='Perfect')
        
        ax15.set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
        ax15.set_ylabel('Predicted RT (s)', fontsize=14, fontweight='bold')
        ax15.set_title('15. Prediction Intervals\nUncertainty Visualization', 
                      fontsize=16, fontweight='bold')
        ax15.legend(fontsize=9, loc='upper left')
        ax15.grid(True, alpha=0.3)
        
        # 16. Summary statistics
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('off')
        
        summary_text = "16. Model Summary\n\n"
        summary_text += f"Model: {self.model_name}\n"
        summary_text += f"Samples: {len(y_true):,}\n"
        summary_text += f"MedAE: {metrics['MedAE']:.2f} s\n"
        summary_text += f"R²: {metrics['R2']:.3f}\n"
        summary_text += f"Pearson: {metrics['Pearson_r']:.3f}\n\n"
        
        # Performance classification
        medae = metrics['MedAE']
        if medae < 20:
            performance = "Excellent"
        elif medae < 30:
            performance = "Good"
        elif medae < 40:
            performance = "Acceptable"
        elif medae < 50:
            performance = "Moderate"
        else:
            performance = "Needs Improvement"
        
        summary_text += f"Performance: {performance}\n\n"
        
        # Key findings
        summary_text += "Key Findings:\n"
        if metrics['within_30s'] > 70:
            summary_text += "• High accuracy (≥70% within 30s)\n"
        if metrics['R2'] > 0.8:
            summary_text += "• Strong correlation (R² > 0.8)\n"
        if 'shapiro_normal' in metrics and metrics['shapiro_normal']:
            summary_text += "• Normally distributed errors\n"
        
        ax16.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✓ Comprehensive diagnostic plot saved to {save_path}")
        
        return fig
    
    def create_comparison_plot(self, y_true: np.ndarray, 
                              y_pred1: np.ndarray, 
                              y_pred2: np.ndarray,
                              model1_name: str = "Model 1",
                              model2_name: str = "Model 2",
                              save_path: str = None) -> plt.Figure:
        """
        Create comprehensive comparison plot between two models
        
        Args:
            y_true: True values
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model1_name: Name of first model
            model2_name: Name of second model
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
        
        # Compute comparison metrics
        comparison = self.compare_models(y_true, y_pred1, y_pred2, 
                                        model1_name, model2_name)
        
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f'MODEL COMPARISON: {model1_name} vs {model2_name}', 
                    fontsize=24, fontweight='bold', y=1.02)
        
        # 1. Error comparison scatter
        ax1 = plt.subplot(3, 3, 1)
        ax1.scatter(errors1, errors2, alpha=0.6, s=20, c='steelblue', edgecolors='black', linewidth=0.5)
        lims = [min(errors1.min(), errors2.min()), max(errors1.max(), errors2.max())]
        ax1.plot(lims, lims, 'r--', lw=2, alpha=0.7, label='Equal performance')
        ax1.set_xlabel(f'{model1_name} Error (s)', fontsize=14, fontweight='bold')
        ax1.set_ylabel(f'{model2_name} Error (s)', fontsize=14, fontweight='bold')
        ax1.set_title('1. Error Comparison\nPoints below red line: Model 2 better', 
                     fontsize=16, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Error distribution comparison
        ax2 = plt.subplot(3, 3, 2)
        bins = np.linspace(0, np.max([errors1.max(), errors2.max()]), 40)
        ax2.hist(errors1, bins=bins, alpha=0.5, color='blue', label=model1_name, density=True)
        ax2.hist(errors2, bins=bins, alpha=0.5, color='red', label=model2_name, density=True)
        ax2.axvline(np.median(errors1), color='blue', linestyle='--', linewidth=2, 
                   label=f'{model1_name} MedAE: {np.median(errors1):.1f}s')
        ax2.axvline(np.median(errors2), color='red', linestyle='--', linewidth=2, 
                   label=f'{model2_name} MedAE: {np.median(errors2):.1f}s')
        ax2.set_xlabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=14, fontweight='bold')
        ax2.set_title('2. Error Distribution Comparison', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # 3. CDF comparison
        ax3 = plt.subplot(3, 3, 3)
        sorted_err1 = np.sort(errors1)
        sorted_err2 = np.sort(errors2)
        cdf1 = np.arange(1, len(sorted_err1) + 1) / len(sorted_err1)
        cdf2 = np.arange(1, len(sorted_err2) + 1) / len(sorted_err2)
        ax3.plot(sorted_err1, cdf1, 'b-', linewidth=3, label=model1_name)
        ax3.plot(sorted_err2, cdf2, 'r-', linewidth=3, label=model2_name)
        
        thresholds = [10, 30, 60]
        for threshold in thresholds:
            ax3.axvline(threshold, color='gray', linestyle='--', alpha=0.5)
            prop1 = (errors1 <= threshold).mean() * 100
            prop2 = (errors2 <= threshold).mean() * 100
            ax3.text(threshold, 0.3, f'Δ={prop2-prop1:+.1f}%', 
                    ha='center', va='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Absolute Error (s)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Cumulative Probability', fontsize=14, fontweight='bold')
        ax3.set_title('3. CDF Comparison\nVertical lines show thresholds', 
                     fontsize=16, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. Improvement scatter
        ax4 = plt.subplot(3, 3, 4)
        improvement = errors1 - errors2
        ax4.scatter(y_true, improvement, alpha=0.6, s=20, 
                   c=np.sign(improvement), cmap='RdYlGn', vmin=-1, vmax=1,
                   edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        # Add LOESS smoothing
        if len(y_true) > 50:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smooth = lowess(improvement, y_true, frac=0.3)
                ax4.plot(smooth[:, 0], smooth[:, 1], 'purple', linewidth=3, 
                        alpha=0.8, label='Trend')
            except:
                pass
        
        ax4.set_xlabel('True RT (s)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Improvement (Model1 - Model2 Error) (s)', 
                      fontsize=14, fontweight='bold')
        ax4.set_title('4. Improvement by RT\nGreen: Model2 better, Red: Model1 better', 
                     fontsize=16, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        
        # 5. Statistical tests results
        ax5 = plt.subplot(3, 3, 5)
        ax5.axis('off')
        
        stat_text = "5. Statistical Test Results\n\n"
        
        # Paired t-test
        ttest = comparison['paired_ttest']
        stat_text += f"Paired t-test:\n"
        stat_text += f"  t = {ttest['statistic']:.3f}\n"
        stat_text += f"  p = {ttest['p_value']:.2e}\n"
        stat_text += f"  {'✓ Significant' if ttest['significant'] else '✗ Not significant'}\n\n"
        
        # Wilcoxon test
        wilcoxon = comparison['wilcoxon']
        stat_text += f"Wilcoxon signed-rank:\n"
        stat_text += f"  W = {wilcoxon['statistic']:.0f}\n"
        stat_text += f"  p = {wilcoxon['p_value']:.2e}\n"
        stat_text += f"  {'✓ Significant' if wilcoxon['significant'] else '✗ Not significant'}\n\n"
        
        # Effect sizes
        effects = comparison['effect_sizes']
        stat_text += f"Effect Sizes:\n"
        stat_text += f"  Cohen's d = {effects['cohens_d']:.3f}\n"
        if not np.isnan(effects['cliffs_delta']):
            stat_text += f"  Cliff's δ = {effects['cliffs_delta']:.3f}\n"
        stat_text += f"  P(Model2 better) = {effects['probability_superiority']:.1%}\n"
        
        ax5.text(0.1, 0.5, stat_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8))
        
        # 6. Improvement metrics
        ax6 = plt.subplot(3, 3, 6)
        ax6.axis('off')
        
        metrics_text = "6. Improvement Metrics\n\n"
        metrics_text += f"Mean Error:\n"
        metrics_text += f"  {model1_name}: {comparison['model1_mean_error']:.2f} s\n"
        metrics_text += f"  {model2_name}: {comparison['model2_mean_error']:.2f} s\n"
        metrics_text += f"  Difference: {comparison['mean_difference']:.2f} s\n\n"
        
        metrics_text += f"Median Error:\n"
        metrics_text += f"  {model1_name}: {comparison['model1_median_error']:.2f} s\n"
        metrics_text += f"  {model2_name}: {comparison['model2_median_error']:.2f} s\n"
        metrics_text += f"  Difference: {comparison['median_difference']:.2f} s\n\n"
        
        metrics_text += f"Improvement:\n"
        metrics_text += f"  Mean: {comparison['mean_improvement_pct']:.1f}%\n"
        metrics_text += f"  Median: {comparison['median_improvement_pct']:.1f}%\n"
        
        if 'bootstrap_ci_95' in comparison:
            ci = comparison['bootstrap_ci_95']
            metrics_text += f"  95% CI: [{ci[0]:.1f}%, {ci[1]:.1f}%]\n"
        
        ax6.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))
        
        # 7. R² comparison
        ax7 = plt.subplot(3, 3, 7)
        r2_values = [comparison['r2_comparison']['model1_r2'], 
                    comparison['r2_comparison']['model2_r2']]
        models = [model1_name, model2_name]
        colors = ['blue', 'red']
        
        bars = ax7.bar(models, r2_values, color=colors, edgecolor='black')
        ax7.set_ylabel('R²', fontsize=14, fontweight='bold')
        ax7.set_title('7. R² Comparison', fontsize=16, fontweight='bold')
        ax7.set_ylim(0, 1)
        ax7.grid(True, alpha=0.3, axis='y')
        
        for bar, r2 in zip(bars, r2_values):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # 8. Win/loss analysis
        ax8 = plt.subplot(3, 3, 8)
        win_loss = [comparison['sign_test']['n_better'],
                   comparison['sign_test']['n_worse']]
        labels = [f'{model2_name} better', f'{model1_name} better']
        colors = ['#2ecc71', '#e74c3c']
        
        ax8.pie(win_loss, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax8.set_title('8. Win/Loss Analysis\n(Which model has smaller error?)', 
                     fontsize=16, fontweight='bold')
        
        # 9. Summary and recommendation
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Determine which model is better
        mean_imp = comparison['mean_improvement_pct']
        median_imp = comparison['median_improvement_pct']
        ttest_sig = comparison['paired_ttest']['significant']
        wilcoxon_sig = comparison['wilcoxon']['significant']
        
        if ttest_sig and wilcoxon_sig and mean_imp > 0 and median_imp > 0:
            conclusion = f"{model2_name} is statistically significantly better"
            color = '#2ecc71'
        elif ttest_sig or wilcoxon_sig:
            if mean_imp > 0 and median_imp > 0:
                conclusion = f"{model2_name} is better with some statistical significance"
                color = '#27ae60'
            elif mean_imp < 0 and median_imp < 0:
                conclusion = f"{model1_name} is better with some statistical significance"
                color = '#e74c3c'
            else:
                conclusion = "Mixed results, no clear winner"
                color = '#f39c12'
        else:
            conclusion = "No statistically significant difference"
            color = '#95a5a6'
        
        summary_text = "9. Conclusion\n\n"
        summary_text += f"{conclusion}\n\n"
        
        if ttest_sig and wilcoxon_sig:
            summary_text += f"• Both tests significant (p < {self.alpha})\n"
        elif ttest_sig:
            summary_text += f"• T-test significant (p < {self.alpha})\n"
        elif wilcoxon_sig:
            summary_text += f"• Wilcoxon test significant (p < {self.alpha})\n"
        
        summary_text += f"• Mean improvement: {mean_imp:.1f}%\n"
        summary_text += f"• Median improvement: {median_imp:.1f}%\n\n"
        
        if mean_imp > 5 and ttest_sig:
            summary_text += "Recommendation: Use the better model"
        elif mean_imp > 0:
            summary_text += "Recommendation: Consider the better model\nif the improvement is meaningful"
        else:
            summary_text += "Recommendation: Models are comparable"
        
        ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f" Model comparison plot saved to {save_path}")
        
        return fig
    
    def export_results(self, export_dir: str = "results"):
        """
        Export all results to files
        
        Args:
            export_dir: Directory to save results
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export metrics to JSON
        if 'metrics' in self.results:
            metrics_file = export_path / f"{self.model_name}_metrics.json"
            with open(metrics_file, 'w') as f:
                # Convert numpy types to Python types
                def convert(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert(item) for item in obj]
                    else:
                        return obj
                
                json.dump(convert(self.results), f, indent=2)
            print(f" Metrics exported to {metrics_file}")
        
        # Export to CSV
        if 'metrics' in self.results:
            # Flatten metrics for CSV
            flat_metrics = {}
            for key, value in self.results['metrics'].items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        flat_metrics[f"{key}_{subkey}"] = subvalue
                else:
                    flat_metrics[key] = value
            
            df = pd.DataFrame([flat_metrics])
            csv_file = export_path / f"{self.model_name}_metrics.csv"
            df.to_csv(csv_file, index=False)
            print(f" Metrics exported to {csv_file}")
        
        # Export comparison results
        if 'comparison' in self.results:
            comp_file = export_path / f"{self.model_name}_comparison.json"
            with open(comp_file, 'w') as f:
                json.dump(self.results['comparison'], f, indent=2, default=str)
            print(f" Comparison results exported to {comp_file}")
    
    def print_summary_report(self):
        """Print comprehensive summary report to console"""
        print("\n" + "="*80)
        print(f"COMPREHENSIVE STATISTICAL REPORT: {self.model_name}")
        print("="*80)
        
        if 'metrics' in self.results:
            metrics = self.results['metrics']
            
            print(f"\n SAMPLE INFORMATION:")
            print(f"  Samples: {metrics['n_samples']:,}")
            print(f"  True RT: {metrics['y_true_mean']:.1f} ± {metrics['y_true_std']:.1f} s")
            print(f"  Predicted RT: {metrics['y_pred_mean']:.1f} ± {metrics['y_pred_std']:.1f} s")
            
            print(f"\n CORE PERFORMANCE METRICS:")
            print(f"  MedAE:      {metrics['MedAE']:8.2f} s")
            print(f"  MAE:        {metrics['MAE']:8.2f} s")
            print(f"  RMSE:       {metrics['RMSE']:8.2f} s")
            print(f"  R²:         {metrics['R2']:8.4f}")
            print(f"  Pearson r:  {metrics['Pearson_r']:8.4f} (p={metrics['Pearson_p']:.2e})")
            print(f"  Spearman ρ: {metrics['Spearman_rho']:8.4f} (p={metrics['Spearman_p']:.2e})")
            print(f"  MAPE:       {metrics['MAPE']:8.2f} %")
            
            print(f"\n THRESHOLD PERFORMANCE:")
            print(f"  ≤10s:  {metrics.get('within_10s', 0):8.2f} %")
            print(f"  ≤30s:  {metrics.get('within_30s', 0):8.2f} %")
            print(f"  ≤60s:  {metrics.get('within_60s', 0):8.2f} %")
            print(f"  ≤120s: {metrics.get('within_120s', 0):8.2f} %")
            
            print(f"\n ERROR DISTRIBUTION:")
            print(f"  Mean error:      {metrics['residual_mean']:8.2f} ± {metrics['residual_std']:8.2f} s")
            print(f"  Median error:    {metrics['residual_median']:8.2f} s")
            print(f"  MAD:             {metrics['residual_mad']:8.2f} s")
            print(f"  Skewness:        {metrics['residual_skew']:8.2f}")
            print(f"  Kurtosis:        {metrics['residual_kurtosis']:8.2f}")
            print(f"  Q90 abs error:   {metrics['abs_error_q90']:8.2f} s")
            print(f"  Q95 abs error:   {metrics['abs_error_q95']:8.2f} s")
            
            print(f"\n REGRESSION ANALYSIS:")
            if 'regression_slope' in metrics:
                print(f"  Slope:       {metrics['regression_slope']:8.4f}")
                print(f"  Intercept:   {metrics['regression_intercept']:8.2f}")
                print(f"  R:           {metrics['regression_r']:8.4f}")
                print(f"  p-value:     {metrics['regression_p']:8.2e}")
                print(f"  Std error:   {metrics['regression_std_err']:8.4f}")
            
            print(f"\n STATISTICAL TESTS:")
            if 'shapiro_normal' in metrics:
                normal = "NORMAL" if metrics['shapiro_normal'] else "NON-NORMAL"
                print(f"  Normality (Shapiro-Wilk): {normal} (p={metrics['shapiro_p']:.2e})")
            
            if 'dagostino_normal' in metrics:
                normal = "NORMAL" if metrics['dagostino_normal'] else "NON-NORMAL"
                print(f"  Normality (D'Agostino):  {normal} (p={metrics['dagostino_p']:.2e})")
            
            if 'bartlett_homoscedastic' in metrics:
                homo = "HOMOSCEDASTIC" if metrics['bartlett_homoscedastic'] else "HETEROSCEDASTIC"
                print(f"  Homoscedasticity:        {homo} (p={metrics['bartlett_p']:.2e})")
            
            print(f"\n BLAND-ALTMAN ANALYSIS:")
            print(f"  Mean difference:     {metrics['bland_altman_mean_diff']:8.2f} s")
            print(f"  Std difference:      {metrics['bland_altman_std_diff']:8.2f} s")
            print(f"  Limits of agreement: [{metrics['bland_altman_lower_limit']:.1f}, {metrics['bland_altman_upper_limit']:.1f}] s")
            print(f"  Within limits:       {metrics['bland_altman_within_limits']:8.1f} %")
        
        if 'comparison' in self.results:
            comp = self.results['comparison']
            
            print(f"\n" + "="*80)
            print(f"MODEL COMPARISON: {comp['model1_name']} vs {comp['model2_name']}")
            print(f"="*80)
            
            print(f"\n ERROR STATISTICS:")
            print(f"  {comp['model1_name']}:")
            print(f"    Mean error:   {comp['model1_mean_error']:8.2f} ± {comp['model1_std_error']:8.2f} s")
            print(f"    Median error: {comp['model1_median_error']:8.2f} s")
            
            print(f"\n  {comp['model2_name']}:")
            print(f"    Mean error:   {comp['model2_mean_error']:8.2f} ± {comp['model2_std_error']:8.2f} s")
            print(f"    Median error: {comp['model2_median_error']:8.2f} s")
            
            print(f"\n IMPROVEMENT:")
            print(f"  Mean improvement:   {comp['mean_improvement_pct']:8.1f} %")
            print(f"  Median improvement: {comp['median_improvement_pct']:8.1f} %")
            print(f"  Absolute reduction: {comp['improvement_absolute']:8.2f} s")
            
            print(f"\n STATISTICAL SIGNIFICANCE:")
            ttest = comp['paired_ttest']
            print(f"  Paired t-test:     {'✓ Significant' if ttest['significant'] else '✗ Not significant'}")
            print(f"    t = {ttest['statistic']:.3f}, p = {ttest['p_value']:.2e}")
            
            wilcoxon = comp['wilcoxon']
            print(f"  Wilcoxon test:     {'✓ Significant' if wilcoxon['significant'] else '✗ Not significant'}")
            print(f"    W = {wilcoxon['statistic']:.0f}, p = {wilcoxon['p_value']:.2e}")
            
            print(f"\n EFFECT SIZES:")
            effects = comp['effect_sizes']
            print(f"  Cohen's d:         {effects['cohens_d']:.3f}")
            if not np.isnan(effects['cliffs_delta']):
                print(f"  Cliff's delta:     {effects['cliffs_delta']:.3f}")
            print(f"  P(Model2 better):  {effects['probability_superiority']:.1%}")
            
            print(f"\n R² COMPARISON:")
            r2_comp = comp['r2_comparison']
            print(f"  {comp['model1_name']}: {r2_comp['model1_r2']:.4f}")
            print(f"  {comp['model2_name']}: {r2_comp['model2_r2']:.4f}")
            print(f"  Difference:        {r2_comp['r2_difference']:+.4f}")
            print(f"  % Improvement:     {r2_comp['r2_improvement_pct']:+.1f} %")
        
        print("\n" + "="*80)


def run_comprehensive_analysis(y_true, y_pred, model_name="Model", 
                             comparison_y_pred=None, comparison_name="Comparison Model",
                             save_dir="results"):
    """
    Run complete statistical analysis pipeline
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        comparison_y_pred: Predictions from comparison model (optional)
        comparison_name: Name of comparison model
        save_dir: Directory to save results
    
    Returns:
        analyzer: StatisticalAnalyzer object with all results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING COMPREHENSIVE STATISTICAL ANALYSIS")
    print(f"{'='*80}")
    
    # Initialize analyzer
    analyzer = ComprehensiveStatisticalAnalyzer(model_name=model_name)
    
    # Compute all metrics
    print("\n Computing all metrics...")
    metrics = analyzer.compute_all_metrics(y_true, y_pred)
    
    # Create comprehensive diagnostic plot
    print(" Creating comprehensive diagnostic plot...")
    fig = analyzer.create_comprehensive_diagnostic_plot(
        y_true, y_pred, metrics,
        save_path=Path(save_dir) / f"{model_name}_diagnostics.png"
    )
    
    # If comparison model provided
    if comparison_y_pred is not None:
        print(" Comparing models...")
        comparison = analyzer.compare_models(
            y_true, y_pred, comparison_y_pred,
            model_name, comparison_name
        )
        
        # Create comparison plot
        print(" Creating model comparison plot...")
        comp_fig = analyzer.create_comparison_plot(
            y_true, y_pred, comparison_y_pred,
            model_name, comparison_name,
            save_path=Path(save_dir) / f"{model_name}_vs_{comparison_name}_comparison.png"
        )
    
    # Export results
    print(" Exporting results...")
    analyzer.export_results(save_dir)
    
    # Print summary
    analyzer.print_summary_report()
    
    print(f"\n Analysis complete! Results saved to '{save_dir}'")
    print("="*80)
    
    return analyzer


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 1000
    y_true = np.random.uniform(0, 600, n)
    y_pred = y_true + np.random.normal(0, 50, n)  # Add noise
    y_pred_comparison = y_true + np.random.normal(0, 45, n)  # Slightly better model
    
    # Run analysis
    analyzer = run_comprehensive_analysis(
        y_true=y_true,
        y_pred=y_pred,
        model_name="Test Model",
        comparison_y_pred=y_pred_comparison,
        comparison_name="Improved Model",
        save_dir="example_results"
    )
