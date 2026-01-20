"""
Comprehensive metrics calculation for retention time prediction.

This module provides the RTMetrics class for computing and saving
all relevant performance metrics.
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class RTMetrics:
    """
    Compute and store comprehensive retention time prediction metrics.
    
    This class calculates all relevant metrics once and provides methods
    to save/load results as JSON files.
    """
    
    def __init__(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        y_pred_baseline: Optional[np.ndarray] = None
    ):
        """
        Initialize RTMetrics and compute all metrics.
        
        Args:
            y_true: True retention time values
            y_pred: Predicted retention time values
            model_name: Name of the model for identification
            y_pred_baseline: Baseline predictions for comparison (optional)
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.y_pred_baseline = y_pred_baseline
        
        # Compute all metrics
        self.metrics = self._compute_all_metrics()
        
        # Compute statistical tests if baseline provided
        if y_pred_baseline is not None:
            self.statistical_tests = self._compute_statistical_tests()
        else:
            self.statistical_tests = {}
    
    def _compute_all_metrics(self) -> Dict[str, float]:
        """
        Compute all performance metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        abs_errors = np.abs(self.y_true - self.y_pred)
        errors = self.y_pred - self.y_true
        
        # Avoid division by zero in relative error calculation
        rel_errors = np.abs((self.y_true - self.y_pred) / np.where(self.y_true != 0, self.y_true, 1)) * 100
        
        metrics = {
            # Core metrics
            'MedAE': float(np.median(abs_errors)),
            'MAE': float(np.mean(abs_errors)),
            'RMSE': float(np.sqrt(mean_squared_error(self.y_true, self.y_pred))),
            'R2': float(r2_score(self.y_true, self.y_pred)),
            
            # Correlation metrics
            'Pearson': float(pearsonr(self.y_true, self.y_pred)[0]),
            'Pearson_pval': float(pearsonr(self.y_true, self.y_pred)[1]),
            'Spearman': float(spearmanr(self.y_true, self.y_pred)[0]),
            'Spearman_pval': float(spearmanr(self.y_true, self.y_pred)[1]),
            
            # Threshold accuracy metrics
            'Pct_le_60s': float((abs_errors <= 60).mean() * 100),
            'Pct_le_30s': float((abs_errors <= 30).mean() * 100),
            'Pct_le_10s': float((abs_errors <= 10).mean() * 100),
            
            # Relative error metrics
            'Median_Rel_Error': float(np.median(rel_errors)),
            'Mean_Rel_Error': float(np.mean(rel_errors)),
            
            # Error distribution metrics
            'Mean_Error': float(np.mean(errors)),
            'Std_Error': float(np.std(errors)),
            'Error_Q25': float(np.percentile(abs_errors, 25)),
            'Error_Q75': float(np.percentile(abs_errors, 75)),
            'Error_95CI_lower': float(np.percentile(abs_errors, 2.5)),
            'Error_95CI_upper': float(np.percentile(abs_errors, 97.5)),
            
            # Dataset info
            'n_samples': int(len(self.y_true)),
            'model_name': self.model_name
        }
        
        return metrics
    
    def _compute_statistical_tests(self) -> Dict[str, float]:
        """
        Perform statistical significance tests against baseline.
        
        Returns:
            Dictionary containing statistical test results
        """
        from scipy.stats import wilcoxon, ttest_rel
        
        errors_current = np.abs(self.y_true - self.y_pred)
        errors_baseline = np.abs(self.y_true - self.y_pred_baseline)
        
        # Wilcoxon signed-rank test (non-parametric paired test)
        wilcoxon_stat, wilcoxon_p = wilcoxon(errors_current, errors_baseline)
        
        # Paired t-test
        ttest_stat, ttest_p = ttest_rel(errors_current, errors_baseline)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(errors_current - errors_baseline)
        pooled_std = np.sqrt((np.std(errors_current)**2 + np.std(errors_baseline)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0.0
        
        return {
            'Wilcoxon_statistic': float(wilcoxon_stat),
            'Wilcoxon_p_value': float(wilcoxon_p),
            'T_test_statistic': float(ttest_stat),
            'T_test_p_value': float(ttest_p),
            'Cohens_d': float(cohens_d),
            'Mean_Error_Diff': float(mean_diff),
            'Baseline_Mean_Error': float(np.mean(errors_baseline)),
            'Current_Mean_Error': float(np.mean(errors_current))
        }
    
    def save(self, path: Path):
        """
        Save metrics to JSON file.
        
        Args:
            path: Path to save the metrics JSON file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'metrics': self.metrics,
            'statistical_tests': self.statistical_tests
        }
        
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Metrics saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> Dict:
        """
        Load metrics from JSON file.
        
        Args:
            path: Path to the metrics JSON file
            
        Returns:
            Dictionary containing metrics and statistical tests
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def print_summary(self, benchmark: Optional[Dict[str, float]] = None):
        """
        Print a formatted summary of metrics.
        
        Args:
            benchmark: Optional benchmark metrics for comparison
        """
        print("=" * 80)
        print(f"METRICS SUMMARY: {self.model_name}")
        print("=" * 80)
        
        print("\nCORE METRICS:")
        print(f"  MedAE       : {self.metrics['MedAE']:8.2f} s", end="")
        if benchmark and 'MedAE' in benchmark:
            diff = self.metrics['MedAE'] - benchmark['MedAE']
            print(f"  (Benchmark: {benchmark['MedAE']:.2f} s, Δ={diff:+.2f} s)")
        else:
            print()
        
        print(f"  MAE         : {self.metrics['MAE']:8.2f} s", end="")
        if benchmark and 'MAE' in benchmark:
            diff = self.metrics['MAE'] - benchmark['MAE']
            print(f"  (Benchmark: {benchmark['MAE']:.2f} s, Δ={diff:+.2f} s)")
        else:
            print()
        
        print(f"  RMSE        : {self.metrics['RMSE']:8.2f} s", end="")
        if benchmark and 'RMSE' in benchmark:
            diff = self.metrics['RMSE'] - benchmark['RMSE']
            print(f"  (Benchmark: {benchmark['RMSE']:.2f} s, Δ={diff:+.2f} s)")
        else:
            print()
        
        print(f"  R²          : {self.metrics['R2']:8.3f}   ", end="")
        if benchmark and 'R2' in benchmark:
            diff = self.metrics['R2'] - benchmark['R2']
            print(f"  (Benchmark: {benchmark['R2']:.3f}, Δ={diff:+.3f})")
        else:
            print()
        
        print(f"  Pearson     : {self.metrics['Pearson']:8.3f}   ", end="")
        if benchmark and 'Pearson' in benchmark:
            diff = self.metrics['Pearson'] - benchmark['Pearson']
            print(f"  (Benchmark: {benchmark['Pearson']:.3f}, Δ={diff:+.3f})")
        else:
            print()
        
        print(f"  Spearman    : {self.metrics['Spearman']:8.3f}")
        
        print("\nTHRESHOLD ACCURACY:")
        print(f"  % ≤ 60s     : {self.metrics['Pct_le_60s']:8.2f} %", end="")
        if benchmark and 'Pct_le_60s' in benchmark:
            diff = self.metrics['Pct_le_60s'] - benchmark['Pct_le_60s']
            print(f"  (Benchmark: {benchmark['Pct_le_60s']:.2f} %, Δ={diff:+.2f} %)")
        else:
            print()
        
        print(f"  % ≤ 30s     : {self.metrics['Pct_le_30s']:8.2f} %", end="")
        if benchmark and 'Pct_le_30s' in benchmark:
            diff = self.metrics['Pct_le_30s'] - benchmark['Pct_le_30s']
            print(f"  (Benchmark: {benchmark['Pct_le_30s']:.2f} %, Δ={diff:+.2f} %)")
        else:
            print()
        
        print(f"  % ≤ 10s     : {self.metrics['Pct_le_10s']:8.2f} %", end="")
        if benchmark and 'Pct_le_10s' in benchmark:
            diff = self.metrics['Pct_le_10s'] - benchmark['Pct_le_10s']
            print(f"  (Benchmark: {benchmark['Pct_le_10s']:.2f} %, Δ={diff:+.2f} %)")
        else:
            print()
        
        print("\nRELATIVE ERRORS:")
        print(f"  Median Rel Error : {self.metrics['Median_Rel_Error']:.2f} %")
        print(f"  Mean Rel Error   : {self.metrics['Mean_Rel_Error']:.2f} %")
        
        if self.statistical_tests:
            print("\nSTATISTICAL TESTS (vs. Baseline):")
            print(f"  Wilcoxon p-value : {self.statistical_tests['Wilcoxon_p_value']:.2e}")
            print(f"  T-test p-value   : {self.statistical_tests['T_test_p_value']:.2e}")
            print(f"  Cohen's d        : {self.statistical_tests['Cohens_d']:.3f}")
            
            significance = "✓ Significant" if self.statistical_tests['Wilcoxon_p_value'] < 0.05 else "✗ Not significant"
            print(f"  Significance     : {significance} (p < 0.05)")
        
        print("=" * 80)
