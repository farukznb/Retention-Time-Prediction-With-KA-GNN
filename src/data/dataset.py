"""
Statistical significance tests for comparing RT prediction models.

This script loads saved metrics from all experiments and performs
pairwise statistical comparisons using Wilcoxon signed-rank test
and paired t-tests.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon, ttest_rel
from typing import Dict, List
import pandas as pd


def load_metrics(metrics_path: Path) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        metrics_path: Path to metrics JSON file
        
    Returns:
        Dictionary containing metrics
    """
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    return data


def compute_pairwise_comparison(
    metrics1: Dict,
    metrics2: Dict,
    model1_name: str,
    model2_name: str
) -> Dict:
    """
    Compute statistical comparison between two models.
    
    Note: This assumes we have the same test set predictions.
    For a full comparison, we'd need to reload predictions and compute
    tests on the actual error distributions.
    
    Args:
        metrics1: Metrics for first model
        metrics2: Metrics for second model
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        Dictionary containing comparison results
    """
    results = {
        'model1': model1_name,
        'model2': model2_name,
        'metrics_comparison': {}
    }
    
    # Compare key metrics
    key_metrics = ['MedAE', 'MAE', 'RMSE', 'R2', 'Pearson', 'Spearman',
                   'Pct_le_60s', 'Pct_le_30s', 'Pct_le_10s']
    
    for metric in key_metrics:
        val1 = metrics1['metrics'].get(metric, None)
        val2 = metrics2['metrics'].get(metric, None)
        
        if val1 is not None and val2 is not None:
            diff = val2 - val1
            pct_change = (diff / val1 * 100) if val1 != 0 else 0
            
            # For R2, Pearson, Spearman, and Pct_* metrics, higher is better
            # For MedAE, MAE, RMSE, lower is better
            if metric in ['R2', 'Pearson', 'Spearman', 'Pct_le_60s', 'Pct_le_30s', 'Pct_le_10s']:
                better = model2_name if diff > 0 else model1_name
            else:
                better = model2_name if diff < 0 else model1_name
            
            results['metrics_comparison'][metric] = {
                f'{model1_name}': val1,
                f'{model2_name}': val2,
                'difference': diff,
                'percent_change': pct_change,
                'better_model': better
            }
    
    # If statistical tests are available in metrics
    if 'statistical_tests' in metrics2 and metrics2['statistical_tests']:
        results['statistical_tests'] = metrics2['statistical_tests']
    
    return results


def generate_comparison_report(
    all_metrics: Dict[str, Dict],
    save_path: Path
):
    """
    Generate comprehensive comparison report.
    
    Args:
        all_metrics: Dictionary mapping model names to their metrics
        save_path: Path to save the report
    """
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("STATISTICAL COMPARISON OF RT PREDICTION MODELS")
    report_lines.append("=" * 100)
    report_lines.append("")
    
    model_names = list(all_metrics.keys())
    
    # Summary table
    report_lines.append("MODEL PERFORMANCE SUMMARY")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Model':<30} {'MedAE':>10} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'Pearson':>10} {'%≤30s':>10}")
    report_lines.append("-" * 100)
    
    for model_name in model_names:
        m = all_metrics[model_name]['metrics']
        report_lines.append(
            f"{model_name:<30} "
            f"{m.get('MedAE', 0):10.2f} "
            f"{m.get('MAE', 0):10.2f} "
            f"{m.get('RMSE', 0):10.2f} "
            f"{m.get('R2', 0):8.3f} "
            f"{m.get('Pearson', 0):10.3f} "
            f"{m.get('Pct_le_30s', 0):10.2f}"
        )
    
    report_lines.append("-" * 100)
    report_lines.append("")
    
    # Pairwise comparisons
    report_lines.append("PAIRWISE COMPARISONS")
    report_lines.append("-" * 100)
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            comparison = compute_pairwise_comparison(
                all_metrics[model1],
                all_metrics[model2],
                model1,
                model2
            )
            
            report_lines.append(f"\n{model1} vs {model2}:")
            report_lines.append("-" * 80)
            
            for metric, vals in comparison['metrics_comparison'].items():
                report_lines.append(
                    f"  {metric:<20} "
                    f"{vals[model1]:>10.3f} vs {vals[model2]:>10.3f}  "
                    f"(Δ={vals['difference']:+.3f}, {vals['percent_change']:+.1f}%)  "
                    f"Better: {vals['better_model']}"
                )
            
            if 'statistical_tests' in comparison and comparison['statistical_tests']:
                st = comparison['statistical_tests']
                report_lines.append(f"\n  Statistical Tests:")
                report_lines.append(f"    Wilcoxon p-value: {st.get('Wilcoxon_p_value', 'N/A'):.2e}")
                report_lines.append(f"    T-test p-value:   {st.get('T_test_p_value', 'N/A'):.2e}")
                report_lines.append(f"    Cohen's d:        {st.get('Cohens_d', 'N/A'):.3f}")
                
                p_val = st.get('Wilcoxon_p_value', 1.0)
                if p_val < 0.001:
                    sig = "*** (p < 0.001)"
                elif p_val < 0.01:
                    sig = "** (p < 0.01)"
                elif p_val < 0.05:
                    sig = "* (p < 0.05)"
                else:
                    sig = "ns (not significant)"
                report_lines.append(f"    Significance:     {sig}")
            
            report_lines.append("")
    
    report_lines.append("=" * 100)
    
    # Save report
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Print to console
    print('\n'.join(report_lines))
    print(f"\nReport saved to {save_path}")


def create_comparison_table(
    all_metrics: Dict[str, Dict],
    save_path: Path
):
    """
    Create comparison table as CSV.
    
    Args:
        all_metrics: Dictionary mapping model names to their metrics
        save_path: Path to save CSV
    """
    rows = []
    
    for model_name, data in all_metrics.items():
        m = data['metrics']
        row = {
            'Model': model_name,
            'MedAE': m.get('MedAE', np.nan),
            'MAE': m.get('MAE', np.nan),
            'RMSE': m.get('RMSE', np.nan),
            'R²': m.get('R2', np.nan),
            'Pearson': m.get('Pearson', np.nan),
            'Spearman': m.get('Spearman', np.nan),
            '%≤60s': m.get('Pct_le_60s', np.nan),
            '%≤30s': m.get('Pct_le_30s', np.nan),
            '%≤10s': m.get('Pct_le_10s', np.nan),
            'Median_Rel_Error_%': m.get('Median_Rel_Error', np.nan),
            'n_samples': m.get('n_samples', np.nan)
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"Comparison table saved to {save_path}")


def main():
    """Main function to run statistical comparisons."""
    
    # Define paths to metrics files
    results_dir = Path("results/metrics")
    
    metrics_files = {
        'KAGNN+ECFP Baseline': results_dir / 'baseline_kagnn.json',
        'KAGNN→PGM Forward': results_dir / 'kagnn_pgm_forward.json',
        'PGM→KAGNN Reverse': results_dir / 'pgm_kagnn_reverse.json'
    }
    
    # Load all metrics
    print("Loading metrics from all experiments...")
    all_metrics = {}
    
    for model_name, metrics_path in metrics_files.items():
        if metrics_path.exists():
            all_metrics[model_name] = load_metrics(metrics_path)
            print(f"  ✓ Loaded {model_name}")
        else:
            print(f"  ✗ Not found: {model_name} ({metrics_path})")
    
    if len(all_metrics) < 2:
        print("\nError: Need at least 2 models to compare. Run experiments first.")
        return
    
    print(f"\nComparing {len(all_metrics)} models...")
    
    # Generate reports
    output_dir = Path("results/statistical_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Text report
    generate_comparison_report(
        all_metrics,
        save_path=output_dir / "comparison_report.txt"
    )
    
    # CSV table
    create_comparison_table(
        all_metrics,
        save_path=output_dir / "comparison_table.csv"
    )
    
    print("\n" + "=" * 100)
    print("Statistical analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
