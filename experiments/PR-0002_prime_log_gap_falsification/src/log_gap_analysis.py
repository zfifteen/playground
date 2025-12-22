#!/usr/bin/env python3
"""
Log-Gap Statistical Analysis

Core statistical analysis for quintile/decile regression and decay analysis.

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from scipy import stats
import pandas as pd


def compute_quintile_analysis(log_gaps, n_bins=5):
    """
    Compute quintile (or n-tile) means of log-gaps.
    
    Tests H-MAIN-A: Mean log-gap decreases monotonically as primes increase.
    
    Args:
        log_gaps: numpy array of log-gaps
        n_bins: number of bins (5 for quintiles, 10 for deciles)
        
    Returns:
        Dictionary with analysis results
    """
    n = len(log_gaps)
    bin_size = n // n_bins
    
    bin_means = []
    bin_stds = []
    bin_indices = []
    
    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        bin_data = log_gaps[start:end]
        bin_means.append(np.mean(bin_data))
        bin_stds.append(np.std(bin_data))
        bin_indices.append(i + 1)
    
    bin_means = np.array(bin_means)
    bin_stds = np.array(bin_stds)
    bin_indices = np.array(bin_indices)
    
    # Linear regression: bin_means vs bin_indices
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        bin_indices, bin_means
    )
    
    # Check monotonic decrease
    differences = np.diff(bin_means)
    is_monotonic_decreasing = np.all(differences < 0)
    
    # Check if slope is significantly negative
    is_significantly_negative = (slope < 0) and (p_value < 0.001)
    
    return {
        'n_bins': n_bins,
        'bin_indices': bin_indices,
        'bin_means': bin_means,
        'bin_stds': bin_stds,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
        'is_monotonic_decreasing': is_monotonic_decreasing,
        'is_significantly_negative': is_significantly_negative,
        'decay_ratio': bin_means[0] / bin_means[-1] if bin_means[-1] != 0 else np.inf,
        # Falsification: F1 triggers if slope >= 0 with p > 0.05
        'f1_falsified': slope >= 0 and p_value > 0.05
    }


def compute_decile_analysis(log_gaps):
    """
    Compute decile analysis (10 bins) for finer granularity.
    """
    return compute_quintile_analysis(log_gaps, n_bins=10)


def compute_descriptive_stats(log_gaps):
    """
    Compute descriptive statistics for log-gaps.
    
    Args:
        log_gaps: numpy array of log-gaps
        
    Returns:
        Dictionary with descriptive statistics
    """
    return {
        'count': len(log_gaps),
        'mean': np.mean(log_gaps),
        'std': np.std(log_gaps),
        'min': np.min(log_gaps),
        'max': np.max(log_gaps),
        'median': np.median(log_gaps),
        'q1': np.percentile(log_gaps, 25),
        'q3': np.percentile(log_gaps, 75),
        'iqr': np.percentile(log_gaps, 75) - np.percentile(log_gaps, 25),
        'skewness': stats.skew(log_gaps),
        'kurtosis': stats.kurtosis(log_gaps),  # Excess kurtosis
        'range': np.max(log_gaps) - np.min(log_gaps),
    }


def compute_scale_comparison(results_by_scale):
    """
    Compare results across different scales (10^5, 10^6, 10^7, 10^8).
    
    Tests for scale consistency: Results should be consistent across scales.
    
    Args:
        results_by_scale: Dictionary mapping scale to analysis results
        
    Returns:
        Dictionary with comparison results
    """
    scales = sorted(results_by_scale.keys())
    
    # Extract key metrics across scales
    decay_ratios = []
    slopes = []
    mean_log_gaps = []
    
    for scale in scales:
        result = results_by_scale[scale]
        if 'quintile' in result:
            decay_ratios.append(result['quintile']['decay_ratio'])
            slopes.append(result['quintile']['slope'])
        if 'descriptive' in result:
            mean_log_gaps.append(result['descriptive']['mean'])
    
    # Check directional consistency (all negative slopes)
    slopes_array = np.array(slopes)
    directional_consistent = np.all(slopes_array < 0)
    
    # Check if decay ratios are all > 1 (indicating decay)
    decay_ratios_array = np.array(decay_ratios)
    decay_consistent = np.all(decay_ratios_array > 1)
    
    return {
        'scales': scales,
        'decay_ratios': decay_ratios,
        'slopes': slopes,
        'mean_log_gaps': mean_log_gaps,
        'directional_consistent': directional_consistent,
        'decay_consistent': decay_consistent,
        # F6 falsification: Scale-dependent reversals
        'f6_falsified': not directional_consistent
    }


def generate_summary_dataframe(results):
    """
    Generate a summary DataFrame from analysis results.
    
    Args:
        results: Dictionary with analysis results
        
    Returns:
        pandas DataFrame
    """
    rows = []
    
    if 'descriptive' in results:
        desc = results['descriptive']
        rows.append({'Metric': 'Count', 'Value': desc['count']})
        rows.append({'Metric': 'Mean', 'Value': f"{desc['mean']:.6f}"})
        rows.append({'Metric': 'Std Dev', 'Value': f"{desc['std']:.6f}"})
        rows.append({'Metric': 'Min', 'Value': f"{desc['min']:.6f}"})
        rows.append({'Metric': 'Max', 'Value': f"{desc['max']:.6f}"})
        rows.append({'Metric': 'Median', 'Value': f"{desc['median']:.6f}"})
        rows.append({'Metric': 'Skewness', 'Value': f"{desc['skewness']:.4f}"})
        rows.append({'Metric': 'Excess Kurtosis', 'Value': f"{desc['kurtosis']:.4f}"})
    
    if 'quintile' in results:
        q = results['quintile']
        rows.append({'Metric': 'Q1 Mean', 'Value': f"{q['bin_means'][0]:.6f}"})
        rows.append({'Metric': 'Q5 Mean', 'Value': f"{q['bin_means'][-1]:.6f}"})
        rows.append({'Metric': 'Decay Ratio (Q1/Q5)', 'Value': f"{q['decay_ratio']:.2f}"})
        rows.append({'Metric': 'Regression Slope', 'Value': f"{q['slope']:.6e}"})
        rows.append({'Metric': 'R²', 'Value': f"{q['r_squared']:.4f}"})
        rows.append({'Metric': 'p-value', 'Value': f"{q['p_value']:.4e}"})
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '.')
    from prime_generator import generate_primes_to_limit, compute_log_gaps
    
    print("Testing log-gap analysis...")
    
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    log_gaps = data['log_gaps']
    
    # Descriptive stats
    desc = compute_descriptive_stats(log_gaps)
    print("\nDescriptive Statistics:")
    for k, v in desc.items():
        print(f"  {k}: {v}")
    
    # Quintile analysis
    quintile = compute_quintile_analysis(log_gaps)
    print("\nQuintile Analysis:")
    print(f"  Bin means: {quintile['bin_means']}")
    print(f"  Slope: {quintile['slope']:.6e}")
    print(f"  R²: {quintile['r_squared']:.4f}")
    print(f"  p-value: {quintile['p_value']:.4e}")
    print(f"  Decay ratio: {quintile['decay_ratio']:.2f}")
    print(f"  Monotonic decreasing: {quintile['is_monotonic_decreasing']}")
