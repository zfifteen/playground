#!/usr/bin/env python3
"""
Autocorrelation Analysis

ACF, PACF, and Ljung-Box tests for log-gap autocorrelation structure.

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from scipy import stats

# Try to import statsmodels, but provide fallback if not available
try:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def compute_acf_manual(data, nlags=20):
    """
    Manual ACF computation (fallback if statsmodels not available).
    
    Args:
        data: numpy array
        nlags: number of lags to compute
        
    Returns:
        numpy array of ACF values for lags 0 to nlags
    """
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    if var == 0:
        return np.zeros(nlags + 1)
    
    acf_vals = np.zeros(nlags + 1)
    acf_vals[0] = 1.0
    
    for lag in range(1, nlags + 1):
        if lag < n:
            cov = np.mean((data[:-lag] - mean) * (data[lag:] - mean))
            acf_vals[lag] = cov / var
    
    return acf_vals


def compute_ljungbox_manual(data, nlags=20):
    """
    Manual Ljung-Box test computation (fallback).
    
    Q = n(n+2) * sum_{k=1}^{h} (rho_k^2 / (n-k))
    
    Under H0, Q ~ chi-squared(h)
    
    Args:
        data: numpy array
        nlags: number of lags to test
        
    Returns:
        Dictionary with Q statistic and p-value for each lag
    """
    n = len(data)
    acf_vals = compute_acf_manual(data, nlags)
    
    results = []
    for h in range(1, nlags + 1):
        Q = 0
        for k in range(1, h + 1):
            Q += (acf_vals[k] ** 2) / (n - k)
        Q *= n * (n + 2)
        
        p_value = 1 - stats.chi2.cdf(Q, h)
        results.append({
            'lag': h,
            'Q': Q,
            'p_value': p_value
        })
    
    return results


def compute_autocorrelation_analysis(data, nlags=20, max_sample_size=500000):
    """
    Compute ACF, PACF, and Ljung-Box tests.
    
    Tests H-MAIN-C: Log-gap autocorrelation exhibits short-range structure.
    
    Args:
        data: numpy array of log-gaps
        nlags: number of lags to analyze
        max_sample_size: maximum sample size for performance (large datasets will be sampled)
        
    Returns:
        Dictionary with ACF, PACF, and Ljung-Box results
    """
    results = {'nlags': nlags}
    
    # For very large datasets, use systematic sampling to improve performance
    # while maintaining statistical properties for ordered sequences like prime gaps.
    # Systematic sampling preserves the sequential structure better than random sampling.
    if len(data) > max_sample_size:
        # Calculate step size for systematic sampling
        step = len(data) // max_sample_size
        # Use systematic sampling: start from random offset, then take every nth element
        np.random.seed(42)  # Reproducibility
        offset = np.random.randint(0, step) if step > 1 else 0
        indices = np.arange(offset, len(data), step)[:max_sample_size]
        data_sample = data[indices]
        results['sampled'] = True
        results['sample_size'] = len(data_sample)
        results['sample_method'] = 'systematic'
    else:
        data_sample = data
        results['sampled'] = False
        results['sample_size'] = len(data)
    
    if STATSMODELS_AVAILABLE:
        # Use statsmodels
        try:
            acf_vals = acf(data_sample, nlags=nlags, fft=True)
            results['acf'] = acf_vals
        except Exception:
            acf_vals = compute_acf_manual(data_sample, nlags)
            results['acf'] = acf_vals
        
        try:
            # PACF might fail for short series
            pacf_vals = pacf(data_sample, nlags=min(nlags, len(data_sample) // 2 - 1))
            results['pacf'] = pacf_vals
        except Exception:
            results['pacf'] = None
        
        try:
            lb_result = acorr_ljungbox(data_sample, lags=nlags, return_df=True)
            results['ljungbox'] = {
                'Q_stats': lb_result['lb_stat'].values,
                'p_values': lb_result['lb_pvalue'].values,
                'lags': lb_result.index.values
            }
        except Exception:
            lb_manual = compute_ljungbox_manual(data_sample, nlags)
            results['ljungbox'] = {
                'Q_stats': [r['Q'] for r in lb_manual],
                'p_values': [r['p_value'] for r in lb_manual],
                'lags': [r['lag'] for r in lb_manual]
            }
    else:
        # Use manual implementations
        results['acf'] = compute_acf_manual(data_sample, nlags)
        results['pacf'] = None  # PACF is complex to implement manually
        
        lb_manual = compute_ljungbox_manual(data_sample, nlags)
        results['ljungbox'] = {
            'Q_stats': [r['Q'] for r in lb_manual],
            'p_values': [r['p_value'] for r in lb_manual],
            'lags': [r['lag'] for r in lb_manual]
        }
    
    # Analysis
    acf_vals = results['acf']
    lb_pvalues = np.array(results['ljungbox']['p_values'])
    
    # Confidence bounds for ACF (approximate)
    n = len(data_sample)
    conf_bound = 1.96 / np.sqrt(n)
    
    # Check for significant ACF at low lags
    significant_lags = []
    for lag in range(1, min(len(acf_vals), nlags + 1)):
        if abs(acf_vals[lag]) > conf_bound:
            significant_lags.append(lag)
    
    results['conf_bound'] = conf_bound
    results['significant_lags'] = significant_lags
    results['has_short_range_structure'] = len(significant_lags) > 0
    
    # Ljung-Box interpretation
    # F4: All p-values > 0.05 means white noise
    all_p_above_threshold = np.all(lb_pvalues > 0.05)
    any_p_below_threshold = np.any(lb_pvalues < 0.01)
    
    results['ljungbox_all_p_above_005'] = all_p_above_threshold
    results['ljungbox_any_p_below_001'] = any_p_below_threshold
    
    # F4 falsification: autocorrelation is flat at all lags
    results['f4_falsified'] = all_p_above_threshold
    
    return results


def compute_windowed_autocorrelation(data, window_size=10000, nlags=20):
    """
    Compute ACF in non-overlapping windows to check for consistency.
    
    Args:
        data: numpy array of log-gaps
        window_size: size of each window
        nlags: number of lags
        
    Returns:
        List of ACF results for each window
    """
    n = len(data)
    n_windows = n // window_size
    
    window_results = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = data[start:end]
        
        acf_vals = compute_acf_manual(window_data, nlags)
        window_results.append({
            'window': i + 1,
            'start': start,
            'end': end,
            'acf': acf_vals
        })
    
    return window_results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from prime_generator import generate_primes_to_limit, compute_log_gaps
    
    print("Testing autocorrelation analysis...")
    print(f"statsmodels available: {STATSMODELS_AVAILABLE}")
    
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    log_gaps = data['log_gaps']
    
    # Autocorrelation analysis
    acf_results = compute_autocorrelation_analysis(log_gaps, nlags=20)
    
    print(f"\nACF values (lags 1-10):")
    print(f"  {acf_results['acf'][1:11]}")
    print(f"\nConfidence bound: ±{acf_results['conf_bound']:.4f}")
    print(f"Significant lags: {acf_results['significant_lags']}")
    print(f"Has short-range structure: {acf_results['has_short_range_structure']}")
    
    print(f"\nLjung-Box p-values (lags 1-10):")
    print(f"  {acf_results['ljungbox']['p_values'][:10]}")
    print(f"All p > 0.05 (white noise): {acf_results['ljungbox_all_p_above_005']}")
    print(f"F4 falsified: {acf_results['f4_falsified']}")
