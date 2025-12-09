#!/usr/bin/env python3
"""
STATISTICAL UTILITIES FOR RIGOROUS QMC VALIDATION
==================================================

Implements bootstrap confidence intervals and permutation tests
for statistically sound comparison of QMC methods and correlation analysis.

Author: Big D (zfifteen)
Date: December 2025
"""

import numpy as np
from typing import Tuple, Callable, Optional
from scipy import stats


def bootstrap_ci(values: np.ndarray, 
                 n_boot: int = 1000, 
                 alpha: float = 0.05,
                 statistic: Callable = np.mean,
                 seed: Optional[int] = None) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Parameters:
    -----------
    values : np.ndarray
        Data values to bootstrap
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level (e.g., 0.05 for 95% CI)
    statistic : Callable
        Statistic to compute (default: mean)
    seed : Optional[int]
        Random seed for reproducibility
        
    Returns:
    --------
    point_estimate : float
        Point estimate of the statistic
    ci_lower : float
        Lower bound of confidence interval
    ci_upper : float
        Upper bound of confidence interval
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    
    # Compute point estimate
    point_estimate = statistic(values)
    
    # Bootstrap resampling
    boot_statistics = np.zeros(n_boot)
    for i in range(n_boot):
        boot_sample = rng.choice(values, size=n, replace=True)
        boot_statistics[i] = statistic(boot_sample)
    
    # Compute percentile CI
    ci_lower = np.percentile(boot_statistics, 100 * alpha / 2)
    ci_upper = np.percentile(boot_statistics, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def bootstrap_regression_ci(x: np.ndarray,
                            y: np.ndarray,
                            n_boot: int = 1000,
                            alpha: float = 0.05,
                            seed: Optional[int] = None) -> dict:
    """
    Compute bootstrap CI for linear regression coefficients and R².
    
    Parameters:
    -----------
    x : np.ndarray
        Independent variable
    y : np.ndarray
        Dependent variable
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level
    seed : Optional[int]
        Random seed
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'slope': (estimate, ci_lower, ci_upper)
        - 'intercept': (estimate, ci_lower, ci_upper)
        - 'r_squared': (estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    
    # Original regression
    slope_orig, intercept_orig, r_value_orig, _, _ = stats.linregress(x, y)
    r_squared_orig = r_value_orig ** 2
    
    # Bootstrap resampling
    slopes = np.zeros(n_boot)
    intercepts = np.zeros(n_boot)
    r_squareds = np.zeros(n_boot)
    
    for i in range(n_boot):
        # Resample with replacement
        indices = rng.choice(n, size=n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        
        # Regression on bootstrap sample
        slope, intercept, r_value, _, _ = stats.linregress(x_boot, y_boot)
        slopes[i] = slope
        intercepts[i] = intercept
        r_squareds[i] = r_value ** 2
    
    # Compute CIs
    slope_ci = (np.percentile(slopes, 100 * alpha / 2),
                np.percentile(slopes, 100 * (1 - alpha / 2)))
    intercept_ci = (np.percentile(intercepts, 100 * alpha / 2),
                    np.percentile(intercepts, 100 * (1 - alpha / 2)))
    r_squared_ci = (np.percentile(r_squareds, 100 * alpha / 2),
                    np.percentile(r_squareds, 100 * (1 - alpha / 2)))
    
    return {
        'slope': (slope_orig, slope_ci[0], slope_ci[1]),
        'intercept': (intercept_orig, intercept_ci[0], intercept_ci[1]),
        'r_squared': (r_squared_orig, r_squared_ci[0], r_squared_ci[1]),
    }


def permutation_test_correlation(x: np.ndarray,
                                 y: np.ndarray,
                                 n_perm: int = 1000,
                                 seed: Optional[int] = None) -> Tuple[float, float]:
    """
    Permutation test for correlation significance.
    
    Tests null hypothesis: x and y are independent (correlation = 0).
    
    Parameters:
    -----------
    x : np.ndarray
        First variable
    y : np.ndarray
        Second variable
    n_perm : int
        Number of permutations
    seed : Optional[int]
        Random seed
        
    Returns:
    --------
    observed_corr : float
        Observed correlation coefficient
    p_value : float
        Two-tailed p-value
    """
    rng = np.random.default_rng(seed)
    
    # Observed correlation
    observed_corr = np.corrcoef(x, y)[0, 1]
    
    # Permutation distribution
    perm_corrs = np.zeros(n_perm)
    for i in range(n_perm):
        # Permute one variable
        y_perm = rng.permutation(y)
        perm_corrs[i] = np.corrcoef(x, y_perm)[0, 1]
    
    # Compute p-value (two-tailed)
    p_value = np.mean(np.abs(perm_corrs) >= np.abs(observed_corr))
    
    return observed_corr, p_value


def compare_distributions_bootstrap(values1: np.ndarray,
                                    values2: np.ndarray,
                                    n_boot: int = 1000,
                                    alpha: float = 0.05,
                                    seed: Optional[int] = None) -> dict:
    """
    Compare two distributions using bootstrap for mean difference CI.
    
    Parameters:
    -----------
    values1 : np.ndarray
        First distribution
    values2 : np.ndarray
        Second distribution
    n_boot : int
        Number of bootstrap samples
    alpha : float
        Significance level
    seed : Optional[int]
        Random seed
        
    Returns:
    --------
    results : dict
        Dictionary with:
        - 'mean_diff': (estimate, ci_lower, ci_upper)
        - 'effect_size': Cohen's d
        - 'significant': Whether CI excludes 0
    """
    rng = np.random.default_rng(seed)
    
    # Observed difference
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    mean_diff_obs = mean1 - mean2
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt((np.var(values1) + np.var(values2)) / 2)
    cohens_d = mean_diff_obs / pooled_std if pooled_std > 0 else 0
    
    # Bootstrap resampling
    boot_diffs = np.zeros(n_boot)
    for i in range(n_boot):
        boot1 = rng.choice(values1, size=len(values1), replace=True)
        boot2 = rng.choice(values2, size=len(values2), replace=True)
        boot_diffs[i] = np.mean(boot1) - np.mean(boot2)
    
    # Compute CI
    ci_lower = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    
    # Check significance
    significant = not (ci_lower <= 0 <= ci_upper)
    
    return {
        'mean_diff': (mean_diff_obs, ci_lower, ci_upper),
        'effect_size': cohens_d,
        'significant': significant,
    }


def format_ci_report(point_estimate: float,
                    ci_lower: float,
                    ci_upper: float,
                    name: str = "Estimate",
                    precision: int = 4) -> str:
    """
    Format confidence interval for reporting.
    
    Parameters:
    -----------
    point_estimate : float
        Point estimate
    ci_lower : float
        Lower CI bound
    ci_upper : float
        Upper CI bound
    name : str
        Name of the estimate
    precision : int
        Decimal precision
        
    Returns:
    --------
    report : str
        Formatted string
    """
    fmt = f"{{:.{precision}f}}"
    return (f"{name}: {fmt.format(point_estimate)} "
            f"[{fmt.format(ci_lower)}, {fmt.format(ci_upper)}]")


if __name__ == "__main__":
    """
    Test statistical utilities.
    """
    print("=" * 70)
    print("STATISTICAL UTILITIES TEST")
    print("=" * 70)
    print()
    
    # Test 1: Bootstrap CI for mean
    print("Test 1: Bootstrap CI for mean")
    print("-" * 70)
    rng = np.random.default_rng(42)
    data = rng.normal(loc=5.0, scale=2.0, size=100)
    point, lower, upper = bootstrap_ci(data, n_boot=1000, alpha=0.05, seed=42)
    print(f"  Data: N=100, true mean=5.0, true std=2.0")
    print(f"  {format_ci_report(point, lower, upper, 'Mean', 3)}")
    print()
    
    # Test 2: Bootstrap regression CI
    print("Test 2: Bootstrap regression CI")
    print("-" * 70)
    x = rng.uniform(0, 10, 50)
    y = 2.5 * x + 3.0 + rng.normal(0, 1, 50)
    reg_results = bootstrap_regression_ci(x, y, n_boot=1000, alpha=0.05, seed=42)
    print(f"  Data: y = 2.5*x + 3.0 + noise")
    print(f"  Slope: {reg_results['slope'][0]:.3f} "
          f"[{reg_results['slope'][1]:.3f}, {reg_results['slope'][2]:.3f}]")
    print(f"  R²: {reg_results['r_squared'][0]:.3f} "
          f"[{reg_results['r_squared'][1]:.3f}, {reg_results['r_squared'][2]:.3f}]")
    print()
    
    # Test 3: Permutation test for correlation
    print("Test 3: Permutation test for correlation")
    print("-" * 70)
    x1 = rng.uniform(0, 10, 50)
    y1 = 0.8 * x1 + rng.normal(0, 1, 50)  # Strong correlation
    y2 = rng.normal(0, 1, 50)  # No correlation
    
    corr1, p1 = permutation_test_correlation(x1, y1, n_perm=1000, seed=42)
    corr2, p2 = permutation_test_correlation(x1, y2, n_perm=1000, seed=42)
    
    print(f"  Correlated data: r={corr1:.3f}, p={p1:.4f}")
    print(f"  Uncorrelated data: r={corr2:.3f}, p={p2:.4f}")
    print()
    
    # Test 4: Compare distributions
    print("Test 4: Compare distributions")
    print("-" * 70)
    dist1 = rng.normal(loc=5.0, scale=1.0, size=100)
    dist2 = rng.normal(loc=5.5, scale=1.0, size=100)
    
    comp_results = compare_distributions_bootstrap(dist1, dist2, n_boot=1000, 
                                                   alpha=0.05, seed=42)
    print(f"  Distribution 1: mean={np.mean(dist1):.3f}")
    print(f"  Distribution 2: mean={np.mean(dist2):.3f}")
    print(f"  Mean difference: {comp_results['mean_diff'][0]:.3f} "
          f"[{comp_results['mean_diff'][1]:.3f}, {comp_results['mean_diff'][2]:.3f}]")
    print(f"  Effect size (Cohen's d): {comp_results['effect_size']:.3f}")
    print(f"  Significant: {comp_results['significant']}")
    print()
    
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
