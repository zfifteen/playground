#!/usr/bin/env python3
"""
Visualization Module

Q-Q plots, histograms, trend plots, and ACF/PACF plots.

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def plot_log_gap_histogram(log_gaps, output_path=None, title_suffix=""):
    """
    Plot histogram of log-gaps with fitted distributions.
    
    Args:
        log_gaps: numpy array of log-gaps
        output_path: path to save figure (optional)
        title_suffix: additional title text
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Regular histogram
    ax1 = axes[0]
    ax1.hist(log_gaps, bins=100, density=True, alpha=0.7, color='steelblue', 
             edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Log-Gap: ln(p_{n+1}/p_n)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title(f'Log-Gap Distribution {title_suffix}', fontsize=12)
    
    # Fit and overlay log-normal
    log_gaps_pos = log_gaps[log_gaps > 0]
    if len(log_gaps_pos) > 0:
        shape, loc, scale = stats.lognorm.fit(log_gaps_pos, floc=0)
        x = np.linspace(0.0001, np.percentile(log_gaps, 99), 500)
        pdf = stats.lognorm.pdf(x, shape, loc, scale)
        ax1.plot(x, pdf, 'r-', linewidth=2, label=f'Log-normal fit')
        ax1.legend()
    
    ax1.grid(True, alpha=0.3)
    
    # Right: Log-scale histogram (to see tail behavior)
    ax2 = axes[1]
    log_gaps_pos = log_gaps[log_gaps > 0]
    if len(log_gaps_pos) > 0:
        ax2.hist(np.log10(log_gaps_pos), bins=100, density=True, alpha=0.7, 
                 color='coral', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('log₁₀(Log-Gap)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'Log-Transformed Log-Gap Distribution {title_suffix}', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_qq_lognormal(log_gaps, output_path=None, title_suffix=""):
    """
    Q-Q plot comparing log-gaps to log-normal distribution.
    
    Args:
        log_gaps: numpy array of log-gaps
        output_path: path to save figure (optional)
        title_suffix: additional title text
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    log_gaps_pos = log_gaps[log_gaps > 0]
    
    # Left: Q-Q plot for log-normal
    ax1 = axes[0]
    if len(log_gaps_pos) > 0:
        # Log-normal Q-Q is equivalent to normal Q-Q on log-transformed data
        log_data = np.log(log_gaps_pos)
        stats.probplot(log_data, dist="norm", plot=ax1)
        ax1.set_title(f'Q-Q Plot: Log(Log-Gap) vs Normal {title_suffix}', fontsize=12)
        ax1.get_lines()[0].set_markersize(2)
        ax1.get_lines()[0].set_alpha(0.5)
    ax1.grid(True, alpha=0.3)
    
    # Right: Q-Q plot for exponential (for comparison)
    ax2 = axes[1]
    stats.probplot(log_gaps, dist="expon", plot=ax2)
    ax2.set_title(f'Q-Q Plot: Log-Gap vs Exponential {title_suffix}', fontsize=12)
    ax2.get_lines()[0].set_markersize(2)
    ax2.get_lines()[0].set_alpha(0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_decay_trend(quintile_result, output_path=None, title_suffix=""):
    """
    Plot quintile/decile mean decay trend.
    
    Args:
        quintile_result: output from compute_quintile_analysis
        output_path: path to save figure (optional)
        title_suffix: additional title text
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n_bins = quintile_result['n_bins']
    bin_indices = quintile_result['bin_indices']
    bin_means = quintile_result['bin_means']
    bin_stds = quintile_result['bin_stds']
    
    # Scatter with error bars
    ax.errorbar(bin_indices, bin_means, yerr=bin_stds, fmt='o', 
                color='steelblue', markersize=10, capsize=5,
                label='Bin means ± std')
    
    # Regression line
    slope = quintile_result['slope']
    intercept = quintile_result['intercept']
    r_squared = quintile_result['r_squared']
    
    x_fit = np.linspace(bin_indices[0], bin_indices[-1], 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r--', linewidth=2, 
            label=f'Linear fit (R²={r_squared:.4f})')
    
    ax.set_xlabel(f'{"Quintile" if n_bins == 5 else "Decile"} Index', fontsize=11)
    ax.set_ylabel('Mean Log-Gap', fontsize=11)
    ax.set_title(f'Log-Gap Decay Trend {title_suffix}\nSlope={slope:.2e}, p={quintile_result["p_value"]:.2e}', 
                 fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks to integers
    ax.set_xticks(bin_indices)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_acf_pacf(acf_result, output_path=None, title_suffix=""):
    """
    Plot ACF and PACF.
    
    Args:
        acf_result: output from compute_autocorrelation_analysis
        output_path: path to save figure (optional)
        title_suffix: additional title text
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    nlags = acf_result['nlags']
    acf_vals = acf_result['acf']
    conf_bound = acf_result['conf_bound']
    
    # Left: ACF
    ax1 = axes[0]
    lags = np.arange(0, len(acf_vals))
    ax1.bar(lags[1:], acf_vals[1:], color='steelblue', alpha=0.7)
    ax1.axhline(y=conf_bound, color='red', linestyle='--', 
                label=f'95% CI (±{conf_bound:.4f})')
    ax1.axhline(y=-conf_bound, color='red', linestyle='--')
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_xlabel('Lag', fontsize=11)
    ax1.set_ylabel('ACF', fontsize=11)
    ax1.set_title(f'Autocorrelation Function (ACF) {title_suffix}', fontsize=12)
    ax1.legend()
    ax1.set_xlim(-0.5, nlags + 0.5)
    ax1.grid(True, alpha=0.3)
    
    # Right: PACF (if available)
    ax2 = axes[1]
    if acf_result['pacf'] is not None:
        pacf_vals = acf_result['pacf']
        lags_pacf = np.arange(0, len(pacf_vals))
        ax2.bar(lags_pacf[1:], pacf_vals[1:], color='coral', alpha=0.7)
        ax2.axhline(y=conf_bound, color='red', linestyle='--', 
                    label=f'95% CI (±{conf_bound:.4f})')
        ax2.axhline(y=-conf_bound, color='red', linestyle='--')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xlabel('Lag', fontsize=11)
        ax2.set_ylabel('PACF', fontsize=11)
        ax2.set_title(f'Partial Autocorrelation Function (PACF) {title_suffix}', fontsize=12)
        ax2.legend()
        ax2.set_xlim(-0.5, len(pacf_vals) + 0.5)
    else:
        # Plot Ljung-Box p-values instead
        lb_lags = acf_result['ljungbox']['lags']
        lb_pvals = acf_result['ljungbox']['p_values']
        ax2.bar(lb_lags, lb_pvals, color='coral', alpha=0.7)
        ax2.axhline(y=0.05, color='red', linestyle='--', label='α=0.05')
        ax2.axhline(y=0.01, color='orange', linestyle='--', label='α=0.01')
        ax2.set_xlabel('Lag', fontsize=11)
        ax2.set_ylabel('p-value', fontsize=11)
        ax2.set_title(f'Ljung-Box Test p-values {title_suffix}', fontsize=12)
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


def plot_scale_comparison(scale_results, output_path=None):
    """
    Plot comparison of results across scales.
    
    Args:
        scale_results: dictionary mapping scale to analysis results
        output_path: path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    scales = sorted(scale_results.keys())
    
    # Top-left: Mean log-gap vs scale
    ax1 = axes[0, 0]
    means = [scale_results[s]['descriptive']['mean'] for s in scales]
    ax1.plot(np.log10(scales), means, 'o-', markersize=10, color='steelblue')
    ax1.set_xlabel('log₁₀(Scale)', fontsize=11)
    ax1.set_ylabel('Mean Log-Gap', fontsize=11)
    ax1.set_title('Mean Log-Gap vs Scale', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Top-right: Decay ratio vs scale
    ax2 = axes[0, 1]
    decay_ratios = [scale_results[s]['quintile']['decay_ratio'] for s in scales]
    ax2.plot(np.log10(scales), decay_ratios, 's-', markersize=10, color='coral')
    ax2.set_xlabel('log₁₀(Scale)', fontsize=11)
    ax2.set_ylabel('Decay Ratio (Q1/Q5)', fontsize=11)
    ax2.set_title('Decay Ratio vs Scale', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Bottom-left: Regression slope vs scale
    ax3 = axes[1, 0]
    slopes = [scale_results[s]['quintile']['slope'] for s in scales]
    ax3.plot(np.log10(scales), slopes, '^-', markersize=10, color='green')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('log₁₀(Scale)', fontsize=11)
    ax3.set_ylabel('Regression Slope', fontsize=11)
    ax3.set_title('Quintile Regression Slope vs Scale', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Bottom-right: KS statistic (log-normal) vs scale
    ax4 = axes[1, 1]
    ks_ln = [scale_results[s]['distribution']['lognormal']['ks_statistic'] 
             for s in scales if 'lognormal' in scale_results[s]['distribution']]
    ks_norm = [scale_results[s]['distribution']['normal']['ks_statistic'] 
               for s in scales if 'normal' in scale_results[s]['distribution']]
    scales_filtered = [s for s in scales if 'lognormal' in scale_results[s]['distribution']]
    
    ax4.plot(np.log10(scales_filtered), ks_ln, 'o-', label='Log-normal', markersize=8)
    ax4.plot(np.log10(scales_filtered), ks_norm, 's-', label='Normal', markersize=8)
    ax4.set_xlabel('log₁₀(Scale)', fontsize=11)
    ax4.set_ylabel('KS Statistic', fontsize=11)
    ax4.set_title('Distribution Fit Quality vs Scale', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    plt.close()
    return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from prime_generator import generate_primes_to_limit, compute_log_gaps
    from log_gap_analysis import compute_quintile_analysis
    from autocorrelation import compute_autocorrelation_analysis
    
    print("Testing visualization...")
    
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    log_gaps = data['log_gaps']
    
    # Create output directory
    os.makedirs('../results/figures', exist_ok=True)
    
    # Test plots
    plot_log_gap_histogram(log_gaps, '../results/figures/test_histogram.png', '(10^5)')
    plot_qq_lognormal(log_gaps, '../results/figures/test_qq.png', '(10^5)')
    
    quintile = compute_quintile_analysis(log_gaps)
    plot_decay_trend(quintile, '../results/figures/test_decay.png', '(10^5)')
    
    acf_result = compute_autocorrelation_analysis(log_gaps)
    plot_acf_pacf(acf_result, '../results/figures/test_acf.png', '(10^5)')
    
    print("All test plots generated.")
