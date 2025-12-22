#!/usr/bin/env python3
"""
Distribution Tests and Fitting

Kolmogorov-Smirnov tests and MLE fitting for various distributions.

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from scipy import stats


def fit_distributions(data):
    """
    Fit multiple distributions to the data and compare using KS test.
    
    Tests H-MAIN-B: Log-gaps follow a log-normal or related heavy-tailed distribution.
    
    Args:
        data: numpy array of log-gaps
        
    Returns:
        Dictionary with fitting results for each distribution
    """
    results = {}
    
    # Remove any non-positive values for log-normal fit
    data_positive = data[data > 0]
    
    # 1. Normal distribution
    try:
        mu_norm, std_norm = stats.norm.fit(data)
        ks_stat_norm, p_norm = stats.kstest(data, 'norm', args=(mu_norm, std_norm))
        results['normal'] = {
            'params': {'mu': mu_norm, 'sigma': std_norm},
            'ks_statistic': ks_stat_norm,
            'p_value': p_norm
        }
    except Exception as e:
        results['normal'] = {'error': str(e)}
    
    # 2. Log-normal distribution
    if len(data_positive) > 0:
        try:
            # scipy lognorm uses shape, loc, scale parameterization
            shape, loc, scale = stats.lognorm.fit(data_positive, floc=0)
            ks_stat_ln, p_ln = stats.kstest(
                data_positive, 'lognorm', args=(shape, loc, scale)
            )
            # Convert to standard log-normal params
            mu_ln = np.log(scale)
            sigma_ln = shape
            results['lognormal'] = {
                'params': {'mu': mu_ln, 'sigma': sigma_ln, 'shape': shape, 'scale': scale},
                'ks_statistic': ks_stat_ln,
                'p_value': p_ln
            }
        except Exception as e:
            results['lognormal'] = {'error': str(e)}
    
    # 3. Exponential distribution
    try:
        loc_exp, scale_exp = stats.expon.fit(data)
        ks_stat_exp, p_exp = stats.kstest(data, 'expon', args=(loc_exp, scale_exp))
        results['exponential'] = {
            'params': {'loc': loc_exp, 'scale': scale_exp},
            'ks_statistic': ks_stat_exp,
            'p_value': p_exp
        }
    except Exception as e:
        results['exponential'] = {'error': str(e)}
    
    # 4. Gamma distribution
    if len(data_positive) > 0:
        try:
            a_gamma, loc_gamma, scale_gamma = stats.gamma.fit(data_positive, floc=0)
            ks_stat_gamma, p_gamma = stats.kstest(
                data_positive, 'gamma', args=(a_gamma, loc_gamma, scale_gamma)
            )
            results['gamma'] = {
                'params': {'a': a_gamma, 'loc': loc_gamma, 'scale': scale_gamma},
                'ks_statistic': ks_stat_gamma,
                'p_value': p_gamma
            }
        except Exception as e:
            results['gamma'] = {'error': str(e)}
    
    # 5. Weibull distribution
    if len(data_positive) > 0:
        try:
            c_weibull, loc_weibull, scale_weibull = stats.weibull_min.fit(
                data_positive, floc=0
            )
            ks_stat_weibull, p_weibull = stats.kstest(
                data_positive, 'weibull_min', 
                args=(c_weibull, loc_weibull, scale_weibull)
            )
            results['weibull'] = {
                'params': {'c': c_weibull, 'loc': loc_weibull, 'scale': scale_weibull},
                'ks_statistic': ks_stat_weibull,
                'p_value': p_weibull
            }
        except Exception as e:
            results['weibull'] = {'error': str(e)}
    
    return results


def compare_distributions(fit_results):
    """
    Compare distribution fits and determine best fit.
    
    Args:
        fit_results: Output from fit_distributions
        
    Returns:
        Dictionary with comparison results
    """
    # Filter valid results
    valid_fits = {
        name: res for name, res in fit_results.items() 
        if 'error' not in res
    }
    
    if not valid_fits:
        return {'error': 'No valid distribution fits'}
    
    # Sort by KS statistic (lower is better)
    sorted_fits = sorted(
        valid_fits.items(), 
        key=lambda x: x[1]['ks_statistic']
    )
    
    best_fit_name, best_fit_data = sorted_fits[0]
    
    # Check falsification criteria
    # F2: Normal fits better than log-normal
    normal_ks = fit_results.get('normal', {}).get('ks_statistic', float('inf'))
    lognormal_ks = fit_results.get('lognormal', {}).get('ks_statistic', float('inf'))
    f2_falsified = normal_ks < lognormal_ks
    
    # Get exponential KS for comparison
    exp_ks = fit_results.get('exponential', {}).get('ks_statistic', float('inf'))
    
    # Compare normal vs log-normal with ratio threshold
    ks_ratio = None
    if normal_ks > 0 and lognormal_ks > 0:
        ks_ratio = normal_ks / lognormal_ks
    
    return {
        'best_fit': best_fit_name,
        'best_ks': best_fit_data['ks_statistic'],
        'rankings': [(name, data['ks_statistic']) for name, data in sorted_fits],
        'normal_ks': normal_ks,
        'lognormal_ks': lognormal_ks,
        'exponential_ks': exp_ks,
        'ks_ratio_normal_lognormal': ks_ratio,
        'f2_falsified': f2_falsified,
        'lognormal_is_best_or_second': best_fit_name in ['lognormal', 'gamma', 'weibull']
    }


def compute_skewness_kurtosis_check(data):
    """
    Check if skewness and kurtosis are consistent with normal distribution.
    
    Tests F5: |skewness| < 0.5, |excess kurtosis| < 1 suggests normality.
    
    Args:
        data: numpy array of log-gaps
        
    Returns:
        Dictionary with results
    """
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # Excess kurtosis
    
    # Normal distribution: skewness=0, excess kurtosis=0
    is_normal_like = abs(skewness) < 0.5 and abs(kurtosis) < 1
    
    # Heavy-tailed: high positive skewness and kurtosis
    is_heavy_tailed = skewness > 1 and kurtosis > 3
    
    return {
        'skewness': skewness,
        'excess_kurtosis': kurtosis,
        'is_normal_like': is_normal_like,
        'is_heavy_tailed': is_heavy_tailed,
        # F5 falsification
        'f5_falsified': is_normal_like
    }


def compute_qq_data(data, distribution='lognorm'):
    """
    Compute Q-Q plot data for a given distribution.
    
    Args:
        data: numpy array
        distribution: 'norm', 'lognorm', 'expon'
        
    Returns:
        Dictionary with theoretical and sample quantiles
    """
    data_sorted = np.sort(data)
    n = len(data_sorted)
    
    # Theoretical quantiles
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    
    if distribution == 'norm':
        theoretical = stats.norm.ppf(probabilities)
    elif distribution == 'lognorm':
        # Fit log-normal first
        data_positive = data[data > 0]
        if len(data_positive) > 0:
            shape, loc, scale = stats.lognorm.fit(data_positive, floc=0)
            theoretical = stats.lognorm.ppf(probabilities, shape, loc, scale)
        else:
            theoretical = np.zeros(n)
    elif distribution == 'expon':
        loc, scale = stats.expon.fit(data)
        theoretical = stats.expon.ppf(probabilities, loc, scale)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return {
        'sample_quantiles': data_sorted,
        'theoretical_quantiles': theoretical,
        'distribution': distribution
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from prime_generator import generate_primes_to_limit, compute_log_gaps
    
    print("Testing distribution fitting...")
    
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    log_gaps = data['log_gaps']
    
    # Fit distributions
    fit_results = fit_distributions(log_gaps)
    print("\nDistribution Fits:")
    for name, result in fit_results.items():
        if 'error' in result:
            print(f"  {name}: ERROR - {result['error']}")
        else:
            print(f"  {name}: KS={result['ks_statistic']:.4f}, p={result['p_value']:.4e}")
    
    # Compare
    comparison = compare_distributions(fit_results)
    print(f"\nBest fit: {comparison['best_fit']} (KS={comparison['best_ks']:.4f})")
    print(f"Normal vs Log-normal ratio: {comparison['ks_ratio_normal_lognormal']:.2f}")
    print(f"F2 falsified (normal better): {comparison['f2_falsified']}")
    
    # Skewness/Kurtosis
    sk_check = compute_skewness_kurtosis_check(log_gaps)
    print(f"\nSkewness: {sk_check['skewness']:.4f}")
    print(f"Excess Kurtosis: {sk_check['excess_kurtosis']:.4f}")
    print(f"Heavy-tailed: {sk_check['is_heavy_tailed']}")
    print(f"F5 falsified (normal-like): {sk_check['f5_falsified']}")
