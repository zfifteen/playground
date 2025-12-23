"""
Distribution Testing Module

Tests H-MAIN-B: Lognormal Gap Distribution
Fits various distributions to log(gap) within magnitude bands.
"""

import numpy as np
from typing import Dict, Tuple
from scipy import stats


def compute_cohens_d(ks_stat1: float, ks_stat2: float, n: int) -> float:
    """Compute Cohen's d effect size for comparing two KS statistics.
    
    Cohen's d measures the standardized difference between two means.
    For KS statistics, we compute the effect size as the difference in 
    KS statistics normalized by the pooled standard error.
    
    Args:
        ks_stat1: KS statistic for first distribution (e.g., lognormal)
        ks_stat2: KS statistic for second distribution (e.g., exponential)
        n: Sample size
        
    Returns:
        Cohen's d effect size. Interpretation:
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large
    """
    # Standard error of KS statistic is approximately sqrt(1/(2n))
    # For comparing two KS statistics, we use pooled SE
    se_ks = np.sqrt(1.0 / (2 * n))
    
    # Cohen's d = (mean1 - mean2) / pooled_sd
    # Here we use the difference in KS statistics
    cohens_d = (ks_stat2 - ks_stat1) / se_ks
    
    return cohens_d


def test_distributions_in_band(gaps: np.ndarray, band_name: str) -> Dict:
    """Test distribution fits for gaps within a magnitude band.
    
    Tests whether log(gap) fits normal distribution (lognormal hypothesis)
    vs exponential, gamma, and Weibull alternatives.
    
    Args:
        gaps: Array of gap values within the band
        band_name: Name of the band for reporting
        
    Returns:
        Dictionary with fit statistics for each distribution
    """
    if len(gaps) < 10:
        return {'error': 'Insufficient data'}
    
    log_gaps = np.log(gaps)
    
    results = {}
    
    # Test 1: Normal fit to log(gaps) - implies lognormal for gaps
    try:
        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = stats.shapiro(log_gaps)
        
        # KS test against fitted normal
        mu, sigma = np.mean(log_gaps), np.std(log_gaps)
        ks_normal, ks_normal_p = stats.kstest(log_gaps, 
                                               stats.norm(mu, sigma).cdf)
        
        results['normal_on_log'] = {
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'ks_stat': ks_normal,
            'ks_p': ks_normal_p,
            'mu': mu,
            'sigma': sigma,
        }
    except Exception as e:
        results['normal_on_log'] = {'error': str(e)}
    
    # Test 2: Exponential fit to raw gaps
    try:
        # MLE for exponential
        lambda_exp = 1.0 / np.mean(gaps)
        ks_exp, ks_exp_p = stats.kstest(gaps, 
                                        stats.expon(scale=1/lambda_exp).cdf)
        
        results['exponential'] = {
            'ks_stat': ks_exp,
            'ks_p': ks_exp_p,
            'lambda': lambda_exp,
        }
    except Exception as e:
        results['exponential'] = {'error': str(e)}
    
    # Test 3: Gamma fit to raw gaps
    try:
        # MLE for gamma
        shape, loc, scale = stats.gamma.fit(gaps)
        ks_gamma, ks_gamma_p = stats.kstest(gaps, 
                                            stats.gamma(shape, loc, scale).cdf)
        
        results['gamma'] = {
            'ks_stat': ks_gamma,
            'ks_p': ks_gamma_p,
            'shape': shape,
            'scale': scale,
        }
    except Exception as e:
        results['gamma'] = {'error': str(e)}
    
    # Test 4: Weibull fit to raw gaps
    try:
        # MLE for Weibull
        c, loc, scale = stats.weibull_min.fit(gaps)
        ks_weibull, ks_weibull_p = stats.kstest(gaps, 
                                                stats.weibull_min(c, loc, scale).cdf)
        
        results['weibull'] = {
            'ks_stat': ks_weibull,
            'ks_p': ks_weibull_p,
            'shape': c,
            'scale': scale,
        }
    except Exception as e:
        results['weibull'] = {'error': str(e)}
    
    # Determine best fit by lowest KS statistic
    ks_stats = {}
    for dist_name, dist_result in results.items():
        if 'error' not in dist_result and 'ks_stat' in dist_result:
            ks_stats[dist_name] = dist_result['ks_stat']
    
    if ks_stats:
        best_fit = min(ks_stats, key=ks_stats.get)
        results['best_fit'] = best_fit
        
        # Apply Bonferroni correction for multiple testing
        # Testing 4 distributions across 3 bands = 12 total tests
        n_tests = len(ks_stats)
        n_bands = 3  # Total bands in study
        bonferroni_alpha = 0.05 / (n_tests * n_bands) if n_tests > 0 else 0.05
        results['bonferroni_alpha'] = bonferroni_alpha
        results['n_tests'] = n_tests
        results['n_bands'] = n_bands
        
        # Check if lognormal (normal on log) is best and compute effect size
        if 'normal_on_log' in ks_stats and 'exponential' in ks_stats:
            ks_ratio = ks_stats['exponential'] / ks_stats['normal_on_log']
            results['ks_ratio_exp_to_lognormal'] = ks_ratio
            
            # Compute Cohen's d for practical significance
            # Positive Cohen's d means lognormal fits better (lower KS)
            cohens_d = compute_cohens_d(
                ks_stats['normal_on_log'], 
                ks_stats['exponential'], 
                len(gaps)
            )
            results['cohens_d_lognormal_vs_exponential'] = cohens_d
            results['practical_significance'] = abs(cohens_d) > 0.5
    
    results['band_name'] = band_name
    results['n_samples'] = len(gaps)
    
    return results


def test_distributions(primes: np.ndarray) -> Dict:
    """Test T2: Distribution Fitting Within Magnitude Bands.
    
    Tests distribution of gaps within magnitude bands:
    - [10^5, 10^6)
    - [10^6, 10^7)
    - [10^7, 10^8)
    
    Args:
        primes: Array of prime numbers
        
    Returns:
        Dictionary with results for each band and overall interpretation
    """
    gaps = np.diff(primes)
    
    # Define magnitude bands
    bands = {
        '1e5_1e6': (10**5, 10**6),
        '1e6_1e7': (10**6, 10**7),
        '1e7_1e8': (10**7, 10**8),
    }
    
    band_results = {}
    
    for band_name, (lower, upper) in bands.items():
        # Find gaps in this band
        mask = (primes[:-1] >= lower) & (primes[:-1] < upper)
        band_gaps = gaps[mask]
        
        if len(band_gaps) > 0:
            band_results[band_name] = test_distributions_in_band(band_gaps, band_name)
    
    # Cross-band analysis with family-wise error rate correction
    best_fits = []
    lognormal_count = 0
    exponential_count = 0
    cohens_d_values = []
    practical_sig_count = 0
    
    # Bonferroni-corrected alpha for cross-band decision
    # Testing 3 bands as independent hypotheses
    n_bands_tested = len(band_results)
    cross_band_alpha = 0.05 / n_bands_tested if n_bands_tested > 0 else 0.05
    
    for band_name, results in band_results.items():
        if 'best_fit' in results:
            best_fits.append(results['best_fit'])
            if results['best_fit'] == 'normal_on_log':
                lognormal_count += 1
            elif results['best_fit'] == 'exponential':
                exponential_count += 1
        
        # Collect Cohen's d values for practical significance assessment
        if 'cohens_d_lognormal_vs_exponential' in results:
            cohens_d_values.append(results['cohens_d_lognormal_vs_exponential'])
            if results.get('practical_significance', False):
                practical_sig_count += 1
    
    # Interpret consistency with Bonferroni-corrected threshold
    # Require both statistical consistency (>=2 bands) AND practical significance
    if lognormal_count >= 2 and practical_sig_count >= 1:
        interpretation = "Lognormal structure detected (reject H0 for H-MAIN-B)"
    elif exponential_count >= 2:
        interpretation = "Exponential structure detected (fail to reject H0)"
    elif lognormal_count >= 2:
        interpretation = "Lognormal detected but practical significance not established"
    else:
        interpretation = "Inconsistent across scales"
    
    return {
        'band_results': band_results,
        'best_fits': best_fits,
        'lognormal_count': lognormal_count,
        'exponential_count': exponential_count,
        'practical_sig_count': practical_sig_count,
        'cohens_d_values': cohens_d_values,
        'cross_band_alpha': cross_band_alpha,
        'n_bands_tested': n_bands_tested,
        'interpretation': interpretation,
    }


def compute_qq_data(gaps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Q-Q plot data for normal quantiles vs log(gap).
    
    Args:
        gaps: Array of gap values
        
    Returns:
        Tuple of (theoretical_quantiles, sample_quantiles)
    """
    log_gaps = np.log(gaps)
    log_gaps_sorted = np.sort(log_gaps)
    
    # Theoretical normal quantiles
    n = len(log_gaps)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, n))
    
    # Standardize sample quantiles
    mu = np.mean(log_gaps)
    sigma = np.std(log_gaps)
    sample_quantiles = (log_gaps_sorted - mu) / sigma
    
    return theoretical_quantiles, sample_quantiles
