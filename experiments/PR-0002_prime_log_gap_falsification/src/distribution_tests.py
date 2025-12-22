import numpy as np
from scipy import stats

def fit_distributions(data):
    """
    Fits Normal and Log-Normal distributions to the data.
    Returns KS test statistics and p-values, plus MLE parameters.
    
    Note: For log-normal fit, we fit the distribution to the data directly.
    However, log-gaps are already 'log' in nature? 
    The hypothesis says log-gaps follow a log-normal distribution.
    So we fit lognorm to the data.
    """
    results = {}
    
    # Normal distribution
    mu, std = stats.norm.fit(data)
    ks_stat_norm, p_val_norm = stats.kstest(data, 'norm', args=(mu, std))
    results['normal'] = {
        'ks_stat': ks_stat_norm,
        'p_value': p_val_norm,
        'params': (mu, std)
    }
    
    # Log-Normal distribution
    # shape (s), loc, scale
    # We require strictly positive data for lognorm. Log gaps are > 0 since p_{n+1} > p_n.
    if np.any(data <= 0):
         # Should not happen for primes
         data_clean = data[data > 0]
    else:
         data_clean = data
         
    shape, loc, scale = stats.lognorm.fit(data_clean)
    ks_stat_lognorm, p_val_lognorm = stats.kstest(data_clean, 'lognorm', args=(shape, loc, scale))
    results['lognormal'] = {
        'ks_stat': ks_stat_lognorm,
        'p_value': p_val_lognorm,
        'params': (shape, loc, scale)
    }
    
    # Exponential distribution (Null hypothesis H0-D)
    loc_exp, scale_exp = stats.expon.fit(data_clean)
    ks_stat_exp, p_val_exp = stats.kstest(data_clean, 'expon', args=(loc_exp, scale_exp))
    results['exponential'] = {
        'ks_stat': ks_stat_exp,
        'p_value': p_val_exp,
        'params': (loc_exp, scale_exp)
    }

    # Uniform distribution (Null hypothesis H0-C check)
    loc_uni, scale_uni = stats.uniform.fit(data_clean)
    ks_stat_uni, p_val_uni = stats.kstest(data_clean, 'uniform', args=(loc_uni, scale_uni))
    results['uniform'] = {
        'ks_stat': ks_stat_uni,
        'p_value': p_val_uni,
        'params': (loc_uni, scale_uni)
    }
    
    return results

def calculate_moments(data):
    """
    Calculates Skewness and Kurtosis.
    """
    return {
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data) # This is excess kurtosis (Fisher)
    }