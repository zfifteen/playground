import numpy as np
import pandas as pd
from scipy import stats

def compute_log_gaps(primes):
    """
    Computes log-gaps from an array of primes.
    log_gap[n] = ln(p[n+1]) - ln(p[n])
    """
    if len(primes) < 2:
        return np.array([])
    
    log_primes = np.log(primes)
    log_gaps = np.diff(log_primes)
    return log_gaps

def compute_quintile_stats(log_gaps):
    """
    Splits log_gaps into 5 equal quintiles and computes mean, std for each.
    Returns a DataFrame.
    """
    n = len(log_gaps)
    quintile_size = n // 5
    
    stats_list = []
    for i in range(5):
        start = i * quintile_size
        end = (i + 1) * quintile_size if i < 4 else n
        chunk = log_gaps[start:end]
        
        stats_list.append({
            'quintile': i + 1,
            'mean': np.mean(chunk),
            'std': np.std(chunk),
            'count': len(chunk)
        })
        
    return pd.DataFrame(stats_list)

def compute_decile_stats(log_gaps):
    """
    Splits log_gaps into 10 equal deciles and computes mean, std for each.
    Returns a DataFrame.
    """
    n = len(log_gaps)
    decile_size = n // 10
    
    stats_list = []
    for i in range(10):
        start = i * decile_size
        end = (i + 1) * decile_size if i < 9 else n
        chunk = log_gaps[start:end]
        
        stats_list.append({
            'decile': i + 1,
            'mean': np.mean(chunk),
            'std': np.std(chunk),
            'count': len(chunk)
        })
        
    return pd.DataFrame(stats_list)

def regression_on_means(stats_df, x_col='quintile', y_col='mean'):
    """
    Performs linear regression on the means of quintiles/deciles.
    Returns slope, intercept, r_value, p_value, std_err.
    """
    x = stats_df[x_col].values
    y = stats_df[y_col].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }