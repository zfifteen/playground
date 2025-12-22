import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

def compute_autocorrelation(data, nlags=20):
    """
    Computes ACF and PACF values.
    """
    # Limit data for performance if very large
    # PACF can be slow on 5M points. 
    # 1M points is more than enough for stable ACF/PACF at low lags.
    if len(data) > 1000000:
        calc_data = data[:1000000]
    else:
        calc_data = data
        
    acf_values = acf(calc_data, nlags=nlags, fft=True)
    pacf_values = pacf(calc_data, nlags=nlags, method='yw') # yw = Yule-Walker
    
    return {
        'acf': acf_values,
        'pacf': pacf_values
    }

def perform_ljung_box(data, lags=20):
    """
    Performs Ljung-Box test for autocorrelation.
    Returns a DataFrame with test statistics and p-values.
    """
    if len(data) > 1000000:
        calc_data = data[:1000000]
    else:
        calc_data = data
        
    lb_result = acorr_ljungbox(calc_data, lags=[lags], return_df=True)
    return lb_result
