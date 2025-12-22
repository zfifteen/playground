import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf
import warnings


def ljung_box_test(data: np.ndarray, lags: int = 20):
    """
    Perform Ljung-Box test for autocorrelation up to specified lags.
    """
    # Ljung-Box test
    lb_df = acorr_ljungbox(data, lags=lags, return_df=True)
    results = {}
    for lag in range(1, lags + 1):
        lb_stat = lb_df.loc[lag, "lb_stat"]
        p_value = lb_df.loc[lag, "lb_pvalue"]
        results[f"lag_{lag}"] = {"lb_stat": lb_stat, "p_value": p_value}
    # Overall: are all p_values > 0.05?
    all_uncorrelated = all(res["p_value"] > 0.05 for res in results.values())
    return results, all_uncorrelated


def compute_acf(data: np.ndarray, nlags: int = 20):
    """
    Compute autocorrelation function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_values = acf(data, nlags=nlags, fft=True)
    return acf_values


def compute_pacf(data: np.ndarray, nlags: int = 20):
    """
    Compute partial autocorrelation function.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pacf_values = pacf(data, nlags=nlags)
    return pacf_values


def autocorrelation_analysis(data: np.ndarray, lags: int = 20):
    """
    Full autocorrelation analysis.
    """
    lb_results, all_uncorrelated = ljung_box_test(data, lags)
    acf_values = compute_acf(data, lags)
    pacf_values = compute_pacf(data, lags)

    # Check for significant autocorrelation (p < 0.01)
    significant_acf = [
        i for i in range(1, len(acf_values)) if abs(acf_values[i]) > 0.1
    ]  # rough threshold
    significant_pacf = [
        i for i in range(1, len(pacf_values)) if abs(pacf_values[i]) > 0.1
    ]

    analysis = {
        "ljung_box": lb_results,
        "all_uncorrelated": all_uncorrelated,
        "acf": acf_values,
        "pacf": pacf_values,
        "significant_acf_lags": significant_acf,
        "significant_pacf_lags": significant_pacf,
    }
    return analysis


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    data = np.random.randn(100)
    analysis = autocorrelation_analysis(data)
    print("All uncorrelated:", analysis["all_uncorrelated"])
    print("Significant ACF lags:", analysis["significant_acf_lags"])
