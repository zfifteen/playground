import numpy as np
from scipy import stats


def compute_log_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Compute log-gaps: ln(p_{n+1}) - ln(p_n) = ln(p_{n+1}/p_n)
    """
    log_primes = np.log(primes.astype(np.float64))
    log_gaps = np.diff(log_primes)
    return log_gaps


def compute_regular_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Compute regular gaps: p_{n+1} - p_n
    """
    return np.diff(primes)


def get_quintile_means(log_gaps: np.ndarray) -> np.ndarray:
    """
    Divide log_gaps into 5 quintiles and compute means.
    """
    n = len(log_gaps)
    quintile_size = n // 5
    means = []
    for i in range(5):
        start = i * quintile_size
        end = (i + 1) * quintile_size if i < 4 else n
        mean = np.mean(log_gaps[start:end])
        means.append(mean)
    return np.array(means)


def get_decile_means(log_gaps: np.ndarray) -> np.ndarray:
    """
    Divide log_gaps into 10 deciles and compute means.
    """
    n = len(log_gaps)
    decile_size = n // 10
    means = []
    for i in range(10):
        start = i * decile_size
        end = (i + 1) * decile_size if i < 9 else n
        mean = np.mean(log_gaps[start:end])
        means.append(mean)
    return np.array(means)


def regression_analysis(means: np.ndarray):
    """
    Perform linear regression on means vs index.
    Returns: slope, intercept, r_squared, p_value
    """
    x = np.arange(len(means))
    result = stats.linregress(x, means)
    slope = result[0]
    intercept = result[1]
    r_value = result[2]
    p_value = result[3]
    r_squared = r_value * r_value
    return slope, intercept, r_squared, p_value


def analyze_log_gaps(primes: np.ndarray) -> dict:
    """
    Perform full log-gap analysis.
    """
    log_gaps = compute_log_gaps(primes)
    regular_gaps = compute_regular_gaps(primes)

    # Basic stats
    basic_stats = {
        "count": len(log_gaps),
        "mean": np.mean(log_gaps),
        "std": np.std(log_gaps),
        "min": np.min(log_gaps),
        "max": np.max(log_gaps),
        "skewness": stats.skew(log_gaps),
        "kurtosis": stats.kurtosis(log_gaps),  # excess kurtosis
        "regular_gap_mean": np.mean(regular_gaps),
        "regular_gap_std": np.std(regular_gaps),
        "regular_gap_max": np.max(regular_gaps),
    }

    # Quintile analysis
    quintile_means = get_quintile_means(log_gaps)
    quintile_slope, quintile_intercept, quintile_r2, quintile_p = regression_analysis(
        quintile_means
    )

    # Decile analysis
    decile_means = get_decile_means(log_gaps)
    decile_slope, decile_intercept, decile_r2, decile_p = regression_analysis(
        decile_means
    )

    analysis = {
        "basic_stats": basic_stats,
        "quintile_means": quintile_means,
        "quintile_regression": {
            "slope": quintile_slope,
            "intercept": quintile_intercept,
            "r_squared": quintile_r2,
            "p_value": quintile_p,
        },
        "decile_means": decile_means,
        "decile_regression": {
            "slope": decile_slope,
            "intercept": decile_intercept,
            "r_squared": decile_r2,
            "p_value": decile_p,
        },
        "log_gaps": log_gaps,
        "regular_gaps": regular_gaps,
    }

    return analysis


if __name__ == "__main__":
    # Test with small primes
    from prime_generator import generate_primes_up_to

    primes = generate_primes_up_to(100)
    analysis = analyze_log_gaps(primes)
    print("Basic stats:", analysis["basic_stats"])
    print("Quintile means:", analysis["quintile_means"])
    print("Quintile regression:", analysis["quintile_regression"])
