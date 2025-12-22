"""
Log Gap Analysis Module

This module transforms prime numbers into "log-gaps," which are the logarithmic differences between consecutive primes.
Why logs? Prime gaps (differences like 3-2=1, 5-3=2) grow larger on average as numbers get bigger.
But logarithms (natural log, ln) compress this growth, making gaps relative rather than absolute.
This is key to our hypothesis: in electrical circuits, we often deal with ratios (like voltage logs), not absolute values.
Log-gaps reveal patterns like decay (gaps shrinking relatively) that might mimic a "damped" system.

We also compute regular gaps for comparison and analyze trends across "quintiles" (5 groups) or "deciles" (10 groups) of primes.
"""

import numpy as np
from scipy import stats


def compute_log_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Calculate logarithmic prime gaps.

    For each pair of consecutive primes (p_n, p_{n+1}), compute ln(p_{n+1}) - ln(p_n).
    This is the same as ln(p_{n+1} / p_n), the natural log of their ratio.
    Why? Ratios capture multiplicative relationships, like how circuit voltages relate logarithmically.
    Without logs, gaps are additive and dominated by large numbers; logs make them comparable.
    """
    log_primes = np.log(
        primes.astype(np.float64)
    )  # Convert primes to logs (float for precision)
    log_gaps = np.diff(log_primes)  # Difference between consecutive logs
    return log_gaps  # Array of log-gaps


def compute_regular_gaps(primes: np.ndarray) -> np.ndarray:
    """
    Calculate regular (arithmetic) prime gaps for comparison.

    Just p_{n+1} - p_n, the straight difference.
    These show the raw "distances" between primes, which increase over time (Prime Number Theorem).
    We compute them alongside log-gaps to contrast additive vs. multiplicative views.
    In the experiment, log-gaps are the focus, but regular gaps help validate data.
    """
    return np.diff(primes)  # Simple differences


def get_quintile_means(log_gaps: np.ndarray) -> np.ndarray:
    """
    Split log-gaps into 5 equal groups (quintiles) and find the average gap in each.

    Quintiles divide data into fifths: first 20%, second 20%, etc.
    Here, we group the log-gaps by their position (earlier vs. later in the prime list).
    Computing means per group lets us check if gaps change systematically.
    In the hypothesis, we expect averages to decrease (decay) as primes get larger.
    """
    n = len(log_gaps)
    quintile_size = n // 5  # Size of each group
    means = []
    for i in range(5):  # 5 quintiles
        start = i * quintile_size
        end = (i + 1) * quintile_size if i < 4 else n  # Last group takes remainder
        mean = np.mean(log_gaps[start:end])  # Average gap in this group
        means.append(mean)
    return np.array(means)


def get_decile_means(log_gaps: np.ndarray) -> np.ndarray:
    """
    Split log-gaps into 10 equal groups (deciles) and find the average gap in each.

    Deciles are like quintiles but finer: tenths of the data.
    This gives more detail on trends, useful for checking if decay is consistent.
    Similar to quintiles, we look for decreasing means as a sign of "damping."
    """
    n = len(log_gaps)
    decile_size = n // 10  # Size of each group
    means = []
    for i in range(10):  # 10 deciles
        start = i * decile_size
        end = (i + 1) * decile_size if i < 9 else n  # Last group takes remainder
        mean = np.mean(log_gaps[start:end])  # Average gap in this group
        means.append(mean)
    return np.array(means)


def regression_analysis(means: np.ndarray):
    """
    Fit a straight line to the means (quintile or decile averages) to check for trends.

    Linear regression finds the best-fit line: y = slope * x + intercept.
    Here, x is the group index (0 for first quintile, 1 for second, etc.), y is the mean gap.
    We want to know if gaps are decreasing (negative slope, like damping).
    Outputs: slope (trend strength), intercept (starting point), r_squared (how well the line fits, 0-1), p_value (probability the trend is random, low means significant).
    """
    x = np.arange(len(means))  # x-values: 0, 1, 2, ... for each group
    result = stats.linregress(x, means)  # SciPy's regression function
    slope = result[0]  # How much y changes per x
    intercept = result[1]  # y when x=0
    r_value = result[2]  # Correlation coefficient
    p_value = result[3]  # Significance
    r_squared = r_value * r_value  # Fraction of variance explained
    return slope, intercept, r_squared, p_value


def analyze_log_gaps(primes: np.ndarray) -> dict:
    """
    Run the complete log-gap analysis pipeline.

    This is the heart of the module: takes a list of primes and outputs everything needed to test the hypothesis.
    It computes gaps, stats, trends, and regressions to see if log-gaps behave like a damped circuit.
    The result dict is used by other modules for plotting and falsification checks.
    """
    log_gaps = compute_log_gaps(primes)  # Get the log-gaps
    regular_gaps = compute_regular_gaps(primes)  # And regular gaps for context

    # Basic statistics: summarize the distribution
    basic_stats = {
        "count": len(log_gaps),  # Number of gaps
        "mean": np.mean(log_gaps),  # Average gap
        "std": np.std(log_gaps),  # Spread around the mean
        "min": np.min(log_gaps),  # Smallest gap
        "max": np.max(log_gaps),  # Largest gap
        "skewness": stats.skew(log_gaps),  # Asymmetry (positive = right-tail heavy)
        "kurtosis": stats.kurtosis(log_gaps),  # Tail heaviness (excess over normal)
        "regular_gap_mean": np.mean(regular_gaps),  # Regular gap average
        "regular_gap_std": np.std(regular_gaps),  # Regular gap spread
        "regular_gap_max": np.max(regular_gaps),  # Biggest regular gap
    }

    # Analyze trends in quintiles (5 groups)
    quintile_means = get_quintile_means(log_gaps)  # Averages per group
    quintile_slope, quintile_intercept, quintile_r2, quintile_p = regression_analysis(
        quintile_means  # Fit line to check decay
    )

    # Finer analysis with deciles (10 groups)
    decile_means = get_decile_means(log_gaps)
    decile_slope, decile_intercept, decile_r2, decile_p = regression_analysis(
        decile_means
    )

    # Package everything into a results dictionary
    analysis = {
        "basic_stats": basic_stats,  # Summary numbers
        "quintile_means": quintile_means,  # Group averages
        "quintile_regression": {  # Trend line for quintiles
            "slope": quintile_slope,
            "intercept": quintile_intercept,
            "r_squared": quintile_r2,
            "p_value": quintile_p,
        },
        "decile_means": decile_means,
        "decile_regression": {  # Trend line for deciles
            "slope": decile_slope,
            "intercept": decile_intercept,
            "r_squared": decile_r2,
            "p_value": decile_p,
        },
        "log_gaps": log_gaps,  # Full array for further use
        "regular_gaps": regular_gaps,
    }

    return analysis


if __name__ == "__main__":
    # Quick test to see the analysis in action
    from prime_generator import generate_primes_up_to

    primes = generate_primes_up_to(100)  # Small set for testing
    analysis = analyze_log_gaps(primes)  # Run full analysis
    print("Basic stats:", analysis["basic_stats"])  # Show summary
    print("Quintile means:", analysis["quintile_means"])  # Group averages
    print("Quintile regression:", analysis["quintile_regression"])  # Trend check
