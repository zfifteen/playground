import numpy as np
import os
import sys

sys.path.append("src")

from prime_generator import PrimeGenerator
from log_gap_analysis import analyze_log_gaps
from distribution_tests import run_distribution_tests, find_best_fit
from autocorrelation import autocorrelation_analysis
from visualization import generate_all_plots


def run_full_analysis(limit: int, save_data: bool = True):
    """
    Run full analysis for given limit.
    """
    print(f"Starting analysis for limit {limit}...")

    # Generate primes
    print("Generating primes...")
    primes = PrimeGenerator.generate_primes_array(limit)
    count = len(primes)
    print(f"Generated {count} primes up to {limit}")

    # Validate count
    expected = {10**6: 78498, 10**7: 664579, 10**8: 5761455}[limit]
    if abs(count - expected) > 1:
        print(f"Warning: Prime count {count} deviates from expected {expected}")
    else:
        print("Prime count validated.")

    # Compute analysis
    print("Computing log-gap analysis...")
    analysis = analyze_log_gaps(primes)

    # Distribution tests
    print("Running distribution tests...")
    dist_tests = run_distribution_tests(analysis["log_gaps"])
    best_dist, best_ks = find_best_fit(dist_tests)

    # Autocorrelation
    print("Running autocorrelation analysis...")
    autocorr = autocorrelation_analysis(analysis["log_gaps"])

    # Generate plots
    print("Generating plots...")
    generate_all_plots(
        analysis["log_gaps"],
        analysis["quintile_means"],
        analysis["decile_means"],
        autocorr["acf"],
        autocorr["pacf"],
    )

    # Save data
    if save_data:
        os.makedirs("data", exist_ok=True)
        np.save(f"data/primes_{limit}.npy", primes)
        np.savetxt(f"data/log_gaps_{limit}.csv", analysis["log_gaps"], delimiter=",")

        os.makedirs("results", exist_ok=True)
        # Save summary
        summary = {
            "limit": limit,
            "prime_count": count,
            "basic_stats": analysis["basic_stats"],
            "quintile_means": analysis["quintile_means"].tolist(),
            "decile_means": analysis["decile_means"].tolist(),
            "quintile_regression": analysis["quintile_regression"],
            "decile_regression": analysis["decile_regression"],
            "best_distribution": best_dist,
            "best_ks_stat": best_ks,
            "all_uncorrelated": autocorr["all_uncorrelated"],
            "significant_acf_lags": autocorr["significant_acf_lags"],
            "significant_pacf_lags": autocorr["significant_pacf_lags"],
        }
        np.save(f"results/analysis_{limit}.npy", summary)

    print(f"Analysis complete for {limit}")

    # Return key results for falsification check
    return {
        "analysis": analysis,
        "dist_tests": dist_tests,
        "autocorr": autocorr,
        "best_dist": best_dist,
        "best_ks": best_ks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("limit", type=int, help="Upper limit for primes")
    args = parser.parse_args()
    run_full_analysis(args.limit)
