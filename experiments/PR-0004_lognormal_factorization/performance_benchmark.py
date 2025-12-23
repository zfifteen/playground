#!/usr/bin/env python3
"""Performance benchmark for lognormal factorization pipeline on fitted bands."""

import time
import random
from typing import List
from src.pipeline import factor_with_lognormal_prefilter, factor_classical
from src.model import ModelStore, Band
from src.config import SearchPolicyConfig


def generate_primes_in_range(min_p: int, max_p: int, count: int) -> List[int]:
    """Generate primes in [min_p, max_p) using sieve."""
    # Simple sieve for the range
    sieve = [True] * (max_p - min_p)
    primes = []

    for i in range(2, int(max_p**0.5) + 1):
        start = ((min_p + i - 1) // i) * i
        for j in range(max(start, min_p), max_p, i):
            if j != i and j >= min_p:
                sieve[j - min_p] = False

    for i in range(min_p, max_p):
        if sieve[i - min_p] and i >= 2:
            primes.append(i)
            if len(primes) >= count:
                break
    return primes[:count]


def generate_test_semiprimes(band: Band, count: int) -> List[int]:
    """Generate semiprimes with p,q in band's range."""
    min_p = band.p_min
    max_p = band.p_max
    primes = generate_primes_in_range(min_p, max_p, count * 2)
    semiprimes = []
    for i in range(0, len(primes) - 1, 2):
        p = primes[i]
        q = primes[i + 1]
        semiprimes.append(p * q)
        if len(semiprimes) >= count:
            break
    return semiprimes


def performance_benchmark():
    """Benchmark on semiprimes in fitted bands."""
    model = ModelStore()
    cfg = SearchPolicyConfig(
        max_steps=10000, radius_scale=1.0, direction_mode="ALTERNATE"
    )

    results = []

    for band_idx, band in enumerate(model.bands):
        print(f"Benchmarking band {band_idx + 1}: {band.p_min}-{band.p_max}")
        test_ns = generate_test_semiprimes(band, 10)  # 10 semiprimes per band

        for N in test_ns:
            # Set seed for reproducibility
            cfg.seed = random.randint(0, 1000000)
            cfg.__post_init__()  # Re-init rng

            # Pipeline
            start = time.time()
            pipeline_factor = factor_with_lognormal_prefilter(N, model, cfg)
            pipeline_time = time.time() - start

            # Classical
            start = time.time()
            classical_factor = factor_classical(N)
            classical_time = time.time() - start

            # Verify
            if pipeline_factor and classical_factor:
                factors = {pipeline_factor, N // pipeline_factor}
                classical_factors = {classical_factor, N // classical_factor}
                success = factors == classical_factors
            else:
                success = False

            results.append(
                {
                    "band": band_idx + 1,
                    "N": N,
                    "pipeline_time": pipeline_time,
                    "classical_time": classical_time,
                    "success": success,
                    "speedup": classical_time / pipeline_time
                    if pipeline_time > 0
                    else 0,
                }
            )

    # Summary per band
    for band_idx in range(1, len(model.bands) + 1):
        band_results = [r for r in results if r["band"] == band_idx]
        if band_results:
            avg_pipeline = sum(r["pipeline_time"] for r in band_results) / len(
                band_results
            )
            avg_classical = sum(r["classical_time"] for r in band_results) / len(
                band_results
            )
            avg_speedup = sum(r["speedup"] for r in band_results) / len(band_results)
            success_rate = sum(1 for r in band_results if r["success"]) / len(
                band_results
            )
            print(
                f"Band {band_idx}: Avg pipeline {avg_pipeline:.4f}s, classical {avg_classical:.4f}s, speedup {avg_speedup:.2f}x, success {success_rate:.0%}"
            )


if __name__ == "__main__":
    performance_benchmark()
