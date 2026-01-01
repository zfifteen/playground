#!/usr/bin/env python3
"""
Baseline Euclidean/Riemannian Metric for GVA Search

This module implements the Z5D-based geometric resonance scoring
as the baseline metric for comparison against p-adic ultrametric.
"""

import math


def z5d_n_est(p: int) -> int:
    """IMPLEMENTED: Estimate prime index n such that p(n) ≈ p using PNT."""
    if p < 2:
        return 1
    
    ln_p = math.log(p)
    if ln_p <= 0:
        return 1
    
    # Prime counting function approximation
    inv_ln_p = 1.0 / ln_p
    n_est = (p / ln_p) * (1 + inv_ln_p + 2 * inv_ln_p * inv_ln_p)
    
    return max(1, int(n_est))


def z5d_predict_nth_prime(n: int) -> int:
    """IMPLEMENTED: Predict the nth prime using PNT-based approximation."""
    if n <= 0:
        return 2
    if n == 1:
        return 2
    if n == 2:
        return 3
    if n == 3:
        return 5
    if n == 4:
        return 7
    if n == 5:
        return 11
    
    ln_n = math.log(n)
    if ln_n <= 0:
        return 2
    
    ln_ln_n = math.log(ln_n)
    
    # PNT approximation with correction terms
    correction = (ln_ln_n - 2) / ln_n
    predicted = n * (ln_n + ln_ln_n - 1 + correction)
    
    return max(2, int(predicted))


def compute_z5d_score(p: int, n_est: int = None) -> float:
    """IMPLEMENTED: Compute Z5D score as normalized log-relative deviation."""
    if p <= 0:
        return 0.0
    
    if n_est is None:
        n_est = z5d_n_est(p)
    
    predicted_p = z5d_predict_nth_prime(n_est)
    
    # Compute absolute difference
    diff = abs(p - predicted_p)
    
    if diff == 0:
        return -100.0  # Perfect prediction
    
    # Log-relative error: log10(|p - p'| / p)
    rel_error = diff / p
    
    if rel_error <= 0:
        return -100.0
    
    try:
        score = math.log10(rel_error)
        # Clamp to reasonable range
        return max(-100.0, min(100.0, score))
    except (ValueError, OverflowError):
        # Fallback for edge cases
        return 0.0


def euclidean_distance(a: int, b: int) -> int:
    """IMPLEMENTED: Simple Euclidean distance between two integers."""
    return abs(a - b)


def riemannian_distance(candidate: int, target: int) -> float:
    """IMPLEMENTED: Riemannian-style distance based on Z5D geometric manifold."""
    # Get Z5D scores for both
    score_candidate = compute_z5d_score(candidate)
    score_target = compute_z5d_score(target)
    
    # Distance is the difference in their geometric positions
    euclidean = abs(candidate - target)
    geometric = abs(score_candidate - score_target)
    
    # Combine: geometric distance weighted by relative Euclidean distance
    if target > 0:
        normalized_euclidean = euclidean / target
    else:
        normalized_euclidean = 1.0
    
    # Final distance combines both geometric and Euclidean components
    return geometric + 0.1 * normalized_euclidean


def compute_gva_score(candidate: int, reference_point: int) -> float:
    """IMPLEMENTED: Main GVA scoring function for baseline metric."""
    # Primary score: Z5D geometric resonance
    z5d = compute_z5d_score(candidate)
    
    # Secondary: Riemannian distance from reference
    riem_dist = riemannian_distance(candidate, reference_point)
    
    # Combined score (Z5D is primary, distance is secondary)
    return z5d + 0.01 * riem_dist


if __name__ == "__main__":
    # Test implementation
    test_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    print("Testing baseline metric on small primes:")
    print(f"{'p':<10} {'n_est':<10} {'predicted':<12} {'z5d_score':<12}")
    print("-" * 50)
    
    for p in test_values:
        n_est = z5d_n_est(p)
        predicted = z5d_predict_nth_prime(n_est)
        score = compute_z5d_score(p, n_est)
        print(f"{p:<10} {n_est:<10} {predicted:<12} {score:<12.4f}")
    
    # Test GVA scoring
    print("\nTesting GVA scoring around sqrt(143) = 11.96...")
    sqrt_143 = 12
    for candidate in [7, 11, 13, 17]:
        gva = compute_gva_score(candidate, sqrt_143)
        print(f"  Candidate {candidate}: GVA score = {gva:.4f}")
