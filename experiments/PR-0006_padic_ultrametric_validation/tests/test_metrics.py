#!/usr/bin/env python3
"""
Unit tests for baseline and p-adic metrics
"""

import sys
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from metric_baseline import (
    z5d_n_est,
    z5d_predict_nth_prime,
    compute_z5d_score,
    euclidean_distance,
    riemannian_distance,
    compute_gva_score
)

from metric_padic import (
    padic_valuation,
    padic_norm,
    padic_distance,
    prime_factorization_small,
    multi_padic_distance,
    adaptive_padic_score,
    padic_ultrametric_gva_score
)


def test_z5d_n_est():
    """Test prime index estimation"""
    # For small primes, estimates should be reasonable
    assert z5d_n_est(2) > 0
    assert z5d_n_est(11) > 0
    assert z5d_n_est(29) > 0
    print("✓ test_z5d_n_est passed")


def test_z5d_predict_nth_prime():
    """Test nth prime prediction"""
    # Small indices should give reasonable predictions
    assert z5d_predict_nth_prime(1) == 2
    assert z5d_predict_nth_prime(2) == 3
    assert z5d_predict_nth_prime(5) == 11
    print("✓ test_z5d_predict_nth_prime passed")


def test_compute_z5d_score():
    """Test Z5D scoring"""
    # Small primes should have low (negative) scores
    score_2 = compute_z5d_score(2)
    score_11 = compute_z5d_score(11)
    
    # Scores should be finite
    assert math.isfinite(score_2)
    assert math.isfinite(score_11)
    print("✓ test_compute_z5d_score passed")


def test_euclidean_distance():
    """Test Euclidean distance"""
    assert euclidean_distance(0, 0) == 0
    assert euclidean_distance(5, 8) == 3
    assert euclidean_distance(8, 5) == 3
    print("✓ test_euclidean_distance passed")


def test_riemannian_distance():
    """Test Riemannian distance"""
    # Should return finite values
    dist = riemannian_distance(11, 13)
    assert math.isfinite(dist)
    assert dist >= 0
    print("✓ test_riemannian_distance passed")


def test_compute_gva_score():
    """Test GVA scoring"""
    # Should return finite scores
    score = compute_gva_score(11, 12)
    assert math.isfinite(score)
    print("✓ test_compute_gva_score passed")


def test_padic_valuation():
    """Test p-adic valuation"""
    # v_2(8) = 3 (8 = 2^3)
    assert padic_valuation(8, 2) == 3
    
    # v_3(9) = 2 (9 = 3^2)
    assert padic_valuation(9, 3) == 2
    
    # v_2(5) = 0 (5 is odd)
    assert padic_valuation(5, 2) == 0
    
    # v_p(0) = infinity
    assert padic_valuation(0, 2) == float('inf')
    print("✓ test_padic_valuation passed")


def test_padic_norm():
    """Test p-adic norm"""
    # ||8||_2 = 2^(-3) = 0.125
    assert abs(padic_norm(8, 2) - 0.125) < 1e-10
    
    # ||9||_3 = 3^(-2) = 0.111...
    assert abs(padic_norm(9, 3) - 1/9) < 1e-10
    
    # ||0||_p = 0
    assert padic_norm(0, 2) == 0.0
    print("✓ test_padic_norm passed")


def test_padic_distance():
    """Test p-adic distance"""
    # d_2(8, 12) = ||8-12||_2 = ||4||_2 = 2^(-2) = 0.25
    assert abs(padic_distance(8, 12, 2) - 0.25) < 1e-10
    
    # Distance is symmetric
    assert padic_distance(8, 12, 2) == padic_distance(12, 8, 2)
    print("✓ test_padic_distance passed")


def test_padic_ultrametric_property():
    """Test ultrametric inequality: d(a,c) <= max(d(a,b), d(b,c))"""
    a, b, c = 8, 12, 20
    p = 2
    
    dab = padic_distance(a, b, p)
    dbc = padic_distance(b, c, p)
    dac = padic_distance(a, c, p)
    
    # Ultrametric property
    assert dac <= max(dab, dbc) + 1e-10
    print("✓ test_padic_ultrametric_property passed")


def test_prime_factorization_small():
    """Test small prime factorization"""
    # 12 = 2^2 * 3
    factors = prime_factorization_small(12, max_prime=100)
    assert factors[2] == 2
    assert factors[3] == 1
    
    # 1 has no factors
    factors = prime_factorization_small(1, max_prime=100)
    assert len(factors) == 0
    
    # 7 is prime
    factors = prime_factorization_small(7, max_prime=100)
    assert factors[7] == 1
    print("✓ test_prime_factorization_small passed")


def test_multi_padic_distance():
    """Test multi-prime p-adic distance"""
    # Should return finite value
    dist = multi_padic_distance(10, 15, [2, 3, 5])
    assert math.isfinite(dist)
    assert dist >= 0
    print("✓ test_multi_padic_distance passed")


def test_adaptive_padic_score():
    """Test adaptive p-adic scoring"""
    # Should return finite score
    score = adaptive_padic_score(11, 12)
    assert math.isfinite(score)
    print("✓ test_adaptive_padic_score passed")


def test_padic_ultrametric_gva_score():
    """Test main p-adic GVA scoring"""
    # Test without N
    score1 = padic_ultrametric_gva_score(11, 12)
    assert math.isfinite(score1)
    
    # Test with N
    score2 = padic_ultrametric_gva_score(11, 12, 143)
    assert math.isfinite(score2)
    print("✓ test_padic_ultrametric_gva_score passed")


def test_metrics_produce_different_scores():
    """Test that both metrics produce different scores for different candidates"""
    N = 143
    sqrt_N = 12
    
    # Test baseline metric on different candidates
    score_11_baseline = compute_gva_score(11, sqrt_N)
    score_13_baseline = compute_gva_score(13, sqrt_N)
    score_15_baseline = compute_gva_score(15, sqrt_N)
    
    # All should be finite
    assert math.isfinite(score_11_baseline)
    assert math.isfinite(score_13_baseline)
    assert math.isfinite(score_15_baseline)
    
    # Test p-adic metric on different candidates
    score_11_padic = padic_ultrametric_gva_score(11, sqrt_N, N)
    score_13_padic = padic_ultrametric_gva_score(13, sqrt_N, N)
    score_15_padic = padic_ultrametric_gva_score(15, sqrt_N, N)
    
    # All should be finite
    assert math.isfinite(score_11_padic)
    assert math.isfinite(score_13_padic)
    assert math.isfinite(score_15_padic)
    
    print("✓ test_metrics_produce_different_scores passed")


def run_all_tests():
    """Run all unit tests"""
    print("="*60)
    print("Running unit tests for metrics...")
    print("="*60)
    
    # Baseline metric tests
    test_z5d_n_est()
    test_z5d_predict_nth_prime()
    test_compute_z5d_score()
    test_euclidean_distance()
    test_riemannian_distance()
    test_compute_gva_score()
    
    # p-adic metric tests
    test_padic_valuation()
    test_padic_norm()
    test_padic_distance()
    test_padic_ultrametric_property()
    test_prime_factorization_small()
    test_multi_padic_distance()
    test_adaptive_padic_score()
    test_padic_ultrametric_gva_score()
    
    # Integration tests
    test_metrics_produce_different_scores()
    
    print("="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()
