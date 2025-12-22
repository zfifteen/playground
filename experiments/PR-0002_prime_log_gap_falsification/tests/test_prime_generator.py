#!/usr/bin/env python3
"""
Tests for Prime Generator

Validates prime count against known π(x) values.

Author: GitHub Copilot
Date: December 2025
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prime_generator import (
    simple_sieve, 
    segmented_sieve, 
    generate_primes_to_limit,
    compute_log_gaps
)
import numpy as np


def test_simple_sieve_small():
    """Test simple sieve for small values."""
    primes = simple_sieve(20)
    expected = np.array([2, 3, 5, 7, 11, 13, 17, 19])
    assert np.array_equal(primes, expected), f"Got {primes}, expected {expected}"
    print("✓ test_simple_sieve_small passed")


def test_simple_sieve_edge_cases():
    """Test edge cases for simple sieve."""
    assert len(simple_sieve(0)) == 0
    assert len(simple_sieve(1)) == 0
    assert len(simple_sieve(2)) == 1
    assert simple_sieve(2)[0] == 2
    print("✓ test_simple_sieve_edge_cases passed")


def test_pi_1e5():
    """Test π(10^5) = 9,592"""
    primes = generate_primes_to_limit(10**5, validate=True)
    assert len(primes) == 9592, f"π(10^5) = {len(primes)}, expected 9592"
    print("✓ test_pi_1e5 passed")


def test_pi_1e6():
    """Test π(10^6) = 78,498"""
    primes = generate_primes_to_limit(10**6, validate=True)
    assert len(primes) == 78498, f"π(10^6) = {len(primes)}, expected 78498"
    print("✓ test_pi_1e6 passed")


def test_segmented_vs_simple():
    """Test that segmented sieve matches simple sieve."""
    limit = 10**5
    simple = simple_sieve(limit)
    segmented = segmented_sieve(limit, segment_size=10**4)
    assert np.array_equal(simple, segmented), "Segmented and simple sieves don't match"
    print("✓ test_segmented_vs_simple passed")


def test_primes_are_sorted():
    """Test that generated primes are in ascending order."""
    primes = generate_primes_to_limit(10**5)
    assert np.all(np.diff(primes) > 0), "Primes are not strictly ascending"
    print("✓ test_primes_are_sorted passed")


def test_compute_log_gaps():
    """Test log-gap computation."""
    primes = np.array([2, 3, 5, 7, 11])
    data = compute_log_gaps(primes)
    
    # log_gaps should be [ln(3/2), ln(5/3), ln(7/5), ln(11/7)]
    expected_log_gaps = np.log(np.array([3/2, 5/3, 7/5, 11/7]))
    
    assert len(data['log_gaps']) == 4
    assert np.allclose(data['log_gaps'], expected_log_gaps)
    
    # Regular gaps should be [1, 2, 2, 4]
    expected_regular_gaps = np.array([1, 2, 2, 4])
    assert np.array_equal(data['regular_gaps'], expected_regular_gaps)
    
    print("✓ test_compute_log_gaps passed")


def test_log_gaps_positive():
    """Test that all log-gaps are positive (primes are increasing)."""
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    assert np.all(data['log_gaps'] > 0), "Some log-gaps are not positive"
    print("✓ test_log_gaps_positive passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Prime Generator Tests")
    print("=" * 60)
    
    test_simple_sieve_small()
    test_simple_sieve_edge_cases()
    test_pi_1e5()
    test_segmented_vs_simple()
    test_primes_are_sorted()
    test_compute_log_gaps()
    test_log_gaps_positive()
    
    print("\n" + "=" * 60)
    print("Testing π(10^6) (may take a moment)...")
    test_pi_1e6()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
