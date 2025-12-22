#!/usr/bin/env python3
"""
Prime Generation using Segmented Sieve of Eratosthenes

Memory-efficient prime generation for large ranges up to 10^9.
Includes validation against known prime counting function values.

Author: GitHub Copilot
Date: December 2025
"""

import numpy as np
from math import isqrt

# Known values of π(x) for validation
KNOWN_PI_VALUES = {
    10**5: 9592,
    10**6: 78498,
    10**7: 664579,
    10**8: 5761455,
    10**9: 50847534,
}


def simple_sieve(limit):
    """
    Simple Sieve of Eratosthenes for small ranges.
    
    Args:
        limit: Upper bound (inclusive)
        
    Returns:
        numpy array of primes up to limit
    """
    if limit < 2:
        return np.array([], dtype=np.int64)
    
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    return np.nonzero(is_prime)[0].astype(np.int64)


def segmented_sieve(limit, segment_size=10**6):
    """
    Memory-efficient prime generation using segmented sieve.
    
    Algorithm:
    1. Generate small primes up to sqrt(limit) using basic sieve
    2. Process [sqrt(limit), limit] in segments of segment_size
    3. For each segment, mark composites using small primes
    4. Collect primes from each segment
    
    Args:
        limit: Upper bound for prime generation
        segment_size: Size of each segment (default 10^6)
        
    Returns:
        numpy array of all primes up to limit
    """
    if limit < 2:
        return np.array([], dtype=np.int64)
    
    sqrt_limit = isqrt(limit)
    
    # Step 1: Generate small primes using simple sieve
    small_primes = simple_sieve(sqrt_limit)
    
    # If limit <= sqrt threshold, we're done
    if limit <= sqrt_limit:
        return small_primes
    
    # Step 2: Process segments
    all_primes = list(small_primes)
    
    low = sqrt_limit + 1
    while low <= limit:
        high = min(low + segment_size - 1, limit)
        
        # Create segment sieve
        sieve_size = high - low + 1
        is_prime_segment = np.ones(sieve_size, dtype=bool)
        
        # Mark composites in this segment
        for p in small_primes:
            # Find the minimum multiple of p >= low
            start = ((low + p - 1) // p) * p
            if start < p * p:
                start = p * p
            
            if start <= high:
                # Mark multiples of p in segment
                is_prime_segment[start - low::p] = False
        
        # Collect primes from segment
        segment_primes = np.nonzero(is_prime_segment)[0] + low
        all_primes.extend(segment_primes)
        
        low = high + 1
    
    return np.array(all_primes, dtype=np.int64)


def generate_primes_to_limit(limit, validate=True):
    """
    Generate all primes up to a given limit.
    
    Args:
        limit: Upper bound for prime generation
        validate: If True, validate count against known π(x) values
        
    Returns:
        numpy array of primes
        
    Raises:
        ValueError: If validation fails
    """
    # Choose algorithm based on limit size
    if limit <= 10**6:
        primes = simple_sieve(limit)
    else:
        primes = segmented_sieve(limit)
    
    # Validate if requested
    if validate:
        count = len(primes)
        for threshold, expected in KNOWN_PI_VALUES.items():
            if limit == threshold:
                if count != expected:
                    raise ValueError(
                        f"Prime count validation failed: π({limit}) = {count}, "
                        f"expected {expected}"
                    )
                break
    
    return primes


def validate_prime_count(limit, count):
    """
    Validate prime count against known values.
    
    Args:
        limit: The upper bound used for generation
        count: The count of primes generated
        
    Returns:
        (is_valid, expected, actual) tuple
    """
    if limit in KNOWN_PI_VALUES:
        expected = KNOWN_PI_VALUES[limit]
        return (count == expected, expected, count)
    return (True, None, count)  # No known value to validate against


def compute_log_gaps(primes):
    """
    Compute log-gaps between consecutive primes.
    
    For consecutive prime pairs (p_n, p_{n+1}):
    log_gap[n] = ln(p_{n+1}) - ln(p_n) = ln(p_{n+1} / p_n)
    
    Args:
        primes: numpy array of primes
        
    Returns:
        Dictionary with:
        - 'log_gaps': log-gaps array
        - 'regular_gaps': regular gaps array  
        - 'log_primes': log of primes array
    """
    log_primes = np.log(primes.astype(np.float64))
    log_gaps = np.diff(log_primes)  # ln(p_{n+1}/p_n)
    regular_gaps = np.diff(primes)  # p_{n+1} - p_n
    
    return {
        'log_gaps': log_gaps,
        'regular_gaps': regular_gaps,
        'log_primes': log_primes[:-1],  # Align with gaps
        'primes': primes[:-1]  # Align with gaps
    }


if __name__ == "__main__":
    # Quick test
    print("Testing prime generator...")
    
    for limit in [10**5, 10**6]:
        primes = generate_primes_to_limit(limit, validate=True)
        print(f"π({limit:,}) = {len(primes):,} ✓")
    
    print("\nComputing log-gaps for π(10^5)...")
    primes = generate_primes_to_limit(10**5)
    data = compute_log_gaps(primes)
    print(f"Log-gap range: [{data['log_gaps'].min():.6f}, {data['log_gaps'].max():.6f}]")
    print(f"Log-gap mean: {data['log_gaps'].mean():.6f}")
    print(f"Log-gap std: {data['log_gaps'].std():.6f}")
