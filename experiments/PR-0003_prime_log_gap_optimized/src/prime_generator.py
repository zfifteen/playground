"""
Prime Generator Module with Disk Caching

Generates primes up to 10^9 using segmented sieve and caches results to disk.
Enables rerunning analysis without regenerating primes.
"""

import numpy as np
import math
import os
from pathlib import Path
from typing import Optional, Tuple


# Known prime counts for validation
KNOWN_PI_VALUES = {
    10**6: 78498,
    10**7: 664579,
    10**8: 5761455,
    10**9: 50847534,
}


def sieve_of_eratosthenes(limit: int) -> np.ndarray:
    """
    IMPLEMENTED: Basic Sieve of Eratosthenes for small primes.
    
    Generates all primes up to limit using the classic sieve algorithm.
    Used to generate small primes needed for segmented sieving.
    
    Args:
        limit: Upper bound for prime generation (inclusive)
        
    Returns:
        NumPy array of primes up to limit
        
    Algorithm:
        1. Create boolean array of size limit+1, all True
        2. Mark 0, 1 as not prime
        3. For each i from 2 to sqrt(limit):
            - If i is prime, mark all multiples as composite
        4. Return indices where array is True
    """
    if limit < 2:
        return np.array([], dtype=np.uint64)
    
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i:limit+1:i] = False
    
    return np.where(sieve)[0].astype(np.uint64)


def segmented_sieve(limit: int, segment_size: int = 10**6) -> np.ndarray:
    """
    Generate primes up to limit using memory-efficient segmented sieve.
    
    Args:
        limit: Maximum value for prime generation
        segment_size: Size of each segment (default 10^6)
        
    Returns:
        NumPy array of all primes up to limit
    """
    if limit < 2:
        return np.array([], dtype=np.uint64)
    
    # Step 1: Get small primes up to sqrt(limit)
    sqrt_limit = int(math.sqrt(limit)) + 1
    small_primes = sieve_of_eratosthenes(sqrt_limit)
    
    # Step 2: Initialize result with small primes
    result = list(small_primes)
    
    # Step 3: Process segments from sqrt(limit) to limit
    low_start = sqrt_limit if sqrt_limit > small_primes[-1] else small_primes[-1] + 1
    
    for low in range(low_start, limit + 1, segment_size):
        high = min(low + segment_size, limit + 1)
        segment = np.ones(high - low, dtype=bool)
        
        # Mark composites using small primes
        for p in small_primes:
            if p * p > high:
                break
            
            # Find first multiple of p in segment
            start = max(p * p, ((low + p - 1) // p) * p)
            
            # Mark all multiples as composite
            for j in range(start, high, p):
                segment[j - low] = False
        
        # Extract primes from segment
        segment_primes = [low + i for i in range(len(segment)) if segment[i]]
        result.extend(segment_primes)
    
    return np.array(result, dtype=np.uint64)


def generate_primes_to_limit(limit: int, 
                             cache_dir: Optional[str] = None,
                             validate: bool = True) -> np.ndarray:
    """
    Generate or load primes up to limit with optional disk caching.
    
    Args:
        limit: Maximum value for prime generation
        cache_dir: Directory for caching primes (default: ../data)
        validate: Whether to validate count against KNOWN_PI_VALUES
        
    Returns:
        NumPy array of primes up to limit
    """
    # Determine cache directory
    if cache_dir is None:
        script_dir = Path(__file__).parent.parent
        cache_dir = script_dir / 'data'
    else:
        cache_dir = Path(cache_dir)
    
    cache_file = cache_dir / f'primes_{limit}.npy'
    
    # Try to load from cache
    if cache_file.exists():
        primes = np.load(cache_file)
        print(f"Loaded {len(primes):,} primes from cache")
    else:
        # Generate primes
        print(f"Generating primes up to {limit:,}...")
        primes = segmented_sieve(limit)
        
        # Cache to disk
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, primes)
        print(f"Generated and cached {len(primes):,} primes")
    
    # Validate if requested
    if validate and limit in KNOWN_PI_VALUES:
        expected = KNOWN_PI_VALUES[limit]
        actual = len(primes)
        if actual != expected:
            raise ValueError(
                f"Prime count mismatch for limit {limit}: "
                f"expected {expected}, got {actual}"
            )
        print(f"Validation passed: π({limit:,}) = {actual:,}")
    
    return primes


def compute_gaps(primes: np.ndarray, 
                cache_dir: Optional[str] = None,
                limit: Optional[int] = None) -> dict:
    """
    Compute regular and log-gaps with optional disk caching.
    
    Args:
        primes: Array of prime numbers
        cache_dir: Directory for caching gaps
        limit: Max prime value (used for cache filename)
        
    Returns:
        Dictionary with 'primes', 'regular_gaps', 'log_gaps', 'log_primes'
    """
    # Determine cache directory and file
    if cache_dir is None and limit is not None:
        script_dir = Path(__file__).parent.parent
        cache_dir = script_dir / 'data'
    
    if cache_dir is not None and limit is not None:
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / f'gaps_{limit}.npz'
        
        # Try to load from cache
        if cache_file.exists():
            data = np.load(cache_file)
            result = {
                'primes': primes,
                'regular_gaps': data['regular_gaps'],
                'log_gaps': data['log_gaps'],
                'log_primes': data['log_primes']
            }
            print(f"Loaded gaps from cache")
            return result
    
    # Compute gaps
    print(f"Computing gaps...")
    regular_gaps = np.diff(primes)
    log_primes = np.log(primes.astype(np.float64))
    log_gaps = np.diff(log_primes)
    
    # Cache if requested
    if cache_dir is not None and limit is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, 
                regular_gaps=regular_gaps,
                log_gaps=log_gaps,
                log_primes=log_primes)
        print(f"Computed and cached gaps")
    
    return {
        'primes': primes,
        'regular_gaps': regular_gaps,
        'log_gaps': log_gaps,
        'log_primes': log_primes
    }
