import numpy as np
import math
from typing import List, Generator


def sieve_of_eratosthenes(limit: int) -> List[int]:
    """
    Basic sieve to generate primes up to sqrt(limit).
    """
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def segmented_sieve(
    limit: int, segment_size: int = 10**6
) -> Generator[int, None, None]:
    """
    Memory-efficient prime generation using segmented sieve.

    1. Generate small primes up to sqrt(limit) using basic sieve
    2. Process [sqrt(limit), limit] in segments of segment_size
    3. For each segment, mark composites using small primes
    4. Yield primes from each segment
    """
    if limit < 2:
        return

    # Step 1: Handle small primes
    sqrt_limit = int(math.sqrt(limit)) + 1
    small_primes = sieve_of_eratosthenes(sqrt_limit)
    yield from small_primes

    # Step 2: Segmented sieve
    low_start = max(sqrt_limit, small_primes[-1] + 1) if small_primes else 2
    for low in range(low_start, limit + 1, segment_size):
        high = min(low + segment_size, limit + 1)
        sieve = [True] * (high - low)
        for p in small_primes:
            if p * p > high:
                break
            # Find the smallest multiple of p in [low, high)
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high, p):
                sieve[j - low] = False
        # Yield primes in this segment
        for i in range(high - low):
            if sieve[i]:
                yield low + i


def generate_primes_up_to(limit: int) -> np.ndarray:
    """
    Generate all primes up to limit as numpy array.
    """
    primes = list(segmented_sieve(limit))
    return np.array(primes, dtype=np.uint64)


def count_primes_up_to(limit: int) -> int:
    """
    Count primes up to limit without storing all primes.
    """
    return sum(1 for _ in segmented_sieve(limit))


if __name__ == "__main__":
    # Test
    print("Testing prime generation...")
    primes = generate_primes_up_to(100)
    print(f"Primes up to 100: {primes}")
    print(f"Count: {len(primes)}")
