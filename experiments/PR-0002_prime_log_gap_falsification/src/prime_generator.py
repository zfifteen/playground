import numpy as np
import math

class PrimeGenerator:
    """
    Implements a Segmented Sieve of Eratosthenes for efficient prime generation
    up to large limits (e.g., 10^9) while keeping memory usage constrained.
    """
    
    @staticmethod
    def simple_sieve(limit):
        """
        Generates primes up to limit using the basic Sieve of Eratosthenes.
        Returns a list of primes.
        """
        if limit < 2:
            return []
        
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0:2] = False
        
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i : limit+1 : i] = False
                
        return np.nonzero(sieve)[0].tolist()

    @staticmethod
    def segmented_sieve(limit, segment_size=1_000_000):
        """
        Generates primes up to limit using a segmented sieve.
        Yields primes as they are found.
        """
        if limit < 2:
            return
            
        # 1. Generate small primes up to sqrt(limit)
        sqrt_limit = int(math.isqrt(limit))
        small_primes = PrimeGenerator.simple_sieve(sqrt_limit)
        
        # Yield small primes first
        for p in small_primes:
            yield p
            
        # 2. Process segments
        low = sqrt_limit + 1
        high = min(low + segment_size, limit + 1)
        
        while low <= limit:
            # Initialize segment sieve
            # segment corresponds to numbers [low, high-1]
            segment_len = high - low
            if segment_len <= 0:
                break
                
            sieve = np.ones(segment_len, dtype=bool)
            
            # Mark composites in this segment using small_primes
            for p in small_primes:
                # For segmented sieve, start marking from max(p*p, first_multiple_in_segment)
                # Since p <= sqrt_limit and low > sqrt_limit, p*p might be less than low,
                # so we take the maximum of p*p and the first multiple of p >= low
                start_val = max(p*p, (low + p - 1) // p * p)
                start_idx = start_val - low
                
                if start_idx < segment_len:
                    sieve[start_idx : segment_len : p] = False
            
            # Yield primes from this segment
            # Indices where sieve is True map to numbers low + index
            primes_indices = np.nonzero(sieve)[0]
            for idx in primes_indices:
                p = low + idx
                if p <= limit:
                    yield int(p)
            
            # Move to next segment
            low = high
            high = min(low + segment_size, limit + 1)

    @staticmethod
    def generate_primes_array(limit):
        """
        Convenience method to generate all primes up to limit and return as a numpy array.
        """
        # For very large limits, this might consume a lot of memory.
        # Use with caution for limit > 10^8 on low memory machines.
        return np.array(list(PrimeGenerator.segmented_sieve(limit)), dtype=np.int64)

if __name__ == "__main__":
    # Quick test
    limit = 100
    print(f"Primes up to {limit}: {list(PrimeGenerator.segmented_sieve(limit))}")