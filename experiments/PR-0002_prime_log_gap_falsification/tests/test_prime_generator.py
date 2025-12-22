import unittest
import numpy as np
from src.prime_generator import PrimeGenerator

class TestPrimeGenerator(unittest.TestCase):
    
    def test_simple_sieve_small(self):
        primes = PrimeGenerator.simple_sieve(30)
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self.assertEqual(primes, expected)

    def test_segmented_sieve_small(self):
        primes = list(PrimeGenerator.segmented_sieve(30))
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        self.assertEqual(primes, expected)

    def test_segmented_sieve_medium(self):
        limit = 1000
        primes = list(PrimeGenerator.segmented_sieve(limit))
        # Pi(1000) = 168
        self.assertEqual(len(primes), 168)
        self.assertEqual(primes[-1], 997)

    def test_known_counts(self):
        # Validate against known Pi(x) values
        # Pi(10^4) = 1229
        limit = 10_000
        count = sum(1 for _ in PrimeGenerator.segmented_sieve(limit))
        self.assertEqual(count, 1229)

        # Pi(10^5) = 9592
        limit = 100_000
        count = sum(1 for _ in PrimeGenerator.segmented_sieve(limit))
        self.assertEqual(count, 9592)

    def test_segment_boundary(self):
        # Test around segment boundaries
        # segment_size default is 1,000,000
        # Let's force a smaller segment size to test logic
        limit = 100
        primes = list(PrimeGenerator.segmented_sieve(limit, segment_size=10))
        expected = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        self.assertEqual(primes, expected)

if __name__ == '__main__':
    unittest.main()