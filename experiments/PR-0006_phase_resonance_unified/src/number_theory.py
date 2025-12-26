"""
Number Theory: Phase-Resonance for Semiprime Factorization

This module implements geometric resonance using irrational constants
(golden ratio φ and Euler's number e) to detect factors of semiprimes.
"""

import numpy as np
from typing import List, Tuple, Dict
import math


# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio: 1.618...
E = np.e  # Euler's number: 2.718...


def generate_semiprimes(n: int, min_value: int = 100, max_value: int = 10000) -> List[Tuple[int, int, int]]:
    """
    IMPLEMENTED: Generate random semiprimes (products of two primes) for testing.
    
    Args:
        n: Number of semiprimes to generate
        min_value: Minimum value for prime factors
        max_value: Maximum value for prime factors
    
    Returns:
        List of tuples (semiprime, prime1, prime2) where semiprime = prime1 * prime2
    """
    # Generate primes using simple sieve
    def sieve_of_eratosthenes(limit):
        """Generate all primes up to limit"""
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        
        return [i for i in range(2, limit + 1) if is_prime[i]]
    
    # Get all primes in range
    primes = sieve_of_eratosthenes(max_value)
    primes = [p for p in primes if p >= min_value]
    
    if len(primes) < 2:
        raise ValueError(f"Not enough primes in range [{min_value}, {max_value}]")
    
    # Generate random semiprime pairs
    semiprimes = []
    rng = np.random.RandomState(42)  # Fixed seed for reproducibility
    
    for _ in range(n):
        # Select two random primes
        idx1, idx2 = rng.choice(len(primes), size=2, replace=True)
        p1, p2 = primes[idx1], primes[idx2]
        
        # Ensure p1 <= p2 for consistency
        if p1 > p2:
            p1, p2 = p2, p1
        
        semiprime = p1 * p2
        semiprimes.append((semiprime, p1, p2))
    
    return semiprimes


def compute_resonance(n: int, k: int, theta: float = 0.0) -> float:
    """
    IMPLEMENTED: Compute geometric resonance value for candidate divisor k.
    
    Uses irrational constants φ (golden ratio) and e (Euler's number) to
    create phase-aligned resonance patterns that may reveal prime factors.
    
    Args:
        n: The number being factored (semiprime)
        k: Candidate divisor to test
        theta: Phase offset, default 0.0
    
    Returns:
        float - resonance value R(k). Higher values indicate stronger resonance.
    """
    # Validate inputs
    if k <= 1:
        return 0.0  # Invalid divisor
    
    # Compute ln(k) safely
    log_k = np.log(k)
    
    if log_k == 0:  # k == 1, already handled above but be safe
        return 0.0
    
    # Calculate first term: cos(θ + ln(k)·φ) / ln(k)
    # This uses the golden ratio for phase alignment
    term1 = np.cos(theta + log_k * PHI) / log_k
    
    # Calculate second term: cos(ln(k)·e) · 0.5
    # This uses Euler's number for additional resonance component
    term2 = np.cos(log_k * E) * 0.5
    
    # Return sum of both terms
    resonance = term1 + term2
    
    return resonance


def scan_divisors(n: int, theta: float = 0.0, max_k: int = None) -> np.ndarray:
    """
    IMPLEMENTED: Scan all candidate divisors of n and compute resonance for each.
    
    Args:
        n: The semiprime to analyze
        theta: Phase offset for resonance
        max_k: Maximum divisor to test (default: sqrt(n))
    
    Returns:
        np.ndarray - resonance values for each candidate divisor
        Array index i corresponds to divisor i+2 (skip 0 and 1)
    """
    # Set max_k to sqrt(n) if not provided
    if max_k is None:
        max_k = int(np.sqrt(n)) + 1
    
    # Create array of candidates from 2 to max_k
    candidates = np.arange(2, max_k + 1)
    
    # For each k, call compute_resonance(n, k, theta)
    resonance_values = np.array([compute_resonance(n, k, theta) for k in candidates])
    
    return resonance_values


def detect_peaks(resonance_values: np.ndarray, threshold: float = None) -> List[int]:
    """
    IMPLEMENTED: Identify peaks in resonance signal that may indicate factors.
    
    Args:
        resonance_values: Output from scan_divisors()
        threshold: Minimum resonance value to consider (default: mean + 2*std)
    
    Returns:
        List[int] - candidate divisor values (not indices)
        May include false positives; needs validation against true factors
    """
    if len(resonance_values) == 0:
        return []
    
    # Calculate threshold if not provided (mean + 2*std)
    if threshold is None:
        mean_val = np.mean(resonance_values)
        std_val = np.std(resonance_values)
        threshold = mean_val + 2.0 * std_val
    
    # Find local maxima using comparison with neighbors
    # A point is a local maximum if it's greater than both neighbors
    peaks_indices = []
    
    for i in range(len(resonance_values)):
        # Get neighbor values (handle edges)
        left_val = resonance_values[i-1] if i > 0 else -np.inf
        center_val = resonance_values[i]
        right_val = resonance_values[i+1] if i < len(resonance_values)-1 else -np.inf
        
        # Check if local maximum
        is_local_max = (center_val > left_val) and (center_val > right_val)
        
        # Filter peaks above threshold
        if is_local_max and center_val >= threshold:
            peaks_indices.append(i)
    
    # Convert array indices back to divisor values (add 2)
    # Because scan_divisors starts from divisor=2 at index=0
    candidate_divisors = [idx + 2 for idx in peaks_indices]
    
    # Return sorted list of candidate factors
    return sorted(candidate_divisors)


def evaluate_factor_detection(semiprime: int, true_factors: Tuple[int, int], 
                              detected_peaks: List[int], tolerance: int = 1) -> Dict[str, float]:
    # PURPOSE: Evaluate accuracy of resonance-based factor detection
    # INPUTS:
    #   - semiprime (int): The number that was factored
    #   - true_factors (tuple): (p1, p2) true prime factors
    #   - detected_peaks (list): Candidate factors from detect_peaks() [IMPLEMENTED ✓]
    #   - tolerance (int): Allow detection within ±tolerance of true factor
    # PROCESS:
    #   1. Check if p1 is in detected_peaks (within tolerance)
    #   2. Check if p2 is in detected_peaks (within tolerance)
    #   3. Count false positives (detected peaks that aren't factors)
    #   4. Calculate precision = true_positives / (true_positives + false_positives)
    #   5. Calculate recall = true_positives / 2 (two factors to find)
    #   6. Return metrics dictionary
    # OUTPUTS: Dict with keys 'precision', 'recall', 'f1_score', 'true_positives', 'false_positives'
    # DEPENDENCIES: Basic Python arithmetic
    # NOTES: Perfect detection = precision 1.0, recall 1.0
    pass


def compute_snr_at_factors(n: int, true_factors: Tuple[int, int], 
                          resonance_values: np.ndarray) -> float:
    # PURPOSE: Calculate signal-to-noise ratio of resonance at true factors
    # INPUTS:
    #   - n (int): The semiprime
    #   - true_factors (tuple): (p1, p2) true prime factors
    #   - resonance_values (np.ndarray): Resonance scan from scan_divisors()
    # PROCESS:
    #   1. Extract resonance values at true factor positions (p1-2, p2-2 as indices)
    #   2. Calculate signal = mean of resonance at true factors
    #   3. Calculate noise = mean of all other resonance values
    #   4. Compute SNR = signal / noise (or in dB: 20*log10(signal/noise))
    #   5. Return SNR value
    # OUTPUTS: float - signal-to-noise ratio
    # DEPENDENCIES: numpy mean, log10
    # NOTES: Higher SNR indicates clearer factor detection
    pass


def run_batch_analysis(semiprimes: List[Tuple[int, int, int]], 
                      theta: float = 0.0) -> Dict[str, any]:
    # PURPOSE: Run resonance analysis on multiple semiprimes and aggregate results
    # INPUTS:
    #   - semiprimes (list): Output from generate_semiprimes() [IMPLEMENTED ✓]
    #   - theta (float): Phase offset for all analyses
    # PROCESS:
    #   1. Initialize results collectors (lists for metrics)
    #   2. For each (semiprime, p1, p2) in semiprimes:
    #      a. Call scan_divisors(semiprime, theta)
    #      b. Call detect_peaks(resonance_values)
    #      c. Call evaluate_factor_detection(semiprime, (p1,p2), detected_peaks)
    #      d. Call compute_snr_at_factors(semiprime, (p1,p2), resonance_values)
    #      e. Store all metrics
    #   3. Calculate aggregate statistics (mean, std, median for each metric)
    #   4. Return comprehensive results dictionary
    # OUTPUTS: Dict with aggregated metrics and individual results
    # DEPENDENCIES: All above functions [SOME IMPLEMENTED ✓]
    # NOTES: This is the main entry point for number theory experiments
    pass


def compare_to_random_baseline(semiprimes: List[Tuple[int, int, int]]) -> Dict[str, float]:
    # PURPOSE: Generate random baseline for comparison with resonance method
    # INPUTS:
    #   - semiprimes (list): Same dataset used for resonance analysis
    # PROCESS:
    #   1. For each semiprime, generate random "detected" factors
    #   2. Apply same evaluation metrics (precision, recall, SNR)
    #   3. Aggregate results same way as run_batch_analysis()
    #   4. Return baseline performance metrics
    # OUTPUTS: Dict with baseline metrics
    # DEPENDENCIES: evaluate_factor_detection() [TO BE IMPLEMENTED]
    # NOTES: Should show much worse performance than resonance if method works
    pass
