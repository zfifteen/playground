"""
Module for acquiring Riemann zeta zeros data.

This module provides functionality to fetch zeta zeros from LMFDB or compute them
using mpmath for the Z-domain framework experiment.
"""

import numpy as np
import mpmath as mp
from pathlib import Path
import json
import pickle
from typing import Tuple, List
import requests
from tqdm import tqdm


class ZetaZerosDataset:
    """Handler for Riemann zeta zeros data acquisition and caching."""
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the dataset handler.
        
        Args:
            cache_dir: Directory for caching zeros data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_from_lmfdb(self, n_zeros: int = 10000, start_index: int = 1) -> np.ndarray:
        """
        Attempt to fetch zeta zeros from LMFDB database.
        
        Args:
            n_zeros: Number of zeros to fetch
            start_index: Starting index (1-based)
            
        Returns:
            Array of imaginary parts of zeros on critical line
            
        Note:
            LMFDB API may have limitations. Falls back to mpmath if unavailable.
        """
        cache_file = self.cache_dir / f"lmfdb_zeros_{start_index}_{n_zeros}.npy"
        
        if cache_file.exists():
            print(f"Loading zeros from cache: {cache_file}")
            return np.load(cache_file)
        
        print(f"Attempting to fetch {n_zeros} zeros from LMFDB...")
        try:
            # LMFDB API endpoint (this is a simplified approach)
            # In practice, LMFDB may require different handling
            url = "https://www.lmfdb.org/api/zeros/zeta"
            # Note: The actual API may differ; this is illustrative
            
            # For this experiment, we'll use mpmath instead
            raise NotImplementedError("LMFDB API integration pending")
            
        except Exception as e:
            print(f"LMFDB fetch failed: {e}")
            print("Falling back to mpmath computation...")
            return self.compute_with_mpmath(n_zeros, start_index)
    
    def compute_with_mpmath(self, n_zeros: int = 10000, start_index: int = 1,
                           precision: int = 50) -> np.ndarray:
        """
        Compute zeta zeros using mpmath library.
        
        Args:
            n_zeros: Number of zeros to compute
            start_index: Starting index (1-based)
            precision: Decimal precision for computation
            
        Returns:
            Array of imaginary parts of zeros on critical line
        """
        cache_file = self.cache_dir / f"mpmath_zeros_{start_index}_{n_zeros}_p{precision}.npy"
        
        if cache_file.exists():
            print(f"Loading zeros from cache: {cache_file}")
            return np.load(cache_file)
        
        print(f"Computing {n_zeros} zeta zeros with mpmath (precision={precision})...")
        mp.mp.dps = precision
        
        zeros = []
        for n in tqdm(range(start_index, start_index + n_zeros), desc="Computing zeros"):
            # Find the nth zero using mpmath
            # mpmath.zetazero(n) returns the nth nontrivial zero
            try:
                zero = mp.zetazero(n)
                # Extract imaginary part (zeros are on critical line s = 1/2 + it)
                gamma_n = float(mp.im(zero))
                zeros.append(gamma_n)
            except Exception as e:
                print(f"Error computing zero {n}: {e}")
                # Use approximate formula for high zeros if computation fails
                # Approximation: γ_n ≈ 2πn/log(n/(2πe))
                if n > 1:
                    gamma_n = 2 * np.pi * n / np.log(n / (2 * np.pi * np.e))
                    zeros.append(gamma_n)
        
        zeros_array = np.array(zeros, dtype=np.float64)
        
        # Cache the results
        np.save(cache_file, zeros_array)
        print(f"Cached zeros to: {cache_file}")
        
        return zeros_array
    
    def load_precomputed_zeros(self, filepath: str) -> np.ndarray:
        """
        Load precomputed zeros from a file.
        
        Args:
            filepath: Path to file containing zeros (npy, txt, or csv)
            
        Returns:
            Array of zeros
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npy':
            return np.load(filepath)
        elif filepath.suffix in ['.txt', '.csv']:
            return np.loadtxt(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def get_zeros(self, n_zeros: int = 10000, start_index: int = 1,
                  method: str = 'mpmath', **kwargs) -> np.ndarray:
        """
        Get zeta zeros using specified method.
        
        Args:
            n_zeros: Number of zeros to get
            start_index: Starting index (1-based)
            method: Method to use ('mpmath', 'lmfdb', 'file')
            **kwargs: Additional arguments for specific methods
            
        Returns:
            Array of imaginary parts of zeros
        """
        if method == 'mpmath':
            return self.compute_with_mpmath(n_zeros, start_index, 
                                           kwargs.get('precision', 50))
        elif method == 'lmfdb':
            return self.fetch_from_lmfdb(n_zeros, start_index)
        elif method == 'file':
            return self.load_precomputed_zeros(kwargs['filepath'])
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_known_zeros_sample(self) -> np.ndarray:
        """
        Return a small sample of known zeta zeros for testing.
        
        Returns:
            Array of first 10 known zeros (imaginary parts)
        """
        # First 10 nontrivial zeros (imaginary parts, γ_n)
        known_zeros = np.array([
            14.134725141734693790,
            21.022039638771554993,
            25.010857580145688763,
            30.424876125859513210,
            32.935061587739189691,
            37.586178158825671257,
            40.918719012147495187,
            43.327073280914999519,
            48.005150881167159727,
            49.773832477672302181
        ])
        return known_zeros


def compute_zero_differences(zeros: np.ndarray) -> np.ndarray:
    """
    Compute differences between consecutive zeros: δ_n = γ_{n+1} - γ_n
    
    Args:
        zeros: Array of zero imaginary parts
        
    Returns:
        Array of differences (length = len(zeros) - 1)
    """
    return np.diff(zeros)


def compute_normalized_gaps(zeros: np.ndarray) -> np.ndarray:
    """
    Compute normalized gaps according to local density.
    
    Under GUE hypothesis, normalized gaps should follow specific distribution.
    
    Args:
        zeros: Array of zero imaginary parts
        
    Returns:
        Array of normalized gaps
    """
    deltas = compute_zero_differences(zeros)
    
    # Local average spacing: mean gap in neighborhood
    # For large γ, average spacing ≈ 2π/log(γ/(2π))
    midpoints = (zeros[:-1] + zeros[1:]) / 2
    local_spacing = 2 * np.pi / np.log(midpoints / (2 * np.pi))
    
    # Normalize gaps by local spacing
    normalized_gaps = deltas / local_spacing
    
    return normalized_gaps


if __name__ == "__main__":
    # Test the module
    print("Testing ZetaZerosDataset...")
    
    dataset = ZetaZerosDataset()
    
    # Test with known zeros sample
    print("\n1. Testing with known zeros sample:")
    known_zeros = dataset.get_known_zeros_sample()
    print(f"First 5 zeros: {known_zeros[:5]}")
    
    deltas = compute_zero_differences(known_zeros)
    print(f"First 5 differences: {deltas[:5]}")
    
    # Test mpmath computation with small sample
    print("\n2. Testing mpmath computation (first 100 zeros):")
    zeros = dataset.get_zeros(n_zeros=100, method='mpmath', precision=50)
    print(f"Computed {len(zeros)} zeros")
    print(f"First 5: {zeros[:5]}")
    print(f"Last 5: {zeros[-5:]}")
    
    # Compute statistics
    deltas = compute_zero_differences(zeros)
    print(f"\nGap statistics:")
    print(f"  Mean: {np.mean(deltas):.4f}")
    print(f"  Std: {np.std(deltas):.4f}")
    print(f"  Min: {np.min(deltas):.4f}")
    print(f"  Max: {np.max(deltas):.4f}")
