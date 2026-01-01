"""
Z-domain transform implementation for Riemann hypothesis verification.

This module implements the Z-transform framework as described in the hypothesis:

    Z_n = δ_n · log(δ_{n+1}/δ_n) / ((1/2)log γ_k)

where:
- δ_n = γ_{n+1} - γ_n (raw gap structure)
- log(δ_{n+1}/δ_n) captures multiplicative rate changes
- (1/2)log γ_k is the RH-bounded normalizer from O(γ^{1/2} log γ)
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class ZTransform:
    """Z-domain transform for zeta zero differences."""
    
    def __init__(self, zeros: np.ndarray):
        """
        Initialize Z-transform with zeta zeros.
        
        Args:
            zeros: Array of imaginary parts of zeta zeros (γ_n values)
        """
        self.zeros = zeros
        self.n_zeros = len(zeros)
        
        # Precompute differences
        self.deltas = np.diff(zeros)  # δ_n = γ_{n+1} - γ_n
        
    def compute_z_values(self, k: int = 0) -> np.ndarray:
        """
        Compute Z-transform values.
        
        Z_n = δ_n · log(δ_{n+1}/δ_n) / ((1/2)log γ_k)
        
        Args:
            k: Starting index for γ_k in the normalizer (default: 0)
            
        Returns:
            Array of Z_n values (length = n_zeros - 2)
            
        Note:
            We lose 2 elements: one from diff, one from consecutive ratio
        """
        # Need at least 3 zeros to compute Z-transform
        if self.n_zeros < 3:
            raise ValueError("Need at least 3 zeros for Z-transform")
        
        # Component A: δ_n (gaps)
        delta_n = self.deltas[:-1]  # δ_n for n = 0 to N-2
        delta_n_plus_1 = self.deltas[1:]  # δ_{n+1} for n = 0 to N-2
        
        # Component B: Multiplicative rate log(δ_{n+1}/δ_n)
        # Protect against division by zero and log of negative/zero
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ratio = delta_n_plus_1 / delta_n
            # Filter out non-positive ratios (should be rare for zeta zeros)
            ratio = np.where(ratio > 0, ratio, np.nan)
            log_ratio = np.log(ratio)
        
        # Component C: RH-bounded normalizer (1/2)log γ_k
        # Use γ_k corresponding to each δ_n
        gamma_k = self.zeros[k:-2]  # Align with delta_n indices
        
        # Protect against log of values <= 0
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            normalizer = 0.5 * np.log(gamma_k)
            normalizer = np.where(gamma_k > 0, normalizer, np.nan)
        
        # Compute Z_n = A · B / C
        Z_n = delta_n * log_ratio / normalizer
        
        return Z_n
    
    def compute_phases(self, k: int = 0) -> np.ndarray:
        """
        Compute phase mapping θ_n = 2π(Z_n mod 1).
        
        This projects Z_n values onto the unit circle [0, 2π).
        
        Args:
            k: Starting index for γ_k in the normalizer
            
        Returns:
            Array of phases in [0, 2π)
        """
        Z_n = self.compute_z_values(k)
        
        # Remove NaN values
        Z_n_clean = Z_n[~np.isnan(Z_n)]
        
        if len(Z_n_clean) == 0:
            raise ValueError("All Z_n values are NaN")
        
        # Map to [0, 2π) via modulo operation
        theta_n = 2 * np.pi * np.mod(Z_n_clean, 1)
        
        return theta_n
    
    def compute_alternative_z(self) -> np.ndarray:
        """
        Compute alternative Z-transform using local average spacing.
        
        This variant uses the theoretical average spacing 2π/log(γ/(2π))
        as the normalizer instead of (1/2)log γ_k.
        
        Returns:
            Array of alternative Z values
        """
        delta_n = self.deltas[:-1]
        delta_n_plus_1 = self.deltas[1:]
        
        # Multiplicative rate
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            ratio = delta_n_plus_1 / delta_n
            ratio = np.where(ratio > 0, ratio, np.nan)
            log_ratio = np.log(ratio)
        
        # Alternative normalizer: 2π/log(γ/(2π))
        gamma_k = self.zeros[:-2]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            normalizer = 2 * np.pi / np.log(gamma_k / (2 * np.pi))
            normalizer = np.where(gamma_k > 0, normalizer, np.nan)
        
        # Z_n = δ_n · log(δ_{n+1}/δ_n) / normalizer
        Z_n_alt = delta_n * log_ratio / normalizer
        
        return Z_n_alt
    
    def get_statistics(self, k: int = 0) -> dict:
        """
        Compute statistics for Z-transform values.
        
        Args:
            k: Starting index for γ_k
            
        Returns:
            Dictionary with statistical measures
        """
        Z_n = self.compute_z_values(k)
        Z_n_clean = Z_n[~np.isnan(Z_n)]
        
        theta_n = self.compute_phases(k)
        
        stats = {
            'n_values': len(Z_n_clean),
            'n_nans': np.sum(np.isnan(Z_n)),
            'z_mean': np.mean(Z_n_clean),
            'z_std': np.std(Z_n_clean),
            'z_min': np.min(Z_n_clean),
            'z_max': np.max(Z_n_clean),
            'z_median': np.median(Z_n_clean),
            'z_skewness': self._compute_skewness(Z_n_clean),
            'z_kurtosis': self._compute_kurtosis(Z_n_clean),
            'phase_mean': np.mean(theta_n),
            'phase_std': np.std(theta_n),
            'phase_min': np.min(theta_n),
            'phase_max': np.max(theta_n),
        }
        
        return stats
    
    @staticmethod
    def _compute_skewness(data: np.ndarray) -> float:
        """Compute skewness of data."""
        from scipy import stats
        return stats.skew(data)
    
    @staticmethod
    def _compute_kurtosis(data: np.ndarray) -> float:
        """Compute kurtosis of data."""
        from scipy import stats
        return stats.kurtosis(data)
    
    def analyze_phase_clustering(self, k: int = 0, n_bins: int = 36) -> dict:
        """
        Analyze phase clustering on the unit circle.
        
        Under RH with simple zeros, phases should be dispersed.
        Clustering indicates anomalous structure.
        
        Args:
            k: Starting index for γ_k
            n_bins: Number of bins for phase histogram (default: 36 = 10° bins)
            
        Returns:
            Dictionary with clustering analysis results
        """
        theta_n = self.compute_phases(k)
        
        # Histogram of phases
        hist, bin_edges = np.histogram(theta_n, bins=n_bins, range=(0, 2*np.pi))
        
        # Expected count under uniform distribution
        expected_count = len(theta_n) / n_bins
        
        # Chi-square test for uniformity
        chi_square = np.sum((hist - expected_count)**2 / expected_count)
        
        # Degrees of freedom = n_bins - 1
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_square, df=n_bins - 1)
        
        # Compute circular variance (1 = uniform, 0 = concentrated)
        # R = |<exp(iθ)>| where <> is mean
        complex_phases = np.exp(1j * theta_n)
        mean_resultant_length = np.abs(np.mean(complex_phases))
        circular_variance = 1 - mean_resultant_length
        
        # Compute entropy of phase distribution
        prob = hist / len(theta_n)
        prob = prob[prob > 0]  # Remove zeros for log
        entropy = -np.sum(prob * np.log(prob))
        max_entropy = np.log(n_bins)  # For uniform distribution
        normalized_entropy = entropy / max_entropy
        
        return {
            'n_bins': n_bins,
            'chi_square': chi_square,
            'p_value': p_value,
            'circular_variance': circular_variance,
            'mean_resultant_length': mean_resultant_length,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': normalized_entropy,
            'histogram': hist.tolist(),
            'bin_edges': bin_edges.tolist(),
        }


def compute_gap_autocorrelation(deltas: np.ndarray, max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute autocorrelation of gap sequence.
    
    Args:
        deltas: Array of gap differences
        max_lag: Maximum lag to compute
        
    Returns:
        Tuple of (lags, autocorrelations)
    """
    from statsmodels.tsa.stattools import acf
    
    acf_values = acf(deltas, nlags=max_lag, fft=True)
    lags = np.arange(max_lag + 1)
    
    return lags, acf_values


if __name__ == "__main__":
    # Test the Z-transform
    print("Testing Z-transform implementation...")
    
    # Use known zeros for testing
    from zeta_zeros import ZetaZerosDataset
    
    dataset = ZetaZerosDataset()
    zeros = dataset.get_zeros(n_zeros=1000, method='mpmath', precision=50)
    
    print(f"\nLoaded {len(zeros)} zeros")
    print(f"Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
    
    # Create Z-transform
    ztrans = ZTransform(zeros)
    
    # Compute Z values
    print("\nComputing Z-transform values...")
    Z_n = ztrans.compute_z_values(k=0)
    print(f"Computed {len(Z_n)} Z values")
    print(f"NaN count: {np.sum(np.isnan(Z_n))}")
    
    Z_clean = Z_n[~np.isnan(Z_n)]
    if len(Z_clean) > 0:
        print(f"Mean: {np.mean(Z_clean):.4f}")
        print(f"Std: {np.std(Z_clean):.4f}")
        print(f"Range: [{np.min(Z_clean):.4f}, {np.max(Z_clean):.4f}]")
    
    # Compute phases
    print("\nComputing phase mapping...")
    theta = ztrans.compute_phases(k=0)
    print(f"Computed {len(theta)} phases")
    print(f"Mean phase: {np.mean(theta):.4f} rad ({np.degrees(np.mean(theta)):.2f}°)")
    print(f"Std phase: {np.std(theta):.4f} rad ({np.degrees(np.std(theta)):.2f}°)")
    
    # Get statistics
    print("\nZ-transform statistics:")
    stats = ztrans.get_statistics(k=0)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Analyze phase clustering
    print("\nPhase clustering analysis:")
    clustering = ztrans.analyze_phase_clustering(k=0, n_bins=36)
    print(f"  Chi-square: {clustering['chi_square']:.4f}")
    print(f"  p-value: {clustering['p_value']:.4e}")
    print(f"  Circular variance: {clustering['circular_variance']:.4f}")
    print(f"  Mean resultant length: {clustering['mean_resultant_length']:.4f}")
    print(f"  Normalized entropy: {clustering['normalized_entropy']:.4f}")
    
    if clustering['p_value'] < 0.05:
        print("  ⚠️  Phases are NOT uniformly distributed (reject uniformity)")
    else:
        print("  ✓ Phases appear uniformly distributed")
