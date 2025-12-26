"""
GUE (Gaussian Unitary Ensemble) comparison and statistical analysis.

This module implements tests to compare zeta zero spacing statistics
against the predictions of Random Matrix Theory (GUE).
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Tuple, Dict, Optional
import warnings


class GUEComparison:
    """Compare zeta zero statistics with GUE predictions."""
    
    @staticmethod
    def wigner_surmise(s: np.ndarray) -> np.ndarray:
        """
        Wigner surmise for GUE (nearest-neighbor spacing distribution).
        
        P(s) = (32/π²) s² exp(-4s²/π)
        
        Args:
            s: Normalized spacing values
            
        Returns:
            Probability density values
        """
        return (32 / np.pi**2) * s**2 * np.exp(-4 * s**2 / np.pi)
    
    @staticmethod
    def poisson_distribution(s: np.ndarray) -> np.ndarray:
        """
        Poisson (independent) spacing distribution.
        
        P(s) = exp(-s)
        
        Args:
            s: Normalized spacing values
            
        Returns:
            Probability density values
        """
        return np.exp(-s)
    
    def test_spacing_distribution(self, normalized_gaps: np.ndarray) -> Dict:
        """
        Test normalized gap distribution against GUE and Poisson.
        
        Args:
            normalized_gaps: Array of normalized gap values
            
        Returns:
            Dictionary with test results
        """
        # Remove NaN and infinite values
        gaps_clean = normalized_gaps[np.isfinite(normalized_gaps)]
        gaps_clean = gaps_clean[gaps_clean > 0]
        
        if len(gaps_clean) == 0:
            return {'error': 'No valid gaps for testing'}
        
        # Kolmogorov-Smirnov test against Wigner surmise
        # Generate theoretical CDF
        s_theory = np.linspace(0, 5, 1000)
        wigner_pdf = self.wigner_surmise(s_theory)
        wigner_cdf = np.cumsum(wigner_pdf) * (s_theory[1] - s_theory[0])
        wigner_cdf /= wigner_cdf[-1]  # Normalize
        
        # KS test (using empirical CDF)
        ks_stat_wigner, p_value_wigner = self._ks_test_custom(
            gaps_clean, s_theory, wigner_cdf
        )
        
        # Poisson CDF: F(s) = 1 - exp(-s)
        poisson_cdf = 1 - np.exp(-s_theory)
        ks_stat_poisson, p_value_poisson = self._ks_test_custom(
            gaps_clean, s_theory, poisson_cdf
        )
        
        # Compute mean and variance
        mean_gap = np.mean(gaps_clean)
        var_gap = np.var(gaps_clean)
        
        # For GUE, mean ≈ 1, variance ≈ 0.286
        # For Poisson, mean = 1, variance = 1
        gue_mean, gue_var = 1.0, 0.286
        poisson_mean, poisson_var = 1.0, 1.0
        
        return {
            'n_gaps': len(gaps_clean),
            'mean_gap': mean_gap,
            'var_gap': var_gap,
            'ks_stat_wigner': ks_stat_wigner,
            'p_value_wigner': p_value_wigner,
            'ks_stat_poisson': ks_stat_poisson,
            'p_value_poisson': p_value_poisson,
            'mean_deviation_gue': abs(mean_gap - gue_mean),
            'var_deviation_gue': abs(var_gap - gue_var),
            'mean_deviation_poisson': abs(mean_gap - poisson_mean),
            'var_deviation_poisson': abs(var_gap - poisson_var),
        }
    
    def _ks_test_custom(self, data: np.ndarray, x_theory: np.ndarray, 
                       cdf_theory: np.ndarray) -> Tuple[float, float]:
        """
        Custom KS test against theoretical CDF.
        
        Args:
            data: Empirical data
            x_theory: x values for theoretical CDF
            cdf_theory: Theoretical CDF values
            
        Returns:
            Tuple of (KS statistic, p-value approximation)
        """
        # Sort data
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        # Empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Interpolate theoretical CDF at data points
        theoretical_cdf_at_data = np.interp(data_sorted, x_theory, cdf_theory)
        
        # KS statistic
        ks_stat = np.max(np.abs(empirical_cdf - theoretical_cdf_at_data))
        
        # Approximate p-value using Kolmogorov distribution
        # For large n, KS statistic ~ sqrt(n) D converges to Kolmogorov distribution
        try:
            from scipy.stats import kstwobign
            p_value = kstwobign.sf(ks_stat * np.sqrt(n))
        except:
            # Fallback approximation
            p_value = 2 * np.exp(-2 * n * ks_stat**2)
        
        return ks_stat, p_value
    
    def test_level_repulsion(self, normalized_gaps: np.ndarray) -> Dict:
        """
        Test for level repulsion (gaps should avoid zero).
        
        GUE predicts P(s) ~ s² near s=0 (level repulsion).
        Poisson predicts P(s) ~ const near s=0 (no repulsion).
        
        Args:
            normalized_gaps: Array of normalized gaps
            
        Returns:
            Dictionary with level repulsion analysis
        """
        gaps_clean = normalized_gaps[np.isfinite(normalized_gaps)]
        gaps_clean = gaps_clean[gaps_clean > 0]
        
        # Count very small gaps
        small_threshold = 0.1
        n_small = np.sum(gaps_clean < small_threshold)
        frac_small = n_small / len(gaps_clean)
        
        # Expected fraction under Poisson: P(s < 0.1) = 1 - exp(-0.1) ≈ 0.095
        expected_poisson = 1 - np.exp(-small_threshold)
        
        # Expected under GUE (numerical integration)
        s_vals = np.linspace(0, small_threshold, 100)
        gue_pdf = self.wigner_surmise(s_vals)
        from scipy import integrate
        expected_gue = integrate.trapezoid(gue_pdf, s_vals)
        
        return {
            'n_gaps': len(gaps_clean),
            'n_small_gaps': n_small,
            'frac_small_gaps': frac_small,
            'expected_frac_poisson': expected_poisson,
            'expected_frac_gue': expected_gue,
            'repulsion_score': frac_small / expected_poisson,  # >1 = less repulsion
        }
    
    def montgomery_pair_correlation(self, zeros: np.ndarray, 
                                   r_max: float = 10.0, 
                                   n_bins: int = 50) -> Dict:
        """
        Compute Montgomery's pair correlation function.
        
        R_2(r) measures correlation between pairs of zeros separated by distance r
        (in units of local average spacing).
        
        GUE prediction: R_2(r) = 1 - (sin(πr)/(πr))²
        
        Args:
            zeros: Array of zero imaginary parts
            r_max: Maximum r value
            n_bins: Number of bins for histogram
            
        Returns:
            Dictionary with pair correlation results
        """
        # Compute all pairwise distances
        n_zeros = len(zeros)
        
        # For large datasets, subsample to avoid O(n²) cost
        max_pairs = 1000000
        if n_zeros * (n_zeros - 1) // 2 > max_pairs:
            # Subsample zeros
            subsample_size = int(np.sqrt(2 * max_pairs))
            indices = np.random.choice(n_zeros, subsample_size, replace=False)
            zeros_sub = zeros[indices]
        else:
            zeros_sub = zeros
        
        # Compute pairwise distances
        distances = []
        for i in range(len(zeros_sub)):
            for j in range(i + 1, len(zeros_sub)):
                distances.append(abs(zeros_sub[j] - zeros_sub[i]))
        
        distances = np.array(distances)
        
        # Normalize by local average spacing
        # Average spacing ≈ 2π/log(γ/(2π)) at height γ
        mean_gamma = np.mean(zeros_sub)
        avg_spacing = 2 * np.pi / np.log(mean_gamma / (2 * np.pi))
        
        normalized_distances = distances / avg_spacing
        
        # Histogram
        hist, bin_edges = np.histogram(normalized_distances, 
                                       bins=n_bins, 
                                       range=(0, r_max),
                                       density=True)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # GUE prediction
        r_theory = bin_centers
        r_theory = np.where(r_theory > 1e-10, r_theory, 1e-10)  # Avoid division by zero
        R2_gue = 1 - (np.sin(np.pi * r_theory) / (np.pi * r_theory))**2
        
        # Poisson prediction: R_2(r) = 1 (no correlation)
        R2_poisson = np.ones_like(r_theory)
        
        return {
            'r_values': bin_centers.tolist(),
            'R2_empirical': hist.tolist(),
            'R2_gue': R2_gue.tolist(),
            'R2_poisson': R2_poisson.tolist(),
            'n_pairs': len(distances),
            'avg_spacing': avg_spacing,
        }


class MultiplicitySensitivity:
    """Analyze sensitivity to zero multiplicity."""
    
    @staticmethod
    def detect_anomalous_contractions(deltas: np.ndarray, threshold: float = 3.0) -> Dict:
        """
        Detect anomalously small gaps that could indicate multiplicity.
        
        Multiple zeros would create δ_n ≈ 0, spiking the B term in Z-transform.
        
        Args:
            deltas: Array of gap differences
            threshold: Number of standard deviations below mean for anomaly
            
        Returns:
            Dictionary with anomaly detection results
        """
        deltas_clean = deltas[np.isfinite(deltas)]
        
        mean_delta = np.mean(deltas_clean)
        std_delta = np.std(deltas_clean)
        
        # Anomaly threshold
        anomaly_threshold = mean_delta - threshold * std_delta
        
        # Find anomalies
        anomalies = deltas_clean < anomaly_threshold
        n_anomalies = np.sum(anomalies)
        
        # Get indices and values of anomalies
        anomaly_indices = np.where(anomalies)[0]
        anomaly_values = deltas_clean[anomalies]
        
        return {
            'mean_delta': mean_delta,
            'std_delta': std_delta,
            'anomaly_threshold': anomaly_threshold,
            'n_anomalies': n_anomalies,
            'frac_anomalies': n_anomalies / len(deltas_clean),
            'anomaly_indices': anomaly_indices[:20].tolist(),  # First 20
            'anomaly_values': anomaly_values[:20].tolist(),
            'smallest_gap': np.min(deltas_clean),
            'smallest_gap_normalized': np.min(deltas_clean) / mean_delta,
        }
    
    @staticmethod
    def compute_b_term_spikes(deltas: np.ndarray, threshold: float = 5.0) -> Dict:
        """
        Compute spikes in B term: log(δ_{n+1}/δ_n).
        
        Large spikes indicate sudden contractions/expansions.
        
        Args:
            deltas: Array of gap differences
            threshold: Number of standard deviations for spike detection
            
        Returns:
            Dictionary with spike analysis
        """
        deltas_clean = deltas[np.isfinite(deltas)]
        deltas_clean = deltas_clean[deltas_clean > 0]
        
        # Compute B term
        ratios = deltas_clean[1:] / deltas_clean[:-1]
        log_ratios = np.log(ratios)
        
        # Detect spikes
        mean_b = np.mean(log_ratios)
        std_b = np.std(log_ratios)
        
        spike_threshold_high = mean_b + threshold * std_b
        spike_threshold_low = mean_b - threshold * std_b
        
        spikes_high = log_ratios > spike_threshold_high
        spikes_low = log_ratios < spike_threshold_low
        
        return {
            'mean_b_term': mean_b,
            'std_b_term': std_b,
            'n_spikes_high': np.sum(spikes_high),
            'n_spikes_low': np.sum(spikes_low),
            'n_spikes_total': np.sum(spikes_high) + np.sum(spikes_low),
            'frac_spikes': (np.sum(spikes_high) + np.sum(spikes_low)) / len(log_ratios),
            'max_b_term': np.max(log_ratios),
            'min_b_term': np.min(log_ratios),
        }


if __name__ == "__main__":
    # Test GUE comparison
    print("Testing GUE comparison module...")
    
    from zeta_zeros import ZetaZerosDataset, compute_normalized_gaps
    
    dataset = ZetaZerosDataset()
    zeros = dataset.get_zeros(n_zeros=1000, method='mpmath', precision=50)
    
    print(f"\nLoaded {len(zeros)} zeros")
    
    # Compute normalized gaps
    normalized_gaps = compute_normalized_gaps(zeros)
    print(f"Computed {len(normalized_gaps)} normalized gaps")
    
    # GUE comparison
    gue = GUEComparison()
    
    print("\n=== Spacing Distribution Test ===")
    spacing_test = gue.test_spacing_distribution(normalized_gaps)
    for key, value in spacing_test.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== Level Repulsion Test ===")
    repulsion = gue.test_level_repulsion(normalized_gaps)
    for key, value in repulsion.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n=== Multiplicity Sensitivity ===")
    deltas = np.diff(zeros)
    sensitivity = MultiplicitySensitivity()
    
    anomalies = sensitivity.detect_anomalous_contractions(deltas)
    print("Anomalous contractions:")
    for key, value in anomalies.items():
        if isinstance(value, (list,)):
            print(f"  {key}: {value}")
        elif isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    spikes = sensitivity.compute_b_term_spikes(deltas)
    print("\nB-term spikes:")
    for key, value in spikes.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
