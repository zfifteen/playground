"""
Visualization module for Z-domain framework experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import warnings

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ZDomainVisualizer:
    """Visualization tools for Z-domain analysis."""
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory for saving plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_z_transform(self, Z_n: np.ndarray, filename: str = "z_transform.png"):
        """
        Plot Z-transform values.
        
        Args:
            Z_n: Array of Z-transform values
            filename: Output filename
        """
        Z_clean = Z_n[np.isfinite(Z_n)]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Time series
        axes[0, 0].plot(Z_clean, alpha=0.6, linewidth=0.5)
        axes[0, 0].set_xlabel('Index n')
        axes[0, 0].set_ylabel('Z_n')
        axes[0, 0].set_title('Z-Transform Time Series')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram
        axes[0, 1].hist(Z_clean, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Z_n')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Z-Transform Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot (normality check)
        from scipy import stats
        stats.probplot(Z_clean, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 1].boxplot(Z_clean, vert=True)
        axes[1, 1].set_ylabel('Z_n')
        axes[1, 1].set_title('Box Plot of Z-Transform Values')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_phase_circle(self, theta: np.ndarray, filename: str = "phase_circle.png"):
        """
        Plot phases on unit circle.
        
        Args:
            theta: Array of phase values in [0, 2π)
            filename: Output filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Polar plot
        ax_polar = plt.subplot(1, 2, 1, projection='polar')
        ax_polar.scatter(theta, np.ones_like(theta), alpha=0.3, s=5)
        ax_polar.set_title('Phase Distribution on Unit Circle', pad=20)
        ax_polar.set_ylim([0, 1.2])
        
        # Histogram
        axes[1].hist(theta, bins=36, alpha=0.7, edgecolor='black', range=(0, 2*np.pi))
        axes[1].axhline(len(theta)/36, color='red', linestyle='--', 
                       label='Expected (uniform)')
        axes[1].set_xlabel('Phase θ (radians)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Phase Histogram (36 bins = 10° each)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_autocorrelation(self, data: np.ndarray, max_lag: int = 50,
                           title: str = "Autocorrelation", 
                           filename: str = "autocorrelation.png"):
        """
        Plot autocorrelation function.
        
        Args:
            data: Time series data
            max_lag: Maximum lag
            title: Plot title
            filename: Output filename
        """
        from statsmodels.tsa.stattools import acf, pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # ACF
        plot_acf(data, lags=max_lag, ax=axes[0], alpha=0.05)
        axes[0].set_title(f'{title} - ACF')
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(data, lags=max_lag, ax=axes[1], alpha=0.05, method='ywm')
        axes[1].set_title(f'{title} - PACF')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_gue_comparison(self, normalized_gaps: np.ndarray,
                           filename: str = "gue_comparison.png"):
        """
        Plot comparison with GUE predictions.
        
        Args:
            normalized_gaps: Array of normalized gap values
            filename: Output filename
        """
        from gue_analysis import GUEComparison
        
        gaps_clean = normalized_gaps[np.isfinite(normalized_gaps)]
        gaps_clean = gaps_clean[gaps_clean > 0]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram with theoretical distributions
        s_vals = np.linspace(0, 4, 100)
        gue = GUEComparison()
        wigner_pdf = gue.wigner_surmise(s_vals)
        poisson_pdf = gue.poisson_distribution(s_vals)
        
        axes[0, 0].hist(gaps_clean, bins=50, density=True, alpha=0.6, 
                       label='Empirical', edgecolor='black')
        axes[0, 0].plot(s_vals, wigner_pdf, 'r-', linewidth=2, label='GUE (Wigner)')
        axes[0, 0].plot(s_vals, poisson_pdf, 'g--', linewidth=2, label='Poisson')
        axes[0, 0].set_xlabel('Normalized spacing s')
        axes[0, 0].set_ylabel('Probability density')
        axes[0, 0].set_title('Spacing Distribution vs GUE/Poisson')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xlim([0, 4])
        
        # CDF comparison
        gaps_sorted = np.sort(gaps_clean)
        empirical_cdf = np.arange(1, len(gaps_sorted) + 1) / len(gaps_sorted)
        
        wigner_cdf = np.cumsum(wigner_pdf) * (s_vals[1] - s_vals[0])
        wigner_cdf /= wigner_cdf[-1]
        poisson_cdf = 1 - np.exp(-s_vals)
        
        axes[0, 1].plot(gaps_sorted, empirical_cdf, 'b-', alpha=0.6, label='Empirical')
        axes[0, 1].plot(s_vals, wigner_cdf, 'r-', linewidth=2, label='GUE')
        axes[0, 1].plot(s_vals, poisson_cdf, 'g--', linewidth=2, label='Poisson')
        axes[0, 1].set_xlabel('Normalized spacing s')
        axes[0, 1].set_ylabel('Cumulative probability')
        axes[0, 1].set_title('Cumulative Distribution Function')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, 4])
        
        # Level repulsion zoom (near s=0)
        axes[1, 0].hist(gaps_clean[gaps_clean < 1], bins=30, density=True, 
                       alpha=0.6, label='Empirical', edgecolor='black')
        s_zoom = np.linspace(0, 1, 100)
        wigner_zoom = gue.wigner_surmise(s_zoom)
        poisson_zoom = gue.poisson_distribution(s_zoom)
        axes[1, 0].plot(s_zoom, wigner_zoom, 'r-', linewidth=2, label='GUE (quadratic)')
        axes[1, 0].plot(s_zoom, poisson_zoom, 'g--', linewidth=2, label='Poisson (flat)')
        axes[1, 0].set_xlabel('Normalized spacing s')
        axes[1, 0].set_ylabel('Probability density')
        axes[1, 0].set_title('Level Repulsion (zoom near s=0)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot against Wigner
        from scipy.stats import probplot
        
        # Generate Wigner samples for Q-Q plot
        # Use inverse transform sampling
        n_samples = len(gaps_clean)
        uniform = np.linspace(1/(2*n_samples), 1 - 1/(2*n_samples), n_samples)
        
        # Numerical inverse CDF for Wigner (approximate)
        wigner_quantiles = np.interp(uniform, wigner_cdf, s_vals)
        
        axes[1, 1].scatter(wigner_quantiles, gaps_sorted, alpha=0.5, s=10)
        axes[1, 1].plot([0, 4], [0, 4], 'r--', linewidth=2, label='Perfect fit')
        axes[1, 1].set_xlabel('Theoretical quantiles (GUE)')
        axes[1, 1].set_ylabel('Empirical quantiles')
        axes[1, 1].set_title('Q-Q Plot vs GUE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_gap_structure(self, zeros: np.ndarray, deltas: np.ndarray,
                          filename: str = "gap_structure.png"):
        """
        Plot gap structure and relationships.
        
        Args:
            zeros: Array of zero imaginary parts
            deltas: Array of gap differences
            filename: Output filename
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Gaps vs height
        axes[0, 0].scatter(zeros[:-1], deltas, alpha=0.3, s=5)
        axes[0, 0].set_xlabel('Zero height γ_n')
        axes[0, 0].set_ylabel('Gap δ_n')
        axes[0, 0].set_title('Gap Size vs Zero Height')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log-log plot
        axes[0, 1].scatter(np.log(zeros[:-1]), np.log(deltas), alpha=0.3, s=5)
        axes[0, 1].set_xlabel('log(γ_n)')
        axes[0, 1].set_ylabel('log(δ_n)')
        axes[0, 1].set_title('Log-Log: Gap vs Height')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gap differences (second derivative)
        gap_diff = np.diff(deltas)
        axes[1, 0].plot(gap_diff, alpha=0.6, linewidth=0.5)
        axes[1, 0].set_xlabel('Index n')
        axes[1, 0].set_ylabel('δ_{n+1} - δ_n')
        axes[1, 0].set_title('Gap Differences (Acceleration)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # Gap ratios (multiplicative view)
        gap_ratios = deltas[1:] / deltas[:-1]
        gap_ratios_clean = gap_ratios[np.isfinite(gap_ratios)]
        
        axes[1, 1].hist(np.log(gap_ratios_clean), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, 
                          label='No change (ratio=1)')
        axes[1, 1].set_xlabel('log(δ_{n+1}/δ_n)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Gap Ratios (B term in Z-transform)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")
    
    def plot_phase_analysis(self, theta: np.ndarray, clustering_results: dict,
                           filename: str = "phase_analysis.png"):
        """
        Comprehensive phase analysis plots.
        
        Args:
            theta: Array of phases
            clustering_results: Results from phase clustering analysis
            filename: Output filename
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Polar scatter
        ax1 = fig.add_subplot(gs[0, 0], projection='polar')
        ax1.scatter(theta, np.ones_like(theta), alpha=0.2, s=10, c=theta, 
                   cmap='hsv')
        ax1.set_title('Phases on Unit Circle', pad=20)
        ax1.set_ylim([0, 1.2])
        
        # Histogram with chi-square test
        ax2 = fig.add_subplot(gs[0, 1:])
        n_bins = clustering_results['n_bins']
        hist = np.array(clustering_results['histogram'])
        bin_edges = np.array(clustering_results['bin_edges'])
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        expected = len(theta) / n_bins
        ax2.bar(bin_centers, hist, width=2*np.pi/n_bins, alpha=0.7, 
               edgecolor='black', label='Observed')
        ax2.axhline(expected, color='red', linestyle='--', linewidth=2,
                   label=f'Expected (uniform)')
        ax2.set_xlabel('Phase θ (radians)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Phase Histogram (χ² = {clustering_results["chi_square"]:.2f}, ' +
                     f'p = {clustering_results["p_value"]:.4f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mean resultant vector
        ax3 = fig.add_subplot(gs[1, 0], projection='polar')
        mean_angle = np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
        mean_length = clustering_results['mean_resultant_length']
        ax3.arrow(0, 0, mean_angle, mean_length, head_width=0.1, head_length=0.05,
                 fc='red', ec='red', linewidth=2)
        ax3.set_title(f'Mean Resultant\nLength = {mean_length:.4f}', pad=20)
        ax3.set_ylim([0, 1])
        
        # Circular variance visualization
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = ['Circular\nVariance', 'Normalized\nEntropy']
        values = [clustering_results['circular_variance'], 
                 clustering_results['normalized_entropy']]
        colors = ['steelblue', 'orange']
        bars = ax4.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.set_xlim([0, 1])
        ax4.set_xlabel('Value')
        ax4.set_title('Phase Clustering Metrics\n(0 = clustered, 1 = uniform)')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center')
        
        # Phase differences (successive)
        ax5 = fig.add_subplot(gs[1, 2])
        phase_diffs = np.diff(np.sort(theta))
        ax5.hist(phase_diffs, bins=30, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Phase difference (radians)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Distribution of Successive Phase Differences')
        ax5.grid(True, alpha=0.3)
        
        # Phase autocorrelation
        ax6 = fig.add_subplot(gs[2, :2])
        from statsmodels.tsa.stattools import acf
        theta_sorted = np.sort(theta)
        acf_vals = acf(theta_sorted, nlags=min(50, len(theta_sorted)//2), fft=True)
        lags = np.arange(len(acf_vals))
        ax6.stem(lags, acf_vals, linefmt='b-', markerfmt='bo', basefmt='r-')
        ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax6.axhline(1.96/np.sqrt(len(theta_sorted)), color='red', linestyle='--', 
                   label='95% CI')
        ax6.axhline(-1.96/np.sqrt(len(theta_sorted)), color='red', linestyle='--')
        ax6.set_xlabel('Lag')
        ax6.set_ylabel('ACF')
        ax6.set_title('Phase Autocorrelation (sorted phases)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Summary statistics box
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        summary_text = f"""
Phase Statistics:
─────────────────
N phases: {len(theta)}
Mean: {np.mean(theta):.4f} rad
Std: {np.std(theta):.4f} rad

Clustering Tests:
─────────────────
χ² statistic: {clustering_results['chi_square']:.2f}
p-value: {clustering_results['p_value']:.4e}
Circular var: {clustering_results['circular_variance']:.4f}
Entropy: {clustering_results['normalized_entropy']:.4f}

Interpretation:
─────────────────
"""
        if clustering_results['p_value'] < 0.05:
            summary_text += "❌ NON-UNIFORM\n(reject uniformity)"
            color = 'red'
        else:
            summary_text += "✓ UNIFORM\n(consistent with RH)"
            color = 'green'
        
        ax7.text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9,
                verticalalignment='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved: {filename}")


if __name__ == "__main__":
    # Test visualizations
    print("Testing visualization module...")
    
    from zeta_zeros import ZetaZerosDataset, compute_normalized_gaps
    from z_transform import ZTransform
    
    dataset = ZetaZerosDataset()
    zeros = dataset.get_zeros(n_zeros=500, method='mpmath', precision=50)
    
    print(f"Loaded {len(zeros)} zeros")
    
    # Create visualizer
    viz = ZDomainVisualizer(output_dir="results/test_viz")
    
    # Test Z-transform plot
    ztrans = ZTransform(zeros)
    Z_n = ztrans.compute_z_values()
    viz.plot_z_transform(Z_n)
    
    # Test phase plot
    theta = ztrans.compute_phases()
    viz.plot_phase_circle(theta)
    
    # Test GUE comparison
    normalized_gaps = compute_normalized_gaps(zeros)
    viz.plot_gue_comparison(normalized_gaps)
    
    # Test gap structure
    deltas = np.diff(zeros)
    viz.plot_gap_structure(zeros, deltas)
    
    # Test phase analysis
    clustering = ztrans.analyze_phase_clustering(n_bins=36)
    viz.plot_phase_analysis(theta, clustering)
    
    print("\nAll test visualizations completed!")
