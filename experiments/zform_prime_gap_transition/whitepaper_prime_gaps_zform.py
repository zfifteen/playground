#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Paper (Executable): Z-Form Analysis of Real Prime Gap Distributions

Title:
  Z-Form Normalization of Prime Gaps: Testing Phase Structure in Actual Data

Synopsis
--------
This script tests the Z-Form hypothesis using REAL prime gap data:
  
  Z = A(B/C) where:
    A = gₙ (individual gap value)
    B = Δg/Δn (gap velocity - rate of change)
    C = 2log²pₙ (Cramér bound - theoretical maximum gap)

The hypothesis: Z-normalization reveals phase structure in actual prime gaps,
distinguishing between lognormal-dominated and exponential-dominated regimes
based on local gap dynamics.

The script:
  * Generates REAL primes using segmented sieve
  * Computes actual gaps gₙ = pₙ₊₁ - pₙ
  * Calculates gap velocity B = Δg/Δn
  * Applies Cramér bound C = 2log²pₙ
  * Constructs Z(n) = gₙ · (Δg/Δn) / (2log²pₙ)
  * Tests for phase structure in Z across scales

References
----------
[1] Cohen, "Gaps Between Consecutive Primes and the Exponential Distribution" (2024)
[2] Cramér conjecture on prime gaps
[3] PR-0003: Real prime gap analysis showing log-normal distribution (ACF=0.796)

Run
---
$ python whitepaper_prime_gaps_zform.py

"""

import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add PR-0003 prime generator to path
sys.path.insert(0, str(Path(__file__).parent.parent / "PR-0003_prime_log_gap_optimized" / "src"))
from prime_generator import generate_primes_to_limit


# =============================================================================
# 1. Real Prime Gap Generation
# =============================================================================

def generate_real_prime_gaps(limit: int, cache_dir: str = "data") -> tuple:
    """
    IMPLEMENTED: Generate real primes and compute gaps using segmented sieve.
    
    Args:
        limit: Maximum prime value (e.g., 10^6, 10^7)
        cache_dir: Directory for caching primes
    
    Returns:
        (primes, gaps) where:
            primes: np.array of prime numbers
            gaps: np.array of gaps[i] = primes[i+1] - primes[i]
    """
    print(f"Generating real primes up to {limit:,}...")
    
    # Create cache directory if it doesn't exist
    cache_path = Path(cache_dir)
    cache_path.mkdir(exist_ok=True)
    
    # Generate primes using PR-0003's segmented sieve
    primes = generate_primes_to_limit(limit, cache_dir=cache_dir)
    print(f"  Generated {len(primes):,} primes")
    
    # Compute gaps as consecutive differences
    gaps = np.diff(primes)
    print(f"  Computed {len(gaps):,} gaps")
    
    return primes, gaps


def compute_cramer_bound(primes: np.ndarray) -> np.ndarray:
    """
    IMPLEMENTED: Compute Cramér bound C = 2(log p)² for each prime.
    
    Args:
        primes: Array of prime numbers
    
    Returns:
        Array of Cramér bounds, one per prime
    """
    log_primes = np.log(primes)
    cramer = 2.0 * (log_primes ** 2)
    return cramer


# =============================================================================
# 2. Gap Velocity Calculation
# =============================================================================

def compute_gap_velocity(gaps: np.ndarray, window: int = 10) -> np.ndarray:
    """
    IMPLEMENTED: Compute gap velocity B = Δg/Δn using windowed differences.
    
    Args:
        gaps: Array of gap values
        window: Window size for computing velocity
    
    Returns:
        Array of gap velocities (rate of change)
    """
    # Convert to float to avoid integer overflow
    gaps = gaps.astype(np.float64)
    n = len(gaps)
    velocity = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i < window:
            # Forward difference for start
            if i + window < n:
                velocity[i] = (gaps[i + window] - gaps[i]) / window
            else:
                velocity[i] = 0.0
        elif i >= n - window:
            # Backward difference for end
            if i - window >= 0:
                velocity[i] = (gaps[i] - gaps[i - window]) / window
            else:
                velocity[i] = 0.0
        else:
            # Central difference for middle
            velocity[i] = (gaps[i + window] - gaps[i - window]) / (2 * window)
    
    return velocity


# =============================================================================
# 3. Z-Form Construction
# =============================================================================

def compute_z_form(gaps: np.ndarray, primes: np.ndarray, velocity: np.ndarray, 
                   cramer: np.ndarray) -> np.ndarray:
    """
    IMPLEMENTED: Compute Z = A(B/C) where A = gap, B = velocity, C = Cramér bound.
    
    Args:
        gaps: Gap values (A component)
        primes: Prime numbers (for alignment)
        velocity: Gap velocities (B component)
        cramer: Cramér bounds (C component)
    
    Returns:
        Z-form normalized values
    """
    # Align arrays - gaps is one element shorter than primes
    # We use cramer[:-1] to match gaps length
    cramer_aligned = cramer[:-1]
    
    # Compute Z = gaps * (velocity / cramer)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        Z = gaps * (velocity / cramer_aligned)
        # Replace inf and nan with 0
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    
    return Z


# =============================================================================
# 4. Phase Structure Analysis
# =============================================================================

def analyze_phase_structure(Z: np.ndarray, primes: np.ndarray, 
                            num_bands: int = 10) -> dict:
    """
    IMPLEMENTED: Analyze Z values across log-prime bands to detect phase structure.
    
    Args:
        Z: Z-form values
        primes: Corresponding primes (for binning)
        num_bands: Number of logarithmic bands
    
    Returns:
        Dictionary with phase analysis results
    """
    # Use primes[:-1] to match Z length (gaps are one shorter)
    primes_aligned = primes[:-1]
    log_primes = np.log(primes_aligned)
    
    # Create logarithmic bins
    log_min, log_max = log_primes.min(), log_primes.max()
    band_edges = np.linspace(log_min, log_max, num_bands + 1)
    
    # Analyze each band
    band_centers = []
    Z_means = []
    Z_stds = []
    Z_medians = []
    classifications = []
    
    # Classification thresholds
    THRESHOLD_HIGH = 0.01  # Lognormal-dominated
    THRESHOLD_LOW = -0.01  # Exponential-dominated
    
    for i in range(num_bands):
        mask = (log_primes >= band_edges[i]) & (log_primes < band_edges[i + 1])
        if i == num_bands - 1:  # Include right edge in last band
            mask = (log_primes >= band_edges[i]) & (log_primes <= band_edges[i + 1])
        
        Z_band = Z[mask]
        
        if len(Z_band) > 0:
            band_centers.append((band_edges[i] + band_edges[i + 1]) / 2)
            Z_means.append(np.mean(Z_band))
            Z_stds.append(np.std(Z_band))
            Z_medians.append(np.median(Z_band))
            
            # Classify based on mean Z
            mean_z = Z_means[-1]
            if mean_z > THRESHOLD_HIGH:
                classifications.append("lognormal-dominated")
            elif mean_z < THRESHOLD_LOW:
                classifications.append("exponential-dominated")
            else:
                classifications.append("transition")
    
    return {
        "band_edges": band_edges,
        "band_centers": np.array(band_centers),
        "Z_means": np.array(Z_means),
        "Z_stds": np.array(Z_stds),
        "Z_medians": np.array(Z_medians),
        "classifications": classifications,
        "thresholds": (THRESHOLD_LOW, THRESHOLD_HIGH),
    }


def test_z_distribution(Z: np.ndarray, primes: np.ndarray) -> dict:
    """
    IMPLEMENTED: Test if Z shows different distributions across scales.
    
    Args:
        Z: Z-form values
        primes: Corresponding primes
    
    Returns:
        Dictionary with distribution test results
    """
    # Align primes with gaps/Z
    primes_aligned = primes[:-1]
    
    # Split into three regions based on prime percentiles
    p33 = np.percentile(primes_aligned, 33)
    p67 = np.percentile(primes_aligned, 67)
    
    low_mask = primes_aligned < p33
    mid_mask = (primes_aligned >= p33) & (primes_aligned < p67)
    high_mask = primes_aligned >= p67
    
    regions = {
        "low": (primes_aligned[low_mask], Z[low_mask]),
        "mid": (primes_aligned[mid_mask], Z[mid_mask]),
        "high": (primes_aligned[high_mask], Z[high_mask]),
    }
    
    results = {
        "region_bounds": {"low": (primes_aligned.min(), p33),
                         "mid": (p33, p67),
                         "high": (p67, primes_aligned.max())},
        "z_stats": {},
    }
    
    for region_name, (primes_r, Z_r) in regions.items():
        results["z_stats"][region_name] = {
            "mean": np.mean(Z_r),
            "std": np.std(Z_r),
            "median": np.median(Z_r),
            "percentile_25": np.percentile(Z_r, 25),
            "percentile_75": np.percentile(Z_r, 75),
        }
    
    return results


# =============================================================================
# 5. Visualization
# =============================================================================

def plot_z_across_scales(Z: np.ndarray, primes: np.ndarray, filename: str = "z_vs_primes.png"):
    """
    IMPLEMENTED: Plot Z values vs log(prime) to visualize phase structure.
    
    Args:
        Z: Z-form values
        primes: Corresponding primes
        filename: Output filename
    """
    primes_aligned = primes[:-1]
    log_primes = np.log10(primes_aligned)
    
    plt.figure(figsize=(12, 6))
    
    # Scatter plot with transparency
    plt.scatter(log_primes, Z, alpha=0.1, s=1, c='blue', label='Z values')
    
    # Add phase threshold lines
    plt.axhline(y=0.01, color='green', linestyle='--', linewidth=2, label='Lognormal threshold (+0.01)')
    plt.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Neutral (Z=0)')
    plt.axhline(y=-0.01, color='red', linestyle='--', linewidth=2, label='Exponential threshold (-0.01)')
    
    # Add smoothed trend
    from scipy.ndimage import gaussian_filter1d
    # Bin data for smoothing
    bins = 50
    bin_edges = np.linspace(log_primes.min(), log_primes.max(), bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    for i in range(bins):
        mask = (log_primes >= bin_edges[i]) & (log_primes < bin_edges[i + 1])
        if np.sum(mask) > 0:
            bin_means.append(np.mean(Z[mask]))
        else:
            bin_means.append(np.nan)
    bin_means = np.array(bin_means)
    valid = ~np.isnan(bin_means)
    if np.sum(valid) > 5:
        smoothed = gaussian_filter1d(bin_means[valid], sigma=2)
        plt.plot(bin_centers[valid], smoothed, color='orange', linewidth=3, label='Smoothed trend')
    
    plt.xlabel('log₁₀(prime)', fontsize=12)
    plt.ylabel('Z = (gap)(Δg/Δn)/(2log²p)', fontsize=12)
    plt.title('Z-Form Phase Structure Across Prime Scales', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


def plot_phase_bands(analysis: dict, filename: str = "phase_bands.png"):
    """
    IMPLEMENTED: Plot phase structure analysis results.
    
    Args:
        analysis: Results from analyze_phase_structure
        filename: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    centers = analysis["band_centers"]
    means = analysis["Z_means"]
    stds = analysis["Z_stds"]
    classifications = analysis["classifications"]
    
    # Color map for classifications
    color_map = {
        "lognormal-dominated": "green",
        "transition": "orange",
        "exponential-dominated": "red",
    }
    colors = [color_map[c] for c in classifications]
    
    # Plot 1: Mean Z per band with error bars
    ax1.bar(range(len(means)), means, color=colors, alpha=0.7, edgecolor='black')
    ax1.errorbar(range(len(means)), means, yerr=stds, fmt='none', ecolor='black', capsize=5)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.axhline(y=0.01, color='green', linestyle='--', alpha=0.5)
    ax1.axhline(y=-0.01, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Band Index', fontsize=12)
    ax1.set_ylabel('Mean Z', fontsize=12)
    ax1.set_title('Phase Classification by Log-Prime Band', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[k], label=k) for k in color_map.keys()]
    ax1.legend(handles=legend_elements, fontsize=10)
    
    # Plot 2: Band centers in log-prime space
    log10_centers = centers / np.log(10)  # Convert to log10
    ax2.plot(log10_centers, means, 'o-', color='blue', linewidth=2, markersize=8)
    for i, (x, y, c) in enumerate(zip(log10_centers, means, colors)):
        ax2.plot(x, y, 'o', color=c, markersize=10)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(y=0.01, color='green', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.01, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('log₁₀(prime) - Band Center', fontsize=12)
    ax2.set_ylabel('Mean Z', fontsize=12)
    ax2.set_title('Z Transition Across Prime Scales', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  Saved {filename}")


# =============================================================================
# 6. Main Routine
# =============================================================================

def main():
    """
    Main execution: generate real primes, compute Z-form, analyze phases.
    """
    # PURPOSE: Orchestrate complete Z-Form analysis
    # INPUTS: None (uses hardcoded scales)
    # PROCESS:
    #   1. Generate real primes at multiple scales (10^6, 10^7)
    #   2. For each scale:
    #      a. Compute gaps
    #      b. Compute Cramér bounds
    #      c. Compute gap velocity
    #      d. Compute Z-form
    #      e. Analyze phase structure
    #      f. Generate plots
    #   3. Print summary results
    #   4. Return analysis results
    # OUTPUTS: Prints to console, saves plots
    # DEPENDENCIES: All above functions
    print("=" * 80)
    print("Z-FORM ANALYSIS OF REAL PRIME GAP DISTRIBUTIONS")
    print("=" * 80)
    print()
    
    # Test at scale 10^6 for reasonable execution time
    limit = 10**6
    print(f"Scale: 10^6 (limit = {limit:,})")
    print()
    
    # Step 1: Generate real primes and gaps
    print("Step 1: Generating real primes...")
    primes, gaps = generate_real_prime_gaps(limit)
    print(f"  ✓ Primes: {len(primes):,}")
    print(f"  ✓ Gaps: {len(gaps):,}")
    print()
    
    # Step 2: Compute Cramér bounds
    print("Step 2: Computing Cramér bounds C = 2(log p)²...")
    cramer = compute_cramer_bound(primes)
    print(f"  ✓ Bounds computed for {len(cramer):,} primes")
    print(f"  ✓ C range: [{cramer.min():.2f}, {cramer.max():.2f}]")
    print()
    
    # Step 3: Compute gap velocity
    print("Step 3: Computing gap velocity B = Δg/Δn...")
    velocity = compute_gap_velocity(gaps, window=10)
    print(f"  ✓ Velocity computed for {len(velocity):,} gaps")
    print(f"  ✓ B range: [{velocity.min():.6f}, {velocity.max():.6f}]")
    print()
    
    # Step 4: Compute Z-form
    print("Step 4: Computing Z = A(B/C) where A=gap, B=velocity, C=Cramér...")
    Z = compute_z_form(gaps, primes, velocity, cramer)
    print(f"  ✓ Z computed for {len(Z):,} gaps")
    print(f"  ✓ Z range: [{Z.min():.6f}, {Z.max():.6f}]")
    print(f"  ✓ Z mean: {Z.mean():.6f}")
    print(f"  ✓ Z std: {Z.std():.6f}")
    print()
    
    # Step 5: Analyze phase structure
    print("Step 5: Analyzing phase structure across log-prime bands...")
    analysis = analyze_phase_structure(Z, primes, num_bands=10)
    print(f"  ✓ Analyzed {len(analysis['band_centers'])} bands")
    print()
    print("  Phase Classification:")
    regime_counts = {}
    for c in analysis['classifications']:
        regime_counts[c] = regime_counts.get(c, 0) + 1
    for regime, count in sorted(regime_counts.items()):
        pct = 100.0 * count / len(analysis['classifications'])
        print(f"    {regime}: {count} bands ({pct:.1f}%)")
    print()
    
    # Step 6: Test Z distribution across scales
    print("Step 6: Testing Z distribution across scale regions...")
    dist_test = test_z_distribution(Z, primes)
    print("  Z Statistics by Region:")
    for region in ["low", "mid", "high"]:
        stats = dist_test["z_stats"][region]
        bounds = dist_test["region_bounds"][region]
        print(f"    {region:5s} [{bounds[0]:.0f}, {bounds[1]:.0f}]: " +
              f"mean={stats['mean']:.6f}, std={stats['std']:.6f}")
    print()
    
    # Step 7: Generate visualizations
    print("Step 7: Generating visualizations...")
    plot_z_across_scales(Z, primes, "z_vs_primes.png")
    plot_phase_bands(analysis, "phase_bands.png")
    print()
    
    # Summary
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"Phase structure detected: {len(set(analysis['classifications']))} distinct regimes")
    print(f"Z range: [{Z.min():.6f}, {Z.max():.6f}]")
    print()
    print("Outputs:")
    print("  - z_vs_primes.png (Z scatter plot with phase thresholds)")
    print("  - phase_bands.png (band-wise phase classification)")
    print()
    
    return analysis, dist_test


if __name__ == "__main__":
    main()
