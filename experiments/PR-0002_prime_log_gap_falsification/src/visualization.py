"""
Visualization Module

This module creates plots to make the log-gap data "visible" and testable.
Humans think in pictures – histograms show shapes, Q-Q plots check fits, trends reveal decay.
These visuals help "see" if log-gaps behave like damped circuit signals.
All plots can be saved as PNGs for the experiment's figures folder.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def plot_log_gap_histogram(log_gaps: np.ndarray, save_path: str = None):
    """
    Create a histogram to visualize the shape of log-gap distribution.

    A histogram is like a bar chart of frequencies: how often do gaps fall in each range?
    This shows if gaps are symmetric (normal) or skewed (log-normal, with a long tail).
    50 bins give detail without noise; alpha/transparency makes it readable.
    Grid helps estimate values; saves to file if path given, else displays.
    """
    plt.figure(figsize=(10, 6))  # Wide figure for clarity
    plt.hist(
        log_gaps, bins=50, alpha=0.7, edgecolor="black"
    )  # 50 bins, semi-transparent
    plt.xlabel("Log Gap")  # X: gap value
    plt.ylabel("Frequency")  # Y: count in each bin
    plt.title("Histogram of Prime Log-Gaps")  # Descriptive title
    plt.grid(True, alpha=0.3)  # Light grid
    if save_path:
        plt.savefig(save_path)  # Save as PNG
        plt.close()  # Don't show if saving
    else:
        plt.show()  # Interactive display


def plot_qq_lognormal(log_gaps: np.ndarray, save_path: str = None):
    """
    Q-Q (Quantile-Quantile) plot to check if log-gaps match log-normal perfectly.

    Q-Q compares data quantiles (percentiles) to a theoretical distribution's quantiles.
    If points lie on the diagonal line, the fit is perfect; deviations show mismatches.
    We fit log-normal parameters first, then plot. This visually confirms (or rejects) our hypothesis.
    Square figure for symmetry; title explains the comparison.
    """
    plt.figure(figsize=(8, 8))  # Square for balance
    # Fit log-normal to data (fix loc at 0 since gaps >0)
    shape, loc, scale = stats.lognorm.fit(log_gaps, floc=0)
    # Generate Q-Q plot: data vs. fitted log-normal
    stats.probplot(log_gaps, dist="lognorm", sparams=(shape, loc, scale), plot=plt)
    plt.title("Q-Q Plot: Log-Gaps vs Log-Normal")  # Clear title
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_decay_trend(
    quintile_means: np.ndarray, decile_means: np.ndarray, save_path: str = None
):
    """
    Line plot showing how average log-gaps change across prime groups.

    Quintiles (5 groups) and deciles (10 groups) divide the prime sequence.
    X-axis: group index (0 = smallest primes, higher = larger primes).
    Y-axis: average gap in that group.
    A downward slope indicates "decay" – gaps shrinking relatively as primes grow.
    Circles for quintiles, squares for deciles; legend distinguishes them.
    This plot visually tests the damping hypothesis.
    """
    plt.figure(figsize=(10, 6))  # Wide for trend visibility
    x_quint = np.arange(len(quintile_means))  # Indices 0-4
    x_decile = np.arange(len(decile_means))  # Indices 0-9
    plt.plot(
        x_quint, quintile_means, "o-", label="Quintiles", markersize=8
    )  # Circles with line
    plt.plot(
        x_decile, decile_means, "s-", label="Deciles", markersize=6
    )  # Squares with line
    plt.xlabel("Bin Index")  # Group position
    plt.ylabel("Mean Log-Gap")  # Average value
    plt.title("Log-Gap Mean Decay Trend")  # Emphasizes decay
    plt.legend()  # Show labels
    plt.grid(True, alpha=0.3)  # Light grid
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_acf_pacf(
    acf_values: np.ndarray, pacf_values: np.ndarray, save_path: str = None
):
    """
    Two-panel plot of autocorrelation functions to reveal gap "memory."

    Top: ACF bars show correlation strength at each lag (delay in gaps).
    Bars above zero = positive memory; below = negative.
    Dashed line at zero for reference.

    Bottom: PACF isolates direct correlations, helping identify patterns like AR(1).
    Together, they check if gaps behave like a circuit's filtered signal.
    Tall figure for two subplots; bars are narrow for clarity.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # Two rows, one column
    lags = np.arange(len(acf_values))  # Lag indices: 0, 1, 2, ...
    # Top subplot: ACF
    ax1.bar(lags, acf_values, width=0.3)  # Bar chart of correlations
    ax1.axhline(y=0, color="black", linestyle="--")  # Zero reference line
    ax1.set_title("Autocorrelation Function (ACF)")  # Title for top
    ax1.set_xlabel("Lag")  # X: time delay
    ax1.set_ylabel("ACF")  # Y: correlation value
    ax1.grid(True, alpha=0.3)  # Light grid

    # Bottom subplot: PACF
    ax2.bar(lags, pacf_values, width=0.3)  # Similar bar chart
    ax2.axhline(y=0, color="black", linestyle="--")  # Zero line
    ax2.set_title("Partial Autocorrelation Function (PACF)")  # Title for bottom
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("PACF")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()  # Adjust spacing
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def generate_all_plots(
    log_gaps: np.ndarray,
    quintile_means: np.ndarray,
    decile_means: np.ndarray,
    acf_values: np.ndarray,
    pacf_values: np.ndarray,
    output_dir: str = "results/figures",
):
    """
    Batch-generate all four key plots and save them as PNGs.

    This is the main entry point for visualization in the experiment.
    It creates the full figure set: histogram, Q-Q, decay, ACF/PACF.
    Output directory is created if needed; files are named clearly for reports.
    Used by run_analysis.py to produce the results/figures/ folder.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure dir exists
    # Generate each plot with its filename
    plot_log_gap_histogram(log_gaps, os.path.join(output_dir, "log_gap_histogram.png"))
    plot_qq_lognormal(log_gaps, os.path.join(output_dir, "qq_plot_lognormal.png"))
    plot_decay_trend(
        quintile_means, decile_means, os.path.join(output_dir, "decay_trend.png")
    )
    plot_acf_pacf(acf_values, pacf_values, os.path.join(output_dir, "acf_pacf.png"))


if __name__ == "__main__":
    # Test the plotting functions with dummy data
    np.random.seed(42)  # Reproducible
    log_gaps = np.random.lognormal(0, 1, 1000)  # Fake log-normal gaps
    quintile_means = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Decreasing averages
    decile_means = np.array(
        [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )  # More steps
    acf_values = np.random.randn(21) * 0.1  # Random correlations
    pacf_values = np.random.randn(21) * 0.1
    generate_all_plots(
        log_gaps, quintile_means, decile_means, acf_values, pacf_values
    )  # Save to default dir
