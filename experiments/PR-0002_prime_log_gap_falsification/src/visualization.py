import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


def plot_log_gap_histogram(log_gaps: np.ndarray, save_path: str = None):
    """
    Plot histogram of log-gaps.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(log_gaps, bins=50, alpha=0.7, edgecolor="black")
    plt.xlabel("Log Gap")
    plt.ylabel("Frequency")
    plt.title("Histogram of Prime Log-Gaps")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_qq_lognormal(log_gaps: np.ndarray, save_path: str = None):
    """
    Q-Q plot against log-normal distribution.
    """
    plt.figure(figsize=(8, 8))
    # Fit log-normal
    shape, loc, scale = stats.lognorm.fit(log_gaps, floc=0)
    stats.probplot(log_gaps, dist="lognorm", sparams=(shape, loc, scale), plot=plt)
    plt.title("Q-Q Plot: Log-Gaps vs Log-Normal")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_decay_trend(
    quintile_means: np.ndarray, decile_means: np.ndarray, save_path: str = None
):
    """
    Plot decay trend for quintiles and deciles.
    """
    plt.figure(figsize=(10, 6))
    x_quint = np.arange(len(quintile_means))
    x_decile = np.arange(len(decile_means))
    plt.plot(x_quint, quintile_means, "o-", label="Quintiles", markersize=8)
    plt.plot(x_decile, decile_means, "s-", label="Deciles", markersize=6)
    plt.xlabel("Bin Index")
    plt.ylabel("Mean Log-Gap")
    plt.title("Log-Gap Mean Decay Trend")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_acf_pacf(
    acf_values: np.ndarray, pacf_values: np.ndarray, save_path: str = None
):
    """
    Plot ACF and PACF.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    lags = np.arange(len(acf_values))
    ax1.bar(lags, acf_values, width=0.3)
    ax1.axhline(y=0, color="black", linestyle="--")
    ax1.set_title("Autocorrelation Function (ACF)")
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("ACF")
    ax1.grid(True, alpha=0.3)

    ax2.bar(lags, pacf_values, width=0.3)
    ax2.axhline(y=0, color="black", linestyle="--")
    ax2.set_title("Partial Autocorrelation Function (PACF)")
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("PACF")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
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
    Generate all plots and save to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_log_gap_histogram(log_gaps, os.path.join(output_dir, "log_gap_histogram.png"))
    plot_qq_lognormal(log_gaps, os.path.join(output_dir, "qq_plot_lognormal.png"))
    plot_decay_trend(
        quintile_means, decile_means, os.path.join(output_dir, "decay_trend.png")
    )
    plot_acf_pacf(acf_values, pacf_values, os.path.join(output_dir, "acf_pacf.png"))


if __name__ == "__main__":
    # Test plots
    np.random.seed(42)
    log_gaps = np.random.lognormal(0, 1, 1000)
    quintile_means = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    decile_means = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    acf_values = np.random.randn(21) * 0.1
    pacf_values = np.random.randn(21) * 0.1
    generate_all_plots(log_gaps, quintile_means, decile_means, acf_values, pacf_values)
