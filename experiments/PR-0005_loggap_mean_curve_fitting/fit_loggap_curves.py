#!/usr/bin/env python3
"""
Experiment: Fit simple deterministic `loggap_mean_curve_*` models to
binned mean log-gaps vs log-prime.

Context and existing code:
- Repo: https://github.com/zfifteen/playground/tree/main/experiments/PR-0004_lognormal_factorization
- Binning logic is in the binning module (e.g., src/binning.py), which
  computes:
    - `centers`: bin centers on the log-prime axis
    - `bin_means`: mean log-gap per bin (with possible NaNs)

- Example JSON outputs with `centers` and `bin_means`:
    - data/bin_stats_1e5.json
    - data/bin_stats_1e6.json
    - data/bin_stats_1e7.json
    - data/bin_stats_1e8.json
    - data/bin_stats_1e9.json

Goal:
- Load existing binning JSON files (do NOT regenerate primes or recompute bins).
- Define several `loggap_mean_curve_*` candidate functions mapping logp -> mean log-gap.
- Fit each candidate to (centers, bin_means) using curve_fit.
- Compare fits by MSE and R^2 and print results.

Constraints:
- Reuse existing JSON structure: access `binning.centers` and `binning.bin_means`.
- Ignore bins where `bin_means` is NaN.
- Keep code small and explicit; no magical data generation.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit


# Initial parameter guess for slope in curve fitting
# Based on observed decay trend in mean log-gaps
INITIAL_SLOPE = -0.002


@dataclass
class TrendFitResult:
    """IMPLEMENTED: Result container for trend fitting."""
    name: str
    params: Tuple[float, ...]
    mse: float
    r2: float


def load_bin_data(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    IMPLEMENTED: Load bin centers (log_primes) and bin mean log-gaps from a bin_stats JSON file.

    Expected structure (from existing binning code):
    {
      "binning": {
        "centers": [...],      # May not exist in older files
        "bin_means": [...],
        "n_bins": ...          # May also be in metadata
      },
      "metadata": {
        "max_prime": "...",
        "n_bins": "..."
      }
    }
    
    If centers are not present, compute them from max_prime and n_bins.
    """
    with path.open() as f:
        data = json.load(f)
    
    binning = data["binning"]
    
    # Try to get centers directly, or compute them
    if "centers" in binning:
        centers = np.array(binning["centers"], dtype=float)
    else:
        # Compute centers from metadata
        max_prime = float(data["metadata"]["max_prime"])
        # n_bins may be in binning or metadata
        n_bins = int(binning.get("n_bins", data["metadata"]["n_bins"]))
        
        # Estimate range: from ln(2) to ln(max_prime)
        min_log = np.log(2)  # First prime is 2
        max_log = np.log(max_prime)
        
        # Create bin edges and centers
        edges = np.linspace(min_log, max_log, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
    
    means = np.array(binning["bin_means"], dtype=float)
    
    # Filter out NaN values
    mask = ~np.isnan(means)
    return centers[mask], means[mask]


# --- Candidate loggap_mean_curve_* models --- #

def loggap_mean_curve_linear(logp: np.ndarray, a: float, b: float) -> np.ndarray:
    """IMPLEMENTED: Linear trend in logp: a + b * logp."""
    return a + b * logp


def loggap_mean_curve_normed(logp: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    IMPLEMENTED: Linear trend in normalized logp in [0,1]:
    t = (logp - min_log) / (max_log - min_log)
    """
    logp_min = np.min(logp)
    logp_max = np.max(logp)
    t = (logp - logp_min) / (logp_max - logp_min)
    return a + b * t


def loggap_mean_curve_loglog(logp: np.ndarray, a: float, b: float) -> np.ndarray:
    """IMPLEMENTED: Linear trend in log(logp): a + b * log(logp)."""
    return a + b * np.log(logp)


def loggap_mean_curve_power_normed(logp: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """
    IMPLEMENTED: Power-law trend in normalized logp:
    t in [0,1], curve = a + b * t**c.
    """
    logp_min = np.min(logp)
    logp_max = np.max(logp)
    t = (logp - logp_min) / (logp_max - logp_min)
    return a + b * t**c


def fit_trend(
    logp: np.ndarray,
    means: np.ndarray,
    func: Callable,
    name: str,
    p0: Tuple[float, ...],
    bounds: Tuple = (-np.inf, np.inf),
) -> TrendFitResult:
    """IMPLEMENTED: Fit a loggap_mean_curve_* model and compute MSE and R^2."""
    popt, _ = curve_fit(func, logp, means, p0=p0, bounds=bounds, maxfev=10000)
    preds = func(logp, *popt)
    residuals = means - preds
    mse = float(np.mean(residuals**2))
    ss_tot = float(np.sum((means - means.mean())**2))
    r2 = 1.0 - float(np.sum(residuals**2) / ss_tot)
    return TrendFitResult(name=name, params=tuple(popt), mse=mse, r2=r2)


def run_for_scale(json_path: Path) -> List[TrendFitResult]:
    """IMPLEMENTED: Run all trend fitting models for a single scale."""
    logp, means = load_bin_data(json_path)

    configs = [
        ("loggap_mean_curve_linear", loggap_mean_curve_linear, (means.mean(), INITIAL_SLOPE), (-np.inf, np.inf)),
        ("loggap_mean_curve_normed", loggap_mean_curve_normed, (means.mean(), INITIAL_SLOPE), (-np.inf, np.inf)),
        ("loggap_mean_curve_loglog", loggap_mean_curve_loglog, (means.mean(), INITIAL_SLOPE), (-np.inf, np.inf)),
        ("loggap_mean_curve_power_normed", loggap_mean_curve_power_normed, (means.mean(), INITIAL_SLOPE, 1.0), ([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 10.0])),
    ]

    results: List[TrendFitResult] = []
    for name, func, p0, bounds in configs:
        res = fit_trend(logp, means, func, name, p0, bounds)
        results.append(res)
    results.sort(key=lambda r: r.mse)
    return results


def main():
    """IMPLEMENTED: Main entry point - process all JSON files and display results."""
    # Adjust paths to match real files in your repo
    base = Path(__file__).resolve().parent / "data"
    json_files = [
        base / "bin_stats_1e5.json",
        base / "bin_stats_1e6.json",
        base / "bin_stats_1e7.json",
        base / "bin_stats_1e8.json",
        base / "bin_stats_1e9.json",
    ]

    for path in json_files:
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        print(f"\n=== {path.name} ===")
        results = run_for_scale(path)
        for r in results:
            print(f"{r.name:28s} mse={r.mse:.3e}  r2={r.r2:.4f}  params={r.params}")


if __name__ == "__main__":
    main()
