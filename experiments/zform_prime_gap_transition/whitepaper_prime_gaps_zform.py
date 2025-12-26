#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Paper (Executable): Z-Form Hybrid Modeling of Prime Gap Distributions

Title:
  Smooth Transition from Lognormal to Exponential in Prime Gap Distributions:
  A Unified Z-Form Perspective with Adaptive Sieve Demonstration

Synopsis
--------
This script is an executable "white paper" that demonstrates how to:
  1. Model a smooth, scale-dependent transition from lognormal to exponential
     distributions for prime gaps.
  2. Embed this transition in the Unified Z-Form Z = A(B/C), where:
        A = gap length g,
        B = local scale derivative of a distributional diagnostic w.r.t. log n,
        C = an invariant upper bound on |B|.
  3. Use Z as a phase-like coordinate that drives an adaptive "sieve policy"
     which changes behavior smoothly with scale instead of invoking a sharp
     "crossover."

The code does NOT use actual primes (for speed and reproducibility) but
constructs a synthetic model which mirrors the empirically observed behavior:
  - At smaller scales, gaps are better fit by a lognormal distribution.
  - At large scales, gaps approach an exponential with mean log n, consistent
    with exponential-moment theory for prime gaps.[web:22][web:21]
  - There is a smooth, monotone transition in a per-gap log-likelihood
    advantage ε(n) of lognormal vs exponential across bands in log n,
    reminiscent of the fixed log-band protocol with random starts described
    in numerical experiments.[web:43]

The script:
  * Generates synthetic gaps across many scales n.
  * Performs band-wise fits of lognormal and exponential models.
  * Computes the per-gap log-likelihood advantage ε(n).
  * Constructs a Z-Form normalization Z(n) = A * B/C with B = dε/d(log n).
  * Demonstrates a toy "adaptive sieve" that maps Z to policy parameters.

References (conceptual, not imported here)
-----------------------------------------
[1] Cohen, "Gaps Between Consecutive Primes and the Exponential Distribution"
    (2024).[web:21][web:22][web:23][web:45]
[2] Prime gap statistics and maximal gaps via Cramér–Shanks-type heuristics.[web:21][web:19]
[3] zfifteen unified-framework wiki, PREDICTIONS_01:
    log-band protocol and lognormal vs exponential windowed fits.[web:43]

Run
---
$ python whitepaper_prime_gaps_zform.py

"""

import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# =============================================================================
# 1. Utility: scales, bands, and synthetic "prime-gap-like" generation
# =============================================================================

def generate_scales(num_scales=25, n_min=1e6, n_max=1e14):
    """
    Generate logarithmically spaced 'scales' n between n_min and n_max.

    These represent positions along the number line where prime gaps are being
    sampled. In actual prime data, n would index prime order or magnitude.

    Here we only need log n to define:
      - the mean gap ~ log n (as in classical heuristics),[web:21][web:22]
      - the mixing weight between lognormal and exponential behavior.
    """
    logs = np.linspace(math.log(n_min), math.log(n_max), num_scales)
    return np.exp(logs)


def smooth_lognormal_weight(log_n, log_n_lo, log_n_hi):
    """
    Smoothly map log n to a mixing weight w(n) in [0, 1]:

      w(n) ~ 1  for n << 10^10 (lognormal-dominated),
      w(n) ~ 0  for n >> 10^11 (exponential-dominated),
      w(n) ~ 0.5 near transition region.[web:43]

    We implement this with a logistic curve in log n:

      w(log n) = 1 / (1 + exp(k * (log n - m))),

    where m is mid log-scale and k controls slope (steepness).
    """
    m = 0.5 * (log_n_lo + log_n_hi)
    k = 2.5 / (log_n_hi - log_n_lo)  # slope tuned empirically
    return 1.0 / (1.0 + np.exp(k * (log_n - m)))


def sample_gaps_at_scale(n, num_gaps=50000, rng=None):
    """
    Sample synthetic "prime gaps" at scale n.

    At each scale we:
      - Set mean gap mu_g = log n (Cramér-like mean).[web:21][web:22]
      - Define a lognormal component Ln(mu, sigma^2) with mean mu_g.
      - Define an exponential component Exp(1/mu_g).
      - Mix them with weight w(n) that decreases smoothly with log n.

    The key behavior is that the mixture transitions from mostly lognormal
    (small scales) to mostly exponential (large scales), with the mean gap
    remaining comparable to log n across scales.
    """
    if rng is None:
        rng = np.random.default_rng()

    log_n = math.log(n)
    # Chosen "transition window" roughly between 1e10 and 1e11
    log_n_lo = math.log(1e8)
    log_n_hi = math.log(1e12)
    w = smooth_lognormal_weight(log_n, log_n_lo, log_n_hi)

    # Mean gap ~ log n
    mu_g = max(log_n, 1.0)  # avoid degenerate small logs

    # Construct lognormal parameters so E[lognormal] ~ mu_g.
    # For lognormal LN(m, s^2), mean = exp(m + s^2/2).
    # Fix s, then choose m so that mean matches mu_g approximately.
    s = 0.7  # variance parameter; reflects finite-structure multiplicativity
    m = math.log(mu_g) - 0.5 * s ** 2

    # Sample lognormal and exponential components
    ln_samples = np.exp(rng.normal(loc=m, scale=s, size=num_gaps))
    exp_samples = rng.exponential(scale=mu_g, size=num_gaps)

    # Mix
    mix_mask = rng.uniform(size=num_gaps) < w
    gaps = np.where(mix_mask, ln_samples, exp_samples)

    # Force integer-like behavior and minimum gap of 2 (rough prime-gap proxy)
    gaps = np.maximum(2, np.round(gaps)).astype(int)

    return gaps, w


# =============================================================================
# 2. Band-wise lognormal vs exponential fitting and ε(n)
# =============================================================================

def fit_exponential_mle(data):
    """
    Fit an exponential distribution with MLE for the rate parameter λ.

    For an exponential with density λ exp(-λ x) on x >= 0, MLE is:
        λ_hat = 1 / mean(data_shifted)
    Here we shift data to min 0, but for prime gaps we are interested
    mostly in the shape; so we keep them as positive and just use mean.
    """
    data = np.asarray(data, dtype=float)
    mean = np.mean(data)
    lam = 1.0 / mean
    return lam


def logpdf_exponential(data, lam):
    """
    Log-density of exponential(λ) on data > 0.
    """
    data = np.asarray(data, dtype=float)
    return np.log(lam) - lam * data


def fit_lognormal_mle(data):
    """
    Fit a lognormal LN(m, s^2) via MLE:

      m_hat = mean(log data)
      s_hat^2 = var(log data)
    """
    data = np.asarray(data, dtype=float)
    logd = np.log(data)
    m_hat = np.mean(logd)
    s_hat = np.std(logd, ddof=1)
    return m_hat, s_hat


def logpdf_lognormal(data, m, s):
    """
    Log-density of lognormal LN(m, s^2) on data > 0.
    """
    data = np.asarray(data, dtype=float)
    logd = np.log(data)
    return -np.log(data * s * math.sqrt(2 * math.pi)) - (logd - m) ** 2 / (2 * s ** 2)


def compute_band_loglik_advantage(scales, all_gaps):
    """
    For each scale n in 'scales', compute the per-gap log-likelihood advantage
    of lognormal over exponential:

        ε(n) = (L_lognormal - L_exponential) / N

    where L_* are summed log-likelihoods over the gaps in that band.

    Returns:
      log_n: np.array of log-scales,
      eps:   np.array of ε(n),
      w_emp: the synthetic "true" mixing weights used in generation.
    """
    log_n = []
    eps = []
    w_emp = []

    for n, gaps, w in zip(scales, all_gaps["gaps"], all_gaps["weights"]):
        # Fit exponential
        lam_hat = fit_exponential_mle(gaps)
        loglik_exp = np.sum(logpdf_exponential(gaps, lam_hat))

        # Fit lognormal
        m_hat, s_hat = fit_lognormal_mle(gaps)
        loglik_ln = np.sum(logpdf_lognormal(gaps, m_hat, s_hat))

        N = len(gaps)
        eps_n = (loglik_ln - loglik_exp) / N

        log_n.append(math.log(n))
        eps.append(eps_n)
        w_emp.append(w)

    return np.array(log_n), np.array(eps), np.array(w_emp)


# =============================================================================
# 3. Z-Form construction: Z = A(B/C)
# =============================================================================

def estimate_B_from_eps(log_n, eps, smoothing=3):
    """
    Estimate B = dε/d(log n) from discrete (log_n, eps) pairs via a
    smoothed finite-difference.

    We first sort by log_n, then compute central differences, and finally
    optionally smooth with a moving average.
    """
    idx = np.argsort(log_n)
    log_n_sorted = log_n[idx]
    eps_sorted = eps[idx]

    # Finite differences
    B = np.zeros_like(eps_sorted)
    for i in range(1, len(log_n_sorted) - 1):
        dlog = log_n_sorted[i + 1] - log_n_sorted[i - 1]
        if dlog != 0:
            B[i] = (eps_sorted[i + 1] - eps_sorted[i - 1]) / dlog
        else:
            B[i] = 0.0
    B[0] = B[1]
    B[-1] = B[-2]

    if smoothing > 1:
        kernel = np.ones(smoothing) / smoothing
        B_s = np.convolve(B, kernel, mode='same')
    else:
        B_s = B

    return log_n_sorted, eps_sorted, B_s


def compute_Z_for_bands(scales, all_gaps, log_n, B):
    """
    Compute Z(n) = A * (B/C) for each band, with:

      A = mean gap at that scale,
      B = local derivative dε/d(log n) at that scale (interpolated),
      C = max |B| across all scales (global invariant bound).[web:43]

    Returns:
      Z:       np.array of Z-values per band,
      A_vals:  mean gaps per band,
      B_vals:  B-values mapped to each band,
      C:       global |B|-bound used.
    """
    # Interpolate B(log n) at the scales' log n
    log_n_scales = np.log(scales)
    B_interp = np.interp(log_n_scales, log_n, B)

    # A = mean gap per band
    A_vals = np.array([np.mean(g) for g in all_gaps["gaps"]])

    # C = invariant bound (max |B|)
    C = np.max(np.abs(B_interp)) if np.max(np.abs(B_interp)) > 0 else 1.0

    Z = A_vals * (B_interp / C)
    return Z, A_vals, B_interp, C


# =============================================================================
# 4. Toy adaptive sieve policy driven by Z
# =============================================================================

def adaptive_sieve_policy(Z, log_n_scales):
    """
    Demonstrate how a Z-based policy might adjust sieve parameters.

    Conceptual mapping:
      - High positive Z: "lognormal-dominated" finite structure regime.
        -> Use wider windows, slightly higher sieve limits.
      - Z near 0: transition regime.
        -> Use moderate windows, combined sieve across many intervals.
      - Negative Z: "exponential-dominated" asymptotic regime.
        -> Use windows ~ [6 log n, 10 log n], standard limits.[web:2]

    Returns:
      policy: list of dicts, one per band.
    """
    policy = []
    for Z_n, ln in zip(Z, log_n_scales):
        n = math.exp(ln)
        # Base window length ~ c * log n
        base_window = 8.0 * ln  # 8 * log n as a neutral baseline

        if Z_n > 0.15:
            # Lognormal-dominated
            window = 1.2 * base_window   # slightly wider window
            sieve_limit_scale = 1.3      # higher sieve limit factor
            regime = "lognormal-dominated"
        elif Z_n < -0.15:
            # Exponential-dominated
            window = base_window         # baseline ~ exponential tail heuristic
            sieve_limit_scale = 1.0
            regime = "exponential-dominated"
        else:
            # Transition
            window = 1.05 * base_window  # modest adjustment
            sieve_limit_scale = 1.1
            regime = "transition"

        policy.append({
            "n": n,
            "log_n": ln,
            "Z": Z_n,
            "window_length": window,
            "sieve_limit_scale": sieve_limit_scale,
            "regime": regime,
        })
    return policy


# =============================================================================
# 5. Visualization and main routine
# =============================================================================

def plot_eps_and_eps_derivative(log_n, eps, B):
    """
    Plot ε(log n) and its derivative B = dε/d(log n).

    This demonstrates:
      - Smooth monotone decline of ε with log n.
      - B approaching zero at large scales, consistent with convergence to an
        exponential fixed point in the Z-domain.[web:43][web:22]
    """
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color_eps = 'tab:blue'
    color_B = 'tab:red'

    ax1.set_xlabel('log n')
    ax1.set_ylabel('ε(n): per-gap loglik advantage (LN - EXP)', color=color_eps)
    ax1.plot(log_n, eps, 'o-', color=color_eps, label='ε(n) (lognormal − exponential)')
    ax1.tick_params(axis='y', labelcolor=color_eps)

    ax2 = ax1.twinx()
    ax2.set_ylabel('B = dε/d(log n)', color=color_B)
    ax2.plot(log_n, B, 's--', color=color_B, label='B = dε/d(log n)')
    ax2.tick_params(axis='y', labelcolor=color_B)

    fig.tight_layout()
    plt.title('Smooth lognormal → exponential transition: ε(n) and its derivative B')
    plt.grid(True, axis='x', linestyle=':')
    plt.show()


def plot_Z_vs_scale(log_n_scales, Z):
    """
    Plot Z(n) vs log n to show the phase-like motion in the Z-domain.

    Here, Z ~ 0 corresponds to the neutral fixed circle where lognormal and
    exponential are equally good; positive Z is lognormal-dominated; negative Z
    is exponential-dominated.[web:43]
    """
    plt.figure(figsize=(8, 4.5))
    plt.axhline(0.0, color='k', linewidth=1, linestyle='--')
    plt.plot(log_n_scales, Z, 'o-', label='Z(n) = A * (B/C)')
    plt.xlabel('log n')
    plt.ylabel('Z(n)')
    plt.title('Z-Form phase: A(B/C) across scales')
    plt.grid(True, linestyle=':')
    plt.legend()
    plt.show()


def print_policy_table(policy, max_rows=12):
    """
    Print a compact table summarizing the toy adaptive sieve policy
    driven by Z.

    Columns:
      log10 n, Z, regime, window_length/(log n), sieve_limit_scale.
    """
    print("\nToy adaptive sieve policy driven by Z = A(B/C):")
    print("  (showing up to {} scales)".format(max_rows))
    header = "{:>10} {:>8} {:>24} {:>16} {:>18}".format(
        "log10 n", "Z", "regime", "window/log n", "sieve_limit_scale")
    print(header)
    print("-" * len(header))

    subset = policy[::max(1, len(policy) // max_rows)]
    for row in subset:
        log10n = math.log10(row["n"])
        ln = row["log_n"]
        Z_n = row["Z"]
        regime = row["regime"]
        window_per_log = row["window_length"] / ln
        print("{:10.3f} {:8.3f} {:>24} {:16.3f} {:18.3f}".format(
            log10n, Z_n, regime, window_per_log, row["sieve_limit_scale"]))


def main():
    # -------------------------------------------------------------------------
    # 5.1 Generate synthetic gaps across scales
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(42)
    scales = generate_scales(num_scales=30, n_min=1e6, n_max=1e14)

    all_gaps = {"gaps": [], "weights": []}
    for n in scales:
        gaps, w = sample_gaps_at_scale(n, num_gaps=50000, rng=rng)
        all_gaps["gaps"].append(gaps)
        all_gaps["weights"].append(w)

    # -------------------------------------------------------------------------
    # 5.2 Compute ε(n) per band
    # -------------------------------------------------------------------------
    log_n_raw, eps_raw, w_emp = compute_band_loglik_advantage(scales, all_gaps)

    # -------------------------------------------------------------------------
    # 5.3 Estimate B = dε/d(log n) and construct Z
    # -------------------------------------------------------------------------
    log_n_sorted, eps_sorted, B = estimate_B_from_eps(log_n_raw, eps_raw, smoothing=5)
    Z, A_vals, B_vals, C = compute_Z_for_bands(scales, all_gaps, log_n_sorted, B)

    # -------------------------------------------------------------------------
    # 5.4 Toy adaptive sieve policy driven by Z
    # -------------------------------------------------------------------------
    log_n_scales = np.log(scales)
    policy = adaptive_sieve_policy(Z, log_n_scales)

    # -------------------------------------------------------------------------
    # 5.5 Visualizations
    # -------------------------------------------------------------------------
    # Plot ε(n) and B
    plot_eps_and_eps_derivative(log_n_sorted, eps_sorted, B)

    # Plot Z vs log n
    plot_Z_vs_scale(log_n_scales, Z)

    # Print a compact policy table
    print_policy_table(policy)


if __name__ == "__main__":
    main()
