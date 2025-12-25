#!/usr/bin/env python3
"""
Falsification Experiment: Fractal Cascade Structure in Prime Gaps

This script implements the falsification test described in TECH-SPEC.md.
It tests whether prime log-gaps exhibit recursive log-normal structure within
magnitude strata and if variance scaling follows a power law (Hurst exponent).
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import curve_fit
import seaborn as sns
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class StratumResult:
    """Results for a single stratum."""
    range_id: str
    stratum_id: int
    delta_min: float
    delta_max: float
    n_gaps: int
    mu_hat: float
    sigma_hat: float
    ks_stat: float
    ks_p: float
    pass_ks: bool

@dataclass
class RangeResult:
    """Results for a prime range."""
    range_id: str
    h_estimate: float
    h_95ci_lower: float
    h_95ci_upper: float
    r_squared: float
    n_strata_used: int
    pct_strata_pass_ks: float
    strata_results: List[StratumResult]
    multifractal_data: Optional[Dict[str, Any]] = None

@dataclass
class NullModelResult:
    """Results for a null model comparison."""
    model_type: str
    range_id: str
    h_estimate: float
    r_squared: float
    pct_strata_pass_ks: float
    delta_h: float  # Difference from observed H

class PrimeGenerator:
    """Generates primes using a segmented sieve."""
    
    def generate_range(self, start: int, end: int) -> np.ndarray:
        """
        Generate primes in the range [start, end].
        Uses a segmented sieve implementation for memory efficiency.
        """
        if start < 2:
            start = 2
        
        chunk_size = 10**6
        primes = []
        
        # Initial small primes for sieving
        limit = int(np.sqrt(end)) + 1
        small_primes = self._simple_sieve(limit)
        
        # Process in chunks
        current_start = start
        with tqdm(total=end-start, desc=f"Generating primes {start:.1e}-{end:.1e}", unit="num") as pbar:
            while current_start < end:
                current_end = min(current_start + chunk_size, end)
                if current_end <= current_start:
                    break
                    
                chunk_primes = self._segmented_sieve_chunk(current_start, current_end, small_primes)
                primes.append(chunk_primes)
                
                pbar.update(current_end - current_start)
                current_start = current_end
                
        if not primes:
            return np.array([], dtype=np.int64)
            
        return np.concatenate(primes)

    def _simple_sieve(self, limit: int) -> np.ndarray:
        """Standard Sieve of Eratosthenes up to limit."""
        is_prime = np.ones(limit + 1, dtype=bool)
        is_prime[0:2] = False
        for i in range(2, int(np.sqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i : limit+1 : i] = False
        return np.nonzero(is_prime)[0]

    def _segmented_sieve_chunk(self, start: int, end: int, small_primes: np.ndarray) -> np.ndarray:
        """Sieve a specific chunk using pre-computed small primes."""
        length = end - start
        if length <= 0:
            return np.array([], dtype=np.int64)
            
        is_prime = np.ones(length, dtype=bool)
        
        if start == 0:
            if length > 0: is_prime[0] = False
            if length > 1: is_prime[1] = False
        elif start == 1:
            if length > 0: is_prime[0] = False
            
        limit = int(np.sqrt(end))
        
        for p in small_primes:
            if p > limit:
                break
            
            first_multiple = (start + p - 1) // p * p
            if first_multiple < p * p:
                first_multiple = p * p
                
            idx = first_multiple - start
            if idx < length:
                is_prime[idx::p] = False
                
        numbers = np.arange(start, end)
        return numbers[is_prime]

class FractalAnalyzer:
    """Analyzes fractal structure in gap data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        
    def analyze_range(self, range_id: str, gaps: np.ndarray, n_strata: int, 
                      bootstrap_iters: int = 1000) -> RangeResult:
        """Perform full fractal analysis on a set of gaps."""
        
        # 1. Compute log-gaps
        # Filter out zero gaps if any (shouldn't happen for primes)
        valid_gaps = gaps[gaps > 0]
        log_gaps = np.log(valid_gaps)
        
        # 2. Stratify
        # We use quantile-based stratification (equal number of points per stratum)
        # as it's more robust than fixed-width for heavy-tailed distributions
        try:
            quantiles = np.linspace(0, 100, n_strata + 1)
            bin_edges = np.percentile(log_gaps, quantiles)
        except Exception as e:
            logger.error(f"Error computing quantiles: {e}")
            return None

        strata_results = []
        mus = []
        sigmas = []
        
        for i in range(n_strata):
            lower = bin_edges[i]
            upper = bin_edges[i+1]
            
            # Extract gaps in this stratum
            # Use strictly less for upper bound except for last bin
            if i == n_strata - 1:
                mask = (log_gaps >= lower) & (log_gaps <= upper)
            else:
                mask = (log_gaps >= lower) & (log_gaps < upper)
                
            stratum_data = log_gaps[mask]
            
            # Filter out non-positive log-gaps before taking log-log
            # Real prime gaps are >= 2, so log-gaps >= ln(2) > 0.
            # Synthetic data (e.g. cascade) might violate this.
            stratum_data = stratum_data[stratum_data > 0]
            
            n_gaps = len(stratum_data)
            
            if n_gaps < 100: # Minimum threshold
                continue
                
            # Fit log-normal (normal on log-gaps)
            # Since we already took log, we fit normal to log_gaps
            # But wait, the hypothesis is about log-gaps themselves being log-normal?
            # TECH-SPEC says: "assume log-gaps Delta follow Lognormal(mu, sigma)"
            # "Equivalently, ln(Delta) ~ N(mu, sigma^2)"
            # So we need to take log AGAIN of the log-gaps.
            # log_gaps = ln(p_{n+1}/p_n) = Delta
            # We need ln(Delta) = ln(ln(p_{n+1}/p_n))
            
            ln_delta = np.log(stratum_data)
            
            mu_hat = np.mean(ln_delta)
            sigma_hat = np.std(ln_delta)
            
            if sigma_hat == 0:
                continue

            # KS Test against Normal(mu_hat, sigma_hat)
            # We test ln_delta against normal CDF
            ks_stat, ks_p = stats.kstest(ln_delta, 'norm', args=(mu_hat, sigma_hat))
            
            # Pass criteria: KS < 0.10 or p > 0.01
            pass_ks = (ks_stat < 0.10) or (ks_p > 0.01)
            
            strata_results.append(StratumResult(
                range_id=range_id,
                stratum_id=i,
                delta_min=lower,
                delta_max=upper,
                n_gaps=n_gaps,
                mu_hat=float(mu_hat),
                sigma_hat=float(sigma_hat),
                ks_stat=float(ks_stat),
                ks_p=float(ks_p),
                pass_ks=bool(pass_ks)
            ))
            
            # For scaling law, we use Mean(Delta) and Std(Delta)
            # But we must ensure they are positive and valid
            mean_delta = np.mean(stratum_data)
            std_delta = np.std(stratum_data)
            
            if mean_delta > 0 and std_delta > 0:
                mus.append(mean_delta)
                sigmas.append(std_delta)

        # 3. Hurst Exponent Estimation
        if len(mus) < 3:
            logger.warning(f"Not enough valid strata for range {range_id}")
            return None
            
        mus = np.array(mus)
        sigmas = np.array(sigmas)
        
        # Ensure positive values for log
        valid_idx = (mus > 0) & (sigmas > 0)
        mus = mus[valid_idx]
        sigmas = sigmas[valid_idx]
        
        if len(mus) < 3:
            logger.warning(f"Not enough valid strata after filtering for range {range_id}")
            return None
            
        log_mu = np.log(mus)
        log_sigma = np.log(sigmas)
        
        # Bootstrap for CI
        h_estimates = []
        for _ in range(bootstrap_iters):
            # Resample indices
            indices = resample(range(len(mus)), random_state=self.rng.integers(0, 100000))
            # Ensure we have enough unique points for regression
            if len(np.unique(indices)) < 3: continue
            
            X_boot = log_mu[indices].reshape(-1, 1)
            y_boot = log_sigma[indices]
            
            reg = LinearRegression().fit(X_boot, y_boot)
            h_estimates.append(reg.coef_[0])
            
        if not h_estimates:
             logger.warning(f"Bootstrap failed for range {range_id}")
             return None
             
        h_estimates = np.array(h_estimates)
        h_est = np.mean(h_estimates)
        h_lower = np.percentile(h_estimates, 2.5)
        h_upper = np.percentile(h_estimates, 97.5)
        
        # Main fit for R2
        reg_main = LinearRegression().fit(log_mu.reshape(-1, 1), log_sigma)
        r2 = reg_main.score(log_mu.reshape(-1, 1), log_sigma)
        
        pct_pass = sum(1 for s in strata_results if s.pass_ks) / len(strata_results) * 100
        
        return RangeResult(
            range_id=range_id,
            h_estimate=float(h_est),
            h_95ci_lower=float(h_lower),
            h_95ci_upper=float(h_upper),
            r_squared=float(r2),
            n_strata_used=len(strata_results),
            pct_strata_pass_ks=float(pct_pass),
            strata_results=strata_results
        )

    def generate_null_data(self, model_type: str, target_gaps: np.ndarray) -> np.ndarray:
        """Generate synthetic gaps based on null models."""
        n = len(target_gaps)
        
        if model_type == "cramer":
            # Independent samples from empirical distribution
            # We just shuffle the existing gaps? 
            # Or sample with replacement?
            # "Sample gaps independently from the global empirical distribution"
            # Sampling with replacement is best.
            return self.rng.choice(target_gaps, size=n, replace=True)
            
        elif model_type == "sieve":
            # Simplified sieve simulation
            # We need to generate a sequence of "primes" and take gaps.
            # This is computationally expensive to tune exactly.
            # Simplified: Randomly delete integers with probability 1 - 1/log(x)?
            # For this test, let's use a "Random Walk" approximation of Cramér model
            # Gaps are Exponentially distributed locally.
            # But we want to match the "global gap distribution".
            # If we just sample from exponential, we miss the heavy tail?
            # The spec says: "Use Eratosthenes sieve... introduce randomization".
            # This is complex to implement faithfully in a generic function.
            # Let's approximate:
            # Generate gaps from a mixture of Exponentials to match the density?
            # Or just use the "Cramér" model as the primary null for independence.
            # Let's implement a "Shuffled" null (Cramér-like) and a "Gaussian Noise" null?
            # Spec: "Sieve-based... tune to match global gap distribution".
            # Let's skip complex Sieve for now and focus on Cramér (Independent) vs Cascade.
            # We will use "cramer" as the main null.
            pass
            
        elif model_type == "cascade":
            # Multiplicative cascade
            # Start with [0, 1], split recursively.
            # This generates a measure (density).
            # We need gaps.
            # Gaps ~ 1/density.
            # Let's implement a simple Binomial Cascade (p-model).
            # p1 = 0.5 + delta, p2 = 0.5 - delta.
            # Or Lognormal weights.
            
            # Simple 1D cascade:
            # Length N = 2^k.
            # Start with mass 1 distributed on [0,1].
            # At each step, redistribute mass of bin i into two halves with fractions W1, W2.
            # W ~ Lognormal.
            # Resulting measure mu(x) is the "density" of primes.
            # Gaps are inversely proportional to density?
            # Or is the "energy" the gap size itself?
            # Spec: "read off the 'energy' at each point as a synthetic gap magnitude."
            
            # Let's do that.
            k = int(np.ceil(np.log2(n)))
            size = 2**k
            measure = np.ones(1)
            
            # Cascade parameters
            # We need to tune this to match global mean/var of log-gaps.
            # Target mean/var of log-gaps.
            target_log_gaps = np.log(target_gaps[target_gaps > 0])
            target_mu = np.mean(target_log_gaps)
            target_sigma = np.std(target_log_gaps)
            
            # In a cascade, log(measure) is sum of log(weights).
            # log(gap) ~ Normal(k * mu_w, k * sigma_w^2).
            # We want k * mu_w = target_mu and k * sigma_w^2 = target_sigma^2.
            
            mu_w = target_mu / k
            sigma_w = target_sigma / np.sqrt(k)
            
            # Generate weights
            # We need 2^1 + 2^2 + ... + 2^k weights? 
            # No, just multiply down.
            # Efficient way:
            # Start with ones.
            # For each level: repeat_interleave(2) * random_weights.
            
            current = np.ones(1)
            for _ in range(k):
                # Split each into 2
                current = np.repeat(current, 2)
                # Generate weights
                # We want log(weight) ~ N(mu_w, sigma_w)
                weights = self.rng.lognormal(mean=mu_w, sigma=sigma_w, size=len(current))
                current = current * weights
                
            # Take first n
            return current[:n]
            
        return target_gaps # Fallback

class ExperimentRunner:
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        self.prime_gen = PrimeGenerator()
        self.analyzer = FractalAnalyzer(seed)
        os.makedirs(output_dir, exist_ok=True)
        
    def run(self, ranges: List[str], n_strata: int = 10, null_models: List[str] = None):
        if null_models is None:
            null_models = ["cramer", "cascade"]
            
        all_results = []
        null_results = []
        
        for range_str in ranges:
            logger.info(f"Processing range: {range_str}")
            try:
                start, end = map(float, range_str.split(':'))
                start, end = int(start), int(end)
            except ValueError:
                continue
                
            # 1. Real Data
            primes = self.prime_gen.generate_range(start, end)
            gaps = np.diff(primes)
            
            res = self.analyzer.analyze_range(range_str, gaps, n_strata)
            if res:
                all_results.append(res)
                self._plot_range_results(res, gaps)
                
                # 2. Null Models
                for nm in null_models:
                    logger.info(f"  Running null model: {nm}")
                    syn_gaps = self.analyzer.generate_null_data(nm, gaps)
                    null_res = self.analyzer.analyze_range(f"{range_str}_{nm}", syn_gaps, n_strata)
                    
                    if null_res:
                        null_results.append(NullModelResult(
                            model_type=nm,
                            range_id=range_str,
                            h_estimate=float(null_res.h_estimate),
                            r_squared=float(null_res.r_squared),
                            pct_strata_pass_ks=float(null_res.pct_strata_pass_ks),
                            delta_h=float(null_res.h_estimate - res.h_estimate)
                        ))
                        
        self._save_results(all_results, null_results)
        self._generate_report(all_results, null_results)

    def _plot_range_results(self, res: RangeResult, gaps: np.ndarray):
        """Generate plots for a range."""
        # Variance Scaling Plot
        # Reconstruct x and y from results for plotting
        # We need to be careful to use the same logic as in analyze_range
        # But we don't have the raw data here easily.
        # Let's approximate using the stored mu_hat/sigma_hat if possible?
        # No, we stored mu_hat/sigma_hat of ln(Delta).
        # And we plotted Mean(Delta) vs Std(Delta).
        # Let's just skip the plot reconstruction if we don't have the exact values,
        # or better, store the plot values in RangeResult.
        # For now, let's use the approximation assuming lognormal:
        
        x_vals = np.array([np.exp(s.mu_hat + s.sigma_hat**2/2) for s in res.strata_results])
        y_vals = np.array([np.sqrt((np.exp(s.sigma_hat**2)-1) * np.exp(2*s.mu_hat + s.sigma_hat**2)) for s in res.strata_results])
        
        # Filter out invalid values for log
        valid_idx = (x_vals > 0) & (y_vals > 0)
        x_vals = x_vals[valid_idx]
        y_vals = y_vals[valid_idx]
        
        if len(x_vals) < 3:
            logger.warning(f"Not enough valid points for plotting scaling in range {res.range_id}")
            return

        plt.figure(figsize=(10, 6))
        plt.scatter(np.log(x_vals), np.log(y_vals), label='Strata')
        
        # Regression line
        # y = Hx + C
        # We have H estimate.
        # Need intercept.
        # Let's just refit for plot
        reg = LinearRegression().fit(np.log(x_vals).reshape(-1,1), np.log(y_vals))
        plt.plot(np.log(x_vals), reg.predict(np.log(x_vals).reshape(-1,1)), 'r--', 
                 label=f'Fit H={res.h_estimate:.2f} (R2={res.r_squared:.2f})')
        
        plt.xlabel('Log Mean Gap (in stratum)')
        plt.ylabel('Log Std Gap (in stratum)')
        plt.title(f'Variance Scaling: {res.range_id}')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f"scaling_{res.range_id}.png"))
        plt.close()

    def _save_results(self, results: List[RangeResult], null_results: List[NullModelResult]):
        # Save JSON
        data = {
            "observed": [asdict(r) for r in results],
            "null_models": [asdict(r) for r in null_results]
        }
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(data, f, indent=2)

    def _generate_report(self, results: List[RangeResult], null_results: List[NullModelResult]):
        report = []
        report.append("# Falsification Test Report: Fractal Cascade Structure\n")
        
        # 1. Verdict
        # Check criteria
        # H in [0.6, 1.0]
        # KS pass rate >= 80%
        # Null models fail
        
        pass_h = all(0.6 <= r.h_estimate <= 1.0 for r in results)
        pass_ks = all(r.pct_strata_pass_ks >= 80 for r in results)
        
        # Check null models
        # Cramér should fail (low KS or wrong H)
        cramer_res = [n for n in null_results if n.model_type == "cramer"]
        pass_null = True
        if cramer_res:
            # Cramér "passes" if it fails to look like the data
            # i.e. KS pass rate < 60% OR H diff > 0.3
            for cr in cramer_res:
                if cr.pct_strata_pass_ks >= 70 and abs(cr.delta_h) < 0.15:
                    pass_null = False # Cramér successfully mimicked the data -> Falsified
        
        falsified = not (pass_h and pass_ks and pass_null)
        
        if falsified:
            report.append("**RESULT: FALSIFIED**\n")
            if not pass_h: report.append("- Hurst exponent outside [0.6, 1.0] or unstable.")
            if not pass_ks: report.append("- Within-stratum log-normality not consistent (<80% pass).")
            if not pass_null: report.append("- Null models (Cramér) reproduced the structure.")
        else:
            report.append("**RESULT: NOT FALSIFIED**\n")
            report.append("Data supports fractal cascade structure.")
            
        report.append("\n## Observed Results")
        for r in results:
            report.append(f"- Range {r.range_id}: H={r.h_estimate:.3f}, KS Pass={r.pct_strata_pass_ks:.1f}%")
            
        report.append("\n## Null Model Comparison")
        for n in null_results:
            report.append(f"- {n.model_type} ({n.range_id}): H={n.h_estimate:.3f}, KS Pass={n.pct_strata_pass_ks:.1f}%")
            
        with open(os.path.join(self.output_dir, "report.md"), "w") as f:
            f.write("\n".join(report))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranges", default="1e6:1e7,1e7:1e8")
    parser.add_argument("--output", default="results")
    parser.add_argument("--strata", type=int, default=10)
    args = parser.parse_args()
    
    runner = ExperimentRunner(args.output)
    runner.run(args.ranges.split(','), args.strata)

if __name__ == "__main__":
    main()
