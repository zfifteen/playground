#!/usr/bin/env python3
"""
Falsification Experiment: Lognormal vs Exponential Distribution for Prime Gaps

This script implements the falsification test described in TECH-SPEC.md.
It tests whether prime gaps in log-space are better modeled by lognormal distributions
than by exponential distributions across multiple disjoint ranges and bands.
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
from scipy.optimize import minimize
import seaborn as sns
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Metrics for a fitted model on test data."""
    name: str
    params: Dict[str, float]
    log_likelihood: float
    aic: float
    bic: float
    ks_stat: float
    ks_pvalue: float
    ad_stat: Optional[float] = None
    ad_pvalue: Optional[float] = None

@dataclass
class BandResult:
    """Results for a single band."""
    range_id: str
    band_id: int
    p_min: float
    p_max: float
    n_train: int
    n_test: int
    exp_metrics: ModelMetrics
    ln_metrics: ModelMetrics
    winner: str  # "lognormal", "exponential", or "ambiguous"
    delta_bic: float  # BIC_exp - BIC_ln (positive favors lognormal)
    delta_logl: float # logL_ln - logL_exp (positive favors lognormal)

class PrimeGenerator:
    """Generates primes using a segmented sieve."""
    
    def __init__(self):
        pass
        
    def generate_range(self, start: int, end: int) -> np.ndarray:
        """
        Generate primes in the range [start, end].
        Uses a segmented sieve implementation for memory efficiency.
        """
        # For very large ranges, we might need a more optimized C++ implementation
        # or a library like prime-sieve. For this experiment, we'll use a 
        # pure Python implementation but chunked.
        
        # Ensure start is odd (if > 2)
        if start < 2:
            start = 2
        
        # Simple sieve for small ranges, segmented for large
        # For the purpose of this script, we'll use a simple implementation
        # but in a real large-scale run, we'd want something faster.
        # Here we use a basic segmented sieve approach.
        
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
        
        # Handle 0 and 1 if they appear in the range
        if start == 0:
            if length > 0: is_prime[0] = False
            if length > 1: is_prime[1] = False
        elif start == 1:
            if length > 0: is_prime[0] = False
            
        limit = int(np.sqrt(end))
        
        for p in small_primes:
            if p > limit:
                break
                
            # Find first multiple of p >= start
            first_multiple = (start + p - 1) // p * p
            if first_multiple < p * p:
                first_multiple = p * p
                
            # Index in is_prime array
            idx = first_multiple - start
            if idx < length:
                is_prime[idx::p] = False
                
        numbers = np.arange(start, end)
        return numbers[is_prime]

class DistributionFitter:
    """Fits and evaluates distributions."""
    
    def fit_exponential(self, data: np.ndarray) -> Tuple[float, float]:
        """
        Fit exponential distribution.
        Returns (loc, scale) where scale = 1/lambda.
        """
        # MLE for exponential: scale = mean
        loc = 0 # Fixed at 0 for standard exponential
        scale = np.mean(data)
        return loc, scale
        
    def fit_lognormal(self, data: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit lognormal distribution.
        Returns (shape, loc, scale).
        """
        # MLE for lognormal: fit normal to log(data)
        # shape (s) is sigma, scale is exp(mu)
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data)
        
        # Scipy parameterization: s=sigma, scale=exp(mu)
        return sigma, 0, np.exp(mu)
        
    def evaluate_exponential(self, data: np.ndarray, params: Tuple[float, float]) -> ModelMetrics:
        """Evaluate exponential fit on data."""
        loc, scale = params
        n = len(data)
        
        # Log likelihood
        log_l = np.sum(stats.expon.logpdf(data, loc=loc, scale=scale))
        
        # AIC/BIC (k=1 parameter: scale)
        k = 1
        aic = 2*k - 2*log_l
        bic = k*np.log(n) - 2*log_l
        
        # KS Test
        ks_stat, ks_p = stats.kstest(data, 'expon', args=(loc, scale))
        
        return ModelMetrics(
            name="exponential",
            params={"loc": loc, "scale": scale, "lambda": 1.0/scale},
            log_likelihood=log_l,
            aic=aic,
            bic=bic,
            ks_stat=ks_stat,
            ks_pvalue=ks_p
        )
        
    def evaluate_lognormal(self, data: np.ndarray, params: Tuple[float, float, float]) -> ModelMetrics:
        """Evaluate lognormal fit on data."""
        s, loc, scale = params
        n = len(data)
        
        # Log likelihood
        log_l = np.sum(stats.lognorm.logpdf(data, s=s, loc=loc, scale=scale))
        
        # AIC/BIC (k=2 parameters: s, scale)
        k = 2
        aic = 2*k - 2*log_l
        bic = k*np.log(n) - 2*log_l
        
        # KS Test
        ks_stat, ks_p = stats.kstest(data, 'lognorm', args=(s, loc, scale))
        
        return ModelMetrics(
            name="lognormal",
            params={"s": s, "loc": loc, "scale": scale, "mu": np.log(scale), "sigma": s},
            log_likelihood=log_l,
            aic=aic,
            bic=bic,
            ks_stat=ks_stat,
            ks_pvalue=ks_p
        )

class FalsificationExperiment:
    """Main experiment controller."""
    
    def __init__(self, output_dir: str, seed: int = 42):
        self.output_dir = output_dir
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.prime_gen = PrimeGenerator()
        self.fitter = DistributionFitter()
        
        os.makedirs(output_dir, exist_ok=True)
        
    def run(self, ranges: List[str], n_bands: int = 6, min_gaps: int = 5000):
        """Run the full experiment."""
        all_results = []
        
        for range_str in ranges:
            logger.info(f"Processing range: {range_str}")
            try:
                start_str, end_str = range_str.split(':')
                start, end = int(float(start_str)), int(float(end_str))
            except ValueError:
                logger.error(f"Invalid range format: {range_str}. Expected start:end (e.g. 1e8:1e9)")
                continue
                
            # 1. Generate Primes and Gaps
            primes = self.prime_gen.generate_range(start, end)
            if len(primes) < 2:
                logger.warning(f"Not enough primes in range {range_str}")
                continue
                
            gaps = np.diff(primes)
            # Associate gaps with the left prime
            gap_primes = primes[:-1]
            
            # 2. Banding
            log_min = np.log10(start)
            log_max = np.log10(end)
            band_edges = np.logspace(log_min, log_max, n_bands + 1)
            
            for i in range(n_bands):
                band_min = band_edges[i]
                band_max = band_edges[i+1]
                
                # Filter gaps in this band
                mask = (gap_primes >= band_min) & (gap_primes < band_max)
                band_gaps = gaps[mask]
                
                if len(band_gaps) < min_gaps:
                    logger.warning(f"Band {i} ({band_min:.1e}-{band_max:.1e}) has insufficient gaps: {len(band_gaps)}")
                    continue
                    
                logger.info(f"  Band {i}: {len(band_gaps)} gaps")
                
                # 3. Train/Test Split
                # Shuffle
                indices = np.arange(len(band_gaps))
                self.rng.shuffle(indices)
                shuffled_gaps = band_gaps[indices]
                
                split_idx = int(0.7 * len(shuffled_gaps))
                train_gaps = shuffled_gaps[:split_idx]
                test_gaps = shuffled_gaps[split_idx:]
                
                # 4. Fit Models (on Train)
                exp_params = self.fitter.fit_exponential(train_gaps)
                ln_params = self.fitter.fit_lognormal(train_gaps)
                
                # 5. Evaluate Models (on Test)
                exp_metrics = self.fitter.evaluate_exponential(test_gaps, exp_params)
                ln_metrics = self.fitter.evaluate_lognormal(test_gaps, ln_params)
                
                # 6. Compare
                delta_bic = exp_metrics.bic - ln_metrics.bic # Positive means exp has higher BIC (worse) -> Lognormal wins
                delta_logl = ln_metrics.log_likelihood - exp_metrics.log_likelihood
                
                winner = "ambiguous"
                if delta_bic >= 10:
                    winner = "lognormal"
                elif delta_bic <= -10:
                    winner = "exponential"
                    
                result = BandResult(
                    range_id=range_str,
                    band_id=i,
                    p_min=band_min,
                    p_max=band_max,
                    n_train=len(train_gaps),
                    n_test=len(test_gaps),
                    exp_metrics=exp_metrics,
                    ln_metrics=ln_metrics,
                    winner=winner,
                    delta_bic=delta_bic,
                    delta_logl=delta_logl
                )
                all_results.append(result)
                
                # Plotting (optional, for first band of each range or all)
                self._plot_band_fit(test_gaps, exp_params, ln_params, result, range_str, i)

        # Save results
        self._save_results(all_results)
        self._generate_report(all_results)
        
    def _plot_band_fit(self, data: np.ndarray, exp_params, ln_params, result: BandResult, range_id: str, band_id: int):
        """Generate diagnostic plots for the fit."""
        plt.figure(figsize=(12, 5))
        
        # Histogram and PDFs
        plt.subplot(1, 2, 1)
        sns.histplot(data, stat="density", alpha=0.3, label="Data")
        
        x = np.linspace(min(data), max(data), 1000)
        
        # Exp PDF
        loc_exp, scale_exp = exp_params
        y_exp = stats.expon.pdf(x, loc=loc_exp, scale=scale_exp)
        plt.plot(x, y_exp, 'r-', label=f"Exp (BIC={result.exp_metrics.bic:.0f})")
        
        # Lognormal PDF
        s_ln, loc_ln, scale_ln = ln_params
        y_ln = stats.lognorm.pdf(x, s=s_ln, loc=loc_ln, scale=scale_ln)
        plt.plot(x, y_ln, 'g-', label=f"Lognorm (BIC={result.ln_metrics.bic:.0f})")
        
        plt.title(f"Range {range_id} Band {band_id}\nWinner: {result.winner}")
        plt.legend()
        
        # Q-Q Plot (Log-space for Lognormal)
        plt.subplot(1, 2, 2)
        stats.probplot(np.log(data), dist="norm", plot=plt)
        plt.title("Q-Q Plot (Log-Data vs Normal)")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"fit_{range_id.replace(':','_')}_band{band_id}.png"))
        plt.close()

    def _save_results(self, results: List[BandResult]):
        """Save results to JSON and CSV."""
        # JSON
        json_data = [asdict(r) for r in results]
        with open(os.path.join(self.output_dir, "results.json"), "w") as f:
            json.dump(json_data, f, indent=2)
            
        # CSV
        flat_data = []
        for r in results:
            item = {
                "range_id": r.range_id,
                "band_id": r.band_id,
                "p_min": r.p_min,
                "p_max": r.p_max,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "winner": r.winner,
                "delta_bic": r.delta_bic,
                "delta_logl": r.delta_logl,
                "exp_bic": r.exp_metrics.bic,
                "ln_bic": r.ln_metrics.bic,
                "exp_ks": r.exp_metrics.ks_stat,
                "ln_ks": r.ln_metrics.ks_stat,
                "exp_ks_p": r.exp_metrics.ks_pvalue,
                "ln_ks_p": r.ln_metrics.ks_pvalue
            }
            flat_data.append(item)
        
        pd.DataFrame(flat_data).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)

    def _generate_report(self, results: List[BandResult]):
        """Generate a summary report."""
        if not results:
            logger.warning("No results to report.")
            return

        df = pd.DataFrame([asdict(r) for r in results])
        
        report = []
        report.append("# Falsification Test Report: Lognormal vs Exponential\n")
        
        # Overall stats
        n_total = len(results)
        n_ln_wins = sum(1 for r in results if r.winner == "lognormal")
        n_exp_wins = sum(1 for r in results if r.winner == "exponential")
        n_ambiguous = sum(1 for r in results if r.winner == "ambiguous")
        
        report.append("## Summary\n")
        report.append(f"- Total Bands Tested: {n_total}")
        report.append(f"- Lognormal Wins: {n_ln_wins} ({n_ln_wins/n_total*100:.1f}%)")
        report.append(f"- Exponential Wins: {n_exp_wins} ({n_exp_wins/n_total*100:.1f}%)")
        report.append(f"- Ambiguous: {n_ambiguous} ({n_ambiguous/n_total*100:.1f}%)")
        
        # Per Range
        report.append("\n## Per-Range Breakdown\n")
        ranges = sorted(list(set(r.range_id for r in results)))
        
        falsified = False
        falsification_reasons = []
        
        for rid in ranges:
            range_results = [r for r in results if r.range_id == rid]
            n_range = len(range_results)
            n_r_ln = sum(1 for r in range_results if r.winner == "lognormal")
            
            report.append(f"### Range {rid}")
            report.append(f"- Bands: {n_range}")
            report.append(f"- Lognormal Wins: {n_r_ln} ({n_r_ln/n_range*100:.1f}%)")
            
            # Check falsification criteria per range
            if n_r_ln / n_range < 0.5:
                report.append(f"  - **WARNING**: Lognormal win rate < 50%")
        
        # Global Falsification Check
        # Criterion: Lognormal fails if it doesn't beat exponential by ΔBIC ≥ 10 in ≥50% of bands across two independent ranges
        ranges_below_50pct = []
        for rid in ranges:
            range_results = [r for r in results if r.range_id == rid]
            if not range_results: continue
            n_r_ln = sum(1 for r in range_results if r.winner == "lognormal")
            if n_r_ln / len(range_results) < 0.5:
                ranges_below_50pct.append(rid)
                
        if len(ranges_below_50pct) >= 2:
            falsified = True
            falsification_reasons.append(f"Lognormal win rate < 50% in {len(ranges_below_50pct)} ranges: {ranges_below_50pct}")
            
        report.append("\n## Conclusion\n")
        if falsified:
            report.append("**RESULT: FALSIFIED**")
            report.append("The claim that lognormal distributions better model prime gaps has been falsified under the current protocol.")
            for reason in falsification_reasons:
                report.append(f"- {reason}")
        else:
            report.append("**RESULT: NOT FALSIFIED**")
            report.append("The data supports the claim that lognormal distributions provide a better fit than exponential distributions.")
            
        with open(os.path.join(self.output_dir, "report.md"), "w") as f:
            f.write("\n".join(report))
            
        logger.info(f"Report generated at {os.path.join(self.output_dir, 'report.md')}")

def main():
    parser = argparse.ArgumentParser(description="Run Lognormal vs Exponential Falsification Test")
    parser.add_argument("--ranges", type=str, default="1e6:1e7,1e7:1e8", 
                        help="Comma-separated list of ranges (e.g. '1e6:1e7,1e7:1e8')")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bands", type=int, default=6, help="Number of bands per range")
    
    args = parser.parse_args()
    
    ranges = args.ranges.split(',')
    
    experiment = FalsificationExperiment(args.output, args.seed)
    experiment.run(ranges, n_bands=args.bands)

if __name__ == "__main__":
    main()
