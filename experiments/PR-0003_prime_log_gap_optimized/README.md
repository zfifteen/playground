# PR-0003: Prime Log-Gap Analysis with 100 Bins (Optimized)

## Overview

This experiment reimplements the prime log-gap analysis from PR-0002 with significant improvements:
- **100 equal-width bins on the log-prime axis** (not on prime index)
- **Support for primes up to 10⁹** (vs 10⁸ in PR-0002) - design capability
- **Tested and validated at 10^7 scale** (664,579 primes)
- **Disk caching** for primes and computed gaps
- **Segmented sieve** for memory-efficient prime generation
- **17 total plots** (12 2D + 5 3D) for comprehensive visualization

## Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Max prime (design) | 10⁹ | Extended from 10⁸ for better statistical power |
| Max prime (tested) | 10^7 | Validated scale (664,579 primes) |
| Number of bins | 100 | Increased from 50 for finer granularity |
| Binning strategy | Equal-width on log-prime axis | Differs from PR-0002 (bins by index) |
| Segment size | 10⁶ | Memory-efficient sieving |

## File Structure

```
experiments/PR-0003_prime_log_gap_optimized/
├── README.md (this file)
├── run_experiment.py (main entry point)
├── src/
│   ├── __init__.py
│   ├── prime_generator.py (segmented sieve with caching)
│   ├── binning.py (100 bins on log-prime axis)
│   ├── statistics.py (regression, KS, Ljung-Box)
│   ├── visualization_2d.py (12 2D plots)
│   └── visualization_3d.py (5 3D plots)
├── data/ (cached primes and gaps)
└── results/ (plots and results.json)
```

## How to Run

### Basic Usage

```bash
cd experiments/PR-0003_prime_log_gap_optimized
python run_experiment.py
```

This will:
1. Generate primes up to the specified limit (default: 10⁹) or load from cache
2. Compute log-gaps (or load from cache)
3. Bin data into 100 equal-width bins on log-prime axis
4. Run all statistical tests
5. Generate all 17 plots
6. Save results to `results/results.json`

**Estimated time:** 
- 10^6: ~13 seconds
- 10^7: ~95 seconds (validated)
- 10^8: ~15 minutes (estimated)
- 10⁹: ~3 hours* (estimated, not tested)

*Note: Default is 10⁹ but only 10^7 has been validated. Use `--max-prime` to specify scale.

### Advanced Options

```bash
# Use smaller max prime for testing
python run_experiment.py --max-prime 1e8 --bins 100

# Disable caching (always regenerate)
python run_experiment.py --no-cache

# Verbose output
python run_experiment.py --verbose
```

## Output Files

### Data Files

| File | Description |
|------|-------------|
| `data/primes_<max_prime>.npy` | Cached prime array (e.g., `primes_10000000.npy` for 10^7) |
| `data/gaps_<max_prime>.npz` | Cached regular gaps, log-gaps, log-primes |

Note: Data files are excluded from git (regenerable via caching).

### Results

| File | Description |
|------|-------------|
| `results/results.json` | Complete analysis results and parameters |

### 2D Plots (12 total)

| File | Description |
|------|-------------|
| `decay_trend.png` | Bin index vs mean log-gap with regression line |
| `log_gap_histogram.png` | Histogram of all log-gaps (100 bins) |
| `qq_plot_lognormal.png` | Q-Q plot against log-normal distribution |
| `acf.png` | Autocorrelation function for lags 1-50 |
| `pacf.png` | Partial autocorrelation function |
| `log_prime_vs_log_gap.png` | Scatter: ln(prime) vs log-gap |
| `box_plot_per_bin.png` | Box plots by 100 bins |
| `cdf.png` | Empirical vs log-normal CDF |
| `kde.png` | Kernel density estimate with log-normal overlay |
| `regression_residuals.png` | Residuals vs bin index |
| `log_gap_vs_regular_gap.png` | Regular gap vs log-gap scatter |
| `prime_density.png` | Prime index vs ln(prime) |

### 3D Plots (5 total)

| File | Description |
|------|-------------|
| `scatter_3d.png` | 3D scatter: (index, log-prime, log-gap) |
| `surface_3d.png` | 2D histogram surface of (log-prime, log-gap) |
| `contour_3d.png` | ACF as function of lag and scale |
| `wireframe_3d.png` | Bin means by bin index and scale |
| `bar_3d.png` | Skewness and kurtosis per bin group |

## Key Differences from PR-0002

| Aspect | PR-0002 | PR-0003 (This) |
|--------|---------|----------------|
| Max prime (design) | 10⁸ | 10⁹ |
| Max prime (tested) | 10⁸ | 10^7 |
| Binning strategy | 50 bins by index | 100 bins on log-prime axis |
| Bin count | 50 | 100 |
| Caching | Partial | Full (primes + gaps) |
| 2D plots | 4 | 12 |
| 3D plots | 0 | 5 |
| Total plots | 4 | 17 |

## Binning Strategy Explained

**PR-0002 approach:** Divided primes into 50 equal-count groups by their index position.
- Bin 1: primes 1 to N/50
- Bin 2: primes N/50+1 to 2N/50
- etc.

**PR-0003 approach (this):** Divides the log-prime range into 100 equal-width intervals.
- If primes range from p_min to p_max
- Bin edges at: ln(p_min), ln(p_min) + Δ, ln(p_min) + 2Δ, ..., ln(p_max)
- Where Δ = (ln(p_max) - ln(p_min)) / 100
- Each bin contains primes whose logarithm falls in that interval

**Why this matters:** The new approach captures how log-gaps change with the magnitude of primes (log-scale), not just their sequential position. This better aligns with the hypothesis that gaps behave multiplicatively.

## Statistical Tests (Preserved from PR-0002)

All tests from PR-0002 are preserved with updated binning:

| Test | Purpose | Hypothesis |
|------|---------|------------|
| Linear regression | Bin mean decay | H-MAIN-A: Mean log-gap decreases |
| Kolmogorov-Smirnov | Distribution fit | H-MAIN-B: Log-normal distribution |
| Ljung-Box | Autocorrelation | H-MAIN-C: Short-range memory |
| Skewness/Kurtosis | Distribution shape | Check for heavy tails |

See PR-0002/SPEC.md for detailed hypothesis definitions.

## Dependencies

```
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.5.0
statsmodels >= 0.13.0
```

Install with:
```bash
pip install numpy scipy matplotlib statsmodels
```

## Performance Notes

### Tested at Scale

| Scale | Primes | Time | Data Size | Status |
|-------|--------|------|-----------|--------|
| 10^6 | 78,498 | ~13s | 2.5 MB | ✅ Validated |
| 10^7 | 664,579 | ~95s | 26 MB | ✅ Validated |
| 10^8 | 5.76M | ~12-20 min* | ~180 MB* | ⚠️ Estimated |
| 10^9 | 50.8M | ~1.5-3 hrs* | ~1.6 GB* | ⚠️ Estimated |

*Estimated based on observed sub-linear scaling (time grows ~0.86x as fast as data). See `RESULTS_AT_SCALE.md` for detailed 10^7 results.

### Resource Requirements

- **Memory usage:** ~3 GB peak at 10^7
- **Disk space:** ~26 MB cached data at 10^7
- **Caching:** Subsequent runs ~10x faster

## Validation Status

### Actually Tested
- ✅ **10^6 scale:** 78,498 primes, 13s execution
- ✅ **10^7 scale:** 664,579 primes, 95s execution (comprehensive validation)

### Design Capability
- ⚙️ **10^8 scale:** Estimated 5.76M primes, ~15 minutes (not tested)
- ⚙️ **10^9 scale:** Estimated 50.8M primes, ~3 hours (not tested)

**Note:** The implementation supports scales up to 10^9, but only 10^6 and 10^7 have been validated. See `RESULTS_AT_SCALE.md` for comprehensive 10^7 analysis.

## References

1. **PR-0002 SPEC.md:** Technical design specification for hypothesis and tests
2. **Prime Number Theorem:** π(x) ~ x/ln(x)
3. **Cramér's model:** Prime gap distribution theory

## Author

GitHub Copilot (Incremental Coder Agent)

## License

Same as parent repository
