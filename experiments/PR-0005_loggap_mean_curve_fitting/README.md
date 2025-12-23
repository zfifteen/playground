# PR-0005: Log-Gap Mean Curve Fitting

## Overview

This experiment fits simple deterministic models to the relationship between log-prime values and mean log-gaps. The goal is to identify which functional form best describes the decay trend observed in the binned log-gap means.

## Background

From PR-0003, we have binned statistics showing that mean log-gaps decrease as log-primes increase. This experiment fits several candidate curve models to quantify this relationship.

## Data Source

The experiment uses pre-computed binning statistics from PR-0003:
- `data/bin_stats_1e5.json` - Statistics for primes up to 10^5
- `data/bin_stats_1e6.json` - Statistics for primes up to 10^6
- `data/bin_stats_1e7.json` - Statistics for primes up to 10^7
- `data/bin_stats_1e8.json` - Statistics for primes up to 10^8
- `data/bin_stats_1e9.json` - Statistics for primes up to 10^9

Each file contains:
- Bin centers on the log-prime axis
- Mean log-gap per bin (with NaNs for empty bins)
- Additional statistics (variance, skewness, kurtosis)

## Models Tested

1. **loggap_mean_curve_linear**: Simple linear model
   - Formula: `y = a + b * logp`
   - Parameters: a (intercept), b (slope)

2. **loggap_mean_curve_normed**: Normalized linear model
   - Formula: `y = a + b * t` where `t = (logp - min) / (max - min)`
   - Parameters: a (intercept), b (slope)

3. **loggap_mean_curve_loglog**: Log-log linear model
   - Formula: `y = a + b * log(logp)`
   - Parameters: a (intercept), b (slope)

4. **loggap_mean_curve_power_normed**: Power-law model
   - Formula: `y = a + b * t^c` where `t = (logp - min) / (max - min)`
   - Parameters: a (intercept), b (coefficient), c (exponent)

## Usage

```bash
cd experiments/PR-0005_loggap_mean_curve_fitting
python3 fit_loggap_curves.py
```

## Results

The script outputs for each scale:
- Model name
- Mean Squared Error (MSE)
- R² (coefficient of determination)
- Fitted parameters

Models are sorted by MSE (best fit first).

### Key Findings

Across all scales (10^5 to 10^9):

1. **Best performing model**: `loggap_mean_curve_loglog`
   - Consistently achieves the lowest MSE
   - R² ranges from ~0.65 to ~0.77
   - This suggests the relationship is approximately linear in log(log(p))

2. **Second best**: `loggap_mean_curve_power_normed`
   - Similar performance to loglog
   - Provides a flexible power-law form

3. **Simple linear models** perform worst
   - R² ranges from ~0.34 to ~0.52
   - The relationship is clearly non-linear in logp

### Trend Observation

As the scale increases from 10^5 to 10^9:
- MSE for all models tends to increase slightly
- R² for all models tends to decrease
- This suggests the relationship becomes slightly noisier at larger scales

## Implementation Details

- Uses `scipy.optimize.curve_fit` for parameter estimation
- Filters out NaN values from bin_means before fitting
- Computes bin centers from metadata if not present in JSON
- Uses bounds on power-law exponent to ensure numerical stability

## Dependencies

- numpy
- scipy
- Python 3.8+

## References

- Based on binning methodology from PR-0003
- Data generated using prime generation from PR-0003
