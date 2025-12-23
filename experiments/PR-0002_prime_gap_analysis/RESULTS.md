# Prime Gap Distribution Analysis - Results Summary

**Date:** 2025-12-23  
**Scale:** 10^6 (1,000,000)  
**Status:** Phase 1 Validation Complete ✓

## Overview

This experiment tested whether prime number gaps exhibit systematic deviations from Prime Number Theorem (PNT) predictions and whether their distributions follow lognormal or other structured patterns.

## Hypothesis Testing Results

### H-MAIN-A: Gap Growth Relative to PNT

**Result:** **REJECTED H0** - Sub-logarithmic growth detected

- **Overall mean gap/log(p):** 1.0017 (very close to PNT prediction of 1.0)
- **Regression slope:** -0.00344 (slightly negative)
- **R²:** 0.146
- **p-value:** 0.000234 (< 0.01, highly significant)
- **95% CI:** [-0.00522, -0.00166]

**Interpretation:** While the mean is extremely close to 1.0, there is a statistically significant negative trend across scales. This suggests gaps grow *slightly* slower than logarithmically, though the effect size is very small. The practical significance is minimal, but the statistical evidence is clear.

### H-MAIN-B: Lognormal Gap Distribution

**Result:** **INCONCLUSIVE** - Inconsistent across scales

- **Lognormal best fit:** 1 band (1e5_1e6)
- **Exponential best fit:** 0 bands
- **Other distributions:** Not tested at this scale (need 10^7 and 10^8)

**Interpretation:** Only one magnitude band available at 10^6 scale. Need larger scales (10^7, 10^8) to test consistency requirement (≥2 of 3 bands).

### H-MAIN-C: Gap Autocorrelation

**Result:** **REJECTED H0** - Autocorrelation detected

- **Ljung-Box Q:** 381.69
- **p-value:** < 0.000001 (highly significant)
- **Significant lags:** 25 out of 40 tested
- **Key lags:** 1, 2, 9, 12, 15, 16, 17, 19, 20, 21

**Interpretation:** Strong evidence for autocorrelation in prime gaps. Consecutive gaps are NOT independent. This invalidates simple "random sieve" models and suggests gaps have memory/structure.

## Validation Results

### Prime Count Validation ✓

All prime counts match expected values:
- π(10³) = count correct
- π(10⁴) = count correct
- π(10⁵) = 9,592 ✓
- π(10⁶) = 78,498 ✓

### Maximal Gap Validation ✓

All maximal gaps match computed values:
- Up to 10³: gap=20, prime=887 ✓
- Up to 10⁴: gap=36, prime=9,551 ✓
- Up to 10⁵: gap=72, prime=31,397 ✓
- Up to 10⁶: gap=114, prime=492,113 ✓

### Gap Properties ✓

- All gaps positive ✓
- First gap is 1 (2→3) ✓
- All other gaps even ✓
- Mode gap: 6 (at this scale)

## Implementation Correctness

All validation tests passed:
- ✓ Gap calculation correct
- ✓ log(gap) computed as log of magnitude, not log of ratio
- ✓ Array alignment correct
- ✓ PNT normalization within tolerance (mean ≈ 1.0)
- ✓ Maximal gaps match expected values

## Key Findings

1. **PNT is remarkably accurate:** mean(gap/log(p)) = 1.0017, within 0.17% of prediction
2. **Slight sub-logarithmic trend:** Statistically significant but practically negligible
3. **Strong autocorrelation:** Gaps are NOT random - they have memory
4. **Lognormal hypothesis:** Needs larger scales to test properly

## Generated Outputs

### Figures
- `results/figures/pnt_deviation.png` - Visualization of gap/log(p) vs prime magnitude
- `results/figures/acf_plot.png` - Autocorrelation function with confidence bands
- `results/figures/gap_histogram.png` - Distribution of raw and log gaps
- `results/figures/qq_plot_1e5_1e6.png` - Q-Q plot for lognormal test

### Data
- `results/analysis_results_1000000.json` - Complete numerical results
- `data/primes_1000000.npy` - Cached prime array

## Next Steps

### Phase 2: Extension to 10^7
- Generate primes to 10,000,000
- Test distribution consistency (2 bands available)
- Verify autocorrelation persists at larger scale
- Compare PNT deviation across scales

### Phase 3: Full Scale to 10^8
- Generate primes to 100,000,000
- Full 3-band distribution testing
- Final cross-scale validation
- Document final conclusions

## Reproducibility

All results are reproducible using:
```bash
python run_experiment.py --scale 1e6
```

Random seed: Fixed in code
Environment: Python 3.12, numpy, scipy, matplotlib
Git commit: [to be added]

## Conclusion

Phase 1 validation at 10^6 scale is **SUCCESSFUL**. All implementation checks passed, and we have found:
1. PNT is confirmed to high precision
2. Autocorrelation in gaps is real and strong
3. Need larger scales to test lognormal hypothesis

The implementation is correct and ready for Phase 2 scaling.
