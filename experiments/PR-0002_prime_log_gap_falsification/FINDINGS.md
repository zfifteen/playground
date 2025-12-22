# FINDINGS: Prime Log-Gap Falsification Experiment

**PR Number:** PR-0002  
**Date:** 2025-12-22  
**Author:** GitHub Copilot  
**Status:** Complete  

---

## CONCLUSION

**The hypothesis that prime number gaps in logarithmic space exhibit circuit-like damped impulse response behavior is SUPPORTED (not falsified).**

This conclusion is based on rigorous statistical testing across three orders of magnitude (10⁶, 10⁷, 10⁸ primes) with all pre-defined falsification criteria failing to trigger. The evidence consistently shows:

1. **Log-normal distribution fits significantly better than normal** (KS ratio > 9x)
2. **Monotonic decay of mean log-gap** across quintiles at all scales
3. **Strong autocorrelation structure** at all lags (not white noise)
4. **Heavy-tailed distribution** with extreme skewness and kurtosis
5. **Scale-consistent results** across all tested magnitudes

---

## Executive Summary

| Falsification Criterion | Status | Evidence |
|------------------------|--------|----------|
| F1: Non-decreasing quintile trend | NOT FALSIFIED | Negative slopes at all scales |
| F2: Normal fits better than log-normal | NOT FALSIFIED | KS ratio = 9.36 to 11.28 |
| F4: White noise (no autocorrelation) | NOT FALSIFIED | Significant ACF at lags 1-20 |
| F5: Normal-like skewness/kurtosis | NOT FALSIFIED | Skewness = 89 to 768 |
| F6: Scale-dependent reversals | NOT FALSIFIED | Consistent results across scales |

---

## Detailed Results by Scale

### Scale 10⁶ (78,498 primes)

| Metric | Value |
|--------|-------|
| Log-gap count | 78,497 |
| Log-gap range | [0.000002, 0.510826] |
| Mean log-gap | 0.000167 |
| Standard deviation | 0.003816 |
| Skewness | 89.81 |
| Excess kurtosis | 9,716.91 |

**Quintile Analysis:**
| Quintile | Mean Log-Gap |
|----------|--------------|
| Q1 | 0.000724 |
| Q2 | 0.000048 |
| Q3 | 0.000028 |
| Q4 | 0.000020 |
| Q5 | 0.000015 |

- **Decay ratio (Q1/Q5):** 46.81
- **Regression slope:** -1.45 × 10⁻⁴
- **R²:** 0.538

**Distribution Fit:**
| Distribution | KS Statistic |
|--------------|--------------|
| Log-normal | 0.0516 (BEST) |
| Normal | 0.4827 |
| Exponential | — |

**KS Ratio (Normal/Log-normal):** 9.36

---

### Scale 10⁷ (664,579 primes)

| Metric | Value |
|--------|-------|
| Log-gap count | 664,578 |
| Log-gap range | [0.000000, 0.510826] |
| Mean log-gap | 0.000023 |
| Standard deviation | 0.001312 |
| Skewness | 261.02 |
| Excess kurtosis | 82,136.14 |

**Quintile Analysis:**
| Quintile | Mean Log-Gap |
|----------|--------------|
| Q1 | 0.000103 |
| Q2 | 0.000006 |
| Q3 | 0.000003 |
| Q4 | 0.000002 |
| Q5 | 0.000002 |

- **Decay ratio (Q1/Q5):** 57.20
- **Regression slope:** -2.06 × 10⁻⁵
- **R²:** 0.531

**Distribution Fit:**
| Distribution | KS Statistic |
|--------------|--------------|
| Log-normal | 0.0466 (BEST) |
| Normal | 0.4930 |

**KS Ratio (Normal/Log-normal):** 10.58

---

### Scale 10⁸ (5,761,455 primes)

| Metric | Value |
|--------|-------|
| Log-gap count | 5,761,454 |
| Log-gap range | [0.000000, 0.510826] |
| Mean log-gap | 0.000003 |
| Standard deviation | 0.000446 |
| Skewness | 768.37 |
| Excess kurtosis | 711,847.52 |

**Quintile Analysis:**
| Quintile | Mean Log-Gap |
|----------|--------------|
| Q1 | 0.000014 |
| Q2 | 0.000001 |
| Q3 | 0.000000 |
| Q4 | 0.000000 |
| Q5 | 0.000000 |

- **Decay ratio (Q1/Q5):** 67.58
- **Regression slope:** -2.78 × 10⁻⁶
- **R²:** 0.526

**Distribution Fit:**
| Distribution | KS Statistic |
|--------------|--------------|
| Log-normal | 0.0441 (BEST) |
| Normal | 0.4973 |

**KS Ratio (Normal/Log-normal):** 11.28

---

## Cross-Scale Consistency

| Scale | Mean Log-Gap | Decay Ratio | KS Ratio |
|-------|--------------|-------------|----------|
| 10⁶ | 0.000167 | 46.81 | 9.36 |
| 10⁷ | 0.000023 | 57.20 | 10.58 |
| 10⁸ | 0.000003 | 67.58 | 11.28 |

**Observations:**
1. Mean log-gap decreases systematically with scale (as expected from prime thinning)
2. Decay ratio increases slightly with scale, suggesting stronger decay at larger primes
3. KS ratio (normal/log-normal) increases with scale, indicating log-normal fit improves

---

## Autocorrelation Analysis

At all scales, the Ljung-Box test showed significant autocorrelation (p < 0.05) at all lags from 1 to 20. This strongly rejects the null hypothesis of white noise (H0-C).

**Significant lags at all scales:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20

This indicates:
- Prime gaps exhibit **short-range memory** structure
- The "filter memory" model from the circuit analogy is consistent with data
- Log-gaps are not independent draws from any distribution

---

## Falsification Criteria Evaluation

### F1: Non-decreasing quintile trend

**Status: NOT FALSIFIED**

At all scales, the quintile means show a clear decreasing pattern from Q1 to Q5. The regression slopes are all negative:
- 10⁶: slope = -1.45 × 10⁻⁴
- 10⁷: slope = -2.06 × 10⁻⁵
- 10⁸: slope = -2.78 × 10⁻⁶

While the R² values (~0.53) indicate the relationship is not perfectly linear, the decreasing trend is unmistakable and consistent.

### F2: Normal fits better than log-normal

**Status: NOT FALSIFIED**

The log-normal distribution consistently fits the data much better than the normal distribution:
- KS statistic for log-normal: 0.044 - 0.052
- KS statistic for normal: 0.483 - 0.497
- KS ratio: 9.36 to 11.28

This strongly supports the hypothesis of multiplicative (log-normal) rather than additive (normal) structure.

### F4: White noise (no autocorrelation)

**Status: NOT FALSIFIED**

The Ljung-Box test rejected white noise at all lags. The ACF shows significant positive autocorrelation at low lags, consistent with the "filter memory" aspect of the circuit analogy.

### F5: Normal-like skewness/kurtosis

**Status: NOT FALSIFIED**

The distribution exhibits extreme skewness and kurtosis:
- Skewness: 89.81 to 768.37 (normal: 0)
- Excess kurtosis: 9,716 to 711,847 (normal: 0)

These values are vastly larger than the falsification thresholds (|skewness| < 0.5, |excess kurtosis| < 1), confirming heavy-tailed behavior.

### F6: Scale-dependent reversals

**Status: NOT FALSIFIED**

Results are directionally consistent across all scales:
- All slopes are negative
- All decay ratios are > 1 (Q1 > Q5)
- Log-normal consistently beats normal
- Autocorrelation structure persists

---

## Theoretical Implications

The results support the circuit analogy interpretation:

| Circuit Concept | Number-Theoretic Observation |
|-----------------|------------------------------|
| Damped response | Mean log-gap decreases with prime index |
| Multiplicative structure | Log-normal distribution fits best |
| Filter memory | Short-range autocorrelation present |
| Heavy tails | Extreme events (large gaps) occur more often than normal |

The observation that log-gaps decrease as primes increase is consistent with treating ln(n) as a "voltage" that becomes denser at higher values—analogous to an RC circuit approaching steady state.

---

## Limitations

1. **R² for quintile regression is moderate (~0.53)**: The decay is not perfectly linear, suggesting the relationship may be more complex than a simple linear trend.

2. **Numerical precision**: At very large scales, very small log-gaps may approach machine precision limits.

3. **Maximum scale tested (10⁸)**: Results may change at scales beyond 10⁹.

4. **Causation vs correlation**: While the data is consistent with the circuit analogy, this does not prove the analogy captures the true underlying mechanism.

---

## Reproducibility

All results can be reproduced by running:

```bash
cd experiments/PR-0002_prime_log_gap_falsification
python3 run_experiment.py
```

**Dependencies:**
- Python 3.8+
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- pandas >= 1.3.0

**Prime count validation:**
- π(10⁶) = 78,498 ✓
- π(10⁷) = 664,579 ✓
- π(10⁸) = 5,761,455 ✓

---

## Figures

All generated figures are available in `results/figures/`:

- `log_gap_histogram_*.png` - Distribution histograms with log-normal fit
- `qq_plot_*.png` - Q-Q plots for log-normal assessment
- `decay_trend_*.png` - Quintile mean decay visualization
- `acf_pacf_*.png` - Autocorrelation function plots
- `scale_comparison.png` - Cross-scale comparison summary

---

## Conclusion

This experiment was designed to rigorously test and potentially falsify the hypothesis that prime log-gaps exhibit circuit-like damped impulse response behavior. After testing across three orders of magnitude with pre-defined falsification criteria:

**No falsification criteria were triggered.**

The hypothesis remains **supported but not proven**. The data shows:
- Log-normal (not normal) distribution
- Monotonic decay of mean log-gap
- Short-range autocorrelation (not white noise)
- Consistent behavior across scales

Future work should:
1. Extend to 10⁹ and beyond
2. Investigate the specific functional form of decay
3. Explore transfer function formulations
4. Test connections to Riemann zeta structure
