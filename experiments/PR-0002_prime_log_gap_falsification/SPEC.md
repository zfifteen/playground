# Technical Design Specification: Prime Log-Gap Falsification Experiment

**PR Number:** PR-0002  
**Date:** 2025-12-21  
**Author:** zfifteen  
**Status:** Draft  

---

## 1. Executive Summary

This document specifies an experiment designed to rigorously test and potentially falsify the hypothesis that prime number gaps in logarithmic space exhibit circuit-like damped impulse response behavior. The experiment extends preliminary findings (9,592 primes to 10⁵) to larger scales (10⁸+ primes) with formal statistical validation.

---

## 2. Hypothesis Under Test

### 2.1 Primary Hypothesis (H-MAIN)

**Statement:** Prime gaps in log-space, defined as Δₙ = ln(pₙ₊₁) - ln(pₙ) = ln(pₙ₊₁/pₙ), exhibit statistical properties consistent with a multiplicative damped system, specifically:

1. **H-MAIN-A (Decay):** Mean log-gap decreases monotonically as primes increase
2. **H-MAIN-B (Distribution):** Log-gaps follow a log-normal or related heavy-tailed multiplicative distribution
3. **H-MAIN-C (Memory):** Log-gap autocorrelation exhibits short-range structure (non-zero at low lags)

### 2.2 Null Hypotheses (For Falsification)

| ID | Null Hypothesis | Falsification Condition |
|----|-----------------|------------------------|
| H0-A | Log-gap mean is constant or increasing | Quintile means show no decrease or increase with p |
| H0-B | Log-gaps are normally distributed | KS test p-value > 0.05 for normal fit, AND normal KS < log-normal KS |
| H0-C | Log-gaps are uncorrelated (white noise) | Ljung-Box test p-value > 0.05 at all lags ≤ 20 |
| H0-D | Log-gaps follow simple exponential decay | Exponential KS statistic < log-normal KS statistic |

### 2.3 Connection to Circuit Analogy

The hypothesis derives from the mapping:

| Electrical Domain | Number-Theoretic Domain |
|-------------------|------------------------|
| Voltage v | Logarithm ln(n) |
| Current i | Log-ratio ln(pₙ₊₁/pₙ) |
| Impulse response | Prime gap distribution |
| Damping coefficient | Log-gap decay rate |
| RC time constant | Characteristic scale of gap correlation |

### 2.4 Connection to Relativistic Bounds

The Gist "Proof and Analysis of Relativistic Doppler Shift Bounds" establishes:

$$\frac{\delta}{1+\delta} < \beta < \frac{\delta}{1-\delta} \quad (0 < \delta < 1)$$

via the rapidity identity artanh(β) = ln(1+δ). This logarithmic compression parallels the treatment of integers as "potentials" in the circuit analogy, where logarithms bound multiplicative growth analogously to how β < 1 bounds relativistic velocity.

---

## 3. Experimental Design

### 3.1 Data Generation

#### 3.1.1 Prime Generation Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Minimum prime | 2 | Include full range |
| Maximum prime | 10⁸ (extendable to 10⁹) | Sufficient for statistical power |
| Expected count | ~5.76 × 10⁶ primes | From π(10⁸) ≈ 5,761,455 |
| Generation method | Segmented Sieve of Eratosthenes | Memory-efficient for large ranges |
| Validation | Cross-check count against known π(x) values | Ensure no computational errors |

#### 3.1.2 Known Validation Points

| x | π(x) | Source |
|---|------|--------|
| 10⁶ | 78,498 | Prime counting function tables |
| 10⁷ | 664,579 | Prime counting function tables |
| 10⁸ | 5,761,455 | Prime counting function tables |

### 3.2 Derived Quantities

For each consecutive prime pair (pₙ, pₙ₊₁):

```
log_gap[n] = ln(p[n+1]) - ln(p[n]) = ln(p[n+1] / p[n])
regular_gap[n] = p[n+1] - p[n]
log_prime[n] = ln(p[n])
```

### 3.3 Statistical Tests

#### 3.3.1 Test Suite

| Test ID | Test Name | Target Hypothesis | Implementation |
|---------|-----------|-------------------|----------------|
| T1 | Quintile Mean Regression | H-MAIN-A | Linear regression of quintile means vs quintile index |
| T2 | Decile Mean Regression | H-MAIN-A | Finer granularity check (10 bins) |
| T3 | KS Test Battery | H-MAIN-B | Fit and compare: normal, log-normal, exponential, gamma, Weibull |
| T4 | Maximum Likelihood Estimation | H-MAIN-B | MLE for log-normal parameters (μ, σ) |
| T5 | Ljung-Box Test | H-MAIN-C | Autocorrelation significance at lags 1-20 |
| T6 | ACF/PACF Analysis | H-MAIN-C | Partial autocorrelation to identify AR structure |
| T7 | Skewness/Kurtosis | H-MAIN-B | Compare to theoretical values for candidate distributions |
| T8 | Q-Q Plots | H-MAIN-B | Visual and quantitative assessment of distribution fit |

#### 3.3.2 Significance Thresholds

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| p-value for rejection | < 0.01 | Conservative threshold to avoid false positives |
| KS statistic comparison | Ratio > 1.5 | One distribution fits substantially better |
| Regression R² | > 0.90 | Strong linear relationship in decay |
| Slope significance | p < 0.001 | Decay slope is significantly negative |

### 3.4 Falsification Criteria

**The hypothesis is FALSIFIED if ANY of the following occur:**

1. **F1:** Quintile/decile means show non-decreasing trend (T1, T2 regression slope ≥ 0 with p > 0.05)
2. **F2:** Normal distribution fits log-gaps better than log-normal (T3: KS_normal < KS_lognormal)
3. **F3:** Log-gaps are indistinguishable from uniform random (T3: uniform KS p-value > 0.05)
4. **F4:** Autocorrelation is flat at all lags (T5: Ljung-Box p > 0.05 for all lags)
5. **F5:** Skewness and kurtosis are consistent with normal distribution (T7: |skewness| < 0.5, |excess kurtosis| < 1)
6. **F6:** Results at 10⁸ contradict results at 10⁵ (directional inconsistency)

**The hypothesis is SUPPORTED (not proven) if:**

1. All falsification criteria fail to trigger
2. Log-normal or heavy-tailed fit is statistically superior
3. Decay trend is monotonic and statistically significant
4. Results are consistent across scales (10⁵ → 10⁶ → 10⁷ → 10⁸)

---

## 4. Implementation Specification

### 4.1 File Structure

```
experiments/PR-0002_prime_log_gap_falsification/
├── SPEC.md                    # This document
├── FINDINGS.md                # Results and conclusions
├── src/
│   ├── prime_generator.py     # Segmented sieve implementation
│   ├── log_gap_analysis.py    # Core statistical analysis
│   ├── distribution_tests.py  # KS tests and MLE fitting
│   ├── autocorrelation.py     # ACF/PACF and Ljung-Box
│   └── visualization.py       # Q-Q plots, histograms, trend plots
├── data/
│   ├── primes_1e6.npy         # Cached prime arrays
│   ├── primes_1e7.npy
│   ├── primes_1e8.npy
│   └── log_gaps_*.csv         # Computed log-gaps
├── results/
│   ├── quintile_analysis.csv
│   ├── distribution_fits.csv
│   ├── autocorrelation.csv
│   └── figures/
│       ├── log_gap_histogram.png
│       ├── qq_plot_lognormal.png
│       ├── decay_trend.png
│       └── acf_pacf.png
└── tests/
    └── test_prime_generator.py # Validation against known π(x)
```

### 4.2 Dependencies

```
numpy >= 1.21.0
scipy >= 1.7.0
statsmodels >= 0.13.0
matplotlib >= 3.5.0
pandas >= 1.3.0
sympy >= 1.9  # For symbolic verification
```

### 4.3 Computational Requirements

| Scale | Estimated Primes | Memory (primes) | Memory (gaps) | Time (est.) |
|-------|------------------|-----------------|---------------|-------------|
| 10⁶ | 78,498 | ~0.6 MB | ~0.6 MB | < 1 sec |
| 10⁷ | 664,579 | ~5 MB | ~5 MB | ~5 sec |
| 10⁸ | 5,761,455 | ~46 MB | ~46 MB | ~60 sec |
| 10⁹ | 50,847,534 | ~407 MB | ~407 MB | ~15 min |

### 4.4 Algorithm: Segmented Sieve

```python
def segmented_sieve(limit, segment_size=10**6):
    """
    Memory-efficient prime generation using segmented sieve.

    1. Generate small primes up to sqrt(limit) using basic sieve
    2. Process [sqrt(limit), limit] in segments of segment_size
    3. For each segment, mark composites using small primes
    4. Yield primes from each segment
    """
    pass  # Implementation in src/prime_generator.py
```

---

## 5. Analysis Protocol

### 5.1 Phase 1: Validation (Scale 10⁶)

1. Generate primes to 10⁶
2. Verify count matches π(10⁶) = 78,498
3. Compute log-gaps
4. Run full test suite (T1-T8)
5. Compare to preliminary results (10⁵ scale)
6. Document any discrepancies

### 5.2 Phase 2: Extension (Scale 10⁷)

1. Generate primes to 10⁷
2. Verify count matches π(10⁷) = 664,579
3. Compute log-gaps
4. Run full test suite
5. Compare decay rates across scales
6. Check for scale-dependent effects

### 5.3 Phase 3: Full Scale (10⁸)

1. Generate primes to 10⁸
2. Verify count matches π(10⁸) = 5,761,455
3. Compute log-gaps
4. Run full test suite
5. Final statistical analysis
6. Aggregate cross-scale comparison

### 5.4 Phase 4: Robustness Checks

1. **Windowed analysis:** Analyze log-gaps in non-overlapping windows of 10⁵ primes
2. **Bootstrap confidence intervals:** 1000 resamples for key statistics
3. **Sensitivity analysis:** Vary bin counts, significance thresholds
4. **Alternative metrics:** Test ln(1 + gap/p) as alternative to ln(p_{n+1}/p_n)

---

## 6. Expected Outcomes

### 6.1 If Hypothesis is Supported

| Finding | Implication |
|---------|-------------|
| Log-normal fit confirmed at scale | Multiplicative structure in prime gaps is real |
| Decay rate consistent across scales | "Damping coefficient" is a stable invariant |
| Short-range autocorrelation | "Filter memory" model is viable |
| Heavy tails persist | Rare large gaps follow predictable statistics |

### 6.2 If Hypothesis is Falsified

| Finding | Implication |
|---------|-------------|
| Normal distribution fits better | Circuit analogy breaks down; gaps are additive |
| No decay in log-gap means | Logarithmic "voltage" is not the right variable |
| White noise autocorrelation | No memory/filter structure exists |
| Scale-dependent reversals | Preliminary results were artifacts of small sample |

### 6.3 Theoretical Connections

**If supported**, the results suggest:

1. Rational bounds on prime density analogous to β–δ Doppler bounds
2. Transfer function formulation H(z) for prime-counting approximations
3. Poles/zeros corresponding to Riemann zeta structure

**These must be tested separately** and are not part of this experiment's scope.

---

## 7. Constraints and Limitations

### 7.1 What This Experiment Does NOT Test

1. Riemann Hypothesis implications
2. Cryptographic applications (key generation speedup)
3. CRISPR/bioinformatics applications
4. Specific transfer function designs
5. Predictions beyond the computed range

### 7.2 Known Limitations

1. Finite computational range (10⁸ or 10⁹ maximum practical)
2. Cannot distinguish between "true" log-normal and close approximations
3. Autocorrelation may be sensitive to window size choices
4. Heavy tails make variance estimates unstable

### 7.3 Potential Confounds

1. **Numerical precision:** Log of large primes may lose precision
   - Mitigation: Use np.float128 or mpmath for extended precision
2. **Edge effects:** First/last quintiles may behave differently
   - Mitigation: Exclude extreme quintiles in robustness checks
3. **Sieve artifacts:** Computational errors could create spurious patterns
   - Mitigation: Validate against known π(x) at multiple checkpoints

---

## 8. Success Criteria

### 8.1 Experiment Success (Independent of Hypothesis Outcome)

- [ ] All phases (10⁶, 10⁷, 10⁸) complete
- [ ] Prime counts match known values within ±1
- [ ] All statistical tests execute without error
- [ ] Results are reproducible (seed-fixed where applicable)
- [ ] FINDINGS.md documents clear conclusion with evidence

### 8.2 Scientific Rigor Checklist

- [ ] Falsification criteria defined BEFORE running tests
- [ ] No data fabrication or selective reporting
- [ ] Negative results documented equally to positive
- [ ] Code is version-controlled and reproducible
- [ ] Statistical thresholds are conventional (not cherry-picked)

---

## 9. References

### 9.1 Mathematical Foundations

1. Prime Number Theorem: π(x) ~ x/ln(x)
2. Cramér's model for prime gaps
3. Napier's inequality: z/(z+1) < ln(1+z) < z for z > 0
4. Mechanical–electrical analogies (impedance/mobility forms)
5. Z-transform and Dirichlet series correspondence

### 9.2 Prior Work in This Project

1. Gist: "Proof and Analysis of Relativistic Doppler Shift Bounds" (zfifteen/d095ef208231ec991a8cacb7bdc5cc92)
2. Preliminary experiment: experiments/electrical_number_theory_analogy_test/

### 9.3 Statistical Methods

1. Kolmogorov-Smirnov test for distribution fitting
2. Ljung-Box test for autocorrelation
3. Maximum likelihood estimation for distribution parameters
4. Q-Q plots for distribution assessment

---

## 10. Appendix: Preliminary Results Summary

From initial experiment (9,592 primes to 10⁵):

| Metric | Value |
|--------|-------|
| Log-gap range | [0.000020, 0.510826] |
| Log-gap mean | 0.001128 |
| Log-gap std | 0.010868 |
| Q1 mean | 0.004704 |
| Q5 mean | 0.000128 |
| Decay ratio (Q1/Q5) | 36.8 |
| Best distribution fit | Log-normal (KS=0.0438) |
| Skewness | 31.55 |
| Excess kurtosis | 1194.87 |

These results motivate the current experiment but do not predetermine its outcome.