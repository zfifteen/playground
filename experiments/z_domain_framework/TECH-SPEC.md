# Z-Domain Framework: Technical Specification

## Experiment Metadata

- **Experiment ID:** z_domain_framework
- **Date:** 2025-12-26
- **Status:** Completed (pilot study)
- **Scale:** 1,000 zeta zeros analyzed
- **Hypothesis:** Z-domain phase mapping can detect RH violations

## Mathematical Definitions

### Primary Transform

$$Z_n = \delta_n \cdot \frac{\log(\delta_{n+1}/\delta_n)}{(1/2)\log \gamma_k}$$

**Components:**
- $\delta_n = \gamma_{n+1} - \gamma_n$ where $\gamma_k$ are imaginary parts of zeta zeros
- $\log(\delta_{n+1}/\delta_n)$ is the B-term (multiplicative rate)
- $(1/2)\log \gamma_k$ is the C-term (RH-theoretic normalizer)

### Phase Projection

$$\theta_n = 2\pi(Z_n \bmod 1)$$

Maps $Z_n \in \mathbb{R}$ to $\theta_n \in [0, 2\pi)$

## Test Statistics

### 1. Phase Uniformity (Primary Criterion)

**Null Hypothesis:** $\theta_n \sim \text{Uniform}(0, 2\pi)$

**Test:** Chi-square goodness-of-fit
$$\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}$$

where:
- $k = 36$ bins (10° each)
- $O_i$ = observed count in bin $i$
- $E_i = n/k$ = expected count under uniformity

**Decision rule:** Reject $H_0$ if $p < 0.05$

**Result:** $\chi^2 = 362.64$, $p < 10^{-6}$ → **REJECT uniformity**

### 2. GUE Spacing (Secondary Criterion)

**Null Hypothesis:** Normalized gaps follow Wigner surmise
$$P_{\text{GUE}}(s) = \frac{32}{\pi^2} s^2 e^{-4s^2/\pi}$$

**Test:** Kolmogorov-Smirnov
$$D = \sup_x |F_n(x) - F_0(x)|$$

where $F_n$ is empirical CDF, $F_0$ is theoretical CDF

**Decision rule:** Reject $H_0$ if $p < 0.05$

**Result:** $D = 0.044$, $p = 0.042$ → **MARGINAL rejection**

### 3. Level Repulsion (Tertiary Criterion)

**Metric:** Fraction of small gaps
$$f_{\text{small}} = \frac{\#\{s_i : s_i < 0.1\}}{n}$$

**Expected values:**
- GUE: $f \approx 0.001$ (strong repulsion)
- Poisson: $f \approx 0.095$ (no repulsion)

**Repulsion score:** $R = f_{\text{small}} / f_{\text{Poisson}}$

**Decision rule:** Accept RH if $R < 1.5$

**Result:** $f = 0$, $R = 0$ → **STRONG repulsion (RH-consistent)**

### 4. Multiplicity Detection (Anomaly Criterion)

**Metric 1:** Anomalous contractions
$$N_{\text{anomaly}} = \#\{\delta_n : \delta_n < \mu - 3\sigma\}$$

**Metric 2:** B-term spikes
$$N_{\text{spike}} = \#\{|B_n| : |B_n| > \mu + 5\sigma\}$$

where $B_n = \log(\delta_{n+1}/\delta_n)$

**Decision rule:** Flag concern if anomaly rate $> 1\%$

**Result:** $N_{\text{anomaly}} = 0$, $N_{\text{spike}} = 0$ → **NO multiplicities detected**

## Circular Statistics

### Mean Resultant Length

$$R = \left|\frac{1}{n}\sum_{i=1}^n e^{i\theta_i}\right|$$

**Interpretation:**
- $R = 0$: Uniform distribution
- $R = 1$: Concentrated at one angle

**Result:** $R = 0.369$ (moderate clustering)

### Circular Variance

$$V = 1 - R$$

**Result:** $V = 0.631$

### Shannon Entropy

$$H = -\sum_{i=1}^k p_i \log p_i$$

**Normalized:** $H_{\text{norm}} = H / \log k$

**Interpretation:**
- $H_{\text{norm}} = 0$: Completely clustered
- $H_{\text{norm}} = 1$: Perfectly uniform

**Result:** $H_{\text{norm}} = 0.952$ (high entropy, near uniform)

## Implementation Details

### Numerical Precision

- **Zero computation:** mpmath with 50 decimal places
- **Floating point:** IEEE 754 double precision (53-bit mantissa)
- **Precision limit:** $\log(\gamma) \approx 7.26$ at $\gamma = 1419$ → relative error $\sim 10^{-15}$

### Edge Cases Handled

1. **Division by zero:** Filter $\delta_n > 0$ before computing $B_n$
2. **Log of negative:** Filter ratios $> 0$ before log
3. **NaN propagation:** Remove NaNs before statistics
4. **Infinite values:** Check `np.isfinite()` before analysis

### Computational Complexity

| Operation | Complexity | Time (n=1000) |
|-----------|------------|---------------|
| Zero computation | $O(n \log n)$ | ~6 min |
| Z-transform | $O(n)$ | <1 sec |
| Phase mapping | $O(n)$ | <1 sec |
| χ² test | $O(n)$ | <1 sec |
| KS test | $O(n \log n)$ | <1 sec |
| Visualizations | $O(n)$ | ~3 sec |

**Total runtime:** ~6-7 minutes (dominated by zero computation)

## Data Products

### Primary Outputs

1. **results.json** (4.5 KB)
   - Machine-readable summary
   - All test statistics
   - Conclusion and confidence

2. **Visualizations** (6 PNG files, ~3.3 MB total)
   - z_transform.png (548 KB)
   - phase_circle.png (450 KB)
   - phase_analysis.png (693 KB) ⭐ **Key result**
   - gue_comparison.png (541 KB)
   - gap_structure.png (944 KB)
   - gap_autocorrelation.png (122 KB)

3. **Cached zeros** (results/data/)
   - mpmath_zeros_1_1000_p50.npy (8 KB)
   - Enables instant re-run without recomputation

### Documentation

1. **FINDINGS.md** (12 KB)
   - Conclusion-first format
   - Technical evidence
   - Critical assessment

2. **README.md** (11 KB)
   - Installation and usage
   - Mathematical framework
   - Scientific context

3. **This file (TECH-SPEC.md)**
   - Formal definitions
   - Test procedures
   - Implementation notes

## Quality Assurance

### Validation Checks

- ✅ Known zeros match: First 10 zeros match published values
- ✅ Gap positivity: All $\delta_n > 0$ (no multiplicities)
- ✅ Phase range: All $\theta_n \in [0, 2\pi)$
- ✅ Distribution moments: Finite mean, variance, skewness, kurtosis
- ✅ Visualization quality: All 6 plots generated successfully

### Known Limitations

1. **Sample size:** 1,000 zeros insufficient for definitive conclusions
2. **Height range:** $\gamma < 1500$ in transitional regime (skewness not converged)
3. **Normalizer:** $(1/2)\log\gamma_k$ is theoretical; may need empirical tuning
4. **Statistical power:** χ² test highly sensitive to deviations at large $n$

## Reproducibility

### Environment

```
Python 3.12.3
numpy >= 1.21.0
scipy >= 1.7.0
matplotlib >= 3.5.0
mpmath >= 1.2.0
statsmodels >= 0.13.0
```

### Random Seed

Not applicable - computation is deterministic

### Cache Policy

- Zeros cached by $(n_{\text{zeros}}, \text{precision})$ key
- Subsequent runs reuse cached data
- Delete `results/data/` to force recomputation

### Verification Command

```bash
cd experiments/z_domain_framework
python run_experiment.py --quick --verbose
# Should produce identical results to committed files
```

## Future Extensions

### Planned Improvements

1. **Scale to 10,000 zeros:** Test convergence to uniformity
2. **Height stratification:** Analyze $[10^2, 10^3]$, $[10^3, 10^4]$, etc.
3. **Alternative normalizers:** Test $C = 2\pi/\log(\gamma/(2\pi))$
4. **Johnson transform:** Pre-process gaps to remove skewness
5. **LMFDB integration:** Use pre-computed zeros for speed

### Research Questions

1. Does phase distribution converge to uniform at $\gamma > 10^5$?
2. Can we quantify convergence rate vs Takalo's predictions?
3. Is there optimal normalizer that yields uniformity at all heights?
4. How does this compare to Montgomery pair correlation?
5. What is the connection to de Bruijn-Newman constant?

## References

See FINDINGS.md and README.md for complete bibliography.

---

**Document version:** 1.0  
**Last updated:** 2025-12-26  
**Author:** GitHub Copilot (Incremental Coder Agent)
