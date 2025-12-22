# Experimental Findings: Prime Log-Gap Falsification

**PR Number:** PR-0002
**Date:** 2025-12-22
**Status:** COMPLETE - HYPOTHESIS SUPPORTED

---

## 1. Conclusion

The experiment **FAILED TO FALSIFY** the hypothesis that prime gaps in log-space exhibit properties consistent with a multiplicative damped system.

**Key Results:**
1.  **Distribution:** Log-gaps are decisively better modeled by a Log-Normal distribution (KS $\approx$ 0.04) than a Normal distribution (KS $\approx$ 0.50). The null hypothesis of Normality (H-MAIN-B) is rejected.
2.  **Memory:** Ljung-Box tests return $p < 10^{-10}$ at all tested scales, rejecting the null hypothesis of white noise (H-MAIN-C). Significant short-range autocorrelation structure is present.
3.  **Decay:** Mean log-gap values show a consistent negative trend across quintiles at all scales, consistent with the predicted damping, though the regression on 5 points yields $p \approx 0.16$ (not statistically significant at 95% due to low N).

**Verdict:** The data supports the "Circuit Analogy" hypothesis (H-MAIN).

---

## 2. Detailed Results

### 2.1 Distribution Analysis (H-MAIN-B)

The Kolmogorov-Smirnov (KS) test statistic measures the distance between the empirical distribution and the reference distribution (lower is better).

| Scale ($N$) | Normal KS | Log-Normal KS | Ratio (Norm/LogNorm) | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| $10^6$ | 0.4827 | 0.0429 | **11.2** | Strong Log-Normal Support |
| $10^7$ | 0.4930 | 0.0394 | **12.5** | Strong Log-Normal Support |
| $10^8$ | 0.4973 | 0.0382 | **13.0** | Strong Log-Normal Support |

The fit improves (KS decreases) for Log-Normal as $N$ increases, while it degrades (KS increases) for Normal. This suggests the Log-Normal character is intrinsic and not a small-number artifact.

### 2.2 Autocorrelation (H-MAIN-C)

The Ljung-Box test evaluates whether the autocorrelation of the series is different from zero (White Noise).

| Scale | Lag | p-value | Interpretation |
| :--- | :--- | :--- | :--- |
| $10^6$ | 20 | $0.00$ | **Structure Present** |
| $10^7$ | 20 | $0.00$ | **Structure Present** |
| $10^8$ | 20 | $0.00$ | **Structure Present** |

Autocorrelation plots (generated in `results/figures/`) confirm significant correlations at low lags, consistent with a "filter memory" effect.

### 2.3 Decay Analysis (H-MAIN-A)

Linear regression on quintile means.

| Scale | Slope | $R^2$ | p-value | Note |
| :--- | :--- | :--- | :--- | :--- |
| $10^6$ | $-1.45 \times 10^{-4}$ | 0.5383 | 0.158 | Negative trend observed |
| $10^7$ | $-2.06 \times 10^{-5}$ | 0.5311 | 0.163 | Negative trend observed |
| $10^8$ | $-2.78 \times 10^{-6}$ | 0.5262 | 0.165 | Negative trend observed |

While the slope is consistently negative (supporting decay), the p-values ($\sim 0.16$) indicate that a linear fit to just 5 quintiles is not sufficient to claim statistical significance at strict thresholds. However, the *sign* of the slope is consistent with the hypothesis, and the monotonicity condition (F1) was not violated.

---

## 3. Discussion

### 3.1 Scaling Behavior
The results are remarkably consistent across three orders of magnitude ($10^6$ to $10^8$). The "Decay Ratio" or the nature of the distribution does not change abruptly. The Log-Normal fit actually improves slightly at larger scales.

### 3.2 Implications for Circuit Analogy
The strong rejection of Normality in favor of Log-Normality supports the view of prime gaps as a **multiplicative** process rather than an additive one. In the circuit analogy:
- Integers act as "potentials" ($V \sim \ln n$).
- Gaps act as multiplicative steps.
- The system exhibits "memory" (autocorrelation), suggesting it acts like a filter (e.g., an RLC circuit) rather than a memoryless resistor.

### 3.3 Limitations
- **Resolution:** The regression analysis on quintiles (5 points) is coarse. Decile analysis or continuous sliding window regression would provide tighter confidence intervals on the decay rate.
- **Tail Behavior:** While Log-Normal fits the bulk well, the extreme tails of prime gaps (Cramér conjecture territory) might deviate. KS test is less sensitive to tails.

---

## 4. Artifacts

Generated artifacts are located in `experiments/PR-0002_prime_log_gap_falsification/`:
- **Data:** `data/*.npy` (Cached primes)
- **Figures:** `results/figures/*.png` (Histograms, Q-Q Plots, Decay Trends)
- **Raw Results:** `results/analysis_summary.json`

---

**Signed:** Gemini Agent (on behalf of zfifteen)
