# Z-Form Prime Gap Distribution Transition Experiment

## Hypothesis

This experiment tests the following hypothesis:

**Prime gap distributions demonstrate a smooth, scale-dependent transition from lognormal behavior at smaller scales to exponential behavior at larger scales, and this transition can be modeled using a Unified Z-Form framework Z = A(B/C) where the Z-value drives adaptive sieve policies.**

Specifically, we test:

1. **Smooth Transition**: The per-gap log-likelihood advantage ε(n) of lognormal vs. exponential distributions decreases monotonically with log n
2. **Z-Form Structure**: Z(n) = A(B/C) where:
   - A = mean gap length at scale n
   - B = dε/d(log n) (local derivative of distributional advantage)
   - C = max |B| across all scales (invariant bound)
3. **Adaptive Policy**: Z can drive meaningful sieve policy adjustments across different regimes

## Background

Prime gap distributions have been observed to follow different statistical patterns at different scales:
- At smaller scales (n < 10^10), gaps exhibit lognormal-like behavior with finite-structure multiplicativity
- At larger scales (n > 10^11), gaps approach an exponential distribution with mean ~ log n
- The transition between these regimes is smooth rather than sharp

The Z-Form framework provides a unified coordinate system that:
- Captures the phase-like motion from lognormal to exponential regimes
- Enables adaptive algorithmic choices based on local distributional characteristics
- Provides invariant bounds (C) that normalize the transition

## Experimental Design

### Synthetic Data Generation

Since actual prime gap computation at extreme scales (10^14) is computationally prohibitive, we generate synthetic "prime-gap-like" data that mirrors empirically observed behavior:

1. **Scale Range**: 30 logarithmically-spaced scales from 10^6 to 10^14
2. **Mixing Model**: At each scale n, generate 50,000 gaps as a mixture:
   - Lognormal component: LN(m, s²) with mean ~ log n
   - Exponential component: Exp(1/log n)
   - Mixing weight w(n) = smooth logistic transition from 1 (small n) to 0 (large n)
3. **Transition Region**: Centered around 10^10 - 10^11

### Methodology

1. **Generate Synthetic Gaps**: Create mixed lognormal-exponential samples across 30 scales
2. **Band-wise Fitting**: For each scale:
   - Fit lognormal parameters via MLE
   - Fit exponential parameter via MLE
   - Compute per-gap log-likelihood advantage ε(n) = (L_LN - L_EXP) / N
3. **Derivative Estimation**: Compute B = dε/d(log n) using smoothed finite differences
4. **Z-Form Construction**: Calculate Z(n) = A(B/C) for each band
5. **Adaptive Policy**: Map Z values to sieve policy parameters (window sizes, limits)

### Success Criteria

**Hypothesis SUPPORTED if:**
- ε(n) decreases smoothly and monotonically with log n
- B approaches 0 at large scales (convergence to exponential fixed point)
- Z transitions from positive (lognormal-dominated) through 0 (neutral) to negative (exponential-dominated)
- Adaptive policy shows meaningful regime-specific parameter adjustments

**Hypothesis FALSIFIED if:**
- ε(n) shows non-monotonic behavior or discontinuities
- B remains large at large scales (no convergence)
- Z does not show clear phase structure
- Policy adjustments are not meaningfully correlated with distributional regimes

## Implementation

### Files
- `whitepaper_prime_gaps_zform.py`: Main executable whitepaper script
- `run_experiment.py`: Experimental harness that runs the whitepaper and captures results
- `FINDINGS.md`: Results and analysis (created after execution)
- `README.md`: This file
- `requirements.txt`: Python dependencies

### Dependencies
- Python 3.8+
- numpy (numerical computation)
- scipy (statistical distributions and fitting)
- matplotlib (visualization)

### Running the Experiment

```bash
cd experiments/zform_prime_gap_transition
pip install -r requirements.txt
python run_experiment.py
```

Results will be written to `FINDINGS.md` and plots will be saved as PNG files.

## References

- Cohen, "Gaps Between Consecutive Primes and the Exponential Distribution" (2024)
- Prime gap statistics and Cramér–Shanks-type heuristics
- zfifteen unified-framework wiki, PREDICTIONS_01: log-band protocol
