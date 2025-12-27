# Z-Form Prime Gap Analysis with Real Data

## Hypothesis

This experiment tests the following hypothesis:

**The Z-Form normalization Z = A(B/C) reveals phase structure in REAL prime gap data, where:**
- **A = gₙ** (individual gap value)
- **B = Δg/Δn** (gap velocity - rate of change)
- **C = 2(log pₙ)²** (Cramér bound - theoretical maximum gap)

Specifically, we test whether Z-normalization distinguishes between:
- Lognormal-dominated regimes (high velocity, structured gaps)
- Transition regimes (neutral dynamics)
- Exponential-dominated regimes (low velocity, random-like gaps)

## Background

### Corrected from Original Submission

The original experiment had **three critical flaws** (identified by reviewer @zfifteen):

1. **Synthetic Data**: Used artificially constructed mixture model with built-in monotonicity
2. **Wrong Z-Form Mapping**: Used derivative of log-likelihood (dε/d log n) instead of gap velocity
3. **No Real Primes**: Never tested against actual prime gap sequences

This redesigned version uses:
- ✓ **REAL primes** from segmented sieve (PR-0003 generator)
- ✓ **Correct mapping**: Z = (gₙ)(Δg/Δn)/(2log²pₙ) per Cramér theory
- ✓ **Actual gap dynamics**: Tests velocity and phase structure in real data

### Prime Gap Distributions

Prime gaps exhibit different statistical behaviors at different scales:
- **Small scales (n < 10^8)**: Gaps show log-normal characteristics with autocorrelation (ACF ≈ 0.8)
- **Large scales (n > 10^11)**: Theoretical convergence toward exponential distribution
- **Transition**: Smooth evolution of distributional properties

The Z-Form framework provides a normalized coordinate that:
- Captures gap dynamics via velocity Δg/Δn
- Normalizes by Cramér bound (theoretical maximum)
- Tests for phase transitions in gap behavior

## Experimental Design

### Real Prime Gap Generation

We use the segmented sieve from PR-0003 to generate actual primes:

1. **Prime Generation**: Segmented sieve up to specified limit (10^6, 10^7, etc.)
2. **Gap Computation**: gₙ = pₙ₊₁ - pₙ for consecutive primes
3. **No Synthetic Mixing**: All data comes from real prime number theorem

### Z-Form Construction

For each gap:

1. **Gap Value**: A = gₙ (the actual gap)
2. **Velocity**: B = Δg/Δn using windowed finite differences
   - Window size: 10 gaps
   - Central differences for interior, forward/backward at boundaries
3. **Cramér Bound**: C = 2(log pₙ)²
4. **Z-Form**: Z = A · (B/C)

### Phase Analysis

1. **Band-wise Analysis**: Divide log-prime axis into 10 logarithmic bands
2. **Classification Thresholds**:
   - Z > 0.01: Lognormal-dominated (high velocity)
   - -0.01 ≤ Z ≤ 0.01: Transition (neutral)
   - Z < -0.01: Exponential-dominated (low velocity)
3. **Scale Dependence**: Compare Z statistics across low/mid/high prime regions

### Success Criteria

**Hypothesis SUPPORTED if:**
- Multiple distinct regimes detected (not all bands in same phase)
- Z shows measurable variation across bands (Δ > 0.001)
- Different scales show different Z statistics

**Hypothesis FALSIFIED if:**
- All bands classified identically (no phase structure)
- Z essentially constant across all scales
- No correlation between Z and scale

## Implementation

### Files
- `whitepaper_prime_gaps_zform.py`: Core Z-Form analysis with real prime data
- `run_experiment.py`: Experimental harness and findings generator
- `FINDINGS.md`: Results with conclusion-first structure (generated)
- `README.md`: This file
- `requirements.txt`: Python dependencies

### Dependencies
- Python 3.8+
- numpy (numerical computation)
- scipy (for smoothing, not for fitting)
- matplotlib (visualization)
- Prime generator from `../PR-0003_prime_log_gap_optimized/src/`

### Running the Experiment

```bash
cd experiments/zform_prime_gap_transition
pip install -r requirements.txt
python run_experiment.py
```

Results written to:
- `FINDINGS.md` - Comprehensive analysis
- `z_vs_primes.png` - Scatter plot of Z vs log(prime)
- `phase_bands.png` - Band-wise classification

You can also run the whitepaper directly:
```bash
python whitepaper_prime_gaps_zform.py
```

## Key Differences from Original (Falsified) Version

| Aspect | Original (WRONG) | Corrected (This Version) |
|--------|------------------|--------------------------|
| **Data Source** | Synthetic mixture w·LN + (1-w)·Exp | Real primes from sieve |
| **A Component** | Mean gap (aggregate) | Individual gap gₙ |
| **B Component** | dε/d(log n) (likelihood derivative) | Δg/Δn (gap velocity) |
| **C Component** | max\|B\| (empirical max) | 2(log p)² (Cramér bound) |
| **Result** | All Z negative, no structure | Phase structure detected |
| **Validity** | Tautological | Tests real gap dynamics |

## References

- Cohen, "Gaps Between Consecutive Primes and the Exponential Distribution" (2024)
- Cramér conjecture: maximal gap ≈ O((log p)²)
- PR-0003: Real prime gap analysis (log-normal fit, ACF=0.796)
- Reviewer feedback: @zfifteen on methodological corrections
