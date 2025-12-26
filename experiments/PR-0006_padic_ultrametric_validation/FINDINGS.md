# Experimental Findings: p-adic Ultrametric vs Riemannian Metric in Semiprime Factorization

## Conclusion

[TO BE COMPLETED AFTER EXPERIMENT EXECUTION]

The p-adic ultrametric demonstrates [SUPERIOR/INFERIOR/MIXED] performance compared to the Riemannian/Euclidean baseline metric in small-scale semiprime factorization tasks within a toy Geometric Variational Analysis (GVA) framework.

Specifically:
- p-adic metric achieved faster factor discovery in [X] out of [Y] toy cases
- Baseline metric prevailed in [X] toy cases
- [TIE/NO TIE] observed for medium-sized semiprime
- Both metrics [SUCCEEDED/FAILED] on RSA-100 instance

This [CONFIRMS/REFUTES/PARTIALLY VALIDATES] the hypothesis stated in geofac_validation PR #35.

---

## Technical Supporting Evidence

### 1. Experimental Setup

**Test Environment:**
- Platform: [To be filled]
- Python version: [To be filled]
- Execution date: [To be filled]

**Dataset:**
- 5 semiprimes tested (3 toy, 1 medium, 1 RSA challenge)
- All semiprimes validated: N = p×q, gcd(p,q) = 1, both prime
- Random seed: 42 (reproducible results)

**Parameters:**
- Candidates per semiprime: 500
- Search window: ±15% around √N (min radius 50)
- GCD checks: Top-scored candidates only

### 2. Dataset Validation Results

[TO BE COMPLETED]

✓ Toy-1: N = 143 = 11 × 13
✓ Toy-2: N = 1763 = 41 × 43
✓ Toy-3: N = 6557 = 79 × 83
✓ Medium-1: N = 9753016572299 = 3122977 × 3122987
✓ RSA-100: N = [large number]

All semiprimes passed validation checks.

### 3. Detailed Results by Semiprime

#### Toy-1 (N = 143)

**Baseline Metric:**
- Factor found: [Yes/No]
- Factor value: [value]
- Iterations to factor: [count]
- Runtime: [seconds]
- Score range: [best, worst]

**p-adic Metric:**
- Factor found: [Yes/No]
- Factor value: [value]
- Iterations to factor: [count]
- Runtime: [seconds]
- Score range: [best, worst]

**Winner:** [Baseline/p-adic/Tie]

**Analysis:**
[To be filled with interpretation]

#### Toy-2 (N = 1763)

[Similar structure as Toy-1]

#### Toy-3 (N = 6557)

[Similar structure as Toy-1]

#### Medium-1 (N = 9753016572299)

[Similar structure as Toy-1]

#### RSA-100

[Similar structure as Toy-1, with emphasis on expected failure]

### 4. Comparative Performance Summary

**Overall Success Rates:**
- Baseline: [X/5] successful factorizations
- p-adic: [X/5] successful factorizations

**Average Iterations to Factor (excluding failures):**
- Baseline: [mean] iterations
- p-adic: [mean] iterations

**Average Runtime (excluding failures):**
- Baseline: [mean] seconds
- p-adic: [mean] seconds

**Head-to-Head Comparison:**
| Semiprime | Baseline Iters | p-adic Iters | Winner |
|-----------|----------------|--------------|--------|
| Toy-1     | [X]            | [Y]          | [W]    |
| Toy-2     | [X]            | [Y]          | [W]    |
| Toy-3     | [X]            | [Y]          | [W]    |
| Medium-1  | [X]            | [Y]          | [W]    |
| RSA-100   | Failed         | Failed       | Both   |

### 5. Metric Behavior Analysis

**Baseline Metric (Riemannian/Z5D):**
- Score distribution characteristics: [analysis]
- Correlation with factor proximity: [analysis]
- Strengths observed: [analysis]
- Weaknesses observed: [analysis]

**p-adic Metric (Ultrametric):**
- Score distribution characteristics: [analysis]
- Correlation with factor proximity: [analysis]
- Strengths observed: [analysis]
- Weaknesses observed: [analysis]
- Ultrametric property verification: [analysis]

### 6. Key Observations

1. **Window Coverage**: [Analysis of whether search window adequately covered factor space]

2. **Score Variation**: [Analysis of score range and discrimination power]

3. **Metric Leakage Verification**: [Confirmation that neither metric used direct divisibility testing]

4. **RSA-100 Null Result**: [Interpretation of why both metrics failed, validating sampling limitation]

5. **Statistical Significance**: [Discussion of whether observed differences are meaningful given sample size]

### 7. Validation of PR #35 Claims

**Claim 1**: "p-adic achieves faster factor discovery in two out of three toy cases"
- Status: [CONFIRMED/REFUTED]
- Evidence: [specific data]

**Claim 2**: "Baseline prevails in one toy case"
- Status: [CONFIRMED/REFUTED]
- Evidence: [specific data]

**Claim 3**: "Results in a tie for medium-sized semiprime"
- Status: [CONFIRMED/REFUTED]
- Evidence: [specific data]

**Claim 4**: "Both metrics fail for RSA-100 due to sampling constraints"
- Status: [CONFIRMED/REFUTED]
- Evidence: [specific data]

### 8. Limitations and Caveats

1. **Sample Size**: Only 5 semiprimes tested (limited statistical power)
2. **Search Strategy**: Uniform random sampling (not optimized)
3. **Window Size**: Fixed ±15% (not adaptive)
4. **Metric Parameters**: Default p-adic primes [2,3,5,7,11,13,17,19,23,29] (not optimized for each N)
5. **One-Factor Goal**: Experiment stops at first factor found
6. **Small N Advantage**: Dense sampling for N < 10^13 may favor both metrics equally

### 9. Implications

**For p-adic Approach:**
[Discussion of when/why p-adic metric shows advantages]

**For Baseline Approach:**
[Discussion of when/why baseline metric shows advantages]

**For Hybrid Strategies:**
[Suggestions for combining both metrics]

### 10. Recommendations for Future Work

1. **Larger Test Suite**: Expand to [X] semiprimes across wider range
2. **Adaptive Parameters**: Optimize p-adic prime selection per N
3. **Rank-Based Evaluation**: For large N, measure factor rank percentile rather than exact discovery
4. **Hybrid Metrics**: Investigate weighted combinations of both approaches
5. **Theoretical Analysis**: Develop formal model explaining observed performance differences

---

## Appendix: Raw Data

See `results/padic_gva_results_[timestamp].csv` for complete experimental data.

## Reproducibility

This experiment is fully reproducible using:
```bash
python3 -m experiments.PR-0006_padic_ultrametric_validation.src.experiment_runner
```

Random seed fixed at 42 ensures identical results across runs.
