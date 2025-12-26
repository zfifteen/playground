# p-adic Ultrametric vs Riemannian Metric Validation Experiment

## Executive Summary

This experiment validates the hypothesis from [geofac_validation PR #35](https://github.com/zfifteen/geofac_validation/pull/35) that the p-adic ultrametric demonstrates superior performance over the Riemannian/Euclidean baseline metric in certain small-scale semiprime factorization tasks within a toy Geometric Variational Analysis (GVA) framework.

## Purpose

To definitively prove or falsify the claim that:
> The p-adic ultrametric demonstrates superior performance over the Riemannian/Euclidean baseline metric in certain small-scale semiprime factorization tasks within a toy GVA experiment, achieving faster factor discovery in two out of three toy cases, while the baseline prevails in one toy case, results in a tie for a medium-sized semiprime, and both metrics fail for the larger RSA-100 instance due to inherent probabilistic sampling constraints.

## Experimental Design

### Test Cases

Five semiprimes of varying scale:
1. **Toy-1**: N = 143 = 11 × 13 (minimal test case)
2. **Toy-2**: N = 1763 = 41 × 43 (small twin-prime product)
3. **Toy-3**: N = 6557 = 79 × 83 (small twin-prime product)
4. **Medium-1**: N = 9753016572299 = 3122977 × 3122987 (~22-bit prime factors)
5. **RSA-100**: N = 1522605...6139 (actual RSA-100 challenge)

### Methodology

For each semiprime N = p × q:

1. **Candidate Generation**: Sample 500 candidates uniformly in a ±15% window around √N (minimum radius 50 for small N)
2. **Dual Metric Scoring**:
   - **Baseline**: Z5D geometric resonance based on Prime Number Theorem predictions (Riemannian-style)
   - **p-adic**: Ultrametric distance based on p-adic valuations and norms
3. **Factor Discovery**: Sort candidates by score, perform GCD checks to find factors
4. **Metrics Recorded**:
   - Factor found (yes/no)
   - Iterations to first factor
   - Runtime
   - Score statistics

### Critical Design Constraints

- **No Metric Leakage**: Neither metric computes gcd(candidate, N) during scoring
- **Dataset Validation**: All semiprimes validated at runtime (N = p×q, coprimality, primality)
- **Reproducibility**: Fixed random seed (42) for consistent results

## Implementation Structure

```
experiments/PR-0006_padic_ultrametric_validation/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── metric_baseline.py       # Baseline Riemannian/Z5D metric
│   ├── metric_padic.py          # p-adic ultrametric implementation
│   └── experiment_runner.py     # Main experiment orchestrator
├── results/
│   └── *.csv                    # Output CSV files (timestamped)
├── tests/
│   └── test_*.py                # Unit tests
├── FINDINGS.md                  # Detailed findings (conclusion-first)
├── README.md                    # This file
└── .gitignore                   # Ignore patterns
```

## How to Run

### Prerequisites

Python 3.6+ (standard library only, no external dependencies)

### Execution

From repository root:

```bash
# Run using Python module syntax
python3 -m experiments.PR-0006_padic_ultrametric_validation.src.experiment_runner

# Or run directly
cd experiments/PR-0006_padic_ultrametric_validation/src
python3 experiment_runner.py
```

### Testing Individual Metrics

```bash
# Test baseline metric
python3 experiments/PR-0006_padic_ultrametric_validation/src/metric_baseline.py

# Test p-adic metric
python3 experiments/PR-0006_padic_ultrametric_validation/src/metric_padic.py
```

## Expected Outcomes

Based on PR #35 findings, we expect:

- **Toy-1**: p-adic wins (1 iteration vs 38 for baseline)
- **Toy-2**: p-adic wins (1 iteration vs 67 for baseline)
- **Toy-3**: Baseline wins (86 iterations vs 166 for p-adic)
- **Medium-1**: Tie (2 iterations each)
- **RSA-100**: Both fail (sampling limitation, Pr ≈ 10^-47)

## Key Validation Points

1. **Dataset Integrity**: All semiprimes must satisfy N = p×q with coprime prime factors
2. **Metric Independence**: No direct divisibility testing during scoring
3. **Window Coverage**: For small N, window must ensure score variation
4. **RSA-100 Null Result**: Expected failure validates sampling constraints, not metric quality

## Output Format

Results saved to `results/padic_gva_results_<timestamp>.csv` with fields:
- Semiprime identification (name, N, p, q)
- Metric used (baseline/padic)
- Search parameters (num_candidates, window_pct)
- Results (factor_found, factor_value, iterations_to_factor)
- Performance (runtime_seconds, gcd_checks)
- Score statistics (best_score, worst_score, total_scored)

## Documentation

- **FINDINGS.md**: Detailed experimental findings with conclusion-first format
- **README.md**: This file (setup and methodology)
- **Source code**: Inline documentation with implementation notes

## References

- Original hypothesis: [geofac_validation PR #35](https://github.com/zfifteen/geofac_validation/pull/35)
- p-adic number theory: Gouvêa, "p-adic Numbers: An Introduction"
- Prime Number Theorem: Standard reference for baseline metric

## License

Part of the zfifteen/playground repository experiments collection.
