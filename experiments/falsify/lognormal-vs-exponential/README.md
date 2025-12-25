# Lognormal vs Exponential Distribution Falsification Test

This experiment tests the claim that prime gaps in log-space are better modeled by lognormal distributions than by exponential distributions.

## Overview

The experiment:
1. Generates primes in specified disjoint ranges (up to 10^18 using primesieve for efficiency).
2. Bands the primes by magnitude (log-spaced).
3. Extracts consecutive gaps within each band.
4. Fits both Exponential and Lognormal distributions to a training subset (70%).
5. Evaluates the fits on a held-out test subset (30%) using BIC and Log-Likelihood.
6. Determines a "winner" for each band based on $\Delta BIC \ge 10$.

## Running the Experiment

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

For large prime ranges (e.g., up to 10^18), install primesieve:

```bash
brew install primesieve  # macOS with Homebrew
# Or download from https://primesieve.org/
```

### Execution

Run the experiment with default settings:

```bash
python run_experiment.py
```

Customize ranges and output:

```bash
python run_experiment.py --ranges "1e7:2e7,2e7:3e7" --output results_custom/ --seed 123
```

For large ranges, use primesieve:

```bash
python run_experiment.py --ranges "1e16:1e17" --prime-source primesieve --output results_large/
```

Note: Use --prime-source primesieve for ranges > 10^10 to leverage optimized prime generation. Default is segmented sieve for smaller ranges.

### Output

The script produces:
- `results.json`: Detailed metrics for every band.
- `results.csv`: Flattened summary table.
- `report.md`: Human-readable summary and falsification verdict.
- `fit_*.png`: Diagnostic plots for the first band of each range.

## Falsification Criteria

The claim "Lognormal > Exponential" is considered **falsified** if:
- Lognormal fails to beat Exponential (by $\Delta BIC \ge 10$) in $\ge 50\%$ of bands across at least two independent ranges.
