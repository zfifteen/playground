# Lognormal vs Exponential Distribution Falsification Test

This experiment tests the claim that prime gaps in log-space are better modeled by lognormal distributions than by exponential distributions.

## Overview

The experiment:
1. Generates primes in specified disjoint ranges.
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

### Execution

Run the experiment with default settings:

```bash
python run_experiment.py
```

Customize ranges and output:

```bash
python run_experiment.py --ranges "1e7:2e7,2e7:3e7" --output results_custom/ --seed 123
```

### Output

The script produces:
- `results.json`: Detailed metrics for every band.
- `results.csv`: Flattened summary table.
- `report.md`: Human-readable summary and falsification verdict.
- `fit_*.png`: Diagnostic plots for the first band of each range.

## Falsification Criteria

The claim "Lognormal > Exponential" is considered **falsified** if:
- Lognormal fails to beat Exponential (by $\Delta BIC \ge 10$) in $\ge 50\%$ of bands across at least two independent ranges.
