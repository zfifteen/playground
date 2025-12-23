# Lognormal Pre-Filter Factorization Pipeline

This experiment implements a factorization pipeline for semiprimes \(N = pq\) using a lognormal prime-gap model as a pre-filter in front of standard algorithms (Fermat-style and Pollard Rho).

## Overview

The pipeline biases search efforts near \(\sqrt{N}\) based on empirical lognormal distributions derived from prime gaps up to \(10^9\). Bands are hardcoded for \(\sqrt{N}\) in \(10^5\)–\(10^9\), with approximations for smaller/larger values. This is a research experiment for semiprimes in the fitted regime.

## Usage

### Python API
```python
from src.pipeline import factor_with_lognormal_prefilter
from src.model import ModelStore
from src.config import SearchPolicyConfig

model = ModelStore()
cfg = SearchPolicyConfig()
factor = factor_with_lognormal_prefilter(N, model, cfg)
```

### CLI
```bash
python factor_cli.py <N> [--max-steps INT] [--radius-scale FLOAT] [--direction-mode ALTERNATE|RANDOM] [--seed INT]
```

Example: `python factor_cli.py 77` outputs `7 11`

## Parameters

Bands are hardcoded from empirical fits (results-10-6.json to 10-9.json):
- Band 1: \(10^5 \leq p < 10^6\)
- Band 2: \(10^6 \leq p < 10^7\)
- Band 3: \(10^7 \leq p < 10^8\)
- Band 4: \(10^8 \leq p \leq 10^9\)

## Testing

Run tests: `pytest tests/`

Run functional smoke test: `python run_experiment.py`

Run performance benchmark on fitted bands: `python performance_benchmark.py`