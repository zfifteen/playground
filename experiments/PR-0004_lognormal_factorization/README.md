# Lognormal Pre-Filter Factorization Pipeline

This experiment implements a factorization pipeline for semiprimes \(N = pq\) using a lognormal prime-gap model as a pre-filter in front of standard algorithms (Fermat-style and Pollard Rho).

## Overview

The pipeline biases search efforts near \(\sqrt{N}\) based on empirical lognormal gap distributions derived from prime gaps up to \(10^9\).

## Structure

- `src/`: Core implementation
- `tests/`: Unit and integration tests
- `data/`: Any generated data or benchmarks
- `run_experiment.py`: Script to run benchmarks

## Usage

```python
from src.pipeline import factor_with_lognormal_prefilter
from src.model import ModelStore
from src.config import SearchPolicyConfig

model = ModelStore()
cfg = SearchPolicyConfig()
factor = factor_with_lognormal_prefilter(N, model, cfg)
```

## Parameters

Bands are hardcoded from empirical fits (results-10-6.json to 10-9.json):
- Band 1: \(10^5 \leq p < 10^6\)
- Band 2: \(10^6 \leq p < 10^7\)
- Band 3: \(10^7 \leq p < 10^8\)
- Band 4: \(10^8 \leq p \leq 10^9\)

## Testing

Run tests: `pytest tests/`

Run benchmarks: `python run_experiment.py`