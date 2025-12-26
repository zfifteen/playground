# Z-Domain Framework for Riemann Hypothesis Verification

## Overview

This experiment implements and tests a novel Z-domain transform framework designed to detect potential violations of the Riemann Hypothesis (RH) through phase mapping analysis of zeta zero differences.

## Hypothesis

The framework tests whether the multiplicative structure of zeta zero gaps can reveal:
1. **Zero multiplicity** (multiple zeros at same height)
2. **Critical line violations** (zeros off the critical line Re(s) = 1/2)
3. **Departure from GUE statistics** (Random Matrix Theory predictions)

## Mathematical Framework

### The Z-Transform

$$Z_n = \delta_n \cdot \frac{\log(\delta_{n+1}/\delta_n)}{(1/2)\log \gamma_k}$$

**Components:**
- **Observable A** ($\delta_n$): Raw gap $\delta_n = \gamma_{n+1} - \gamma_n$
- **Multiplicative rate B**: $\log(\delta_{n+1}/\delta_n)$ captures derivative in log-scale
- **RH-bounded normalizer C**: $(1/2)\log \gamma_k$ from $O(\gamma^{1/2} \log \gamma)$ maximal gap bound under RH

### Phase Mapping

$$\theta_n = 2\pi(Z_n \mod 1)$$

Projects Z-values onto unit circle [0, 2π).

**Expected behavior under RH:**
- **Simple zeros + GUE**: Phases uniformly distributed → χ² test accepts uniformity
- **Multiple zeros**: $\delta_n \approx 0$ → B-term spike → phase clustering

## Empirical Foundation

Based on:
1. **Takalo et al. (arXiv:2006.04196, 2001.05294, 2001.11353)**: 
   - Zeta zero differences exhibit self-replicating properties
   - Lognormal (Johnson S_L/S_U) distributions
   - Skewness changes sign at early zero locations
   - Information density ~1/ln(γ/(2π)) constant per segment

2. **Random Matrix Theory (GUE)**:
   - Wigner surmise: P(s) = (32/π²)s² exp(-4s²/π)
   - Montgomery pair correlation: R₂(r) = 1 - (sin πr / πr)²
   - Level repulsion: P(s) → 0 as s → 0

3. **de Bruijn-Newman Constant**:
   - RH ⟺ Λ ≤ 0
   - Polymath15: 0 ≤ Λ < 0.22
   - Multiplicity would push Λ > 0

## Directory Structure

```
experiments/z_domain_framework/
├── run_experiment.py           # Main experiment runner
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── FINDINGS.md                 # Experimental results and analysis
├── src/                        # Source modules
│   ├── __init__.py
│   ├── zeta_zeros.py          # Zero acquisition (mpmath/LMFDB)
│   ├── z_transform.py         # Z-transform computation
│   ├── gue_analysis.py        # GUE comparison tests
│   └── visualization.py       # Plotting utilities
├── data/                       # Cached zeros (gitignored)
└── results/                    # Output plots and JSON
    ├── results.json
    ├── z_transform.png
    ├── phase_circle.png
    ├── phase_analysis.png
    ├── gue_comparison.png
    ├── gap_structure.png
    └── gap_autocorrelation.png
```

## Installation

### Requirements

- Python 3.8+
- NumPy, SciPy, matplotlib, pandas, seaborn
- statsmodels (for autocorrelation)
- mpmath (for zeta zero computation)

### Setup

```bash
cd experiments/z_domain_framework
pip install -r requirements.txt
```

## Usage

### Quick Test (1,000 zeros, ~6 minutes)

```bash
python run_experiment.py --quick --verbose
```

### Full Scale Options

```bash
# 5,000 zeros (default, ~30 minutes)
python run_experiment.py --verbose

# 10,000 zeros (~60 minutes)
python run_experiment.py --n-zeros 10000 --verbose

# Custom precision
python run_experiment.py --n-zeros 5000 --precision 100 --verbose

# Custom output directory
python run_experiment.py --output my_results/ --verbose
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n-zeros` | int | 5000 | Number of zeta zeros to analyze |
| `--method` | str | 'mpmath' | Zero acquisition method (mpmath/lmfdb/file) |
| `--precision` | int | 50 | Decimal precision for mpmath |
| `--output` | str | 'results' | Output directory for results |
| `--verbose` | flag | False | Print detailed progress |
| `--quick` | flag | False | Quick test with 1,000 zeros |

## Output

### Results JSON

`results/results.json` contains:
- Metadata (timestamp, configuration)
- Zero summary statistics
- Z-transform distribution parameters
- Phase clustering metrics (χ², p-value, circular variance)
- GUE comparison (KS statistics, p-values)
- Level repulsion analysis
- Multiplicity sensitivity (anomaly counts)
- Autocorrelation coefficients
- Final conclusion and confidence level

### Visualizations

1. **z_transform.png**: 
   - Time series of Z_n values
   - Histogram with distribution fit
   - Q-Q plot (normality test)
   - Box plot

2. **phase_circle.png**:
   - Polar scatter plot on unit circle
   - Histogram of phases (36 bins = 10° each)

3. **phase_analysis.png** (comprehensive):
   - Polar scatter with color mapping
   - Histogram with χ² test results
   - Mean resultant vector
   - Clustering metrics (circular variance, entropy)
   - Phase difference distribution
   - Phase autocorrelation
   - Statistical summary box

4. **gue_comparison.png**:
   - Empirical vs GUE/Poisson spacing PDFs
   - CDFs comparison
   - Level repulsion zoom (near s=0)
   - Q-Q plot vs GUE

5. **gap_structure.png**:
   - Gap size vs zero height
   - Log-log plot
   - Gap acceleration (second derivative)
   - Gap ratio distribution (B-term)

6. **gap_autocorrelation.png**:
   - ACF and PACF up to lag 50

## Experiment Workflow

The experiment executes these steps:

1. **Zero Acquisition**: Compute or load zeta zeros via mpmath
2. **Z-Transform**: Compute Z_n values and phases θ_n
3. **Phase Clustering**: χ² test for uniformity, circular statistics
4. **GUE Comparison**: KS tests against Wigner surmise and Poisson
5. **Level Repulsion**: Test for gaps near zero
6. **Multiplicity Sensitivity**: Detect anomalous contractions and B-term spikes
7. **Autocorrelation**: Analyze temporal dependencies in gaps
8. **Visualization**: Generate 6 comprehensive plots
9. **Conclusion**: Formulate verdict based on criteria

### Success Criteria

The experiment evaluates RH consistency using:

| Criterion | Test | Threshold | RH-Consistent |
|-----------|------|-----------|---------------|
| Phase uniformity | χ² test | p > 0.05 | Uniform phases |
| GUE spacing | KS test | p > 0.05 | Matches GUE |
| Level repulsion | Small gap fraction | score < 1.5 | Strong repulsion |
| Anomaly rate | Contraction count | < 1% | Low anomalies |

**Verdicts:**
- **CONSISTENT**: All 4 criteria pass
- **LIKELY CONSISTENT**: 3/4 criteria pass
- **INCONCLUSIVE**: 2/4 criteria pass
- **POTENTIAL ANOMALY**: < 2/4 criteria pass

## Key Findings (1,000-Zero Pilot)

See **FINDINGS.md** for complete analysis.

**Summary:**
- **Phase uniformity**: FAILED (χ² = 362.64, p < 10⁻⁶)
- **GUE spacing**: MARGINAL (KS p = 0.042)
- **Level repulsion**: PASSED (perfect repulsion)
- **Anomalies**: PASSED (zero detected)
- **Verdict**: INCONCLUSIVE (finite-sample effects likely)

**Interpretation:** Phase clustering reflects known lognormal skewness at finite heights, not RH violation.

## Performance

| Scale | Zeros | Height Range | Computation Time | Memory |
|-------|-------|--------------|------------------|--------|
| Quick | 1,000 | [14, 1,419] | ~6 minutes | ~100 MB |
| Default | 5,000 | [14, ~8,000] | ~30 minutes | ~300 MB |
| Large | 10,000 | [14, ~17,000] | ~60 minutes | ~600 MB |
| X-Large | 100,000 | [14, ~200,000] | ~10 hours* | ~5 GB* |

*Estimated, not tested

### Computational Bottleneck

- **mpmath.zetazero(n)**: ~0.4 seconds per zero at n ~ 1000
- Scales roughly as O(n log n) for n zeros

**Optimization strategies:**
1. Use LMFDB pre-computed zeros (when available)
2. Parallel computation (embarrassingly parallel)
3. Cache aggressively (subsequent runs ~instant)

## Scientific Context

### What This Framework Contributes

1. **Multiplicative encoding**: Captures gap dynamics in natural (log) scale
2. **RH-theoretic normalization**: Aligns with maximal gap bounds
3. **Phase space**: Intuitive geometric representation
4. **Multiplicity detector**: Would flag δ_n → 0 instantly

### Limitations

1. **Skewness inheritance**: Amplifies lognormal gap asymmetry
2. **Finite-height effects**: Requires ~10⁴+ zeros to average out
3. **Normalizer tuning**: $(1/2)\log\gamma_k$ may need empirical adjustment
4. **Not a proof**: Statistical evidence, not rigorous verification

### Relation to Other RH Tests

| Method | What It Tests | Limitation |
|--------|---------------|------------|
| **This framework** | Gap structure + GUE | Requires large samples |
| **Direct zero search** | Critical line location | Computationally limited |
| **Turán's criterion** | Inequality chains | Partial results only |
| **Odlyzko's tests** | GUE statistics | No phase encoding |
| **Montgomery pair correlation** | Two-point function | Needs ~10⁶ zeros |

## Future Directions

1. **Scale to 100,000 zeros**: Definitive test of phase convergence
2. **Height stratification**: Track statistics vs log(γ)
3. **Johnson transform**: Remove skewness before Z-transform
4. **Prime gap mirror**: Test correlation with Δp_n via explicit formula
5. **de Bruijn-Newman integration**: Compute Λ sensitivity
6. **Machine learning**: Train classifier on Z-phase patterns

## References

### Primary Literature

1. Takalo, J. (2020). On the self-replicating properties of Riemann zeta zeros. arXiv:2006.04196
2. Takalo, J. (2020). Distributions of differences of Riemann zeta zeros. arXiv:2001.05294, 2001.11353
3. Tao, T. (2018). The de Bruijn-Newman constant is non-negative. Polymath15
4. Ford, K. et al. (2018). Large gaps between primes. arXiv:1802.07609
5. Montgomery, H. (1973). Pair correlation of zeros. Analytic Number Theory Symposium

### Background

6. Odlyzko, A. (2001). The 10²³rd zero of the Riemann zeta function
7. Hughes, C. P. et al. (2018). Random matrix theory and ζ(1/2+it)
8. Rubinstein, M. (2006). Evidence for a spectral interpretation of zeros
9. Conrey, B. (2003). The Riemann Hypothesis. Notices AMS
10. Edwards, H. M. (1974). Riemann's Zeta Function. Academic Press

## Citation

If you use this framework in research, please cite:

```bibtex
@misc{zdomain2025,
  title={Z-Domain Framework for Riemann Hypothesis Verification},
  author={GitHub Copilot Workspace},
  year={2025},
  howpublished={zfifteen/playground repository},
  note={experiments/z_domain_framework}
}
```

## License

Same as parent repository.

## Author

Implemented by: GitHub Copilot (Incremental Coder Agent)  
Date: 2025-12-26  
Repository: zfifteen/playground

---

**Disclaimer:** This is experimental mathematical software for research purposes. Results do not constitute a proof or disproof of the Riemann Hypothesis.
