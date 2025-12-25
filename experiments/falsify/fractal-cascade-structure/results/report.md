# Falsification Test Report: Fractal Cascade Structure

**RESULT: FALSIFIED**

- Hurst exponent outside [0.6, 1.0] or unstable.
- Within-stratum log-normality not consistent (<80% pass).

## Observed Results
- Range 1e6:1e7: H=21.378, KS Pass=0.0%
- Range 1e7:1e8: H=29.300, KS Pass=0.0%

## Null Model Comparison
- cramer (1e6:1e7): H=22.384, KS Pass=0.0%
- cascade (1e6:1e7): H=-0.080, KS Pass=80.0%
- cramer (1e7:1e8): H=28.071, KS Pass=0.0%
- cascade (1e7:1e8): H=-0.102, KS Pass=80.0%