# PR-0005: Log-Gap Mean Curve Fitting Results

## Executive Summary

This experiment successfully fitted four different deterministic models to the relationship between log-prime values and mean log-gaps across five different scales (10^5 to 10^9).

## Best Performing Model

**Winner: `loggap_mean_curve_loglog`** - Linear model in log(log(p))

Formula: `mean_loggap = a + b * log(logp)`

This model consistently achieved:
- **Lowest MSE** across all scales
- **Highest R²** values (0.647 to 0.769)
- Simple 2-parameter form

## Detailed Results by Scale

### 10^5 (100,000)
```
loggap_mean_curve_loglog     mse=2.325e-03  r2=0.7687
loggap_mean_curve_power_normed mse=2.680e-03  r2=0.7335
loggap_mean_curve_linear     mse=4.839e-03  r2=0.5187
loggap_mean_curve_normed     mse=4.839e-03  r2=0.5187
```

### 10^6 (1,000,000)
```
loggap_mean_curve_loglog     mse=2.671e-03  r2=0.7243
loggap_mean_curve_power_normed mse=2.936e-03  r2=0.6969
loggap_mean_curve_linear     mse=5.380e-03  r2=0.4446
loggap_mean_curve_normed     mse=5.380e-03  r2=0.4446
```

### 10^7 (10,000,000)
```
loggap_mean_curve_loglog     mse=2.804e-03  r2=0.7026
loggap_mean_curve_power_normed mse=2.959e-03  r2=0.6860
loggap_mean_curve_linear     mse=5.567e-03  r2=0.4094
loggap_mean_curve_normed     mse=5.567e-03  r2=0.4094
```

### 10^8 (100,000,000)
```
loggap_mean_curve_loglog     mse=3.030e-03  r2=0.6703
loggap_mean_curve_power_normed mse=3.135e-03  r2=0.6589
loggap_mean_curve_linear     mse=5.836e-03  r2=0.3649
loggap_mean_curve_normed     mse=5.836e-03  r2=0.3649
```

### 10^9 (1,000,000,000)
```
loggap_mean_curve_loglog     mse=3.207e-03  r2=0.6474
loggap_mean_curve_power_normed mse=3.244e-03  r2=0.6433
loggap_mean_curve_linear     mse=5.963e-03  r2=0.3443
loggap_mean_curve_normed     mse=5.963e-03  r2=0.3443
```

## Key Insights

1. **Non-linear Relationship**: Simple linear models (linear, normed) perform significantly worse than log-log and power-law models, confirming the relationship is non-linear.

2. **Log-Log Dominance**: The log-log model suggests the relationship is approximately:
   ```
   mean_loggap ≈ a + b * log(log(p))
   ```
   This is theoretically interesting as it relates to the Prime Number Theorem.

3. **Scale Dependency**: As scale increases from 10^5 to 10^9:
   - MSE increases slightly (2.3e-3 → 3.2e-3)
   - R² decreases slightly (0.77 → 0.65)
   - This suggests increased noise or complexity at larger scales

4. **Power-Law Alternative**: The power-law normalized model performs nearly as well as log-log, providing flexibility with a third parameter (exponent c ≈ 0.2-0.3).

## Implications

The strong performance of the log-log model provides a simple, interpretable formula for predicting mean log-gaps from log-primes. This could be useful for:
- Prime generation algorithms
- Theoretical analysis of prime distributions
- Validation of other prime-related hypotheses

## Next Steps

Potential follow-up investigations:
1. Investigate why R² decreases at larger scales
2. Explore theoretical connections between log-log relationship and PNT
3. Test models on even larger scales (10^10+)
4. Incorporate additional parameters (e.g., bin variance)
