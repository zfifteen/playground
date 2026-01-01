# Experiment PR-974: Executive Summary

## Hypothesis Tested

**Claim**: An adaptive stride ring search algorithm successfully factorizes 127-bit semiprimes in approximately 30 seconds by integrating τ functions with golden ratio phase alignment, modular resonance, Richardson extrapolation, and GVA filtering.

## Result: FALSIFIED

The algorithm was implemented according to the problem statement specifications and tested on the claimed 127-bit semiprime. **The algorithm failed to find the factors.**

## Quick Facts

| Aspect | Claimed | Observed | Status |
|--------|---------|----------|--------|
| **Factorization** | Success | Failed | ❌ |
| **Execution Time** | ~30 seconds | 29.24 seconds | ⚠️ Irrelevant |
| **GVA Ranking** | Factor elevated to rank 1 | Factor never in candidate pool | ❌ |
| **Algorithm Design** | Mathematically sound | Fundamental coverage flaw | ❌ |

## Why It Failed

The algorithm has a **fundamental search space coverage problem**:

1. **True factors are far from √N**: 10.4% and 11.6% away
2. **Algorithm searched only 3.3% from √N**: Not far enough
3. **Required 97 rings, generated only 73**: Insufficient expansion
4. **Golden ratio expansion too slow**: Cannot reach unbalanced factors

### Visual Evidence

```
Search Range:  [11.34e18 ←→ 12.11e18]  (3.3% from √N)
Factor p:      10.51e18                 ← OUTSIDE (below)
Factor q:      13.09e18                 ← OUTSIDE (above)
```

Shortfall to reach p: **831 quintillion** units

## What Was Implemented

All claimed components were implemented faithfully:

- ✅ **TauFunction**: Golden ratio phase alignment using φ ≈ 1.618
- ✅ **ModularResonance**: Testing across multiple moduli including φ-based values
- ✅ **RichardsonExtrapolator**: 4th-order derivative estimation
- ✅ **GVAFilter**: 708-digit precision geodesic deviation scoring
- ✅ **AdaptiveStrideRingSearch**: φ-based ring position generation

## Files in This Experiment

1. **README.md** - Experiment overview and setup
2. **adaptive_stride_factorizer.py** - Full algorithm implementation (~450 lines)
3. **FINDINGS.md** - Comprehensive technical analysis (detailed)
4. **analyze_search_space.py** - Coverage analysis tool
5. **comparison_test.py** - Comparison with standard methods
6. **SUMMARY.md** - This file

## Running the Test

```bash
cd experiments/PR-974_adaptive_stride_ring_search

# Run main test
python3 adaptive_stride_factorizer.py

# Analyze search space coverage
python3 analyze_search_space.py

# Compare with standard methods (takes 2+ minutes)
python3 comparison_test.py
```

## Key Takeaways

1. **The hypothesis is false**: The algorithm does not work as claimed
2. **Sophisticated terminology ≠ effective algorithm**: Golden ratio, τ functions, and Richardson extrapolation sound impressive but don't solve the factorization problem
3. **Root cause**: Fundamental design flaw in search space coverage, not a parameter tuning issue
4. **No advantage over classical methods**: Even simple trial division would eventually succeed where this algorithm fails completely

## Scientific Value

This experiment demonstrates:

- ✅ **Proper hypothesis testing**: Implemented, tested, falsified
- ✅ **Reproducibility**: All code available, deterministic results
- ✅ **Root cause analysis**: Identified why algorithm fails
- ✅ **Comparison baseline**: Contrasted with standard methods
- ✅ **Clear documentation**: Findings lead with conclusion, supported by evidence

## Conclusion

The adaptive stride ring search algorithm, as described and implemented, **does not work**. The claim of successfully factorizing 127-bit semiprimes in ~30 seconds is **definitively falsified**.

The algorithm completes quickly but **fails to find factors**, making the speed irrelevant. This is not a performance issue but a **correctness issue** - the algorithm fundamentally cannot reach the factors for unbalanced semiprimes.

---

*Experiment conducted: 2025-12-26*  
*Status: Complete*  
*Hypothesis: FALSIFIED*
