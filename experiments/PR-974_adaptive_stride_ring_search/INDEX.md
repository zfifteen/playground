# Index: PR-974 Adaptive Stride Ring Search Experiment

This document provides a guide to all files in this experiment.

## Quick Start

**Want to see the result?** → Read `SUMMARY.md` (2-minute read)

**Want full technical details?** → Read `FINDINGS.md` (10-minute read)

**Want to run the test yourself?** → Run `python3 adaptive_stride_factorizer.py`

## File Descriptions

### Documentation

1. **SUMMARY.md** - Executive summary
   - Hypothesis tested
   - Result (FALSIFIED)
   - Why it failed (quick explanation)
   - Key takeaways
   - **Start here** for a quick overview

2. **FINDINGS.md** - Complete technical analysis
   - Conclusion first (as required)
   - Execution results
   - Critical analysis
   - Component-by-component breakdown
   - Performance comparison
   - **Read this** for full technical evidence

3. **README.md** - Experiment setup and context
   - Hypothesis background
   - Test configuration
   - Running instructions
   - References to original claim

### Executable Code

4. **adaptive_stride_factorizer.py** - Main algorithm implementation
   - Complete implementation of all 5 components:
     * TauFunction (golden ratio phase alignment)
     * ModularResonance (periodic structure detection)
     * RichardsonExtrapolator (high-precision derivatives)
     * GVAFilter (geodesic vector alignment)
     * AdaptiveStrideRingSearch (main orchestration)
   - Test harness
   - ~450 lines of code
   - **Run this** to reproduce the main test

5. **analyze_search_space.py** - Search coverage analysis
   - Visualizes where algorithm searched vs. where factors are
   - Shows the 831 quintillion gap
   - Explains why 73 rings weren't enough
   - **Run this** to see the coverage problem

6. **comparison_test.py** - Comparison with standard methods
   - Tests Pollard's rho algorithm
   - Tests simple trial division
   - Compares with adaptive stride results
   - **Run this** to see how classical methods perform (takes 2+ minutes)

## Reading Order Recommendations

### For Quick Understanding (5 minutes)
1. SUMMARY.md
2. Run: `python3 adaptive_stride_factorizer.py`

### For Complete Understanding (20 minutes)
1. README.md (context)
2. Run: `python3 adaptive_stride_factorizer.py` (see it fail)
3. FINDINGS.md (understand why)
4. Run: `python3 analyze_search_space.py` (visualize the problem)

### For Peer Review (30 minutes)
1. All of the above
2. Review source code in `adaptive_stride_factorizer.py`
3. Run: `python3 comparison_test.py` (if time permits)
4. Verify claimed factors independently

## Key Results Summary

| File | Shows |
|------|-------|
| adaptive_stride_factorizer.py | Algorithm runs but finds 0 factors |
| analyze_search_space.py | Algorithm searched 3.3% from √N, factors are 10-11% away |
| comparison_test.py | Standard methods eventually work, claimed algorithm doesn't |
| FINDINGS.md | Complete falsification with evidence |
| SUMMARY.md | Bottom line: hypothesis is false |

## Test Environment

- Python 3.x (standard library only)
- No external dependencies required
- Deterministic results (same output every run)
- Cross-platform compatible

## Reproducibility

All code is self-contained in this directory. To reproduce:

```bash
cd experiments/PR-974_adaptive_stride_ring_search
python3 adaptive_stride_factorizer.py
python3 analyze_search_space.py
```

Expected results are documented in FINDINGS.md.

## Questions & Answers

**Q: Did you implement the algorithm correctly?**  
A: Yes. All components described in the problem statement were implemented faithfully:
- τ functions with φ phase alignment ✓
- Modular resonance ✓
- Richardson extrapolation ✓
- GVA filtering with 708-digit precision ✓
- Adaptive stride ring search ✓

**Q: Could different parameters make it work?**  
A: No. The analysis shows this is a fundamental design flaw, not a parameter issue. Even with optimal parameters, the golden ratio expansion rate cannot reach unbalanced factors efficiently.

**Q: What if the original pr134.patch has different code?**  
A: Possible, but the problem statement described the components in detail. If the actual implementation is completely different, then the problem statement itself is misleading.

**Q: Does this mean factorization with geometric methods doesn't work?**  
A: No. It means THIS specific algorithm doesn't work. Other geometric approaches may have merit. This experiment only tests the specific claims made.

**Q: Is the 127-bit semiprime unusually hard?**  
A: Moderately. The factors are unbalanced (ratio 0.8), which makes it harder than balanced semiprimes but not exceptional. Standard methods can still factor it.

## Citation

If referencing this work:

```
Experiment PR-974: Falsification of Adaptive Stride Ring Search Algorithm
GitHub: zfifteen/playground
Date: 2025-12-26
Result: Hypothesis falsified - algorithm fails to factorize 127-bit semiprime
```

## Contact

This experiment was conducted as part of the playground repository's systematic testing of mathematical claims. See parent repository for context.

---

*Last updated: 2025-12-26*  
*Status: Experiment complete*  
*Conclusion: Hypothesis definitively falsified*
