# PR-974: Adaptive Stride Ring Search Algorithm Test

## Objective

This experiment tests the hypothesis that an adaptive stride ring search algorithm can successfully factorize 127-bit semiprimes in approximately 30 seconds by integrating τ functions with golden ratio phase alignment, modular resonance, and Richardson extrapolation, elevating the true factor from rank 317 to rank 1 through GVA filtering.

## Hypothesis to Test

**Claim**: The adaptive stride ring search algorithm factorizes the 127-bit semiprime N = 137524771864208156028430259349934309717 (with factors p = 10508623501177419659 and q = 13086849276577416863) in approximately 30 seconds.

**Key Components**:
1. τ (tau) functions with golden ratio (φ) phase alignment
2. Modular resonance for periodic structure detection
3. Richardson extrapolation for precise derivative calculations
4. GVA (Geodesic Vector Alignment) filtering mechanism
5. Adaptive stride ring search with dynamic parameter adjustment

**Performance Target**: ~30 seconds factorization time

**Claimed Result**: GVA filtering elevates true factor from rank 317 to rank 1

## Test Semiprime

- **N** = 137524771864208156028430259349934309717
- **p** = 10508623501177419659
- **q** = 13086849276577416863
- **Bit length**: 127 bits
- **Verification**: p × q = N ✓, both p and q are prime ✓

## Implementation

The algorithm is implemented in `adaptive_stride_factorizer.py` with the following components:

1. **TauFunction**: Implements τ functions with golden ratio phase alignment
2. **ModularResonance**: Detects periodic structure in the search space
3. **RichardsonExtrapolator**: Provides high-precision derivative calculations
4. **GVAFilter**: Geodesic Vector Alignment filtering for candidate ranking
5. **AdaptiveStrideRingSearch**: Main algorithm orchestrating all components

## Running the Test

```bash
python3 adaptive_stride_factorizer.py
```

This will:
1. Initialize the algorithm with the test semiprime
2. Run the factorization with timing
3. Report performance metrics
4. Compare results against claimed performance

## Expected Outcomes

### Success Criteria (Hypothesis Confirmed)
- Algorithm finds factors p and q
- Factorization completes in ≤ 60 seconds
- GVA filter demonstrates significant rank improvement for true factor

### Failure Criteria (Hypothesis Falsified)
- Algorithm fails to find factors within reasonable time (>300 seconds)
- Factors found but time significantly exceeds claim (>120 seconds)
- GVA filtering shows no meaningful rank improvement
- Algorithm requires modifications to work at all

## Results

See `FINDINGS.md` for detailed test results and analysis.

## References

- Original claim: GitHub unified-framework PR #974
- Context: Issue #132 (previous failures with precision offsets and ranking errors)
- Related: Isospectral tori hypothesis (falsified, 100% failure rate)
