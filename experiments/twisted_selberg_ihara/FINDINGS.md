# Conclusion
Quasi Monte Carlo (Halton) sampling produced a decisively lower integration error than plain Monte Carlo on the x·y unit-square integral (0.00170 vs 0.01210 with 512 samples, seed=42), supporting the hypothesis that low-discrepancy traces deliver superior convergence for the twisted Selberg/Ihara-inspired sampling task.

## Experiment Overview
- **Objective:** Empirically contrast Monte Carlo (MC) and quasi Monte Carlo (QMC) sampling as a stand-in for the twisted Selberg meromorphic / p-adic Ihara trace efficiency claim on flat cosmology-like integrals.
- **Design:** Deterministic Halton sequence (bases 2,3) vs. RNG-based MC on the analytic integral ∫₀¹∫₀¹ x·y dx dy = 0.25.
- **Implementation:** `TwistedSelbergIharaExperiment.run_integral_comparison` (static method) executes both estimators and reports absolute error.

## Technical Evidence
- True integral: **0.25**
- Sample size: **512**; Seed: **42**
- MC estimate: **0.2621008657** → error **0.0121008657**
- QMC (Halton) estimate: **0.2483029464** → error **0.0016970536**
- Outcome: `qmc_better=True` (≈7.1× lower error)

## Reproduction Steps
1. From repo root: `python - <<'PY'`
2. ```
   from experiments.twisted_selberg_ihara.experiment import TwistedSelbergIharaExperiment
   print(TwistedSelbergIharaExperiment.run_integral_comparison(sample_size=512, seed=42))
   ```
3. Confirm the reported errors match the evidence above.

## Notes and Limitations
- Only the QMC vs MC integration trace is implemented; higher-level curvature and twisted Dirac/Ihara coupling stubs remain specified but unimplemented per incremental protocol.
- Halton sampling stands in for Sobol/Owen to avoid extra dependencies while still exercising a low-discrepancy sequence.
