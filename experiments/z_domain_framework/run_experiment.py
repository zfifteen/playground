"""
Main experiment runner for Z-Domain Framework RH Verification.

This experiment tests the hypothesis that the Z-domain transform can serve
as a diagnostic tool for Riemann Hypothesis verification by detecting
phase disruptions that would indicate zero multiplicity.
"""

import numpy as np
import json
from pathlib import Path
import argparse
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from zeta_zeros import ZetaZerosDataset, compute_zero_differences, compute_normalized_gaps
from z_transform import ZTransform, compute_gap_autocorrelation
from gue_analysis import GUEComparison, MultiplicitySensitivity
from visualization import ZDomainVisualizer


def run_experiment(n_zeros: int = 5000, 
                  method: str = 'mpmath',
                  precision: int = 50,
                  output_dir: str = 'results',
                  verbose: bool = True):
    """
    Run the complete Z-domain framework experiment.
    
    Args:
        n_zeros: Number of zeta zeros to analyze
        method: Method for obtaining zeros ('mpmath', 'lmfdb', 'file')
        precision: Precision for mpmath computation
        output_dir: Directory for saving results
        verbose: Print detailed progress
        
    Returns:
        Dictionary with all experimental results
    """
    if verbose:
        print("=" * 80)
        print("Z-DOMAIN FRAMEWORK EXPERIMENT")
        print("Riemann Hypothesis Verification via Phase Mapping")
        print("=" * 80)
        print(f"\nConfiguration:")
        print(f"  Number of zeros: {n_zeros}")
        print(f"  Method: {method}")
        print(f"  Precision: {precision}")
        print(f"  Output: {output_dir}")
        print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'n_zeros': n_zeros,
            'method': method,
            'precision': precision if method == 'mpmath' else None,
        }
    }
    
    # Step 1: Acquire zeta zeros
    if verbose:
        print("Step 1: Acquiring zeta zeros...")
    
    dataset = ZetaZerosDataset(cache_dir=str(output_path / 'data'))
    zeros = dataset.get_zeros(n_zeros=n_zeros, method=method, precision=precision)
    
    if verbose:
        print(f"  ✓ Loaded {len(zeros)} zeros")
        print(f"  Range: [{zeros[0]:.4f}, {zeros[-1]:.4f}]")
        print()
    
    results['zeros_summary'] = {
        'n_zeros': len(zeros),
        'min_zero': float(zeros[0]),
        'max_zero': float(zeros[-1]),
        'mean_zero': float(np.mean(zeros)),
    }
    
    # Step 2: Compute gaps and Z-transform
    if verbose:
        print("Step 2: Computing Z-transform...")
    
    ztrans = ZTransform(zeros)
    Z_n = ztrans.compute_z_values(k=0)
    theta_n = ztrans.compute_phases(k=0)
    
    if verbose:
        print(f"  ✓ Computed {len(Z_n)} Z-values")
        print(f"  ✓ Computed {len(theta_n)} phases")
        print()
    
    # Get Z-transform statistics
    z_stats = ztrans.get_statistics(k=0)
    results['z_transform'] = z_stats
    
    if verbose:
        print("  Z-Transform Statistics:")
        print(f"    Mean: {z_stats['z_mean']:.6f}")
        print(f"    Std:  {z_stats['z_std']:.6f}")
        print(f"    Skewness: {z_stats['z_skewness']:.6f}")
        print(f"    Kurtosis: {z_stats['z_kurtosis']:.6f}")
        print()
    
    # Step 3: Analyze phase clustering
    if verbose:
        print("Step 3: Analyzing phase clustering...")
    
    clustering = ztrans.analyze_phase_clustering(k=0, n_bins=36)
    results['phase_clustering'] = clustering
    
    if verbose:
        print(f"  Chi-square: {clustering['chi_square']:.4f}")
        print(f"  p-value: {clustering['p_value']:.6e}")
        print(f"  Circular variance: {clustering['circular_variance']:.6f}")
        print(f"  Normalized entropy: {clustering['normalized_entropy']:.6f}")
        
        if clustering['p_value'] < 0.05:
            print("  ⚠️  Phases are NOT uniformly distributed")
            print("      This could indicate anomalous zero structure!")
        else:
            print("  ✓ Phases appear uniformly distributed")
            print("      Consistent with RH and simple zeros")
        print()
    
    # Step 4: GUE comparison
    if verbose:
        print("Step 4: GUE comparison analysis...")
    
    normalized_gaps = compute_normalized_gaps(zeros)
    gue = GUEComparison()
    
    spacing_test = gue.test_spacing_distribution(normalized_gaps)
    results['gue_spacing'] = spacing_test
    
    if verbose:
        print(f"  KS statistic (GUE): {spacing_test['ks_stat_wigner']:.6f}")
        print(f"  p-value (GUE): {spacing_test['p_value_wigner']:.6e}")
        print(f"  KS statistic (Poisson): {spacing_test['ks_stat_poisson']:.6f}")
        print(f"  p-value (Poisson): {spacing_test['p_value_poisson']:.6e}")
        
        if spacing_test['p_value_wigner'] > 0.05:
            print("  ✓ Gap distribution consistent with GUE")
        else:
            print("  ⚠️  Gap distribution deviates from GUE")
        print()
    
    # Step 5: Level repulsion test
    if verbose:
        print("Step 5: Testing level repulsion...")
    
    repulsion = gue.test_level_repulsion(normalized_gaps)
    results['level_repulsion'] = repulsion
    
    if verbose:
        print(f"  Small gaps fraction: {repulsion['frac_small_gaps']:.6f}")
        print(f"  Expected (GUE): {repulsion['expected_frac_gue']:.6f}")
        print(f"  Expected (Poisson): {repulsion['expected_frac_poisson']:.6f}")
        print(f"  Repulsion score: {repulsion['repulsion_score']:.6f}")
        
        if repulsion['repulsion_score'] < 1.2:
            print("  ✓ Strong level repulsion (consistent with GUE)")
        else:
            print("  ⚠️  Weak level repulsion")
        print()
    
    # Step 6: Multiplicity sensitivity
    if verbose:
        print("Step 6: Analyzing multiplicity sensitivity...")
    
    deltas = compute_zero_differences(zeros)
    sensitivity = MultiplicitySensitivity()
    
    anomalies = sensitivity.detect_anomalous_contractions(deltas, threshold=3.0)
    results['anomalies'] = {k: v for k, v in anomalies.items() 
                           if not isinstance(v, list)}
    
    b_spikes = sensitivity.compute_b_term_spikes(deltas, threshold=5.0)
    results['b_term_spikes'] = b_spikes
    
    if verbose:
        print(f"  Anomalous contractions: {anomalies['n_anomalies']}")
        print(f"  Fraction: {anomalies['frac_anomalies']:.6f}")
        print(f"  Smallest gap (normalized): {anomalies['smallest_gap_normalized']:.6f}")
        print(f"  B-term spikes: {b_spikes['n_spikes_total']}")
        print(f"  Spike fraction: {b_spikes['frac_spikes']:.6f}")
        print()
    
    # Step 7: Autocorrelation analysis
    if verbose:
        print("Step 7: Computing autocorrelation...")
    
    lags, acf_vals = compute_gap_autocorrelation(deltas, max_lag=50)
    results['autocorrelation'] = {
        'acf_lag1': float(acf_vals[1]),
        'acf_lag5': float(acf_vals[5]),
        'acf_lag10': float(acf_vals[10]),
    }
    
    if verbose:
        print(f"  ACF(1): {acf_vals[1]:.6f}")
        print(f"  ACF(5): {acf_vals[5]:.6f}")
        print(f"  ACF(10): {acf_vals[10]:.6f}")
        print()
    
    # Step 8: Generate visualizations
    if verbose:
        print("Step 8: Generating visualizations...")
    
    viz = ZDomainVisualizer(output_dir=output_dir)
    
    viz.plot_z_transform(Z_n, filename="z_transform.png")
    viz.plot_phase_circle(theta_n, filename="phase_circle.png")
    viz.plot_gue_comparison(normalized_gaps, filename="gue_comparison.png")
    viz.plot_gap_structure(zeros, deltas, filename="gap_structure.png")
    viz.plot_phase_analysis(theta_n, clustering, filename="phase_analysis.png")
    viz.plot_autocorrelation(deltas, max_lag=50, title="Gap Autocorrelation",
                            filename="gap_autocorrelation.png")
    
    if verbose:
        print()
    
    # Step 9: Formulate conclusion
    if verbose:
        print("Step 9: Formulating conclusion...")
        print()
    
    conclusion = formulate_conclusion(results)
    results['conclusion'] = conclusion
    
    if verbose:
        print("=" * 80)
        print("PRELIMINARY CONCLUSION")
        print("=" * 80)
        print(conclusion['summary'])
        print()
        print(f"RH Status: {conclusion['rh_status']}")
        print(f"Confidence: {conclusion['confidence']}")
        print("=" * 80)
        print()
    
    # Save results
    results_file = output_path / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x))
    
    if verbose:
        print(f"Results saved to: {results_file}")
    
    return results


def formulate_conclusion(results: dict) -> dict:
    """
    Formulate conclusion based on experimental results.
    
    Args:
        results: Dictionary with all experimental results
        
    Returns:
        Dictionary with conclusion and interpretation
    """
    # Criteria for RH consistency
    criteria = []
    
    # 1. Phase uniformity
    phase_p = results['phase_clustering']['p_value']
    if phase_p > 0.05:
        criteria.append(('Phase uniformity', True, 
                        f'p={phase_p:.4e} > 0.05'))
    else:
        criteria.append(('Phase uniformity', False, 
                        f'p={phase_p:.4e} < 0.05 (ANOMALOUS)'))
    
    # 2. GUE spacing
    gue_p = results['gue_spacing']['p_value_wigner']
    if gue_p > 0.05:
        criteria.append(('GUE spacing', True, 
                        f'p={gue_p:.4e} > 0.05'))
    else:
        criteria.append(('GUE spacing', False, 
                        f'p={gue_p:.4e} < 0.05 (deviation)'))
    
    # 3. Level repulsion
    repulsion_score = results['level_repulsion']['repulsion_score']
    if repulsion_score < 1.5:
        criteria.append(('Level repulsion', True, 
                        f'score={repulsion_score:.4f} < 1.5'))
    else:
        criteria.append(('Level repulsion', False, 
                        f'score={repulsion_score:.4f} > 1.5'))
    
    # 4. Anomaly frequency
    anomaly_frac = results['anomalies']['frac_anomalies']
    if anomaly_frac < 0.01:
        criteria.append(('Low anomalies', True, 
                        f'{anomaly_frac:.4%} < 1%'))
    else:
        criteria.append(('Low anomalies', False, 
                        f'{anomaly_frac:.4%} > 1%'))
    
    # Count passes
    n_pass = sum(1 for _, passed, _ in criteria if passed)
    n_total = len(criteria)
    
    # Determine status
    if n_pass == n_total:
        rh_status = "CONSISTENT"
        confidence = "High"
        summary = ("All statistical tests support RH consistency. Phase distribution "
                  "is uniform, gap spacing follows GUE, level repulsion is strong, "
                  "and anomaly rate is low. No evidence of zero multiplicity detected.")
    elif n_pass >= n_total * 0.75:
        rh_status = "LIKELY CONSISTENT"
        confidence = "Moderate"
        summary = ("Most tests support RH, but some minor deviations observed. "
                  "These could be statistical fluctuations or finite-sample effects.")
    elif n_pass >= n_total * 0.5:
        rh_status = "INCONCLUSIVE"
        confidence = "Low"
        summary = ("Mixed results. Some tests support RH while others show deviations. "
                  "Larger sample size or higher precision may be needed.")
    else:
        rh_status = "POTENTIAL ANOMALY"
        confidence = "Moderate"
        summary = ("Multiple tests show deviations from RH predictions. "
                  "Further investigation recommended. Could indicate computational "
                  "issues, statistical artifacts, or genuine mathematical structure.")
    
    return {
        'summary': summary,
        'rh_status': rh_status,
        'confidence': confidence,
        'criteria': [{'name': name, 'passed': passed, 'detail': detail}
                    for name, passed, detail in criteria],
        'pass_rate': f"{n_pass}/{n_total}",
    }


def main():
    parser = argparse.ArgumentParser(
        description='Z-Domain Framework Experiment for RH Verification'
    )
    parser.add_argument('--n-zeros', type=int, default=5000,
                       help='Number of zeta zeros to analyze (default: 5000)')
    parser.add_argument('--method', type=str, default='mpmath',
                       choices=['mpmath', 'lmfdb', 'file'],
                       help='Method for obtaining zeros (default: mpmath)')
    parser.add_argument('--precision', type=int, default=50,
                       help='Precision for mpmath (default: 50)')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with 1000 zeros')
    
    args = parser.parse_args()
    
    if args.quick:
        args.n_zeros = 1000
        print("Quick test mode: Using 1000 zeros")
    
    try:
        results = run_experiment(
            n_zeros=args.n_zeros,
            method=args.method,
            precision=args.precision,
            output_dir=args.output,
            verbose=args.verbose or True,
        )
        
        print("\n✓ Experiment completed successfully!")
        print(f"See {args.output}/ for results and visualizations")
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
