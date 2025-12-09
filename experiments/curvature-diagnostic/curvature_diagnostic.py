"""
Curvature Diagnostic Analysis

This script implements a curvature-based diagnostic for prime classification
using the κ(n) = d(n) · ln(n) / e² metric, where d(n) is the divisor count.

The hypothesis being tested: κ(n) provides a structural signature that can
classify primes with accuracy significantly better than random (50%).
"""

import numpy as np
import math
import argparse
import json
from pathlib import Path


def divisor_count(n: int) -> int:
    """Count divisors of n."""
    if n <= 0:
        return 0
    count = 0
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            count += 1 if i * i == n else 2
    return count


def kappa(n: int) -> float:
    """κ(n) = d(n) · ln(n) / e²."""
    if n < 2:
        return 0.0
    return divisor_count(n) * math.log(n) / math.exp(2)


def is_prime(n: int) -> bool:
    """Simple primality check for validation."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def run_analysis(max_n: int, v_param: float, bootstrap_samples: int) -> dict:
    """Compute κ(n), classification accuracy with bootstrap CI, and Z(n)."""
    print(f"Running analysis for n=2 to n={max_n}...")
    print(f"Parameters: v={v_param}, bootstrap_samples={bootstrap_samples}")
    
    ns = np.arange(2, max_n + 1)
    print(f"Computing κ(n) for {len(ns)} values...")
    kappas = np.array([kappa(n) for n in ns])
    
    print(f"Checking primality for {len(ns)} values...")
    primes = np.array([is_prime(n) for n in ns])
    
    threshold = 1.5  # Empirical threshold for classification
    classified_primes = kappas < threshold
    accuracy = np.mean(classified_primes == primes)
    
    print(f"Base accuracy: {accuracy:.2%}")
    print(f"Computing bootstrap confidence interval with {bootstrap_samples} samples...")
    
    # Bootstrap CI for accuracy
    accuracies = []
    for i in range(bootstrap_samples):
        if (i + 1) % 100 == 0:
            print(f"  Bootstrap sample {i + 1}/{bootstrap_samples}...")
        idx = np.random.choice(len(ns), len(ns), replace=True)
        acc = np.mean(classified_primes[idx] == primes[idx])
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    ci = np.percentile(accuracies, [2.5, 97.5])
    
    # Z-normalization: Transform number space according to curvature
    # Z(n) = n / exp(v × κ(n)) where v is the traversal rate parameter
    # This transformation reveals structural invariants in the cognitive number theory framework
    deltas = v_param * kappas
    z = ns / np.exp(deltas)
    
    # Calculate delta from baseline (50%)
    baseline = 0.5
    delta_from_baseline = (accuracy - baseline) / baseline * 100
    
    # Additional statistics
    prime_count = np.sum(primes)
    composite_count = len(primes) - prime_count
    true_positives = np.sum(classified_primes & primes)
    false_positives = np.sum(classified_primes & ~primes)
    true_negatives = np.sum(~classified_primes & ~primes)
    false_negatives = np.sum(~classified_primes & primes)
    
    # Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,  # Use full dataset accuracy
        'bootstrap_mean_accuracy': mean_acc,  # Mean of bootstrap samples (for reference)
        'ci': ci.tolist(),
        'delta_from_baseline_pct': delta_from_baseline,
        'kappas': kappas,
        'z': z,
        'ns': ns,
        'threshold': threshold,
        'prime_count': int(prime_count),
        'composite_count': int(composite_count),
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'true_negatives': int(true_negatives),
        'false_negatives': int(false_negatives),
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'max_n': max_n,
        'v_param': v_param,
        'bootstrap_samples': bootstrap_samples
    }


def main():
    parser = argparse.ArgumentParser(
        description="Curvature diagnostic analysis for prime classification."
    )
    parser.add_argument(
        '--max-n', 
        type=int, 
        default=50, 
        help="Max n for analysis (default: 50)"
    )
    parser.add_argument(
        '--v-param', 
        type=float, 
        default=1.0, 
        help="Traversal rate v (default: 1.0)"
    )
    parser.add_argument(
        '--bootstrap-samples', 
        type=int, 
        default=1000, 
        help="Bootstrap samples for CI (default: 1000)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help="Output directory for artifacts (default: current directory)"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run analysis
    results = run_analysis(args.max_n, args.v_param, args.bootstrap_samples)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"95% CI: [{results['ci'][0]:.4f}, {results['ci'][1]:.4f}]")
    print(f"Delta from baseline (50%): {results['delta_from_baseline_pct']:+.2f}%")
    print(f"Precision: {results['precision']:.2%}")
    print(f"Recall: {results['recall']:.2%}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"\nPrimes: {results['prime_count']}, Composites: {results['composite_count']}")
    print(f"True Positives: {results['true_positives']}, False Positives: {results['false_positives']}")
    print(f"True Negatives: {results['true_negatives']}, False Negatives: {results['false_negatives']}")
    print("=" * 60)
    
    # Save CSV data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'kappas.csv'
    data = np.column_stack((results['ns'], results['kappas'], results['z']))
    np.savetxt(csv_path, data, delimiter=',', header='n,kappa,z', comments='')
    print(f"\nSaved kappas data to: {csv_path}")
    
    # Save CI and metrics JSON
    ci_data = {
        'accuracy': results['accuracy'],
        'ci_lower': results['ci'][0],
        'ci_upper': results['ci'][1],
        'delta_from_baseline_pct': results['delta_from_baseline_pct'],
        'threshold': results['threshold'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1_score': results['f1_score'],
        'prime_count': results['prime_count'],
        'composite_count': results['composite_count'],
        'confusion_matrix': {
            'true_positives': results['true_positives'],
            'false_positives': results['false_positives'],
            'true_negatives': results['true_negatives'],
            'false_negatives': results['false_negatives']
        },
        'parameters': {
            'max_n': results['max_n'],
            'v_param': results['v_param'],
            'bootstrap_samples': results['bootstrap_samples']
        }
    }
    
    ci_path = output_dir / 'ci.json'
    with open(ci_path, 'w') as f:
        json.dump(ci_data, f, indent=2)
    print(f"Saved CI and metrics to: {ci_path}")


if __name__ == "__main__":
    main()
