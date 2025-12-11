#!/usr/bin/env python3
"""
SL(2, Z) MATRIX ENUMERATION
============================

Enumerates hyperbolic SL(2, Z) matrices systematically for testing Anosov
dynamics and QMC sampling quality.

Key features:
1. Enumerate by trace with bounded entries
2. Filter for hyperbolicity: |trace| > 2 (ensures real distinct eigenvalues)
3. Enforce determinant = 1 (unimodular condition)
4. Optional de-duplication by simple invariants (trace, eigenvalue magnitude)
5. Numeric stability guardrails

Author: Big D (zfifteen)
Date: December 2025
"""

import numpy as np
from scipy.linalg import eigvals
from typing import List, Tuple, Optional, Set
import itertools


class SL2ZEnumerator:
    """
    Enumerates hyperbolic SL(2, Z) matrices with specified constraints.
    """
    
    def __init__(self, max_entry: int = 20, min_trace_abs: int = 3):
        """
        Initialize enumerator.
        
        Parameters:
        -----------
        max_entry : int
            Maximum absolute value for matrix entries
        min_trace_abs : int
            Minimum absolute trace for hyperbolicity (default: 3, ensures |tr| > 2)
        """
        self.max_entry = max_entry
        self.min_trace_abs = min_trace_abs
        
    def is_hyperbolic(self, matrix: np.ndarray) -> bool:
        """
        Check if matrix is hyperbolic (|trace| > 2).
        
        For SL(2,Z), this ensures real distinct eigenvalues with |λ₁| > 1 > |λ₂|.
        """
        trace = np.trace(matrix)
        return abs(trace) > 2
    
    def compute_invariants(self, matrix: np.ndarray) -> Tuple[int, int]:
        """
        Compute conjugacy invariants for de-duplication.
        
        For SL(2,Z), matrices are conjugate if they have the same:
        - trace (invariant under conjugation)
        - discriminant = trace² - 4 (determines eigenvalue structure)
        
        This is more mathematically sound than using Frobenius norm.
        
        Returns:
        --------
        trace : int
            Matrix trace
        discriminant : int
            trace² - 4 (determines conjugacy class)
        
        Note: This is still an approximation. Full conjugacy classification
        in SL(2,Z) requires orbit representatives, but this is adequate for
        generating diverse test matrices.
        """
        trace = int(np.trace(matrix))
        discriminant = trace * trace - 4
        return trace, discriminant
    
    def enumerate_by_trace(self, 
                          target_trace: int,
                          deduplicate: bool = True,
                          limit_per_trace: Optional[int] = None) -> List[np.ndarray]:
        """
        Enumerate all SL(2,Z) matrices with a specific trace.
        
        For a 2x2 matrix [[a, b], [c, d]] with det = 1:
        - trace = a + d = target_trace
        - ad - bc = 1
        
        Parameters:
        -----------
        target_trace : int
            Target trace value
        deduplicate : bool
            Whether to remove duplicates by conjugacy/similarity
        limit_per_trace : Optional[int]
            Maximum matrices to return per trace (None = no limit)
            
        Returns:
        --------
        matrices : List[np.ndarray]
            List of unique hyperbolic SL(2,Z) matrices
        """
        if abs(target_trace) <= 2:
            return []  # Not hyperbolic
        
        matrices = []
        seen_invariants: Set[Tuple[int, int]] = set()
        
        # Iterate over possible values of a, b, c
        # Then compute d = trace - a and check det = ad - bc = 1
        for a in range(-self.max_entry, self.max_entry + 1):
            d_target = target_trace - a
            
            # Check if d is in valid range
            if abs(d_target) > self.max_entry:
                continue
            
            for b in range(-self.max_entry, self.max_entry + 1):
                for c in range(-self.max_entry, self.max_entry + 1):
                    # Compute determinant: ad - bc should equal 1
                    det = a * d_target - b * c
                    
                    if det != 1:
                        continue
                    
                    matrix = np.array([[a, b], [c, d_target]], dtype=int)
                    
                    # Verify hyperbolicity (redundant but safe)
                    if not self.is_hyperbolic(matrix):
                        continue
                    
                    # De-duplicate if requested
                    if deduplicate:
                        inv = self.compute_invariants(matrix)
                        if inv in seen_invariants:
                            continue
                        seen_invariants.add(inv)
                    
                    matrices.append(matrix)
                    
                    if limit_per_trace and len(matrices) >= limit_per_trace:
                        return matrices
        
        return matrices
    
    def enumerate_by_trace_range(self,
                                 min_trace: int,
                                 max_trace: int,
                                 deduplicate: bool = True,
                                 max_matrices: Optional[int] = None,
                                 limit_per_trace: Optional[int] = 3) -> List[np.ndarray]:
        """
        Enumerate matrices across a range of traces.
        
        Parameters:
        -----------
        min_trace : int
            Minimum trace (absolute value)
        max_trace : int
            Maximum trace (absolute value)
        deduplicate : bool
            Whether to de-duplicate
        max_matrices : Optional[int]
            Maximum number of matrices to return (None = no limit)
        limit_per_trace : Optional[int]
            Maximum matrices per trace value (for diversity)
            
        Returns:
        --------
        matrices : List[np.ndarray]
            List of matrices, sorted by trace
        """
        all_matrices = []
        
        # Consider both positive and negative traces
        for trace_abs in range(max(min_trace, self.min_trace_abs), max_trace + 1):
            for trace in [trace_abs, -trace_abs]:
                matrices = self.enumerate_by_trace(trace, deduplicate=deduplicate, limit_per_trace=limit_per_trace)
                all_matrices.extend(matrices)
                
                if max_matrices is not None and len(all_matrices) >= max_matrices:
                    return all_matrices[:max_matrices]
        
        return all_matrices
    
    def get_standard_test_set(self, 
                              n_matrices: int = 50,
                              diversity: str = 'trace') -> List[np.ndarray]:
        """
        Get a standard test set of matrices with good diversity.
        
        Parameters:
        -----------
        n_matrices : int
            Target number of matrices (may return fewer if insufficient diversity)
        diversity : str
            Diversity criterion: 'trace' (default), 'entropy', or 'mixed'
            
        Returns:
        --------
        matrices : List[np.ndarray]
            Diverse set of test matrices
        """
        # Generate a large pool with more matrices per trace for diversity
        pool = self.enumerate_by_trace_range(
            min_trace=3, 
            max_trace=45,  # Larger range to ensure >= 50 matrices
            deduplicate=True,
            max_matrices=None,  # Get all
            limit_per_trace=5   # More per trace
        )
        
        if len(pool) <= n_matrices:
            # If we don't have enough, try to get more by reducing deduplication
            if len(pool) < n_matrices:
                # Get more without strict deduplication
                pool_extra = self.enumerate_by_trace_range(
                    min_trace=3, 
                    max_trace=50,  # Even larger
                    deduplicate=False,  # Allow more duplicates
                    max_matrices=n_matrices * 2,
                    limit_per_trace=15
                )
                # Add unique ones
                seen = {self.compute_invariants(m) for m in pool}
                for m in pool_extra:
                    inv = self.compute_invariants(m)
                    if inv not in seen:
                        pool.append(m)
                        seen.add(inv)
                    if len(pool) >= n_matrices:
                        break
            return pool[:n_matrices]  # Always return exactly n_matrices or less
        
        if diversity == 'trace':
            # Select matrices with diverse traces
            traces = [int(np.trace(m)) for m in pool]
            unique_traces = sorted(set(traces))
            
            selected = []
            for trace in unique_traces:
                candidates = [m for m, t in zip(pool, traces) if t == trace]
                selected.extend(candidates[:max(1, n_matrices // len(unique_traces))])
                if len(selected) >= n_matrices:
                    break
            
            return selected[:n_matrices]
        
        elif diversity == 'entropy':
            # Select matrices with diverse entropies
            entropies = []
            for m in pool:
                evals = eigvals(m)
                entropy = np.log(max(abs(evals)))
                entropies.append(entropy)
            
            # Sort by entropy and take evenly spaced samples
            sorted_indices = np.argsort(entropies)
            step = len(pool) // n_matrices
            selected_indices = sorted_indices[::max(1, step)][:n_matrices]
            
            return [pool[i] for i in selected_indices]
        
        elif diversity == 'mixed':
            # Mix of trace and entropy diversity
            half = n_matrices // 2
            trace_diverse = self.get_standard_test_set(half, diversity='trace')
            entropy_diverse = self.get_standard_test_set(n_matrices - half, diversity='entropy')
            
            # Combine and de-duplicate
            combined = trace_diverse + entropy_diverse
            seen = set()
            unique = []
            for m in combined:
                inv = self.compute_invariants(m)
                if inv not in seen:
                    seen.add(inv)
                    unique.append(m)
            
            return unique[:n_matrices]
        
        else:
            raise ValueError(f"Unknown diversity criterion: {diversity}")


def validate_matrix(matrix: np.ndarray, verbose: bool = False) -> dict:
    """
    Validate and characterize an SL(2,Z) matrix.
    
    Returns a dictionary with validation results and key properties.
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'properties': {}
    }
    
    # Check determinant
    det = np.linalg.det(matrix)
    if abs(det - 1.0) > 1e-10:
        result['valid'] = False
        result['errors'].append(f"Determinant is {det:.10f}, not 1")
    
    # Check integer entries
    if not np.allclose(matrix, matrix.astype(int)):
        result['valid'] = False
        result['errors'].append("Matrix has non-integer entries")
    
    # Check hyperbolicity
    trace = np.trace(matrix)
    result['properties']['trace'] = int(trace)
    
    if abs(trace) <= 2:
        result['valid'] = False
        result['errors'].append(f"Not hyperbolic: |trace| = {abs(trace)} <= 2")
    
    # Compute spectral properties
    evals = eigvals(matrix)
    evals_abs = sorted([abs(ev) for ev in evals], reverse=True)
    
    result['properties']['eigenvalues'] = evals
    result['properties']['lambda_max'] = evals_abs[0]
    result['properties']['lambda_min'] = evals_abs[-1]
    result['properties']['entropy'] = np.log(evals_abs[0])
    
    if len(evals_abs) > 1:
        result['properties']['spectral_gap'] = np.log(evals_abs[0] / evals_abs[1])
    
    # Check numeric stability
    if evals_abs[0] > 1e10:
        result['warnings'].append(f"Very large eigenvalue: {evals_abs[0]:.2e}")
    
    if verbose:
        print(f"Matrix validation:")
        print(f"  Valid: {result['valid']}")
        if result['errors']:
            print(f"  Errors: {result['errors']}")
        if result['warnings']:
            print(f"  Warnings: {result['warnings']}")
        print(f"  Properties: {result['properties']}")
    
    return result


if __name__ == "__main__":
    """
    Test SL(2,Z) enumeration and generate a standard test set.
    """
    print("=" * 70)
    print("SL(2, Z) MATRIX ENUMERATION TEST")
    print("=" * 70)
    print()
    
    enumerator = SL2ZEnumerator(max_entry=15, min_trace_abs=3)
    
    # Test 1: Enumerate by specific traces
    print("Test 1: Enumerate matrices with specific traces")
    print("-" * 70)
    for trace in [3, 4, 5, 6]:
        matrices = enumerator.enumerate_by_trace(trace, deduplicate=True)
        print(f"  Trace {trace}: {len(matrices)} unique matrices")
    print()
    
    # Test 2: Generate standard test set
    print("Test 2: Generate standard test set (50 matrices)")
    print("-" * 70)
    test_set = enumerator.get_standard_test_set(n_matrices=50, diversity='mixed')
    print(f"  Generated {len(test_set)} matrices")
    
    # Show trace distribution
    traces = [int(np.trace(m)) for m in test_set]
    unique_traces = sorted(set(traces))
    print(f"  Trace range: [{min(unique_traces)}, {max(unique_traces)}]")
    print(f"  Unique traces: {len(unique_traces)}")
    
    # Show entropy distribution
    entropies = []
    for m in test_set:
        evals = eigvals(m)
        entropy = np.log(max(abs(evals)))
        entropies.append(entropy)
    
    print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    print()
    
    # Test 3: Validate a few sample matrices
    print("Test 3: Validate sample matrices")
    print("-" * 70)
    
    sample_matrices = [
        np.array([[2, 1], [1, 1]]),  # Fibonacci
        np.array([[3, 2], [1, 1]]),  # Trace-4
        np.array([[10, 1], [9, 1]]), # Trace-11
    ]
    
    for i, m in enumerate(sample_matrices, 1):
        print(f"  Matrix {i}: {m.tolist()}")
        result = validate_matrix(m)
        print(f"    Valid: {result['valid']}")
        print(f"    Trace: {result['properties']['trace']}")
        print(f"    Entropy: {result['properties']['entropy']:.3f}")
        print()
    
    print("=" * 70)
    print(f"SUCCESS: Generated {len(test_set)} hyperbolic SL(2,Z) matrices")
    print("=" * 70)
