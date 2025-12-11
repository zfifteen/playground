#!/usr/bin/env python3
"""
QMC BASELINE GENERATORS AND DISCREPANCY METRICS
================================================

Implements standard QMC sequence generators and standardized discrepancy
measurements for comparative evaluation of Anosov-driven sampling.

This module provides:
1. Sobol sequences (scrambled, via scipy.stats.qmc)
2. Halton sequences (scrambled, via scipy.stats.qmc)
3. Random sequences (numpy RNG)
4. Simple lattice points
5. Standardized discrepancy measurements (Centered 'CD' and Wrap-around 'WD')

Author: Big D (zfifteen)
Date: December 2025
"""

import numpy as np
from scipy.stats import qmc
from typing import Tuple, Optional, List


class QMCBaselineGenerator:
    """
    Factory for generating baseline QMC sequences for comparison.
    """
    
    def __init__(self, dimension: int = 2, seed: Optional[int] = None):
        """
        Initialize baseline generator.
        
        Parameters:
        -----------
        dimension : int
            Dimension of the sequences (default: 2)
        seed : Optional[int]
            Random seed for reproducibility
        """
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def generate_sobol(self, n_points: int, scramble: bool = True) -> np.ndarray:
        """
        Generate Sobol sequence points.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
        scramble : bool
            Whether to use scrambling (recommended for better uniformity)
            
        Returns:
        --------
        points : np.ndarray of shape (n_points, dimension)
            Sobol sequence points in [0,1)^d
        """
        sampler = qmc.Sobol(d=self.dimension, scramble=scramble, seed=self.seed)
        # Skip first point (often [0, 0, ...])
        sampler.reset()
        _ = sampler.random(1)
        points = sampler.random(n_points)
        return points
    
    def generate_halton(self, n_points: int, scramble: bool = True) -> np.ndarray:
        """
        Generate Halton sequence points.
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
        scramble : bool
            Whether to use scrambling (recommended for higher dimensions)
            
        Returns:
        --------
        points : np.ndarray of shape (n_points, dimension)
            Halton sequence points in [0,1)^d
        """
        sampler = qmc.Halton(d=self.dimension, scramble=scramble, seed=self.seed)
        points = sampler.random(n_points)
        return points
    
    def generate_random(self, n_points: int) -> np.ndarray:
        """
        Generate pseudo-random points (baseline comparison).
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate
            
        Returns:
        --------
        points : np.ndarray of shape (n_points, dimension)
            Random points in [0,1)^d
        """
        return self.rng.uniform(0, 1, size=(n_points, self.dimension))
    
    def generate_lattice(self, n_points: int) -> np.ndarray:
        """
        Generate simple lattice points (uniform grid).
        
        Parameters:
        -----------
        n_points : int
            Number of points to generate (will be adjusted to nearest perfect square for 2D)
            
        Returns:
        --------
        points : np.ndarray of shape (actual_n_points, dimension)
            Lattice points in [0,1)^d
        """
        if self.dimension == 2:
            # For 2D, create a square grid
            n_per_dim = int(np.ceil(np.sqrt(n_points)))
            x = np.linspace(0, 1, n_per_dim, endpoint=False)
            y = np.linspace(0, 1, n_per_dim, endpoint=False)
            xx, yy = np.meshgrid(x, y)
            points = np.column_stack([xx.ravel(), yy.ravel()])
            # Return exactly n_points
            return points[:n_points]
        else:
            # For higher dimensions, use a simple grid along each axis
            n_per_dim = int(np.ceil(n_points ** (1.0 / self.dimension)))
            axes = [np.linspace(0, 1, n_per_dim, endpoint=False) for _ in range(self.dimension)]
            grid = np.meshgrid(*axes, indexing='ij')
            points = np.column_stack([g.ravel() for g in grid])
            return points[:n_points]


class DiscrepancyMetrics:
    """
    Standardized discrepancy measurements using scipy.stats.qmc.
    
    Replaces ad-hoc Monte Carlo star discrepancy with reproducible metrics:
    - 'CD' (Centered Discrepancy)
    - 'WD' (Wrap-around Discrepancy)
    """
    
    @staticmethod
    def compute_discrepancy(points: np.ndarray, method: str = 'CD') -> float:
        """
        Compute discrepancy using scipy.stats.qmc.discrepancy.
        
        Parameters:
        -----------
        points : np.ndarray of shape (n_points, dimension)
            Points in [0,1)^d
        method : str
            Discrepancy method: 'CD' (Centered) or 'WD' (Wrap-around)
            
        Returns:
        --------
        discrepancy : float
            Computed discrepancy value
        """
        if method not in ['CD', 'WD']:
            raise ValueError(f"Method must be 'CD' or 'WD', got '{method}'")
        
        return qmc.discrepancy(points, method=method)
    
    @staticmethod
    def compute_both_discrepancies(points: np.ndarray) -> Tuple[float, float]:
        """
        Compute both CD and WD discrepancies.
        
        Parameters:
        -----------
        points : np.ndarray of shape (n_points, dimension)
            Points in [0,1)^d
            
        Returns:
        --------
        cd_discrepancy : float
            Centered discrepancy
        wd_discrepancy : float
            Wrap-around discrepancy
        """
        cd = qmc.discrepancy(points, method='CD')
        wd = qmc.discrepancy(points, method='WD')
        return cd, wd


def compare_baselines(n_points: int = 1000, 
                      dimension: int = 2,
                      seed: Optional[int] = 42,
                      methods: Optional[List[str]] = None) -> dict:
    """
    Generate all baseline sequences and compute their discrepancies.
    
    Parameters:
    -----------
    n_points : int
        Number of points to generate for each method
    dimension : int
        Dimension of the sequences
    seed : Optional[int]
        Random seed for reproducibility
    methods : Optional[List[str]]
        List of methods to compare. Default: all methods
        Options: ['sobol', 'halton', 'random', 'lattice']
        
    Returns:
    --------
    results : dict
        Dictionary with keys for each method, containing:
        - 'points': generated points
        - 'cd_discrepancy': centered discrepancy
        - 'wd_discrepancy': wrap-around discrepancy
    """
    if methods is None:
        methods = ['sobol', 'halton', 'random', 'lattice']
    
    generator = QMCBaselineGenerator(dimension=dimension, seed=seed)
    metrics = DiscrepancyMetrics()
    results = {}
    
    method_map = {
        'sobol': generator.generate_sobol,
        'halton': generator.generate_halton,
        'random': generator.generate_random,
        'lattice': generator.generate_lattice,
    }
    
    for method in methods:
        if method not in method_map:
            raise ValueError(f"Unknown method: {method}")
        
        points = method_map[method](n_points)
        cd, wd = metrics.compute_both_discrepancies(points)
        
        results[method] = {
            'points': points,
            'cd_discrepancy': cd,
            'wd_discrepancy': wd,
        }
    
    return results


if __name__ == "__main__":
    """
    Quick test of baseline generators and discrepancy metrics.
    """
    print("=" * 70)
    print("QMC BASELINE COMPARISON TEST")
    print("=" * 70)
    print()
    
    # Test with different point counts
    n_values = [1000, 5000, 10000]
    
    for n in n_values:
        print(f"Testing with N = {n} points:")
        print("-" * 70)
        
        results = compare_baselines(n_points=n, dimension=2, seed=42)
        
        # Sort by CD discrepancy for display
        sorted_methods = sorted(results.keys(), 
                              key=lambda m: results[m]['cd_discrepancy'])
        
        for method in sorted_methods:
            cd = results[method]['cd_discrepancy']
            wd = results[method]['wd_discrepancy']
            print(f"  {method:10s}: CD = {cd:.6f}, WD = {wd:.6f}")
        
        print()
    
    print("=" * 70)
    print("Expected ordering (typically): Sobol < Halton < Random")
    print("Lattice may vary depending on N and how points align")
    print("=" * 70)
