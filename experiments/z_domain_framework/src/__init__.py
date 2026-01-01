"""
Z-Domain Framework for Riemann Hypothesis Verification

This package implements the Z-domain transform framework for testing
the Riemann Hypothesis through phase mapping and GUE comparison.
"""

__version__ = '1.0.0'

from .zeta_zeros import ZetaZerosDataset, compute_zero_differences, compute_normalized_gaps
from .z_transform import ZTransform, compute_gap_autocorrelation
from .gue_analysis import GUEComparison, MultiplicitySensitivity
from .visualization import ZDomainVisualizer

__all__ = [
    'ZetaZerosDataset',
    'compute_zero_differences',
    'compute_normalized_gaps',
    'ZTransform',
    'compute_gap_autocorrelation',
    'GUEComparison',
    'MultiplicitySensitivity',
    'ZDomainVisualizer',
]
