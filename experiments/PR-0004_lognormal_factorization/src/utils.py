"""Utilities for sampling and factorization helpers."""

import math
import random
from typing import Optional
from .model import Band


def sample_lognormal(shape: float, scale: float, seed: Optional[int] = None) -> float:
    """Sample from lognormal distribution (loc=0)."""
    if seed is not None:
        random.seed(seed)
    # Using Box-Muller transform for normal
    u1 = random.random()
    u2 = random.random()
    z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
    return scale * math.exp(shape * z)


def clamp_gap(gap: float, band: Band) -> float:
    """Clamp gap to band limits or generic range."""
    if band.min_gap is not None and band.max_gap is not None:
        return max(band.min_gap, min(gap, band.max_gap))
    else:
        # Generic clamp: [scale/100, 100 * scale]
        return max(band.scale / 100, min(gap, 100 * band.scale))


def is_perfect_square(n: int) -> bool:
    """Check if n is a perfect square."""
    if n < 0:
        return False
    root = math.isqrt(n)
    return root * root == n


def isqrt(n: int) -> int:
    """Integer square root."""
    return math.isqrt(n)


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def pollard_rho(N: int) -> Optional[int]:
    """Simple Pollard Rho factorization."""
    if N % 2 == 0:
        return 2
    if N < 2:
        return None

    # f(x) = x^2 + c mod N
    c = 1
    x = 3  # Changed from 2
    y = 3
    d = 1
    iterations = 0
    max_iterations = 10000  # Prevent infinite loop

    while d == 1 and iterations < max_iterations:
        x = (x * x + c) % N
        y = (y * y + c) % N
        y = (y * y + c) % N
        d = gcd(abs(x - y), N)
        iterations += 1

    if d == N or iterations >= max_iterations:
        return None  # N is prime or not found
    return d
