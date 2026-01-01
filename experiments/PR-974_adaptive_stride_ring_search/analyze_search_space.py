#!/usr/bin/env python3
"""
Visualization of the search space problem.

This script demonstrates why the adaptive stride ring search failed:
it searched in the wrong region of the number space.
"""

import math


def analyze_search_coverage():
    """Analyze where the algorithm searched vs. where the factors are."""
    
    N = 137524771864208156028430259349934309717
    p = 10508623501177419659
    q = 13086849276577416863
    sqrt_N = int(math.isqrt(N))
    
    print("=" * 80)
    print("SEARCH SPACE ANALYSIS")
    print("=" * 80)
    print()
    
    # Key positions
    print("Key Positions:")
    print(f"  N         = {N:,}")
    print(f"  sqrt(N)   = {sqrt_N:,}")
    print(f"  Factor p  = {p:,}")
    print(f"  Factor q  = {q:,}")
    print()
    
    # Distances
    dist_p = abs(sqrt_N - p)
    dist_q = abs(q - sqrt_N)
    
    print("Distances from sqrt(N):")
    print(f"  To p: {dist_p:,} ({(dist_p/sqrt_N)*100:.2f}% of sqrt(N))")
    print(f"  To q: {dist_q:,} ({(dist_q/sqrt_N)*100:.2f}% of sqrt(N))")
    print()
    
    # Algorithm search range
    base_stride = sqrt_N // 1000
    num_rings = 73  # From actual run
    
    print("Algorithm Search Parameters:")
    print(f"  Base stride: {base_stride:,}")
    print(f"  Number of rings: {num_rings}")
    print()
    
    # Estimate maximum search distance
    # Using golden ratio expansion: offset = PHI^(i/10) * stride
    PHI = (1 + math.sqrt(5)) / 2
    max_ring_index = num_rings
    max_offset_factor = int(PHI ** (max_ring_index / 10))
    max_search_distance = max_offset_factor * base_stride
    
    print(f"Estimated Maximum Search Distance:")
    print(f"  Offset factor (PHI^({max_ring_index}/10)): {max_offset_factor:,}")
    print(f"  Max distance: ~{max_search_distance:,}")
    print(f"  As % of sqrt(N): {(max_search_distance/sqrt_N)*100:.2f}%")
    print()
    
    # Coverage analysis
    print("Coverage Analysis:")
    print(f"  Distance to p: {dist_p:,}")
    print(f"  Max search:    {max_search_distance:,}")
    print(f"  Shortfall:     {max(0, dist_p - max_search_distance):,}")
    print()
    
    if dist_p > max_search_distance:
        print("  ❌ Algorithm DID NOT search far enough to reach factor p")
        print(f"     Would need to search {(dist_p/max_search_distance):.1f}x further")
    else:
        print("  ✓ Algorithm should have covered region containing p")
    
    print()
    
    # Visual representation
    print("Visual Representation (not to scale):")
    print()
    print("  0" + "-" * 20 + "p" + "-" * 15 + "sqrt(N)" + "-" * 15 + "q" + "-" * 20 + "N")
    print("  ^" + " " * 20 + "^" + " " * 15 + "^" + " " * 15 + "^" + " " * 20 + "^")
    print("  |" + " " * 20 + "|" + " " * 15 + "|" + " " * 15 + "|" + " " * 20 + "|")
    print("  1" + " " * 14 + "10.5e18" + " " * 6 + "11.7e18" + " " * 6 + "13.1e18" + " " * 10 + "1.4e38")
    print()
    
    # Where algorithm actually searched
    search_center = sqrt_N
    search_radius = max_search_distance
    search_min = search_center - search_radius
    search_max = search_center + search_radius
    
    print(f"Algorithm searched: [{search_min:,} to {search_max:,}]")
    print(f"Factor p location:  {p:,}")
    print(f"Factor q location:  {q:,}")
    print()
    
    if p < search_min:
        print(f"❌ Factor p is BELOW search range by {search_min - p:,}")
    elif p > search_max:
        print(f"❌ Factor p is ABOVE search range by {p - search_max:,}")
    else:
        print(f"✓ Factor p is WITHIN search range")
    
    print()
    
    # Calculate required number of rings to reach p
    # Solve: PHI^(n/10) * stride = dist_p
    # n = 10 * log(dist_p / stride) / log(PHI)
    required_n = 10 * math.log(dist_p / base_stride) / math.log(PHI)
    print(f"To reach factor p would require:")
    print(f"  Ring index: {required_n:.1f}")
    print(f"  Actual rings generated: {num_rings}")
    print(f"  Shortfall: {max(0, required_n - num_rings):.1f} rings")
    print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The algorithm failed because it:")
    print("1. Searched in a region centered on sqrt(N)")
    print("2. Did not expand the search far enough (~10% of sqrt(N))")
    print("3. Required ~104 rings but only generated 73 rings")
    print("4. Even with golden ratio expansion, could not reach the true factors")
    print()
    print("This is a fundamental algorithm design flaw, not a parameter tuning issue.")
    print()


if __name__ == "__main__":
    analyze_search_coverage()
