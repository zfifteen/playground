"""Tests for prefilter."""

import pytest
from src.model import ModelStore
from src.config import SearchPolicyConfig
from src.prefilter import (
    generate_lognormal_offsets,
    factor_with_candidate_prefilter,
    probably_prime,
)


def test_probably_prime():
    assert probably_prime(2)
    assert probably_prime(3)
    assert not probably_prime(4)
    assert probably_prime(7)
    assert not probably_prime(9)


def test_generate_lognormal_offsets():
    model = ModelStore()
    cfg = SearchPolicyConfig(
        max_steps=10, direction_mode="ALTERNATE", radius_scale=10000
    )
    band = model.get_band_for_p(500000)
    offsets = generate_lognormal_offsets(500000, band, cfg, seed=42)
    assert len(offsets) == 10
    # With large radius_scale, should have non-zero
    assert any(off != 0 for off in offsets)


def test_factor_with_candidate_prefilter():
    model = ModelStore()
    cfg = SearchPolicyConfig(max_steps=100, radius_scale=10000)
    # Small semiprime: 7*11=77
    factor = factor_with_candidate_prefilter(77, model, cfg, seed=42)
    assert factor in [7, 11]
