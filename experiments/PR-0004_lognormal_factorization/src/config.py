"""Configuration for search policy."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchPolicyConfig:
    """Configuration for the pre-filter search policy."""

    max_steps: int = 10000
    radius_scale: float = 1.0
    direction_mode: str = "ALTERNATE"  # "ALTERNATE" or "RANDOM"
    seed: Optional[int] = None

    def __post_init__(self):
        if self.direction_mode not in ["ALTERNATE", "RANDOM"]:
            raise ValueError("direction_mode must be 'ALTERNATE' or 'RANDOM'")
