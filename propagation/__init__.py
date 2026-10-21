"""Channel propagation physics, geometry, and observable transforms."""

from .physics import (
    add_estimation_noise,
    frequency_response,
    observable_features,
    subcarrier_frequencies,
)

__all__ = [
    "add_estimation_noise",
    "frequency_response",
    "observable_features",
    "subcarrier_frequencies",
]
