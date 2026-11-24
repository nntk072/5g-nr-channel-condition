"""Deterministic seed rosters for bounded G4 native regressions."""

from __future__ import annotations

from collections.abc import Sequence


def cdl_temporal_velocity_seed_roster(
    profiles: Sequence[str],
    velocity_directions: Sequence[str],
    *,
    seed_base: int = 932_400,
) -> dict[tuple[str, str], int]:
    """Assign one unique consecutive seed to each ordered profile/direction pair."""
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise ValueError("seed_base must be a non-negative integer")
    if isinstance(profiles, (str, bytes)) or not isinstance(profiles, Sequence):
        raise TypeError("profiles must be a sequence of canonical strings")
    if isinstance(velocity_directions, (str, bytes)) or not isinstance(
        velocity_directions, Sequence
    ):
        raise TypeError("velocity_directions must be a sequence of canonical strings")
    if not profiles or not velocity_directions:
        raise ValueError("profiles and velocity_directions must be non-empty")
    if any(not isinstance(value, str) or not value or value != value.strip() for value in profiles):
        raise ValueError("profile names must be non-empty canonical strings")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for value in velocity_directions
    ):
        raise ValueError("velocity direction names must be non-empty canonical strings")
    if len(set(profiles)) != len(profiles):
        raise ValueError("profile names must be unique")
    if len(set(velocity_directions)) != len(velocity_directions):
        raise ValueError("velocity direction names must be unique")

    direction_count = len(velocity_directions)
    return {
        (profile, direction): seed_base + profile_index * direction_count + direction_index
        for profile_index, profile in enumerate(profiles)
        for direction_index, direction in enumerate(velocity_directions)
    }
