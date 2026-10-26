"""Dependency-free seed-roster checks for bounded G4 regressions."""

from __future__ import annotations

import pytest
from evaluation.g4_seed_roster import cdl_temporal_velocity_seed_roster


def test_cdl_temporal_velocity_roster_assigns_unique_consecutive_seeds() -> None:
    profiles = ("A", "B", "C", "D", "E")
    directions = tuple(f"direction_{index:02d}" for index in range(16))

    roster = cdl_temporal_velocity_seed_roster(profiles, directions)

    seeds = tuple(roster.values())
    assert len(roster) == 80
    assert len(set(seeds)) == 80
    assert seeds == tuple(range(932_400, 932_480))
    assert roster[("A", directions[15])] == 932_415
    assert roster[("B", directions[0])] == 932_416
    assert roster[("E", directions[15])] == 932_479


@pytest.mark.parametrize(
    ("profiles", "directions", "seed_base"),
    [
        ((), ("x",), 1),
        (("A",), (), 1),
        (("A", "A"), ("x",), 1),
        (("A",), ("x", "x"), 1),
        ((" ",), ("x",), 1),
        ((" A",), ("x",), 1),
        (("A",), ("x ",), 1),
        (("A",), ("x",), -1),
        (("A",), ("x",), True),
    ],
)
def test_cdl_temporal_velocity_roster_rejects_invalid_inputs(
    profiles: tuple[str, ...], directions: tuple[str, ...], seed_base: int
) -> None:
    with pytest.raises(ValueError):
        cdl_temporal_velocity_seed_roster(profiles, directions, seed_base=seed_base)


@pytest.mark.parametrize(
    ("profiles", "directions"),
    [
        ("ABCDE", ("x",)),
        (("A",), "xyz"),
        (b"AB", ("x",)),
        (("A",), 1),
    ],
)
def test_cdl_temporal_velocity_roster_rejects_non_sequence_rosters(
    profiles: object, directions: object
) -> None:
    with pytest.raises(TypeError, match="sequence of canonical strings"):
        cdl_temporal_velocity_seed_roster(profiles, directions)  # type: ignore[arg-type]
