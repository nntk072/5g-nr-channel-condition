"""Torch-independent checks for retained G5 SNR-curve inference contracts."""

import pytest
from experiments.audit_g5_ordered_impairments_result import (
    _inference_seed_roster_consistent,
)


@pytest.mark.parametrize("version", (7, 8))
def test_acceptance_curve_audit_requires_the_exact_ordered_seed_roster(version: int) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert _inference_seed_roster_consistent(
        schema,
        {"channel_seeds": [101, 103, 107]},
        [101, 103, 107],
    )


@pytest.mark.parametrize(
    "inference_seeds",
    (
        [103, 101, 107],
        [101, 103, 109],
        [101, True, 107],
        [101, 103, 103],
        [101, 103, -1],
        "101,103,107",
    ),
)
@pytest.mark.parametrize("version", (7, 8))
def test_acceptance_curve_audit_rejects_malformed_or_different_roster(
    version: int, inference_seeds: object
) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert not _inference_seed_roster_consistent(
        schema,
        {"channel_seeds": inference_seeds},
        [101, 103, 107],
    )


@pytest.mark.parametrize("version", range(1, 7))
def test_historical_curve_audit_keeps_inference_shape_unchanged(version: int) -> None:
    schema = f"umcd.g5_ordered_impairments_independent_channels_snr_curve.v{version}"
    assert _inference_seed_roster_consistent(schema, {"cluster_count": 3}, [101, 103, 107])
    assert not _inference_seed_roster_consistent(
        schema,
        {"cluster_count": 3, "channel_seeds": [101, 103, 107]},
        [101, 103, 107],
    )
