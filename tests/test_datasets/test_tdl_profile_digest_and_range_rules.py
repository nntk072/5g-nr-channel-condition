"""TDL resolved-profile digest and carrier-range rules: source identity pairing, precision and FR ranges."""

from __future__ import annotations

import pytest
from datasets.standardized_specs import (
    carrier_frequency_range,
    tdl_resolved_profile_parameter_sha256,
)

DIGEST = "a" * 64


def _tdl_metadata(**overrides):
    metadata = {
        "profile_proxy": "tdl-a",
        "spec_version": "19.2",
        "fixed_profile_spec_version": "19.2.2",
        "delay_profile_semantics": "absolute",
        "profile_rms_delay_spread_s": 3.0e-8,
        "profile_los_component": False,
        "num_profile_taps": 2,
        "nominal_tap_delays_s": [0.0, 1.0e-7],
        "mean_tap_powers": [0.7, 0.3],
        "k_factor_linear": 0.0,
    }
    metadata.update(overrides)
    return metadata


def test_profile_source_identity_and_digest_must_appear_together() -> None:
    with pytest.raises(
        ValueError, match="resolved profile parameters are incomplete or non-finite"
    ):
        tdl_resolved_profile_parameter_sha256(
            _tdl_metadata(profile_parameter_table_source="TR 38.901:19.2:TDL-A.json")
        )
    with pytest.raises(
        ValueError, match="resolved profile parameters are incomplete or non-finite"
    ):
        tdl_resolved_profile_parameter_sha256(_tdl_metadata(profile_parameter_table_sha256=DIGEST))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"precision": "half"},
            "resolved profile parameters are incomplete or non-finite",
            id="precision",
        ),
        pytest.param(
            {"precision": 32},
            "resolved profile parameters are incomplete or non-finite",
            id="precision-type",
        ),
        pytest.param(
            {"profile_parameter_table_source": "", "profile_parameter_table_sha256": DIGEST},
            "resolved profile parameters are incomplete or non-finite",
            id="empty-source",
        ),
        pytest.param(
            {
                "profile_parameter_table_source": "TR 38.901:19.2:TDL-A.json",
                "profile_parameter_table_sha256": "A" * 64,
            },
            "resolved profile parameters are incomplete or non-finite",
            id="uppercase-digest",
        ),
    ],
)
def test_profile_digest_refuses_invalid_precision_and_identity(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        tdl_resolved_profile_parameter_sha256(_tdl_metadata(**overrides))


def test_precision_and_source_identity_are_bound_into_the_digest() -> None:
    base = tdl_resolved_profile_parameter_sha256(_tdl_metadata())
    single = tdl_resolved_profile_parameter_sha256(_tdl_metadata(precision="single"))
    assert single != base
    bound = tdl_resolved_profile_parameter_sha256(
        _tdl_metadata(
            profile_parameter_table_source="TR 38.901:19.2:TDL-A.json",
            profile_parameter_table_sha256=DIGEST,
        )
    )
    assert bound != base


@pytest.mark.parametrize(
    ("carrier_hz", "expected"),
    [(3.5e9, "FR1"), (28e9, "FR2"), (10e9, None), (100e9, None)],
    ids=["fr1", "fr2", "between-ranges", "above-fr2"],
)
def test_carrier_frequency_ranges(carrier_hz, expected) -> None:
    assert carrier_frequency_range(carrier_hz) == expected
