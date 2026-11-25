"""Standardized channel admission: each rejected metadata combination and its declared reason."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from datasets.standardized import (
    StandardizedFamily,
    validate_standardized_channel_admission,
)
from tests.test_datasets.test_standardized import _channel, _metadata


def _admit(family, model, metadata, h_path=None, **kwargs):
    channel = _channel(model=model)
    validate_standardized_channel_admission(
        family,
        model,
        channel.h_path[0] if h_path is None else h_path,
        metadata,
        **kwargs,
    )


def test_family_must_be_the_standardized_enum() -> None:
    channel = _channel()
    with pytest.raises(TypeError, match="StandardizedFamily"):
        validate_standardized_channel_admission("tdl", "A", channel.h_path, channel.metadata)


def test_unknown_profile_names_the_family_specific_scope() -> None:
    channel = _channel()
    channel_h = channel.h_path[0]
    with pytest.raises(ValueError, match="TDL profile must be scalable A–E or fixed TS 38.101-4"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL, "Z", channel_h, channel.metadata
        )
    with pytest.raises(ValueError, match="CDL profile must be one of A–E"):
        validate_standardized_channel_admission(
            StandardizedFamily.CDL, "Z", channel_h, channel.metadata
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"delay_profile_semantics": "wrong"}, "delay profile semantics disagree"),
        ({"precision": "half"}, "TDL precision must be"),
        ({"profile_parameter_table_source": "x"}, "must appear together"),
        ({"profile_parameter_table_sha256": "a" * 64}, "must appear together"),
        (
            {
                "spec_version": "99.9",
                "profile_parameter_table_source": "TR 38.901:99.9:TDL-A.json",
                "profile_parameter_table_sha256": "a" * 64,
                "resolved_profile_parameters_sha256": "b" * 64,
            },
            "TDL resolved profile parameters are incomplete",
        ),
        (
            {
                "profile_parameter_table_source": "TR 38.901:19.2:TDL-A.json",
                "profile_parameter_table_sha256": "a" * 64,
            },
            "must be bound by the profile digest",
        ),
        (
            {"resolved_profile_parameters_sha256": "b" * 64},
            "TDL resolved profile parameters are incomplete",
        ),
        (
            {"resolved_profile_parameters_sha256": "B" * 64},
            "resolved TDL profile parameter digest is invalid",
        ),
        ({"fixed_profile_spec_version": "0.0", "spec_version": "0.0"}, "unsupported"),
    ],
)
def test_tdl_metadata_rejections(change, message) -> None:
    metadata = copy.deepcopy(_metadata(model="A"))
    metadata.update(change)
    with pytest.raises(ValueError, match=message):
        _admit(StandardizedFamily.TDL, "A", metadata)


def test_tdl_parameter_table_identity_must_belong_to_tdl_records() -> None:
    metadata = copy.deepcopy(_metadata(model="A"))
    metadata["resolved_profile_parameters_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="resolved TDL profile digest is attached to a non-TDL"):
        _admit(StandardizedFamily.CDL, "A", metadata)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"parameter_table_sha256": "A" * 64},
            "CDL parameter table digest must be lowercase SHA-256",
        ),
        (
            {
                "parameter_table_sha256": "a" * 64,
                "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
                "parameter_table": "wrong.json",
                "parameter_table_binding_sha256": "c" * 64,
            },
            "source identity is inconsistent",
        ),
        (
            {
                "parameter_table_sha256": None,
                "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
                "parameter_table": float("nan"),
            },
            "source identity is inconsistent",
        ),
    ],
)
def test_cdl_parameter_table_rejections(change, message) -> None:
    metadata = copy.deepcopy(_metadata(model="A"))
    metadata.update(change)
    with pytest.raises(ValueError, match=message):
        _admit(StandardizedFamily.CDL, "A", metadata)


def test_cdl_parameter_table_identity_is_accepted_when_fully_bound() -> None:
    from datasets.standardized import cdl_parameter_table_binding_sha256

    metadata = copy.deepcopy(_metadata(model="A"))
    metadata.update(
        {
            "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
            "parameter_table_sha256": "a" * 64,
            "parameter_table": "CDL-A.json",
        }
    )
    metadata["parameter_table_binding_sha256"] = cdl_parameter_table_binding_sha256(metadata)
    _admit(StandardizedFamily.CDL, "A", metadata)


def test_cdl_parameter_table_binding_rejects_non_json_numeric_values() -> None:
    from datasets.standardized_specs import cdl_parameter_table_binding_sha256

    with pytest.raises(ValueError, match="binding is incomplete or invalid"):
        cdl_parameter_table_binding_sha256(
            {
                "profile_model": "A",
                "spec_version": "19.2",
                "parameter_table": float("nan"),
                "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
                "parameter_table_sha256": "a" * 64,
            }
        )


def test_cdl_parameter_table_binding_rejects_noncanonical_digest() -> None:
    from datasets.standardized_specs import cdl_parameter_table_binding_sha256

    with pytest.raises(ValueError, match="binding is incomplete or invalid"):
        cdl_parameter_table_binding_sha256(
            {
                "profile_model": "A",
                "spec_version": "19.2",
                "parameter_table": "CDL-A.json",
                "parameter_table_source": "TR 38.901:19.2:CDL-A.json",
                "parameter_table_sha256": "A" * 64,
            }
        )


@pytest.mark.parametrize(
    "shape", [(2, 1, 2, 1, 1, 2, 2), (1, 1, 1, 1, 1, 2, 2), (2, 1, 1, 1, 1, 2, 2, 1)]
)
def test_link_shape_must_be_one_rx_and_one_tx(shape) -> None:
    channel = _channel()
    bad = np.zeros(shape, dtype=np.complex64)
    with pytest.raises(ValueError, match="one RX and one TX link"):
        _admit(StandardizedFamily.TDL, "A", channel.metadata, h_path=bad)


@pytest.mark.parametrize("carrier", [float("nan"), float("inf"), True, 1.0e12])
def test_carrier_frequency_must_be_finite_and_in_scope(carrier) -> None:
    metadata = copy.deepcopy(_metadata(model="A", carrier_hz=3.5e9))
    metadata["carrier_frequency_hz"] = carrier
    with pytest.raises(ValueError, match="carrier_frequency_hz must be within"):
        _admit(StandardizedFamily.TDL, "A", metadata)


def test_fixed_tdl_profile_is_refused_outside_its_fr1_or_fr2_carrier_range() -> None:
    metadata = copy.deepcopy(_metadata(model="A10", carrier_hz=3.5e9))
    metadata["fixed_profile_spec_version"] = "19.2.2"
    metadata["carrier_frequency_hz"] = 10e9  # between FR1 and FR2: outside A10's declared range
    with pytest.raises(ValueError, match="carrier_frequency_hz must be within 24.25–71 GHz"):
        _admit(StandardizedFamily.TDL, "A10", metadata)


def test_channel_admission_uses_the_declared_module_name_in_its_errors() -> None:
    channel = _channel()
    with pytest.raises(ValueError, match="standardized channel admission rejected"):
        validate_standardized_channel_admission(
            StandardizedFamily.TDL,
            "A",
            channel.h_path,
            {**channel.metadata, "precision": "fp16"},
        )
