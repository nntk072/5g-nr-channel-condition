"""Resolved SRS configuration records: comb, bandwidth and fixed-allocation consistency with TS 38.211."""

from __future__ import annotations

import dataclasses

import pytest
from integrations.sionna.srs.srs_configuration import (
    NRSRSBandwidthConfiguration,
    NRSRSFixedFrequencyAllocation,
    NRSRSTransmissionCombConfiguration,
    nr_srs_type1_bandwidth_configuration,
)


def _comb(**overrides):
    values = {"transmission_comb": 4, "comb_offset": 1, "base_cyclic_shift": 0}
    values.update(overrides)
    return NRSRSTransmissionCombConfiguration(**values)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param(
            {"transmission_comb": True},
            TypeError,
            "transmission_comb must be an integer",
            id="comb-type",
        ),
        pytest.param(
            {"comb_offset": 4}, ValueError, "comb_offset must lie inside", id="offset-range"
        ),
        pytest.param(
            {"base_cyclic_shift": 12}, ValueError, "base_cyclic_shift must be in", id="shift-range"
        ),
    ],
)
def test_comb_configuration_rejects_out_of_range_fields(overrides, error, message) -> None:
    with pytest.raises(error, match=message):
        _comb(**overrides)


def test_bandwidth_configuration_is_cross_checked_against_its_table_row() -> None:
    config = nr_srs_type1_bandwidth_configuration(0, transmission_comb=2, b_srs=0)
    with pytest.raises(ValueError, match="subband_count disagrees"):
        dataclasses.replace(config, subband_count=config.subband_count + 1)
    with pytest.raises(ValueError, match="allocated bandwidth disagrees"):
        dataclasses.replace(
            config, allocated_bandwidth_subcarriers=config.allocated_bandwidth_subcarriers + 2
        )
    with pytest.raises(ValueError, match="sequence_length disagrees"):
        dataclasses.replace(config, sequence_length=config.sequence_length + 1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"c_srs": 64}, "c_srs must be in", id="c-srs"),
        pytest.param({"b_srs": 4}, "b_srs must be in", id="b-srs"),
    ],
)
def test_bandwidth_configuration_rejects_indices_outside_the_table(overrides, message) -> None:
    config = nr_srs_type1_bandwidth_configuration(0, transmission_comb=2, b_srs=0)
    with pytest.raises(ValueError, match=message):
        dataclasses.replace(config, **overrides)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"c_srs": True}, TypeError, "c_srs must be an integer"),
        ({"b_srs": True}, TypeError, "b_srs must be an integer"),
        ({"transmission_comb": True}, TypeError, "transmission_comb must be an integer"),
    ],
)
def test_bandwidth_configuration_rejects_invalid_public_field_types(
    overrides, error, message
) -> None:
    config = nr_srs_type1_bandwidth_configuration(0, transmission_comb=2, b_srs=0)
    with pytest.raises(error, match=message):
        dataclasses.replace(config, **overrides)


def test_bandwidth_configuration_rejects_boolean_nonindex_field() -> None:
    with pytest.raises(TypeError, match="bandwidth_rb must be an integer"):
        NRSRSBandwidthConfiguration(
            c_srs=0,
            b_srs=0,
            transmission_comb=2,
            bandwidth_rb=True,
            subband_count=1,
            sequence_length=48,
            allocated_bandwidth_subcarriers=96,
        )


def test_fixed_allocation_must_carry_a_bandwidth_record_and_consistent_positions() -> None:
    bandwidth = nr_srs_type1_bandwidth_configuration(0, transmission_comb=2, b_srs=0)
    base = {
        "bandwidth": bandwidth,
        "position_indices": (0,),
        "frequency_domain_position": 0,
        "reference_subcarrier": 0,
        "frequency_domain_shift_rb": 0,
        "comb_offset": 0,
        "start_subcarrier": 0,
    }
    NRSRSFixedFrequencyAllocation(**base)
    with pytest.raises(TypeError, match="must be an NRSRSBandwidthConfiguration"):
        NRSRSFixedFrequencyAllocation(**{**base, "bandwidth": "not-a-bandwidth"})
    with pytest.raises(ValueError, match="one entry per bandwidth level"):
        NRSRSFixedFrequencyAllocation(**{**base, "position_indices": (0, 0)})
    with pytest.raises(ValueError, match="nonnegative integers|must be nonnegative"):
        NRSRSFixedFrequencyAllocation(**{**base, "position_indices": (-1,)})
    with pytest.raises(ValueError, match="comb_offset must be smaller"):
        NRSRSFixedFrequencyAllocation(**{**base, "comb_offset": 2})
    with pytest.raises(ValueError, match="n_srs must be nonnegative"):
        NRSRSFixedFrequencyAllocation(**{**base, "n_srs": -1})


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"frequency_domain_position": True}, TypeError, "must be an integer"),
        ({"reference_subcarrier": -1}, ValueError, "must be nonnegative"),
        (
            {"position_indices": (True,)},
            ValueError,
            "position indices must be nonnegative integers",
        ),
        (
            {"position_indices": (1,)},
            ValueError,
            "disagree with the resolved SRS frequency position",
        ),
        ({"start_subcarrier": 1}, ValueError, "start_subcarrier disagrees"),
    ],
)
def test_fixed_allocation_rejects_invalid_frequency_fields(overrides, error, message) -> None:
    bandwidth = nr_srs_type1_bandwidth_configuration(0, transmission_comb=2, b_srs=0)
    base = {
        "bandwidth": bandwidth,
        "position_indices": (0,),
        "frequency_domain_position": 0,
        "reference_subcarrier": 0,
        "frequency_domain_shift_rb": 0,
        "comb_offset": 0,
        "start_subcarrier": 0,
    }
    with pytest.raises(error, match=message):
        NRSRSFixedFrequencyAllocation(**{**base, **overrides})
