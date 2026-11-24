"""Positioning SRS resource mapping, symbol offsets and hyper-SFN gating."""

import numpy as np
import pytest
from integrations.sionna.srs import (
    NRSRSPositioningMultiportResourceGrid,
    NRSRSPositioningResourceGrid,
    NRSRSTransmissionCombConfiguration,
    map_nr_srs_hopped_resource,
    map_nr_srs_positioning_multiport_resource,
    map_nr_srs_positioning_resource,
    nr_srs_fixed_frequency_allocation,
    nr_srs_periodic_resource_schedule,
    nr_srs_port_symbol_activity,
    nr_srs_positioning_symbol_offset,
    nr_srs_resource_frequency_allocations,
    nr_srs_transmission_comb_from_rrc,
    nr_srs_transmission_counter,
)


def test_positioning_srs_frequency_allocation_applies_hyper_sfn_gate() -> None:
    common = {
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
        "resource_type": "periodic",
        "absolute_slot_index": 0,
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
    }

    eligible = nr_srs_resource_frequency_allocations(
        **common, hyper_frame_number=3, hyper_sfn_index=1
    )
    ineligible = nr_srs_resource_frequency_allocations(
        **common, hyper_frame_number=3, hyper_sfn_index=0
    )

    assert eligible is not None
    assert ineligible is None


def test_positioning_srs_periodic_schedule_applies_hyper_sfn_gate() -> None:
    common = {
        "absolute_slot_index": 0,
        "c_srs": 9,
        "transmission_comb": 4,
        "b_srs": 3,
        "b_hop": 0,
        "srs_symbol_count": 2,
        "resource_type": "periodic",
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
        "hyper_frame_number": 3,
    }

    active = nr_srs_periodic_resource_schedule(**common, hyper_sfn_index=1)
    gated = nr_srs_periodic_resource_schedule(**common, hyper_sfn_index=0)

    assert active is not None
    assert active.target_slot_index == 0
    assert len(active.allocations) == 2
    assert gated is None


def test_srs_hopped_mapper_combines_hopping_and_positioning_offsets() -> None:
    allocations = tuple(
        nr_srs_fixed_frequency_allocation(
            55,
            transmission_comb=4,
            b_srs=3,
            b_hop=1,
            n_srs=counter,
        )
        for counter in range(4)
    )
    sequence = np.ones((4, allocations[0].bandwidth.sequence_length), dtype=np.complex128)

    mapped = map_nr_srs_hopped_resource(
        sequence,
        frequency_allocations=allocations,
        slot_symbol_count=14,
        subcarrier_count=1024,
        start_position_offset=3,
        srs_symbol_count=4,
        positioning_resource=True,
    )

    assert mapped.positioning_offsets == (0, 2, 1, 3)
    assert mapped.n_srs_by_symbol == (0, 1, 2, 3)
    expected_starts = (0, 194, 385, 99)
    for symbol, indices, first in zip(
        mapped.symbol_indices,
        mapped.subcarrier_indices_by_symbol,
        expected_starts,
        strict=True,
    ):
        assert indices == tuple(range(first, first + 96, 4))
        assert mapped.mask[symbol, list(indices)].all()


def test_positioning_srs_long_period_applies_hyper_sfn_gate() -> None:
    common = {
        "resource_type": "periodic",
        "srs_symbol_count": 2,
        "resource_symbol_index": 0,
        "absolute_slot_index": 0,
        "periodicity_slots": 20480,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 1,
    }

    assert nr_srs_transmission_counter(**common, hyper_frame_number=3, hyper_sfn_index=1) == 0
    assert nr_srs_transmission_counter(**common, hyper_frame_number=3, hyper_sfn_index=0) is None


def test_positioning_srs_hyper_sfn_configuration_is_validated_off_occasion() -> None:
    with pytest.raises(ValueError, match=r"at least 2\^mu \* 10240 slots"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=1,
            periodicity_slots=10240,
            slot_offset=0,
            positioning_resource=True,
            numerology=1,
            hyper_frame_number=0,
            hyper_sfn_index=0,
        )


def test_hyper_sfn_gate_is_positioning_only() -> None:
    with pytest.raises(ValueError, match="only defined for positioning SRS"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=0,
            periodicity_slots=10240,
            slot_offset=0,
            numerology=0,
            hyper_frame_number=0,
            hyper_sfn_index=0,
        )


@pytest.mark.parametrize("hyper_frame_number", [-1, 1024])
def test_positioning_srs_hyper_frame_number_uses_nr_range(hyper_frame_number) -> None:
    with pytest.raises(ValueError, match=r"hyper_frame_number must be in \[0, 1023\]"):
        nr_srs_transmission_counter(
            resource_type="periodic",
            srs_symbol_count=2,
            resource_symbol_index=0,
            absolute_slot_index=0,
            periodicity_slots=10240,
            slot_offset=0,
            positioning_resource=True,
            numerology=0,
            hyper_frame_number=hyper_frame_number,
            hyper_sfn_index=0,
        )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"numerology": True}, TypeError, "numerology must be an integer"),
        ({"numerology": 7}, ValueError, "numerology must be between 0 and 6"),
        (
            {"hyper_frame_number": True},
            TypeError,
            "hyper_frame_number must be a nonnegative integer",
        ),
        (
            {"hyper_frame_number": "0"},
            TypeError,
            "hyper_frame_number must be a nonnegative integer",
        ),
        ({"hyper_sfn_index": True}, TypeError, "hyper_sfn_index must be 0 or 1"),
        ({"hyper_sfn_index": 2}, ValueError, "hyper_sfn_index must be 0 or 1"),
    ],
)
def test_positioning_srs_hyper_sfn_fields_reject_invalid_types_and_ranges(
    changes, error, message
) -> None:
    arguments = {
        "resource_type": "periodic",
        "srs_symbol_count": 2,
        "resource_symbol_index": 0,
        "absolute_slot_index": 0,
        "periodicity_slots": 10240,
        "slot_offset": 0,
        "positioning_resource": True,
        "numerology": 0,
        "hyper_frame_number": 0,
        "hyper_sfn_index": 0,
    }
    arguments.update(changes)

    with pytest.raises(error, match=message):
        nr_srs_transmission_counter(**arguments)


@pytest.mark.parametrize(
    ("comb", "symbol_count", "expected"),
    [
        (2, 1, (0,)),
        (2, 2, (0, 1)),
        (2, 4, (0, 1, 0, 1)),
        (4, 2, (0, 2)),
        (4, 4, (0, 2, 1, 3)),
        (4, 8, (0, 2, 1, 3, 0, 2, 1, 3)),
        (4, 12, (0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3)),
        (8, 4, (0, 4, 2, 6)),
        (8, 8, (0, 4, 2, 6, 1, 5, 3, 7)),
        (8, 12, (0, 4, 2, 6, 1, 5, 3, 7, 0, 4, 2, 6)),
    ],
)
def test_srs_positioning_symbol_offsets_match_all_supported_table_rows(
    comb, symbol_count, expected
) -> None:
    actual = tuple(
        nr_srs_positioning_symbol_offset(
            comb,
            srs_symbol_count=symbol_count,
            resource_symbol_index=index,
        )
        for index in range(symbol_count)
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("comb", "symbol_count", "symbol_index", "message"),
    [
        (2, 8, 0, "unsupported"),
        (4, 1, 0, "unsupported"),
        (8, 2, 0, "unsupported"),
        (4, 4, 4, "inside"),
    ],
)
def test_srs_positioning_symbol_offset_rejects_unsupported_table_cases(
    comb, symbol_count, symbol_index, message
) -> None:
    with pytest.raises(ValueError, match=message):
        nr_srs_positioning_symbol_offset(
            comb,
            srs_symbol_count=symbol_count,
            resource_symbol_index=symbol_index,
        )


def test_srs_positioning_mapper_applies_distinct_comb_offsets_by_symbol() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55,
        transmission_comb=4,
        b_srs=3,
        b_hop=3,
        frequency_domain_position=7,
    )
    sequence = (
        np.arange(4 * allocation.bandwidth.sequence_length, dtype=np.float64)
        .reshape(4, allocation.bandwidth.sequence_length)
        .astype(np.complex128)
        + 1j
    )

    mapped = map_nr_srs_positioning_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=2,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
    )

    assert isinstance(mapped, NRSRSPositioningResourceGrid)
    assert mapped.symbol_indices == (10, 11, 12, 13)
    assert mapped.positioning_offsets == (0, 2, 1, 3)
    expected_first_subcarriers = (290, 288, 291, 289)
    for row, (symbol, indices, first) in enumerate(
        zip(
            mapped.symbol_indices,
            mapped.subcarrier_indices_by_symbol,
            expected_first_subcarriers,
            strict=True,
        )
    ):
        assert indices == tuple(range(first, first + 96, 4))
        np.testing.assert_array_equal(mapped.grid[symbol, list(indices)], sequence[row])
        assert mapped.mask[symbol, list(indices)].all()
    assert int(np.count_nonzero(mapped.mask)) == 4 * allocation.bandwidth.sequence_length
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


def test_srs_positioning_mapper_broadcasts_one_sequence_row_and_applies_beta() -> None:
    allocation = nr_srs_fixed_frequency_allocation(55, transmission_comb=4, b_srs=3, b_hop=3)
    sequence = np.full(allocation.bandwidth.sequence_length, 1.0 + 2.0j)
    mapped = map_nr_srs_positioning_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=0,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
        beta_srs=0.5,
    )

    for symbol, indices in zip(
        mapped.symbol_indices, mapped.subcarrier_indices_by_symbol, strict=True
    ):
        np.testing.assert_array_equal(mapped.grid[symbol, list(indices)], 0.5 + 1.0j)
    assert int(np.count_nonzero(mapped.mask)) == 4 * allocation.bandwidth.sequence_length


def test_srs_positioning_multiport_mapper_keeps_port_and_symbol_rosters() -> None:
    allocation = nr_srs_fixed_frequency_allocation(55, transmission_comb=4, b_srs=3, b_hop=3)
    sequences = np.stack(
        [
            np.full((4, allocation.bandwidth.sequence_length), 1.0 + 1.0j),
            np.full((4, allocation.bandwidth.sequence_length), 2.0 - 1.0j),
        ]
    )

    mapped = map_nr_srs_positioning_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=4,
        comb_offset=0,
        frequency_start_subcarrier=allocation.start_subcarrier,
        allocated_bandwidth_subcarriers=allocation.bandwidth.allocated_bandwidth_subcarriers,
    )

    assert isinstance(mapped, NRSRSPositioningMultiportResourceGrid)
    assert mapped.grid.shape == (2, 14, 512)
    assert mapped.symbol_indices == (10, 11, 12, 13)
    assert mapped.positioning_offsets == (0, 2, 1, 3)
    for port, sequence_rows in enumerate(sequences):
        for row, symbol in enumerate(mapped.symbol_indices):
            indices = mapped.subcarrier_indices_by_symbol[row]
            np.testing.assert_array_equal(
                mapped.grid[port, symbol, list(indices)], sequence_rows[row] / np.sqrt(2)
            )
            assert mapped.mask[port, symbol, list(indices)].all()
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


def test_srs_positioning_multiport_mapper_applies_eight_port_tdm() -> None:
    sequences = np.ones((8, 2, 36), dtype=np.complex128)
    mapped = map_nr_srs_positioning_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=72,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
        frequency_start_subcarrier=0,
        allocated_bandwidth_subcarriers=72,
        ports8_tdm=True,
    )

    assert mapped.symbol_indices == (12, 13)
    assert mapped.active_port_symbols == nr_srs_port_symbol_activity(
        tuple(range(1000, 1008)), srs_symbol_count=2, ports8_tdm=True
    )
    assert mapped.mask[:, 12, :].sum(axis=1).tolist() == [36, 36, 0, 0, 36, 36, 0, 0]
    assert mapped.mask[:, 13, :].sum(axis=1).tolist() == [0, 0, 36, 36, 0, 0, 36, 36]
    assert np.all(mapped.grid[~mapped.mask] == 0)
    assert not mapped.grid.flags.writeable


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"srs_symbol_count": 10}, ValueError, "symbol count is unsupported"),
        (
            {"comb_size": 8, "srs_symbol_count": 2},
            ValueError,
            "unsupported for this SRS-PosResource comb",
        ),
        ({"allocated_bandwidth_subcarriers": 95}, ValueError, "comb-aligned"),
        ({"start_position_offset": 2}, ValueError, "start_position_offset"),
    ],
)
def test_srs_positioning_mapper_rejects_invalid_configuration(overrides, error, message) -> None:
    allocation = nr_srs_fixed_frequency_allocation(55, transmission_comb=4, b_srs=3, b_hop=3)
    config = {
        "slot_symbol_count": 14,
        "subcarrier_count": 512,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 4,
        "comb_offset": 0,
        "frequency_start_subcarrier": allocation.start_subcarrier,
        "allocated_bandwidth_subcarriers": allocation.bandwidth.allocated_bandwidth_subcarriers,
    }
    config.update(overrides)
    with pytest.raises(error, match=message):
        map_nr_srs_positioning_resource(
            np.ones((config["srs_symbol_count"], 24), dtype=np.complex128), **config
        )


def test_srs_positioning_transmission_comb_adapter_handles_r16_names() -> None:
    assert nr_srs_transmission_comb_from_rrc(
        {"n8-r16": {"combOffset-n8-r16": 6, "cyclicShift-n8-r16": 4}},
        positioning_resource=True,
    ) == NRSRSTransmissionCombConfiguration(8, 6, 4)
    with pytest.raises(ValueError, match="choice must be n2, n4, or n8"):
        nr_srs_transmission_comb_from_rrc(
            {"n8": {"combOffset-n8": 0, "cyclicShift-n8": 0}},
            positioning_resource=True,
        )
