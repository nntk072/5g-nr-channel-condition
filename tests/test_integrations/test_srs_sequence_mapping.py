"""SRS sequences, bandwidth tables, cyclic shifts and multiport grid mapping."""

import numpy as np
import pytest
from integrations.sionna.srs import (
    NRSRSBandwidthConfiguration,
    NRSRSHoppedMultiportResourceGrid,
    NRSRSMultiportResourceGrid,
    NRSRSPortConfiguration,
    NRSRSResourceGrid,
    NRSRSScheduledResourceOccurrence,
    NRSRSTransmissionCombConfiguration,
    audit_nr_srs_intercell_re_overlap,
    map_nr_srs_hopped_multiport_resource,
    map_nr_srs_hopped_resource,
    map_nr_srs_multiport_resource,
    map_nr_srs_resource,
    nr_low_papr_sequence_type1,
    nr_srs_cyclic_shift_hopping_offsets,
    nr_srs_fixed_frequency_allocation,
    nr_srs_group_hopping_index,
    nr_srs_hopped_sequence_type1,
    nr_srs_multiport_ls_channel_estimate,
    nr_srs_port_configuration_from_rrc,
    nr_srs_port_cyclic_shift_indices,
    nr_srs_port_sequences_type1,
    nr_srs_port_symbol_activity,
    nr_srs_resource_sequences_type1,
    nr_srs_sequence_hopping_index,
    nr_srs_sequence_type1,
    nr_srs_single_port_ls_channel_estimate,
    nr_srs_transmission_comb_from_rrc,
    nr_srs_type1_bandwidth_configuration,
    srs_cyclic_shift_count,
)
from tests.test_integrations.srs_resource_mapping_support import (
    _reference_low_papr_phase,
)


def test_srs_hopped_multiport_mapper_preserves_per_port_symbols_and_offsets() -> None:
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
    sequence_length = allocations[0].bandwidth.sequence_length
    sequences = np.stack(
        [
            np.full((4, sequence_length), 1.0 + 1.0j),
            np.full((4, sequence_length), 2.0 - 1.0j),
        ]
    )

    mapped = map_nr_srs_hopped_multiport_resource(
        sequences,
        frequency_allocations=allocations,
        slot_symbol_count=14,
        subcarrier_count=1024,
        start_position_offset=3,
        srs_symbol_count=4,
        positioning_resource=True,
    )

    assert isinstance(mapped, NRSRSHoppedMultiportResourceGrid)
    assert mapped.grid.shape == (2, 14, 1024)
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
        for port, value in enumerate((1.0 + 1.0j, 2.0 - 1.0j)):
            np.testing.assert_array_equal(
                mapped.grid[port, symbol, list(indices)],
                np.full(len(indices), value / np.sqrt(2), dtype=np.complex128),
            )
            assert mapped.mask[port, symbol, list(indices)].all()
    assert not mapped.grid.flags.writeable
    assert not mapped.mask.flags.writeable


def test_hopped_positioning_mapper_applies_per_symbol_transmission_comb_offsets() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=0, n_srs=0
    )
    mapped = map_nr_srs_hopped_resource(
        np.ones(allocation.bandwidth.sequence_length, dtype=np.complex128),
        frequency_allocations=(allocation,),
        slot_symbol_count=14,
        subcarrier_count=512,
        start_position_offset=3,
        srs_symbol_count=4,
        positioning_resource=True,
    )

    assert mapped.positioning_offsets == (0, 2, 1, 3)
    first_subcarriers = tuple(indices[0] for indices in mapped.subcarrier_indices_by_symbol)
    assert first_subcarriers == tuple(
        allocation.start_subcarrier + offset for offset in mapped.positioning_offsets
    )


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        ({"positioning_resource": 1}, TypeError, "positioning_resource must be a boolean"),
        ({"frequency_allocations": ()}, TypeError, "must contain resolved SRS allocation"),
        ({"frequency_allocations": (None,)}, TypeError, "must contain resolved SRS allocation"),
        (
            {
                "frequency_allocations": (
                    nr_srs_fixed_frequency_allocation(
                        55, transmission_comb=4, b_srs=3, b_hop=1, n_srs=0
                    ),
                )
                * 2
            },
            ValueError,
            "one or one-per-symbol",
        ),
        ({"slot_symbol_count": 13}, ValueError, "slot_symbol_count must be 12 or 14"),
        ({"subcarrier_count": 0}, ValueError, "subcarrier_count must be positive"),
        ({"srs_symbol_count": 3}, ValueError, "unsupported SRS symbol count"),
        ({"start_position_offset": 2}, ValueError, "start_position_offset must fit"),
        ({"beta_srs": 0.0}, ValueError, "beta_srs must be a finite positive"),
        ({"subcarrier_count": 1}, ValueError, "does not fit inside the subcarrier grid"),
        (
            {"sequence": np.ones(24, dtype=np.float64)},
            TypeError,
            "sequence must be complex-valued",
        ),
        (
            {"sequence": np.ones((1, 1, 24), dtype=np.complex128)},
            ValueError,
            "sequence must have shape",
        ),
        (
            {"sequence": np.ones(23, dtype=np.complex128)},
            ValueError,
            "sequence length does not match",
        ),
        (
            {"sequence": np.ones((3, 24), dtype=np.complex128)},
            ValueError,
            "exactly srs_symbol_count rows",
        ),
        (
            {"sequence": np.full(24, np.nan + 0j, dtype=np.complex128)},
            ValueError,
            "only finite values",
        ),
    ],
)
def test_hopped_mapper_rejects_invalid_allocations_geometry_and_sequences(
    changes, error, message
) -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=1, n_srs=0
    )
    arguments = {
        "frequency_allocations": (allocation,),
        "slot_symbol_count": 14,
        "subcarrier_count": 1024,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
    }
    sequence = changes.get(
        "sequence", np.ones(allocation.bandwidth.sequence_length, dtype=np.complex128)
    )
    arguments.update({key: value for key, value in changes.items() if key != "sequence"})

    with pytest.raises(error, match=message):
        map_nr_srs_hopped_resource(sequence, **arguments)


def test_hopped_mapper_rejects_allocations_from_different_resource_configurations() -> None:
    allocations = (
        nr_srs_fixed_frequency_allocation(55, transmission_comb=4, b_srs=3, b_hop=1, n_srs=0),
        nr_srs_fixed_frequency_allocation(
            55,
            transmission_comb=4,
            b_srs=3,
            b_hop=1,
            frequency_domain_position=1,
            n_srs=1,
        ),
    )
    with pytest.raises(ValueError, match="share one SRS configuration"):
        map_nr_srs_hopped_resource(
            np.ones(allocations[0].bandwidth.sequence_length, dtype=np.complex128),
            frequency_allocations=allocations,
            slot_symbol_count=14,
            subcarrier_count=1024,
            start_position_offset=1,
            srs_symbol_count=2,
        )


@pytest.mark.parametrize(
    ("sequences", "error", "message"),
    [
        (np.ones((2, 2, 24)), TypeError, "complex-valued"),
        (np.ones((2, 2, 24), dtype=np.complex128)[:, :, :23], ValueError, "sequence length"),
        (np.ones((2, 1, 24), dtype=np.complex128), ValueError, "exactly srs_symbol_count rows"),
    ],
)
def test_hopped_multiport_mapper_rejects_invalid_port_sequences(
    sequences: np.ndarray, error: type[Exception], message: str
) -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=0, n_srs=0
    )
    with pytest.raises(error, match=message):
        map_nr_srs_hopped_multiport_resource(
            sequences,
            frequency_allocations=(allocation,),
            slot_symbol_count=14,
            subcarrier_count=512,
            start_position_offset=1,
            srs_symbol_count=2,
        )


def test_hopped_multiport_mapper_rejects_wrong_sequence_rank() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=0, n_srs=0
    )
    with pytest.raises(ValueError, match=r"shape \[ports, SRS symbols, occupied REs\]"):
        map_nr_srs_hopped_multiport_resource(
            np.ones((2, allocation.bandwidth.sequence_length), dtype=np.complex128),
            frequency_allocations=(allocation,),
            slot_symbol_count=14,
            subcarrier_count=512,
            start_position_offset=1,
            srs_symbol_count=2,
        )


def test_hopped_mapper_rejects_noninteger_grid_geometry() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=0, n_srs=0
    )
    with pytest.raises(TypeError, match="start_position_offset must be an integer"):
        map_nr_srs_hopped_resource(
            np.ones(allocation.bandwidth.sequence_length, dtype=np.complex128),
            frequency_allocations=(allocation,),
            slot_symbol_count=14,
            subcarrier_count=512,
            start_position_offset=1.5,
            srs_symbol_count=1,
        )


@pytest.mark.parametrize("transmission_comb", [2, 4, 8])
def test_srs_type1_bandwidth_table_b_srs_zero_matches_all_64_c_srs_rows(
    transmission_comb,
) -> None:
    # TS 38.211 V19.2.0 Table 6.4.1.4.3-1, m_SRS,0 (b_SRS=0).
    expected_bandwidth_rb = (
        4,
        8,
        12,
        16,
        16,
        20,
        24,
        24,
        28,
        32,
        36,
        40,
        48,
        48,
        52,
        56,
        60,
        64,
        72,
        72,
        76,
        80,
        88,
        96,
        96,
        104,
        112,
        120,
        120,
        120,
        128,
        128,
        128,
        132,
        136,
        144,
        144,
        144,
        144,
        152,
        160,
        160,
        160,
        168,
        176,
        184,
        192,
        192,
        192,
        192,
        208,
        216,
        224,
        240,
        240,
        240,
        240,
        256,
        256,
        256,
        264,
        272,
        272,
        272,
    )
    for c_srs, bandwidth_rb in enumerate(expected_bandwidth_rb):
        config = nr_srs_type1_bandwidth_configuration(c_srs, transmission_comb=transmission_comb)
        assert isinstance(config, NRSRSBandwidthConfiguration)
        assert config.c_srs == c_srs
        assert config.b_srs == 0
        assert config.transmission_comb == transmission_comb
        assert config.bandwidth_rb == bandwidth_rb
        assert config.subband_count == 1
        assert config.allocated_bandwidth_subcarriers == 12 * bandwidth_rb
        assert config.sequence_length == 12 * bandwidth_rb // transmission_comb


@pytest.mark.parametrize(
    ("c_srs", "b_srs", "bandwidth_rb", "subband_count"),
    [
        (0, 0, 4, 1),
        (0, 1, 4, 1),
        (0, 2, 4, 1),
        (0, 3, 4, 1),
        (9, 1, 16, 2),
        (9, 2, 8, 2),
        (9, 3, 4, 2),
        (13, 1, 24, 2),
        (13, 2, 12, 2),
        (13, 3, 4, 3),
        (36, 1, 48, 3),
        (36, 2, 24, 2),
        (36, 3, 12, 2),
        (49, 1, 24, 8),
        (49, 2, 8, 3),
        (49, 3, 4, 2),
        (55, 1, 48, 5),
        (55, 2, 16, 3),
        (55, 3, 8, 2),
        (61, 1, 136, 2),
        (61, 2, 68, 2),
        (61, 3, 4, 17),
        (62, 1, 68, 4),
        (62, 2, 4, 17),
        (62, 3, 4, 1),
        (63, 1, 16, 17),
        (63, 2, 8, 2),
        (63, 3, 4, 2),
    ],
)
def test_srs_type1_bandwidth_table_resolves_all_b_srs_levels(
    c_srs, b_srs, bandwidth_rb, subband_count
) -> None:
    config = nr_srs_type1_bandwidth_configuration(c_srs, b_srs=b_srs, transmission_comb=4)

    assert config.bandwidth_rb == bandwidth_rb
    assert config.subband_count == subband_count
    assert config.allocated_bandwidth_subcarriers == 12 * bandwidth_rb
    assert config.sequence_length == 3 * bandwidth_rb


@pytest.mark.parametrize(
    ("c_srs", "comb", "error", "message"),
    [
        (True, 2, TypeError, "c_srs"),
        (-1, 2, ValueError, "c_srs"),
        (64, 2, ValueError, "c_srs"),
        (0, 3, ValueError, "transmission_comb"),
    ],
)
def test_srs_type1_bandwidth_configuration_rejects_invalid_inputs(
    c_srs, comb, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_type1_bandwidth_configuration(c_srs, transmission_comb=comb)


@pytest.mark.parametrize(
    ("b_srs", "error"),
    [(True, TypeError), (1.0, TypeError), (-1, ValueError), (4, ValueError)],
)
def test_srs_type1_bandwidth_configuration_rejects_invalid_b_srs(b_srs, error) -> None:
    with pytest.raises(error, match="b_srs"):
        nr_srs_type1_bandwidth_configuration(0, b_srs=b_srs, transmission_comb=2)


def test_srs_type1_bandwidth_configuration_rejects_inconsistent_public_value() -> None:
    with pytest.raises(ValueError, match="standardized c_srs row"):
        NRSRSBandwidthConfiguration(
            c_srs=0,
            b_srs=0,
            transmission_comb=2,
            bandwidth_rb=8,
            subband_count=1,
            sequence_length=48,
            allocated_bandwidth_subcarriers=96,
        )


def test_srs_mapper_respects_allocated_bandwidth_inside_wider_grid() -> None:
    sequence = np.exp(1j * np.arange(108, dtype=np.float64) * 0.1)
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=600,
        start_position_offset=0,
        srs_symbol_count=1,
        comb_size=4,
        comb_offset=1,
        frequency_start_subcarrier=24,
        allocated_bandwidth_subcarriers=432,
    )

    assert result.subcarrier_indices == tuple(range(25, 456, 4))
    assert len(result.subcarrier_indices) == len(sequence)
    assert not result.mask[:, :25].any()
    assert not result.mask[:, 456:].any()
    np.testing.assert_array_equal(result.grid[13, 25:456:4], sequence)


def test_srs_fixed_bandwidth_sequences_map_to_exactly_the_resolved_extent() -> None:
    bandwidth = nr_srs_type1_bandwidth_configuration(9, transmission_comb=4)
    sequences = nr_srs_port_sequences_type1(
        bandwidth.sequence_length,
        cyclic_shift_indices=(0, 1),
        srs_symbol_count=2,
        sequence_id=21,
        slot_index=3,
        start_symbol_index=10,
        hopping_mode="neither",
        transmission_comb=4,
    )
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=600,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=bandwidth.transmission_comb,
        comb_offset=2,
        frequency_start_subcarrier=48,
        allocated_bandwidth_subcarriers=bandwidth.allocated_bandwidth_subcarriers,
    )

    assert sequences.shape == (2, 2, 96)
    assert mapped.grid.shape == (2, 14, 600)
    assert mapped.subcarrier_indices == tuple(range(50, 434, 4))
    assert len(mapped.subcarrier_indices) == bandwidth.sequence_length
    assert not mapped.mask[:, :, :50].any()
    assert not mapped.mask[:, :, 434:].any()


def test_srs_mapper_rejects_allocated_bandwidth_outside_grid() -> None:
    with pytest.raises(ValueError, match="allocated bandwidth"):
        map_nr_srs_resource(
            np.ones(24, dtype=np.complex128),
            slot_symbol_count=14,
            subcarrier_count=48,
            start_position_offset=1,
            srs_symbol_count=2,
            comb_size=2,
            comb_offset=0,
            frequency_start_subcarrier=24,
            allocated_bandwidth_subcarriers=25,
        )


def test_srs_multiport_sequences_use_explicit_cyclic_shifts_and_are_orthogonal() -> None:
    sequences = nr_srs_port_sequences_type1(
        72,
        cyclic_shift_indices=(0, 1, 5),
        srs_symbol_count=2,
        sequence_id=37,
        slot_index=4,
        start_symbol_index=10,
        hopping_mode="groupHopping",
        transmission_comb=4,
    )

    assert sequences.shape == (3, 2, 72)
    for symbol_index in range(2):
        for left_port in range(3):
            for right_port in range(left_port + 1, 3):
                cross_correlation = np.vdot(
                    sequences[left_port, symbol_index],
                    sequences[right_port, symbol_index],
                )
                assert abs(cross_correlation) < 1e-10


@pytest.mark.parametrize(
    ("ports", "comb", "base", "ports8_tdm", "expected"),
    [
        ((1000, 1001), 2, 3, False, (3, 7)),
        ((1000, 1001, 1002, 1003), 4, 2, False, (2, 5, 8, 11)),
        ((1000, 1001, 1002, 1003), 8, 1, False, (1, 1, 4, 4)),
        (
            (1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007),
            2,
            0,
            True,
            (0, 2, 0, 2, 4, 6, 4, 6),
        ),
    ],
)
def test_srs_port_cyclic_shift_roster_matches_standard_mapping(
    ports, comb, base, ports8_tdm, expected
) -> None:
    assert (
        nr_srs_port_cyclic_shift_indices(
            ports,
            transmission_comb=comb,
            base_cyclic_shift=base,
            ports8_tdm=ports8_tdm,
        )
        == expected
    )


def test_srs_port_cyclic_shift_roster_rejects_partial_or_invalid_ports() -> None:
    with pytest.raises(ValueError, match="complete configured"):
        nr_srs_port_cyclic_shift_indices((1000, 1002), transmission_comb=2, base_cyclic_shift=0)


def test_srs_cyclic_shift_finer_granularity_adds_half_shift() -> None:
    ordinary = nr_srs_sequence_type1(72, group_index=5, transmission_comb=2)
    half_shifted = nr_srs_sequence_type1(
        72,
        group_index=5,
        transmission_comb=2,
        cyclic_shift_hopping_offset=1,
        hopping_finer_granularity=True,
    )
    n = np.arange(72)
    np.testing.assert_allclose(half_shifted, ordinary * np.exp(2j * np.pi * n / 16))
    with pytest.raises(ValueError, match="complete configured"):
        nr_srs_port_cyclic_shift_indices(
            (1000, 1001),
            transmission_comb=2,
            base_cyclic_shift=0,
            ports8_tdm=True,
        )


def test_srs_multiport_resource_mapping_preserves_port_axis_and_mask() -> None:
    sequences = nr_srs_port_sequences_type1(
        24,
        cyclic_shift_indices=(0, 1),
        srs_symbol_count=2,
        sequence_id=12,
        slot_index=0,
        start_symbol_index=12,
        hopping_mode="neither",
        transmission_comb=2,
    )
    result = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
        beta_srs=0.5,
    )

    assert isinstance(result, NRSRSMultiportResourceGrid)
    assert result.grid.shape == (2, 14, 48)
    assert result.mask.shape == result.grid.shape
    assert result.symbol_indices == (12, 13)
    assert int(np.count_nonzero(result.mask)) == 2 * 2 * 24
    np.testing.assert_array_equal(result.grid[0, 12, 0::2], 0.5 / np.sqrt(2) * sequences[0, 0])
    np.testing.assert_array_equal(result.grid[1, 13, 0::2], 0.5 / np.sqrt(2) * sequences[1, 1])
    assert not result.grid.flags.writeable
    assert not result.mask.flags.writeable


def test_srs_multiport_mapping_normalizes_four_port_three_tx_power() -> None:
    sequences = np.ones((4, 1, 24), dtype=np.complex128)
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 0,
        "srs_symbol_count": 1,
        "comb_size": 2,
        "comb_offset": 0,
    }

    result = map_nr_srs_multiport_resource(sequences, four_port_srs_3tx=True, **common)
    np.testing.assert_array_equal(
        result.grid[:, 13, 0::2],
        np.full((4, 24), 1 / np.sqrt(3), dtype=np.complex128),
    )
    with pytest.raises(ValueError, match="requires exactly four"):
        map_nr_srs_multiport_resource(sequences[:3], four_port_srs_3tx=True, **common)


def test_srs_eight_port_tdm_activity_matches_resource_local_symbol_pattern() -> None:
    ports = tuple(range(1000, 1008))
    activity = nr_srs_port_symbol_activity(ports, srs_symbol_count=4, ports8_tdm=True)
    assert activity == (
        (True, False, True, False),
        (True, False, True, False),
        (False, True, False, True),
        (False, True, False, True),
        (True, False, True, False),
        (True, False, True, False),
        (False, True, False, True),
        (False, True, False, True),
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (
            {"n2": {"combOffset-n2": 1, "cyclicShift-n2": 7}},
            NRSRSTransmissionCombConfiguration(2, 1, 7),
        ),
        (
            {"n4": {"combOffset-n4": 3, "cyclicShift-n4": 11}},
            NRSRSTransmissionCombConfiguration(4, 3, 11),
        ),
        (
            {"n8": {"combOffset-n8": 7, "cyclicShift-n8": 5}},
            NRSRSTransmissionCombConfiguration(8, 7, 5),
        ),
    ],
)
def test_srs_transmission_comb_adapter_resolves_rrc_choices(choice, expected) -> None:
    assert nr_srs_transmission_comb_from_rrc(choice) == expected


@pytest.mark.parametrize(
    "choice",
    [
        {},
        {"n2": {"combOffset-n2": 0}},
        {"n2": {"combOffset-n2": 0, "cyclicShift-n2": 0, "extra": 1}},
        {"n4": {"combOffset-n4": True, "cyclicShift-n4": 0}},
        {"n8": {"combOffset-n8": 8, "cyclicShift-n8": 0}},
    ],
)
def test_srs_transmission_comb_adapter_rejects_malformed_decoded_choices(choice) -> None:
    with pytest.raises((TypeError, ValueError)):
        nr_srs_transmission_comb_from_rrc(choice)


@pytest.mark.parametrize(
    ("port_count", "expected"),
    [
        (None, NRSRSPortConfiguration((1000,), False)),
        ("port1", NRSRSPortConfiguration((1000,), False)),
        ("ports2", NRSRSPortConfiguration((1000, 1001), False)),
        ("ports4", NRSRSPortConfiguration((1000, 1001, 1002, 1003), False)),
    ],
)
def test_srs_rrc_port_configuration_resolves_standard_enums(port_count, expected) -> None:
    assert nr_srs_port_configuration_from_rrc(port_count) == expected


@pytest.mark.parametrize("port_mode", ["ports8", "ports8tdm"])
def test_srs_rrc_eight_port_configuration(port_mode) -> None:
    assert nr_srs_port_configuration_from_rrc(
        "ports4", nrof_srs_ports_n8=port_mode
    ) == NRSRSPortConfiguration(tuple(range(1000, 1008)), port_mode == "ports8tdm")


def test_srs_multiport_grid_rejects_inconsistent_mask() -> None:
    with pytest.raises(ValueError, match="mask disagrees"):
        NRSRSMultiportResourceGrid(
            grid=np.zeros((2, 14, 48), dtype=np.complex128),
            mask=np.zeros((2, 14, 48), dtype=np.bool_),
            symbol_indices=(13,),
            subcarrier_indices=(0,),
        )


def test_srs_multiport_mapper_rejects_wrong_rank_and_symbol_count() -> None:
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 1,
        "srs_symbol_count": 2,
        "comb_size": 2,
        "comb_offset": 0,
    }
    with pytest.raises(ValueError, match=r"shape \[ports, SRS symbols"):
        map_nr_srs_multiport_resource(np.ones((2, 24), dtype=np.complex128), **common)
    with pytest.raises(ValueError, match="exactly srs_symbol_count"):
        map_nr_srs_multiport_resource(np.ones((2, 1, 24), dtype=np.complex128), **common)


@pytest.mark.parametrize(
    (
        "sequence_length",
        "group_index",
        "base_sequence_index",
        "cyclic_shift_index",
        "transmission_comb",
    ),
    [(30, 17, 0, 6, 2), (36, 0, 0, 0, 2), (48, 3, 0, 7, 2), (72, 29, 1, 5, 8)],
)
def test_srs_low_papr_sequence_matches_independent_scalar_reference(
    sequence_length, group_index, base_sequence_index, cyclic_shift_index, transmission_comb
) -> None:
    shift_count = srs_cyclic_shift_count(transmission_comb)
    sequence = nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        base_sequence_index=base_sequence_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=shift_count,
    )
    expected = np.asarray(
        [
            np.exp(
                -1j
                * np.pi
                * (
                    group_index * (n + 1) * (n + 2) / 31
                    if sequence_length == 30
                    else _reference_low_papr_phase(
                        sequence_length, group_index, base_sequence_index, n
                    )
                )
                + 2j * np.pi * cyclic_shift_index * n / shift_count
            )
            for n in range(sequence_length)
        ],
        dtype=np.complex128,
    )

    np.testing.assert_allclose(sequence, expected, atol=5e-12, rtol=5e-12)
    np.testing.assert_allclose(np.abs(sequence), 1.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("sequence_length", "group_index", "phase_row"),
    [
        (6, 0, (-3, -1, 3, 3, -1, -3)),
        (12, 29, (-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3)),
        (18, 0, (-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1)),
        (
            24,
            29,
            (3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3),
        ),
    ],
)
def test_srs_short_type1_sequences_match_ts_phase_tables(
    sequence_length, group_index, phase_row
) -> None:
    cyclic_shift_index = 3
    cyclic_shift_count = 8
    sequence = nr_low_papr_sequence_type1(
        sequence_length,
        group_index=group_index,
        cyclic_shift_index=cyclic_shift_index,
        cyclic_shift_count=cyclic_shift_count,
    )
    indices = np.arange(sequence_length)
    expected = np.exp(
        1j * np.pi * np.asarray(phase_row) / 4
        + 2j * np.pi * cyclic_shift_index * indices / cyclic_shift_count
    )

    np.testing.assert_allclose(sequence, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(np.abs(sequence), 1.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    ("args", "error", "message"),
    [
        ((32,), ValueError, "supported type-1 sequence lengths"),
        ((36,), TypeError, "group_index"),
        ((36,), ValueError, "group_index"),
        ((36,), ValueError, "cyclic_shift_index"),
        ((48,), ValueError, "sequence lengths below 72"),
    ],
)
def test_srs_low_papr_sequence_rejects_unsupported_parameters(args, error, message) -> None:
    kwargs = {"group_index": 0}
    if message == "group_index" and error is TypeError:
        kwargs["group_index"] = True
    elif message == "group_index":
        kwargs["group_index"] = 30
    elif message == "cyclic_shift_index":
        kwargs.update(cyclic_shift_index=8, cyclic_shift_count=8)
    elif message == "sequence lengths below 72":
        kwargs["base_sequence_index"] = 1
    with pytest.raises(error, match=message):
        nr_low_papr_sequence_type1(*args, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"hopping_finer_granularity": 1}, TypeError, "hopping_finer_granularity"),
        ({"cyclic_shift_count": 0}, ValueError, "cyclic_shift_count must be positive"),
        (
            {"cyclic_shift_count": 2, "cyclic_shift_hopping_offset": 2},
            ValueError,
            "outside the configured shift range",
        ),
    ],
)
def test_low_papr_sequence_rejects_invalid_cyclic_shift_hopping(kwargs, error, message) -> None:
    with pytest.raises(error, match=message):
        nr_low_papr_sequence_type1(72, group_index=0, **kwargs)


@pytest.mark.parametrize(
    ("sequence_length", "error", "message"),
    [
        (True, TypeError, "sequence_length must be an integer"),
        (0, ValueError, "sequence_length must be positive"),
    ],
)
def test_sequence_hopping_index_rejects_invalid_sequence_lengths(
    sequence_length, error, message
) -> None:
    with pytest.raises(error, match=message):
        nr_srs_sequence_hopping_index(1, 0, 13, 0, sequence_length)


@pytest.mark.parametrize(("comb", "expected"), [(2, 8), (4, 12), (8, 6)])
def test_srs_cyclic_shift_count_matches_transmission_comb(comb, expected) -> None:
    assert srs_cyclic_shift_count(comb) == expected


@pytest.mark.parametrize(("comb", "error"), [(3, ValueError), (True, TypeError)])
def test_srs_cyclic_shift_count_rejects_invalid_comb(comb, error) -> None:
    with pytest.raises(error, match="transmission_comb"):
        srs_cyclic_shift_count(comb)


def test_srs_sequence_binds_cyclic_shift_count_to_comb() -> None:
    sequence = nr_srs_sequence_type1(
        48,
        group_index=3,
        base_sequence_index=0,
        cyclic_shift_index=7,
        transmission_comb=2,
    )
    explicit = nr_low_papr_sequence_type1(
        48,
        group_index=3,
        base_sequence_index=0,
        cyclic_shift_index=7,
        cyclic_shift_count=8,
    )
    np.testing.assert_array_equal(sequence, explicit)


def test_srs_sequence_rejects_shift_index_outside_comb_range() -> None:
    with pytest.raises(ValueError, match="cyclic_shift_index"):
        nr_srs_sequence_type1(
            48,
            group_index=3,
            cyclic_shift_index=8,
            transmission_comb=2,
        )


@pytest.mark.parametrize("hopping_mode", ["neither", "groupHopping"])
def test_srs_hopped_sequence_resolves_group_before_sequence_generation(hopping_mode) -> None:
    sequence_id = 71
    slot_index = 2
    start_symbol_index = 8
    resource_symbol_index = 3
    group_index = (
        sequence_id % 30
        if hopping_mode == "neither"
        else nr_srs_group_hopping_index(
            sequence_id, slot_index, start_symbol_index, resource_symbol_index
        )
    )
    sequence = nr_srs_hopped_sequence_type1(
        72,
        sequence_id=sequence_id,
        slot_index=slot_index,
        start_symbol_index=start_symbol_index,
        resource_symbol_index=resource_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=4,
        cyclic_shift_index=4,
    )
    expected = nr_srs_sequence_type1(
        72,
        group_index=group_index,
        base_sequence_index=0,
        cyclic_shift_index=4,
        transmission_comb=4,
    )
    np.testing.assert_array_equal(sequence, expected)


@pytest.mark.parametrize(
    ("hopping_mode", "sequence_length", "slot_symbol_count"),
    [("neither", 24, 14), ("groupHopping", 48, 14), ("sequenceHopping", 96, 12)],
)
def test_srs_resource_sequence_rows_follow_each_resource_symbol_hop(
    hopping_mode, sequence_length, slot_symbol_count
) -> None:
    start_symbol_index = 10 if slot_symbol_count == 14 else 8
    rows = nr_srs_resource_sequences_type1(
        sequence_length,
        srs_symbol_count=4,
        sequence_id=91,
        slot_index=7,
        start_symbol_index=start_symbol_index,
        hopping_mode=hopping_mode,
        transmission_comb=2,
        cyclic_shift_index=3,
        slot_symbol_count=slot_symbol_count,
    )
    expected = np.stack(
        [
            nr_srs_hopped_sequence_type1(
                sequence_length,
                sequence_id=91,
                slot_index=7,
                start_symbol_index=start_symbol_index,
                resource_symbol_index=index,
                hopping_mode=hopping_mode,
                transmission_comb=2,
                cyclic_shift_index=3,
                slot_symbol_count=slot_symbol_count,
            )
            for index in range(4)
        ]
    )

    np.testing.assert_array_equal(rows, expected)


def test_srs_resource_sequence_rows_map_directly_to_each_resource_symbol() -> None:
    rows = nr_srs_resource_sequences_type1(
        6,
        srs_symbol_count=2,
        sequence_id=12,
        slot_index=0,
        start_symbol_index=12,
        hopping_mode="neither",
        transmission_comb=2,
    )
    mapped = map_nr_srs_resource(
        rows,
        slot_symbol_count=14,
        subcarrier_count=12,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )

    assert mapped.symbol_indices == (12, 13)
    np.testing.assert_array_equal(
        mapped.grid[
            np.asarray(mapped.symbol_indices)[:, None],
            list(mapped.subcarrier_indices),
        ],
        rows,
    )


def test_srs_resource_sequences_apply_configured_cyclic_shift_subset() -> None:
    subset = (False, True, False, True, False, False, True, False)
    coordinates = {
        "transmission_comb": 2,
        "cyclic_shift_hopping_id": 913,
        "radio_frame_index": 4,
        "slot_index": 7,
        "slots_per_frame": 20,
        "slot_symbol_count": 14,
        "start_symbol_index": 9,
        "srs_symbol_count": 4,
        "hopping_subset": subset,
    }
    offsets = nr_srs_cyclic_shift_hopping_offsets(**coordinates)
    assert set(offsets) <= {1, 3, 6}
    automatic = nr_srs_resource_sequences_type1(
        72,
        sequence_id=91,
        slot_index=coordinates["slot_index"],
        start_symbol_index=coordinates["start_symbol_index"],
        srs_symbol_count=coordinates["srs_symbol_count"],
        hopping_mode="neither",
        transmission_comb=2,
        cyclic_shift_hopping_id=coordinates["cyclic_shift_hopping_id"],
        radio_frame_index=coordinates["radio_frame_index"],
        slots_per_frame=coordinates["slots_per_frame"],
        hopping_subset=subset,
    )
    explicit = nr_srs_resource_sequences_type1(
        72,
        sequence_id=91,
        slot_index=coordinates["slot_index"],
        start_symbol_index=coordinates["start_symbol_index"],
        srs_symbol_count=coordinates["srs_symbol_count"],
        hopping_mode="neither",
        transmission_comb=2,
        cyclic_shift_hopping_offsets=offsets,
    )
    np.testing.assert_array_equal(automatic, explicit)


def test_srs_mapper_places_one_sequence_on_end_relative_symbols_and_comb() -> None:
    sequence = np.exp(1j * np.arange(24) * 0.17)
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=2,
        comb_offset=1,
        beta_srs=0.5,
    )

    assert result.symbol_indices == (10, 11, 12, 13)
    assert result.subcarrier_indices == tuple(range(1, 48, 2))
    assert int(np.count_nonzero(result.mask)) == 4 * 24
    np.testing.assert_array_equal(result.grid[10, 1::2], 0.5 * sequence)
    np.testing.assert_array_equal(result.grid[13, 1::2], 0.5 * sequence)
    assert not result.mask[:, 0::2].any()
    assert not result.grid[0:10].any()
    assert not result.grid.flags.writeable
    assert not result.mask.flags.writeable


def test_srs_mapper_accepts_per_symbol_sequences_and_frequency_origin() -> None:
    sequence = np.arange(30, dtype=np.float64).reshape(2, 15).astype(np.complex128) + 1j
    result = map_nr_srs_resource(
        sequence,
        slot_symbol_count=12,
        subcarrier_count=64,
        start_position_offset=1,
        srs_symbol_count=2,
        comb_size=4,
        comb_offset=2,
        frequency_start_subcarrier=5,
    )

    assert result.symbol_indices == (10, 11)
    # The configured offset counts backward from the slot end, then the rows
    # map consecutively from the derived first SRS symbol.
    assert result.subcarrier_indices == tuple(range(7, 64, 4))
    np.testing.assert_array_equal(result.grid[10, list(result.subcarrier_indices)], sequence[0])
    np.testing.assert_array_equal(result.grid[11, list(result.subcarrier_indices)], sequence[1])


def test_srs_mapper_rejects_sequence_shape_and_type_mismatch() -> None:
    common = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 3,
        "srs_symbol_count": 4,
        "comb_size": 2,
        "comb_offset": 1,
    }
    with pytest.raises(TypeError, match="complex-valued"):
        map_nr_srs_resource(np.ones(24), **common)
    with pytest.raises(ValueError, match="sequence length"):
        map_nr_srs_resource(np.ones(23, dtype=np.complex128), **common)
    with pytest.raises(ValueError, match="exactly srs_symbol_count"):
        map_nr_srs_resource(np.ones((3, 24), dtype=np.complex128), **common)


@pytest.mark.parametrize(
    ("sequence", "changes", "message"),
    [
        (np.ones((2, 2, 12), dtype=np.complex128), {}, r"shape \[occupied REs\]"),
        (np.full(24, np.nan + 1j, dtype=np.complex128), {}, "only finite values"),
        (
            np.ones(24, dtype=np.complex128),
            {"subcarrier_count": 0},
            "subcarrier_count must be positive",
        ),
        (
            np.ones(24, dtype=np.complex128),
            {"frequency_start_subcarrier": -1},
            "must be nonnegative",
        ),
    ],
)
def test_srs_mapper_rejects_invalid_grid_coordinates_and_nonfinite_sequences(
    sequence, changes, message
) -> None:
    arguments = {
        "slot_symbol_count": 14,
        "subcarrier_count": 48,
        "start_position_offset": 1,
        "srs_symbol_count": 1,
        "comb_size": 2,
        "comb_offset": 0,
    }
    arguments.update(changes)

    with pytest.raises(ValueError, match=message):
        map_nr_srs_resource(sequence, **arguments)


def test_srs_grid_rejects_mask_that_does_not_match_declared_resource() -> None:
    grid = np.zeros((14, 48), dtype=np.complex128)
    mask = np.zeros((14, 48), dtype=np.bool_)
    with pytest.raises(ValueError, match="mask disagrees"):
        NRSRSResourceGrid(
            grid=grid,
            mask=mask,
            symbol_indices=(10,),
            subcarrier_indices=(1,),
        )


def test_srs_single_port_ls_estimator_recovers_per_re_channel() -> None:
    sequence = np.exp(1j * np.arange(24, dtype=np.float64) * 0.13)
    mapped = map_nr_srs_resource(
        sequence,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=1,
        beta_srs=0.7,
    )
    channel = np.exp(1j * np.arange(14 * 48).reshape(14, 48) * 0.003)
    result = nr_srs_single_port_ls_channel_estimate(mapped, mapped.grid * channel)

    np.testing.assert_allclose(result.channel_by_re[mapped.mask], channel[mapped.mask])
    assert result.symbol_indices == mapped.symbol_indices
    assert np.array_equal(result.mask, mapped.mask)
    assert not result.channel_by_re.flags.writeable
    assert not result.mask.flags.writeable
    assert np.count_nonzero(result.channel_by_re[~result.mask]) == 0


def test_srs_single_port_ls_estimator_recovers_hopped_resource_channel() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        0, transmission_comb=2, b_srs=0, b_hop=0, reference_subcarrier=0
    )
    mapped = map_nr_srs_hopped_resource(
        np.ones((2, allocation.bandwidth.sequence_length), dtype=np.complex128),
        frequency_allocations=(allocation,),
        slot_symbol_count=14,
        subcarrier_count=allocation.bandwidth.allocated_bandwidth_subcarriers,
        start_position_offset=1,
        srs_symbol_count=2,
    )
    channel = np.full(mapped.grid.shape, 0.4 - 0.2j, dtype=np.complex128)
    result = nr_srs_single_port_ls_channel_estimate(mapped, mapped.grid * channel)

    np.testing.assert_allclose(result.channel_by_re[result.mask], 0.4 - 0.2j)


def test_srs_multiport_ls_estimator_separates_ports_and_reports_noise_covariance() -> None:
    sequences = np.ones((2, 2, 24), dtype=np.complex128)
    sequences[1, 1] *= -1.0
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    channel = np.empty((2, 2, 48), dtype=np.complex128)
    channel[:, 0, :] = 0.4 + 0.2j
    channel[:, 1, :] = -0.1 + 0.3j
    received = np.einsum("rpk,psk->rsk", channel, mapped.grid)
    noise = np.array([0.01, 0.04], dtype=np.float64)

    result = nr_srs_multiport_ls_channel_estimate(mapped, received, noise_variance_by_rx_ant=noise)

    active_subcarriers = np.any(mapped.mask, axis=(0, 1))
    np.testing.assert_array_equal(result.valid_subcarrier_mask, active_subcarriers)
    np.testing.assert_allclose(
        result.channel_by_rx_port_subcarrier[:, :, active_subcarriers],
        channel[:, :, active_subcarriers],
    )
    design = mapped.grid[:, mapped.symbol_indices[0] : mapped.symbol_indices[-1] + 1, 0].T
    expected_base_covariance = np.linalg.inv(design.conj().T @ design)
    np.testing.assert_allclose(
        result.noise_covariance_by_rx_subcarrier[:, active_subcarriers],
        np.broadcast_to(
            noise[:, None, None, None] * expected_base_covariance[None, None, :, :],
            result.noise_covariance_by_rx_subcarrier[:, active_subcarriers].shape,
        ),
        atol=1e-18,
    )
    assert set(result.reason_codes) <= {None, "not_occupied"}
    assert not result.channel_by_rx_port_subcarrier.flags.writeable
    assert not result.noise_covariance_by_rx_subcarrier.flags.writeable


def test_srs_multiport_weighted_ls_uses_per_symbol_noise_variance() -> None:
    sequences = np.ones((2, 4, 24), dtype=np.complex128)
    sequences[1, 1::2] *= -1.0
    mapped = map_nr_srs_multiport_resource(
        sequences,
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=3,
        srs_symbol_count=4,
        comb_size=2,
        comb_offset=0,
    )
    channel = np.empty((2, 2, 48), dtype=np.complex128)
    channel[:, 0, :] = 0.4 + 0.2j
    channel[:, 1, :] = -0.1 + 0.3j
    received = np.einsum("rpk,psk->rsk", channel, mapped.grid)
    residuals = np.asarray(
        [
            [0.1 + 0.2j, -0.3 + 0.05j],
            [-0.2 + 0.1j, 0.05 + 0.25j],
            [0.3 - 0.1j, 0.2 - 0.15j],
            [-0.05 + 0.3j, -0.25 - 0.1j],
        ]
    )
    for symbol_index, residual in zip(mapped.symbol_indices, residuals, strict=True):
        received[:, symbol_index, :] += residual[:, None]
    symbol_noise = np.ones((2, 14), dtype=np.float64)
    symbol_noise[0, list(mapped.symbol_indices)] = [0.01, 5.0, 0.02, 3.0]
    symbol_noise[1, list(mapped.symbol_indices)] = [5.0, 0.01, 3.0, 0.02]

    result = nr_srs_multiport_ls_channel_estimate(
        mapped,
        received,
        noise_variance_by_rx_symbol=symbol_noise,
    )

    active_subcarrier = int(np.flatnonzero(np.any(mapped.mask, axis=(0, 1)))[0])
    symbols = np.asarray(mapped.symbol_indices)
    design = mapped.grid[:, symbols, active_subcarrier].T
    occupied = np.any(mapped.mask[:, symbols, active_subcarrier], axis=0)
    design = design[occupied]
    observed = received[:, symbols[occupied], active_subcarrier].T
    expected = np.empty((2, 2), dtype=np.complex128)
    expected_covariance = np.empty((2, 2, 2), dtype=np.complex128)
    for rx_index in range(2):
        weights = 1.0 / symbol_noise[rx_index, symbols[occupied]]
        gram = design.conj().T @ (weights[:, None] * design)
        expected_covariance[rx_index] = np.linalg.inv(gram)
        expected[rx_index] = expected_covariance[rx_index] @ (
            design.conj().T @ (weights * observed[:, rx_index])
        )
    ordinary_least_squares = np.linalg.lstsq(design, observed, rcond=None)[0].T
    assert not np.allclose(expected, ordinary_least_squares)

    np.testing.assert_allclose(
        result.channel_by_rx_port_subcarrier[:, :, active_subcarrier],
        expected,
    )
    for rx_index in range(2):
        np.testing.assert_allclose(
            result.noise_covariance_by_rx_subcarrier[rx_index, active_subcarrier],
            expected_covariance[rx_index],
        )
    np.testing.assert_array_equal(result.noise_variance_by_rx_symbol, symbol_noise)
    assert not result.noise_variance_by_rx_symbol.flags.writeable


def test_srs_multiport_ls_estimator_marks_rank_deficient_port_signatures() -> None:
    mapped = map_nr_srs_multiport_resource(
        np.ones((2, 2, 24), dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    received = np.ones((1, 14, 48), dtype=np.complex128)

    result = nr_srs_multiport_ls_channel_estimate(mapped, received, noise_variance_by_rx_ant=0.01)

    active_subcarriers = np.any(mapped.mask, axis=(0, 1))
    assert not result.valid_subcarrier_mask[active_subcarriers].any()
    assert set(result.reason_codes[i] for i in np.flatnonzero(active_subcarriers)) == {
        "rank_deficient_port_signature"
    }
    assert np.count_nonzero(result.channel_by_rx_port_subcarrier) == 0


@pytest.mark.parametrize(
    ("case", "options", "error", "message"),
    [
        ("mapped_type", {}, TypeError, "mapped multiport SRS resource"),
        ("real_grid", {}, ValueError, "received_grid must be complex"),
        ("zero_rx", {}, ValueError, "at least one RX antenna"),
        ("shape", {}, ValueError, "axes must match"),
        ("nonfinite", {}, ValueError, "only finite values"),
        (
            "both_noise_modes",
            {"noise_variance_by_rx_symbol": np.ones((1, 14))},
            ValueError,
            "provide per-RX or per-RX/per-symbol",
        ),
        (
            "missing_noise",
            {"noise_variance_by_rx_ant": None},
            ValueError,
            "per-RX noise variance is required",
        ),
        (
            "complex_noise",
            {"noise_variance_by_rx_ant": 0.01 + 0.0j},
            TypeError,
            "must be real-valued",
        ),
        (
            "noise_shape",
            {"noise_variance_by_rx_ant": np.ones((1, 2))},
            ValueError,
            "scalar or one value per RX",
        ),
        ("nonpositive_noise", {"noise_variance_by_rx_ant": 0.0}, ValueError, "finite and positive"),
        (
            "unrepresentable_noise",
            {"noise_variance_by_rx_ant": 1e-320},
            ValueError,
            "reciprocals must be representable",
        ),
        (
            "complex_symbol_noise",
            {"noise_variance_by_rx_symbol": np.ones((1, 14), dtype=np.complex128)},
            TypeError,
            "must be real-valued",
        ),
        (
            "symbol_noise_shape",
            {"noise_variance_by_rx_symbol": np.ones((1, 13))},
            ValueError,
            "must have shape",
        ),
        (
            "nonpositive_symbol_noise",
            {"noise_variance_by_rx_symbol": np.zeros((1, 14))},
            ValueError,
            "finite and positive",
        ),
        (
            "unrepresentable_symbol_noise",
            {"noise_variance_by_rx_symbol": np.full((1, 14), 1e-320)},
            ValueError,
            "reciprocals must be representable",
        ),
    ],
)
def test_multiport_ls_rejects_invalid_received_grids_and_noise_profiles(
    case, options, error, message
) -> None:
    mapped = map_nr_srs_multiport_resource(
        np.ones((2, 2, 24), dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    received = np.zeros((1, 14, 48), dtype=np.complex128)
    if case == "mapped_type":
        mapped = None
    elif case == "real_grid":
        received = np.zeros((1, 14, 48), dtype=np.float64)
    elif case == "zero_rx":
        received = np.zeros((0, 14, 48), dtype=np.complex128)
    elif case == "shape":
        received = np.zeros((1, 14, 47), dtype=np.complex128)
    elif case == "nonfinite":
        received[0, 0, 0] = complex(float("nan"), 0)

    with pytest.raises(error, match=message):
        kwargs = {"noise_variance_by_rx_ant": 0.01}
        if case in {
            "complex_symbol_noise",
            "symbol_noise_shape",
            "nonpositive_symbol_noise",
            "unrepresentable_symbol_noise",
        }:
            kwargs["noise_variance_by_rx_ant"] = None
        kwargs.update(options)
        nr_srs_multiport_ls_channel_estimate(mapped, received, **kwargs)


def test_srs_multiport_ls_estimator_rejects_noise_and_receive_shape_errors() -> None:
    mapped = map_nr_srs_multiport_resource(
        np.ones((2, 2, 24), dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=2,
        srs_symbol_count=2,
        comb_size=2,
        comb_offset=0,
    )
    received = np.ones((1, 14, 48), dtype=np.complex128)
    with pytest.raises(ValueError, match="symbol/subcarrier axes"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received[:, :-1], noise_variance_by_rx_ant=0.01
        )
    with pytest.raises(ValueError, match="finite and positive"):
        nr_srs_multiport_ls_channel_estimate(mapped, received, noise_variance_by_rx_ant=0.0)
    with pytest.raises(TypeError, match="real-valued"):
        nr_srs_multiport_ls_channel_estimate(mapped, received, noise_variance_by_rx_ant=1.0 + 0.0j)
    symbol_noise = np.ones((1, 14), dtype=np.float64)
    with pytest.raises(TypeError, match="real-valued"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=symbol_noise.astype(np.complex128)
        )
    with pytest.raises(ValueError, match="shape"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=np.ones((1, 13))
        )
    with pytest.raises(ValueError, match="finite and positive"):
        nr_srs_multiport_ls_channel_estimate(
            mapped, received, noise_variance_by_rx_symbol=np.zeros((1, 14))
        )
    with pytest.raises(ValueError, match="reciprocals must be representable"):
        nr_srs_multiport_ls_channel_estimate(
            mapped,
            received,
            noise_variance_by_rx_symbol=np.full((1, 14), np.nextafter(0.0, 1.0)),
        )
    with pytest.raises(ValueError, match="not both"):
        nr_srs_multiport_ls_channel_estimate(
            mapped,
            received,
            noise_variance_by_rx_ant=0.01,
            noise_variance_by_rx_symbol=np.ones((1, 14)),
        )


@pytest.mark.parametrize(
    ("received", "message"),
    [
        (np.ones((14, 48)), "complex"),
        (np.ones((12, 48), dtype=np.complex128), "shape"),
        (np.full((14, 48), np.nan + 0j), "finite"),
    ],
)
def test_srs_single_port_ls_estimator_rejects_invalid_receive_grid(received, message) -> None:
    mapped = map_nr_srs_resource(
        np.ones(24, dtype=np.complex128),
        slot_symbol_count=14,
        subcarrier_count=48,
        start_position_offset=1,
        srs_symbol_count=1,
        comb_size=2,
        comb_offset=0,
    )
    with pytest.raises(ValueError, match=message):
        nr_srs_single_port_ls_channel_estimate(mapped, received)


def test_srs_intercell_overlap_audit_rejects_incomparable_grid_contexts() -> None:
    allocation = nr_srs_fixed_frequency_allocation(0, transmission_comb=2, b_srs=0, b_hop=0)
    base = {
        "resource_set_id": 0,
        "resource_id": 0,
        "grid_id": "carrier-a-grid",
        "carrier_hz": 3.5e9,
        "numerology": 1,
        "subcarrier_spacing_hz": 30e3,
        "target_slot_index": 4,
        "slot_symbol_count": 14,
        "start_symbol_index": 12,
        "srs_symbol_count": 1,
        "allocations": (allocation,),
    }
    first = NRSRSScheduledResourceOccurrence(cell_id="cell-a", **base)
    second = NRSRSScheduledResourceOccurrence(
        cell_id="cell-b", **{**base, "grid_id": "different-grid"}
    )
    with pytest.raises(ValueError, match="single common grid"):
        audit_nr_srs_intercell_re_overlap([first, second])


def test_srs_intercell_overlap_audit_counts_shared_re_and_cross_cell_pairs() -> None:
    allocation = nr_srs_fixed_frequency_allocation(0, transmission_comb=2, b_srs=0, b_hop=0)
    context = {
        "grid_id": "common-grid",
        "carrier_hz": 3.5e9,
        "numerology": 0,
        "subcarrier_spacing_hz": 15_000.0,
        "target_slot_index": 8,
        "slot_symbol_count": 14,
        "start_symbol_index": 13,
        "srs_symbol_count": 1,
        "allocations": (allocation,),
    }
    occurrences = (
        NRSRSScheduledResourceOccurrence(
            cell_id="cell-a", resource_set_id=0, resource_id=0, **context
        ),
        NRSRSScheduledResourceOccurrence(
            cell_id="cell-a", resource_set_id=0, resource_id=1, **context
        ),
        NRSRSScheduledResourceOccurrence(
            cell_id="cell-b", resource_set_id=2, resource_id=0, **context
        ),
    )

    result = audit_nr_srs_intercell_re_overlap(occurrences)

    assert result["status"] == "INTERCELL_RE_OVERLAP_DETECTED"
    assert result["expanded_resource_element_count"] == 3 * allocation.bandwidth.sequence_length
    assert result["shared_intercell_resource_element_count"] == allocation.bandwidth.sequence_length
    assert len(result["collision_coordinates"]) == allocation.bandwidth.sequence_length
    assert [
        pair["shared_resource_element_count"] for pair in result["pairwise_resource_overlaps"]
    ] == [allocation.bandwidth.sequence_length] * 2

    disjoint_allocation = nr_srs_fixed_frequency_allocation(
        0, transmission_comb=2, b_srs=0, b_hop=0, comb_offset=1
    )
    disjoint = NRSRSScheduledResourceOccurrence(
        cell_id="cell-c",
        resource_set_id=0,
        resource_id=0,
        **{**context, "allocations": (disjoint_allocation,)},
    )
    no_overlap = audit_nr_srs_intercell_re_overlap((occurrences[0], disjoint))
    assert no_overlap["status"] == "NO_INTERCELL_RE_OVERLAP"
    assert no_overlap["pairwise_resource_overlaps"] == []


def test_srs_intercell_overlap_audit_bounds_coordinate_details() -> None:
    allocation = nr_srs_fixed_frequency_allocation(63, transmission_comb=2, b_srs=0, b_hop=0)
    context = {
        "resource_set_id": 0,
        "grid_id": "common-grid",
        "carrier_hz": 3.5e9,
        "numerology": 0,
        "subcarrier_spacing_hz": 15_000.0,
        "target_slot_index": 8,
        "slot_symbol_count": 14,
        "start_symbol_index": 10,
        "srs_symbol_count": 4,
        "allocations": (allocation,),
    }
    occurrences = (
        NRSRSScheduledResourceOccurrence(cell_id="cell-a", resource_id=0, **context),
        NRSRSScheduledResourceOccurrence(cell_id="cell-b", resource_id=0, **context),
    )

    result = audit_nr_srs_intercell_re_overlap(occurrences)

    assert result["shared_intercell_resource_element_count"] > 1_000
    assert len(result["collision_coordinates"]) == 1_000


def test_srs_intercell_overlap_audit_rejects_bad_rosters_and_work_limits() -> None:
    allocation = nr_srs_fixed_frequency_allocation(63, transmission_comb=2, b_srs=0, b_hop=0)
    base = {
        "cell_id": "cell-0",
        "resource_set_id": 0,
        "resource_id": 0,
        "grid_id": "common-grid",
        "carrier_hz": 3.5e9,
        "numerology": 0,
        "subcarrier_spacing_hz": 15_000.0,
        "target_slot_index": 8,
        "slot_symbol_count": 14,
        "start_symbol_index": 10,
        "srs_symbol_count": 4,
        "allocations": (allocation,),
    }
    occurrence = NRSRSScheduledResourceOccurrence(**base)
    with pytest.raises(ValueError, match="must be a sequence"):
        audit_nr_srs_intercell_re_overlap("not-a-roster")
    with pytest.raises(ValueError, match="nonempty sequence"):
        audit_nr_srs_intercell_re_overlap(())
    with pytest.raises(ValueError, match="identities must be unique"):
        audit_nr_srs_intercell_re_overlap((occurrence, occurrence))

    many_cells = tuple(
        NRSRSScheduledResourceOccurrence(
            **{**base, "cell_id": f"cell-{index}", "resource_id": index}
        )
        for index in range(40)
    )
    with pytest.raises(ValueError, match="exceeds its resource-element limit"):
        audit_nr_srs_intercell_re_overlap(many_cells)


def test_scheduled_srs_occurrence_rejects_partial_per_symbol_allocations() -> None:
    allocation = nr_srs_fixed_frequency_allocation(
        55, transmission_comb=4, b_srs=3, b_hop=1, n_srs=0
    )
    with pytest.raises(ValueError, match="shared or provided once per SRS symbol"):
        NRSRSScheduledResourceOccurrence(
            cell_id="cell-a",
            resource_set_id=0,
            resource_id=0,
            grid_id="common-grid",
            carrier_hz=3.5e9,
            numerology=0,
            subcarrier_spacing_hz=15_000.0,
            target_slot_index=8,
            slot_symbol_count=14,
            start_symbol_index=8,
            srs_symbol_count=4,
            allocations=(allocation, allocation),
        )
