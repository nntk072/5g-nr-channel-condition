from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest
from integrations.sionna.nr_prs import (
    NRPRSDelayCFOProfile,
    NRPRSDLRSTDMeasurement,
    NRPRSResourceConfiguration,
    NRPRSResourceSetTiming,
    NRPRSTDReport,
    NRPRSTimingQuality,
    build_nr_dl_tdoa_measurement_report,
    map_nr_dl_prs_resource,
    map_nr_dl_prs_resource_set,
    measure_nr_dl_rstd,
    nr_dl_prs_c_init,
    nr_dl_prs_delay_cfo_profile,
    nr_dl_prs_matched_delay_profile,
    nr_dl_prs_resource_occasion,
    nr_dl_prs_sequence,
)


def _scalar_gold_bits(c_init: int, count: int) -> list[int]:
    stop = 1600 + count
    x1 = [0] * (stop + 31)
    x2 = [0] * (stop + 31)
    x1[0] = 1
    for index in range(31):
        x2[index] = (c_init >> index) & 1
    for index in range(stop):
        x1[index + 31] = (x1[index + 3] + x1[index]) % 2
        x2[index + 31] = (
            x2[index + 3] + x2[index + 2] + x2[index + 1] + x2[index]
        ) % 2
    return [x1[index] ^ x2[index] for index in range(1600, stop)]


def _scalar_prs(
    *, sequence_id: int, slot: int, numerology: int, symbol: int, length: int, symbols_per_slot: int
) -> np.ndarray:
    low_id = sequence_id % 1024
    c_init = (
        (1 << 22) * (sequence_id // 1024)
        + (1 << 10) * (symbols_per_slot * slot + symbol + 1) * (2 * low_id + 1)
        + low_id
    ) % (1 << 31)
    bits = _scalar_gold_bits(c_init, 2 * length)
    return np.asarray(
        [
            ((1 - 2 * bits[2 * index]) + 1j * (1 - 2 * bits[2 * index + 1]))
            / math.sqrt(2.0)
            for index in range(length)
        ],
        dtype=np.complex128,
    )


@pytest.mark.parametrize(
    ("sequence_id", "slot", "mu", "symbol", "symbols_per_slot"),
    [(0, 0, 0, 0, 14), (1023, 9, 0, 13, 14), (1024, 19, 1, 7, 14), (4095, 39, 2, 11, 12)],
)
def test_prs_c_init_and_sequence_match_independent_scalar_reference(
    sequence_id, slot, mu, symbol, symbols_per_slot
):
    low_id = sequence_id % 1024
    expected_seed = (
        (1 << 22) * (sequence_id // 1024)
        + (1 << 10) * (symbols_per_slot * slot + symbol + 1) * (2 * low_id + 1)
        + low_id
    ) % (1 << 31)
    assert (
        nr_dl_prs_c_init(
            prs_sequence_id=sequence_id,
            slot_index=slot,
            numerology=mu,
            symbol_index=symbol,
            symbols_per_slot=symbols_per_slot,
        )
        == expected_seed
    )
    actual = nr_dl_prs_sequence(
        prs_sequence_id=sequence_id,
        slot_index=slot,
        numerology=mu,
        symbol_index=symbol,
        length=37,
        symbols_per_slot=symbols_per_slot,
    )
    expected = _scalar_prs(
        sequence_id=sequence_id,
        slot=slot,
        numerology=mu,
        symbol=symbol,
        length=37,
        symbols_per_slot=symbols_per_slot,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(np.abs(actual) ** 2, 1.0, rtol=0.0, atol=5e-16)


def test_prs_mapping_uses_symbol_specific_comb_offsets_and_point_a_indices():
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=512,
        resource_start_subcarrier=36,
        resource_bandwidth_rbs=4,
        prs_sequence_id=1234,
        slot_index=3,
        numerology=1,
        subcarrier_spacing_hz=30_000.0,
        comb_size=4,
        resource_element_offset=1,
        start_symbol=2,
        num_symbols=4,
        beta_prs=0.5,
    )
    expected_k_primes = (0, 2, 1, 3)
    for symbol_offset, symbol in enumerate(range(2, 6)):
        residue = (1 + expected_k_primes[symbol_offset]) % 4
        expected_indices = 36 + np.arange(residue, 48, 4)
        np.testing.assert_array_equal(resource.subcarrier_indices[symbol_offset], expected_indices)
        expected_sequence = _scalar_prs(
            sequence_id=1234,
            slot=3,
            numerology=1,
            symbol=symbol,
            length=len(expected_indices),
            symbols_per_slot=14,
        )
        np.testing.assert_array_equal(
            resource.grid[symbol, expected_indices], 0.5 * expected_sequence
        )
        assert np.count_nonzero(resource.resource_mask[symbol]) == 12
    assert np.count_nonzero(resource.resource_mask) == 48
    assert np.count_nonzero(resource.grid[0]) == 0
    assert not resource.grid.flags.writeable
    assert not resource.resource_mask.flags.writeable


@pytest.mark.parametrize(
    ("num_symbols", "comb_size"),
    [
        (1, 2),
        (2, 2),
        (4, 2),
        (6, 2),
        (12, 2),
        (1, 4),
        (4, 4),
        (12, 4),
        (1, 6),
        (6, 6),
        (12, 6),
        (1, 12),
        (12, 12),
    ],
)
def test_prs_mapping_repeats_normative_comb_offset_cycle(
    num_symbols: int, comb_size: int
) -> None:
    normative_offsets = {
        2: (0, 1),
        4: (0, 2, 1, 3),
        6: (0, 3, 1, 4, 2, 5),
        12: (0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11),
    }
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=24,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=1,
        prs_sequence_id=19,
        slot_index=0,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=comb_size,
        resource_element_offset=0,
        start_symbol=1,
        num_symbols=num_symbols,
    )

    for symbol_offset, indices in enumerate(resource.subcarrier_indices):
        k_prime = normative_offsets[comb_size][symbol_offset % comb_size]
        np.testing.assert_array_equal(indices, np.arange(k_prime, 12, comb_size))


def test_prs_mapping_rejects_reserved_ssb_symbols_and_unsupported_pairs():
    reserved = np.zeros(14, dtype=bool)
    reserved[5] = True
    options = dict(
        slot_symbol_count=14,
        subcarrier_count=144,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=12,
        prs_sequence_id=5,
        slot_index=0,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=4,
        num_symbols=2,
    )
    with pytest.raises(ValueError, match="reserved slot symbol"):
        map_nr_dl_prs_resource(**options, reserved_symbol_mask=reserved)
    with pytest.raises(ValueError, match="supported TS 38.211 PRS pair"):
        map_nr_dl_prs_resource(**{**options, "num_symbols": 3})


def test_prs_mapping_checks_exact_reserved_resource_elements():
    options = dict(
        slot_symbol_count=14,
        subcarrier_count=12,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=1,
        prs_sequence_id=5,
        slot_index=0,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=4,
        num_symbols=2,
    )
    reserved = np.zeros((14, 12), dtype=bool)
    reserved[4, 1] = True  # Same symbol but outside this resource's comb.
    resource = map_nr_dl_prs_resource(**options, reserved_resource_mask=reserved)
    assert not bool(resource.resource_mask[4, 1])

    reserved[4, 0] = True
    with pytest.raises(ValueError, match="reserved resource element"):
        map_nr_dl_prs_resource(**options, reserved_resource_mask=reserved)


def test_prs_resource_set_composes_distinct_resources_and_keeps_member_masks():
    resources = (
        NRPRSResourceConfiguration(
            "prs-a", 0, 1, 10, 2, 0, 2, 2, beta_prs=0.5
        ),
        NRPRSResourceConfiguration(
            "prs-b", 12, 1, 20, 2, 1, 2, 2, beta_prs=0.75
        ),
    )

    result = map_nr_dl_prs_resource_set(
        resources=resources,
        slot_symbol_count=14,
        subcarrier_count=24,
        slot_index=3,
        numerology=1,
        subcarrier_spacing_hz=30_000.0,
    )

    assert result.resource_ids == ("prs-a", "prs-b")
    assert result.grid.shape == (14, 24)
    assert len(result.resource_masks) == 2
    assert sum(np.count_nonzero(mask) for mask in result.resource_masks) == 24
    assert not np.any(result.collision_mask)
    assert result.colliding_resource_element_count == 0
    assert not result.grid.flags.writeable
    assert not result.collision_mask.flags.writeable


@pytest.mark.parametrize("resource_id", ["", " ", " prs-a", "prs-a "])
def test_prs_resource_set_rejects_empty_or_padded_resource_ids(resource_id: str) -> None:
    resource = NRPRSResourceConfiguration(resource_id, 0, 1, 10, 2, 0, 2, 2)

    with pytest.raises(ValueError, match="nonempty canonical resource_id"):
        map_nr_dl_prs_resource_set(
            resources=(resource,),
            slot_symbol_count=14,
            subcarrier_count=12,
            slot_index=0,
            numerology=0,
            subcarrier_spacing_hz=15_000.0,
        )


def test_prs_resource_set_reports_overlapping_resource_elements_and_bounds_allocation():
    member = NRPRSResourceConfiguration("prs-a", 0, 1, 10, 2, 0, 2, 2)
    result = map_nr_dl_prs_resource_set(
        resources=(member, NRPRSResourceConfiguration("prs-b", 0, 1, 20, 2, 0, 2, 2)),
        slot_symbol_count=14,
        subcarrier_count=12,
        slot_index=3,
        numerology=1,
        subcarrier_spacing_hz=30_000.0,
    )

    assert result.colliding_resource_element_count == 12
    assert np.count_nonzero(result.collision_mask) == 12
    with pytest.raises(ValueError, match="max_total_resource_cells"):
        map_nr_dl_prs_resource_set(
            resources=(member,),
            slot_symbol_count=14,
            subcarrier_count=12,
            slot_index=3,
            numerology=1,
            subcarrier_spacing_hz=30_000.0,
            max_total_resource_cells=100,
        )


def test_prs_matched_delay_profile_recovers_a_single_delayed_component():
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=240,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=20,
        prs_sequence_id=78,
        slot_index=7,
        numerology=1,
        subcarrier_spacing_hz=30_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=3,
        num_symbols=2,
    )
    true_delay_s = 0.37e-6
    frequencies = np.arange(240, dtype=np.float64) * resource.subcarrier_spacing_hz
    channel = 0.7 * np.exp(-2j * np.pi * frequencies * true_delay_s)
    observed = resource.grid * channel[None, :]
    delays = np.linspace(0.0, 0.8e-6, 801)

    result = nr_dl_prs_matched_delay_profile(observed, resource, delay_grid_s=delays)

    assert result.semantics == "strongest_component_delay_on_caller_grid"
    assert abs(result.peak_delay_s - true_delay_s) <= 5e-9
    assert result.comb_ambiguity_period_s == pytest.approx(1.0 / (2 * 30_000.0))
    assert result.peak_power > 0
    assert result.peak_to_median_ratio > 1.0
    assert not result.profile_power.flags.writeable


def test_prs_joint_delay_cfo_profile_recovers_static_link_hypotheses():
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=96,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=4,
        prs_sequence_id=501,
        slot_index=2,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=2,
        num_symbols=4,
    )
    symbol_times = np.arange(4, dtype=np.float64) * 0.5e-3
    true_delay_s = 80e-9
    true_cfo_hz = 250.0
    received = np.zeros_like(resource.grid)
    for row, (symbol, subcarriers) in enumerate(
        zip(resource.symbol_indices, resource.subcarrier_indices, strict=True)
    ):
        frequency = subcarriers.astype(np.float64) * resource.subcarrier_spacing_hz
        channel = 0.8 * np.exp(-2j * np.pi * frequency * true_delay_s)
        cfo = np.exp(2j * np.pi * true_cfo_hz * symbol_times[row])
        received[symbol, subcarriers] = resource.grid[symbol, subcarriers] * channel * cfo

    result = nr_dl_prs_delay_cfo_profile(
        received,
        resource,
        delay_grid_s=np.arange(0.0, 201e-9, 10e-9),
        cfo_grid_hz=np.arange(0.0, 501.0, 25.0),
        symbol_times_s=symbol_times,
    )

    assert isinstance(result, NRPRSDelayCFOProfile)
    assert result.semantics.endswith("static_channel_no_ici")
    assert result.peak_delay_s == pytest.approx(true_delay_s, abs=1e-15)
    assert result.peak_cfo_hz == pytest.approx(true_cfo_hz, abs=1e-12)
    assert result.profile_power.shape == (21, 21)
    assert result.delay_comb_ambiguity_period_s == pytest.approx(1 / 30_000.0)
    assert result.peak_power == pytest.approx(0.8**2, rel=1e-12)
    assert not result.profile_power.flags.writeable


def test_prs_joint_delay_cfo_profile_bounds_work_before_search():
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=24,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=2,
        prs_sequence_id=8,
        slot_index=0,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=2,
        num_symbols=2,
    )
    with pytest.raises(ValueError, match="exceeds max_work_cells"):
        nr_dl_prs_delay_cfo_profile(
            resource.grid,
            resource,
            delay_grid_s=np.asarray([0.0, 1e-9]),
            cfo_grid_hz=np.asarray([-1.0, 1.0]),
            symbol_times_s=np.asarray([0.0, 1e-3]),
            max_work_cells=1,
        )


def test_prs_resource_set_schedule_resolves_periodicity_and_repetitions():
    timing = NRPRSResourceSetTiming(
        numerology=0,
        periodicity_slots=10,
        periodicity_offset_slots=4,
        resource_slot_offset=2,
        repetition_factor=4,
        time_gap_slots=2,
    )

    first = nr_dl_prs_resource_occasion(system_frame_number=0, slot_in_frame=6, timing=timing)
    second = nr_dl_prs_resource_occasion(system_frame_number=0, slot_in_frame=8, timing=timing)
    third = nr_dl_prs_resource_occasion(system_frame_number=1, slot_in_frame=0, timing=timing)
    fourth = nr_dl_prs_resource_occasion(system_frame_number=1, slot_in_frame=2, timing=timing)
    gap = nr_dl_prs_resource_occasion(system_frame_number=0, slot_in_frame=9, timing=timing)
    next_period = nr_dl_prs_resource_occasion(
        system_frame_number=1, slot_in_frame=6, timing=timing
    )

    assert (first.is_resource_occasion, first.transmitted, first.repetition_index) == (
        True,
        True,
        0,
    )
    assert (second.is_resource_occasion, second.repetition_index) == (True, 1)
    assert (third.is_resource_occasion, third.repetition_index) == (True, 2)
    assert (fourth.is_resource_occasion, fourth.repetition_index) == (True, 3)
    assert (gap.is_resource_occasion, gap.status) == (False, "NOT_A_RESOURCE_OCCASION")
    assert (next_period.is_resource_occasion, next_period.set_instance_index) == (True, 1)


def test_prs_resource_set_schedule_applies_option1_and_option2_muting():
    timing = NRPRSResourceSetTiming(
        numerology=0,
        periodicity_slots=8,
        periodicity_offset_slots=0,
        repetition_factor=2,
        time_gap_slots=1,
        option1_bitmap=(True, False),
        option2_bitmap=(True, False),
        muting_bit_repetition_factor=2,
    )
    first_repetition = nr_dl_prs_resource_occasion(
        system_frame_number=0, slot_in_frame=0, timing=timing
    )
    second_repetition = nr_dl_prs_resource_occasion(
        system_frame_number=0, slot_in_frame=1, timing=timing
    )
    second_set = nr_dl_prs_resource_occasion(
        system_frame_number=1, slot_in_frame=6, timing=timing
    )
    fourth_set = nr_dl_prs_resource_occasion(
        system_frame_number=2, slot_in_frame=4, timing=timing
    )

    assert (first_repetition.transmitted, first_repetition.option1_bit_index) == (True, 0)
    assert (second_repetition.transmitted, second_repetition.status) == (
        False,
        "MUTED_BY_OPTION_2",
    )
    assert (second_set.transmitted, second_set.option1_bit_index) == (False, 1)
    assert (fourth_set.transmitted, fourth_set.option1_bit_index) == (False, 1)


def test_prs_resource_set_schedule_combines_both_muting_options():
    timing = NRPRSResourceSetTiming(
        numerology=0,
        periodicity_slots=8,
        periodicity_offset_slots=0,
        repetition_factor=2,
        time_gap_slots=1,
        option1_bitmap=(False, True),
        option2_bitmap=(False, True),
    )

    result = nr_dl_prs_resource_occasion(system_frame_number=0, slot_in_frame=0, timing=timing)

    assert result.is_resource_occasion
    assert not result.transmitted
    assert result.status == "MUTED_BY_OPTION_1_AND_2"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"numerology": 4, "periodicity_slots": 16},
        {"numerology": 0, "periodicity_slots": 7},
        {"numerology": 0, "periodicity_slots": 8, "periodicity_offset_slots": 8},
        {"numerology": 0, "periodicity_slots": 8, "repetition_factor": 3},
        {"numerology": 0, "periodicity_slots": 8, "option1_bitmap": (True, False, True)},
        {
            "numerology": 0,
            "periodicity_slots": 8,
            "repetition_factor": 2,
            "option2_bitmap": (True,),
        },
        {
            "numerology": 0,
            "periodicity_slots": 10240,
            "option1_bitmap": (True, False),
        },
    ],
)
def test_prs_resource_set_timing_rejects_invalid_decoded_configurations(kwargs):
    config = {"numerology": 0, "periodicity_slots": 8, "periodicity_offset_slots": 0}
    config.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        NRPRSResourceSetTiming(**config)


def test_prs_resource_set_requires_time_gap_for_repeated_resources():
    with pytest.raises(ValueError, match="time_gap_slots is required"):
        NRPRSResourceSetTiming(
            numerology=0,
            periodicity_slots=8,
            periodicity_offset_slots=0,
            repetition_factor=2,
        )


def test_prs_resource_set_schedule_validates_sfn_and_slot_coordinates():
    timing = NRPRSResourceSetTiming(
        numerology=0,
        periodicity_slots=8,
        periodicity_offset_slots=0,
    )
    with pytest.raises(ValueError, match="system_frame_number"):
        nr_dl_prs_resource_occasion(system_frame_number=1024, slot_in_frame=0, timing=timing)
    with pytest.raises(ValueError, match="slot_in_frame"):
        nr_dl_prs_resource_occasion(system_frame_number=0, slot_in_frame=10, timing=timing)


@pytest.mark.parametrize(
    ("frequency_range", "reference_point"),
    [(1, "UE_antenna_connector"), (2, "UE_antenna")],
)
def test_nr_dl_rstd_uses_nearest_reference_and_records_frequency_range(
    frequency_range, reference_point
):
    result = measure_nr_dl_rstd(
        target_tp_id="tp-target",
        reference_tp_id="tp-reference",
        target_subframe_start_time_s=0.0102,
        reference_subframe_start_times_s=[0.009, 0.010, 0.011],
        frequency_range=frequency_range,
    )

    assert isinstance(result, NRPRSDLRSTDMeasurement)
    assert result.selected_reference_index == 1
    assert result.selected_reference_subframe_start_time_s == pytest.approx(0.010)
    assert result.rstd_s == pytest.approx(0.0002)
    assert result.measurement_reference_point == reference_point


def test_nr_dl_rstd_accepts_duplicate_resource_estimates_for_same_reference_start():
    result = measure_nr_dl_rstd(
        target_tp_id="tp-target",
        reference_tp_id="tp-reference",
        target_subframe_start_time_s=0.0102,
        reference_subframe_start_times_s=[0.010, 0.010],
        frequency_range=1,
    )

    assert result.selected_reference_index == 0
    assert result.rstd_s == pytest.approx(0.0002)


def test_nr_timing_quality_rounds_uncertainty_up_at_selected_resolution():
    quality = NRPRSTimingQuality.from_uncertainty_m(0.11, resolution_m=0.1)

    assert quality.value == 2
    assert quality.resolution_m == 0.1
    assert quality.represented_uncertainty_m == pytest.approx(0.2)
    with pytest.raises(ValueError, match="exceeds the selected"):
        NRPRSTimingQuality.from_uncertainty_m(3.2, resolution_m=0.1)


def test_dl_tdoa_report_includes_zero_reference_and_closest_reference_per_tp():
    quality = NRPRSTimingQuality(value=2, resolution_m=1.0)
    report = build_nr_dl_tdoa_measurement_report(
        reference_tp_id="tp-ref",
        reference_subframe_start_times_s=(0.009, 0.010),
        target_subframe_start_times_s={
            "tp-b": 0.0102,
            "tp-a": 0.0088,
        },
        timing_quality_by_tp={
            "tp-ref": quality,
            "tp-a": quality,
            "tp-b": quality,
        },
        frequency_range=1,
    )

    assert isinstance(report, NRPRSTDReport)
    assert report.measurement_reference_point == "UE_antenna_connector"
    assert [element.tp_id for element in report.elements] == ["tp-ref", "tp-a", "tp-b"]
    assert report.elements[0].is_reference is True
    assert report.elements[0].rstd_s == 0.0
    assert report.elements[1].selected_reference_subframe_start_time_s == pytest.approx(0.009)
    assert report.elements[1].rstd_s == pytest.approx(-0.0002)
    assert report.elements[2].selected_reference_subframe_start_time_s == pytest.approx(0.010)
    assert report.elements[2].rstd_s == pytest.approx(0.0002)
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "datasets/schema/nr-dl-tdoa-semantic-report.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(report.to_dict())


def test_dl_tdoa_report_requires_exact_quality_roster_and_rejects_ambiguous_reference():
    quality = NRPRSTimingQuality(value=0, resolution_m=0.1)
    base = {
        "reference_tp_id": "tp-ref",
        "reference_subframe_start_times_s": (0.009, 0.011),
        "target_subframe_start_times_s": {"tp-target": 0.010},
        "timing_quality_by_tp": {"tp-ref": quality, "tp-target": quality},
        "frequency_range": 2,
    }
    with pytest.raises(ValueError, match="equidistant"):
        build_nr_dl_tdoa_measurement_report(**base)
    with pytest.raises(ValueError, match="cover exactly"):
        build_nr_dl_tdoa_measurement_report(
            **{**base, "timing_quality_by_tp": {"tp-ref": quality}}
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"reference_subframe_start_times_s": []},
            "must be nonempty",
        ),
        (
            {
                "target_subframe_start_time_s": 0.010,
                "reference_subframe_start_times_s": [0.009, 0.011],
            },
            "equidistant",
        ),
        (
            {"target_tp_id": "tp-reference"},
            "must differ",
        ),
        (
            {"frequency_range": 3},
            "frequency_range",
        ),
    ],
)
def test_nr_dl_rstd_rejects_missing_ambiguous_or_malformed_inputs(kwargs, message):
    arguments: dict[str, Any] = {
        "target_tp_id": "tp-target",
        "reference_tp_id": "tp-reference",
        "target_subframe_start_time_s": 0.010,
        "reference_subframe_start_times_s": [0.009],
        "frequency_range": 1,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        measure_nr_dl_rstd(**arguments)


@pytest.mark.parametrize("bad_delay_grid", [[0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, np.nan]])
def test_prs_matched_delay_profile_rejects_invalid_delay_grid(bad_delay_grid):
    resource = map_nr_dl_prs_resource(
        slot_symbol_count=14,
        subcarrier_count=12,
        resource_start_subcarrier=0,
        resource_bandwidth_rbs=1,
        prs_sequence_id=0,
        slot_index=0,
        numerology=0,
        subcarrier_spacing_hz=15_000.0,
        comb_size=2,
        resource_element_offset=0,
        start_symbol=0,
        num_symbols=1,
    )
    with pytest.raises((TypeError, ValueError), match="delay_grid_s"):
        nr_dl_prs_matched_delay_profile(resource.grid, resource, delay_grid_s=bad_delay_grid)
