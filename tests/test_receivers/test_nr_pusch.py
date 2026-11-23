import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from receivers.nr import (
    NRChannelEstimate,
    NRPUSCHConfig,
    NRPUSCHIntraSlotHopDMRSGrid,
    build_pusch_intra_slot_hop_dmrs_grid,
    compose_pusch_intra_slot_hops,
    dehop_pusch_intra_slot_received_grid,
    derive_pusch_intra_slot_hop_dmrs_plan,
    estimate_demap_pusch_intra_slot_hop,
    map_pusch_intra_slot_hop_data_symbols,
    verify_pusch_intra_slot_dehopped_grid,
    verify_pusch_intra_slot_hop_allocation_grid,
    verify_pusch_intra_slot_hop_dmrs_grid,
    verify_pusch_intra_slot_hop_dmrs_plan,
    verify_pusch_intra_slot_hops,
)


def test_pusch_config_accepts_valid_dmrs_configuration():
    config = NRPUSCHConfig(
        n_size_bwp=24,
        subcarrier_spacing_khz=30,
        dmrs_length=2,
        dmrs_additional_position=1,
        num_cdm_groups_without_data=2,
    )
    assert config.n_size_bwp == 24
    assert config.n_size_grid == 24


def test_pusch_config_defaults_to_sionna_receiver_linear_interpolation():
    assert NRPUSCHConfig().interpolation_type == "lin"


@pytest.mark.parametrize(
    ("mapping_type", "start", "first", "second", "additional", "type_a", "expected"),
    (
        ("B", 2, 5, 5, 1, 2, ([2, 6], [7, 11])),
        ("B", 0, 4, 4, 0, 2, ([0], [4])),
        ("B", 0, 2, 2, 1, 2, ([0], [2])),
        ("A", 0, 7, 7, 1, 2, ([2, 6], [7, 11])),
        ("A", 0, 5, 6, 1, 3, ([3], [5, 9])),
    ),
)
def test_pusch_intra_slot_hop_dmrs_plan_matches_ts_38211_v192_table(
    mapping_type, start, first, second, additional, type_a, expected
):
    plan = derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type=mapping_type,
        allocation_start_symbol=start,
        first_hop_symbol_count=first,
        second_hop_symbol_count=second,
        dmrs_additional_position=additional,
        dmrs_type_a_position=type_a,
    )
    assert plan.metadata["first_hop_dmrs_symbols"] == expected[0]
    assert plan.metadata["second_hop_dmrs_symbols"] == expected[1]
    assert plan.metadata["standard"] == "3GPP TS 38.211 V19.2.0"
    assert plan.metadata["dmrs_additional_position_used_per_hop"] == min(additional, 1)
    import jsonschema

    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-hop-dmrs-plan.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(plan.metadata)
    assert verify_pusch_intra_slot_hop_dmrs_plan(plan.metadata)["status"] == (
        "PASS_TABLE_DERIVATION_REPLAY"
    )


@pytest.mark.parametrize(
    "configuration",
    (
        {
            "mapping_type": "A",
            "allocation_start_symbol": 1,
            "first_hop_symbol_count": 6,
            "second_hop_symbol_count": 6,
            "dmrs_additional_position": 1,
        },
        {
            "mapping_type": "B",
            "allocation_start_symbol": 0,
            "first_hop_symbol_count": 4,
            "second_hop_symbol_count": 5,
            "dmrs_additional_position": 1,
        },
        {
            "mapping_type": "B",
            "allocation_start_symbol": 0,
            "first_hop_symbol_count": 8,
            "second_hop_symbol_count": 6,
            "dmrs_additional_position": 1,
        },
    ),
)
def test_pusch_intra_slot_hop_dmrs_plan_rejects_out_of_table_allocations(configuration):
    with pytest.raises(ValueError):
        derive_pusch_intra_slot_hop_dmrs_plan(**configuration)


def test_pusch_intra_slot_hop_dmrs_grid_uses_absolute_symbol_gold_sequences():
    import jsonschema

    plan = derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type="B",
        allocation_start_symbol=2,
        first_hop_symbol_count=5,
        second_hop_symbol_count=5,
        dmrs_additional_position=1,
    )
    configuration = {
        "allocation_rb_count": 1,
        "dmrs_config_type": 1,
        "n_cell_id": 17,
        "dmrs_n_id": (100, 321),
        "dmrs_n_scid": 1,
        "subcarrier_spacing_khz": 30,
        "slot_index": 3,
    }
    result = build_pusch_intra_slot_hop_dmrs_grid(plan.metadata, **configuration)
    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-hop-dmrs-grid.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result.metadata)
    assert result.pilot_mask.shape == (14, 12)
    assert result.pilot_mask.dtype == np.bool_
    assert tuple(np.flatnonzero(result.pilot_mask[2])) == (0, 2, 4, 6, 8, 10)
    assert tuple(np.flatnonzero(result.pilot_mask[7])) == (0, 2, 4, 6, 8, 10)
    expected_first = np.asarray(
        [
            1 + 1j,
            1 - 1j,
            -1 - 1j,
        ],
        dtype=np.complex128,
    ) / np.sqrt(2)
    np.testing.assert_allclose(result.pilot_grid[2, result.pilot_mask[2]][:3], expected_first)
    receipt = verify_pusch_intra_slot_hop_dmrs_grid(
        plan.metadata,
        result.pilot_mask,
        result.pilot_grid,
        **configuration,
    )
    assert receipt["status"] == "PASS_DMRS_GRID_REPLAY"
    assert result.metadata["qualification"] == "NOT_QUALIFIED"

    tampered = result.pilot_grid.copy()
    tampered[2, 0] *= -1
    with pytest.raises(ValueError, match="pilot values"):
        verify_pusch_intra_slot_hop_dmrs_grid(
            plan.metadata,
            result.pilot_mask,
            tampered,
            **configuration,
        )


def test_pusch_intra_slot_hop_data_mapping_follows_frequency_then_symbol_order():
    import jsonschema

    plan = derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type="B",
        allocation_start_symbol=2,
        first_hop_symbol_count=5,
        second_hop_symbol_count=5,
        dmrs_additional_position=1,
    )
    dmrs = build_pusch_intra_slot_hop_dmrs_grid(
        plan.metadata,
        allocation_rb_count=1,
        dmrs_config_type=1,
        n_cell_id=17,
        dmrs_n_id=(100, 321),
        dmrs_n_scid=1,
        subcarrier_spacing_khz=30,
        slot_index=3,
    )
    first_data = np.arange(48, dtype=np.float64).astype(np.complex128) + 1j
    second_data = (100 + np.arange(48, dtype=np.float64)).astype(np.complex128) - 2j
    mapped = map_pusch_intra_slot_hop_data_symbols(
        plan.metadata,
        dmrs,
        first_hop_data_symbols=first_data,
        second_hop_data_symbols=second_data,
    )
    assert mapped.allocation_grid.shape == (10, 12)
    assert mapped.data_mask.dtype == np.bool_
    assert mapped.data_mask.shape == mapped.allocation_grid.shape
    first_mask = dmrs.pilot_mask[2:7]
    first_coordinates = np.argwhere((~first_mask).T)
    np.testing.assert_array_equal(
        mapped.allocation_grid[first_coordinates[:, 1], first_coordinates[:, 0]],
        first_data,
    )
    expected_data_mask = np.zeros((10, 12), dtype=np.bool_)
    expected_data_mask[:5] = ~dmrs.pilot_mask[2:7]
    expected_data_mask[5:] = ~dmrs.pilot_mask[7:12]
    np.testing.assert_array_equal(mapped.data_mask, expected_data_mask)
    assert mapped.metadata["data_mapping_order"] == (
        "increasing_frequency_subcarrier_then_ofdm_symbol_per_hop"
    )
    schema = json.loads(
        Path("datasets/schema/nr-pusch-hopped-allocation-grid.v2.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(mapped.metadata)
    replay = verify_pusch_intra_slot_hop_allocation_grid(
        plan.metadata,
        dmrs,
        mapped.allocation_grid,
        mapped.data_mask,
        mapped.metadata,
        first_hop_data_symbols=first_data,
        second_hop_data_symbols=second_data,
    )
    assert replay["status"] == "PASS_HOPPED_ALLOCATION_REPLAY"
    tampered_allocation = mapped.allocation_grid.copy()
    tampered_allocation[0, 0] += 1
    with pytest.raises(ValueError, match="does not replay"):
        verify_pusch_intra_slot_hop_allocation_grid(
            plan.metadata,
            dmrs,
            tampered_allocation,
            mapped.data_mask,
            mapped.metadata,
            first_hop_data_symbols=first_data,
            second_hop_data_symbols=second_data,
        )
    tampered_data_mask = mapped.data_mask.copy()
    tampered_data_mask[0, 0] = not tampered_data_mask[0, 0]
    with pytest.raises(ValueError, match="data mask does not replay"):
        verify_pusch_intra_slot_hop_allocation_grid(
            plan.metadata,
            dmrs,
            mapped.allocation_grid,
            tampered_data_mask,
            mapped.metadata,
            first_hop_data_symbols=first_data,
            second_hop_data_symbols=second_data,
        )

    carrier = compose_pusch_intra_slot_hops(
        mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis],
        carrier_grid_size_rb=24,
        carrier_grid_start_rb=10,
        bwp_start_rb=12,
        bwp_size_rb=12,
        allocation_start_symbol=2,
        first_hop_rb_start=0,
        second_hop_rb_start=7,
        first_hop_symbol_count=5,
    )
    assert carrier.metadata["receiver_hopping_qualification"] == "NOT_QUALIFIED"


def _build_hopped_estimator_fixture():
    plan = derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type="B",
        allocation_start_symbol=2,
        first_hop_symbol_count=5,
        second_hop_symbol_count=5,
        dmrs_additional_position=1,
    )
    dmrs = build_pusch_intra_slot_hop_dmrs_grid(
        plan.metadata,
        allocation_rb_count=1,
        dmrs_config_type=1,
        n_cell_id=17,
        dmrs_n_id=(100, 321),
        dmrs_n_scid=1,
        subcarrier_spacing_khz=30,
        slot_index=3,
    )
    first_data = (1 + np.arange(48, dtype=np.float64)).astype(np.complex128) + 0.25j
    second_data = (101 + np.arange(48, dtype=np.float64)).astype(np.complex128) - 0.5j
    mapped = map_pusch_intra_slot_hop_data_symbols(
        plan.metadata,
        dmrs,
        first_hop_data_symbols=first_data,
        second_hop_data_symbols=second_data,
    )
    return plan, dmrs, mapped, first_data, second_data


def test_pusch_hop_estimator_demapper_round_trips_mapper_order_on_flat_channel():
    plan, dmrs, mapped, first_data, second_data = _build_hopped_estimator_fixture()
    channel = np.complex128(0.75 + 0.5j)
    received = (mapped.allocation_grid * channel)[np.newaxis, np.newaxis, np.newaxis]

    result = estimate_demap_pusch_intra_slot_hop(
        plan.metadata,
        dmrs,
        data_mask=mapped.data_mask,
        allocation_metadata=mapped.metadata,
        received_allocation_grid=received,
        noise_variance=0.0,
    )

    expected_data = np.concatenate((first_data, second_data)) * channel
    np.testing.assert_allclose(result.received_data_symbols[0, 0, 0], expected_data)
    np.testing.assert_allclose(result.channel_estimate, channel)
    np.testing.assert_array_equal(result.error_variance, 0.0)
    assert result.received_data_symbols.shape == (1, 1, 1, 96)
    assert result.channel_estimate.shape == result.received_data_symbols.shape
    assert result.metadata["allocation_start_symbol"] == 2
    assert result.metadata["allocation_symbol_count"] == 10
    assert result.metadata["active_allocation_shape"] == [10, 12]
    assert result.metadata["data_mapping_order"] == mapped.metadata["data_mapping_order"]
    assert result.metadata["qualification"] == "NOT_QUALIFIED"
    assert not result.received_data_symbols.flags.writeable
    assert not result.channel_estimate.flags.writeable
    assert not result.error_variance.flags.writeable


def test_pusch_hop_estimator_demapper_broadcasts_noise_and_preserves_inputs():
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = np.broadcast_to(
        mapped.allocation_grid,
        (2, 1, 3, *mapped.allocation_grid.shape),
    ).copy()
    mask_before = mapped.data_mask.copy()
    dmrs_mask_before = dmrs.pilot_mask.copy()
    dmrs_grid_before = dmrs.pilot_grid.copy()
    received_before = received.copy()
    metadata_before = dict(mapped.metadata)

    result = estimate_demap_pusch_intra_slot_hop(
        plan.metadata,
        dmrs,
        data_mask=mapped.data_mask,
        allocation_metadata=mapped.metadata,
        received_allocation_grid=received,
        noise_variance=np.asarray([0.25, 0.5]),
    )

    assert result.error_variance.shape == (2, 1, 3, 96)
    np.testing.assert_allclose(result.error_variance[0], 0.25)
    np.testing.assert_allclose(result.error_variance[1], 0.5)
    np.testing.assert_array_equal(mapped.data_mask, mask_before)
    np.testing.assert_array_equal(dmrs.pilot_mask, dmrs_mask_before)
    np.testing.assert_array_equal(dmrs.pilot_grid, dmrs_grid_before)
    np.testing.assert_array_equal(received, received_before)
    assert dict(mapped.metadata) == metadata_before


def test_pusch_hop_estimator_demapper_rejects_mask_and_replay_mismatches():
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]

    punctured = mapped.data_mask.copy()
    punctured[0, 1] = False
    with pytest.raises(ValueError, match="puncturing or muting"):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=punctured,
            allocation_metadata=mapped.metadata,
            received_allocation_grid=received,
            noise_variance=0.0,
        )

    wrong_digest = dict(mapped.metadata)
    wrong_digest["data_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=wrong_digest,
            received_allocation_grid=received,
            noise_variance=0.0,
        )

    wrong_span = dict(mapped.metadata)
    wrong_span["allocation_start_symbol"] = 1
    with pytest.raises(ValueError, match="active allocation span"):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=wrong_span,
            received_allocation_grid=received,
            noise_variance=0.0,
        )


@pytest.mark.parametrize("digest_key", ("pilot_mask_sha256", "pilot_grid_sha256"))
def test_pusch_hop_estimator_demapper_rejects_tampered_dmrs_metadata_digest(digest_key):
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]
    tampered_dmrs = NRPUSCHIntraSlotHopDMRSGrid(
        dmrs.pilot_mask,
        dmrs.pilot_grid,
        {**dmrs.metadata, digest_key: "0" * 64},
    )
    tampered_allocation = dict(mapped.metadata)
    if digest_key == "pilot_grid_sha256":
        tampered_allocation["dmrs_grid_sha256"] = "0" * 64

    with pytest.raises(ValueError, match=f"DM-RS {digest_key} metadata does not match replay"):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            tampered_dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=tampered_allocation,
            received_allocation_grid=received,
            noise_variance=0.0,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("data_mapping_order", "tampered_order"),
        ("first_hop_data_symbol_count", 47),
        ("second_hop_data_symbol_count", 47),
        ("data_mapping_clause", "tampered_clause"),
    ),
)
def test_pusch_hop_estimator_demapper_rejects_tampered_mapper_contract(field, value):
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]
    tampered = {**mapped.metadata, field: value}

    with pytest.raises(ValueError, match="mapper contract"):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=tampered,
            received_allocation_grid=received,
            noise_variance=0.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"num_cdm_groups_without_data": 2}, "CDM groups"),
        ({"pt_rs_enabled": True}, "PT-RS"),
        ({"puncture_mask": np.eye(10, 12, dtype=np.bool_)}, "punctured REs"),
    ),
)
def test_pusch_hop_estimator_demapper_rejects_unsupported_re_modes(kwargs, message):
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]
    with pytest.raises(ValueError, match=message):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=mapped.metadata,
            received_allocation_grid=received,
            noise_variance=0.0,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda received: received.real, "complex64 or complex128"),
        (lambda received: received[..., :-1, :], "active allocation span"),
        (
            lambda received: np.where(
                np.indices(received.shape)[-1] == 0,
                np.complex128(np.nan + 0j),
                received,
            ),
            "finite",
        ),
    ),
)
def test_pusch_hop_estimator_demapper_rejects_invalid_received_grid(mutator, message):
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]
    with pytest.raises(ValueError, match=message):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=mapped.metadata,
            received_allocation_grid=mutator(received),
            noise_variance=0.0,
        )


@pytest.mark.parametrize(
    ("noise", "exception", "message"),
    (
        (1 + 0j, TypeError, "real numeric"),
        (-0.1, ValueError, "nonnegative"),
        (np.nan, ValueError, "finite"),
        (np.ones((2, 2, 2, 2)), ValueError, "at most three"),
    ),
)
def test_pusch_hop_estimator_demapper_rejects_invalid_noise(noise, exception, message):
    plan, dmrs, mapped, _, _ = _build_hopped_estimator_fixture()
    received = mapped.allocation_grid[np.newaxis, np.newaxis, np.newaxis]
    with pytest.raises(exception, match=message):
        estimate_demap_pusch_intra_slot_hop(
            plan.metadata,
            dmrs,
            data_mask=mapped.data_mask,
            allocation_metadata=mapped.metadata,
            received_allocation_grid=received,
            noise_variance=noise,
        )


def test_pusch_config_accepts_bwp_offset_inside_wider_carrier_grid():
    config = NRPUSCHConfig(
        n_size_bwp=6,
        n_size_grid=24,
        n_start_grid=2,
        n_start_bwp=7,
    )
    assert config.n_start_bwp - config.n_start_grid == 5


def test_pusch_config_keeps_legacy_positional_field_order():
    config = NRPUSCHConfig(6, 15)
    assert config.n_size_bwp == 6
    assert config.subcarrier_spacing_khz == 15
    assert config.carrier_grid_size == 6


def test_pusch_intra_slot_hops_map_bwp_segments_to_absolute_carrier_crbs():
    bwp_grid = np.ones((1, 1, 1, 6, 24), dtype=np.complex64)
    bwp_grid[..., 3:, :] = 2 + 1j

    result = compose_pusch_intra_slot_hops(
        bwp_grid,
        carrier_grid_size_rb=52,
        carrier_grid_start_rb=0,
        bwp_start_rb=8,
        bwp_size_rb=24,
        allocation_start_symbol=4,
        first_hop_rb_start=2,
        second_hop_rb_start=14,
        first_hop_symbol_count=3,
    )

    assert result.frequency_grid.shape == (1, 1, 1, 14, 624)
    np.testing.assert_array_equal(result.frequency_grid[..., 4:7, 120:144], 1.0)
    np.testing.assert_array_equal(result.frequency_grid[..., 7:10, 264:288], 2 + 1j)
    assert np.count_nonzero(result.frequency_grid[..., :4, :]) == 0
    assert np.count_nonzero(result.frequency_grid[..., 10:, :]) == 0
    assert result.metadata["first_hop_start_crb"] == 10
    assert result.metadata["second_hop_start_crb"] == 22
    assert result.metadata["receiver_hopping_qualification"] == "NOT_QUALIFIED"

    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-hop-grid.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result.metadata)
    replay = verify_pusch_intra_slot_hops(bwp_grid, result.frequency_grid, result.metadata)
    assert replay["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert replay["qualification_status"] == "NOT_QUALIFIED"
    replay_schema = json.loads(
        Path("datasets/schema/nr-pusch-hop-grid-replay.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(replay_schema).validate(replay)
    with pytest.raises(TypeError, match="carrier_slot_grid must be a NumPy array"):
        verify_pusch_intra_slot_hops(
            bwp_grid,
            result.frequency_grid.tolist(),
            result.metadata,
            max_output_elements=1,
        )
    malformed_shape_metadata = dict(result.metadata)
    malformed_shape_metadata["carrier_grid_shape"] = [1, 1, 1, 13, 624]
    with pytest.raises(ValueError, match="shape differs from replay metadata"):
        verify_pusch_intra_slot_hops(
            bwp_grid,
            result.frequency_grid,
            malformed_shape_metadata,
            max_output_elements=1,
        )
    with pytest.raises(ValueError, match="input and output exceed max_output_elements"):
        verify_pusch_intra_slot_hops(
            bwp_grid,
            result.frequency_grid,
            result.metadata,
            max_output_elements=100,
        )
    changed_grid = result.frequency_grid.copy()
    changed_grid[0, 0, 0, 4, 120] += 1
    with pytest.raises(ValueError, match="carrier grid does not replay"):
        verify_pusch_intra_slot_hops(bwp_grid, changed_grid, result.metadata)

    changed_metadata = dict(result.metadata)
    changed_metadata["second_hop_start_crb"] += 1
    with pytest.raises(ValueError, match="metadata does not replay"):
        verify_pusch_intra_slot_hops(bwp_grid, result.frequency_grid, changed_metadata)


def test_pusch_dehop_rehomes_received_segments_to_first_allocation():
    source = np.zeros((1, 1, 1, 6, 24), dtype=np.complex64)
    source[..., :3, :] = 1.0 + 2.0j
    source[..., 3:, :] = 3.0 + 4.0j
    composed = compose_pusch_intra_slot_hops(
        source,
        carrier_grid_size_rb=52,
        carrier_grid_start_rb=0,
        bwp_start_rb=8,
        bwp_size_rb=24,
        allocation_start_symbol=4,
        first_hop_rb_start=2,
        second_hop_rb_start=14,
        first_hop_symbol_count=3,
    )

    received = composed.frequency_grid.copy()
    received[..., 4:7, 120:144] *= 0.5 - 0.25j
    received[..., 7:10, 264:288] *= 0.75 + 0.125j
    result = dehop_pusch_intra_slot_received_grid(received, composed.metadata)

    assert result.frequency_grid.shape == received.shape
    np.testing.assert_array_equal(result.frequency_grid[..., 4:7, 120:144], 1.0 + 0.75j)
    np.testing.assert_array_equal(result.frequency_grid[..., 7:10, 120:144], 1.75 + 3.375j)
    assert np.count_nonzero(result.frequency_grid[..., :4, :]) == 0
    assert np.count_nonzero(result.frequency_grid[..., 10:, :]) == 0
    assert np.count_nonzero(result.frequency_grid[..., 4:10, :120]) == 0
    assert result.metadata["receiver_bwp_start_rb"] == 10
    assert result.metadata["receiver_bwp_size_rb"] == 2
    assert result.metadata["receiver_hopping_qualification"] == "NOT_QUALIFIED"
    import json
    from pathlib import Path

    import jsonschema

    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-dehopped-grid.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result.metadata)
    replay = verify_pusch_intra_slot_dehopped_grid(
        received, composed.metadata, result.frequency_grid, result.metadata
    )
    replay_schema = json.loads(
        Path("datasets/schema/nr-pusch-dehopped-grid-replay.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(replay_schema).validate(replay)
    assert replay["qualification_status"] == "NOT_QUALIFIED"

    tampered = result.frequency_grid.copy()
    tampered[..., 4, 120] += 1
    with pytest.raises(ValueError, match="does not replay"):
        verify_pusch_intra_slot_dehopped_grid(
            received, composed.metadata, tampered, result.metadata
        )


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({"hop_boundary_symbol": 9}, "invalid symbol layout"),
        ({"first_hop_start_crb": 0}, "fit within the declared BWP"),
        ({"allocation_grid_shape": [1, 1, 1, 6, 12]}, "shapes disagree"),
        ({"carrier_grid_shape": [2, 1, 1, 14, 624]}, "shapes disagree"),
        ({"receiver_hopping_qualification": "PASS"}, "unqualified status"),
    ],
)
def test_pusch_dehop_rejects_inconsistent_layout_metadata(metadata_change, message):
    allocation = np.ones((1, 1, 1, 6, 24), dtype=np.complex64)
    composed = compose_pusch_intra_slot_hops(
        allocation,
        carrier_grid_size_rb=52,
        carrier_grid_start_rb=0,
        bwp_start_rb=8,
        bwp_size_rb=24,
        allocation_start_symbol=4,
        first_hop_rb_start=2,
        second_hop_rb_start=14,
        first_hop_symbol_count=3,
    )
    metadata = {**composed.metadata, **metadata_change}
    with pytest.raises(ValueError, match=message):
        dehop_pusch_intra_slot_received_grid(composed.frequency_grid, metadata)


@pytest.mark.parametrize(
    ("carrier_start", "carrier_size", "bwp_start", "bwp_size", "expected_offset"),
    [
        (2199, 4, 2200, 2, 12),
        (2199, 275, 2472, 2, 273 * 12),
    ],
)
def test_pusch_intra_slot_hops_support_absolute_crbs_above_274(
    carrier_start, carrier_size, bwp_start, bwp_size, expected_offset
):
    import json
    from pathlib import Path

    import jsonschema

    allocation = np.ones((1, 1, 1, 4, 12), dtype=np.complex64)
    result = compose_pusch_intra_slot_hops(
        allocation,
        carrier_grid_size_rb=carrier_size,
        carrier_grid_start_rb=carrier_start,
        bwp_start_rb=bwp_start,
        bwp_size_rb=bwp_size,
        allocation_start_symbol=2,
        first_hop_rb_start=0,
        second_hop_rb_start=1,
        first_hop_symbol_count=2,
    )

    assert result.metadata["first_hop_start_crb"] == bwp_start
    assert result.metadata["second_hop_start_crb"] == bwp_start + 1
    np.testing.assert_array_equal(
        result.frequency_grid[..., 2:4, expected_offset : expected_offset + 12], 1.0
    )
    np.testing.assert_array_equal(
        result.frequency_grid[..., 4:6, expected_offset + 12 : expected_offset + 24], 1.0
    )
    schema = json.loads(
        Path("datasets/schema/nr-pusch-intra-slot-hop-grid.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(result.metadata)
    verify_pusch_intra_slot_hops(allocation, result.frequency_grid, result.metadata)


def test_pusch_intra_slot_hops_bounds_allocation_before_creating_carrier_grid(monkeypatch):
    import receivers.nr.pusch as pusch

    bwp_grid = np.ones((1, 1, 1, 6, 24), dtype=np.complex64)
    options = {
        "carrier_grid_size_rb": 52,
        "carrier_grid_start_rb": 0,
        "bwp_start_rb": 8,
        "bwp_size_rb": 24,
        "allocation_start_symbol": 4,
        "first_hop_rb_start": 2,
        "second_hop_rb_start": 14,
        "first_hop_symbol_count": 3,
    }

    def unexpected_allocation(*args, **kwargs):
        raise AssertionError("resource rejection must precede output allocation")

    monkeypatch.setattr(np, "zeros", unexpected_allocation)
    with pytest.raises(TypeError, match="must be a NumPy array"):
        pusch.compose_pusch_intra_slot_hops(bwp_grid.tolist(), **options)
    with pytest.raises(ValueError, match="input and output exceed max_output_elements"):
        pusch.compose_pusch_intra_slot_hops(bwp_grid, **options, max_output_elements=100)


def test_pusch_intra_slot_hops_hashes_strided_inputs_without_dtype_or_order_drift():
    base = np.arange(1 * 1 * 1 * 6 * 48, dtype=np.float32).reshape(1, 1, 1, 6, 48)
    complex_grid = (base + 1j * base).astype(np.complex64)
    strided_grid = complex_grid[..., ::2]
    options = {
        "carrier_grid_size_rb": 52,
        "carrier_grid_start_rb": 0,
        "bwp_start_rb": 8,
        "bwp_size_rb": 24,
        "allocation_start_symbol": 4,
        "first_hop_rb_start": 2,
        "second_hop_rb_start": 14,
        "first_hop_symbol_count": 3,
    }
    expected_source_digest = hashlib.sha256(
        np.ascontiguousarray(strided_grid).tobytes()
    ).hexdigest()

    result = compose_pusch_intra_slot_hops(strided_grid, **options)

    assert result.metadata["source_grid_sha256"] == expected_source_digest
    assert (
        result.metadata["carrier_grid_sha256"]
        == hashlib.sha256(np.ascontiguousarray(result.frequency_grid).tobytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("changes", "exception"),
    [
        ({"first_hop_symbol_count": 0}, ValueError),
        ({"second_hop_rb_start": 23}, ValueError),
        ({"allocation_start_symbol": 9}, ValueError),
        ({"carrier_grid_size_rb": True}, TypeError),
        ({"max_output_elements": True}, ValueError),
    ],
)
def test_pusch_intra_slot_hop_composition_rejects_invalid_layout(changes, exception):
    options = {
        "carrier_grid_size_rb": 52,
        "carrier_grid_start_rb": 0,
        "bwp_start_rb": 8,
        "bwp_size_rb": 24,
        "allocation_start_symbol": 4,
        "first_hop_rb_start": 2,
        "second_hop_rb_start": 14,
        "first_hop_symbol_count": 3,
    }
    options.update(changes)
    with pytest.raises(exception):
        compose_pusch_intra_slot_hops(np.ones((1, 1, 1, 6, 24), dtype=np.complex64), **options)


def test_pusch_config_normalizes_json_symbol_allocation_to_immutable_tuple():
    config = NRPUSCHConfig(mapping_type="B", symbol_allocation=[3, 10])
    assert config.symbol_allocation == (3, 10)
    assert isinstance(config.symbol_allocation, tuple)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_size_bwp": 0},
        {"subcarrier_spacing_khz": 20},
        {"num_antenna_ports": 3},
        {"num_layers": 2},
        {"dmrs_config_type": 3},
        {"dmrs_length": 3},
        {"dmrs_length": 2, "dmrs_additional_position": 2},
        {"num_cdm_groups_without_data": 0},
        {"n_cell_id": 1008},
        {"dmrs_n_id": (1,)},
        {"dmrs_n_id": (1, 65536)},
        {"dmrs_n_scid": 2},
        {"mapping_type": "C"},
        {"symbol_allocation": (3,)},
        {"symbol_allocation": (False, 10)},
        {"symbol_allocation": (-1, 10)},
        {"symbol_allocation": (3, 12)},
        {"mapping_type": "A", "symbol_allocation": (1, 13)},
        {"precoding": "unknown"},
        {"interpolation_type": "spline"},
        {"n_size_bwp": True},
        {"n_size_grid": 0},
        {"n_size_grid": 276},
        {"n_size_grid": True},
        {"n_start_grid": -1},
        {"n_start_grid": True},
        {"n_start_bwp": -1},
        {"n_start_bwp": 2474},
        {"n_start_bwp": 1, "n_start_grid": 2},
        {"n_size_bwp": 6, "n_size_grid": 24, "n_start_bwp": 19},
    ],
)
def test_pusch_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        NRPUSCHConfig(**kwargs)


def test_nr_channel_estimate_preserves_named_axis_contract():
    h_hat = np.zeros((2, 1, 1, 1, 1, 14, 24), dtype=np.complex64)
    result = NRChannelEstimate(
        h_hat=h_hat,
        error_variance=np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        pilot_mask=np.zeros((1, 1, 14, 24), dtype=bool),
        metadata={"sionna_version": "2.2.0"},
    )
    assert result.h_hat.shape == (2, 1, 1, 1, 1, 14, 24)


def test_nr_channel_estimate_rejects_non_broadcastable_variance():
    h_hat = np.zeros((2, 1, 1, 1, 1, 14, 24), dtype=np.complex64)
    with pytest.raises(ValueError, match="broadcastable"):
        NRChannelEstimate(
            h_hat=h_hat,
            error_variance=np.zeros((3, 1, 1, 1, 1, 14, 24), dtype=np.float32),
            pilot_mask=np.zeros((1, 1, 14, 24), dtype=bool),
            metadata={"sionna_version": "2.2.0"},
        )


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("h_hat", np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32), TypeError, "complex"),
        (
            "h_hat",
            np.full((1, 1, 1, 1, 1, 14, 24), np.nan + 0j, dtype=np.complex64),
            ValueError,
            "finite",
        ),
        (
            "error_variance",
            np.full((1, 1, 1, 1, 1, 14, 24), -1.0, dtype=np.float32),
            ValueError,
            "nonnegative",
        ),
        (
            "pilot_mask",
            np.zeros((1, 1, 14, 24), dtype=np.int32),
            TypeError,
            "Boolean",
        ),
        ("h_hat", np.zeros((1, 1, 1, 1, 1, 14, 0), dtype=np.complex64), ValueError, "non-empty"),
    ],
)
def test_nr_channel_estimate_rejects_invalid_array_domains(field, value, exception, message):
    kwargs = {
        "h_hat": np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.complex64),
        "error_variance": np.zeros((1, 1, 1, 1, 1, 14, 24), dtype=np.float32),
        "pilot_mask": np.zeros((1, 1, 14, 24), dtype=bool),
        "metadata": {"sionna_version": "2.2.0"},
    }
    kwargs[field] = value
    with pytest.raises(exception, match=message):
        NRChannelEstimate(**kwargs)
