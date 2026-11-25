from dataclasses import replace

import numpy as np
import pytest
from receivers.nr import (
    build_pusch_intra_slot_hop_dmrs_grid,
    derive_pusch_intra_slot_hop_dmrs_plan,
    map_pusch_intra_slot_hop_data_symbols,
    verify_pusch_intra_slot_hop_allocation_grid,
)
from receivers.nr.pusch_core.mapping import (
    _nearest_sorted_positions,
    _normalized_hop_noise_variance,
)


def build_mapping_fixture():
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
    first = np.arange(48, dtype=np.complex128) + 1j
    second = np.arange(48, dtype=np.complex128) - 2j
    return plan.metadata, dmrs, first, second


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("first_hop_data_symbols", [1 + 1j], TypeError, "one-dimensional NumPy array"),
        (
            "first_hop_data_symbols",
            np.ones((1, 48), dtype=np.complex128),
            TypeError,
            "one-dimensional",
        ),
        ("first_hop_data_symbols", np.ones(48), ValueError, "complex64 or complex128"),
        ("first_hop_data_symbols", np.array([np.inf + 0j]), ValueError, "finite values"),
        (
            "second_hop_data_symbols",
            np.arange(48, dtype=np.complex64),
            ValueError,
            "same complex dtype",
        ),
        ("max_output_elements", True, ValueError, "positive integer"),
        ("max_output_elements", 0, ValueError, "positive integer"),
        ("max_output_elements", 1.5, ValueError, "positive integer"),
    ],
)
def test_pusch_data_mapper_rejects_malformed_vectors_and_limits(field, value, error, message):
    plan, dmrs, first, second = build_mapping_fixture()
    if field == "max_output_elements":
        options = {
            "first_hop_data_symbols": first,
            "second_hop_data_symbols": second,
            "max_output_elements": value,
        }
    else:
        first_data, second_data = first.copy(), second.copy()
        if field == "first_hop_data_symbols":
            first_data = value
        else:
            second_data = value
        options = {
            "first_hop_data_symbols": first_data,
            "second_hop_data_symbols": second_data,
        }

    with pytest.raises(error, match=message):
        map_pusch_intra_slot_hop_data_symbols(plan, dmrs, **options)


def test_pusch_data_mapper_enforces_output_budget_and_dmrs_grid_type():
    plan, dmrs, first, second = build_mapping_fixture()
    with pytest.raises(ValueError, match="exceed max_output_elements"):
        map_pusch_intra_slot_hop_data_symbols(
            plan,
            dmrs,
            first_hop_data_symbols=first,
            second_hop_data_symbols=second,
            max_output_elements=100,
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("schema", "unsupported hop DM-RS grid metadata"),
        ("dmrs_plan_sha256", "different timing plan"),
        ("pilot_mask_sha256", "metadata does not match replay"),
    ],
)
def test_pusch_data_mapper_rejects_dmrs_metadata_mismatch(change, message):
    plan, dmrs, first, second = build_mapping_fixture()
    metadata = dict(dmrs.metadata)
    metadata[change] = "unsupported" if change == "schema" else "bad"
    malformed_dmrs = replace(dmrs, metadata=metadata)

    with pytest.raises(ValueError, match=message):
        map_pusch_intra_slot_hop_data_symbols(
            plan,
            malformed_dmrs,
            first_hop_data_symbols=first,
            second_hop_data_symbols=second,
        )


def test_pusch_data_mapper_rejects_data_count_that_disagrees_with_available_re():
    plan, dmrs, first, second = build_mapping_fixture()
    with pytest.raises(ValueError, match="must equal available non-DM-RS REs"):
        map_pusch_intra_slot_hop_data_symbols(
            plan,
            dmrs,
            first_hop_data_symbols=first[:-1],
            second_hop_data_symbols=second,
        )


@pytest.mark.parametrize("failure", ["shape", "undeclared-pilot"])
def test_pusch_mapper_defense_checks_reject_dmrs_grid_disagreement(monkeypatch, failure):
    import receivers.nr.pusch_core.mapping as mapping

    plan, dmrs, first, second = build_mapping_fixture()
    malformed = dmrs
    if failure == "shape":
        malformed = replace(
            dmrs,
            pilot_mask=np.zeros((14, 11), dtype=np.bool_),
            pilot_grid=np.zeros((14, 11), dtype=np.complex128),
        )
    else:
        mask = dmrs.pilot_mask.copy()
        grid = dmrs.pilot_grid.copy()
        mask[3, 0] = True
        grid[3, 0] = 1 + 0j
        malformed = replace(dmrs, pilot_mask=mask, pilot_grid=grid)

    monkeypatch.setattr(
        mapping,
        "verify_pusch_intra_slot_hop_dmrs_grid",
        lambda *_args, **_kwargs: {
            "pilot_mask_sha256": dmrs.metadata["pilot_mask_sha256"],
            "pilot_grid_sha256": dmrs.metadata["pilot_grid_sha256"],
        },
    )
    expected_message = (
        "axes do not match" if failure == "shape" else "outside its declared hop positions"
    )
    with pytest.raises(ValueError, match=expected_message):
        map_pusch_intra_slot_hop_data_symbols(
            plan,
            malformed,
            first_hop_data_symbols=first,
            second_hop_data_symbols=second,
        )

    with pytest.raises(TypeError, match="generated NRPUSCHIntraSlotHopDMRSGrid"):
        map_pusch_intra_slot_hop_data_symbols(
            plan,
            object(),
            first_hop_data_symbols=first,
            second_hop_data_symbols=second,
        )


def test_pusch_allocation_replay_rejects_wrong_input_types_and_metadata():
    plan, dmrs, first, second = build_mapping_fixture()
    mapped = map_pusch_intra_slot_hop_data_symbols(
        plan,
        dmrs,
        first_hop_data_symbols=first,
        second_hop_data_symbols=second,
    )
    args = dict(
        plan_metadata=plan,
        dmrs_grid=dmrs,
        allocation_grid=mapped.allocation_grid,
        data_mask=mapped.data_mask,
        metadata=mapped.metadata,
        first_hop_data_symbols=first,
        second_hop_data_symbols=second,
    )

    with pytest.raises(TypeError, match="allocation_grid must be a NumPy array"):
        verify_pusch_intra_slot_hop_allocation_grid(**{**args, "allocation_grid": []})
    with pytest.raises(TypeError, match="data_mask must be a Boolean"):
        verify_pusch_intra_slot_hop_allocation_grid(**{**args, "data_mask": np.ones((10, 12))})
    with pytest.raises(TypeError, match="metadata must be a mapping"):
        verify_pusch_intra_slot_hop_allocation_grid(**{**args, "metadata": []})
    changed_metadata = dict(mapped.metadata, qualification="PASS")
    with pytest.raises(ValueError, match="metadata does not replay"):
        verify_pusch_intra_slot_hop_allocation_grid(**{**args, "metadata": changed_metadata})


@pytest.mark.parametrize(
    ("noise", "leading_shape", "expected"),
    [
        (0.25, (2, 2, 3), np.full((2, 2, 3), 0.25)),
        ([0.1, 0.2], (2, 2, 1), np.array([[[0.1], [0.1]], [[0.2], [0.2]]])),
        ([[0.1, 0.2]], (2, 2, 3), np.array([[[0.1] * 3, [0.2] * 3]] * 2)),
        (np.ones((2, 2, 3)), (2, 2, 3), np.ones((2, 2, 3))),
    ],
)
def test_hop_noise_variance_broadcasts_supported_shapes(noise, leading_shape, expected):
    result = _normalized_hop_noise_variance(noise, leading_shape=leading_shape)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("noise", "leading_shape", "error", "message"),
    [
        (True, (1, 1, 1), TypeError, "real numeric values"),
        (1 + 1j, (1, 1, 1), TypeError, "real numeric values"),
        ("0.1", (1, 1, 1), TypeError, "real numeric values"),
        (np.ones((1, 1, 1, 1)), (1, 1, 1), ValueError, "at most three dimensions"),
        ([0.1, 0.2, 0.3], (2, 1, 1), ValueError, "scalar-like or per batch"),
        ([[0.1, 0.2, 0.3]], (1, 2, 1), ValueError, r"broadcast over \[batch, rx\]"),
        (np.ones((2, 2, 2)), (2, 2, 1), ValueError, "must broadcast over"),
        (-1.0, (1, 1, 1), ValueError, "finite and nonnegative"),
        (np.nan, (1, 1, 1), ValueError, "finite and nonnegative"),
    ],
)
def test_hop_noise_variance_rejects_invalid_shapes_and_values(noise, leading_shape, error, message):
    with pytest.raises(error, match=message):
        _normalized_hop_noise_variance(noise, leading_shape=leading_shape)


def test_nearest_dmrs_positions_prefers_lower_ties_and_rejects_empty_sources():
    source = np.array([2, 8, 13])
    targets = np.array([0, 5, 10, 16])
    np.testing.assert_array_equal(_nearest_sorted_positions(source, targets), [2, 2, 8, 13])
    with pytest.raises(ValueError, match="at least one DM-RS position"):
        _nearest_sorted_positions(np.array([], dtype=np.int64), np.array([1]))
    with pytest.raises(ValueError, match="at least one DM-RS position"):
        _nearest_sorted_positions(np.array([[1]]), np.array([1]))
