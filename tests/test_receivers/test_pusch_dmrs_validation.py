import copy

import numpy as np
import pytest
from receivers.nr.pusch_core.dmrs import (
    build_pusch_intra_slot_hop_dmrs_grid,
    derive_pusch_intra_slot_hop_dmrs_plan,
    verify_pusch_intra_slot_hop_dmrs_grid,
    verify_pusch_intra_slot_hop_dmrs_plan,
)


def _plan():
    return derive_pusch_intra_slot_hop_dmrs_plan(
        mapping_type="B",
        allocation_start_symbol=0,
        first_hop_symbol_count=5,
        second_hop_symbol_count=5,
        dmrs_additional_position=1,
    ).metadata


def _configuration(**updates):
    return {
        "allocation_rb_count": 1,
        "dmrs_config_type": 1,
        "n_cell_id": 17,
        "dmrs_n_id": (100, 321),
        "dmrs_n_scid": 1,
        "subcarrier_spacing_khz": 30,
        "slot_index": 3,
        **updates,
    }


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("allocation_start_symbol", True, TypeError, "must be an integer"),
        ("first_hop_symbol_count", 2.0, TypeError, "must be an integer"),
        ("second_hop_symbol_count", 0, ValueError, "must be positive"),
        ("allocation_start_symbol", 14, ValueError, r"\[0, 13\]"),
        ("allocation_start_symbol", 13, ValueError, "fit in the 14-symbol slot"),
        ("dmrs_additional_position", 4, ValueError, r"\[0, 3\]"),
        ("dmrs_type_a_position", 1, ValueError, "must be 2 or 3"),
        ("mapping_type", "C", ValueError, "mapping_type"),
        ("mapping_type", "A", ValueError, "slot-start"),
        ("mapping_type", "A", ValueError, "shorter than 4 symbols"),
    ],
)
def test_dmrs_plan_rejects_invalid_configuration(field, value, error, message):
    configuration = {
        "mapping_type": "B",
        "allocation_start_symbol": 0,
        "first_hop_symbol_count": 5,
        "second_hop_symbol_count": 5,
        "dmrs_additional_position": 1,
        "dmrs_type_a_position": 2,
    }
    if field == "mapping_type" and value == "A" and message == "shorter than 4 symbols":
        configuration.update(first_hop_symbol_count=3, second_hop_symbol_count=3)
    if field == "mapping_type" and value == "A" and message == "slot-start":
        configuration["allocation_start_symbol"] = 1
    configuration[field] = value
    with pytest.raises(error, match=message):
        derive_pusch_intra_slot_hop_dmrs_plan(**configuration)


def test_dmrs_plan_verifier_checks_contract_and_replay():
    plan = _plan()
    with pytest.raises(TypeError, match="mapping"):
        verify_pusch_intra_slot_hop_dmrs_plan([])
    with pytest.raises(ValueError, match="fields do not match"):
        verify_pusch_intra_slot_hop_dmrs_plan({**plan, "unexpected": True})
    changed = copy.deepcopy(plan)
    changed["first_hop_dmrs_symbols"] = [1]
    with pytest.raises(ValueError, match="does not replay"):
        verify_pusch_intra_slot_hop_dmrs_plan(changed)


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("allocation_rb_count", 0, ValueError, r"\[1, 275\]"),
        ("allocation_rb_count", True, TypeError, "must be an integer"),
        ("dmrs_config_type", 3, ValueError, "dmrs_config_type"),
        ("n_cell_id", 1008, ValueError, "n_cell_id"),
        ("dmrs_n_id", [1, 2], ValueError, "dmrs_n_id"),
        ("dmrs_n_id", (1,), ValueError, "dmrs_n_id"),
        ("dmrs_n_id", (1, True), ValueError, "dmrs_n_id"),
        ("dmrs_n_id", (-1, 2), ValueError, "dmrs_n_id"),
        ("dmrs_n_scid", 2, ValueError, "dmrs_n_scid"),
        ("subcarrier_spacing_khz", 45, ValueError, "unsupported"),
        ("slot_index", 20, ValueError, "frame range"),
    ],
)
def test_dmrs_grid_builder_rejects_invalid_configuration(field, value, error, message):
    with pytest.raises(error, match=message):
        build_pusch_intra_slot_hop_dmrs_grid(_plan(), **_configuration(**{field: value}))


def test_dmrs_grid_verifier_checks_array_types_and_mask_replay():
    plan = _plan()
    configuration = _configuration()
    result = build_pusch_intra_slot_hop_dmrs_grid(plan, **configuration)
    with pytest.raises(TypeError, match="NumPy Boolean"):
        verify_pusch_intra_slot_hop_dmrs_grid(
            plan, result.pilot_mask.astype(np.int8), result.pilot_grid, **configuration
        )
    with pytest.raises(TypeError, match="NumPy complex128"):
        verify_pusch_intra_slot_hop_dmrs_grid(
            plan, result.pilot_mask, result.pilot_grid.astype(np.complex64), **configuration
        )
    changed_mask = result.pilot_mask.copy()
    changed_mask[0, 0] = False
    with pytest.raises(ValueError, match="pilot mask"):
        verify_pusch_intra_slot_hop_dmrs_grid(
            plan, changed_mask, result.pilot_grid, **configuration
        )
