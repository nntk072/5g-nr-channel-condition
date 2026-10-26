from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from datasets.targets import (
    TargetDefinition as DatasetTargetDefinition,
)
from datasets.targets import (
    TargetValidity,
    TargetValue,
    target_definition_digest,
)
from evaluation.target_coverage import (
    ExpectedInventoryCell,
    TargetDefinition,
    TargetObservation,
    TargetRecord,
    audit_target_coverage,
    convert_target_map,
)

DEFINITION = TargetDefinition(
    target_id="path_loss",
    units="dB",
    definition_sha256="a" * 64,
    element_count=3,
)


def _observation(
    values: tuple[int | float | None, ...] = (0.0, 2.0, None),
    valid_mask: tuple[bool, ...] = (True, True, False),
    reason_codes: tuple[str | None, ...] = (None, None, "measurement_occluded"),
    *,
    target_id: str = "path_loss",
    units: str = "dB",
    definition_sha256: str = "a" * 64,
) -> TargetObservation:
    return TargetObservation(
        target_id=target_id,
        units=units,
        definition_sha256=definition_sha256,
        values=values,
        valid_mask=valid_mask,
        reason_codes=reason_codes,
    )


def _record(
    record_id: str,
    split_id: str = "test",
    family_id: str = "urban",
    observation: TargetObservation | None = None,
    independent_group_id: str | None = None,
) -> TargetRecord:
    return TargetRecord(
        record_id=record_id,
        split_id=split_id,
        family_id=family_id,
        targets=(_observation() if observation is None else observation,),
        independent_group_id=(record_id if independent_group_id is None else independent_group_id),
    )


def _inventory(
    *cells: tuple[str, str, str], minimum_independent_groups: int = 1
) -> tuple[ExpectedInventoryCell, ...]:
    return tuple(
        ExpectedInventoryCell(
            *cell, minimum_independent_groups=minimum_independent_groups
        )
        for cell in cells
    )


def test_reports_element_counts_reason_breakdown_and_fraction_deterministically() -> None:
    inventory = _inventory(("test", "urban", "path_loss"))
    records = (_record("r2"), _record("r1"))

    report = audit_target_coverage(records, (DEFINITION,), inventory)
    cell = report.cells[0]

    assert (cell.valid_element_count, cell.unavailable_element_count) == (4, 2)
    assert cell.independent_group_count == 2
    assert cell.minimum_independent_group_count == 1
    assert cell.coverage_fraction == pytest.approx(2 / 3)
    assert [(item.reason_code, item.element_count) for item in cell.unavailable_by_reason] == [
        ("measurement_occluded", 2)
    ]
    assert report == audit_target_coverage(tuple(reversed(records)), (DEFINITION,), inventory)


def test_target_coverage_counts_distinct_independent_groups_not_correlated_records() -> None:
    report = audit_target_coverage(
        (
            _record("r1", independent_group_id="component-1"),
            _record("r2", independent_group_id=" component-1 "),
            _record("r3", independent_group_id="component-2"),
        ),
        (DEFINITION,),
        _inventory(("test", "urban", "path_loss")),
    )

    assert report.cells[0].independent_group_count == 2


def test_target_coverage_keeps_scene_band_topology_antenna_and_task_cells_separate() -> None:
    records = (
        replace(
            _record("urban-a"),
            scene_id="scene-a",
            band_id="n78",
            topology_id="2x2",
            antenna_config_id="array-a",
            task_id="path-loss",
            independent_group_id="component-a",
        ),
        replace(
            _record("urban-b"),
            scene_id="scene-b",
            band_id="n78",
            topology_id="2x2",
            antenna_config_id="array-a",
            task_id="path-loss",
            independent_group_id="component-b",
        ),
    )
    inventory = (
        ExpectedInventoryCell(
            "test", "urban", "path_loss", 1,
            scene_id="scene-a", band_id="n78", topology_id="2x2",
            antenna_config_id="array-a", task_id="path-loss",
        ),
        ExpectedInventoryCell(
            "test", "urban", "path_loss", 1,
            scene_id="scene-b", band_id="n78", topology_id="2x2",
            antenna_config_id="array-a", task_id="path-loss",
        ),
    )

    report = audit_target_coverage(records, (DEFINITION,), inventory)

    assert [cell.scene_id for cell in report.cells] == ["scene-a", "scene-b"]
    assert [cell.independent_group_count for cell in report.cells] == [1, 1]
    assert {cell.band_id for cell in report.cells} == {"n78"}
    assert {cell.topology_id for cell in report.cells} == {"2x2"}
    assert {cell.antenna_config_id for cell in report.cells} == {"array-a"}
    assert {cell.task_id for cell in report.cells} == {"path-loss"}


def test_unspecified_target_record_cannot_fill_named_scene_inventory() -> None:
    inventory = (
        ExpectedInventoryCell(
            "test", "urban", "path_loss", 1,
            scene_id="reviewed-scene",
            band_id="n78",
            topology_id="2x2",
            antenna_config_id="array-a",
            task_id="path-loss",
        ),
    )
    with pytest.raises(ValueError, match="no declared inventory cells"):
        audit_target_coverage((_record("r1"),), (DEFINITION,), inventory)


def test_target_coverage_requires_explicit_independent_group_identity() -> None:
    with pytest.raises(TypeError, match="independent_group_id"):
        TargetRecord(
            record_id="r1",
            split_id="test",
            family_id="urban",
            targets=(_observation(),),
        )


def test_target_coverage_rejects_many_rows_that_miss_frozen_independent_group_minimum() -> None:
    records = (
        _record("r1", independent_group_id="component-1"),
        _record("r2", independent_group_id="component-1"),
        _record("r3", independent_group_id="component-1"),
    )

    with pytest.raises(ValueError, match="minimum independent-group support"):
        audit_target_coverage(
            records,
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss"), minimum_independent_groups=2),
        )


def test_zero_is_a_valid_value_and_missing_value_is_never_imputed() -> None:
    cell = audit_target_coverage(
        (
            _record(
                "r1",
                observation=_observation(
                    (0.0, None, None), (True, False, False), (None, "not_measured", "invalid")
                ),
            ),
        ),
        (DEFINITION,),
        _inventory(("test", "urban", "path_loss")),
    ).cells[0]

    assert cell.valid_element_count == 1
    assert cell.unavailable_element_count == 2
    assert cell.coverage_fraction == pytest.approx(1 / 3)
    assert [(item.reason_code, item.element_count) for item in cell.unavailable_by_reason] == [
        ("invalid", 1),
        ("not_measured", 1),
    ]


def test_missing_declared_split_family_target_cell_fails_closed() -> None:
    inventory = _inventory(
        ("test", "urban", "path_loss"),
        ("holdout", "urban", "path_loss"),
    )

    with pytest.raises(ValueError, match="no records"):
        audit_target_coverage((_record("r1"),), (DEFINITION,), inventory)


def test_duplicate_record_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate record_id"):
        audit_target_coverage(
            (_record("same"), _record("same")),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )


def test_missing_target_from_record_fails_closed() -> None:
    with pytest.raises(ValueError, match="missing inventoried targets"):
        audit_target_coverage(
            (TargetRecord("r1", "test", "urban", (), "component-r1"),),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )


def test_unknown_target_definition_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown target"):
        audit_target_coverage(
            (_record("r1", observation=_observation(target_id="unknown")),),
            (DEFINITION,),
            _inventory(("test", "urban", "unknown")),
        )


@pytest.mark.parametrize(
    ("units", "definition_sha256", "message"),
    [
        ("m", "a" * 64, "inconsistent units"),
        ("dB", "b" * 64, "inconsistent definition hash"),
    ],
)
def test_inconsistent_target_identity_fails_closed(
    units: str, definition_sha256: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_target_coverage(
            (
                _record(
                    "r1", observation=_observation(units=units, definition_sha256=definition_sha256)
                ),
            ),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )


def test_observation_shape_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        audit_target_coverage(
            (_record("r1", observation=_observation(values=(1.0, 2.0))),),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )


@pytest.mark.parametrize(
    ("values", "valid_mask", "reason_codes", "message"),
    [
        ((1.0, 2.0, 3.0), (True, True, False), (None, None, "unavailable"), "must use None"),
        ((1.0, 2.0, None), (True, True, False), (None, None, None), "reason code"),
        ((1.0, 2.0, None), (True, True, False), (None, None, ""), "nonempty string"),
        ((1.0, 2.0, float("nan")), (True, True, True), (None, None, None), "must be finite"),
    ],
)
def test_invalid_element_encodings_fail_closed(
    values: tuple[int | float | None, ...],
    valid_mask: tuple[bool, ...],
    reason_codes: tuple[str | None, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        audit_target_coverage(
            (_record("r1", observation=_observation(values, valid_mask, reason_codes)),),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )


def test_duplicate_inventory_cell_fails_closed() -> None:
    cell = ExpectedInventoryCell("test", "urban", "path_loss", 1)
    with pytest.raises(ValueError, match="duplicate expected inventory cell"):
        audit_target_coverage((_record("r1"),), (DEFINITION,), (cell, cell))


def test_invalid_definition_digest_and_duplicate_definitions_fail_closed() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        audit_target_coverage(
            (_record("r1"),),
            (replace(DEFINITION, definition_sha256="not-a-digest"),),
            _inventory(("test", "urban", "path_loss")),
        )
    with pytest.raises(ValueError, match="duplicate target definition"):
        audit_target_coverage(
            (_record("r1"),),
            (DEFINITION, DEFINITION),
            _inventory(("test", "urban", "path_loss")),
        )


def test_non_inventoried_assignment_and_target_fail_closed() -> None:
    with pytest.raises(ValueError, match="no declared inventory cells"):
        audit_target_coverage(
            (_record("r1", split_id="train"),),
            (DEFINITION,),
            _inventory(("test", "urban", "path_loss")),
        )

    two_targets = (
        DEFINITION,
        TargetDefinition("delay", "ns", "c" * 64, 1),
    )
    unexpected = TargetRecord(
        "r2",
        "test",
        "urban",
        (
            _observation(),
            TargetObservation("delay", "ns", "c" * 64, (None,), (False,), ("missing",)),
        ),
        independent_group_id="component-r2",
    )
    with pytest.raises(ValueError, match="not inventoried"):
        audit_target_coverage(
            (unexpected,),
            two_targets,
            _inventory(("test", "urban", "path_loss")),
        )


def _dataset_definition() -> DatasetTargetDefinition:
    fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "complex_baseband_voltage_phase_gauge_v1",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    return DatasetTargetDefinition(
        **fields,
        definition_sha256=target_definition_digest(**fields),
    )


def test_target_map_bridge_preserves_canonical_definition_and_maps_unavailable_values() -> None:
    dataset_definition = _dataset_definition()
    target_value = TargetValue(
        entity_id="link-1",
        definition=dataset_definition,
        values=np.asarray([[1.5e-9, 0.0], [0.0, 4.5e-9]]),
        valid_mask=np.asarray([[True, False], [False, True]]),
        reason_codes=(
            TargetValidity.VALID,
            TargetValidity.NOT_OBSERVABLE,
            TargetValidity.CENSORED,
            TargetValidity.VALID,
        ),
        confidence="rt_solver_truth",
    )

    definitions, observations = convert_target_map({"delay_spread": target_value})

    assert definitions == (
        TargetDefinition(
            target_id="delay_spread",
            units="s",
            definition_sha256=dataset_definition.definition_sha256,
            element_count=4,
        ),
    )
    assert observations == (
        TargetObservation(
            target_id="delay_spread",
            units="s",
            definition_sha256=dataset_definition.definition_sha256,
            values=(1.5e-9, None, None, 4.5e-9),
            valid_mask=(True, False, False, True),
            reason_codes=(None, "not_observable", "censored", None),
        ),
    )


@pytest.mark.parametrize(
    ("field", "tampered_value"),
    [
        ("reference", "different_phase_gauge_v2"),
        ("aggregation_rule", "maximum over valid paths"),
    ],
)
def test_target_map_bridge_revalidates_gauge_and_aggregation_digest(
    field: str, tampered_value: str
) -> None:
    target_value = TargetValue(
        entity_id="link-1",
        definition=_dataset_definition(),
        values=np.asarray([2e-9]),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="rt_solver_truth",
    )
    tampered_fields = {
        "semantic": "rms_delay_spread",
        "units": "s",
        "source": "path_oracle",
        "reference": "complex_baseband_voltage_phase_gauge_v1",
        "aggregation_rule": "RMS over valid paths after power normalization",
        "definition_version": "umcd.rms_delay_spread.v1",
    }
    tampered_fields[field] = tampered_value
    tampered_definition = DatasetTargetDefinition(
        **tampered_fields,
        definition_sha256=target_value.definition.definition_sha256,
    )
    object.__setattr__(target_value, "definition", tampered_definition)

    with pytest.raises(ValueError, match="does not match its canonical fields"):
        convert_target_map({"delay_spread": target_value})
