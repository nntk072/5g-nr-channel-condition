"""Qualification run specification: field types, scope, waveform and path-matching rules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from evaluation.acceptance.qualification_contract import (
    build_qualification_run_contract,
    verify_qualification_run_contract,
)
from tests.test_evaluation.test_qualification_contract import _spec


def _mutated(mutate) -> dict[str, Any]:
    spec = deepcopy(_spec())
    mutate(spec)
    return spec


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        pytest.param(
            lambda s: s.update(candidate="x"),
            TypeError,
            "candidate must be an object",
            id="candidate-type",
        ),
        pytest.param(
            lambda s: s["candidate"].update(configuration="x"),
            TypeError,
            "must be an object",
            id="configuration-type",
        ),
        pytest.param(
            lambda s: s["task_scope"].update(carrier_hz=True),
            TypeError,
            "must be a finite number",
            id="carrier-bool",
        ),
        pytest.param(
            lambda s: s["task_scope"].update(channel_conditions=["LOS", "UNKNOWN"]),
            ValueError,
            "may contain only LOS and NLOS",
            id="condition",
        ),
        pytest.param(
            lambda s: s["task_scope"].update(scenario_ids=[]),
            ValueError,
            "must be a nonempty array",
            id="scenarios-empty",
        ),
        pytest.param(
            lambda s: s["task_scope"].update(scenario_ids=["a", "a"]),
            ValueError,
            "entries must be unique",
            id="scenarios-duplicate",
        ),
        pytest.param(
            lambda s: s["waveform"].update(configuration="x"),
            TypeError,
            "waveform.configuration must be an object",
            id="waveform-config",
        ),
        pytest.param(
            lambda s: s["waveform"]["configuration"].update(carrier_hz=1e9),
            ValueError,
            "waveform carrier_hz differs",
            id="waveform-carrier",
        ),
        pytest.param(
            lambda s: s["path_matching"].update(minimum_bidirectional_coverage=1.5),
            ValueError,
            "must be in \\[0, 1\\]",
            id="coverage",
        ),
        pytest.param(
            lambda s: s.update(
                joint_acceptance_family={**s["joint_acceptance_family"], "schema": "x"}
            ),
            ValueError,
            "requires acceptance-family v4|family",
            id="family-schema",
        ),
    ],
)
def test_specification_fields_are_typed_and_scoped(mutate, error, message) -> None:
    with pytest.raises(error, match=message):
        build_qualification_run_contract(_mutated(mutate))


@pytest.mark.parametrize(
    ("contract", "message"),
    [
        pytest.param("not-an-object", "must be an object", id="type"),
        pytest.param(
            {"schema": "other", "contract_sha256": "0" * 64}, "digest does not match", id="digest"
        ),
    ],
)
def test_verification_checks_the_contract_shape(contract, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        verify_qualification_run_contract(contract)


def test_qualification_contract_rejects_path_roster_and_stratum_shape_drift() -> None:
    spec = _spec()
    spec["sampling"]["path_equivalence_scenarios"] = [["site-a", "NLOS", "episode-01"]]
    with pytest.raises(ValueError, match="must exactly match the acceptance family"):
        build_qualification_run_contract(spec)

    spec = _spec()
    spec["sampling"]["planned_units_by_stratum"]["complex_csi"][0]["extra"] = True
    with pytest.raises(ValueError, match="rows must contain exactly"):
        build_qualification_run_contract(spec)

    spec = _spec()
    duplicate = deepcopy(spec["sampling"]["planned_units_by_stratum"]["complex_csi"][0])
    spec["sampling"]["planned_units_by_stratum"]["complex_csi"].append(duplicate)
    with pytest.raises(ValueError, match="contains duplicate strata"):
        build_qualification_run_contract(spec)


def test_qualification_contract_family_validator_guards_internal_rosters(monkeypatch) -> None:
    import evaluation.acceptance.qualification_contract as contract_module

    original = contract_module.validate_joint_acceptance_family
    family = _spec()["joint_acceptance_family"]
    monkeypatch.setattr(
        contract_module,
        "validate_joint_acceptance_family",
        lambda _family: {**family, "material_expected_strata": []},
    )
    with pytest.raises(ValueError, match="no expected roster for material_calibration"):
        build_qualification_run_contract(_spec())

    monkeypatch.setattr(
        contract_module,
        "validate_joint_acceptance_family",
        lambda _family: {**original(_family), "schema": "loslab.joint_acceptance_family.v3"},
    )
    with pytest.raises(ValueError, match="requires acceptance-family v4"):
        build_qualification_run_contract(_spec())
