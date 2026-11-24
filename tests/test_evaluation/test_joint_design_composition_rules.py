"""Joint sample-size design composition: component identity and non-qualification checks.

Only the component-level admission rules are exercised here, in the order the composer
applies them. Full planner outputs are covered by the sample-size planning tests.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family_v4
from evaluation.sample_size.joint_design import compose_joint_sample_size_design

COMPONENTS = {
    "joint": "loslab.joint_sample_size_planning.v1",
    "material_assurance": "loslab.bootstrap_assurance_planning.v1",
    "complex_csi_assurance": "loslab.complex_csi_bootstrap_assurance_planning.v1",
    "propagation_assurance": "loslab.propagation_bootstrap_assurance_planning.v1",
    "receiver_assurance": "loslab.receiver_impact_bootstrap_assurance_planning.v1",
}


@pytest.fixture
def family() -> dict[str, Any]:
    return build_joint_acceptance_family_v4(
        family_id="composition-rules-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[("urban", "LOS", "episode-1")],
    )


def _plans(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    identity = {"family_id": family["family_id"], "family_sha256": family["family_sha256"]}
    plans: dict[str, dict[str, Any]] = {}
    for name, schema in COMPONENTS.items():
        plan: dict[str, Any] = {
            "schema": schema,
            "admission_possible": False,
            "qualification_status": "NOT_QUALIFIED",
            "joint_acceptance_family": dict(identity),
        }
        if name != "joint":
            plan["scenario_provenance_verified"] = False
        plans[name] = plan
    return plans


def _compose(family, plans, **overrides):
    arguments = {
        "joint_acceptance_family": family,
        "joint_plan": plans["joint"],
        "material_assurance_plan": plans["material_assurance"],
        "complex_csi_assurance_plan": plans["complex_csi_assurance"],
        "propagation_assurance_plan": plans["propagation_assurance"],
        "receiver_assurance_plan": plans["receiver_assurance"],
    }
    arguments.update(overrides)
    return compose_joint_sample_size_design(**arguments)


@pytest.mark.parametrize("bound", [0.0, 1.0, 1.5, float("nan"), True])
def test_target_assurance_must_be_a_probability(family, bound) -> None:
    plans = _plans(family)
    with pytest.raises(ValueError):
        _compose(family, plans, target_joint_assurance_lower_bound=bound)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_each_component_plan_must_have_its_schema(family, name) -> None:
    plans = _plans(family)
    plans[name]["schema"] = "loslab.other.v1"
    with pytest.raises(ValueError, match=f"{name} plan has an unsupported schema"):
        _compose(family, plans)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_each_component_plan_must_not_be_a_non_mapping(family, name) -> None:
    plans = _plans(family)
    plans[name] = "not-a-mapping"
    with pytest.raises(ValueError, match=f"{name} plan has an unsupported schema"):
        _compose(family, plans)


@pytest.mark.parametrize(
    "change",
    [
        {"admission_possible": True},
        {"qualification_status": "QUALIFIED"},
    ],
    ids=["admission", "qualification"],
)
@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_plans_must_stay_non_qualifying(family, name, change) -> None:
    plans = _plans(family)
    plans[name].update(change)
    with pytest.raises(ValueError, match=f"{name} plan must remain explicitly non-qualifying"):
        _compose(family, plans)


@pytest.mark.parametrize("name", [name for name in sorted(COMPONENTS) if name != "joint"])
def test_scenario_provenance_cannot_be_claimed_by_branch_plans(family, name) -> None:
    plans = _plans(family)
    plans[name]["scenario_provenance_verified"] = True
    with pytest.raises(
        ValueError, match=f"{name} plan must not claim verified scenario provenance"
    ):
        _compose(family, plans)


@pytest.mark.parametrize("name", sorted(COMPONENTS))
def test_component_plans_must_name_the_supplied_family(family, name) -> None:
    plans = _plans(family)
    plans[name] = deepcopy(plans[name])
    plans[name]["joint_acceptance_family"] = {"family_id": "other", "family_sha256": "0" * 64}
    with pytest.raises(
        ValueError, match=f"{name} plan does not match the supplied acceptance family"
    ):
        _compose(family, plans)
