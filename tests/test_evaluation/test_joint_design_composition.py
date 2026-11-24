"""Joint sample-size design composition: mutation rejections of a composed, verified design.

The composed design is built with the same branch plans as the planner's composition
test; each mutation changes one field and re-digests the design where the contract
requires it, so the named verifier rule is what rejects it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family_v4
from evaluation.sample_size_planning import (
    compose_joint_sample_size_design,
    verify_joint_sample_size_design,
)


def _composed_result() -> dict[str, Any]:
    family = build_joint_acceptance_family_v4(
        family_id="composition-v4",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
        propagation_expected_strata=[("urban", "LOS", "total")],
        receiver_impact_expected_strata=[("urban", "LOS")],
        path_equivalence_expected_scenarios=[
            ("urban", "LOS", "episode-1"),
            ("urban", "LOS", "episode-2"),
        ],
    )
    identity = {
        "family_id": family["family_id"],
        "family_sha256": family["family_sha256"],
    }
    common = {"admission_possible": False, "qualification_status": "NOT_QUALIFIED"}
    joint = {
        "schema": "loslab.joint_sample_size_planning.v1",
        **common,
        "single_combined_sample_size_is_reported": False,
        "complete_sample_size": None,
        "joint_acceptance_family": identity,
        "material_design": {
            "stratum_count": 1,
            "required_independent_measurement_units_per_stratum_for_supported_claims": 40,
            "complete_precision_powered_design": False,
            "angular_maximum_claim": {"assumption_supplied": False},
        },
        "complex_csi_design": {
            "stratum_count": 1,
            "required_independent_scene_clusters_per_stratum_for_supported_claims": 25,
            "complete_precision_powered_design": False,
        },
        "propagation_design": {
            "stratum_count": 1,
            "strata": [["urban", "LOS", "total"]],
            "required_independent_clusters_per_stratum_for_supported_claims": 30,
            "complete_precision_powered_design": False,
        },
        "receiver_impact_design": {
            "stratum_count": 1,
            "strata": [["urban", "LOS"]],
            "complete_precision_powered_design": False,
        },
    }

    def assurance(
        schema: str, stratum: list[str], count: int, key_fields: tuple[str, ...]
    ) -> dict[str, Any]:
        return {
            "schema": schema,
            **common,
            "scenario_provenance_verified": False,
            "joint_acceptance_family": identity,
            "selected_clusters_per_stratum_among_evaluated_candidates": count,
            "target_assurance": 0.8,
            "candidate_designs": [
                {
                    "candidate_clusters_per_stratum": count,
                    "joint_assurance_lower_bound_union_bound": 0.81,
                    "meets_target_assurance": True,
                    "stratum_assurance": [
                        {
                            **dict(zip(key_fields, stratum, strict=True)),
                            "cluster_count": count,
                            "assurance_one_sided_exact_lower_bound": 0.81,
                        }
                    ],
                }
            ],
        }

    result = compose_joint_sample_size_design(
        joint_acceptance_family=family,
        joint_plan=joint,
        material_assurance_plan={
            "schema": "loslab.bootstrap_assurance_planning.v1",
            **common,
            "scenario_provenance_verified": False,
            "joint_acceptance_family": identity,
            "stratum_roster_by_id": {"material-1": ["brick", "V", "normal"]},
            "selected_units_per_stratum_among_evaluated_candidates": 50,
            "target_assurance": 0.8,
            "candidate_designs": [
                {
                    "independent_units_per_stratum": 50,
                    "joint_assurance_lower_bound_union_bound": 0.81,
                    "meets_target_assurance": True,
                    "stratum_assurance": {
                        "material-1": {
                            "assumed_support_atom_count": 2,
                            "joint_assurance_one_sided_exact_lower_bound": 0.81,
                        }
                    },
                }
            ],
        },
        complex_csi_assurance_plan=assurance(
            "loslab.complex_csi_bootstrap_assurance_planning.v1",
            ["urban", "LOS"],
            35,
            ("family_id", "condition"),
        ),
        propagation_assurance_plan=assurance(
            "loslab.propagation_bootstrap_assurance_planning.v1",
            ["urban", "LOS", "total"],
            45,
            ("family_id", "condition", "component"),
        ),
        receiver_assurance_plan=assurance(
            "loslab.receiver_impact_bootstrap_assurance_planning.v1",
            ["urban", "LOS"],
            45,
            ("family_id", "condition"),
        ),
    )
    return result


def _redigest(design: dict[str, Any]) -> dict[str, Any]:
    design = dict(design)
    design.pop("design_sha256", None)
    design["design_sha256"] = hashlib.sha256(
        json.dumps(
            design, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    return design


def test_composed_design_is_schema_valid_and_verifies_as_not_qualified() -> None:
    result = _composed_result()
    schema_path = (
        Path(__file__).parents[2]
        / "datasets/schema/joint-sample-size-design-composition.v1.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(result)
    verification = verify_joint_sample_size_design(result)
    assert verification["status"] == "PASS_DESIGN_INTEGRITY_NOT_QUALIFIED"
    assert verification["admission_possible"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda d: d.update(joint_assurance_lower_bound_union_bound=0.99),
            "joint assurance bound is inconsistent",
            id="bound",
        ),
        pytest.param(
            lambda d: d.update(admission_possible=True),
            "non-qualification contract",
            id="admission",
        ),
        pytest.param(
            lambda d: d.update(component_designs_claim_complete_powered_design=False),
            "component powered-design claim is inconsistent",
            id="component-claim",
        ),
    ],
)
def test_redigested_mutations_are_rejected_by_the_named_rule(mutate, message) -> None:
    mutated = dict(_composed_result())
    mutate(mutated)
    with pytest.raises(ValueError, match=message):
        verify_joint_sample_size_design(_redigest(mutated))


def test_undigested_edits_are_detected_before_any_rule() -> None:
    mutated = dict(_composed_result())
    mutated["admission_possible"] = True
    with pytest.raises(ValueError):
        verify_joint_sample_size_design(mutated)
