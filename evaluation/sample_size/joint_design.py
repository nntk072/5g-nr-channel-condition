"""Composition and verification of the joint sample-size design."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any

from evaluation.acceptance.joint_acceptance import (
    FAMILY_SCHEMA_V4,
    validate_joint_acceptance_family,
)
from evaluation.sample_size.binomial import (
    _probability,
)


def compose_joint_sample_size_design(
    *,
    joint_acceptance_family: dict[str, Any],
    joint_plan: Mapping[str, Any],
    material_assurance_plan: Mapping[str, Any],
    complex_csi_assurance_plan: Mapping[str, Any],
    propagation_assurance_plan: Mapping[str, Any],
    receiver_assurance_plan: Mapping[str, Any],
    target_joint_assurance_lower_bound: float = 0.80,
) -> dict[str, Any]:
    """Join branch-specific G3 planning outputs without merging sampling units.

    The component planners remain authoritative for their own criteria and
    assumptions. This function verifies their family/stratum alignment and
    assembles per-stratum required counts for a future preregistration. It
    never upgrades scenario-conditional planning to qualification or reports
    a single combined G3 sample size.
    """
    family = validate_joint_acceptance_family(joint_acceptance_family)
    target_joint = _probability(
        target_joint_assurance_lower_bound, "target_joint_assurance_lower_bound"
    )
    if family.get("schema") != FAMILY_SCHEMA_V4:
        raise ValueError("joint design composition requires acceptance-family v4")
    expected_family = {
        "family_id": family["family_id"],
        "family_sha256": family["family_sha256"],
    }
    components = {
        "joint": (joint_plan, "loslab.g3_joint_sample_size_planning.v1"),
        "material_assurance": (
            material_assurance_plan,
            "loslab.g3_bootstrap_assurance_planning.v1",
        ),
        "complex_csi_assurance": (
            complex_csi_assurance_plan,
            "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        ),
        "propagation_assurance": (
            propagation_assurance_plan,
            "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        ),
        "receiver_assurance": (
            receiver_assurance_plan,
            "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
        ),
    }
    component_digests: dict[str, dict[str, str]] = {}
    for name, (plan, expected_schema) in components.items():
        if not isinstance(plan, Mapping) or plan.get("schema") != expected_schema:
            raise ValueError(f"{name} plan has an unsupported schema")
        if plan.get("admission_possible") is not False or plan.get("qualification_status") != (
            "NOT_QUALIFIED"
        ):
            raise ValueError(f"{name} plan must remain explicitly non-qualifying")
        if name != "joint" and plan.get("scenario_provenance_verified") is not False:
            raise ValueError(f"{name} plan must not claim verified scenario provenance")
        family_identity = plan.get("joint_acceptance_family")
        if not isinstance(family_identity, Mapping) or any(
            family_identity.get(key) != value for key, value in expected_family.items()
        ):
            raise ValueError(f"{name} plan does not match the supplied acceptance family")
        component_digests[name] = {
            "schema": expected_schema,
            "sha256": hashlib.sha256(
                json.dumps(
                    dict(plan),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        }

    strata_by_branch = {
        "material_calibration": family["material_expected_strata"],
        "complex_csi": family["complex_csi_expected_strata"],
        "propagation_metrics": family["propagation_expected_strata"],
        "receiver_impact": family["receiver_impact_expected_strata"],
        "path_equivalence": family["path_equivalence_expected_scenarios"],
    }
    units_by_branch = {
        "material_calibration": "independent_acquisition_unit",
        "complex_csi": "independent_physical_episode",
        "propagation_metrics": "independent_physical_episode",
        "receiver_impact": "independent_transport_block_cluster",
        "path_equivalence": "physical_episode",
    }

    def positive_count(value: Any, label: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{label} must be a positive integer or null")
        return value

    def selected_assurance_count(
        plan: Mapping[str, Any],
        expected_strata: list[Any],
        label: str,
        *,
        stratum_key_fields: tuple[str, ...] = ("family_id", "condition"),
    ) -> tuple[int | None, float | None]:
        selected = positive_count(
            plan.get("selected_clusters_per_stratum_among_evaluated_candidates"),
            f"{label}.selected_clusters_per_stratum_among_evaluated_candidates",
        )
        if selected is None:
            return None, None
        target = _probability(plan.get("target_assurance"), f"{label}.target_assurance")
        candidates = plan.get("candidate_designs")
        if not isinstance(candidates, list):
            raise ValueError(f"{label}.candidate_designs must be an array")
        matches = [
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("candidate_clusters_per_stratum") == selected
        ]
        if len(matches) != 1 or matches[0].get("meets_target_assurance") is not True:
            raise ValueError(f"{label} selected count is not a passing candidate design")
        joint_lower = matches[0].get("joint_assurance_lower_bound_union_bound")
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or float(joint_lower) < target
            or float(joint_lower) > 1.0
        ):
            raise ValueError(f"{label} selected candidate does not meet joint assurance")
        rows = matches[0].get("stratum_assurance")
        expected_keys = {tuple(row) for row in expected_strata}
        observed: set[tuple[str, ...]] = set()
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{label} selected candidate has no stratum assurance rows")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError(f"{label} stratum assurance rows must be objects")
            raw_key = tuple(row.get(field) for field in stratum_key_fields)
            if any(not isinstance(part, str) or not part for part in raw_key):
                raise ValueError(f"{label} has invalid or duplicate assurance strata")
            key = tuple(part for part in raw_key if isinstance(part, str))
            if key in observed:
                raise ValueError(f"{label} has invalid or duplicate assurance strata")
            observed.add(key)
            if row.get("cluster_count") != selected:
                raise ValueError(f"{label} stratum cluster count differs from selected count")
            bound = row.get("assurance_one_sided_exact_lower_bound")
            if (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                or float(bound) < target
                or float(bound) > 1.0
            ):
                raise ValueError(f"{label} stratum bound does not meet target assurance")
        if observed != expected_keys:
            raise ValueError(f"{label} assurance strata differ from the acceptance family")
        recomputed_joint_lower = max(
            0.0,
            1.0 - sum(1.0 - float(row["assurance_one_sided_exact_lower_bound"]) for row in rows),
        )
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or not 0.0 <= float(joint_lower) <= 1.0
            or not math.isclose(
                float(joint_lower), recomputed_joint_lower, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError(f"{label} joint assurance bound is inconsistent with its strata")
        return selected, float(joint_lower)

    def selected_material_assurance_count(
        plan: Mapping[str, Any],
    ) -> tuple[int | None, float | None]:
        if plan.get("joint_acceptance_family") != expected_family:
            raise ValueError(
                "material assurance plan does not match the supplied acceptance family"
            )
        roster = plan.get("stratum_roster_by_id")
        if not isinstance(roster, Mapping):
            raise ValueError("material assurance plan must bind material stratum IDs")
        normalized_roster: dict[str, tuple[str, ...]] = {}
        for stratum_id, stratum in roster.items():
            if (
                not isinstance(stratum_id, str)
                or not stratum_id
                or isinstance(stratum, (str, bytes))
                or not isinstance(stratum, list)
                or len(stratum) != 3
                or any(not isinstance(part, str) or not part for part in stratum)
            ):
                raise ValueError("material assurance plan contains an invalid stratum binding")
            normalized_roster[stratum_id] = tuple(stratum)
        expected_material_strata = {tuple(row) for row in strata_by_branch["material_calibration"]}
        if (
            len(normalized_roster) != len(expected_material_strata)
            or set(normalized_roster.values()) != expected_material_strata
        ):
            raise ValueError("material assurance plan strata differ from the acceptance family")
        selected = positive_count(
            plan.get("selected_units_per_stratum_among_evaluated_candidates"),
            "material_assurance_plan.selected_units_per_stratum_among_evaluated_candidates",
        )
        if selected is None:
            return None, None
        target = _probability(
            plan.get("target_assurance"), "material_assurance_plan.target_assurance"
        )
        candidates = plan.get("candidate_designs")
        if not isinstance(candidates, list):
            raise ValueError("material_assurance_plan.candidate_designs must be an array")
        matches = [
            row
            for row in candidates
            if isinstance(row, Mapping) and row.get("independent_units_per_stratum") == selected
        ]
        if len(matches) != 1 or matches[0].get("meets_target_assurance") is not True:
            raise ValueError("material assurance selection is not a passing candidate")
        rows = matches[0].get("stratum_assurance")
        if not isinstance(rows, Mapping) or set(rows) != set(normalized_roster):
            raise ValueError("material assurance rows do not cover the bound stratum IDs")
        lower_bounds: list[float] = []
        for stratum_id, row in rows.items():
            if not isinstance(row, Mapping):
                raise ValueError("material assurance stratum rows must be objects")
            bound = row.get("joint_assurance_one_sided_exact_lower_bound")
            atom_count = row.get("assumed_support_atom_count")
            if (
                isinstance(bound, bool)
                or not isinstance(bound, (int, float))
                or not math.isfinite(float(bound))
                or float(bound) < target
                or float(bound) > 1.0
                or isinstance(atom_count, bool)
                or not isinstance(atom_count, int)
                or atom_count < 1
            ):
                raise ValueError(f"material assurance bound is invalid for {stratum_id}")
            lower_bounds.append(float(bound))
        joint_lower = matches[0].get("joint_assurance_lower_bound_union_bound")
        recomputed = max(0.0, 1.0 - sum(1.0 - value for value in lower_bounds))
        if (
            isinstance(joint_lower, bool)
            or not isinstance(joint_lower, (int, float))
            or not math.isfinite(float(joint_lower))
            or not 0.0 <= float(joint_lower) <= 1.0
            or not math.isclose(float(joint_lower), recomputed, rel_tol=1e-12, abs_tol=1e-12)
            or float(joint_lower) < target
        ):
            raise ValueError("material assurance joint bound is inconsistent or below target")
        return selected, float(joint_lower)

    material = joint_plan.get("material_design")
    csi = joint_plan.get("complex_csi_design")
    propagation = joint_plan.get("propagation_design")
    if not isinstance(material, Mapping) or not isinstance(csi, Mapping):
        raise ValueError("joint plan is missing material or complex-CSI design")
    if (
        joint_plan.get("single_combined_sample_size_is_reported") is not False
        or joint_plan.get("complete_sample_size") is not None
    ):
        raise ValueError(
            "joint plan must keep sampling units separate and combined size unresolved"
        )
    material_base_units = positive_count(
        material.get("required_independent_measurement_units_per_stratum_for_supported_claims"),
        "joint_plan.material_design.required_independent_measurement_units_per_stratum_for_supported_claims",
    )
    material_assurance_units, material_assurance_lower = selected_material_assurance_count(
        material_assurance_plan
    )
    material_units = (
        max(material_base_units, material_assurance_units)
        if material_base_units is not None and material_assurance_units is not None
        else None
    )
    csi_base_units = positive_count(
        csi.get("required_independent_scene_clusters_per_stratum_for_supported_claims"),
        "joint_plan.complex_csi_design.required_independent_scene_clusters_per_stratum_for_supported_claims",
    )
    csi_assurance_units, csi_assurance_lower = selected_assurance_count(
        complex_csi_assurance_plan,
        strata_by_branch["complex_csi"],
        "complex_csi_assurance_plan",
    )
    csi_units = (
        max(csi_base_units, csi_assurance_units)
        if csi_base_units is not None and csi_assurance_units is not None
        else None
    )
    propagation_base_units: int | None = None
    if propagation is not None:
        if not isinstance(propagation, Mapping):
            raise ValueError("joint_plan.propagation_design must be an object or null")
        propagation_base_units = positive_count(
            propagation.get("required_independent_clusters_per_stratum_for_supported_claims"),
            "joint_plan.propagation_design.required_independent_clusters_per_stratum_for_supported_claims",
        )
    propagation_assurance_units, propagation_assurance_lower = selected_assurance_count(
        propagation_assurance_plan,
        strata_by_branch["propagation_metrics"],
        "propagation_assurance_plan",
        stratum_key_fields=("family_id", "condition", "component"),
    )
    propagation_units = (
        max(propagation_base_units, propagation_assurance_units)
        if propagation_base_units is not None and propagation_assurance_units is not None
        else None
    )
    receiver_component = joint_plan.get("receiver_impact_design")
    if not isinstance(propagation, Mapping) or not isinstance(receiver_component, Mapping):
        raise ValueError("family v4 joint plan must include propagation and receiver designs")
    for branch, component in (
        ("material_calibration", material),
        ("complex_csi", csi),
        ("propagation_metrics", propagation),
        ("receiver_impact", receiver_component),
    ):
        declared_strata = component.get("stratum_count")
        if (
            isinstance(declared_strata, bool)
            or not isinstance(declared_strata, int)
            or declared_strata != len(strata_by_branch[branch])
        ):
            raise ValueError(f"joint plan {branch} stratum count differs from the family")
        declared_roster = component.get("strata")
        if declared_roster is not None:
            if not isinstance(declared_roster, (list, tuple)) or any(
                not isinstance(row, (list, tuple)) for row in declared_roster
            ):
                raise ValueError(f"joint plan {branch} stratum roster is invalid")
            if {tuple(row) for row in declared_roster} != {
                tuple(row) for row in strata_by_branch[branch]
            }:
                raise ValueError(f"joint plan {branch} stratum roster differs from the family")
    receiver_units, receiver_assurance_lower = selected_assurance_count(
        receiver_assurance_plan,
        strata_by_branch["receiver_impact"],
        "receiver_assurance_plan",
    )

    branch_counts = {
        "material_calibration": material_units,
        "complex_csi": csi_units,
        "propagation_metrics": propagation_units,
        "receiver_impact": receiver_units,
        "path_equivalence": 1,
    }
    branch_designs: dict[str, Any] = {}
    for branch, roster in strata_by_branch.items():
        count = branch_counts[branch]
        branch_designs[branch] = {
            "sampling_unit": units_by_branch[branch],
            "required_units_by_stratum": [
                {"stratum": list(stratum), "required_units": count} for stratum in roster
            ],
            "count_available": count is not None,
            "component_status": (
                "EXACT_EPISODE_CENSUS"
                if branch == "path_equivalence"
                else "SCENARIO_CONDITIONAL_COUNT_AVAILABLE"
                if count is not None
                else "REQUIRED_ASSURANCE_COUNT_UNRESOLVED"
            ),
        }

    all_counts_available = all(design["count_available"] for design in branch_designs.values())
    component_design_complete = all(
        count is not None and lower is not None
        for count, lower in (
            (material_assurance_units, material_assurance_lower),
            (csi_assurance_units, csi_assurance_lower),
            (propagation_assurance_units, propagation_assurance_lower),
            (receiver_units, receiver_assurance_lower),
        )
    )
    assurance_bounds = {
        "material_calibration": material_assurance_lower,
        "complex_csi": csi_assurance_lower,
        "propagation_metrics": propagation_assurance_lower,
        "receiver_impact": receiver_assurance_lower,
    }
    assurance_values = tuple(assurance_bounds.values())
    present_assurance_values = tuple(bound for bound in assurance_values if bound is not None)
    joint_assurance_lower = (
        max(0.0, 1.0 - sum(1.0 - bound for bound in present_assurance_values))
        if len(present_assurance_values) == len(assurance_values)
        else None
    )
    blockers = [
        "planner assumptions are scenario inputs without verified physical/population provenance"
    ]
    if not component_design_complete:
        blockers.append("one or more evaluator-aligned assurance plans have no selected candidate")
    if not all_counts_available:
        blockers.append("one or more branch-specific required counts remain unresolved")
    if joint_assurance_lower is None:
        blockers.append("one or more evaluator-aligned branch assurance bounds remain unresolved")
    elif joint_assurance_lower < target_joint:
        blockers.append("cross-branch union-bound assurance is below the requested joint G3 target")
    angular = material.get("angular_maximum_claim")
    angular_supplied = isinstance(angular, Mapping) and angular.get("assumption_supplied") is True
    angular_compatible = (
        angular.get("compatible_with_material_design") if isinstance(angular, Mapping) else None
    )
    angular_maximum = (
        angular.get("maximum_measurement_units_per_stratum")
        if isinstance(angular, Mapping)
        else None
    )
    if (
        angular_supplied
        and isinstance(angular_maximum, int)
        and material_units is not None
        and material_units > angular_maximum
    ):
        angular_compatible = False
    if not angular_supplied:
        blockers.append("material angular-maximum pass-probability assumption is missing")
    elif angular_compatible is False:
        blockers.append("material angular-maximum assumption conflicts with the planned unit count")
    elif angular_compatible is not True:
        blockers.append("material angular-maximum compatibility remains unresolved")
    complete_powered_design = bool(
        all_counts_available
        and component_design_complete
        and joint_assurance_lower is not None
        and joint_assurance_lower >= target_joint
        and angular_supplied
        and angular_compatible is True
    )

    result: dict[str, Any] = {
        "schema": "loslab.g3_joint_sample_size_design_composition.v1",
        "status": "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "joint_acceptance_family": expected_family,
        "component_plans": component_digests,
        "sampling_units_are_distinct": True,
        "single_combined_sample_size_is_reported": False,
        "branch_designs": branch_designs,
        "all_branch_counts_available": all_counts_available,
        "component_designs_claim_complete_powered_design": component_design_complete,
        "branch_assurance_lower_bounds": assurance_bounds,
        "joint_assurance_lower_bound_union_bound": joint_assurance_lower,
        "target_joint_assurance_lower_bound": target_joint,
        "material_angular_maximum_assumption": {
            "supplied": angular_supplied,
            "compatible_with_material_design": angular_compatible,
            "maximum_units_per_stratum": angular_maximum,
        },
        "complete_precision_powered_design": complete_powered_design,
        "complete_sample_size": None,
        "blockers": blockers,
        "limitations": [
            "Counts are conditional on the exact assumptions and family-bound component plans.",
            "The cross-branch assurance bound covers material, complex-CSI, propagation, and receiver branches; path equivalence is a frozen episode census and carries no population-generalization claim.",
            "Different branches use independent acquisition units, physical episodes, and receiver clusters; their counts must not be summed.",
            "This composition artifact does not verify scenario assumptions, physical independence, measurement provenance, or external review.",
        ],
    }
    result["design_sha256"] = hashlib.sha256(
        json.dumps(
            result,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return result


def verify_joint_sample_size_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Verify composition-artifact integrity while preserving non-admission status."""
    if not isinstance(design, Mapping):
        raise TypeError("G3 joint sample-size design must be an object")
    saved = dict(design)
    declared_digest = saved.pop("design_sha256", None)
    if (
        not isinstance(declared_digest, str)
        or len(declared_digest) != 64
        or any(character not in "0123456789abcdef" for character in declared_digest)
    ):
        raise ValueError("design_sha256 must be a lowercase SHA-256 digest")
    actual_digest = hashlib.sha256(
        json.dumps(
            saved,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if actual_digest != declared_digest:
        raise ValueError("G3 joint sample-size design digest does not match its contents")
    expected_fields = {
        "schema",
        "status",
        "qualification_status",
        "admission_possible",
        "joint_acceptance_family",
        "component_plans",
        "sampling_units_are_distinct",
        "single_combined_sample_size_is_reported",
        "branch_designs",
        "all_branch_counts_available",
        "component_designs_claim_complete_powered_design",
        "branch_assurance_lower_bounds",
        "joint_assurance_lower_bound_union_bound",
        "target_joint_assurance_lower_bound",
        "material_angular_maximum_assumption",
        "complete_precision_powered_design",
        "complete_sample_size",
        "blockers",
        "limitations",
    }
    if set(saved) != expected_fields:
        raise ValueError("G3 joint sample-size design fields differ from its v1 schema")
    if (
        saved.get("schema") != "loslab.g3_joint_sample_size_design_composition.v1"
        or saved.get("status") != "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION"
        or saved.get("qualification_status") != "NOT_QUALIFIED"
        or saved.get("admission_possible") is not False
        or saved.get("sampling_units_are_distinct") is not True
        or saved.get("single_combined_sample_size_is_reported") is not False
        or not isinstance(saved.get("component_designs_claim_complete_powered_design"), bool)
        or not isinstance(saved.get("complete_precision_powered_design"), bool)
        or saved.get("complete_sample_size") is not None
    ):
        raise ValueError("G3 joint sample-size design violates its non-qualification contract")
    angular_assumption = saved.get("material_angular_maximum_assumption")
    if not isinstance(angular_assumption, Mapping) or set(angular_assumption) != {
        "supplied",
        "compatible_with_material_design",
        "maximum_units_per_stratum",
    }:
        raise ValueError("material angular-maximum design declaration is invalid")
    if not isinstance(angular_assumption.get("supplied"), bool):
        raise ValueError("material angular-maximum supplied flag must be Boolean")
    compatible = angular_assumption.get("compatible_with_material_design")
    if compatible is not None and not isinstance(compatible, bool):
        raise ValueError("material angular-maximum compatibility must be Boolean or null")
    maximum = angular_assumption.get("maximum_units_per_stratum")
    if maximum is not None and (
        isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1
    ):
        raise ValueError("material angular-maximum count must be a positive integer or null")
    if not angular_assumption["supplied"] and (compatible is not None or maximum is not None):
        raise ValueError("unsupplied angular-maximum assumptions cannot declare derived values")
    component_plans = saved.get("component_plans")
    expected_component_schemas = {
        "joint": "loslab.g3_joint_sample_size_planning.v1",
        "material_assurance": "loslab.g3_bootstrap_assurance_planning.v1",
        "complex_csi_assurance": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        "propagation_assurance": "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        "receiver_assurance": "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
    }
    if not isinstance(component_plans, Mapping) or set(component_plans) != set(
        expected_component_schemas
    ):
        raise ValueError("component_plans does not cover the exact required planning outputs")
    for name, expected_schema in expected_component_schemas.items():
        component = component_plans[name]
        digest = component.get("sha256") if isinstance(component, Mapping) else None
        if (
            not isinstance(component, Mapping)
            or set(component) != {"schema", "sha256"}
            or component.get("schema") != expected_schema
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"component_plans.{name} is invalid")
    branches = saved.get("branch_designs")
    expected_branches = {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
        "path_equivalence",
    }
    if not isinstance(branches, Mapping) or set(branches) != expected_branches:
        raise ValueError("branch_designs does not cover the exact G3 evaluator branches")
    count_availability: list[bool] = []
    for branch, branch_design in branches.items():
        if not isinstance(branch_design, Mapping):
            raise ValueError(f"branch_designs.{branch} must be an object")
        if set(branch_design) != {
            "sampling_unit",
            "required_units_by_stratum",
            "count_available",
            "component_status",
        }:
            raise ValueError(f"branch_designs.{branch} fields are invalid")
        roster = branch_design.get("required_units_by_stratum")
        if not isinstance(roster, list) or not roster:
            raise ValueError(f"branch_designs.{branch} must contain a nonempty stratum roster")
        keys: set[tuple[str, ...]] = set()
        counts: list[int | None] = []
        for row in roster:
            if not isinstance(row, Mapping) or set(row) != {"stratum", "required_units"}:
                raise ValueError(f"branch_designs.{branch} has an invalid stratum row")
            key = row["stratum"]
            if (
                not isinstance(key, list)
                or not key
                or any(not isinstance(part, str) or not part for part in key)
                or tuple(key) in keys
            ):
                raise ValueError(f"branch_designs.{branch} has an invalid or duplicate stratum")
            keys.add(tuple(key))
            count = row["required_units"]
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 1
            ):
                raise ValueError(f"branch_designs.{branch} required_units is invalid")
            if branch == "path_equivalence" and count != 1:
                raise ValueError("path_equivalence must remain a one-per-episode census")
            counts.append(count)
        count_available = all(count is not None for count in counts)
        if branch_design.get("count_available") is not count_available:
            raise ValueError(f"branch_designs.{branch} count_available is inconsistent")
        count_availability.append(count_available)
    if saved.get("all_branch_counts_available") is not all(count_availability):
        raise ValueError("all_branch_counts_available is inconsistent with branch rosters")
    assurance_bounds = saved.get("branch_assurance_lower_bounds")
    assurance_branches = {
        "material_calibration",
        "complex_csi",
        "propagation_metrics",
        "receiver_impact",
    }
    if not isinstance(assurance_bounds, Mapping) or set(assurance_bounds) != assurance_branches:
        raise ValueError("branch_assurance_lower_bounds does not cover the evaluator branches")
    normalized_bounds: list[float] = []
    for branch, bound in assurance_bounds.items():
        if bound is None:
            continue
        if (
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or not math.isfinite(float(bound))
            or not 0.0 <= float(bound) <= 1.0
        ):
            raise ValueError(f"branch assurance lower bound is invalid for {branch}")
        normalized_bounds.append(float(bound))
    declared_joint_assurance = saved.get("joint_assurance_lower_bound_union_bound")
    if len(normalized_bounds) == len(assurance_branches):
        expected_joint_assurance = max(0.0, 1.0 - sum(1.0 - bound for bound in normalized_bounds))
        if (
            isinstance(declared_joint_assurance, bool)
            or not isinstance(declared_joint_assurance, (int, float))
            or not math.isclose(
                float(declared_joint_assurance),
                expected_joint_assurance,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("joint G3 assurance bound is inconsistent with branch bounds")
    elif declared_joint_assurance is not None:
        raise ValueError("joint G3 assurance bound must be null when a branch bound is unresolved")
    component_design_complete = all(
        branches[branch]["count_available"] and assurance_bounds[branch] is not None
        for branch in assurance_branches
    )
    if saved["component_designs_claim_complete_powered_design"] is not component_design_complete:
        raise ValueError(
            "component powered-design claim is inconsistent with branch counts and assurance bounds"
        )
    target_joint_assurance = _probability(
        saved.get("target_joint_assurance_lower_bound"), "target_joint_assurance_lower_bound"
    )
    if angular_assumption.get("compatible_with_material_design") is True:
        maximum_units = angular_assumption.get("maximum_units_per_stratum")
        material_rows = branches["material_calibration"]["required_units_by_stratum"]
        if maximum_units is not None and any(
            row["required_units"] is not None and row["required_units"] > maximum_units
            for row in material_rows
        ):
            raise ValueError("material angular-maximum assumption conflicts with required counts")
    expected_complete = bool(
        saved["all_branch_counts_available"]
        and component_design_complete
        and declared_joint_assurance is not None
        and float(declared_joint_assurance) >= target_joint_assurance
        and angular_assumption.get("supplied") is True
        and angular_assumption.get("compatible_with_material_design") is True
    )
    if saved["complete_precision_powered_design"] is not expected_complete:
        raise ValueError("complete_precision_powered_design is inconsistent with component plans")
    return {
        "schema": "loslab.g3_joint_sample_size_design_verification.v1",
        "status": "PASS_DESIGN_INTEGRITY_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "design_sha256": declared_digest,
        "complete_precision_powered_design": expected_complete,
    }
