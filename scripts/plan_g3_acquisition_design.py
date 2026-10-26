"""Build a reproducible, assumption-only G3.03 acquisition design report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.g3_joint_acceptance import (  # noqa: E402
    FAMILY_SCHEMA_V2,
    FAMILY_SCHEMA_V3,
    FAMILY_SCHEMA_V4,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_propagation_assurance import (  # noqa: E402
    plan_g3_propagation_bootstrap_assurance,
)
from evaluation.g3_receiver_assurance import (  # noqa: E402
    plan_g3_receiver_impact_bootstrap_assurance,
)
from evaluation.g3_sample_size_planning import (  # noqa: E402
    MIN_INDEPENDENT_MEASUREMENT_UNITS,
    compose_g3_joint_sample_size_design,
    plan_g3_bootstrap_assurance,
    plan_g3_complex_csi_bootstrap_assurance,
    plan_g3_joint_independent_units,
    verify_g3_joint_sample_size_design,
)

_CONFIG_VERSION_V3 = "loslab.g3_acquisition_design_assumptions.v3"
_CONFIG_VERSION_V4 = "loslab.g3_acquisition_design_assumptions.v4"
_CONFIG_VERSION_V5 = "loslab.g3_acquisition_design_assumptions.v5"
_MAX_CONFIG_BYTES = 4 * 1024 * 1024
_FROZEN_FAMILYWISE_ALPHA = 0.05
_STRATUM_FIELDS = {"stratum_id", "material_id", "polarization", "incidence_stratum_id"}
_ASSUMPTION_FIELDS = {
    "planned_quantile_pass_probabilities",
    "planned_interval_coverage",
    "planned_complex_csi_failure_probability",
    "planned_complex_csi_p95_pass_probability",
    "planned_angular_unit_pass_probability",
    "familywise_alpha",
    "familywise_power",
}
_PROPAGATION_ASSUMPTION_FIELDS = {
    "planned_propagation_p95_pass_probabilities",
    "planned_propagation_cluster_failure_probabilities",
}
_PROPAGATION_METRICS = {
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
}
_ASSURANCE_SOURCE_GROUPS = {
    "material_calibration": (
        "evaluation/g3_sample_size_planning.py",
        "evaluation/g3_material_calibration.py",
        "evaluation/g3_joint_acceptance.py",
    ),
    "complex_csi": (
        "evaluation/g3_sample_size_planning.py",
        "evaluation/g3_complex_csi.py",
        "evaluation/g3_joint_acceptance.py",
    ),
    "propagation_metrics": (
        "evaluation/g3_propagation_assurance.py",
        "evaluation/g3_propagation_metrics.py",
        "evaluation/g3_joint_acceptance.py",
    ),
    "receiver_impact": (
        "evaluation/g3_receiver_assurance.py",
        "evaluation/g3_receiver_impact.py",
        "evaluation/g3_joint_acceptance.py",
    ),
}


def _source_group_identity(relative_paths: Sequence[str]) -> dict[str, Any]:
    file_hashes = {
        relative_path: hashlib.sha256(
            (REPOSITORY_ROOT / relative_path).read_bytes()
        ).hexdigest()
        for relative_path in sorted(relative_paths)
    }
    rows = [
        {
            "path": relative_path,
            "sha256": digest,
        }
        for relative_path, digest in file_hashes.items()
    ]
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "files": file_hashes,
        "bundle_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _complete_design_status(
    *,
    schema_version: str,
    missing_designs: list[str],
    material: dict[str, Any] | None,
    complex_csi: dict[str, Any] | None,
    receiver_impact: dict[str, Any] | None,
    propagation: dict[str, Any] | None,
    joint_design: dict[str, Any] | None = None,
) -> str:
    """Summarize missing and unmet scenario-assurance designs without implying qualification."""
    if schema_version == _CONFIG_VERSION_V5:
        if joint_design is None:
            return "JOINT_SCENARIO_ASSURANCE_COMPOSITION_UNAVAILABLE"
        if joint_design.get("complete_precision_powered_g3_design") is True:
            return "SCENARIO_CONDITIONAL_POWERED_G3_DESIGN_COMPLETE_NOT_QUALIFIED"
        return "SCENARIO_CONDITIONAL_G3_DESIGN_INCOMPLETE"
    if schema_version in {_CONFIG_VERSION_V4, _CONFIG_VERSION_V5} and missing_designs:
        return "ASSURANCE_DESIGNS_MISSING:" + ",".join(missing_designs)
    if material is None and complex_csi is None:
        return "MATERIAL_AND_CSI_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"
    if material is None:
        return "MATERIAL_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"
    if complex_csi is None:
        return "CSI_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"

    unmet: list[str] = []
    for name, report, selected_key in (
        (
            "material",
            material,
            "selected_units_per_stratum_among_evaluated_candidates",
        ),
        (
            "complex_csi",
            complex_csi,
            "selected_clusters_per_stratum_among_evaluated_candidates",
        ),
        (
            "receiver_impact",
            receiver_impact,
            "selected_clusters_per_stratum_among_evaluated_candidates",
        ),
        (
            "propagation",
            propagation,
            "selected_clusters_per_stratum_among_evaluated_candidates",
        ),
    ):
        if report is not None and report.get(selected_key) is None:
            unmet.append(name)
    if unmet:
        return "SCENARIO_CONDITIONAL_ASSURANCE_TARGET_NOT_MET:" + ",".join(unmet)
    return "SCENARIO_CONDITIONAL_SEPARATE_DESIGNS_REPORTED"
_BOOTSTRAP_FIELDS = {
    "scenarios_by_stratum",
    "candidate_units_per_stratum",
    "bootstrap_replicates",
    "assurance_replicates",
    "target_assurance",
    "seed",
}
_BOOTSTRAP_METRICS = {
    "median_absolute_power_error_db",
    "p95_absolute_power_error_db",
    "maximum_angular_cdf_ks",
    "nominal_95_interval_coverage",
    "median_prediction_interval_width_db",
}
_CSI_BOOTSTRAP_FIELDS = {
    "scenarios_by_stratum",
    "candidate_clusters_per_stratum",
    "bootstrap_replicates",
    "assurance_replicates",
    "target_assurance",
    "seed",
}
_RECEIVER_BOOTSTRAP_FIELDS = {
    "scenarios_by_stratum",
    "candidate_clusters_per_stratum",
    "snr_grid_db",
    "comparison_snr_db",
    "bootstrap_replicates",
    "assurance_replicates",
    "target_assurance",
    "monte_carlo_confidence",
    "seed",
}
_PROPAGATION_BOOTSTRAP_FIELDS = {
    "scenarios_by_stratum",
    "candidate_clusters_per_stratum",
    "effective_bandwidth_hz",
    "bootstrap_draws",
    "assurance_replicates",
    "target_assurance",
    "monte_carlo_confidence",
    "seed",
}


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _require_object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return value


def _validate_config(
    config: Any,
) -> tuple[
    list[dict[str, str]],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any],
]:
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    version = config.get("schema_version")
    if version not in {_CONFIG_VERSION_V3, _CONFIG_VERSION_V4, _CONFIG_VERSION_V5}:
        raise ValueError("schema_version must identify a supported v3, v4, or v5 planning schema")
    required_fields = {
        "schema_version",
        "scenario_label",
        "assumption_source_description",
        "expected_strata",
        "joint_acceptance_family",
        "worst_stratum_assumptions",
        "bootstrap_assurance",
        "complex_csi_bootstrap_assurance",
    }
    if version in {_CONFIG_VERSION_V4, _CONFIG_VERSION_V5}:
        required_fields.add("receiver_impact_bootstrap_assurance")
    if version == _CONFIG_VERSION_V5:
        required_fields.add("propagation_bootstrap_assurance")
    root = _require_object(
        config,
        "configuration",
        required_fields,
    )
    if root["scenario_label"] not in {
        "hypothetical_unverified",
        "externally_justified_preregistered_unverified",
    }:
        raise ValueError("scenario_label must identify an unverified planning-assumption source")
    source_description = root["assumption_source_description"]
    if not isinstance(source_description, str) or not source_description.strip():
        raise ValueError("assumption_source_description must be a nonempty explanation")

    strata_value = root["expected_strata"]
    if not isinstance(strata_value, list) or not strata_value:
        raise ValueError("expected_strata must be a nonempty array")
    strata: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, str]] = set()
    for index, value in enumerate(strata_value):
        stratum = _require_object(value, f"expected_strata[{index}]", _STRATUM_FIELDS)
        for field in _STRATUM_FIELDS:
            item = stratum[field]
            if not isinstance(item, str) or not item.strip() or item != item.strip():
                raise ValueError(f"expected_strata[{index}].{field} must be nonempty and trimmed")
        if stratum["polarization"] not in {"V", "H"}:
            raise ValueError(f"expected_strata[{index}].polarization must be V or H")
        if stratum["stratum_id"] in seen_ids:
            raise ValueError("expected_strata stratum_id values must be unique")
        key = (
            stratum["material_id"],
            stratum["polarization"],
            stratum["incidence_stratum_id"],
        )
        if key in seen_keys:
            raise ValueError("expected_strata material/polarization/incidence keys must be unique")
        seen_ids.add(stratum["stratum_id"])
        seen_keys.add(key)
        strata.append({field: stratum[field] for field in sorted(_STRATUM_FIELDS)})

    family = validate_g3_joint_acceptance_family(root["joint_acceptance_family"])
    if version == _CONFIG_VERSION_V3 and family["schema"] == FAMILY_SCHEMA_V3:
        raise ValueError(
            "acquisition-design v3 inputs support joint-family v1/v2 only; use input v4"
        )
    if version == _CONFIG_VERSION_V5 and family["schema"] != FAMILY_SCHEMA_V4:
        raise ValueError("acquisition-design v5 inputs require joint-family v4")
    if version == _CONFIG_VERSION_V4 and family["schema"] not in {
        FAMILY_SCHEMA_V3,
        FAMILY_SCHEMA_V4,
    }:
        raise ValueError(f"acquisition-design {version[-2:]} inputs require joint-family v3 or v4")
    config_material_strata = sorted(
        (item["material_id"], item["polarization"], item["incidence_stratum_id"]) for item in strata
    )
    if config_material_strata != [tuple(item) for item in family["material_expected_strata"]]:
        raise ValueError("joint_acceptance_family material strata differ from expected_strata")

    assumption_fields = set(_ASSUMPTION_FIELDS)
    if family["schema"] in {FAMILY_SCHEMA_V2, FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
        assumption_fields.update(_PROPAGATION_ASSUMPTION_FIELDS)
    assumptions = _require_object(
        root["worst_stratum_assumptions"],
        "worst_stratum_assumptions",
        assumption_fields,
    )
    familywise_alpha = assumptions["familywise_alpha"]
    if (
        isinstance(familywise_alpha, bool)
        or not isinstance(familywise_alpha, (int, float))
        or not math.isfinite(float(familywise_alpha))
        or not math.isclose(
            float(familywise_alpha), _FROZEN_FAMILYWISE_ALPHA, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError(
            "familywise_alpha must remain 0.05 for the frozen 95% G3 acceptance criterion"
        )
    if not isinstance(assumptions["planned_quantile_pass_probabilities"], dict):
        raise ValueError("planned_quantile_pass_probabilities must be an object")
    if assumptions["planned_angular_unit_pass_probability"] is not None:
        value = assumptions["planned_angular_unit_pass_probability"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("planned_angular_unit_pass_probability must be a number or null")
    if family["schema"] in {FAMILY_SCHEMA_V2, FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
        for field in _PROPAGATION_ASSUMPTION_FIELDS:
            values = assumptions[field]
            if not isinstance(values, dict) or set(values) != _PROPAGATION_METRICS:
                raise ValueError(
                    f"{field} must define exactly: " + ", ".join(sorted(_PROPAGATION_METRICS))
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) < 1.0
                for value in values.values()
            ):
                raise ValueError(f"{field} values must be finite probabilities in (0, 1)")

    bootstrap = root["bootstrap_assurance"]
    if bootstrap is not None:
        bootstrap = _require_object(bootstrap, "bootstrap_assurance", _BOOTSTRAP_FIELDS)
        candidates = bootstrap["candidate_units_per_stratum"]
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(
                isinstance(candidate, bool)
                or not isinstance(candidate, int)
                or candidate < MIN_INDEPENDENT_MEASUREMENT_UNITS
                for candidate in candidates
            )
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError(
                "candidate_units_per_stratum must contain unique integers at or above "
                f"the evaluator floor of {MIN_INDEPENDENT_MEASUREMENT_UNITS}"
            )
        scenarios = bootstrap["scenarios_by_stratum"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("bootstrap_assurance.scenarios_by_stratum must be a nonempty array")
        by_id: dict[str, dict[str, Any]] = {}
        for index, value in enumerate(scenarios):
            item = _require_object(
                value,
                f"bootstrap_assurance.scenarios_by_stratum[{index}]",
                {"stratum_id", "assumed_iid_scenario"},
            )
            stratum_id = item["stratum_id"]
            if not isinstance(stratum_id, str) or stratum_id not in seen_ids:
                raise ValueError("bootstrap scenario references an unknown stratum_id")
            if stratum_id in by_id:
                raise ValueError("bootstrap scenarios must contain each stratum_id once")
            scenario = _require_object(
                item["assumed_iid_scenario"],
                f"bootstrap scenario {stratum_id!r}",
                {"assumed_unit_summaries", "probabilities"},
            )
            atoms = scenario["assumed_unit_summaries"]
            probabilities = scenario["probabilities"]
            if not isinstance(atoms, list) or not atoms:
                raise ValueError(f"bootstrap scenario {stratum_id!r} needs unit-summary atoms")
            if any(not isinstance(atom, dict) or set(atom) != _BOOTSTRAP_METRICS for atom in atoms):
                raise ValueError(f"bootstrap scenario {stratum_id!r} has invalid metric fields")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(atoms)
                or any(
                    isinstance(probability, bool) or not isinstance(probability, (int, float))
                    for probability in probabilities
                )
            ):
                raise ValueError(f"bootstrap scenario {stratum_id!r} probabilities do not align")
            by_id[stratum_id] = scenario
        if set(by_id) != seen_ids:
            raise ValueError("bootstrap assurance needs exactly one scenario for every stratum")
        bootstrap = {
            **bootstrap,
            "assumed_iid_scenarios_by_stratum": by_id,
        }
        del bootstrap["scenarios_by_stratum"]

    csi_bootstrap = root["complex_csi_bootstrap_assurance"]
    if csi_bootstrap is not None:
        csi_bootstrap = _require_object(
            csi_bootstrap, "complex_csi_bootstrap_assurance", _CSI_BOOTSTRAP_FIELDS
        )
        candidates = csi_bootstrap["candidate_clusters_per_stratum"]
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(
                isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 2
                for candidate in candidates
            )
            or len(set(candidates)) != len(candidates)
            or candidates != sorted(candidates)
        ):
            raise ValueError(
                "candidate_clusters_per_stratum must be unique ascending integers >= 2"
            )
        scenarios = csi_bootstrap["scenarios_by_stratum"]
        expected_csi_strata = {
            (str(item[0]), str(item[1])) for item in family["complex_csi_expected_strata"]
        }
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError(
                "complex_csi_bootstrap_assurance.scenarios_by_stratum must be nonempty"
            )
        by_stratum: dict[tuple[str, str], dict[str, Any]] = {}
        for index, value in enumerate(scenarios):
            item = _require_object(
                value,
                f"complex_csi_bootstrap_assurance.scenarios_by_stratum[{index}]",
                {"family_id", "condition", "assumed_iid_scenario"},
            )
            csi_key = (item["family_id"], item["condition"])
            if csi_key not in expected_csi_strata:
                raise ValueError(
                    "CSI bootstrap scenario references an unknown joint-family stratum"
                )
            if csi_key in by_stratum:
                raise ValueError("CSI bootstrap scenarios must contain each family/condition once")
            scenario = _require_object(
                item["assumed_iid_scenario"],
                f"CSI bootstrap scenario {csi_key!r}",
                {"assumed_cluster_summaries", "probabilities"},
            )
            atoms = scenario["assumed_cluster_summaries"]
            probabilities = scenario["probabilities"]
            if not isinstance(atoms, list) or not atoms:
                raise ValueError(f"CSI bootstrap scenario {csi_key!r} needs cluster-summary atoms")
            if any(
                not isinstance(atom, dict)
                or set(atom) != {"cluster_median_e_h", "cluster_p95_e_h", "cluster_failed"}
                for atom in atoms
            ):
                raise ValueError(
                    f"CSI bootstrap scenario {csi_key!r} has invalid cluster-summary fields"
                )
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(atoms)
                or any(
                    isinstance(p, bool) or not isinstance(p, (int, float)) for p in probabilities
                )
            ):
                raise ValueError(f"CSI bootstrap scenario {csi_key!r} probabilities do not align")
            by_stratum[csi_key] = scenario
        if set(by_stratum) != expected_csi_strata:
            raise ValueError("CSI bootstrap assurance needs exactly every joint-family CSI stratum")
        csi_bootstrap = {**csi_bootstrap, "assumed_iid_scenarios_by_stratum": by_stratum}
        del csi_bootstrap["scenarios_by_stratum"]

    receiver_bootstrap = root.get("receiver_impact_bootstrap_assurance")
    if receiver_bootstrap is not None:
        if family["schema"] not in {FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}:
            raise ValueError("receiver bootstrap assurance requires joint-family v3 or v4")
        receiver_bootstrap = _require_object(
            receiver_bootstrap,
            "receiver_impact_bootstrap_assurance",
            _RECEIVER_BOOTSTRAP_FIELDS,
        )
        receiver_candidates = receiver_bootstrap["candidate_clusters_per_stratum"]
        if (
            not isinstance(receiver_candidates, list)
            or not receiver_candidates
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 20
                for value in receiver_candidates
            )
            or receiver_candidates != sorted(set(receiver_candidates))
        ):
            raise ValueError(
                "receiver candidate cluster counts must be unique ascending integers >= 20"
            )
        receiver_scenarios = receiver_bootstrap["scenarios_by_stratum"]
        expected_receiver_strata = {
            (str(item[0]), str(item[1])) for item in family["receiver_impact_expected_strata"]
        }
        if not isinstance(receiver_scenarios, list) or not receiver_scenarios:
            raise ValueError("receiver scenarios_by_stratum must be a nonempty array")
        by_receiver_stratum: dict[tuple[str, str], dict[str, Any]] = {}
        for index, value in enumerate(receiver_scenarios):
            item = _require_object(
                value,
                f"receiver_impact_bootstrap_assurance.scenarios_by_stratum[{index}]",
                {"family_id", "condition", "assumed_iid_scenario"},
            )
            if not isinstance(item["family_id"], str) or item["condition"] not in {
                "LOS",
                "NLOS",
            }:
                raise ValueError("receiver scenario family_id or condition is invalid")
            receiver_key = (item["family_id"], item["condition"])
            if receiver_key not in expected_receiver_strata or receiver_key in by_receiver_stratum:
                raise ValueError(
                    "receiver scenarios must cover each declared family/condition once"
                )
            scenario = _require_object(
                item["assumed_iid_scenario"],
                f"receiver scenario {receiver_key!r}",
                {"assumed_cluster_outcomes", "probabilities"},
            )
            atoms = scenario["assumed_cluster_outcomes"]
            probabilities = scenario["probabilities"]
            if not isinstance(atoms, list) or not atoms:
                raise ValueError(f"receiver scenario {receiver_key!r} needs cluster-outcome atoms")
            for atom in atoms:
                _require_object(
                    atom,
                    f"receiver outcome atom {receiver_key!r}",
                    {"candidate_crc_errors", "challenger_crc_errors"},
                )
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(atoms)
                or any(
                    isinstance(probability, bool) or not isinstance(probability, (int, float))
                    for probability in probabilities
                )
            ):
                raise ValueError(f"receiver scenario {receiver_key!r} probabilities do not align")
            by_receiver_stratum[receiver_key] = scenario
        if set(by_receiver_stratum) != expected_receiver_strata:
            raise ValueError("receiver bootstrap scenarios must cover every joint-family stratum")
        receiver_bootstrap = {
            **receiver_bootstrap,
            "assumed_iid_scenarios_by_stratum": by_receiver_stratum,
        }
        del receiver_bootstrap["scenarios_by_stratum"]

    propagation_bootstrap = root.get("propagation_bootstrap_assurance")
    if propagation_bootstrap is not None:
        if version != _CONFIG_VERSION_V5 or family["schema"] not in {
            FAMILY_SCHEMA_V3,
            FAMILY_SCHEMA_V4,
        }:
            raise ValueError("propagation bootstrap assurance requires acquisition-design v5")
        propagation_bootstrap = _require_object(
            propagation_bootstrap,
            "propagation_bootstrap_assurance",
            _PROPAGATION_BOOTSTRAP_FIELDS,
        )
        candidates = propagation_bootstrap["candidate_clusters_per_stratum"]
        if (
            not isinstance(candidates, list)
            or not candidates
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in candidates
            )
            or candidates != sorted(set(candidates))
        ):
            raise ValueError(
                "propagation candidate cluster counts must be unique ascending integers >= 2"
            )
        expected_propagation_strata = {
            (str(item[0]), str(item[1]), str(item[2]))
            for item in family["propagation_expected_strata"]
        }
        scenarios = propagation_bootstrap["scenarios_by_stratum"]
        if not isinstance(scenarios, list) or not scenarios:
            raise ValueError("propagation scenarios_by_stratum must be a nonempty array")
        by_propagation_stratum: dict[tuple[str, str, str], dict[str, Any]] = {}
        for index, value in enumerate(scenarios):
            item = _require_object(
                value,
                f"propagation_bootstrap_assurance.scenarios_by_stratum[{index}]",
                {"family_id", "condition", "component", "assumed_iid_scenario"},
            )
            key = (item["family_id"], item["condition"], item["component"])
            if key not in expected_propagation_strata or key in by_propagation_stratum:
                raise ValueError(
                    "propagation scenarios must cover each declared family stratum once"
                )
            scenario = _require_object(
                item["assumed_iid_scenario"],
                f"propagation scenario {key!r}",
                {"assumed_cluster_episodes", "probabilities"},
            )
            atoms = scenario["assumed_cluster_episodes"]
            probabilities = scenario["probabilities"]
            if not isinstance(atoms, list) or not atoms:
                raise ValueError(f"propagation scenario {key!r} needs cluster-episode atoms")
            if (
                not isinstance(probabilities, list)
                or len(probabilities) != len(atoms)
                or any(
                    isinstance(probability, bool) or not isinstance(probability, (int, float))
                    for probability in probabilities
                )
            ):
                raise ValueError(f"propagation scenario {key!r} probabilities do not align")
            by_propagation_stratum[key] = scenario
        if set(by_propagation_stratum) != expected_propagation_strata:
            raise ValueError("propagation scenarios must cover every joint-family stratum")
        propagation_bootstrap = {
            **propagation_bootstrap,
            "assumed_iid_scenarios_by_stratum": by_propagation_stratum,
        }
        del propagation_bootstrap["scenarios_by_stratum"]

    return (
        strata,
        assumptions,
        bootstrap,
        csi_bootstrap,
        receiver_bootstrap,
        propagation_bootstrap,
        family,
    )


def _atomic_write(path: Path, data: bytes) -> None:
    parent = path.parent
    if not parent.is_dir():
        raise ValueError(f"output directory does not exist: {parent}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuration", type=Path, help="assumption JSON matching the v3, v4, or v5 schema"
    )
    parser.add_argument("--output", type=Path, help="write the hash-bound report to this path")
    args = parser.parse_args(argv)
    try:
        if args.configuration.is_symlink() or not args.configuration.is_file():
            raise ValueError("configuration must be a regular non-symlink file")
        raw = args.configuration.read_bytes()
        if len(raw) > _MAX_CONFIG_BYTES:
            raise ValueError(f"configuration exceeds {_MAX_CONFIG_BYTES} bytes")
        config = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
        (
            strata,
            assumptions,
            bootstrap,
            csi_bootstrap,
            receiver_bootstrap,
            propagation_bootstrap,
            family,
        ) = _validate_config(config)
        independent = plan_g3_joint_independent_units(
            joint_acceptance_family=family,
            planned_quantile_pass_probabilities=assumptions["planned_quantile_pass_probabilities"],
            planned_interval_coverage=assumptions["planned_interval_coverage"],
            planned_complex_csi_p95_pass_probability=assumptions[
                "planned_complex_csi_p95_pass_probability"
            ],
            planned_complex_csi_failure_probability=assumptions[
                "planned_complex_csi_failure_probability"
            ],
            planned_angular_unit_pass_probability=assumptions[
                "planned_angular_unit_pass_probability"
            ],
            planned_propagation_p95_pass_probabilities=assumptions.get(
                "planned_propagation_p95_pass_probabilities"
            ),
            planned_propagation_cluster_failure_probabilities=assumptions.get(
                "planned_propagation_cluster_failure_probabilities"
            ),
            familywise_power=assumptions["familywise_power"],
        )
        bootstrap_report = None
        if bootstrap is not None:
            bootstrap_report = plan_g3_bootstrap_assurance(
                **bootstrap,
                joint_acceptance_family=family,
                stratum_roster_by_id={
                    stratum["stratum_id"]: [
                        stratum["material_id"],
                        stratum["polarization"],
                        stratum["incidence_stratum_id"],
                    ]
                    for stratum in strata
                },
                familywise_alpha=family["familywise_alpha"],
                familywise_claim_count=family["claim_count"],
            )
        csi_bootstrap_report = None
        if csi_bootstrap is not None:
            csi_bootstrap_report = plan_g3_complex_csi_bootstrap_assurance(
                joint_acceptance_family=family,
                **csi_bootstrap,
            )
        receiver_bootstrap_report = None
        if receiver_bootstrap is not None:
            receiver_bootstrap_report = plan_g3_receiver_impact_bootstrap_assurance(
                joint_acceptance_family=family,
                **receiver_bootstrap,
            )
        propagation_bootstrap_report = None
        if propagation_bootstrap is not None:
            propagation_bootstrap_report = plan_g3_propagation_bootstrap_assurance(
                joint_acceptance_family=family,
                **propagation_bootstrap,
            )
        missing_designs = []
        if bootstrap_report is None:
            missing_designs.append("material")
        if csi_bootstrap_report is None:
            missing_designs.append("complex_csi")
        if (
            config["schema_version"] in {_CONFIG_VERSION_V4, _CONFIG_VERSION_V5}
            and receiver_bootstrap_report is None
        ):
            missing_designs.append("receiver_impact")
        if config["schema_version"] == _CONFIG_VERSION_V5 and propagation_bootstrap_report is None:
            missing_designs.append("propagation")
        joint_design_composition = None
        if config["schema_version"] == _CONFIG_VERSION_V5 and all(
            plan is not None
            for plan in (
                bootstrap_report,
                csi_bootstrap_report,
                propagation_bootstrap_report,
                receiver_bootstrap_report,
            )
        ):
            joint_design_composition = compose_g3_joint_sample_size_design(
                joint_acceptance_family=family,
                joint_plan=independent,
                material_assurance_plan=bootstrap_report,
                complex_csi_assurance_plan=csi_bootstrap_report,
                propagation_assurance_plan=propagation_bootstrap_report,
                receiver_assurance_plan=receiver_bootstrap_report,
            )
            import jsonschema

            composition_schema_path = (
                REPOSITORY_ROOT
                / "datasets/schema/g3-joint-sample-size-design-composition.v1.schema.json"
            )
            composition_schema = json.loads(
                composition_schema_path.read_text(encoding="utf-8")
            )
            jsonschema.Draft202012Validator(composition_schema).validate(
                joint_design_composition
            )
            verify_g3_joint_sample_size_design(joint_design_composition)
        schema_path = (
            REPOSITORY_ROOT
            / "datasets/schema"
            / (
                "g3.03-acquisition-design-assumptions.v5.schema.json"
                if config["schema_version"] == _CONFIG_VERSION_V5
                else "g3.03-acquisition-design-assumptions.v4.schema.json"
                if config["schema_version"] == _CONFIG_VERSION_V4
                else "g3.03-acquisition-design-assumptions.v3.schema.json"
            )
        )
        report = {
            "schema": "loslab.g3_acquisition_design_report.v6"
            if config["schema_version"] == _CONFIG_VERSION_V5
            else "loslab.g3_acquisition_design_report.v4"
            if config["schema_version"] == _CONFIG_VERSION_V4
            else "loslab.g3_acquisition_design_report.v3",
            "status": "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION",
            "qualification_status": "NOT_QUALIFIED",
            "admission_possible": False,
            "scenario_label": config["scenario_label"],
            "assumption_source_description": config["assumption_source_description"],
            "expected_strata": strata,
            "joint_acceptance_family": {
                "schema": family["schema"],
                "family_id": family["family_id"],
                "family_sha256": family["family_sha256"],
                "claim_count": family["claim_count"],
                "familywise_alpha": family["familywise_alpha"],
                "per_claim_alpha": family["per_claim_alpha"],
                **(
                    {"path_equivalence_expected_scenarios": family[
                        "path_equivalence_expected_scenarios"
                    ]}
                    if family["schema"] == FAMILY_SCHEMA_V4
                    else {}
                ),
            },
            "worst_stratum_assumptions": assumptions,
            "independent_unit_design": independent,
            "bootstrap_assurance_design": bootstrap_report,
            "complex_csi_bootstrap_assurance_design": csi_bootstrap_report,
            "receiver_impact_bootstrap_assurance_design": receiver_bootstrap_report,
            "propagation_bootstrap_assurance_design": propagation_bootstrap_report,
            **(
                {"joint_sample_size_design_composition": joint_design_composition}
                if config["schema_version"] == _CONFIG_VERSION_V5
                else {}
            ),
            "assurance_target_status": {
                "material": (
                    "NOT_SUPPLIED"
                    if bootstrap_report is None
                    else "NO_EVALUATED_CANDIDATE_MET_TARGET"
                    if bootstrap_report[
                        "selected_units_per_stratum_among_evaluated_candidates"
                    ]
                    is None
                    else "SCENARIO_CONDITIONAL_CANDIDATE_FOUND"
                ),
                "complex_csi": (
                    "NOT_SUPPLIED"
                    if csi_bootstrap_report is None
                    else "NO_EVALUATED_CANDIDATE_MET_TARGET"
                    if csi_bootstrap_report[
                        "selected_clusters_per_stratum_among_evaluated_candidates"
                    ]
                    is None
                    else "SCENARIO_CONDITIONAL_CANDIDATE_FOUND"
                ),
                "receiver_impact": (
                    "NOT_SUPPLIED"
                    if receiver_bootstrap_report is None
                    else "NO_EVALUATED_CANDIDATE_MET_TARGET"
                    if receiver_bootstrap_report[
                        "selected_clusters_per_stratum_among_evaluated_candidates"
                    ]
                    is None
                    else "SCENARIO_CONDITIONAL_CANDIDATE_FOUND"
                ),
                "propagation": (
                    "NOT_SUPPLIED"
                    if propagation_bootstrap_report is None
                    else "NO_EVALUATED_CANDIDATE_MET_TARGET"
                    if propagation_bootstrap_report[
                        "selected_clusters_per_stratum_among_evaluated_candidates"
                    ]
                    is None
                    else "SCENARIO_CONDITIONAL_CANDIDATE_FOUND"
                ),
            },
            "complete_design_status": _complete_design_status(
                schema_version=config["schema_version"],
                missing_designs=missing_designs,
                material=bootstrap_report,
                complex_csi=csi_bootstrap_report,
                receiver_impact=receiver_bootstrap_report,
                propagation=propagation_bootstrap_report,
                    joint_design=joint_design_composition,
            ),
            "propagation_assurance_status": (
                "NO_EVALUATED_CANDIDATE_MEETS_TARGET_ASSURANCE"
                if propagation_bootstrap_report is not None
                and propagation_bootstrap_report[
                    "selected_clusters_per_stratum_among_evaluated_candidates"
                ]
                is None
                else "SCENARIO_CONDITIONAL_BOOTSTRAP_ASSURANCE_REPORTED"
                if propagation_bootstrap_report is not None
                else "EVALUATOR_ALIGNED_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"
                if family["schema"] in {FAMILY_SCHEMA_V2, FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}
                else "NOT_INCLUDED_IN_FAMILY_V1"
            ),
            "receiver_impact_assurance_status": (
                "NO_EVALUATED_CANDIDATE_MEETS_TARGET_ASSURANCE"
                if receiver_bootstrap_report is not None
                and receiver_bootstrap_report[
                    "selected_clusters_per_stratum_among_evaluated_candidates"
                ]
                is None
                else "SCENARIO_CONDITIONAL_BOOTSTRAP_ASSURANCE_REPORTED"
                if receiver_bootstrap_report is not None
                else "EVALUATOR_ALIGNED_BOOTSTRAP_ASSURANCE_NOT_SUPPLIED"
                if family["schema"] in {FAMILY_SCHEMA_V3, FAMILY_SCHEMA_V4}
                else "NOT_INCLUDED_IN_FAMILY_V1_V2"
            ),
            "provenance": {
                "input_exact_bytes_sha256": hashlib.sha256(raw).hexdigest(),
                "input_schema_sha256": hashlib.sha256(schema_path.read_bytes()).hexdigest(),
                "planner_source_sha256": hashlib.sha256(
                    (REPOSITORY_ROOT / "evaluation/g3_sample_size_planning.py").read_bytes()
                ).hexdigest(),
                "propagation_assurance_source_sha256": hashlib.sha256(
                    (REPOSITORY_ROOT / "evaluation/g3_propagation_assurance.py").read_bytes()
                ).hexdigest(),
                **(
                    {
                        "joint_design_composition_schema_sha256": hashlib.sha256(
                            composition_schema_path.read_bytes()
                        ).hexdigest()
                        if joint_design_composition is not None
                        else None
                    }
                    if config["schema_version"] == _CONFIG_VERSION_V5
                    else {}
                ),
                **(
                    {
                        "assurance_source_groups": {
                            name: _source_group_identity(paths)
                            for name, paths in _ASSURANCE_SOURCE_GROUPS.items()
                        }
                    }
                    if config["schema_version"] == _CONFIG_VERSION_V5
                    else {}
                ),
                "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
            "interpretation": (
                "The report only evaluates explicitly supplied, unverified assumptions; it does not "
                "independently verify the scenario label or its source. "
                + (
                    "The joint composition joins evaluator-aligned assurance counts while keeping "
                    "acquisition units, scene clusters, receiver clusters, and path episodes separate; "
                    "no combined sample size is inferred. "
                    if config["schema_version"] == _CONFIG_VERSION_V5
                    else "Branch designs remain separate; no combined sample size is inferred. "
                )
                + "No measurement, pilot, "
                "held-out, or qualification data is read."
            ),
        }
        encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if args.output is None:
            sys.stdout.buffer.write(encoded)
        else:
            if args.output.resolve() == args.configuration.resolve():
                raise ValueError("output path must not overwrite the input configuration")
            _atomic_write(args.output, encoded)
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
        TypeError,
    ) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
