"""Frozen, hash-bound inputs for a criterion-driven G3 qualification run.

The contract is a preregistration artifact. It records the design and finite
resource caps; it never promotes numerical output to scientific qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from evaluation.g3_complex_csi import (
    COMPLEX_CSI_P95_LIMIT,
)
from evaluation.g3_complex_csi import (
    MIN_INDEPENDENT_CLUSTERS_PER_STRATUM as MIN_CSI_CLUSTERS,
)
from evaluation.g3_joint_acceptance import validate_g3_joint_acceptance_family
from evaluation.g3_material_calibration import MIN_INDEPENDENT_MEASUREMENT_UNITS
from evaluation.g3_propagation_metrics import (
    MIN_INDEPENDENT_CLUSTERS_PER_STRATUM as MIN_PROPAGATION_CLUSTERS,
)
from evaluation.g3_receiver_impact import (
    MIN_INDEPENDENT_CLUSTERS_PER_STRATUM as MIN_RECEIVER_CLUSTERS,
)
from evaluation.g3_sample_size_planning import verify_g3_joint_sample_size_design

SCHEMA = "loslab.g3_qualification_run_contract.v2"
SCHEMA_V3 = "loslab.g3_qualification_run_contract.v3"
SCHEMA_V4 = "loslab.g3_qualification_run_contract.v4"
_RUN_ID = re.compile(r"^[a-z][a-z0-9._-]{2,63}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{40}$")
_BRANCHES = (
    "material_calibration",
    "complex_csi",
    "propagation_metrics",
    "receiver_impact",
    "path_equivalence",
)
_SAMPLE_UNITS = {
    "material_calibration": "independent_acquisition_unit",
    "complex_csi": "independent_physical_episode",
    "propagation_metrics": "independent_physical_episode",
    "receiver_impact": "independent_transport_block_cluster",
    "path_equivalence": "physical_episode",
}
_MINIMUM_UNITS_PER_STRATUM = {
    "material_calibration": MIN_INDEPENDENT_MEASUREMENT_UNITS,
    "complex_csi": MIN_CSI_CLUSTERS,
    "propagation_metrics": MIN_PROPAGATION_CLUSTERS,
    "receiver_impact": MIN_RECEIVER_CLUSTERS,
    "path_equivalence": 1,
}
_FAMILY_ROSTER_FIELDS = {
    "material_calibration": "material_expected_strata",
    "complex_csi": "complex_csi_expected_strata",
    "propagation_metrics": "propagation_expected_strata",
    "receiver_impact": "receiver_impact_expected_strata",
    "path_equivalence": "path_equivalence_expected_scenarios",
}
_SPEC_FIELDS = {
    "run_id",
    "joint_acceptance_family",
    "task_scope",
    "candidate",
    "challenger",
    "waveform",
    "link_budget_noise",
    "path_matching",
    "sampling",
    "solver_budget",
    "source_identity",
}
_PATH_MATCHING_BASE_FIELDS = {
    "delay_tolerance_s",
    "azimuth_tolerance_deg",
    "elevation_tolerance_deg",
    "coefficient_relative_tolerance",
    "minimum_bidirectional_coverage",
}
_PATH_MATCHING_RAW_FIELDS = {
    "vertex_tolerance_m",
    "direction_tolerance_rad",
    "h_floor",
    "transmit_power_per_re_w",
    "noise_power_per_re_w",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    result = dict(value)
    if set(result) != fields:
        raise ValueError(
            f"{name} must contain exactly the declared fields; "
            f"missing={sorted(fields - set(result))}, extra={sorted(set(result) - fields)}"
        )
    # Also rejects non-JSON values, object arrays, NaN, and infinity.
    _canonical_bytes(result)
    return result


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite and at least {minimum}")
    return number


def _string_list(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, (list, tuple)) or (nonempty and not value):
        raise ValueError(f"{name} must be a {'nonempty ' if nonempty else ''}array of strings")
    items = list(value)
    if any(
        not isinstance(item, str) or not item or item != item.strip()
        for item in items
    ):
        raise ValueError(
            f"{name} entries must be nonempty canonical strings without surrounding whitespace"
        )
    if len(set(items)) != len(items):
        raise ValueError(f"{name} entries must be unique")
    return items


def build_g3_qualification_run_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and freeze all design inputs required before a G3 run starts.

    The source identity fields must come from the caller's repository/runtime
    inventory. This function validates their form but does not authenticate
    Git hosting, measurement sources, signer custody, or physical independence.
    """
    if not isinstance(spec, Mapping) or any(not isinstance(key, str) for key in spec):
        raise TypeError("G3 qualification specification must be an object")
    if set(spec) != _SPEC_FIELDS:
        raise ValueError(
            "G3 qualification specification fields differ from the v2 contract; "
            f"missing={sorted(_SPEC_FIELDS - set(spec))}, "
            f"extra={sorted(set(spec) - _SPEC_FIELDS)}"
        )
    _canonical_bytes(dict(spec))
    run_id = spec["run_id"]
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must be 3–64 lowercase letters, digits, dots, underscores, or hyphens")
    family = validate_g3_joint_acceptance_family(spec["joint_acceptance_family"])
    if family.get("schema") != "loslab.g3_joint_acceptance_family.v4":
        raise ValueError("G3 qualification contract requires acceptance-family v4")

    task = _object(
        spec["task_scope"],
        "task_scope",
        {"task_id", "scenario_ids", "channel_conditions", "carrier_hz"},
    )
    for name in ("task_id",):
        if (
            not isinstance(task[name], str)
            or not task[name]
            or task[name] != task[name].strip()
        ):
            raise ValueError(f"task_scope.{name} must be a nonempty canonical string")
    task["scenario_ids"] = sorted(
        _string_list(task["scenario_ids"], "task_scope.scenario_ids")
    )
    conditions = _string_list(task["channel_conditions"], "task_scope.channel_conditions")
    if set(conditions) - {"LOS", "NLOS"}:
        raise ValueError("channel_conditions may contain only LOS and NLOS")
    task["channel_conditions"] = sorted(conditions)
    task["carrier_hz"] = _number(task["carrier_hz"], "task_scope.carrier_hz", minimum=1.0)
    family_scenarios = family["path_equivalence_expected_scenarios"]
    if not {row[0] for row in family_scenarios}.issubset(task["scenario_ids"]):
        raise ValueError("task_scope.scenario_ids must cover every acceptance-family scenario")
    if not {row[1] for row in family_scenarios}.issubset(task["channel_conditions"]):
        raise ValueError("task_scope.channel_conditions must cover every acceptance-family condition")

    normalized_configs: dict[str, dict[str, Any]] = {}
    for role in ("candidate", "challenger"):
        config = _object(spec[role], role, {"config_id", "configuration"})
        if (
            not isinstance(config["config_id"], str)
            or not config["config_id"]
            or config["config_id"] != config["config_id"].strip()
        ):
            raise ValueError(f"{role}.config_id must be a nonempty canonical string")
        if not isinstance(config["configuration"], Mapping):
            raise TypeError(f"{role}.configuration must be an object")
        configuration = dict(config["configuration"])
        _canonical_bytes(configuration)
        normalized_configs[role] = {
            "config_id": config["config_id"],
            "configuration": configuration,
            "configuration_sha256": _digest(configuration),
        }
    if normalized_configs["candidate"]["config_id"] == normalized_configs["challenger"]["config_id"]:
        raise ValueError("candidate and challenger config_id values must differ")

    waveform = _object(spec["waveform"], "waveform", {"waveform_id", "configuration"})
    if (
        not isinstance(waveform["waveform_id"], str)
        or not waveform["waveform_id"]
        or waveform["waveform_id"] != waveform["waveform_id"].strip()
    ):
        raise ValueError("waveform.waveform_id must be a nonempty canonical string")
    if not isinstance(waveform["configuration"], Mapping):
        raise TypeError("waveform.configuration must be an object")
    waveform_configuration = dict(waveform["configuration"])
    _canonical_bytes(waveform_configuration)
    waveform_carrier = _number(
        waveform_configuration.get("carrier_hz"),
        "waveform.configuration.carrier_hz",
        minimum=1.0,
    )
    if waveform_carrier != task["carrier_hz"]:
        raise ValueError("waveform carrier_hz differs from task_scope.carrier_hz")
    waveform = {
        "waveform_id": waveform["waveform_id"],
        "configuration": waveform_configuration,
        "configuration_sha256": _digest(waveform_configuration),
    }

    budget_fields = {
        "tx_power_dbm",
        "tx_gain_dbi",
        "rx_gain_dbi",
        "noise_figure_db",
        "noise_temperature_k",
        "noise_bandwidth_hz",
    }
    link = _object(spec["link_budget_noise"], "link_budget_noise", budget_fields)
    for key in budget_fields:
        link[key] = _number(
            link[key],
            f"link_budget_noise.{key}",
            minimum=1.0 if key in {"noise_temperature_k", "noise_bandwidth_hz"} else None,
        )

    matching = _object(
        spec["path_matching"],
        "path_matching",
        {
            "delay_tolerance_s",
            "azimuth_tolerance_deg",
            "elevation_tolerance_deg",
            "coefficient_relative_tolerance",
            "minimum_bidirectional_coverage",
        },
    )
    for key in (
        "delay_tolerance_s",
        "azimuth_tolerance_deg",
        "elevation_tolerance_deg",
        "coefficient_relative_tolerance",
    ):
        matching[key] = _number(matching[key], f"path_matching.{key}", minimum=0.0)
    coverage = _number(
        matching["minimum_bidirectional_coverage"],
        "path_matching.minimum_bidirectional_coverage",
        minimum=0.000001,
    )
    if coverage > 1.0:
        raise ValueError("minimum_bidirectional_coverage must be in [0, 1]")
    if coverage != family["path_equivalence_bidirectional_coverage_min"]:
        raise ValueError(
            "path-matching bidirectional coverage must match the acceptance-family criterion"
        )
    matching["minimum_bidirectional_coverage"] = coverage

    sampling = _object(
        spec["sampling"],
        "sampling",
        {
            "episode_ids",
            "path_equivalence_scenarios",
            "unit_by_branch",
            "planned_units_by_branch",
            "planned_units_by_stratum",
            "bootstrap_seed",
            "bootstrap_draws",
        },
    )
    episode_ids = _string_list(sampling["episode_ids"], "sampling.episode_ids")
    path_roster = sampling["path_equivalence_scenarios"]
    if not isinstance(path_roster, (list, tuple)) or not path_roster:
        raise ValueError("sampling.path_equivalence_scenarios must be a nonempty roster")
    normalized_path_roster: list[list[str]] = []
    for row in path_roster:
        if (
            not isinstance(row, (list, tuple))
            or len(row) != 3
            or any(
                not isinstance(item, str) or not item or item != item.strip()
                for item in row
            )
        ):
            raise ValueError("path-equivalence roster rows must be nonempty string triples")
        normalized_path_roster.append(list(row))
    if len({tuple(row) for row in normalized_path_roster}) != len(normalized_path_roster):
        raise ValueError("sampling.path_equivalence_scenarios must be unique")
    family_path_roster = [list(row) for row in family["path_equivalence_expected_scenarios"]]
    if sorted(normalized_path_roster) != sorted(family_path_roster):
        raise ValueError("sampling.path_equivalence_scenarios must exactly match the acceptance family")
    path_episodes = sorted({row[2] for row in family_path_roster})
    if sorted(episode_ids) != path_episodes:
        raise ValueError("sampling.episode_ids must exactly match the acceptance family's path roster")
    units = _object(sampling["unit_by_branch"], "sampling.unit_by_branch", set(_BRANCHES))
    if units != _SAMPLE_UNITS:
        raise ValueError("sampling.unit_by_branch differs from the declared independent-unit contract")
    planned = _object(
        sampling["planned_units_by_branch"], "sampling.planned_units_by_branch", set(_BRANCHES)
    )
    for branch, count in planned.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"planned_units_by_branch.{branch} must be a positive integer")
    stratum_plans_raw = _object(
        sampling["planned_units_by_stratum"],
        "sampling.planned_units_by_stratum",
        set(_BRANCHES),
    )
    stratum_plans: dict[str, list[dict[str, Any]]] = {}
    for branch in _BRANCHES:
        expected_roster = family.get(_FAMILY_ROSTER_FIELDS[branch])
        if not isinstance(expected_roster, (list, tuple)) or not expected_roster:
            raise ValueError(f"acceptance family has no expected roster for {branch}")
        expected_keys = [tuple(item) for item in expected_roster]
        raw_rows = stratum_plans_raw[branch]
        if not isinstance(raw_rows, (list, tuple)) or not raw_rows:
            raise ValueError(f"planned_units_by_stratum.{branch} must be a nonempty roster")
        normalized_rows: list[dict[str, Any]] = []
        observed_keys: list[tuple[str, ...]] = []
        for row in raw_rows:
            if not isinstance(row, Mapping) or set(row) != {"stratum", "planned_units"}:
                raise ValueError(
                    f"planned_units_by_stratum.{branch} rows must contain exactly "
                    "stratum and planned_units"
                )
            stratum = row["stratum"]
            if (
                not isinstance(stratum, (list, tuple))
                or any(
                    not isinstance(value, str) or not value or value != value.strip()
                    for value in stratum
                )
            ):
                raise ValueError(
                    f"planned_units_by_stratum.{branch}.stratum must contain canonical strings"
                )
            key = tuple(stratum)
            if key in observed_keys:
                raise ValueError(f"planned_units_by_stratum.{branch} contains duplicate strata")
            observed_keys.append(key)
            count = row["planned_units"]
            minimum = _MINIMUM_UNITS_PER_STRATUM[branch]
            if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
                raise ValueError(
                    f"planned_units_by_stratum.{branch}{key} must be at least {minimum}; "
                    "this is an evaluator/pilot floor, not a powered sample-size guarantee"
                )
            if branch == "path_equivalence" and count != 1:
                raise ValueError(
                    "path_equivalence planned_units must be exactly one per frozen episode"
                )
            normalized_rows.append({"stratum": list(key), "planned_units": count})
        if set(observed_keys) != set(expected_keys) or len(observed_keys) != len(expected_keys):
            raise ValueError(
                f"planned_units_by_stratum.{branch} must exactly match its acceptance-family roster"
            )
        normalized_rows.sort(key=lambda row: tuple(row["stratum"]))
        stratum_total = sum(row["planned_units"] for row in normalized_rows)
        if planned[branch] != stratum_total:
            raise ValueError(
                f"planned_units_by_branch.{branch} must equal the sum of its per-stratum counts"
            )
        stratum_plans[branch] = normalized_rows
    seed = sampling["bootstrap_seed"]
    draws = sampling["bootstrap_draws"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampling.bootstrap_seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1000:
        raise ValueError("sampling.bootstrap_draws must be an integer of at least 1000")
    sampling = {
        "episode_ids": sorted(episode_ids),
        "path_equivalence_scenarios": sorted(normalized_path_roster),
        "unit_by_branch": units,
        "planned_units_by_branch": planned,
        "planned_units_by_stratum": stratum_plans,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
    }

    caps = _object(
        spec["solver_budget"],
        "solver_budget",
        {"max_rays_per_link", "max_interactions", "max_episodes", "max_wall_seconds", "overflow_policy"},
    )
    for key in ("max_rays_per_link", "max_interactions", "max_episodes"):
        value = caps[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"solver_budget.{key} must be a positive integer")
    caps["max_wall_seconds"] = _number(
        caps["max_wall_seconds"], "solver_budget.max_wall_seconds", minimum=0.000001
    )
    if caps["overflow_policy"] not in {"abort_run", "record_censored_episode"}:
        raise ValueError("solver_budget.overflow_policy is unsupported")

    source = _object(
        spec["source_identity"],
        "source_identity",
        {"git_revision", "working_tree_dirty", "source_bundle_sha256"},
    )
    if not isinstance(source["git_revision"], str) or _REVISION.fullmatch(source["git_revision"]) is None:
        raise ValueError("source_identity.git_revision must be a 40-character lowercase Git SHA")
    if not isinstance(source["working_tree_dirty"], bool):
        raise TypeError("source_identity.working_tree_dirty must be boolean")
    if not isinstance(source["source_bundle_sha256"], str) or _SHA256.fullmatch(
        source["source_bundle_sha256"]
    ) is None:
        raise ValueError("source_identity.source_bundle_sha256 must be a SHA-256 digest")

    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "run_id": run_id,
        "qualification_status": "NOT_QUALIFIED",
        "joint_acceptance_family": family,
        "joint_acceptance_family_sha256": family["family_sha256"],
        "task_scope": task,
        **normalized_configs,
        "waveform": waveform,
        "link_budget_noise": link,
        "path_matching": matching,
        "sampling": sampling,
        "solver_budget": caps,
        "source_identity": source,
        "finite_solver_caps_are_ground_truth": False,
    }
    contract["contract_sha256"] = _digest(contract)
    return contract


def _validate_sample_size_design_binding(
    base_contract: Mapping[str, Any], design: Mapping[str, Any]
) -> dict[str, Any]:
    verification = verify_g3_joint_sample_size_design(design)
    if verification["complete_precision_powered_g3_design"] is not True:
        raise ValueError(
            "G3 contract v3 requires a complete precision- and power-based design"
        )
    family = base_contract["joint_acceptance_family"]
    identity = design.get("joint_acceptance_family")
    if not isinstance(identity, Mapping) or identity != {
        "family_id": family["family_id"],
        "family_sha256": family["family_sha256"],
    }:
        raise ValueError("sample-size design acceptance family differs from contract")
    branches = design.get("branch_designs")
    sampling = base_contract["sampling"]
    for branch in _BRANCHES:
        branch_design = branches[branch]
        requirements = branch_design["required_units_by_stratum"]
        planned = sampling["planned_units_by_stratum"][branch]
        required_by_stratum = {
            tuple(row["stratum"]): row["required_units"] for row in requirements
        }
        planned_by_stratum = {
            tuple(row["stratum"]): row["planned_units"] for row in planned
        }
        if set(required_by_stratum) != set(planned_by_stratum):
            raise ValueError(f"{branch} sample-size strata differ from the frozen contract")
        for stratum, required in required_by_stratum.items():
            if required is None or required < _MINIMUM_UNITS_PER_STRATUM[branch]:
                raise ValueError(
                    f"{branch} powered design requirement is below the evaluator floor"
                )
            if required is None or planned_by_stratum[stratum] < required:
                raise ValueError(
                    f"{branch} planned count for {stratum} is below its powered design requirement"
                )
    return dict(design)


def build_g3_qualification_run_contract_v3(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build a qualification contract bound to a complete joint sample-size design.

    V2 remains available for historical diagnostic receipts. V3 is the only
    contract version admitted to new contract-bound evaluator runs.
    """
    if not isinstance(spec, Mapping) or any(not isinstance(key, str) for key in spec):
        raise TypeError("G3 qualification specification must be an object")
    expected_fields = _SPEC_FIELDS | {"sample_size_design"}
    if set(spec) != expected_fields:
        raise ValueError(
            "G3 qualification v3 specification fields differ; "
            f"missing={sorted(expected_fields - set(spec))}, "
            f"extra={sorted(set(spec) - expected_fields)}"
        )
    base_spec = {key: value for key, value in spec.items() if key in _SPEC_FIELDS}
    base_contract = build_g3_qualification_run_contract(base_spec)
    design = _validate_sample_size_design_binding(base_contract, spec["sample_size_design"])
    contract = {key: value for key, value in base_contract.items() if key != "contract_sha256"}
    contract["schema"] = SCHEMA_V3
    contract["sample_size_design"] = design
    contract["sample_size_design_sha256"] = design["design_sha256"]
    contract["contract_sha256"] = _digest(contract)
    return contract


def build_g3_qualification_run_contract_v4(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze every geometry and receiver-relevance input used by raw path scoring.

    V2 and v3 contracts remain readable for historical diagnostic replay. V4
    adds the parameters needed to recompute path-equivalence comparisons from
    raw path records under one preregistered analysis configuration.
    """
    if not isinstance(spec, Mapping) or any(not isinstance(key, str) for key in spec):
        raise TypeError("G3 qualification specification must be an object")
    expected_fields = _SPEC_FIELDS | {"sample_size_design"}
    if set(spec) != expected_fields:
        raise ValueError(
            "G3 qualification v4 specification fields differ; "
            f"missing={sorted(expected_fields - set(spec))}, "
            f"extra={sorted(set(spec) - expected_fields)}"
        )
    path_matching = spec.get("path_matching")
    required_matching = _PATH_MATCHING_BASE_FIELDS | _PATH_MATCHING_RAW_FIELDS
    if not isinstance(path_matching, Mapping) or set(path_matching) != required_matching:
        raise ValueError("path_matching must contain the exact v4 matching parameter roster")
    normalized_matching = dict(path_matching)
    for key in _PATH_MATCHING_RAW_FIELDS:
        normalized_matching[key] = _number(
            normalized_matching[key], f"path_matching.{key}", minimum=0.0
        )
        if normalized_matching[key] == 0.0:
            raise ValueError(f"path_matching.{key} must be strictly positive")
    if normalized_matching["coefficient_relative_tolerance"] > COMPLEX_CSI_P95_LIMIT:
        raise ValueError(
            "path_matching.coefficient_relative_tolerance must not exceed the frozen "
            f"complex-CSI p95 error criterion ({COMPLEX_CSI_P95_LIMIT})"
        )
    if normalized_matching["direction_tolerance_rad"] > math.pi:
        raise ValueError("path_matching.direction_tolerance_rad must not exceed pi")

    task_scope = spec.get("task_scope")
    if not isinstance(task_scope, Mapping):
        raise TypeError("task_scope must be an object")
    effective_bandwidth_hz = None
    if "effective_bandwidth_hz" in task_scope:
        effective_bandwidth_hz = _number(
            task_scope["effective_bandwidth_hz"],
            "task_scope.effective_bandwidth_hz",
            minimum=1.0,
        )

    base_spec = dict(spec)
    base_spec["task_scope"] = {
        key: value for key, value in task_scope.items() if key != "effective_bandwidth_hz"
    }
    base_spec["path_matching"] = {
        key: path_matching[key] for key in sorted(_PATH_MATCHING_BASE_FIELDS)
    }
    contract = build_g3_qualification_run_contract_v3(base_spec)
    contract = {key: value for key, value in contract.items() if key != "contract_sha256"}
    contract["schema"] = SCHEMA_V4
    if effective_bandwidth_hz is not None:
        contract["task_scope"]["effective_bandwidth_hz"] = effective_bandwidth_hz
    contract["path_matching"].update(
        {key: normalized_matching[key] for key in sorted(_PATH_MATCHING_RAW_FIELDS)}
    )
    contract["contract_sha256"] = _digest(contract)
    return contract


def verify_g3_qualification_run_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute a frozen contract and report integrity without qualifying it."""
    if not isinstance(contract, Mapping):
        raise TypeError("contract must be an object")
    saved = dict(contract)
    declared_digest = saved.pop("contract_sha256", None)
    if not isinstance(declared_digest, str) or _SHA256.fullmatch(declared_digest) is None:
        raise ValueError("contract_sha256 must be a SHA-256 digest")
    if _digest(saved) != declared_digest:
        raise ValueError("G3 qualification contract digest does not match its contents")
    schema = saved.get("schema")
    expected_fields = {
        "schema",
        "run_id",
        "qualification_status",
        "joint_acceptance_family",
        "joint_acceptance_family_sha256",
        "task_scope",
        "candidate",
        "challenger",
        "waveform",
        "link_budget_noise",
        "path_matching",
        "sampling",
        "solver_budget",
        "source_identity",
        "finite_solver_caps_are_ground_truth",
    }
    if schema == SCHEMA_V3:
        expected_fields |= {"sample_size_design", "sample_size_design_sha256"}
    elif schema == SCHEMA_V4:
        expected_fields |= {"sample_size_design", "sample_size_design_sha256"}
    if set(saved) != expected_fields:
        raise ValueError("G3 qualification contract fields differ from its frozen schema")
    if schema not in {SCHEMA, SCHEMA_V3, SCHEMA_V4}:
        raise ValueError("G3 qualification contract schema is unsupported")
    spec = {
        "run_id": saved["run_id"],
        "joint_acceptance_family": saved["joint_acceptance_family"],
        "task_scope": saved["task_scope"],
        "candidate": {
            "config_id": saved["candidate"]["config_id"],
            "configuration": saved["candidate"]["configuration"],
        },
        "challenger": {
            "config_id": saved["challenger"]["config_id"],
            "configuration": saved["challenger"]["configuration"],
        },
        "waveform": {
            "waveform_id": saved["waveform"]["waveform_id"],
            "configuration": saved["waveform"]["configuration"],
        },
        "link_budget_noise": saved["link_budget_noise"],
        "path_matching": saved["path_matching"],
        "sampling": saved["sampling"],
        "solver_budget": saved["solver_budget"],
        "source_identity": saved["source_identity"],
    }
    if schema in {SCHEMA_V3, SCHEMA_V4}:
        design = saved.get("sample_size_design")
        if not isinstance(design, Mapping):
            raise ValueError("sample_size_design must be an object")
        if saved.get("sample_size_design_sha256") != design.get("design_sha256"):
            raise ValueError("sample_size_design_sha256 differs from embedded design")
        spec["sample_size_design"] = dict(design)
        if schema == SCHEMA_V4:
            rebuilt = build_g3_qualification_run_contract_v4(spec)
        else:
            rebuilt = build_g3_qualification_run_contract_v3(spec)
    else:
        rebuilt = build_g3_qualification_run_contract(spec)
    if rebuilt != dict(contract):
        raise ValueError("G3 qualification contract differs from deterministic reconstruction")
    return {
        "schema": "loslab.g3_qualification_run_contract_verification.v1",
        "status": "PASS_CONTRACT_INTEGRITY_NOT_QUALIFIED",
        "qualification_status": "NOT_QUALIFIED",
        "run_id": rebuilt["run_id"],
        "contract_sha256": rebuilt["contract_sha256"],
        "finite_solver_caps_are_ground_truth": False,
    }
