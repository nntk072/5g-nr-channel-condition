"""Roster-complete, scene-episode summaries for G3 path-equivalence diagnostics.

This layer checks the preregistered evaluation roster and applies the frozen
per-comparison bidirectional class-coverage rule to every independent episode.
It does not turn a finite set of path comparisons into population evidence or
qualification.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from numbers import Real
from typing import Any

import numpy as np

from evaluation.g3_joint_acceptance import (
    FAMILY_SCHEMA_V4,
    PATH_COVERAGE_LIMIT,
    validate_g3_joint_acceptance_family,
)

_COVERAGE_LIMIT = PATH_COVERAGE_LIMIT
_SCHEMA = "loslab.g3_path_equivalence_roster_diagnostic.v2"
_RAW_CONFIGURATION_FIELDS = {
    "candidate_configuration_id",
    "challenger_configuration_id",
    "vertex_tolerance_m",
    "delay_tolerance_s",
    "direction_tolerance_rad",
    "h_floor",
    "transmit_power_per_re_w",
    "noise_power_per_re_w",
}
_RAW_RECORD_FIELDS = {
    "family_id",
    "condition",
    "episode_id",
    "candidate_paths",
    "challenger_paths",
    "candidate_full_cfr_real",
    "candidate_full_cfr_imag",
    "challenger_full_cfr_real",
    "challenger_full_cfr_imag",
}


def _recompute_raw_comparison(
    row: dict[str, Any],
    configuration: dict[str, Any],
    *,
    solver_budget_provenance: dict[str, Any] | None = None,
    coefficient_relative_tolerance: float | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None]:
    """Recompute class matching from both solvers' raw paths and full CFRs."""
    from evaluation.g3_path_equivalence import (
        build_path_equivalence_classes,
        compare_diffuse_stochastic_path_distributions,
        compare_path_equivalence_classes,
    )

    if set(configuration) != _RAW_CONFIGURATION_FIELDS:
        raise ValueError("raw path comparison configuration has an unsupported field roster")
    normalized: dict[str, Any] = {}
    for key in ("candidate_configuration_id", "challenger_configuration_id"):
        value = configuration[key]
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{key} must be a nonempty canonical string")
        normalized[key] = value
    numeric_fields = _RAW_CONFIGURATION_FIELDS - {
        "candidate_configuration_id",
        "challenger_configuration_id",
    }
    for key in numeric_fields:
        value = configuration[key]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite positive number")
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{key} must be a finite positive number")
        normalized[key] = number
    if coefficient_relative_tolerance is not None:
        if (
            isinstance(coefficient_relative_tolerance, bool)
            or not isinstance(coefficient_relative_tolerance, Real)
            or not math.isfinite(float(coefficient_relative_tolerance))
            or coefficient_relative_tolerance < 0.0
        ):
            raise ValueError("coefficient_relative_tolerance must be finite and nonnegative")
        normalized["coefficient_relative_tolerance"] = float(
            coefficient_relative_tolerance
        )
    if normalized["direction_tolerance_rad"] > math.pi:
        raise ValueError("direction_tolerance_rad must not exceed pi")

    config_bytes = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    resolution_id = "g3-path-matching-v1:" + hashlib.sha256(config_bytes).hexdigest()
    tolerance_args = {
        "resolution_id": resolution_id,
        "vertex_tolerance_m": normalized["vertex_tolerance_m"],
        "delay_tolerance_s": normalized["delay_tolerance_s"],
        "direction_tolerance_rad": normalized["direction_tolerance_rad"],
    }
    candidate_paths = row.get("candidate_paths")
    challenger_paths = row.get("challenger_paths")
    diffuse_distribution: dict[str, Any] | None = None
    if solver_budget_provenance is None:
        candidate_identity_paths = candidate_paths
        challenger_identity_paths = challenger_paths
    else:
        if not isinstance(candidate_paths, list) or not isinstance(challenger_paths, list):
            raise ValueError("raw candidate/challenger paths must be arrays")

        def split_path_records(
            paths: list[Any], side: str
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
            identity_paths: list[dict[str, Any]] = []
            diffuse_paths: list[dict[str, Any]] = []
            for path in paths:
                if not isinstance(path, dict):
                    raise ValueError(f"{side} raw paths must contain path objects")
                kind = path.get("path_kind")
                if kind in {"specular", "deterministic"}:
                    identity_paths.append(path)
                elif kind == "diffuse_stochastic":
                    diffuse_paths.append(path)
                else:
                    raise ValueError(f"{side} raw path kind is unsupported")
            return identity_paths, diffuse_paths

        candidate_identity_paths, candidate_diffuse_paths = split_path_records(
            candidate_paths, "candidate"
        )
        challenger_identity_paths, challenger_diffuse_paths = split_path_records(
            challenger_paths, "challenger"
        )
        diffuse_distribution = compare_diffuse_stochastic_path_distributions(
            candidate_diffuse_paths,
            challenger_diffuse_paths,
            solver_budget_provenance=solver_budget_provenance,
        )
    candidate_classes = build_path_equivalence_classes(candidate_identity_paths, **tolerance_args)
    challenger_classes = build_path_equivalence_classes(challenger_identity_paths, **tolerance_args)

    def full_cfr(real_key: str, imag_key: str) -> np.ndarray:
        real_values, imag_values = row.get(real_key), row.get(imag_key)
        if (
            not isinstance(real_values, list)
            or not isinstance(imag_values, list)
            or not real_values
            or len(real_values) != len(imag_values)
        ):
            raise ValueError("full CFR components must be equal nonempty arrays")
        components = real_values + imag_values
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in components):
            raise ValueError("full CFR components must contain real numeric values")
        result = np.asarray(real_values, dtype=np.float64) + 1j * np.asarray(
            imag_values, dtype=np.float64
        )
        if not np.isfinite(result).all():
            raise ValueError("full CFR components must be finite")
        return result

    comparison = compare_path_equivalence_classes(
        candidate_classes,
        challenger_classes,
        **tolerance_args,
        candidate_full_cfr=full_cfr("candidate_full_cfr_real", "candidate_full_cfr_imag"),
        challenger_full_cfr=full_cfr("challenger_full_cfr_real", "challenger_full_cfr_imag"),
        h_floor=normalized["h_floor"],
        transmit_power_per_re_w=normalized["transmit_power_per_re_w"],
        noise_power_per_re_w=normalized["noise_power_per_re_w"],
        coefficient_relative_tolerance=normalized.get("coefficient_relative_tolerance"),
        candidate_configuration_id=normalized["candidate_configuration_id"],
        challenger_configuration_id=normalized["challenger_configuration_id"],
    )
    raw_bytes = json.dumps(
        row, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return comparison, hashlib.sha256(raw_bytes).hexdigest(), diffuse_distribution


def _reported_coverage_matches(value: Any, expected: float | None) -> bool:
    if expected is None:
        return value is None
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _validate_comparison_coverage(comparison: dict[str, Any], key: tuple[str, str, str]) -> str:
    """Reconcile a comparator report's directional percentages with class counts."""
    class_status = comparison.get("class_coverage_status")
    if comparison.get("status") == "NO_CLASSES_NOT_ASSESSABLE":
        if (
            class_status != "NOT_ASSESSABLE_NO_CLASSES"
            or comparison.get("matched_classes") != []
            or comparison.get("candidate_only_relevant_classes") != []
            or comparison.get("challenger_only_relevant_classes") != []
            or comparison.get("candidate_bidirectional_coverage") is not None
            or comparison.get("challenger_bidirectional_coverage") is not None
            or comparison.get("coverage_criterion", _COVERAGE_LIMIT) != _COVERAGE_LIMIT
        ):
            raise ValueError(f"no-class comparison is internally inconsistent: {key}")
        for name in (
            "candidate_relevant_class_count",
            "challenger_relevant_class_count",
            "matched_class_pair_count",
        ):
            if name in comparison:
                value = comparison[name]
                if isinstance(value, bool) or not isinstance(value, int) or value != 0:
                    raise ValueError(f"no-class comparison has an invalid {name}: {key}")
        return class_status

    if comparison.get("status") != "PATH_COVERAGE_DIAGNOSTIC_NOT_QUALIFIED":
        raise ValueError(f"comparison has an unsupported diagnostic status: {key}")
    if comparison.get("coverage_criterion") != _COVERAGE_LIMIT:
        raise ValueError(f"comparison uses a different coverage criterion: {key}")
    coefficient_tolerance = comparison.get("coefficient_relative_tolerance")
    if coefficient_tolerance is not None and (
        isinstance(coefficient_tolerance, bool)
        or not isinstance(coefficient_tolerance, (int, float))
        or not math.isfinite(float(coefficient_tolerance))
        or float(coefficient_tolerance) < 0.0
    ):
        raise ValueError(f"comparison coefficient tolerance is invalid: {key}")

    def count(name: str) -> int:
        value = comparison.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"comparison {name} must be a nonnegative integer: {key}")
        return value

    matched = comparison.get("matched_classes")
    candidate_only = comparison.get("candidate_only_relevant_classes")
    challenger_only = comparison.get("challenger_only_relevant_classes")
    if not isinstance(matched, list) or not isinstance(candidate_only, list) or not isinstance(
        challenger_only, list
    ):
        raise ValueError(f"comparison class rosters must be lists: {key}")
    if count("matched_class_pair_count") != len(matched):
        raise ValueError(f"comparison matched-pair count disagrees with its roster: {key}")

    def matched_relevant_count(side: str) -> int:
        field = f"{side}_receiver_relevance"
        relevant = 0
        for item in matched:
            if not isinstance(item, dict):
                raise ValueError(f"comparison matched-class record is invalid: {key}")
            details = item.get(field)
            if not isinstance(details, dict) or not isinstance(
                details.get("receiver_relevant"), bool
            ):
                raise ValueError(f"comparison matched-class relevance is invalid: {key}")
            deltas = item.get("geometry_and_channel_deltas")
            if not isinstance(deltas, dict):
                raise ValueError(f"comparison matched-class deltas are missing: {key}")
            for name in (
                "maximum_interaction_vertex_displacement_m",
                "absolute_delay_difference_s",
                "maximum_endpoint_direction_difference_rad",
                "aggregate_class_cfr_e_h",
            ):
                value = deltas.get(name)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                ):
                    raise ValueError(
                        f"comparison matched-class delta {name} must be finite and nonnegative: {key}"
                    )
            coefficient_error = deltas.get("aggregate_coefficient_relative_error")
            if coefficient_tolerance is not None and (
                isinstance(coefficient_error, bool)
                or not isinstance(coefficient_error, (int, float))
                or not math.isfinite(float(coefficient_error))
                or float(coefficient_error) < 0.0
            ):
                raise ValueError(
                    f"comparison coefficient error must be finite and nonnegative: {key}"
                )
            relevant_pair = bool(
                details["receiver_relevant"]
                or item.get(
                    "challenger_receiver_relevance"
                    if side == "candidate"
                    else "candidate_receiver_relevance",
                    {},
                ).get("receiver_relevant")
            )
            criterion_result = item.get("coefficient_criterion_pass")
            if coefficient_tolerance is None:
                if "coefficient_criterion_pass" in item:
                    raise ValueError(f"unexpected coefficient criterion result: {key}")
            elif relevant_pair:
                expected_criterion_result = (
                    float(coefficient_error) <= float(coefficient_tolerance)
                )
                if criterion_result is not expected_criterion_result:
                    raise ValueError(f"coefficient criterion result disagrees with delta: {key}")
            elif criterion_result is not None:
                raise ValueError(f"irrelevant matched class has a coefficient decision: {key}")
            relevant += int(details["receiver_relevant"])
        return relevant

    def validate_unmatched(rows: list[Any]) -> None:
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError(f"comparison unmatched-class record is invalid: {key}")
            relevance = item.get("receiver_relevance")
            if not isinstance(relevance, dict) or relevance.get("receiver_relevant") is not True:
                raise ValueError(f"comparison unmatched roster contains an irrelevant class: {key}")

    validate_unmatched(candidate_only)
    validate_unmatched(challenger_only)
    candidate_count = count("candidate_relevant_class_count")
    challenger_count = count("challenger_relevant_class_count")
    candidate_matched = matched_relevant_count("candidate")
    challenger_matched = matched_relevant_count("challenger")
    if coefficient_tolerance is not None:
        expected_assessable = sum(
            int(
                item["candidate_receiver_relevance"]["receiver_relevant"]
                or item["challenger_receiver_relevance"]["receiver_relevant"]
            )
            for item in matched
        )
        expected_failures = sum(
            int(item["coefficient_criterion_pass"] is False)
            for item in matched
            if item["candidate_receiver_relevance"]["receiver_relevant"]
            or item["challenger_receiver_relevance"]["receiver_relevant"]
        )
        if (
            count("coefficient_criterion_assessable_class_pairs") != expected_assessable
            or count("coefficient_criterion_failed_class_pairs") != expected_failures
        ):
            raise ValueError(f"coefficient criterion counts disagree with matched classes: {key}")
        expected_coefficient_status = (
            "PASS"
            if expected_assessable and not expected_failures
            else "FAIL"
            if expected_failures
            else "NOT_ASSESSABLE_NO_RELEVANT_MATCHED_CLASSES"
        )
        if comparison.get("coefficient_criterion_status") != expected_coefficient_status:
            raise ValueError(f"coefficient criterion status disagrees with class rows: {key}")
    if candidate_count != candidate_matched + len(candidate_only):
        raise ValueError(f"candidate relevant-class count disagrees with its roster: {key}")
    if challenger_count != challenger_matched + len(challenger_only):
        raise ValueError(f"challenger relevant-class count disagrees with its roster: {key}")
    candidate_coverage = candidate_matched / candidate_count if candidate_count else None
    challenger_coverage = challenger_matched / challenger_count if challenger_count else None
    if not _reported_coverage_matches(
        comparison.get("candidate_bidirectional_coverage"), candidate_coverage
    ) or not _reported_coverage_matches(
        comparison.get("challenger_bidirectional_coverage"), challenger_coverage
    ):
        raise ValueError(f"reported directional coverage disagrees with class counts: {key}")

    assessable = candidate_count > 0 or challenger_count > 0
    passes = (
        assessable
        and candidate_coverage is not None
        and challenger_coverage is not None
        and candidate_coverage >= _COVERAGE_LIMIT
        and challenger_coverage >= _COVERAGE_LIMIT
    )
    coefficient_pass = (
        comparison.get("coefficient_criterion_failed_class_pairs") == 0
        if coefficient_tolerance is not None
        else True
    )
    expected_status = (
        "PASS"
        if passes and coefficient_pass
        else "FAIL"
        if assessable
        else "UNRESOLVED_NO_RELEVANT_CLASSES"
    )
    if class_status != expected_status:
        if class_status == "PASS" and not passes:
            raise ValueError(f"comparison claims PASS below the frozen coverage criterion: {key}")
        raise ValueError(f"comparison coverage status disagrees with class counts: {key}")
    return expected_status


def evaluate_path_equivalence_roster(
    records: Sequence[dict[str, Any]],
    *,
    expected_scenarios: Sequence[tuple[str, str, str]],
    joint_acceptance_family: dict[str, Any] | None = None,
    comparison_configuration: dict[str, Any] | None = None,
    solver_budget_provenance: dict[str, Any] | None = None,
    coefficient_relative_tolerance: float | None = None,
) -> dict[str, Any]:
    """Require one diagnostic comparison for every frozen family/condition/episode.

    Legacy records may contain a precomputed ``comparison``. When a shared
    ``comparison_configuration`` is supplied, every row instead contains raw
    candidate/challenger paths and full CFRs, and the comparison is recomputed.
    When solver-budget provenance is also supplied, diffuse stochastic paths
    are summarized as distributions and excluded from identity matching.
    Coverage is always assessed per episode and never pooled.
    """
    if not records:
        raise ValueError("records must be nonempty")
    if not expected_scenarios:
        raise ValueError("expected_scenarios must be a nonempty preregistered roster")
    if solver_budget_provenance is not None and comparison_configuration is None:
        raise ValueError(
            "solver-budget-bound diffuse diagnostics require raw paths and a comparison configuration"
        )

    expected: set[tuple[str, str, str]] = set()
    for item in expected_scenarios:
        if (
            not isinstance(item, tuple)
            or len(item) != 3
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in item
            )
            or item[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError("expected scenarios must be (family_id, LOS/NLOS, episode_id)")
        if item in expected:
            raise ValueError(f"duplicate expected scenario: {item}")
        expected.add(item)

    family_reference = None
    if joint_acceptance_family is not None:
        family = validate_g3_joint_acceptance_family(joint_acceptance_family)
        if family["schema"] != FAMILY_SCHEMA_V4:
            raise ValueError("path-equivalence roster requires joint acceptance family v4")
        declared = {
            tuple(item) for item in family["path_equivalence_expected_scenarios"]
        }
        if declared != expected:
            raise ValueError("path-equivalence scenario roster differs from joint family v4")
        if family["path_equivalence_bidirectional_coverage_min"] != PATH_COVERAGE_LIMIT:
            raise ValueError("joint family path-coverage criterion is unsupported")
        if family["path_equivalence_admission_possible"] is not False:
            raise ValueError("joint family path-equivalence results must remain diagnostic-only")
        family_reference = {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
        }

    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("each scenario record must be an object")
        family, condition, episode = (
            row.get("family_id"),
            row.get("condition"),
            row.get("episode_id"),
        )
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in (family, episode)
        ):
            raise ValueError("family_id and episode_id must be nonempty canonical strings")
        if not isinstance(condition, str) or condition not in {"LOS", "NLOS"}:
            raise ValueError("condition must be LOS or NLOS")
        key = (family, condition, episode)
        if key in observed:
            raise ValueError(f"duplicate path-equivalence episode: {key}")
        raw_path_input_sha256 = None
        diffuse_distribution = None
        if comparison_configuration is None:
            comparison = row.get("comparison")
            if not isinstance(comparison, dict):
                raise ValueError("each legacy record needs a comparison result")
        else:
            if "comparison" in row:
                raise ValueError("raw path records must not include a precomputed comparison")
            if set(row) != _RAW_RECORD_FIELDS:
                raise ValueError(
                    "raw path records must contain exactly the frozen path and full-CFR fields"
                )
            comparison, raw_path_input_sha256, diffuse_distribution = _recompute_raw_comparison(
                row,
                comparison_configuration,
                solver_budget_provenance=solver_budget_provenance,
                coefficient_relative_tolerance=coefficient_relative_tolerance,
            )
        if comparison.get("schema") != "loslab.g3_path_equivalence_comparison.v1":
            raise ValueError("comparison has an unsupported schema")
        if comparison.get("admission_possible") is not False:
            raise ValueError("path-equivalence comparisons must remain diagnostic-only")
        observed[key] = {
            "comparison": comparison,
            "raw_path_input_sha256": raw_path_input_sha256,
            "diffuse_distribution": diffuse_distribution,
        }

    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        unexpected = sorted(set(observed) - expected)
        raise ValueError(
            f"observed scenario roster differs from preregistration; "
            f"missing={missing}, unexpected={unexpected}"
        )

    scenario_reports: list[dict[str, Any]] = []
    by_stratum: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "coverage_pass": 0, "coverage_fail": 0, "unresolved": 0}
    )
    for key in sorted(expected):
        observed_record = observed[key]
        comparison = observed_record["comparison"]
        status = _validate_comparison_coverage(comparison, key)
        candidate = comparison.get("candidate_bidirectional_coverage")
        challenger = comparison.get("challenger_bidirectional_coverage")
        if status not in {"PASS", "FAIL", "UNRESOLVED_NO_RELEVANT_CLASSES", "NOT_ASSESSABLE_NO_CLASSES"}:
            raise ValueError(f"unsupported path-coverage status for {key}: {status!r}")
        for value, name in ((candidate, "candidate"), (challenger, "challenger")):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} coverage for {key} must be in [0, 1] or null")
        counter = by_stratum[(key[0], key[1])]
        counter["episodes"] += 1
        bucket = (
            "coverage_pass"
            if status == "PASS"
            else "coverage_fail"
            if status == "FAIL"
            else "unresolved"
        )
        counter[bucket] += 1
        scenario_reports.append(
            {
                "family_id": key[0],
                "condition": key[1],
                "episode_id": key[2],
                "class_coverage_status": status,
                "candidate_bidirectional_coverage": candidate,
                "challenger_bidirectional_coverage": challenger,
                "raw_path_input_sha256": observed_record["raw_path_input_sha256"],
                **(
                    {"diffuse_distribution": observed_record["diffuse_distribution"]}
                    if solver_budget_provenance is not None
                    else {}
                ),
            }
        )

    all_pass = all(item["class_coverage_status"] == "PASS" for item in scenario_reports)
    any_fail = any(item["class_coverage_status"] == "FAIL" for item in scenario_reports)
    return {
        "schema": (
            "loslab.g3_path_equivalence_roster_diagnostic.v2"
            if solver_budget_provenance is not None
            else _SCHEMA
        ),
        "status": (
            "ROSTER_PATH_COVERAGE_DIAGNOSTIC_PASS"
            if all_pass
            else "ROSTER_PATH_COVERAGE_DIAGNOSTIC_FAIL"
            if any_fail
            else "ROSTER_PATH_COVERAGE_DIAGNOSTIC_UNRESOLVED"
        ),
        "admission_possible": False,
        "coverage_criterion": _COVERAGE_LIMIT,
        "joint_acceptance_family": family_reference,
        "expected_episode_count": len(expected),
        "observed_episode_count": len(observed),
        "all_expected_episodes_pass": all_pass,
        "strata": [
            {"family_id": family, "condition": condition, **counts}
            for (family, condition), counts in sorted(by_stratum.items())
        ],
        "scenarios": scenario_reports,
        **(
            {
                "solver_budget_provenance": solver_budget_provenance,
                "diffuse_path_diagnostics_are_descriptive_only": True,
            }
            if solver_budget_provenance is not None
            else {}
        ),
        "limitations": [
            "Coverage is assessed independently per scene episode and is not pooled.",
            *(
                [
                    "Per-episode PASS/FAIL also applies the frozen receiver-relevant path-class "
                    "coefficient relative-error tolerance "
                    f"{float(coefficient_relative_tolerance):.17g}."
                ]
                if coefficient_relative_tolerance is not None
                else []
            ),
            "This finite-roster diagnostic does not establish population confidence or physical truth.",
            "Path-equivalence results do not qualify propagation, receiver impact, or production RT.",
        ],
    }
