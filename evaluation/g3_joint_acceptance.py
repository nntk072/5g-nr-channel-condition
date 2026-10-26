"""Shared multiplicity contracts for the G3 numerical evaluators."""

from __future__ import annotations

import hashlib
import json
from typing import Any, cast

FAMILY_SCHEMA = "loslab.g3_joint_acceptance_family.v1"
FAMILY_SCHEMA_V2 = "loslab.g3_joint_acceptance_family.v2"
FAMILY_SCHEMA_V3 = "loslab.g3_joint_acceptance_family.v3"
FAMILY_SCHEMA_V4 = "loslab.g3_joint_acceptance_family.v4"
FAMILYWISE_ALPHA = 0.05
PATH_COVERAGE_LIMIT = 0.95
MATERIAL_CLAIMS = (
    "median_absolute_power_error_db",
    "p95_absolute_power_error_db",
    "maximum_angular_cdf_ks",
    "nominal_95_interval_coverage",
    "median_prediction_interval_width_db",
)
COMPLEX_CSI_CLAIMS = ("p95_e_h_upper_confidence_bound", "cluster_failure_upper_bound")
_PROPAGATION_METRICS = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)
PROPAGATION_CLAIMS = tuple(
    f"{metric}:{bound}"
    for metric in _PROPAGATION_METRICS
    for bound in ("p95_upper_bound", "cluster_failure_upper_bound")
)
RECEIVER_IMPACT_CLAIMS = (
    "candidate_bler_ci_at_comparison_snr",
    "challenger_bler_ci_at_comparison_snr",
    "paired_bler_difference_ci_at_comparison_snr",
    "snr_at_10_percent_bler_shift_ci",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _family_identifier(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(
            "family_id must be a nonempty canonical string without surrounding whitespace"
        )
    return value


def _stratum_keys(value: Any, *, kind: str) -> list[tuple[str, ...]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{kind} expected strata must be a nonempty list")
    result: list[tuple[str, ...]] = []
    width = 3 if kind in {"material", "propagation_metrics"} else 2
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != width:
            raise ValueError(f"each {kind} stratum must contain {width} strings")
        if any(not isinstance(part, str) or not part or part != part.strip() for part in item):
            raise ValueError(
                f"{kind} stratum fields must be nonempty canonical strings "
                "without surrounding whitespace"
            )
        key = tuple(item)
        if kind == "material" and key[1] not in {"V", "H"}:
            raise ValueError("material stratum polarization must be V or H")
        if kind == "complex_csi" and key[1] not in {"LOS", "NLOS"}:
            raise ValueError("complex-CSI stratum condition must be LOS or NLOS")
        if kind == "receiver_impact" and key[1] not in {"LOS", "NLOS"}:
            raise ValueError("receiver-impact stratum condition must be LOS or NLOS")
        if kind == "propagation_metrics":
            if key[1] not in {"LOS", "NLOS"}:
                raise ValueError("propagation-metric stratum condition must be LOS or NLOS")
            if key[2] not in {"total", "specular", "diffuse"}:
                raise ValueError("propagation-metric component must be total, specular, or diffuse")
        if key in result:
            raise ValueError(f"duplicate {kind} expected stratum: {key}")
        result.append(key)
    return sorted(result)


def build_g3_joint_acceptance_family(
    *,
    family_id: str,
    material_expected_strata: list[tuple[str, str, str]],
    complex_csi_expected_strata: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a deterministic, hash-bound family covering both G3 evaluators."""
    family_id = _family_identifier(family_id)
    material = _stratum_keys(material_expected_strata, kind="material")
    complex_csi = _stratum_keys(complex_csi_expected_strata, kind="complex_csi")
    claims = [
        {
            "evaluator": "material_calibration",
            "stratum": list(stratum),
            "claim": claim,
        }
        for stratum in material
        for claim in MATERIAL_CLAIMS
    ] + [
        {"evaluator": "complex_csi", "stratum": list(stratum), "claim": claim}
        for stratum in complex_csi
        for claim in COMPLEX_CSI_CLAIMS
    ]
    claims.sort(
        key=lambda item: (
            item["evaluator"],
            tuple(item["stratum"]),
            item["claim"],
        )
    )
    claim_count = len(claims)
    per_claim_alpha = FAMILYWISE_ALPHA / claim_count
    family: dict[str, Any] = {
        "schema": FAMILY_SCHEMA,
        "family_id": family_id,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "familywise_confidence_level": 1.0 - FAMILYWISE_ALPHA,
        "material_expected_strata": [list(item) for item in material],
        "complex_csi_expected_strata": [list(item) for item in complex_csi],
        "claim_roster": claims,
        "claim_count": claim_count,
        "per_claim_alpha": per_claim_alpha,
        "per_claim_confidence_level": 1.0 - per_claim_alpha,
    }
    family["family_sha256"] = hashlib.sha256(_canonical_bytes(family)).hexdigest()
    return family


def build_g3_joint_acceptance_family_v2(
    *,
    family_id: str,
    material_expected_strata: list[tuple[str, str, str]],
    complex_csi_expected_strata: list[tuple[str, str]],
    propagation_expected_strata: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Build a hash-bound family joining material, CSI, and propagation bounds.

    Receiver-impact intervals are deliberately absent until that evaluator's
    simultaneous interval semantics are added to a later family version.
    """
    family_id = _family_identifier(family_id)
    material = _stratum_keys(material_expected_strata, kind="material")
    complex_csi = _stratum_keys(complex_csi_expected_strata, kind="complex_csi")
    propagation = _stratum_keys(propagation_expected_strata, kind="propagation_metrics")
    claims = (
        [
            {
                "evaluator": "material_calibration",
                "stratum": list(stratum),
                "claim": claim,
            }
            for stratum in material
            for claim in MATERIAL_CLAIMS
        ]
        + [
            {"evaluator": "complex_csi", "stratum": list(stratum), "claim": claim}
            for stratum in complex_csi
            for claim in COMPLEX_CSI_CLAIMS
        ]
        + [
            {"evaluator": "propagation_metrics", "stratum": list(stratum), "claim": claim}
            for stratum in propagation
            for claim in PROPAGATION_CLAIMS
        ]
    )
    claims.sort(
        key=lambda item: (
            item["evaluator"],
            tuple(item["stratum"]),
            item["claim"],
        )
    )
    claim_count = len(claims)
    per_claim_alpha = FAMILYWISE_ALPHA / claim_count
    family: dict[str, Any] = {
        "schema": FAMILY_SCHEMA_V2,
        "family_id": family_id,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "familywise_confidence_level": 1.0 - FAMILYWISE_ALPHA,
        "material_expected_strata": [list(item) for item in material],
        "complex_csi_expected_strata": [list(item) for item in complex_csi],
        "propagation_expected_strata": [list(item) for item in propagation],
        "claim_roster": claims,
        "claim_count": claim_count,
        "per_claim_alpha": per_claim_alpha,
        "per_claim_confidence_level": 1.0 - per_claim_alpha,
    }
    family["family_sha256"] = hashlib.sha256(_canonical_bytes(family)).hexdigest()
    return family


def build_g3_joint_acceptance_family_v3(
    *,
    family_id: str,
    material_expected_strata: list[tuple[str, str, str]],
    complex_csi_expected_strata: list[tuple[str, str]],
    propagation_expected_strata: list[tuple[str, str, str]],
    receiver_impact_expected_strata: list[tuple[str, str]],
) -> dict[str, Any]:
    """Build a family joining material, CSI, propagation, and receiver intervals."""
    base = build_g3_joint_acceptance_family_v2(
        family_id=family_id,
        material_expected_strata=material_expected_strata,
        complex_csi_expected_strata=complex_csi_expected_strata,
        propagation_expected_strata=propagation_expected_strata,
    )
    receiver = _stratum_keys(receiver_impact_expected_strata, kind="receiver_impact")
    claims = list(base["claim_roster"]) + [
        {"evaluator": "receiver_impact", "stratum": list(stratum), "claim": claim}
        for stratum in receiver
        for claim in RECEIVER_IMPACT_CLAIMS
    ]
    claims.sort(
        key=lambda item: (
            item["evaluator"],
            tuple(item["stratum"]),
            item["claim"],
        )
    )
    family = {
        **{
            key: value
            for key, value in base.items()
            if key
            not in {
                "claim_roster",
                "claim_count",
                "per_claim_alpha",
                "per_claim_confidence_level",
                "family_sha256",
                "schema",
            }
        },
        "schema": FAMILY_SCHEMA_V3,
        "receiver_impact_expected_strata": [list(item) for item in receiver],
        "claim_roster": claims,
        "claim_count": len(claims),
    }
    family["per_claim_alpha"] = FAMILYWISE_ALPHA / len(claims)
    family["per_claim_confidence_level"] = 1.0 - family["per_claim_alpha"]
    family["family_sha256"] = hashlib.sha256(_canonical_bytes(family)).hexdigest()
    return family


def build_g3_joint_acceptance_family_v4(
    *,
    family_id: str,
    material_expected_strata: list[tuple[str, str, str]],
    complex_csi_expected_strata: list[tuple[str, str]],
    propagation_expected_strata: list[tuple[str, str, str]],
    receiver_impact_expected_strata: list[tuple[str, str]],
    path_equivalence_expected_scenarios: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Bind run-level path coverage scenarios beside v3 inferential claims.

    The deterministic per-episode path thresholds are in the protocol hash but
    are deliberately excluded from the Bonferroni interval claim count. Every
    path episode must meet both directions; no path/class/episode is treated as
    an independent statistical replicate by this family.
    """
    if not isinstance(path_equivalence_expected_scenarios, (list, tuple)):
        raise ValueError("path-equivalence expected scenarios must be a nonempty list")
    base = build_g3_joint_acceptance_family_v3(
        family_id=family_id,
        material_expected_strata=material_expected_strata,
        complex_csi_expected_strata=complex_csi_expected_strata,
        propagation_expected_strata=propagation_expected_strata,
        receiver_impact_expected_strata=receiver_impact_expected_strata,
    )
    scenarios: list[tuple[str, str, str]] = []
    for item in path_equivalence_expected_scenarios:
        if (
            not isinstance(item, (list, tuple))
            or len(item) != 3
            or any(
                not isinstance(part, str) or not part or part != part.strip()
                for part in item
            )
            or item[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError(
                "path-equivalence scenarios must use canonical strings as "
                "(family_id, LOS/NLOS, episode_id)"
            )
        scenarios.append((item[0], item[1], item[2]))
    if not scenarios:
        raise ValueError("path-equivalence expected scenarios must be nonempty")
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("duplicate path-equivalence expected scenario")
    family = {
        **{key: value for key, value in base.items() if key not in {"schema", "family_sha256"}},
        "schema": FAMILY_SCHEMA_V4,
        "path_equivalence_expected_scenarios": [list(item) for item in sorted(scenarios)],
        "path_equivalence_bidirectional_coverage_min": PATH_COVERAGE_LIMIT,
        "path_equivalence_aggregation": (
            "every preregistered independent scene episode must pass; no pooling"
        ),
        "path_equivalence_admission_possible": False,
    }
    family["family_sha256"] = hashlib.sha256(_canonical_bytes(family)).hexdigest()
    return family


def validate_g3_joint_acceptance_family(value: Any) -> dict[str, Any]:
    """Rebuild and verify a serialized family declaration and its digest."""
    if not isinstance(value, dict):
        raise ValueError("joint_acceptance_family must be an object")
    family_id = _family_identifier(value.get("family_id"))
    material = value.get("material_expected_strata")
    complex_csi = value.get("complex_csi_expected_strata")
    if value.get("schema") == FAMILY_SCHEMA:
        rebuilt = build_g3_joint_acceptance_family(
            family_id=family_id,
            material_expected_strata=cast(
                list[tuple[str, str, str]], _stratum_keys(material, kind="material")
            ),
            complex_csi_expected_strata=cast(
                list[tuple[str, str]], _stratum_keys(complex_csi, kind="complex_csi")
            ),
        )
    elif value.get("schema") == FAMILY_SCHEMA_V2:
        rebuilt = build_g3_joint_acceptance_family_v2(
            family_id=family_id,
            material_expected_strata=cast(
                list[tuple[str, str, str]], _stratum_keys(material, kind="material")
            ),
            complex_csi_expected_strata=cast(
                list[tuple[str, str]], _stratum_keys(complex_csi, kind="complex_csi")
            ),
            propagation_expected_strata=cast(
                list[tuple[str, str, str]],
                _stratum_keys(value.get("propagation_expected_strata"), kind="propagation_metrics"),
            ),
        )
    elif value.get("schema") == FAMILY_SCHEMA_V3:
        rebuilt = build_g3_joint_acceptance_family_v3(
            family_id=family_id,
            material_expected_strata=cast(
                list[tuple[str, str, str]], _stratum_keys(material, kind="material")
            ),
            complex_csi_expected_strata=cast(
                list[tuple[str, str]], _stratum_keys(complex_csi, kind="complex_csi")
            ),
            propagation_expected_strata=cast(
                list[tuple[str, str, str]],
                _stratum_keys(value.get("propagation_expected_strata"), kind="propagation_metrics"),
            ),
            receiver_impact_expected_strata=cast(
                list[tuple[str, str]],
                _stratum_keys(value.get("receiver_impact_expected_strata"), kind="receiver_impact"),
            ),
        )
    elif value.get("schema") == FAMILY_SCHEMA_V4:
        rebuilt = build_g3_joint_acceptance_family_v4(
            family_id=family_id,
            material_expected_strata=cast(
                list[tuple[str, str, str]], _stratum_keys(material, kind="material")
            ),
            complex_csi_expected_strata=cast(
                list[tuple[str, str]], _stratum_keys(complex_csi, kind="complex_csi")
            ),
            propagation_expected_strata=cast(
                list[tuple[str, str, str]],
                _stratum_keys(value.get("propagation_expected_strata"), kind="propagation_metrics"),
            ),
            receiver_impact_expected_strata=cast(
                list[tuple[str, str]],
                _stratum_keys(value.get("receiver_impact_expected_strata"), kind="receiver_impact"),
            ),
            path_equivalence_expected_scenarios=cast(
                list[tuple[str, str, str]], value.get("path_equivalence_expected_scenarios")
            ),
        )
    else:
        raise ValueError("joint_acceptance_family schema is unsupported")
    # Python's structural equality treats bools as integers (for example,
    # ``0 == False``). Compare canonical JSON bytes so a type substitution
    # cannot preserve equality while evading the hash-bound contract.
    if _canonical_bytes(value) != _canonical_bytes(rebuilt):
        raise ValueError("joint_acceptance_family fields or canonical SHA-256 are inconsistent")
    return rebuilt


def family_alpha_for_claim(
    value: Any, *, evaluator: str, expected_strata: Any
) -> tuple[float, int]:
    """Return the shared per-claim alpha after checking evaluator roster coverage."""
    family = validate_g3_joint_acceptance_family(value)
    if evaluator == "material_calibration":
        observed = _stratum_keys(expected_strata, kind="material")
        declared = _stratum_keys(family["material_expected_strata"], kind="material")
    elif evaluator == "complex_csi":
        observed = _stratum_keys(expected_strata, kind="complex_csi")
        declared = _stratum_keys(family["complex_csi_expected_strata"], kind="complex_csi")
    elif evaluator == "propagation_metrics" and family["schema"] == FAMILY_SCHEMA_V2:
        observed = _stratum_keys(expected_strata, kind="propagation_metrics")
        declared = _stratum_keys(family["propagation_expected_strata"], kind="propagation_metrics")
    elif evaluator == "propagation_metrics" and family["schema"] in {
        FAMILY_SCHEMA_V3,
        FAMILY_SCHEMA_V4,
    }:
        observed = _stratum_keys(expected_strata, kind="propagation_metrics")
        declared = _stratum_keys(family["propagation_expected_strata"], kind="propagation_metrics")
    elif evaluator == "receiver_impact" and family["schema"] in {
        FAMILY_SCHEMA_V3,
        FAMILY_SCHEMA_V4,
    }:
        observed = _stratum_keys(expected_strata, kind="receiver_impact")
        declared = _stratum_keys(family["receiver_impact_expected_strata"], kind="receiver_impact")
    else:
        raise ValueError(f"unsupported G3 evaluator: {evaluator!r}")
    if observed != declared:
        raise ValueError(f"{evaluator} expected strata differ from the shared G3 family roster")
    return float(family["per_claim_alpha"]), int(family["claim_count"])
