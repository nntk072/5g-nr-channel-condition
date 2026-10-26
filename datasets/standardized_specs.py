"""Small, version-pinned admission tables for standardized channel profiles."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import TypeAlias

SPEED_OF_LIGHT_MPS = 299_792_458.0


@dataclass(frozen=True, slots=True)
class StandardizedProfileApplicability:
    """Code-supported scope for one standardized channel profile.

    This ledger describes what the adapters and dataset admission code permit;
    it is not a claim that the profile has passed statistical conformance.
    Frequency intervals are inclusive and expressed in hertz.
    """

    family: str
    profile_model: str
    standard: str
    supported_versions: tuple[str, ...]
    delay_profile_semantics: str
    carrier_frequency_ranges_hz: tuple[tuple[float, float], ...]
    maximum_sampled_bandwidth_hz: float
    profile_los_component: bool
    fixed_profile: bool = False
    minimum_channel_bandwidth_exclusive_hz: float | None = None


_TR38901_VERSIONS = ("16.1", "19.2")
_FR1_HZ = (0.5e9, 7.125e9)
_FR2_HZ = (24.25e9, 71e9)
_GENERAL_CARRIER_HZ = (0.5e9, 100e9)
_MAX_SAMPLED_BANDWIDTH_HZ = 2e9
_FIXED_TDL_FR_RANGES: Mapping[str, tuple[tuple[float, float], ...]] = MappingProxyType(
    {
        "A10": (_FR2_HZ,),
        "A30": (_FR1_HZ, _FR2_HZ),
        "B100": (_FR1_HZ,),
        "C60": (_FR2_HZ,),
        "C300": (_FR1_HZ,),
        "D10": (_FR2_HZ,),
        "D30": (_FR1_HZ, _FR2_HZ),
    }
)


def _build_standardized_profile_applicability() -> Mapping[
    tuple[str, str], StandardizedProfileApplicability
]:
    entries: dict[tuple[str, str], StandardizedProfileApplicability] = {}
    for family in ("tdl", "cdl"):
        for model in "ABCDE":
            entries[(family, model)] = StandardizedProfileApplicability(
                family=family,
                profile_model=model,
                standard="3GPP TR 38.901",
                supported_versions=_TR38901_VERSIONS,
                delay_profile_semantics="scalable_tr_38_901_profile",
                carrier_frequency_ranges_hz=(_GENERAL_CARRIER_HZ,),
                maximum_sampled_bandwidth_hz=_MAX_SAMPLED_BANDWIDTH_HZ,
                profile_los_component=model in "DE",
            )
    for model, ranges in _FIXED_TDL_FR_RANGES.items():
        entries[("tdl", model)] = StandardizedProfileApplicability(
            family="tdl",
            profile_model=model,
            standard="3GPP TS 38.101-4",
            supported_versions=("19.2.2",),
            delay_profile_semantics="fixed_ts_38_101_4_absolute_tap_delays",
            carrier_frequency_ranges_hz=ranges,
            maximum_sampled_bandwidth_hz=_MAX_SAMPLED_BANDWIDTH_HZ,
            profile_los_component=model.startswith("D"),
            fixed_profile=True,
            minimum_channel_bandwidth_exclusive_hz=(200e6 if model in {"A10", "D10"} else None),
        )
    return MappingProxyType(entries)


STANDARDIZED_PROFILE_APPLICABILITY = _build_standardized_profile_applicability()


def standardized_profile_applicability(
    family: str, profile_model: str
) -> StandardizedProfileApplicability:
    """Return the canonical code-scope row or reject an unsupported profile."""
    try:
        return STANDARDIZED_PROFILE_APPLICABILITY[(family.lower(), profile_model)]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"unsupported standardized profile: {family}/{profile_model}") from exc


def standardized_profile_applicability_report() -> dict[str, object]:
    """Return a deterministic, hash-bound view of the supported profile scope."""
    rows = [
        asdict(STANDARDIZED_PROFILE_APPLICABILITY[key])
        for key in sorted(STANDARDIZED_PROFILE_APPLICABILITY)
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return {
        "schema": "umcd.standardized_profile_applicability.v1",
        "status": "CODE_SCOPE_ONLY_NOT_CONFORMANCE_QUALIFICATION",
        "rows": rows,
        "rows_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def standardized_generator_capability_report() -> dict[str, object]:
    """Return the explicit adapter topology/antenna and unsupported-family map.

    These rows describe implemented generator contracts. They do not assert
    statistical conformance, standards qualification, or multi-link support.
    """
    families: list[dict[str, object]] = [
        {
            "family": "tdl",
            "supported_models": sorted(
                model for (family, model) in STANDARDIZED_PROFILE_APPLICABILITY if family == "tdl"
            ),
            "supported_profile_table_versions": _standardized_family_versions("tdl"),
            "link_topology": {"tx_node_count": 1, "rx_node_count": 1, "mimo": True},
            "antenna_features": [
                "positive_tx_and_rx_antenna_counts",
                "optional_rx_correlation_matrix",
                "los_angle_of_arrival",
            ],
            "unsupported_antenna_features": [
                "explicit_array_geometry",
                "antenna_element_patterns",
                "transmit_correlation_matrix",
            ],
            "unsupported_topologies": [
                "multiple_tx_or_rx_nodes_per_record",
                "multi_cell_interference_composition",
            ],
            "channel_views": [
                "path_coefficients",
                "path_delays",
                "frequency_response",
                "time_samples",
            ],
        },
        {
            "family": "cdl",
            "supported_models": sorted(
                model for (family, model) in STANDARDIZED_PROFILE_APPLICABILITY if family == "cdl"
            ),
            "supported_profile_table_versions": _standardized_family_versions("cdl"),
            "link_topology": {"tx_node_count": 1, "rx_node_count": 1, "mimo": True},
            "antenna_features": [
                "panel_array_geometry",
                "single_or_dual_polarization",
                "omni_or_38_901_or_38_901_handheld_element_patterns",
                "endpoint_orientation",
                "receiver_velocity",
            ],
            "unsupported_antenna_features": [
                "transmit_correlation_matrix",
            ],
            "unsupported_topologies": [
                "multiple_tx_or_rx_nodes_per_record",
                "multi_cell_interference_composition",
            ],
            "channel_views": [
                "cluster_coefficients",
                "cluster_delays",
                "frequency_response",
                "time_samples",
            ],
        },
    ]
    scenario_gbsm = {
        "family": "3gpp_scenario_gbsm",
        "supported_models": ["InF", "InH", "RMa", "UMa", "UMi"],
        "supported_versions": ["16.1", "19.2"],
        "link_topology": {
            "tx_node_count": 1,
            "rx_node_count": 1,
            "batch_semantics": "explicit_one_bs_one_ut_drops",
        },
        "status": "CODE_SCOPE_ONLY_NOT_CONFORMANCE_QUALIFICATION",
        "antenna_features": [
            "panel_array_geometry",
            "single_or_dual_polarization",
            "omni_or_38_901_or_38_901_handheld_element_patterns",
            "explicit_endpoint_orientation",
            "explicit_ut_velocity",
        ],
        "limitations": [
            "explicit_one_bs_one_ut_topology_per_batch_row",
            "no_multi_cell_interference_composition",
            "no_spatial_consistency_or_blockage_controls",
            "no_geometric_los_truth",
            "no_statistical_conformance_claim",
        ],
    }
    payload = {"families": families, "scenario_gbsm": scenario_gbsm}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "schema": "umcd.standardized_generator_capabilities.v2",
        "status": "CODE_SCOPE_ONLY_NOT_CONFORMANCE_QUALIFICATION",
        **payload,
        "capabilities_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _standardized_family_versions(family: str) -> dict[str, list[str]]:
    entries = [
        row for (row_family, _), row in STANDARDIZED_PROFILE_APPLICABILITY.items()
        if row_family == family
    ]
    return {
        "scalable": sorted(
            {version for row in entries if not row.fixed_profile for version in row.supported_versions}
        ),
        "fixed": sorted(
            {version for row in entries if row.fixed_profile for version in row.supported_versions}
        ),
    }

_TDL_RESOLVED_PROFILE_FIELDS = (
    "profile_proxy",
    "spec_version",
    "fixed_profile_spec_version",
    "delay_profile_semantics",
    "profile_rms_delay_spread_s",
    "profile_los_component",
    "num_profile_taps",
    "nominal_tap_delays_s",
    "mean_tap_powers",
    "k_factor_linear",
)


def tdl_resolved_profile_parameter_sha256(metadata: Mapping[str, object]) -> str:
    """Hash the exact resolved TDL profile parameters carried by a record.

    This binds the emitted tap table and, when present, its exact packaged JSON
    table identity to metadata. The table hash is not an independently sourced
    standards reference or a hash of Sionna's executable source distribution.
    """
    try:
        payload = {name: metadata[name] for name in _TDL_RESOLVED_PROFILE_FIELDS}
        source_id = metadata.get("profile_parameter_table_source")
        source_sha256 = metadata.get("profile_parameter_table_sha256")
        if (source_id is None) != (source_sha256 is None):
            raise ValueError("TDL parameter table source identity and digest must appear together")
        if source_id is not None:
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("TDL parameter table source identity must be nonempty")
            if (
                not isinstance(source_sha256, str)
                or len(source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in source_sha256)
            ):
                raise ValueError("TDL parameter table source digest must be lowercase SHA-256")
            payload["profile_parameter_table_source"] = source_id
            payload["profile_parameter_table_sha256"] = source_sha256
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("TDL resolved profile parameters are incomplete or non-finite") from exc
    return hashlib.sha256(encoded).hexdigest()


def cdl_parameter_table_binding_sha256(metadata: Mapping[str, object]) -> str:
    """Bind a CDL table hash to its profile, release, and package-resource label."""
    fields = (
        "profile_model",
        "spec_version",
        "parameter_table",
        "parameter_table_source",
        "parameter_table_sha256",
    )
    try:
        payload = {name: metadata[name] for name in fields}
        source_digest = payload["parameter_table_sha256"]
        if (
            not isinstance(source_digest, str)
            or len(source_digest) != 64
            or any(character not in "0123456789abcdef" for character in source_digest)
        ):
            raise ValueError("CDL parameter table source digest must be lowercase SHA-256")
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("CDL parameter table binding is incomplete or invalid") from exc
    return hashlib.sha256(encoded).hexdigest()

# TS 38.101-4 V19.2.2 Annex B.2.2, Tables B.2.2-1 and B.2.2-2.
# Values are (fixed TDL profile, frequency range, maximum Doppler in Hz).
FixedTDLDopplerCase: TypeAlias = tuple[str, str, float]
FIXED_TDL_DOPPLER_CASES: Mapping[str, FixedTDLDopplerCase] = MappingProxyType(
    {
        "TDLA30-5": ("A30", "FR1", 5.0),
        "TDLA30-10": ("A30", "FR1", 10.0),
        "TDLA30-20": ("A30", "FR1", 20.0),
        "TDLA30-180": ("A30", "FR1", 180.0),
        "TDLA30-195": ("A30", "FR1", 195.0),
        "TDLA30-1400": ("A30", "FR1", 1_400.0),
        "TDLA30-2700": ("A30", "FR1", 2_700.0),
        "TDLB100-400": ("B100", "FR1", 400.0),
        "TDLC300-100": ("C300", "FR1", 100.0),
        "TDLC300-600": ("C300", "FR1", 600.0),
        "TDLC300-1200": ("C300", "FR1", 1_200.0),
        "TDLD30-5": ("D30", "FR1", 5.0),
        "TDLA30-35": ("A30", "FR2", 35.0),
        "TDLA30-75": ("A30", "FR2", 75.0),
        "TDLA30-300": ("A30", "FR2", 300.0),
        "TDLC60-300": ("C60", "FR2", 300.0),
        "TDLD30-75": ("D30", "FR2", 75.0),
    }
)


def carrier_frequency_range(carrier_frequency_hz: float) -> str | None:
    """Return the NR frequency range for the supported fixed-profile carrier bands."""
    if 0.5e9 <= carrier_frequency_hz <= 7.125e9:
        return "FR1"
    if 24.25e9 <= carrier_frequency_hz <= 71e9:
        return "FR2"
    return None


def _finite_metadata_number(value: object, name: str, prefix: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(prefix + f"conformance case requires finite {name}")
    return float(value)


def validate_fixed_tdl_conformance_metadata(
    profile_model: str, metadata: Mapping[str, object]
) -> None:
    """Validate an optional named Annex B.2.2 case carried by a TDL record."""
    case_id = metadata.get("fixed_profile_conformance_case")
    if case_id is None:
        return
    prefix = "standardized channel admission rejected: "
    if not isinstance(case_id, str) or case_id not in FIXED_TDL_DOPPLER_CASES:
        raise ValueError(prefix + "unknown TS 38.101-4 fixed-profile Doppler case")
    expected_model, expected_range, expected_doppler_hz = FIXED_TDL_DOPPLER_CASES[case_id]
    if profile_model != expected_model:
        raise ValueError(prefix + f"conformance case {case_id} requires TDL-{expected_model}")

    carrier_frequency_hz = _finite_metadata_number(
        metadata.get("carrier_frequency_hz"), "carrier frequency", prefix
    )
    if carrier_frequency_range(carrier_frequency_hz) != expected_range:
        raise ValueError(prefix + f"conformance case {case_id} requires {expected_range}")

    min_speed_mps = _finite_metadata_number(metadata.get("min_speed_mps"), "min_speed_mps", prefix)
    max_speed_mps = _finite_metadata_number(metadata.get("max_speed_mps"), "max_speed_mps", prefix)
    maximum_doppler_hz = _finite_metadata_number(
        metadata.get("maximum_doppler_hz"), "maximum_doppler_hz", prefix
    )
    if min_speed_mps < 0 or min_speed_mps > max_speed_mps:
        raise ValueError(prefix + "conformance case has an invalid speed interval")
    derived_doppler_hz = max_speed_mps * carrier_frequency_hz / SPEED_OF_LIGHT_MPS
    if not math.isclose(
        maximum_doppler_hz, expected_doppler_hz, rel_tol=1e-12, abs_tol=1e-9
    ) or not math.isclose(derived_doppler_hz, expected_doppler_hz, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError(
            prefix + f"conformance case {case_id} requires max Doppler {expected_doppler_hz:g} Hz"
        )
