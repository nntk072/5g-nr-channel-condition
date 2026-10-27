"""SRS positioning measurement primitives and strict report contracts.

The UL SRS-RSRP power averaging semantics follow ETSI TS 138 215 V19.2.0,
clause 5.2.5. Callers supply received per-RE power in watts at the declared
measurement reference point; this module does not calibrate RF power.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECEIVER_REFERENCE_POINTS = {
    "terrestrial_type_1_c": "rx_antenna_connector",
    "terrestrial_type_1_o": "combined_antenna_elements_per_receiver_branch",
    "terrestrial_type_2_o": "combined_antenna_elements_per_receiver_branch",
    "terrestrial_type_1_h": "rx_transceiver_array_boundary_connector",
    "satellite_type_1_o": "combined_antenna_elements_per_receiver_branch",
    "satellite_type_1_h": "rx_transceiver_array_boundary_connector",
}
_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "evaluation/schema/nr-ul-srs-rsrp-report.v1.schema.json"
)


def measure_ul_srs_rsrp(
    received_re_power_w: np.ndarray,
    configured_re_mask: np.ndarray,
    *,
    occasion_ids: Sequence[str],
    receiver_branch_ids: Sequence[str],
    receiver_category: str,
    measurement_frequency_bandwidth_hz: float,
    power_calibration_sha256: str,
) -> dict[str, Any]:
    """Build a linear-average UL SRS-RSRP report from calibrated RE powers.

    ``received_re_power_w`` has axes ``[measurement occasion, receiver branch,
    resource element]``. ``configured_re_mask`` has axes ``[occasion, RE]`` and
    selects only configured SRS REs in the declared measurement bandwidth and
    time occasions. Each receiver branch is averaged independently. The
    reported diversity value uses selection combining (the largest branch
    average), which satisfies the TS 38.215 lower-bound requirement that a
    diversity report not be below any individual branch. It is not a calibrated
    front-end or propagation model.
    """

    powers = np.asarray(received_re_power_w)
    mask = np.asarray(configured_re_mask)
    if powers.ndim != 3 or any(size <= 0 for size in powers.shape):
        raise ValueError("received_re_power_w must be nonempty [occasion, branch, RE]")
    if powers.dtype.kind not in "fiu" or not np.all(np.isfinite(powers)) or np.any(powers < 0):
        raise ValueError("received_re_power_w must contain finite nonnegative real watts")
    if mask.dtype.kind != "b" or mask.shape != (powers.shape[0], powers.shape[2]):
        raise ValueError("configured_re_mask must be Boolean [occasion, RE]")
    if not np.any(mask):
        raise ValueError("configured_re_mask must select at least one configured SRS RE")
    if np.any(np.sum(mask, axis=1) == 0):
        raise ValueError("every configured measurement occasion must contain an SRS RE")

    def validate_ids(values: Sequence[str], expected_count: int, label: str) -> list[str]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of IDs")
        result = list(values)
        if (
            len(result) != expected_count
            or any(
                not isinstance(value, str) or not value or value != value.strip()
                for value in result
            )
            or len(set(result)) != len(result)
        ):
            raise ValueError(f"{label} must contain {expected_count} unique canonical IDs")
        return result

    occasions = validate_ids(occasion_ids, powers.shape[0], "occasion_ids")
    branches = validate_ids(receiver_branch_ids, powers.shape[1], "receiver_branch_ids")
    if receiver_category not in _RECEIVER_REFERENCE_POINTS:
        raise ValueError(
            "receiver_category must be one of the TS 38.215 terrestrial/satellite receiver types"
        )
    if (
        isinstance(measurement_frequency_bandwidth_hz, (bool, np.bool_))
        or not isinstance(measurement_frequency_bandwidth_hz, Real)
        or not math.isfinite(float(measurement_frequency_bandwidth_hz))
        or measurement_frequency_bandwidth_hz <= 0
    ):
        raise ValueError("measurement_frequency_bandwidth_hz must be finite and positive")
    if not isinstance(power_calibration_sha256, str) or not _SHA256_RE.fullmatch(
        power_calibration_sha256
    ):
        raise ValueError("power_calibration_sha256 must be a lowercase SHA-256 digest")

    # Compute each branch over the same configured occasion/RE roster.
    branch_means = [
        float(np.mean(powers[:, branch_index, :][mask])) for branch_index in range(powers.shape[1])
    ]
    combined_w = max(branch_means)
    report = {
        "schema": "loslab.nr_ul_srs_rsrp_report.v1",
        "status": "MEASURED" if combined_w > 0 else "ZERO_POWER_DIAGNOSTIC",
        "measurement": "UL_SRS_RSRP",
        "standard": "3GPP_TS_38.215_V19.2.0_clause_5.2.5",
        "receiver_category": receiver_category,
        "measurement_reference_point": _RECEIVER_REFERENCE_POINTS[receiver_category],
        "measurement_frequency_bandwidth_hz": float(measurement_frequency_bandwidth_hz),
        "power_unit": "W",
        "power_averaging": "linear_average_of_configured_srs_re_power_contributions",
        "configured_measurement_occasion_ids": occasions,
        "configured_re_count_per_occasion": [int(np.sum(row)) for row in mask],
        "configured_re_count_total": int(np.sum(mask)),
        "power_calibration_sha256": power_calibration_sha256,
        "receiver_branch_measurements": [
            {
                "receiver_branch_id": branch_id,
                "ul_srs_rsrp_w": branch_mean,
                "ul_srs_rsrp_dbm": (
                    10.0 * math.log10(branch_mean / 1e-3) if branch_mean > 0 else None
                ),
            }
            for branch_id, branch_mean in zip(branches, branch_means, strict=True)
        ],
        "diversity_combining": "selection_combining_max_receiver_branch_power",
        "diversity_ul_srs_rsrp_w": combined_w,
        "diversity_ul_srs_rsrp_dbm": (
            10.0 * math.log10(combined_w / 1e-3) if combined_w > 0 else None
        ),
        "receiver_diversity_lower_bound_satisfied": combined_w >= max(branch_means),
        "limitations": [
            "Input watts and the power calibration digest are caller supplied and not independently verified.",
            "Selection combining is a declared bounded combiner; no RF receiver or propagation qualification is implied.",
            "The report covers only the supplied configured SRS resource elements and occasions.",
        ],
    }
    _validate_report_schema(report)
    return report


def _validate_report_schema(report: dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ValueError(
            "UL SRS-RSRP report violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
