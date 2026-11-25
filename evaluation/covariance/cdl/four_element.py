"""Independent ETSI-derived covariance reference and claim calculations."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = _ROOT / "config/reference/CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
REFERENCE_SHA256 = "24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090"
ETSI_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
RAY_ANGLE_OFFSETS = (
    0.0447,
    -0.0447,
    0.1413,
    -0.1413,
    0.2492,
    -0.2492,
    0.3715,
    -0.3715,
    0.5129,
    -0.5129,
    0.6797,
    -0.6797,
    0.8844,
    -0.8844,
    1.1481,
    -1.1481,
    1.5195,
    -1.5195,
    2.1551,
    -2.1551,
)
_CLAIM_NAMES = tuple(
    [f"diag_{index}" for index in range(4)]
    + [
        name
        for row in range(4)
        for column in range(row + 1, 4)
        for name in (f"re_{row}{column}", f"im_{row}{column}")
    ]
)


def cdl_spatial_covariance_reference(model: str) -> tuple[np.ndarray, str]:
    """Evaluate a 4x4 spatial covariance from pinned ETSI profile rows."""
    if model not in set("ABCDE"):
        raise ValueError(f"unsupported CDL profile: {model}")
    reference_bytes = REFERENCE_PATH.read_bytes()
    reference_digest = hashlib.sha256(reference_bytes).hexdigest()
    if reference_digest != REFERENCE_SHA256:
        raise ValueError("CDL reference bytes do not match their frozen digest")
    document = json.loads(reference_bytes)
    if document.get("source_sha256") != ETSI_SOURCE_SHA256:
        raise ValueError("CDL reference source digest differs from the frozen ETSI source")
    profile = next(item for item in document["profiles"] if item["profile_model"] == model)
    components = profile["components"]
    specular_components = [item for item in components if item["component"] == "specular"]
    diffuse_components = [item for item in components if item["component"] == "diffuse"]
    if profile["los"]:
        if len(specular_components) != 1 or components[0] is not specular_components[0]:
            raise ValueError("LOS profile must list its single specular component first")
        if components[1:] != diffuse_components:
            raise ValueError("LOS profile diffuse component roster is inconsistent")
    elif specular_components or components != diffuse_components:
        raise ValueError("NLOS profile must contain only diffuse components")
    if len(diffuse_components) != profile["cluster_count"]:
        raise ValueError("CDL reference diffuse-cluster count differs from its declaration")

    component_powers = np.asarray(
        [10.0 ** (float(item["power_db"]) / 10.0) for item in components], dtype=np.float64
    )
    component_powers /= component_powers.sum()
    offsets = np.asarray(RAY_ANGLE_OFFSETS, dtype=np.float64)
    aoa = np.deg2rad(
        np.asarray([item["angles_deg"]["aoa"] for item in diffuse_components])[:, None]
        + float(profile["cluster_spreads_deg"]["asa"]) * offsets[None, :]
    )
    zoa = np.deg2rad(
        np.asarray([item["angles_deg"]["zoa"] for item in diffuse_components])[:, None]
        + float(profile["cluster_spreads_deg"]["zsa"]) * offsets[None, :]
    )
    direction_projection = np.sin(zoa)[:, :, None] * np.sin(aoa)[:, None, :]
    element_indices = np.arange(4, dtype=np.float64)
    separation = element_indices[None, :] - element_indices[:, None]
    phase = np.exp(-1j * math.pi * direction_projection[..., None, None] * separation)
    cluster_covariance = phase.mean(axis=(1, 2))
    diffuse_powers = component_powers[-len(diffuse_components) :]
    covariance = np.sum(diffuse_powers[:, None, None] * cluster_covariance, axis=0)
    if profile["los"]:
        los = specular_components[0]
        los_projection = math.sin(math.radians(float(los["angles_deg"]["zoa"]))) * math.sin(
            math.radians(float(los["angles_deg"]["aoa"]))
        )
        covariance += component_powers[0] * np.exp(-1j * math.pi * los_projection * separation)
    result = covariance.astype(np.complex128, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("CDL covariance reference must be finite")
    return result, reference_digest


def cdl_spatial_covariance_claims(
    samples: Any,
    reference: Any,
    *,
    familywise_alpha: float = 0.01,
    maximum_half_width: float = 0.05,
) -> list[dict[str, Any]]:
    """Compute the frozen 16 real-valued covariance claims from draw samples."""
    values = np.asarray(samples)
    expected = np.asarray(reference)
    if values.dtype.kind != "c" or values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("samples must be a complex [draw, 4] array")
    if values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("samples must contain at least two finite draws")
    if expected.dtype.kind != "c" or expected.shape != (4, 4) or not np.isfinite(expected).all():
        raise ValueError("reference must be a finite complex 4x4 matrix")
    if (
        isinstance(familywise_alpha, bool)
        or not isinstance(familywise_alpha, (int, float))
        or not math.isfinite(float(familywise_alpha))
        or not 0.0 < familywise_alpha < 1.0
    ):
        raise ValueError("familywise_alpha must be a finite value in (0, 1)")
    if (
        isinstance(maximum_half_width, bool)
        or not isinstance(maximum_half_width, (int, float))
        or not math.isfinite(float(maximum_half_width))
        or maximum_half_width <= 0.0
    ):
        raise ValueError("maximum_half_width must be finite and positive")

    outer = values[:, :, None] * values[:, None, :].conj()
    z_value = NormalDist().inv_cdf(1.0 - float(familywise_alpha) / (2.0 * 16.0))
    scalar_claims: list[tuple[str, float, np.ndarray]] = []
    for row in range(4):
        scalar_claims.append(
            (f"diag_{row}", float(expected[row, row].real), outer[:, row, row].real)
        )
    for row in range(4):
        for column in range(row + 1, 4):
            scalar_claims.append(
                (f"re_{row}{column}", float(expected[row, column].real), outer[:, row, column].real)
            )
            scalar_claims.append(
                (f"im_{row}{column}", float(expected[row, column].imag), outer[:, row, column].imag)
            )
    if len(scalar_claims) != 16:
        raise AssertionError("the frozen covariance family must contain 16 coordinates")

    claims: list[dict[str, Any]] = []
    for name, expected_value, observations in scalar_claims:
        estimate = float(observations.mean())
        standard_error = float(observations.std(ddof=1) / math.sqrt(values.shape[0]))
        half_width = z_value * standard_error
        lower, upper = estimate - half_width, estimate + half_width
        claims.append(
            {
                "name": name,
                "estimate": estimate,
                "reference": expected_value,
                "standard_error": standard_error,
                "interval": [lower, upper],
                "half_width": half_width,
                "absolute_error": abs(estimate - expected_value),
                "contained": lower <= expected_value <= upper,
                "precision_pass": half_width <= maximum_half_width,
                "passed": lower <= expected_value <= upper and half_width <= maximum_half_width,
            }
        )
    return claims


def audit_cdl_spatial_covariance_claims(
    claims: Any,
    *,
    familywise_alpha: float = 0.01,
    maximum_half_width: float = 0.05,
) -> dict[str, Any]:
    """Recompute claim arithmetic and the frozen family decision."""
    if not isinstance(claims, list) or len(claims) != 16:
        raise ValueError("covariance report must contain exactly 16 claims")
    if [claim.get("name") for claim in claims if isinstance(claim, dict)] != list(_CLAIM_NAMES):
        raise ValueError("covariance claim names or order differ from the frozen family")
    if (
        isinstance(familywise_alpha, bool)
        or not isinstance(familywise_alpha, Real)
        or not math.isfinite(float(familywise_alpha))
        or not 0.0 < familywise_alpha < 1.0
    ):
        raise ValueError("familywise_alpha must be a finite value in (0, 1)")
    if (
        isinstance(maximum_half_width, bool)
        or not isinstance(maximum_half_width, Real)
        or not math.isfinite(float(maximum_half_width))
        or maximum_half_width <= 0.0
    ):
        raise ValueError("maximum_half_width must be finite and positive")
    z_value = NormalDist().inv_cdf(1.0 - familywise_alpha / 32.0)
    failed = 0
    max_error = 0.0
    max_width = 0.0
    for claim in claims:
        required_fields = {
            "name",
            "estimate",
            "reference",
            "standard_error",
            "interval",
            "half_width",
            "absolute_error",
            "contained",
            "precision_pass",
            "passed",
        }
        if not isinstance(claim, dict) or set(claim) != required_fields:
            raise ValueError("covariance claim fields differ from the frozen report contract")
        for field in ("estimate", "reference", "standard_error", "half_width", "absolute_error"):
            value = claim[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"covariance claim {field} must be finite and real")
        if (
            claim["standard_error"] < 0.0
            or claim["half_width"] < 0.0
            or claim["absolute_error"] < 0.0
        ):
            raise ValueError("covariance claim uncertainty and absolute error must be nonnegative")
        if any(
            type(claim[field]) is not bool for field in ("contained", "precision_pass", "passed")
        ):
            raise ValueError("covariance claim decisions must be Boolean")
        estimate = claim["estimate"]
        reference = claim["reference"]
        standard_error = claim["standard_error"]
        half_width = claim["half_width"]
        interval = claim["interval"]
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                isinstance(bound, bool)
                or not isinstance(bound, Real)
                or not math.isfinite(float(bound))
                for bound in interval
            )
            or interval != [estimate - half_width, estimate + half_width]
            or not math.isclose(half_width, z_value * standard_error, rel_tol=1e-12, abs_tol=1e-15)
            or claim["absolute_error"] != abs(estimate - reference)
        ):
            raise ValueError("covariance claim interval or error arithmetic is inconsistent")
        contained = interval[0] <= reference <= interval[1]
        precision_pass = half_width <= maximum_half_width
        passed = contained and precision_pass
        if (
            claim["contained"] is not contained
            or claim["precision_pass"] is not precision_pass
            or claim["passed"] is not passed
        ):
            raise ValueError("covariance claim decision differs from its frozen criteria")
        failed += not passed
        max_error = max(max_error, abs(estimate - reference))
        max_width = max(max_width, half_width)
    return {
        "claims_passed": len(claims) - failed,
        "claims_failed": failed,
        "maximum_absolute_error": max_error,
        "maximum_half_width": max_width,
        "status": "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE" if failed == 0 else "FAIL",
    }
