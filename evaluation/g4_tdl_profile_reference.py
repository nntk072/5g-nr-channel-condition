"""Independent pinned ETSI power/delay references for G4 TDL checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SCALABLE_REFERENCE_PATH = (
    _ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json"
)
_SCALABLE_REFERENCE_SHA256 = "38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd"
_SCALABLE_SOURCE_SHA256 = "b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4"
_FIXED_REFERENCE_PATH = _ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json"
_FIXED_REFERENCE_SHA256 = "fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3"
_FIXED_SOURCE_SHA256 = "b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81"
_FIXED_PROFILE_MODELS = frozenset({"A10", "A30", "B100", "C60", "C300", "D10", "D30"})


def _read_pinned_json(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{label} bytes do not match their frozen digest")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be an object")
    return value


def _component_profile(profile: Any, *, name: str) -> tuple[list[float], list[float]]:
    if not isinstance(profile, dict) or not isinstance(profile.get("components"), list):
        raise ValueError(f"{name} reference must contain component rows")
    components = profile["components"]
    if not components:
        raise ValueError(f"{name} reference cannot be empty")
    powers: list[float] = []
    delays: list[float] = []
    for expected_index, row in enumerate(components, start=1):
        if not isinstance(row, dict) or row.get("component_index") != expected_index:
            raise ValueError(f"{name} component rows must have an ordered contiguous index")
        power_db = row.get("power_db")
        if (
            isinstance(power_db, bool)
            or not isinstance(power_db, (int, float))
            or not math.isfinite(power_db)
        ):
            raise ValueError(f"{name} component power must be finite real dB")
        power = 10.0 ** (float(power_db) / 10.0)
        if not math.isfinite(power) or power <= 0:
            raise ValueError(f"{name} component power is not representable")
        powers.append(power)
        delay = row.get("normalized_delay")
        if (
            isinstance(delay, bool)
            or not isinstance(delay, (int, float))
            or not math.isfinite(delay)
            or delay < 0
        ):
            raise ValueError(f"{name} normalized delay must be finite and nonnegative")
        delays.append(float(delay))
    total = math.fsum(powers)
    if not math.isfinite(total) or total <= 0:
        raise ValueError(f"{name} total reference power must be finite and positive")
    return [power / total for power in powers], delays


def scalable_tdl_pdp_reference(
    model: str, *, delay_spread_s: float
) -> tuple[list[float], list[float]]:
    """Return ETSI-normalized component powers and scaled delays for TDL-A–E."""
    if not isinstance(model, str) or model not in {"A", "B", "C", "D", "E"}:
        raise ValueError("scalable TDL model must be A, B, C, D, or E")
    if (
        isinstance(delay_spread_s, bool)
        or not isinstance(delay_spread_s, (int, float))
        or not math.isfinite(delay_spread_s)
        or delay_spread_s <= 0
    ):
        raise ValueError("delay_spread_s must be finite and positive")
    reference = _read_pinned_json(
        _SCALABLE_REFERENCE_PATH, _SCALABLE_REFERENCE_SHA256, label="scalable TDL reference"
    )
    if (
        reference.get("schema") != "loslab.g4_scalable_tdl_power_delay_reference.v1"
        or reference.get("source_id") != "tr-138901-v19.2.0"
        or reference.get("source_sha256") != _SCALABLE_SOURCE_SHA256
        or reference.get("status") != "PROFILE_TABLE_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("scalable TDL reference has an invalid normative identity")
    matches = [row for row in reference.get("profiles", []) if row.get("profile_model") == model]
    if len(matches) != 1:
        raise ValueError(f"scalable TDL reference must contain exactly one profile {model}")
    profile = matches[0]
    powers, normalized_delays = _component_profile(profile, name=f"TDL-{model}")
    components = profile["components"]
    if components[0]["fading_distribution"] == "LOS":
        # TR 38.901 lists the deterministic LOS term and the first diffuse
        # zero-delay term separately. Sionna's TDL tap axis merges their power
        # into one first tap, so the independent expected PDP must do likewise.
        if (
            len(components) < 2
            or components[1]["fading_distribution"] != "Rayleigh"
            or normalized_delays[0] != normalized_delays[1]
        ):
            raise ValueError(f"TDL-{model} LOS reference has no paired first diffuse component")
        powers = [powers[0] + powers[1], *powers[2:]]
        normalized_delays = [normalized_delays[0], *normalized_delays[2:]]
    return powers, [delay * float(delay_spread_s) for delay in normalized_delays]


def fixed_tdl_pdp_reference(model: str) -> tuple[list[float], list[float]]:
    """Return normalized fixed-TDL tap powers and absolute delays from ETSI rows.

    Fixed TDL-D profiles transcribe the LOS path and diffuse first-tap power as
    separate components at the same published tap index. The Sionna tap axis
    combines those components; this reference merges their linear powers using
    only the independently pinned TS 38.101-4 table fixture.
    """
    if not isinstance(model, str) or model not in _FIXED_PROFILE_MODELS:
        raise ValueError("fixed TDL model must be A10, A30, B100, C60, C300, D10, or D30")
    reference = _read_pinned_json(
        _FIXED_REFERENCE_PATH, _FIXED_REFERENCE_SHA256, label="fixed TDL reference"
    )
    if (
        reference.get("schema") != "loslab.g4_fixed_tdl_tap_profile_reference.v1"
        or reference.get("source_id") != "ts-138101-4-v19.2.2"
        or reference.get("source_sha256") != _FIXED_SOURCE_SHA256
        or reference.get("status") != "PER_TAP_REFERENCE_ONLY_NOT_CONFORMANCE_QUALIFICATION"
    ):
        raise ValueError("fixed TDL reference has an invalid normative identity")
    profiles = reference.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("fixed TDL reference must contain a profile list")
    matches = [
        row for row in profiles if isinstance(row, dict) and row.get("profile_model") == model
    ]
    if len(matches) != 1:
        raise ValueError(f"fixed TDL reference must contain exactly one profile {model}")
    profile = matches[0]
    components = profile.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"fixed TDL {model} reference must contain component rows")

    power_by_tap: dict[int, float] = {}
    delay_by_tap_ns: dict[int, float] = {}
    for expected_index, component in enumerate(components, start=1):
        if not isinstance(component, dict) or component.get("component_index") != expected_index:
            raise ValueError(f"fixed TDL {model} component rows must be ordered and contiguous")
        tap_index = component.get("tap_index")
        delay_ns = component.get("delay_ns")
        power_db = component.get("power_db")
        distribution = component.get("fading_distribution")
        if (
            isinstance(tap_index, bool)
            or not isinstance(tap_index, int)
            or tap_index < 1
            or isinstance(delay_ns, bool)
            or not isinstance(delay_ns, (int, float))
            or not math.isfinite(float(delay_ns))
            or delay_ns < 0
            or isinstance(power_db, bool)
            or not isinstance(power_db, (int, float))
            or not math.isfinite(float(power_db))
            or distribution not in {"Rayleigh", "LOS", "LOS path"}
        ):
            raise ValueError(f"fixed TDL {model} component fields are invalid")
        delay = float(delay_ns)
        if tap_index in delay_by_tap_ns and delay_by_tap_ns[tap_index] != delay:
            raise ValueError(f"fixed TDL {model} components in one tap have different delays")
        delay_by_tap_ns[tap_index] = delay
        component_power = 10.0 ** (float(power_db) / 10.0)
        if not math.isfinite(component_power) or component_power <= 0.0:
            raise ValueError(f"fixed TDL {model} component power is not representable")
        power_by_tap[tap_index] = power_by_tap.get(tap_index, 0.0) + component_power

    tap_count = profile.get("tap_count")
    tap_indices = sorted(power_by_tap)
    if (
        isinstance(tap_count, bool)
        or not isinstance(tap_count, int)
        or tap_indices != list(range(1, tap_count + 1))
    ):
        raise ValueError(f"fixed TDL {model} tap count/index roster is inconsistent")
    total_power = math.fsum(power_by_tap.values())
    if not math.isfinite(total_power) or total_power <= 0.0:
        raise ValueError(f"fixed TDL {model} total reference power is invalid")
    return (
        [power_by_tap[index] / total_power for index in tap_indices],
        [delay_by_tap_ns[index] * 1e-9 for index in tap_indices],
    )
