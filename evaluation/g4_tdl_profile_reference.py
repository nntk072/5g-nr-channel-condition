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
