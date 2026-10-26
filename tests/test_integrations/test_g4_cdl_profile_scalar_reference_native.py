"""CPU metadata regression against compact, independently derived ETSI CDL scalars."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sionna.phy")

from integrations.sionna.cdl import _profile_metadata  # noqa: E402


_REFERENCE = Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json"
_TAP_ANGLE_REFERENCE = Path(__file__).resolve().parents[2] / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"


@pytest.mark.native_sionna
@pytest.mark.parametrize("profile", list("ABCDE"))
def test_cdl_profile_metadata_matches_independent_etsi_scalar_reference(profile: str) -> None:
    from sionna.phy.channel.tr38901 import models

    reference = json.loads(_REFERENCE.read_text(encoding="utf-8"))
    expected = next(row for row in reference["profiles"] if row["profile_model"] == profile)
    tolerance = reference["derivation"]["rounding_tolerances"]
    metadata = _profile_metadata(profile, "19.2", 1.0, models, torch, "cpu", torch.float64)
    delays = metadata["nominal_cluster_delays_s"]
    powers = metadata["normalized_cluster_powers"]
    mean_delay = sum(power * delay for power, delay in zip(powers, delays, strict=True))
    rms_delay = math.sqrt(sum(
        power * (delay - mean_delay) ** 2 for power, delay in zip(powers, delays, strict=True)
    ))

    assert metadata["profile_los_component"] is expected["los"]
    assert metadata["num_profile_clusters"] == expected["output_delay_cluster_count"]
    assert len(delays) == len(powers) == expected["output_delay_cluster_count"]
    assert expected["source_component_count"] == expected["output_delay_cluster_count"] + int(
        expected["los"]
    )
    assert metadata["parameter_table_source"] == f"TR 38.901:19.2:CDL-{profile}.json"
    assert max(delays) == pytest.approx(expected["max_normalized_delay"], abs=tolerance["max_normalized_delay_abs"])
    assert rms_delay == pytest.approx(expected["rms_normalized_delay"], abs=tolerance["rms_normalized_delay_abs"])
    for name, value in expected["angle_spreads_deg"].items():
        assert metadata["cluster_angle_spreads_deg"][name] == pytest.approx(value, abs=tolerance["angle_spread_deg_abs"])
    assert metadata["cross_polarization_power_ratio_db"] == pytest.approx(expected["xpr_db"], abs=tolerance["xpr_db_abs"])

    k_factor = metadata["profile_k_factor_linear"]
    if expected["los_k_factor_db"] is None:
        assert k_factor is None
    else:
        assert k_factor is not None and k_factor > 0
        assert 10.0 * math.log10(k_factor) == pytest.approx(expected["los_k_factor_db"], abs=tolerance["los_k_factor_db_abs"])


@pytest.mark.native_sionna
@pytest.mark.parametrize("profile", list("ABCDE"))
def test_cdl_profile_metadata_matches_independent_etsi_cluster_table(profile: str) -> None:
    from sionna.phy.channel.tr38901 import models

    reference = json.loads(_TAP_ANGLE_REFERENCE.read_text(encoding="utf-8"))
    expected = next(row for row in reference["profiles"] if row["profile_model"] == profile)
    metadata = _profile_metadata(profile, "19.2", 1.0, models, torch, "cpu", torch.float64)
    components = expected["components"]
    diffuse = [row for row in components if row["component"] == "diffuse"]
    specular = [row for row in components if row["component"] == "specular"]

    expected_delays = sorted(float(row["normalized_delay"]) for row in diffuse)
    actual_delays = [float(value) for value in metadata["nominal_cluster_delays_s"]]
    assert actual_delays == pytest.approx(expected_delays, abs=5.1e-5, rel=0.0)

    power_by_cluster: dict[int, float] = {}
    for row in components:
        power_by_cluster[row["cluster_index"]] = power_by_cluster.get(
            row["cluster_index"], 0.0
        ) + 10.0 ** (float(row["power_db"]) / 10.0)
    total_power = math.fsum(power_by_cluster.values())
    expected_powers = [
        power_by_cluster[row["cluster_index"]] / total_power
        for row in sorted(diffuse, key=lambda item: float(item["normalized_delay"]))
    ]
    actual_powers = [float(value) for value in metadata["normalized_cluster_powers"]]
    expected_relative_db = [10.0 * math.log10(value / max(expected_powers)) for value in expected_powers]
    actual_relative_db = [10.0 * math.log10(value / max(actual_powers)) for value in actual_powers]
    assert actual_relative_db == pytest.approx(expected_relative_db, abs=0.11, rel=0.0)

    expected_angles = {
        name: [
            float(row["angles_deg"][name])
            for row in sorted(diffuse, key=lambda item: float(item["normalized_delay"]))
        ]
        for name in ("aod", "aoa", "zod", "zoa")
    }
    for name, values in expected_angles.items():
        assert metadata["nominal_cluster_angles_deg"][name] == pytest.approx(
            values, abs=0.051, rel=0.0
        )

    for name, value in expected["cluster_spreads_deg"].items():
        assert metadata["cluster_angle_spreads_deg"][name] == pytest.approx(
            float(value), abs=0.051, rel=0.0
        )
    assert metadata["cross_polarization_power_ratio_db"] == pytest.approx(
        float(expected["xpr_db"]), abs=0.051, rel=0.0
    )
    if expected["los"]:
        assert len(specular) == 1
        assert metadata["nominal_los_angles_deg"] == pytest.approx(
            specular[0]["angles_deg"], abs=0.051, rel=0.0
        )
