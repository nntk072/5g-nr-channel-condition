"""Tests for paired diffuse-scattering mechanism audits."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "audit_rt_scattering_control.py"))
audit_scattering_control = MODULE["audit_scattering_control"]


def _write_report(
    path: Path,
    *,
    scattering: float,
    diffuse_count: int,
    carrier_hz: float = 3.5e9,
    depth_zero_diffuse_count: int = 0,
) -> None:
    rows: list[dict[str, Any]] = []
    for depth, count in ((0, depth_zero_diffuse_count), (1, diffuse_count)):
        rows.append(
            {
                "link_id": "blocked",
                "seed": 11,
                "max_depth": depth,
                "samples_per_src": 1000,
                "valid_path_count": 2 + count,
                "valid_diffuse_reflection_path_count": count,
                "diffuse_path_power_sum": count * 0.01,
                "cfr_vector_real": [1.0, 2.0 + scattering],
                "cfr_vector_imag": [0.0, 0.0],
            }
        )
    path.write_text(
        json.dumps(
            {
                "schema": "loslab.sionna_rt_solver_convergence.v4",
                "scene": {"name": "box_one_screen", "scene_tree_sha256": "scene"},
                "configuration": {
                    "cfr_vectors_included": True,
                    "normalize_delays_for_cfr": False,
                    "n_subcarriers": 2,
                    "carrier_hz": carrier_hz,
                    "material_scattering_overrides": (
                        {"screen-mat": scattering} if scattering else {}
                    ),
                    "materials": {
                        "screen-mat": {
                            "scattering_coefficient": [scattering],
                            "scattering_pattern": "LambertianPattern",
                            "xpd_coefficient": [0.0],
                        }
                    },
                },
                "source_revision": "revision",
                "package_source_sha256": "package",
                "script_sha256": "script",
                "versions": {"sionna": "2.2.0"},
                "runs": rows,
            }
        ),
        encoding="utf-8",
    )


def test_scattering_control_confirms_diffuse_paths_and_channel_change(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write_report(baseline, scattering=0.0, diffuse_count=0)
    _write_report(intervention, scattering=0.5, diffuse_count=3)

    result = audit_scattering_control(baseline, intervention, material_name="screen-mat")

    assert result["summary"]["passed"] is True
    assert result["summary"]["matched_case_count"] == 2
    assert result["summary"]["zero_scattering_cases_with_diffuse_paths"] == 0
    assert result["summary"]["nonzero_scattering_cases_with_diffuse_paths"] == 1
    assert result["summary"]["all_depth_zero_cases_have_no_diffuse_paths"] is True
    assert result["comparisons"][1]["relative_cfr_l2_error"] == pytest.approx(0.5 / 5**0.5)


def test_scattering_control_rejects_unmatched_solver_configuration(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write_report(baseline, scattering=0.0, diffuse_count=0)
    _write_report(intervention, scattering=0.5, diffuse_count=3, carrier_hz=3.6e9)

    with pytest.raises(ValueError, match="solver or link configuration"):
        audit_scattering_control(baseline, intervention, material_name="screen-mat")


def test_scattering_control_rejects_diffuse_paths_in_zero_control(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write_report(baseline, scattering=0.0, diffuse_count=1)
    _write_report(intervention, scattering=0.5, diffuse_count=3)

    with pytest.raises(ValueError, match="zero scattering produced"):
        audit_scattering_control(baseline, intervention, material_name="screen-mat")


def test_scattering_control_rejects_diffuse_paths_at_zero_interaction_depth(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write_report(baseline, scattering=0.0, diffuse_count=0)
    _write_report(
        intervention,
        scattering=0.5,
        diffuse_count=3,
        depth_zero_diffuse_count=1,
    )

    with pytest.raises(ValueError, match="zero-interaction depth"):
        audit_scattering_control(baseline, intervention, material_name="screen-mat")


def test_scattering_control_rejects_intervention_without_diffuse_paths(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    intervention = tmp_path / "intervention.json"
    _write_report(baseline, scattering=0.0, diffuse_count=0)
    _write_report(intervention, scattering=0.5, diffuse_count=0)

    with pytest.raises(ValueError, match="nonzero scattering produced no diffuse paths"):
        audit_scattering_control(baseline, intervention, material_name="screen-mat")
