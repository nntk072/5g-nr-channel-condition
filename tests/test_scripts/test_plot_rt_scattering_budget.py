"""Input validation and summary tests for diffuse-path budget plots."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLOTTER = runpy.run_path(str(ROOT / "scripts" / "plot_rt_scattering_budget.py"))
summarize_seed_range = PLOTTER["summarize_seed_range"]
load_report = PLOTTER["load_report"]


def _report(scattering: float) -> dict:
    return {
        "schema": "loslab.sionna_rt_solver_convergence.v3",
        "scene": {"name": "controlled"},
        "configuration": {
            "solver_flags": {"diffuse_reflection": True},
            "materials": {"surface": {"scattering_coefficient": [scattering]}},
            "max_depth_values": [0, 1],
            "samples_per_src_values": [100, 1000],
            "links": [
                {
                    "link_id": "link",
                    "geometric_visibility_state": "visible",
                }
            ],
        },
        "runs": [
            {
                "link_id": "link",
                "max_depth": 1,
                "samples_per_src": budget,
                "valid_diffuse_reflection_path_count": 1,
                "diffuse_path_power_sum": 1e-5,
                "diffuse_power_weighted_rms_delay_spread_ns": 2.0,
                "diffuse_power_weighted_arrival_angular_spread_rad_3d": 0.3,
            }
            for budget in (100, 1000)
        ],
    }


def test_seed_budget_summary_omits_undefined_diffuse_observations():
    rows = [
        {"samples_per_src": 100, "metric": 1.0},
        {"samples_per_src": 100, "metric": None},
        {"samples_per_src": 100, "metric": 3.0},
    ]
    assert summarize_seed_range(rows, x_value=100, metric="metric") == (2.0, 1.0, 3.0)
    assert summarize_seed_range(rows, x_value=1000, metric="metric") is None


def test_loader_rejects_solver_flag_without_nonzero_material_scattering(tmp_path: Path):
    report = tmp_path / "zero-scattering.json"
    report.write_text(json.dumps(_report(0.0)), encoding="utf-8")
    with pytest.raises(ValueError, match="no nonzero material scattering"):
        load_report(report)


def test_loader_requires_diffuse_observables_and_nonzero_scattering(tmp_path: Path):
    report = tmp_path / "scattering.json"
    report.write_text(json.dumps(_report(0.5)), encoding="utf-8")
    assert load_report(report)["configuration"]["materials"]["surface"][
        "scattering_coefficient"
    ] == [0.5]

    invalid = _report(0.5)
    del invalid["runs"][0]["diffuse_path_power_sum"]
    report.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="missing diffuse-component observables"):
        load_report(report)


def test_loader_accepts_cfr_bearing_v4_reports(tmp_path: Path):
    report = _report(0.5)
    report["schema"] = "loslab.sionna_rt_solver_convergence.v4"
    path = tmp_path / "scattering-v4.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert load_report(path)["schema"] == "loslab.sionna_rt_solver_convergence.v4"
