"""Validation for matched path-cap comparisons."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "plot_rt_path_cap_sweep.py"))
summarize_cap_reports = MODULE["summarize_cap_reports"]
plot_cap_sweep = MODULE["plot_cap_sweep"]


def _write_report(path: Path, *, cap: int, path_count: int, scene_hash: str = "scene") -> None:
    configuration = {
        "normalize_delays_for_cfr": False,
        "max_num_paths_per_src": cap,
        "samples_per_src_values": [100_000],
        "max_depth_values": [6],
        "seeds": [11, 23],
        "links": [
            {"link_id": "los", "geometric_visibility_state": "visible"},
        ],
        "materials": {"screen": {"scattering_coefficient": [0.5]}},
        "material_scattering_overrides": {"screen": 0.5},
        "mitsuba_variant": "llvm_ad_mono_polarized",
    }
    runs = [
        {
            "max_depth": 6,
            "samples_per_src": 100_000,
            "seed": seed,
            "link_id": "los",
            "valid_path_count": path_count + (seed - 11) // 12,
            "valid_diffuse_reflection_path_count": path_count - 2 + (seed - 11) // 12,
            "valid_path_power_sum": 0.25,
            "power_weighted_rms_delay_spread_ns": 12.0 + seed / 10,
            "power_weighted_arrival_angular_spread_rad_3d": 0.5,
        }
        for seed in configuration["seeds"]
    ]
    path.write_text(
        json.dumps(
            {
                "schema": "loslab.sionna_rt_solver_convergence.v3",
                "scene": {"name": "fixture", "scene_tree_sha256": scene_hash},
                "configuration": configuration,
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def test_cap_summary_checks_matched_rows_and_summarizes_seed_range(tmp_path: Path):
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write_report(low, cap=1000, path_count=998)
    _write_report(high, cap=10_000, path_count=8000)

    summary = summarize_cap_reports([high, low], max_depth=6, sample_budget=100_000)

    assert [item["max_num_paths_per_src"] for item in summary["inputs"]] == [1000, 10_000]
    assert len(summary["per_seed"]) == 4
    low_summary = next(row for row in summary["summaries"] if row["max_num_paths_per_src"] == 1000)
    assert low_summary["metrics"]["valid_path_count"] == {
        "median": 998.5,
        "min": 998.0,
        "max": 999.0,
    }


def test_cap_summary_rejects_scene_mismatch(tmp_path: Path):
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write_report(low, cap=1000, path_count=998)
    _write_report(high, cap=10_000, path_count=8000, scene_hash="other")

    with pytest.raises(ValueError, match="same scene asset hash"):
        summarize_cap_reports([low, high], max_depth=6, sample_budget=100_000)


def test_cap_figure_writes_image_and_provenance_sidecar(tmp_path: Path):
    pytest.importorskip("matplotlib")
    low = tmp_path / "low.json"
    high = tmp_path / "high.json"
    _write_report(low, cap=1000, path_count=998)
    _write_report(high, cap=10_000, path_count=8000)
    summary = summarize_cap_reports([low, high], max_depth=6, sample_budget=100_000)

    plotted = plot_cap_sweep(summary, tmp_path / "cap-sweep")

    assert (tmp_path / "cap-sweep.png").is_file()
    assert (tmp_path / "cap-sweep.svg").is_file()
    assert (tmp_path / "cap-sweep.json").is_file()
    assert plotted["figure"]["png"]["sha256"]
