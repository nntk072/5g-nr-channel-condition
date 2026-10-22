"""Tests for matched ray-budget comparisons across RT convergence reports."""

from __future__ import annotations

import gzip
import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPARE = runpy.run_path(str(ROOT / "scripts" / "compare_rt_solver_ray_budgets.py"))
compare_ray_budgets = COMPARE["compare_ray_budgets"]


def _row(
    *, budget: int, depth: int, real: list[float], path_count: int, los: bool = True
) -> dict[str, Any]:
    return {
        "link_id": "link-a",
        "seed": 7,
        "max_depth": depth,
        "samples_per_src": budget,
        "geometric_los": los,
        "cfr_vector_real": real,
        "cfr_vector_imag": [0.0] * len(real),
        "valid_path_count": path_count,
        "candidate_path_counts_per_source_before_path_validation": [path_count],
        "candidate_path_cap_filled": False,
        "valid_diffuse_reflection_path_count": 2,
    }


def _report(budget: int, rows: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "loslab.sionna_rt_solver_convergence.v4",
        "source_revision": "revision-" + str(budget),
        "scene": {"scene_tree_sha256": "scene"},
        "script_sha256": "runner",
        "package_source_sha256": "package",
        "versions": {"python": "3.12", "git_sha": "git-" + str(budget)},
        "configuration": {
            "cfr_vectors_included": True,
            "normalize_delays_for_cfr": False,
            "n_subcarriers": 2,
            "carrier_hz": 3.5e9,
            "samples_per_src_values": [budget],
            "reference_samples_per_src": budget,
        },
        "runs": rows,
    }
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(report.get(key), dict):
            report[key].update(value)
        else:
            report[key] = value
    return report


def _write(tmp_path: Path, low: dict[str, Any], ref: dict[str, Any]) -> tuple[Path, Path]:
    low_path, ref_path = tmp_path / "low.json", tmp_path / "reference.json"
    low_path.write_text(json.dumps(low), encoding="utf-8")
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    return low_path, ref_path


def test_reports_cfr_and_path_count_sensitivity_against_larger_budget(tmp_path: Path):
    low = _report(100, [_row(budget=100, depth=2, real=[1.0, 2.0], path_count=4)])
    ref = _report(1000, [_row(budget=1000, depth=2, real=[1.0, 2.1], path_count=5)])
    low_path, ref_path = _write(tmp_path, low, ref)

    result = compare_ray_budgets(low_path, ref_path, low_budget=100, reference_budget=1000)

    group = result["summary"]["grouped_by_depth_and_geometric_condition"][0]
    assert group["condition"] == "LOS"
    assert group["matched_runs"] == 1
    assert group["path_count_changes"] == 1
    assert group["relative_complex_cfr_l2_median"] == pytest.approx(0.1 / (1 + 2.1**2) ** 0.5)


def test_zero_reference_channel_is_reported_without_division_error(tmp_path: Path):
    low = _report(100, [_row(budget=100, depth=0, real=[0.0, 0.0], path_count=0)])
    ref = _report(1000, [_row(budget=1000, depth=0, real=[0.0, 0.0], path_count=0)])
    low_path, ref_path = _write(tmp_path, low, ref)
    result = compare_ray_budgets(low_path, ref_path, low_budget=100, reference_budget=1000)
    assert result["comparisons"][0]["relative_complex_cfr_l2"] == 0.0


def test_reads_reproducible_gzip_json_reports(tmp_path: Path):
    low = _report(100, [_row(budget=100, depth=2, real=[1.0, 2.0], path_count=4)])
    ref = _report(1000, [_row(budget=1000, depth=2, real=[1.0, 2.0], path_count=4)])
    low_path, ref_path = _write(tmp_path, low, ref)
    compressed = tmp_path / "reference.json.gz"
    compressed.write_bytes(gzip.compress(ref_path.read_bytes(), mtime=0))

    result = compare_ray_budgets(low_path, compressed, low_budget=100, reference_budget=1000)

    assert result["reports"][1]["sha256"]
    assert result["comparisons"][0]["relative_complex_cfr_l2"] == 0.0


def test_rejects_different_scene_and_nonincreasing_budget(tmp_path: Path):
    low = _report(100, [_row(budget=100, depth=2, real=[1.0, 2.0], path_count=4)])
    ref = _report(
        1000,
        [_row(budget=1000, depth=2, real=[1.0, 2.0], path_count=4)],
        scene={"scene_tree_sha256": "other"},
    )
    low_path, ref_path = _write(tmp_path, low, ref)
    with pytest.raises(ValueError, match="reports differ"):
        compare_ray_budgets(low_path, ref_path, low_budget=100, reference_budget=1000)
    with pytest.raises(ValueError, match="reference_budget must exceed"):
        compare_ray_budgets(low_path, ref_path, low_budget=100, reference_budget=100)
