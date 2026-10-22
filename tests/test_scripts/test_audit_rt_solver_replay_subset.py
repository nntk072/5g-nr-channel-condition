"""Tests for comparing overlapping rows across compatible RT reports."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = runpy.run_path(str(ROOT / "scripts" / "audit_rt_solver_replay_subset.py"))
audit_replay_subset = AUDIT["audit_replay_subset"]


def _report(*, real: list[float] | None = None, **changes: Any) -> dict[str, Any]:
    vector = real or [1.0, 2.0]
    value: dict[str, Any] = {
        "schema": "loslab.sionna_rt_solver_convergence.v4",
        "source_revision": "revision",
        "scene": {"scene_tree_sha256": "scene"},
        "script_sha256": "script",
        "package_source_sha256": "package",
        "versions": {"python": "3.12", "git_sha": "revision"},
        "configuration": {
            "cfr_vectors_included": True,
            "normalize_delays_for_cfr": False,
            "n_subcarriers": len(vector),
            "carrier_hz": 3.5e9,
            "samples_per_src_values": [100],
            "reference_samples_per_src": 100,
        },
        "runs": [
            {
                "link_id": "link-a",
                "seed": 11,
                "max_depth": 2,
                "samples_per_src": 100,
                "cfr_vector_real": vector,
                "cfr_vector_imag": [0.0] * len(vector),
                "cfr_sha256": "a" * 64 if vector == [1.0, 2.0] else "b" * 64,
                "valid_path_count": 3,
                "candidate_path_count_before_path_validation": 5,
                "candidate_path_counts_per_source_before_path_validation": [5],
            }
        ],
    }
    for key, replacement in changes.items():
        if isinstance(replacement, dict) and isinstance(value.get(key), dict):
            value[key].update(replacement)
        else:
            value[key] = replacement
    return value


def _write(tmp_path: Path, left: dict[str, Any], right: dict[str, Any]) -> tuple[Path, Path]:
    paths = (tmp_path / "a.json", tmp_path / "b.json")
    for path, report in zip(paths, (left, right), strict=True):
        path.write_text(json.dumps(report), encoding="utf-8")
    return paths


def test_compares_numeric_overlap_when_budget_sweep_differs(tmp_path: Path):
    left = _report()
    right = _report(real=[1.0, 2.000001])
    right["source_revision"] = "next-revision"
    right["versions"]["git_sha"] = "next-revision"
    right["configuration"]["samples_per_src_values"] = [100, 1000]
    right["configuration"]["reference_samples_per_src"] = 1000
    a, b = _write(tmp_path, left, right)

    result = audit_replay_subset(a, b)

    assert result["summary"]["matched_runs"] == 1
    assert result["summary"]["exact_cfr_hash_matches"] == 0
    assert result["summary"]["valid_path_count_matches"] == 1
    assert result["comparisons"][0]["relative_complex_cfr_l2"] == pytest.approx(1e-6 / (5**0.5))


@pytest.mark.parametrize(
    "changes",
    [
        {"scene": {"scene_tree_sha256": "other"}},
        {"script_sha256": "other"},
        {"package_source_sha256": "other"},
        {"configuration": {"cfr_vectors_included": True, "carrier_hz": 2.6e9}},
        {"versions": {"python": "other"}},
    ],
)
def test_rejects_incompatible_scene_or_software(tmp_path: Path, changes: dict[str, Any]):
    a, b = _write(tmp_path, _report(), _report(**changes))
    with pytest.raises(ValueError, match="scene, solver configuration, code, or runtime"):
        audit_replay_subset(a, b)


def test_rejects_vector_shape_mismatch(tmp_path: Path):
    malformed = _report(real=[1.0, 2.0, 3.0])
    malformed["configuration"]["n_subcarriers"] = 2
    a, b = _write(tmp_path, _report(), malformed)
    with pytest.raises(ValueError, match="invalid replay row"):
        audit_replay_subset(a, b)
