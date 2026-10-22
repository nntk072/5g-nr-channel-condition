"""Tests for numerical replay audits of CFR-bearing native solver reports."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(str(ROOT / "scripts" / "audit_rt_solver_replay.py"))
audit_solver_replay = MODULE["audit_solver_replay"]


def _write_report(
    path: Path,
    *,
    real: list[float],
    imag: list[float] | None = None,
    path_count: int = 2,
    scene_hash: str = "scene",
    include_vectors: bool = True,
    schema: str = "loslab.sionna_rt_solver_convergence.v4",
    n_subcarriers: int | None = None,
) -> None:
    vector_imag = imag if imag is not None else [0.0] * len(real)
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "scene": {
                    "name": "box_one_screen",
                    "scene_tree_sha256": scene_hash,
                },
                "configuration": {
                    "cfr_vectors_included": include_vectors,
                    "normalize_delays_for_cfr": False,
                    "max_num_paths_per_src": 1000,
                    "n_subcarriers": n_subcarriers if n_subcarriers is not None else len(real),
                    "mitsuba_variant": "llvm_ad_mono_polarized",
                },
                "source_revision": "revision",
                "package_source_sha256": "package",
                "script_sha256": "script",
                "runs": [
                    {
                        "link_id": "los",
                        "seed": 11,
                        "max_depth": 3,
                        "samples_per_src": 1000,
                        "cfr_vector_real": real,
                        "cfr_vector_imag": vector_imag,
                        "cfr_sha256": "a" * 64 if real == [1.0, 2.0] else "b" * 64,
                        "valid_path_count": path_count,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_replay_audit_passes_small_float_drift_without_bitwise_identity(tmp_path: Path):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0])
    _write_report(replay, real=[1.0, 2.0 + 1e-7])

    result = audit_solver_replay(reference, replay, max_relative_cfr_l2_error=1e-6)

    assert result["summary"]["passed"] is True
    assert result["summary"]["case_count"] == 1
    assert result["summary"]["exact_cfr_hash_matches"] == 0
    assert result["summary"]["path_count_matches"] == 1
    assert result["comparisons"][0]["relative_cfr_l2_error"] == pytest.approx(1e-7 / (5**0.5))


def test_replay_audit_fails_when_cfr_drift_exceeds_tolerance(tmp_path: Path):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0])
    _write_report(replay, real=[1.0, 2.1])

    result = audit_solver_replay(reference, replay, max_relative_cfr_l2_error=1e-3)

    assert result["summary"]["passed"] is False
    assert result["failed_cases"] == [
        {"link_id": "los", "seed": 11, "max_depth": 3, "samples_per_src": 1000}
    ]


def test_replay_audit_fails_when_path_count_changes_even_if_cfr_matches(tmp_path: Path):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0], path_count=2)
    _write_report(replay, real=[1.0, 2.0], path_count=3)

    result = audit_solver_replay(reference, replay)

    assert result["summary"]["path_count_matches"] == 0
    assert result["summary"]["passed"] is False


def test_replay_audit_rejects_configuration_and_scene_mismatches(tmp_path: Path):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0])
    _write_report(replay, real=[1.0, 2.0], scene_hash="other")

    with pytest.raises(ValueError, match="scene"):
        audit_solver_replay(reference, replay)


def test_replay_audit_rejects_mismatched_cfr_vector_shapes(tmp_path: Path):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0])
    _write_report(replay, real=[1.0, 2.0, 3.0], n_subcarriers=2)

    with pytest.raises(ValueError, match="CFR-grid dimensions"):
        audit_solver_replay(reference, replay)


@pytest.mark.parametrize(
    ("schema", "include_vectors"),
    [
        ("loslab.sionna_rt_solver_convergence.v3", True),
        ("loslab.sionna_rt_solver_convergence.v4", False),
    ],
)
def test_replay_audit_requires_v4_cfr_vectors(tmp_path: Path, schema: str, include_vectors: bool):
    reference = tmp_path / "reference.json"
    replay = tmp_path / "replay.json"
    _write_report(reference, real=[1.0, 2.0], schema=schema, include_vectors=include_vectors)
    _write_report(replay, real=[1.0, 2.0], schema=schema, include_vectors=include_vectors)

    with pytest.raises(ValueError, match="schema-v4|retain complex CFR vectors"):
        audit_solver_replay(reference, replay)
