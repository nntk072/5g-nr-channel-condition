"""TDL profile sweep: one curve per profile, matched batches and profile identity rules."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from artifacts.provenance import atomic_json
from evaluation.scoring.paired_link_curves import summarize_tdl_profile_sweep


def _curve(letter: str, **config: Any) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "profile": f"TDL-{letter}",
        "speed_mps": 3.0,
        "matched_realization_checks": [
            {
                "seed": 1,
                "noise_seed": 3,
                "noise_real_seed": 3,
                "noise_imag_seed": 4,
                "transmit_bits_sha256": "same-bits",
                "channel_truth_sha256": f"channel-{letter}",
            }
        ],
    }
    configuration.update(config)
    return {
        "schema": "loslab.tdl_pusch_time_bler_curve.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "configuration": configuration,
        "points": [
            {
                "snr_db": 12.0,
                "frames": 4,
                "bler_errors": int(letter in "CE"),
                "bler": 0.25 if letter in "CE" else 0.0,
                "bler_wilson_95_ci": [0.0, 0.5],
                "bit_errors": 4,
                "bits": 400,
                "ber": 0.01,
                "frame_error_mask": [letter in "CE", False, False, False],
            }
        ],
    }


def _write_all(tmp_path: Path, curves: dict[str, dict[str, Any]]) -> list[Path]:
    paths = []
    for name, curve in curves.items():
        path = tmp_path / f"{name}.json"
        atomic_json(path, curve)
        paths.append(path)
    return paths


def _full_set() -> dict[str, dict[str, Any]]:
    return {f"tdl-{letter}": _curve(letter) for letter in "ABCDE"}


def test_full_profile_sweep_summarizes(tmp_path: Path) -> None:
    paths = _write_all(tmp_path, _full_set())
    result = summarize_tdl_profile_sweep(paths)
    assert set(result["comparisons_vs_tdl_a"]) == {"TDL-B", "TDL-C", "TDL-D", "TDL-E"}


def test_sweep_requires_exactly_one_curve_per_profile(tmp_path: Path) -> None:
    curves = _full_set()
    del curves["tdl-E"]
    with pytest.raises(ValueError, match="exactly one curve for every TDL-A through TDL-E"):
        summarize_tdl_profile_sweep(_write_all(tmp_path, curves))


def test_sweep_rejects_unsupported_profiles(tmp_path: Path) -> None:
    curves = _full_set()
    curves["tdl-F"] = _curve("A", profile="TDL-F")
    with pytest.raises(ValueError, match="unsupported profile"):
        summarize_tdl_profile_sweep(_write_all(tmp_path, curves))


def test_sweep_rejects_duplicate_profiles(tmp_path: Path) -> None:
    curves = _full_set()
    curves["tdl-a-again"] = _curve("A")
    with pytest.raises(ValueError, match="duplicate TDL-A curve"):
        summarize_tdl_profile_sweep(_write_all(tmp_path, curves))


def test_sweep_checks_each_profile_against_the_reference(tmp_path: Path) -> None:
    curves = _full_set()
    curves["tdl-C"] = copy.deepcopy(curves["tdl-C"])
    curves["tdl-C"]["configuration"]["matched_realization_checks"][0]["transmit_bits_sha256"] = (
        "other"
    )
    with pytest.raises(ValueError, match="transmitted bits differ from TDL-A"):
        summarize_tdl_profile_sweep(_write_all(tmp_path, curves))


def test_sweep_requires_matching_batch_counts(tmp_path: Path) -> None:
    curves = _full_set()
    curves["tdl-D"] = copy.deepcopy(curves["tdl-D"])
    curves["tdl-D"]["configuration"]["matched_realization_checks"] = []
    with pytest.raises(ValueError, match="matched batch count differs from TDL-A"):
        summarize_tdl_profile_sweep(_write_all(tmp_path, curves))
