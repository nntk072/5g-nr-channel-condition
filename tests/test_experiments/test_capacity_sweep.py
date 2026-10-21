from __future__ import annotations

import json

import pytest
from datasets.synthetic import debug_multipath
from experiments.capacity_sweep import run_capacity_sweep


def test_capacity_sweep_reuses_splits_and_writes_immutable_report(tmp_path) -> None:
    dataset_path = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=67).save(dataset_path)
    output = tmp_path / "sweeps" / "capacities"
    report = run_capacity_sweep([dataset_path], output, methods=("mlp",), seeds=(13,), bootstrap=10)

    assert report["split_identity_count"] == 1
    assert [point["capacity"] for point in report["points"]] == [
        "small",
        "medium",
        "large",
    ]
    parameter_counts = [point["mean_learned_parameter_count"] for point in report["points"]]
    assert parameter_counts[0] < parameter_counts[1] < parameter_counts[2]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert (output / "COMPLETE").is_file()
    assert all((output / tier / "COMPLETE").is_file() for tier in ("small", "medium", "large"))
    with pytest.raises(FileExistsError, match="already exists"):
        run_capacity_sweep([dataset_path], output, methods=("mlp",), seeds=(13,), bootstrap=10)
