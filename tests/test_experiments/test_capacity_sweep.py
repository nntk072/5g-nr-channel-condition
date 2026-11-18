from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from artifacts.run_replay import validate_run_replay_provenance
from datasets.synthetic import debug_multipath
from experiments.capacity_sweep import (
    _capacity_sweep_rng_streams,
    _validate_manifest,
    run_capacity_sweep,
)


def test_capacity_sweep_replay_rng_inventory_covers_optional_controls() -> None:
    streams = _capacity_sweep_rng_streams(
        {
            "seeds": [11],
            "methods": ["label_permuted", "row_shuffled"],
            "observation": "comb_ls",
            "observation_seed": 21,
            "phase_randomization_seed": 22,
            "label_permutation_seed": 23,
            "row_shuffle_seed": 24,
        }
    )

    assert {stream["name"] for stream in streams} == {
        "capacity_child_seed_11",
        "observation_noise",
        "phase_randomization_control",
        "label_permutation_base",
        "row_shuffle_base",
    }
    assert {tuple(stream["seed_material"]) for stream in streams} == {
        (11,),
        (21,),
        (22,),
        (23,),
        (24,),
    }


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
    assert manifest["schema"] == "loslab.capacity_sweep.v2"
    replay = validate_run_replay_provenance(manifest["run_replay"])
    assert replay["state"] == "complete"
    assert replay["config_sha256"] == manifest["config_sha256"]
    assert replay["inputs"] == manifest["inputs"]
    assert {row["name"] for row in replay["rng_streams"]} == {
        "capacity_child_seed_13",
        "phase_randomization_control",
    }
    schema = json.loads(
        (
            Path(__file__).parents[2] / "experiments/schema/capacity-sweep-manifest.v2.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    _validate_manifest(manifest)
    assert (output / "COMPLETE").is_file()
    assert all((output / tier / "COMPLETE").is_file() for tier in ("small", "medium", "large"))
    with pytest.raises(FileExistsError, match="already exists"):
        run_capacity_sweep([dataset_path], output, methods=("mlp",), seeds=(13,), bootstrap=10)


def test_capacity_sweep_persists_failed_parent_replay_receipt(tmp_path, monkeypatch) -> None:
    dataset_path = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=67).save(dataset_path)
    output = tmp_path / "sweeps" / "failed-capacities"

    def fail_child(*args, **kwargs):
        raise RuntimeError("injected child failure")

    monkeypatch.setattr("experiments.capacity_sweep.run_experiment", fail_child)
    with pytest.raises(RuntimeError, match="injected child failure"):
        run_capacity_sweep([dataset_path], output, methods=("mlp",), seeds=(13,), bootstrap=10)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    replay = validate_run_replay_provenance(manifest["run_replay"])
    assert manifest["state"] == replay["state"] == "failed"
    assert replay["failure"]["stage"] == "capacity_sweep"
    _validate_manifest(manifest)
