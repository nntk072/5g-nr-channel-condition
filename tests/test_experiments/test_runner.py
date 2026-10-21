from __future__ import annotations

import json

import numpy as np
from datasets.synthetic import debug_multipath
from experiments.runner import run_experiment


def test_run_experiment_writes_reproducible_complete_artifacts(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=7).save(source)
    output = tmp_path / "runs" / "smoke"

    summary = run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=10,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["inputs"][0]["sha256"]
    assert (output / "COMPLETE").is_file()
    assert (output / "config" / "resolved.json").is_file()
    assert (output / "models" / "logreg_seed11.joblib").is_file()
    assert (output / "evaluation" / "summary.json").is_file()

    prediction = np.load(output / "predictions" / "logreg_seed11.npz")
    train = set(prediction["train_idx"].tolist())
    validation = set(prediction["val_idx"].tolist())
    test = set(prediction["test_idx"].tolist())
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert summary["aggregate_across_fixed_splits"]["logreg"]["seeds"] == 1


def test_run_experiment_refuses_to_overwrite_completed_run(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=7).save(source)
    output = tmp_path / "runs" / "smoke"
    run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=10)

    import pytest

    with pytest.raises(FileExistsError, match="immutable"):
        run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=10)
