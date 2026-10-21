from __future__ import annotations

import json

import numpy as np
from datasets.synthetic import debug_multipath
from experiments.runner import _label_semantics_by_source, _resolved_label_semantics, run_experiment


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
    assert manifest["label_semantics"] == "synthetic_debug_direct_component"
    assert manifest["label_semantics_by_source"] == "SYNTHETIC_DEBUG_CONSTRUCTED"
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


def test_run_experiment_reports_paired_classical_vs_mlp_deltas(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=12, links_per_scene=24, seed=19).save(source)

    summary = run_experiment(
        [source],
        tmp_path / "runs" / "paired",
        methods=("rf", "mlp"),
        seeds=(11, 23, 47),
        bootstrap=10,
    )

    comparison = summary["paired_method_comparisons"]["mlp_versus_rf"]
    assert comparison["paired_seeds"] == [11, 23, 47]
    assert len(comparison["treatment_minus_control"]) == 3
    assert comparison["test"] == "exact paired sign-flip"
    assert comparison["warning"].startswith("Fewer than six seeds")

    results = json.loads((tmp_path / "runs" / "paired" / "results.json").read_text())
    by_seed_and_method = {(row["seed"], row["method"]): row for row in results}
    for seed in (11, 23, 47):
        assert (
            by_seed_and_method[(seed, "rf")]["split"]["indices_sha256"]
            == by_seed_and_method[(seed, "mlp")]["split"]["indices_sha256"]
        )


def test_label_semantics_are_derived_from_each_dataset_source() -> None:
    from datasets.channel import ChannelDataset

    data = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["nlos", "los"]),
        domain=np.asarray(["scene-a", "profile-b"]),
        source=np.asarray(["sionna_rt", "tdl"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"label_semantics": "PROFILE_PROXY"},
        ),
    )

    assert (
        _label_semantics_by_source(data)
        == "sionna_rt=GEOMETRIC_DIRECT_PATH_ORACLE; tdl=PROFILE_PROXY"
    )


def test_uniform_legacy_label_metadata_resolves_to_canonical_semantics() -> None:
    from datasets.channel import ChannelDataset

    legacy_rt = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["nlos", "los"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["sionna_rt", "sionna_rt"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
        ),
    )

    assert _resolved_label_semantics(legacy_rt) == "geometric_direct_path_oracle"


def test_unspecified_legacy_label_metadata_is_not_treated_as_a_task() -> None:
    import pytest
    from datasets.channel import ChannelDataset

    unknown = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["old_source", "old_source"]),
        metadata=({"label_semantics": "UNSPECIFIED"}, {"label_semantics": "unknown"}),
    )

    with pytest.raises(ValueError, match="one explicit, consistent label meaning"):
        _resolved_label_semantics(unknown)


def test_experiment_rejects_mixed_legacy_label_semantics_before_training(tmp_path) -> None:
    import pytest
    from datasets.channel import ChannelDataset

    mixed = ChannelDataset(
        h=np.ones((4, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1, 0, 1]),
        group=np.asarray(["g0", "g1", "g2", "g3"]),
        domain=np.asarray(["d0", "d1", "d2", "d3"]),
        source=np.asarray(["rt", "rt", "tdl", "tdl"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"label_semantics": "PROFILE_PROXY"},
            {"label_semantics": "PROFILE_PROXY"},
        ),
    )
    source = tmp_path / "mixed-legacy.npz"
    mixed.save(source)

    with pytest.raises(ValueError, match="one explicit, consistent label meaning"):
        run_experiment([source], tmp_path / "runs" / "must-not-train")

    assert not (tmp_path / "runs" / "must-not-train").exists()


def test_run_experiment_preserves_precomputed_dmrs_observation_semantics(tmp_path) -> None:
    from datasets.channel import ChannelDataset

    generated = debug_multipath(n_scenes=4, links_per_scene=12, seed=29)
    source = tmp_path / "dmrs-observations.npz"
    observed = ChannelDataset(
        h=generated.h,
        y=generated.y,
        group=generated.group,
        domain=generated.domain,
        source=generated.source,
        metadata=tuple(
            {**metadata, "receiver_observation": "Sionna PUSCH DMRS LS channel estimate"}
            for metadata in generated.metadata
        ),
        generation_report=generated.generation_report,
    )
    observed.save(source)
    output = tmp_path / "runs" / "precomputed"

    run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=10,
        observation="precomputed",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_semantics"] == "Sionna PUSCH DMRS LS channel estimate"
    config = json.loads((output / "config" / "resolved.json").read_text(encoding="utf-8"))
    assert config["values"]["observation"] == "precomputed"
