import hashlib
import json
from pathlib import Path

import pytest
from scripts.summarize_rt_dmrs_losso import summarize_loso_run


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(root: Path) -> None:
    input_path = root / "observations.npz"
    input_path.write_bytes(b"locked observations")
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    domains = ["rt_a", "rt_b", "rt_c"]
    config_base = {
        "methods": ["logreg", "raw_mlp"],
        "seeds": [11, 23],
        "observation": "precomputed",
    }
    for holdout in domains:
        fold_dir = root / f"holdout-{holdout}"
        fold_dir.mkdir()
        (fold_dir / "COMPLETE").write_text("complete", encoding="utf-8")
        config = {**config_base, "holdout": holdout}
        _write_json(
            fold_dir / "manifest.json",
            {
                "config": config,
                "domain_counts": {domain: 4 for domain in domains},
                "inputs": [{"path": str(input_path), "sha256": input_hash}],
            },
        )
        rows = []
        for seed in config["seeds"]:
            for method, auroc in (("logreg", 0.6), ("raw_mlp", 0.8)):
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "metrics": {"auroc": auroc},
                        "split": {
                            "test_domains": [holdout],
                            "train_domains": [domain for domain in domains if domain != holdout],
                            "test_size": 4,
                            "test_groups": 2,
                            "indices_sha256": f"{holdout}-{seed}",
                        },
                    }
                )
        _write_json(fold_dir / "results.json", rows)
        _write_json(fold_dir / "evaluation" / "summary.json", {"ok": True})


def test_summarize_loso_run_requires_all_disjoint_folds_and_aggregates_scenes(tmp_path):
    _make_run(tmp_path)

    report = summarize_loso_run(tmp_path)

    assert report["scene_count"] == 3
    assert report["scene_macro_test_auroc"]["logreg"]["mean_across_scenes"] == pytest.approx(0.6)
    assert report["scene_macro_test_auroc"]["raw_mlp"]["mean_across_scenes"] == pytest.approx(0.8)
    assert len(report["folds"]) == 3


def test_summarize_loso_run_rejects_heldout_scene_in_training(tmp_path):
    _make_run(tmp_path)
    result_path = tmp_path / "holdout-rt_a" / "results.json"
    rows = json.loads(result_path.read_text(encoding="utf-8"))
    rows[0]["split"]["train_domains"].append("rt_a")
    _write_json(result_path, rows)

    with pytest.raises(ValueError, match="held-out scene appears in training"):
        summarize_loso_run(tmp_path)


def test_summarize_loso_run_links_matching_upstream_scene_hash_report(tmp_path):
    _make_run(tmp_path)
    input_path = tmp_path / "observations.npz"
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    source_report = tmp_path / "source-report.json"
    _write_json(
        source_report,
        {
            "dataset": {
                "observation_dataset_sha256": input_hash,
                "source_datasets": {"scene.npz": "scene-hash"},
            }
        },
    )

    report = summarize_loso_run(tmp_path, source_report)

    assert (
        report["source_data_evidence"]["sha256"]
        == hashlib.sha256(source_report.read_bytes()).hexdigest()
    )
    assert report["source_data_evidence"]["source_scene_datasets"] == {"scene.npz": "scene-hash"}
