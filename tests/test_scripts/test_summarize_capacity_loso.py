import hashlib
import json
from pathlib import Path

import pytest
from scripts.summarize_capacity_loso import summarize_capacity_loso


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _make_suite(root: Path) -> Path:
    input_path = root / "observations.npz"
    input_path.write_bytes(b"fixed observations")
    input_row = {
        "path": str(input_path),
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    domains = ("rt_a", "rt_b")
    methods = ("mlp", "raw_mlp")
    seeds = (11, 23)
    tiers = ("small", "medium", "large")
    for domain_index, domain in enumerate(domains):
        fold = root / domain
        fold.mkdir()
        points = []
        children = {}
        for tier_index, tier in enumerate(tiers):
            child = fold / tier
            child.mkdir()
            child_config = {"capacity": tier, "holdout": domain}
            manifest_sha = f"identity-{domain}-{tier}"
            _write_json(
                child / "manifest.json",
                {
                    "state": "complete",
                    "config_identity": manifest_sha,
                    "config": child_config,
                    "inputs": [input_row],
                },
            )
            _write_json(child / "COMPLETE", {"config_identity": manifest_sha})
            rows = []
            for seed_index, seed in enumerate(seeds):
                split_hash = f"{domain}-{seed}"
                for method_index, method in enumerate(methods):
                    score = (
                        0.55
                        + 0.1 * domain_index
                        + 0.02 * tier_index
                        + 0.01 * method_index
                        + 0.001 * seed_index
                    )
                    rows.append(
                        {
                            "seed": seed,
                            "method": method,
                            "metrics": {"auroc": score},
                            "compute": {
                                "training_seconds": 0.1,
                                "learned_parameter_count": 100 + 10 * tier_index,
                                "peak_cuda_memory_bytes": 1024,
                            },
                            "split": {
                                "indices_sha256": split_hash,
                                "test_domains": [domain],
                                "train_domains": [other for other in domains if other != domain],
                                "test_size": 8,
                                "test_groups": 2,
                            },
                        }
                    )
            _write_json(child / "results.json", rows)
            children[tier] = {
                "path": tier,
                "manifest_sha256": hashlib.sha256(
                    (child / "manifest.json").read_bytes()
                ).hexdigest(),
            }
            for method in methods:
                scores = [row["metrics"]["auroc"] for row in rows if row["method"] == method]
                points.append(
                    {
                        "capacity": tier,
                        "method": method,
                        "seeds": list(seeds),
                        "mean_test_auroc": sum(scores) / len(scores),
                        "mean_training_seconds": 0.1,
                        "mean_learned_parameter_count": 100 + 10 * tier_index,
                    }
                )
        report = {
            "config_sha256": f"config-{domain}",
            "split_identity_count": len(methods) * len(seeds),
            "points": points,
            "child_runs": children,
        }
        report_path = fold / "capacity_sweep.json"
        _write_json(report_path, report)
        config = {
            "capacity_tiers": list(tiers),
            "methods": list(methods),
            "seeds": list(seeds),
            "holdout": domain,
            "observation": "precomputed",
            "device": "cuda:0",
            "bootstrap": 10,
        }
        _write_json(
            fold / "manifest.json",
            {
                "schema": "loslab.capacity_sweep.v1",
                "state": "complete",
                "config": config,
                "config_sha256": report["config_sha256"],
                "inputs": [input_row],
                "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
        )
    return root


def test_capacity_loso_audits_all_splits_and_aggregates_scenes(tmp_path):
    root = _make_suite(tmp_path)

    report = summarize_capacity_loso(root, ("rt_a", "rt_b"))

    assert report["fit_count"] == 24
    assert report["scene_count"] == 2
    point = next(
        item
        for item in report["scene_macro_results"]
        if item["capacity"] == "large" and item["method"] == "raw_mlp"
    )
    assert point["scene_macro_mean_test_auroc"] == pytest.approx(0.6505)
    assert len(point["scene_results"]) == 2
    delta = next(
        item
        for item in report["paired_capacity_deltas_vs_small"]
        if item["capacity_vs_small"] == "large" and item["method"] == "mlp"
    )
    assert delta["scenes_improved"] == 2
    assert delta["scene_macro_paired_auroc_delta"] == pytest.approx(0.04)


def test_capacity_loso_rejects_changed_observation_artifact(tmp_path):
    root = _make_suite(tmp_path)
    (tmp_path / "observations.npz").write_bytes(b"changed")

    with pytest.raises(ValueError, match="observation artifact is missing or changed"):
        summarize_capacity_loso(root, ("rt_a", "rt_b"))


def test_capacity_loso_rejects_a_capacity_split_change(tmp_path):
    root = _make_suite(tmp_path)
    results_path = root / "rt_a" / "large" / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    for row in rows:
        if row["seed"] == 11:
            row["split"]["indices_sha256"] = "different-split"
    _write_json(results_path, rows)

    with pytest.raises(ValueError, match="split identity changes across capacity"):
        summarize_capacity_loso(root, ("rt_a", "rt_b"))


def test_capacity_loso_rejects_heldout_scene_in_training(tmp_path):
    root = _make_suite(tmp_path)
    results_path = root / "rt_a" / "medium" / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    rows[0]["split"]["train_domains"].append("rt_a")
    _write_json(results_path, rows)

    with pytest.raises(ValueError, match="held-out scene appears in training"):
        summarize_capacity_loso(root, ("rt_a", "rt_b"))
