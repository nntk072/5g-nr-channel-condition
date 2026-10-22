import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from scripts.summarize_rt_tx_holdout import summarize_tx_holdouts


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _make_run(root: Path) -> None:
    ids = ("west_x1", "west_x2", "west_x3")
    samples = 40 * len(ids)
    labels = np.tile(np.tile([0, 1], 20), len(ids))
    domains = np.repeat([f"rt_box_one_screen_{item}" for item in ids], 40)
    groups = np.asarray(
        [f"box_tx_{item}_block_{index // 10}" for item in ids for index in range(40)]
    )
    metadata = tuple(
        {
            "transmitter_id": item,
            "scene": "box_one_screen",
            "tx": [-float(index + 1), 0.0, 2.5],
            "rx": [
                (-80.0 + 80.0 * index) + float(sample % 20) * 0.2,
                float(sample // 20),
                2.5,
            ],
            "scene_config": {
                "rx_center": [-80.0 + 80.0 * index, 0.0, 2.5],
                "radius_m": 25.0,
            },
            "scene_asset_set_sha256": "a" * 64,
        }
        for index, item in enumerate(ids)
        for sample in range(40)
    )
    observation_path = root / "nr-dmrs-observations.npz"
    ChannelDataset(
        h=np.ones((samples, 1, 8), dtype=np.complex64),
        y=labels,
        group=groups,
        domain=domains,
        source=np.full(samples, "sionna_rt"),
        metadata=metadata,
        label_semantics="geometric_direct_path_oracle",
    ).save(observation_path)
    input_hash = hashlib.sha256(observation_path.read_bytes()).hexdigest()
    for item in ids:
        (root / f"{item}.npz").write_bytes(f"raw-{item}".encode())
    declared_domains = sorted(set(domains))
    for item in ids:
        holdout = f"rt_box_one_screen_{item}"
        fold_dir = root / f"holdout-{item}"
        config = {
            "holdout": holdout,
            "methods": ["logreg"],
            "seeds": [11, 23],
            "observation": "precomputed",
        }
        (fold_dir / "COMPLETE").parent.mkdir(parents=True)
        (fold_dir / "COMPLETE").write_text("complete", encoding="utf-8")
        _write_json(
            fold_dir / "manifest.json",
            {
                "config": config,
                "inputs": [{"path": str(observation_path), "sha256": input_hash}],
            },
        )
        rows = [
            {
                "method": "logreg",
                "seed": seed,
                "metrics": {"auroc": score},
                "split": {
                    "test_domains": [holdout],
                    "train_domains": [domain for domain in declared_domains if domain != holdout],
                    "indices_sha256": f"{item}-{seed}",
                    "test_size": 40,
                },
            }
            for seed, score in ((11, 0.6), (23, 0.8))
        ]
        _write_json(fold_dir / "results.json", rows)
        _write_json(fold_dir / "evaluation" / "summary.json", {"ok": True})


def test_summarize_tx_holdouts_audits_domain_splits_and_receiver_overlap(tmp_path):
    _make_run(tmp_path)

    report = summarize_tx_holdouts(tmp_path)

    assert report["transmitter_count"] == 3
    assert report["observation_dataset"]["sample_count"] == 120
    assert report["study_scope"] == "joint within-scene transmitter-and-receiver location holdout"
    assert report["minimum_receiver_region_separation_m"] == {
        "west_x1__west_x2": pytest.approx(30.0),
        "west_x1__west_x3": pytest.approx(110.0),
        "west_x2__west_x3": pytest.approx(30.0),
    }
    assert all(
        row["rx_samples_with_other_tx_within_1m"] == 0
        for row in report["receiver_position_overlap_diagnostic"].values()
    )
    assert report["mean_across_heldout_tx_locations"]["logreg"][
        "descriptive_mean_auroc"
    ] == pytest.approx(0.7)
    assert len(report["folds"]) == 3
    assert len(report["receiver_position_overlap_diagnostic"]) == 3


def test_summarize_tx_holdouts_rejects_changed_observation_input(tmp_path):
    _make_run(tmp_path)
    observation_path = tmp_path / "nr-dmrs-observations.npz"
    with observation_path.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(ValueError, match="exact shared observation artifact"):
        summarize_tx_holdouts(tmp_path)


def test_summarize_tx_holdouts_rejects_held_out_transmitter_in_training(tmp_path):
    _make_run(tmp_path)
    results_path = tmp_path / "holdout-west_x1" / "results.json"
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    rows[0]["split"]["train_domains"].append("rt_box_one_screen_west_x1")
    _write_json(results_path, rows)

    with pytest.raises(ValueError, match="held-out transmitter leaked"):
        summarize_tx_holdouts(tmp_path)


def test_summarize_tx_holdouts_rejects_overlapping_receiver_regions(tmp_path):
    _make_run(tmp_path)
    observation_path = tmp_path / "nr-dmrs-observations.npz"
    observations = ChannelDataset.load(observation_path)
    for metadata in observations.metadata:
        if metadata["transmitter_id"] == "west_x2":
            metadata["scene_config"]["rx_center"] = [-80.0, 0.0, 2.5]
            metadata["rx"][0] -= 80.0
    observations.save(observation_path)

    with pytest.raises(ValueError, match="receiver-region disks overlap"):
        summarize_tx_holdouts(tmp_path)
