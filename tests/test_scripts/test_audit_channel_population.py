from __future__ import annotations

import json

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from scripts.audit_channel_population import audit_files, main


def _roster():
    return {
        "schema": "loslab.expected_population_roster.v1",
        "roster_id": "population-frozen-v1",
        "units": [
            {
                "unit_id": f"sample-{i}",
                "family_id": "scene-family-a",
                "domain_id": "scene-a",
                "independent_group_id": f"episode-{i}",
                "disposition": "included",
                "reason": None,
            }
            for i in (1, 2)
        ],
    }


def _dataset(path, sample_ids):
    size = len(sample_ids)
    dataset = ChannelDataset(
        h=np.ones((size, 1, 8), dtype=np.complex64),
        y=np.zeros(size, dtype=np.uint8),
        group=np.asarray([f"episode-{i}" for i in range(1, size + 1)]),
        domain=np.asarray(["scene-a"] * size),
        source=np.asarray(["scene-family-a"] * size),
        metadata=tuple({"sample_id": sample_id} for sample_id in sample_ids),
        label_semantics="test_fixture",
    )
    dataset.save(path)


def test_population_audit_workflow_binds_inputs_and_refuses_overwrite(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(_roster()), encoding="utf-8")
    dataset_path = tmp_path / "dataset.npz"
    _dataset(dataset_path, ["sample-1", "sample-2"])
    output_path = tmp_path / "audit.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_channel_population",
            "--roster",
            str(roster_path),
            "--input",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
    )
    main()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "ROSTER_MATCH"
    assert result["roster_source"]["sha256"]
    assert result["dataset_inputs"][0]["sha256"]
    with pytest.raises(FileExistsError):
        main()


def test_population_audit_workflow_publishes_mismatch_then_exits_nonzero(tmp_path, monkeypatch):
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(_roster()), encoding="utf-8")
    dataset_path = tmp_path / "dataset.npz"
    _dataset(dataset_path, ["sample-1"])
    output_path = tmp_path / "audit.json"
    report = audit_files(roster_path, [dataset_path])
    assert report["status"] == "ROSTER_MISMATCH"
    assert report["included_missing_unit_ids"] == ["sample-2"]

    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_channel_population",
            "--roster",
            str(roster_path),
            "--input",
            str(dataset_path),
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert json.loads(output_path.read_text(encoding="utf-8"))["status"] == "ROSTER_MISMATCH"
