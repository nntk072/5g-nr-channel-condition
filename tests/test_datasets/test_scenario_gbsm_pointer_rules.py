"""Scenario GBSM stored runs: pointer, manifest-shape and record-path rules."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from datasets.scenario_gbsm import (
    read_scenario_gbsm_run,
    scenario_gbsm_run_from_channel,
    write_scenario_gbsm_run,
)
from tests.test_datasets.test_scenario_gbsm_store import _native_channel


def _written(tmp_path: Path):
    run = scenario_gbsm_run_from_channel(
        _native_channel(), run_id="pointer-rules", sample_ids=["drop-000"]
    )
    pointer = write_scenario_gbsm_run(tmp_path, run)
    return pointer, tmp_path / pointer.relative_path


def _digest(core: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rewrite(manifest_path: Path, mutate) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest["content_sha256"] = _digest(
        {k: manifest[k] for k in ("schema", "schema_version", "run_id", "records")}
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    return manifest


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"relative_path": "../escape.json"},
            "manifest path must remain inside the store",
            id="escape",
        ),
        pytest.param(
            {"schema_version": 2}, "unsupported scenario GBSM run schema version", id="schema"
        ),
        pytest.param({"content_sha256": "A" * 64}, "run pointer is malformed", id="digest"),
        pytest.param(
            {"sample_ids": ("x", "x")}, "run pointer is malformed", id="duplicate-samples"
        ),
        pytest.param({"run_id": "  "}, "run pointer is malformed", id="run-id"),
    ],
)
def test_pointer_fields_are_checked_before_reading(tmp_path, changes, message) -> None:
    pointer, _ = _written(tmp_path)
    with pytest.raises(ValueError, match=message):
        read_scenario_gbsm_run(tmp_path, dataclasses.replace(pointer, **changes))


def test_manifest_reader_accepts_a_relative_path_string(tmp_path) -> None:
    pointer, _ = _written(tmp_path)
    assert read_scenario_gbsm_run(tmp_path, pointer.relative_path).run_id == "pointer-rules"


def test_manifest_root_and_schema_are_checked(tmp_path) -> None:
    _, manifest_path = _written(tmp_path)
    manifest_path.write_text("[]")
    with pytest.raises(ValueError, match="invalid root or record roster"):
        read_scenario_gbsm_run(tmp_path, manifest_path.relative_to(tmp_path).as_posix())


def test_manifest_fields_and_schema_must_match(tmp_path) -> None:
    pointer, manifest_path = _written(tmp_path)
    _rewrite(manifest_path, lambda m: m.update(schema="scenario.v9"))
    with pytest.raises(ValueError, match="unsupported scenario GBSM run manifest"):
        read_scenario_gbsm_run(tmp_path, pointer)


def test_record_path_must_match_its_content_address(tmp_path) -> None:
    pointer, manifest_path = _written(tmp_path)
    manifest = _rewrite(
        manifest_path,
        lambda m: m["records"][0].update(relative_path="scenario_gbsm/records/00/wrong.npz"),
    )
    rebound = dataclasses.replace(pointer, content_sha256=manifest["content_sha256"])
    with pytest.raises(ValueError, match="does not match its content digest"):
        read_scenario_gbsm_run(tmp_path, rebound)
