"""Stored learning views: manifest, observation and target layer integrity checks."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from datasets import standardized as standardized_module
from datasets.standardized import (
    StandardizedFamily,
    _canonical_json,
    read_standardized_learning_run,
    standardized_learning_run_from_channel_run,
    standardized_run_from_channel,
    write_standardized_learning_run,
)
from tests.test_datasets.test_standardized import _channel


def _write(tmp_path: Path):
    channels = standardized_run_from_channel(
        _channel(), family=StandardizedFamily.TDL, run_id="learning-store"
    )
    learning = standardized_learning_run_from_channel_run(channels, snr_db=15.0, seed=7)
    stored = write_standardized_learning_run(tmp_path, learning)
    return stored, tmp_path / stored.relative_path


def _digest(raw: dict[str, Any]) -> str:
    core = {key: raw[key] for key in ("schema", "schema_version", "run_id", "records")}
    return hashlib.sha256(_canonical_json(core, name="learning run manifest")).hexdigest()


def _edit_manifest(manifest: Path, stored, mutate, *, refresh_digest: bool = True):
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(raw)
    if refresh_digest:
        raw["content_sha256"] = _digest(raw)
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    return dataclasses.replace(stored, content_sha256=raw["content_sha256"])


def _resave_npz(root: Path, reference: dict[str, Any], change) -> None:
    old = root / reference["relative_path"]
    with np.load(old, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(arrays["metadata_json"].tobytes().decode("utf-8"))
    change(metadata)
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(metadata, sort_keys=True).encode("utf-8"), dtype=np.uint8
    )
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    payload = buffer.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    new_path = old.with_name(f"{digest}.npz")
    new_path.write_bytes(payload)
    reference["relative_path"] = new_path.relative_to(root).as_posix()
    reference["content_sha256"] = digest


def test_learning_store_reads_from_a_relative_manifest_string(tmp_path: Path) -> None:
    stored, _ = _write(tmp_path)

    restored = read_standardized_learning_run(tmp_path, stored.relative_path)

    assert restored.run_id == "learning-store"
    assert [item.sample_id for item in restored.examples] == list(stored.sample_ids)


def test_learning_store_reads_a_relative_manifest_path(tmp_path: Path) -> None:
    stored, _ = _write(tmp_path)

    restored = read_standardized_learning_run(tmp_path, Path(stored.relative_path))

    assert [item.sample_id for item in restored.examples] == list(stored.sample_ids)


def test_learning_store_reads_an_absolute_manifest_path(tmp_path: Path) -> None:
    stored, _ = _write(tmp_path)

    restored = read_standardized_learning_run(tmp_path, tmp_path / stored.relative_path)

    assert [item.sample_id for item in restored.examples] == list(stored.sample_ids)


def test_immutable_publish_skips_cleanup_when_tempfile_creation_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_to_create_tempfile(**_kwargs):
        raise OSError("fixture blocks temporary file creation")

    monkeypatch.setattr(standardized_module.tempfile, "NamedTemporaryFile", fail_to_create_tempfile)

    with pytest.raises(OSError, match="blocks temporary file creation"):
        standardized_module._publish_immutable(tmp_path / "manifest.json", b"payload")


@pytest.mark.parametrize(
    ("pointer_change", "message"),
    [
        pytest.param(
            {"relative_path": "../escape.json"}, "learning manifest path must remain", id="escape"
        ),
        pytest.param(
            {"schema_version": 3},
            "unsupported standardized learning run schema",
            id="schema-version",
        ),
        pytest.param(
            {"run_id": "other"}, "run ID disagrees with its manifest pointer", id="run-id"
        ),
    ],
)
def test_learning_pointer_is_checked_against_its_manifest(
    tmp_path: Path, pointer_change, message
) -> None:
    stored, _ = _write(tmp_path)
    pointer = dataclasses.replace(stored, **pointer_change)

    with pytest.raises(ValueError, match=message):
        read_standardized_learning_run(tmp_path, pointer)


def test_learning_manifest_schema_must_be_supported(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    wrong_schema = _edit_manifest(
        manifest, stored, lambda raw: raw.update(schema="standardized_learning_run.v9")
    )
    with pytest.raises(ValueError, match="unsupported standardized learning run schema"):
        read_standardized_learning_run(tmp_path, wrong_schema)


def test_learning_manifest_digest_must_match_its_content(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    tampered = _edit_manifest(
        manifest, stored, lambda raw: raw.update(run_id="tampered"), refresh_digest=False
    )
    with pytest.raises(ValueError, match="learning manifest hash is invalid"):
        read_standardized_learning_run(tmp_path, tampered)


def test_learning_observation_path_must_stay_inside_the_store(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    updated = _edit_manifest(
        manifest,
        stored,
        lambda raw: raw["records"][0]["observation"].update(relative_path="../escape.npz"),
    )
    with pytest.raises(ValueError, match="observation path must remain inside the store"):
        read_standardized_learning_run(tmp_path, updated)


def test_learning_observation_bytes_are_content_addressed(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    observation = tmp_path / raw["records"][0]["observation"]["relative_path"]
    observation.write_bytes(observation.read_bytes() + b"x")

    with pytest.raises(ValueError, match="observation content hash is invalid"):
        read_standardized_learning_run(tmp_path, stored)


def test_learning_observation_identity_must_match_its_sample(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)

    def retarget(raw: dict[str, Any]) -> None:
        _resave_npz(tmp_path, raw["records"][0]["observation"], lambda m: m.update(sample_id="x"))

    updated = _edit_manifest(manifest, stored, retarget)
    with pytest.raises(ValueError, match="observation identity is invalid"):
        read_standardized_learning_run(tmp_path, updated)


def test_learning_target_path_must_stay_inside_the_store(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    escaped = _edit_manifest(
        manifest,
        stored,
        lambda raw: raw["records"][0]["targets"].update(relative_path="../escape.json"),
    )
    with pytest.raises(ValueError, match="target path must remain inside the store"):
        read_standardized_learning_run(tmp_path, escaped)


def test_learning_target_bytes_are_content_addressed(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    target = tmp_path / raw["records"][0]["targets"]["relative_path"]
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="target content hash is invalid"):
        read_standardized_learning_run(tmp_path, stored)


def test_learning_target_schema_must_be_known(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    reference = raw["records"][0]["targets"]
    target = tmp_path / reference["relative_path"]
    payload = json.loads(target.read_bytes())
    payload["schema"] = "standardized_targets.v9"
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    new_target = target.with_name(f"{digest}.json")
    new_target.write_bytes(encoded)
    reference["relative_path"] = new_target.relative_to(tmp_path).as_posix()
    reference["content_sha256"] = digest
    updated = _edit_manifest(manifest, stored, lambda body: body.update(records=raw["records"]))

    with pytest.raises(ValueError, match="unsupported standardized target schema"):
        read_standardized_learning_run(tmp_path, updated)


def test_legacy_target_schema_definitions_are_migrated_on_read(tmp_path: Path) -> None:
    stored, manifest = _write(tmp_path)

    def migrate_payload(body: dict[str, Any]) -> None:
        reference = body["records"][0]["targets"]
        target_path = tmp_path / reference["relative_path"]
        payload = json.loads(target_path.read_text(encoding="utf-8"))
        payload["schema"] = "standardized_targets.v1"
        for item in payload["targets"].values():
            definition = item["definition"]
            item["definition"] = {
                key: definition[key] for key in ("semantic", "units", "source", "reference")
            }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        migrated_path = target_path.with_name(f"{digest}.json")
        migrated_path.write_bytes(encoded)
        reference["relative_path"] = migrated_path.relative_to(tmp_path).as_posix()
        reference["content_sha256"] = digest

    updated = _edit_manifest(manifest, stored, migrate_payload)
    restored = read_standardized_learning_run(tmp_path, updated)
    migrated_definition = next(iter(restored.examples[0].targets.values())).definition

    assert migrated_definition.definition_version == "migrated_from_legacy_target_v1"
    assert migrated_definition.aggregation_rule == "not recorded by legacy target schema v1"


def test_learning_sample_order_must_match_the_pointer(tmp_path: Path) -> None:
    stored, _ = _write(tmp_path)
    reordered = dataclasses.replace(stored, sample_ids=tuple(reversed(stored.sample_ids)))

    with pytest.raises(ValueError, match="learning sample order disagrees with run pointer"):
        read_standardized_learning_run(tmp_path, reordered)
