"""Episode-store reader and layer-payload validation edges."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import datasets.episode_store.storage as storage
import numpy as np
import pytest
from datasets.episode_store import write_episode_bundle
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from tests.test_datasets.test_counterfactual import _paths
from tests.test_datasets.test_umcd_store_integrity import _episode_bundle
from tests.test_datasets.umcd_schema_support import _bundle, _dynamic_episode


def test_object_reference_requires_canonical_internal_content_address(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must remain inside the store"):
        storage._episode_object_size_estimate(
            tmp_path, {"path": "../outside.npz", "sha256": "0" * 64}, max_shard_bytes=100
        )

    with pytest.raises(ValueError, match="must be content-addressed"):
        storage._episode_object_size_estimate(
            tmp_path, {"path": "different.npz", "sha256": "0" * 64}, max_shard_bytes=100
        )


def test_object_reference_rejects_symlink_resolving_outside_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"placeholder")
    link = store / f"{'a' * 64}.npz"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")

    with pytest.raises(ValueError, match="resolves outside the store"):
        storage._episode_object_size_estimate(
            store, {"path": link.name, "sha256": "a" * 64}, max_shard_bytes=100
        )


def test_stable_file_reader_detects_path_replacement_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stable.json"
    path.write_bytes(b"same-size")
    real_read = storage.os.read
    real_fstat = storage.os.fstat
    replaced = False
    first_stat = None

    def replace_after_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        content = real_read(descriptor, size)
        if content and not replaced:
            replaced = True
            replacement = tmp_path / "replacement.json"
            replacement.write_bytes(b"same-size")
            replacement.replace(path)
        return content

    def preserve_open_file_identity(descriptor: int):
        nonlocal first_stat
        current = real_fstat(descriptor)
        if first_stat is None:
            first_stat = current
        elif replaced:
            return first_stat
        return current

    monkeypatch.setattr(storage.os, "read", replace_after_read)
    monkeypatch.setattr(storage.os, "fstat", preserve_open_file_identity)
    with pytest.raises(ValueError, match="path changed while being read"):
        storage._stable_file_bytes(path, label="fixture")


def test_manifest_iteration_skips_non_json_store_files(tmp_path: Path) -> None:
    (tmp_path / "object.npz").write_bytes(b"object")
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")

    assert list(storage._iter_episode_manifest_paths(tmp_path)) == [tmp_path / "manifest.json"]


def test_manifest_reader_accepts_relative_and_absolute_manifest_paths(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored = write_episode_bundle(store, _episode_bundle())
    absolute = store / stored.relative_path

    assert (
        storage._validated_manifest(store, stored.relative_path)[0]["episode_id"]
        == stored.episode_id
    )
    assert storage._validated_manifest(store, absolute)[0]["episode_id"] == stored.episode_id


def test_manifest_reader_rejects_stale_pointer_and_non_list_layers(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored = write_episode_bundle(store, _episode_bundle())
    manifest_path = store / stored.relative_path
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    core = {key: document[key] for key in ("schema", "schema_version", "episode_id", "layers")}

    document["episode_id"] = "different-episode"
    core["episode_id"] = document["episode_id"]
    document["bundle_sha256"] = hashlib.sha256(storage._json_bytes(core)).hexdigest()
    manifest_path.write_bytes(storage._json_bytes(document))
    with pytest.raises(ValueError, match="pointer does not match its manifest"):
        storage._validated_manifest(store, stored)

    document["layers"] = {"bad": "shape"}
    core["layers"] = document["layers"]
    document["bundle_sha256"] = hashlib.sha256(storage._json_bytes(core)).hexdigest()
    manifest_path.write_bytes(storage._json_bytes(document))
    with pytest.raises(ValueError, match="layers must be a list"):
        storage._validated_manifest(store, stored.relative_path)


def test_object_reader_rejects_escaping_paths_and_duplicate_metadata_members(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must remain inside the episode store"):
        storage._read_object(tmp_path, {"path": "../escape.npz", "sha256": "0" * 64})

    external = tmp_path.parent / f"{tmp_path.name}-external.npz"
    external.write_bytes(b"external")
    link = tmp_path / "escape.npz"
    try:
        link.symlink_to(external)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    with pytest.raises(ValueError, match="resolves outside the store"):
        storage._read_object(tmp_path, {"path": link.name, "sha256": "0" * 64})

    link.unlink()
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        for index in range(2):
            encoded = io.BytesIO()
            np.save(encoded, np.frombuffer(storage._json_bytes({"index": index}), dtype=np.uint8))
            if index:
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("metadata_json.npy", encoded.getvalue())
            else:
                archive.writestr("metadata_json.npy", encoded.getvalue())
    payload = archive_bytes.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / f"{digest}.npz"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="unique names and one metadata entry"):
        storage._read_object(tmp_path, {"path": path.name, "sha256": digest})


def test_channel_and_target_payloads_refuse_unpublishable_contracts() -> None:
    channel = dataclasses.replace(_bundle().channel_views[0])
    object.__setattr__(channel, "valid_mask", None)
    with pytest.raises(ValueError, match="missing its validity mask"):
        storage._channel_payload(channel)

    legacy = SimpleNamespace(
        definition=SimpleNamespace(definition_version="umcd.rt-trajectory.geometric-los.v1")
    )
    with pytest.raises(ValueError, match="read-only for canonical UMCD publication"):
        storage._target_payload(legacy)


def test_path_payload_omits_absent_optional_columns() -> None:
    payload = storage._path_payload(_paths())
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert "metadata_json" in archive.files
        assert "doppler_hz" not in archive.files
        assert "interaction_vertices_m" not in archive.files


def test_path_payload_and_layer_validator_bind_present_optional_columns() -> None:
    paths = _bundle().path_oracle
    assert paths is not None
    paths = dataclasses.replace(paths, doppler_hz=np.zeros(len(paths.delay_s)))
    payload = storage._path_payload(paths)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        array_names = set(archive.files) - {"metadata_json"}
    storage._validate_path_oracle_layer(metadata, array_names)


def test_frequency_domain_observation_payload_and_reader_guards() -> None:
    inputs = FrequencyDomainModelInputs(
        observed_cfr=np.ones((1, 1, 3), dtype=np.complex128),
        noise_variance=0.01,
        snr_db=20.0,
        carrier_hz=3.5e9,
        subcarrier_offsets_hz=np.asarray([-30_000.0, 0.0, 30_000.0]),
    )
    binding = LinkedFrequencyDomainObservation(
        "link", 0, FrequencyDomainObservation("frequency-sample", inputs)
    )
    payload = storage._frequency_receiver_payload(binding)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata_json", "observed_cfr", "subcarrier_offsets_hz"}

    with pytest.raises(ValueError, match="unsupported frequency receiver-observation schema"):
        storage._read_frequency_receiver({"schema": "wrong"}, {})
    metadata = {
        "schema": "umcd.receiver_frequency_observation.v1",
        "measurement_semantics": "wrong",
        "waveform_model": "direct_noisy_cfr_not_a_pusch_dmrs_observation",
    }
    with pytest.raises(ValueError, match="measurement semantics"):
        storage._read_frequency_receiver(metadata, {})
    metadata["measurement_semantics"] = "complex_frequency_response_plus_awgn"
    metadata["waveform_model"] = "wrong"
    with pytest.raises(ValueError, match="waveform model"):
        storage._read_frequency_receiver(metadata, {})

    metadata["waveform_model"] = "direct_noisy_cfr_not_a_pusch_dmrs_observation"
    metadata.update(
        link_id="link",
        frame_index=0,
        sample_id="frequency-sample",
        noise_variance=inputs.noise_variance,
        snr_db=inputs.snr_db,
        carrier_hz=inputs.carrier_hz,
    )
    restored = storage._read_frequency_receiver(
        metadata,
        {
            "observed_cfr": inputs.observed_cfr,
            "subcarrier_offsets_hz": inputs.subcarrier_offsets_hz,
        },
    )
    assert restored.observation.sample_id == "frequency-sample"


def test_nr_receiver_reader_rejects_wrong_layer_schema() -> None:
    with pytest.raises(ValueError, match="unsupported receiver-observation layer schema"):
        storage._read_receiver({"schema": "wrong"}, {})


def test_episode_reader_reconstructs_dynamic_pose_and_rejects_bad_split_relations() -> None:
    payload = storage._episode_payload(_dynamic_episode())
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata_json"].tobytes())
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json"
        }

    restored = storage._read_episode(metadata, arrays)
    assert restored.pose_timeline is not None
    metadata["split_leakage_relations"] = [{"namespace": "missing_identity"}]
    with pytest.raises(ValueError, match="split_leakage_relations must be a list"):
        storage._read_episode(metadata, arrays)
