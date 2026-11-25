"""Runner feature helpers: CFR feature domains, digests, benchmark inputs and group statistics."""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.runner.features import (
    _array_sha256,
    _identity_complex_cfr,
    _kish_effective_group_count,
    _label_semantics_by_source,
    _log_power_per_link_subcarrier,
    _run_replay_rng_streams,
    _sparse_multipath_features,
    _stable_input_bytes,
    _stable_single_link_sha256,
    _sync_torch_device,
    _validate_evaluation_payload,
    _validate_json_schema,
    _validate_npz_snapshot,
)


def _cfr(links: int = 1, tones: int = 8, samples: int = 2) -> np.ndarray:
    rng = np.random.default_rng(4)
    return (
        rng.normal(size=(samples, links, tones)) + 1j * rng.normal(size=(samples, links, tones))
    ).astype(np.complex64)


def test_identity_and_log_power_features_keep_their_domains() -> None:
    cfr = _cfr()
    assert _identity_complex_cfr(cfr).shape == cfr.shape
    assert _log_power_per_link_subcarrier(cfr).shape == (2, 8)
    with pytest.raises(ValueError, match="power-only features require complex"):
        _log_power_per_link_subcarrier(np.ones((2, 8)))
    bad = _cfr()
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="power-only CFR input must be finite"):
        _log_power_per_link_subcarrier(bad)
    huge = np.full((1, 1, 8), 1e200 + 0j)
    with pytest.raises(ValueError, match="not representable in float64"):
        _log_power_per_link_subcarrier(huge)


def test_sparse_multipath_features_validate_their_domain() -> None:
    with pytest.raises(ValueError, match="sparse multipath features require complex"):
        _sparse_multipath_features(np.ones((2, 8)))
    with pytest.raises(ValueError, match="at least eight tones"):
        _sparse_multipath_features(_cfr(tones=7))
    bad = _cfr()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="sparse multipath CFR must be finite"):
        _sparse_multipath_features(bad)


def test_array_digest_is_layout_independent() -> None:
    array = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert _array_sha256(array) == _array_sha256(np.asfortranarray(array))


def test_single_link_model_digest_requires_a_regular_file(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    assert len(_stable_single_link_sha256(model)) == 64
    link = tmp_path / "linked.bin"
    link.symlink_to(model)
    with pytest.raises(ValueError, match="single-link regular file|changed|path"):
        _stable_single_link_sha256(link)


def test_kish_group_count_is_the_effective_sample_size() -> None:
    assert _kish_effective_group_count([4, 4, 4]) == pytest.approx(3.0)
    assert _kish_effective_group_count([10]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="group row counts must be positive"):
        _kish_effective_group_count([0, 4])


def test_stable_input_bytes_are_bounded(tmp_path: Path) -> None:
    source = tmp_path / "input.bin"
    source.write_bytes(b"0123456789")
    assert _stable_input_bytes(source) == b"0123456789"
    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        _stable_input_bytes(source, max_bytes=4)
    with pytest.raises(ValueError, match="single-link regular file"):
        _stable_input_bytes(tmp_path)


def test_npz_snapshot_rejects_bad_archives_and_oversized_members() -> None:
    with pytest.raises(ValueError, match="not a valid NPZ archive"):
        _validate_npz_snapshot(b"not a zip", label="dataset", max_uncompressed_bytes=10)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("a.npy", b"x" * 64)
    with pytest.raises(ValueError, match="exceeds the uncompressed dataset size limit"):
        _validate_npz_snapshot(buffer.getvalue(), label="dataset", max_uncompressed_bytes=10)
    crowded = BytesIO()
    with zipfile.ZipFile(crowded, "w") as archive:
        for index in range(33):
            archive.writestr(f"member{index}.npy", b"x")
    with pytest.raises(ValueError, match="invalid NPZ entry roster"):
        _validate_npz_snapshot(crowded.getvalue(), label="dataset", max_uncompressed_bytes=10_000)


def test_stable_hash_rejects_file_replaced_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.runner.features as features

    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    real_fstat = os.fstat

    def changed_inode(descriptor: int) -> SimpleNamespace:
        actual = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=actual.st_dev,
            st_ino=actual.st_ino + 1,
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
        )

    monkeypatch.setattr(features.os, "fstat", changed_inode)
    with pytest.raises(ValueError, match="path changed before opening"):
        _stable_single_link_sha256(model)


def test_stable_input_rejects_replacement_before_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.runner.features as features

    source = tmp_path / "input.bin"
    source.write_bytes(b"payload")
    real_fstat = os.fstat

    def changed_inode(descriptor: int) -> SimpleNamespace:
        actual = real_fstat(descriptor)
        return SimpleNamespace(st_dev=actual.st_dev + 1, st_ino=actual.st_ino)

    monkeypatch.setattr(features.os, "fstat", changed_inode)
    with pytest.raises(ValueError, match="changed before it was opened"):
        _stable_input_bytes(source)


def test_validation_helpers_report_schema_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    import jsonschema

    original_read_text = Path.read_text

    def read_schema(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "fixture.schema.json":
            return '{"type":"object"}'
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_schema)

    class InvalidValidator:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        @classmethod
        def check_schema(cls, _schema: object) -> None:
            return None

        def iter_errors(self, _payload: object) -> list[SimpleNamespace]:
            return [SimpleNamespace(message="missing field")]

        def validate(self, _payload: object) -> None:
            raise jsonschema.ValidationError("missing field")

    monkeypatch.setattr(jsonschema, "Draft202012Validator", InvalidValidator)
    with pytest.raises(ValueError, match="evaluation payload violates fixture.schema.json"):
        _validate_evaluation_payload({}, "fixture.schema.json")
    with pytest.raises(ValueError, match="feature payload violates its schema: missing field"):
        _validate_json_schema({}, {}, "feature payload")


def test_sync_torch_device_only_synchronizes_enabled_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    _sync_torch_device("cpu", True)
    _sync_torch_device("cuda:0", False)
    _sync_torch_device("cuda:0", True)
    assert calls == ["cuda:0"]


def test_replay_rng_streams_record_configured_controls() -> None:
    streams = _run_replay_rng_streams(
        {
            "seeds": [3],
            "prior_shift": {"enabled": True},
            "observation": "comb_ls",
            "observation_seed": 19,
            "phase_randomization_seed": 23,
            "methods": ["label_permuted", "row_shuffled"],
            "label_permutation_seed": 29,
            "row_shuffle_seed": 31,
        }
    )
    assert [stream["name"] for stream in streams] == [
        "run_seed_3",
        "calibration_partition_seed_3",
        "observation_noise",
        "phase_randomization_control",
        "label_permutation_base",
        "row_shuffle_base",
    ]


def test_label_semantics_reports_multiple_values_for_one_source() -> None:
    from datasets.channel import ChannelDataset

    dataset = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["g0", "g1"]),
        domain=np.asarray(["fixture", "fixture"]),
        source=np.asarray(["synthetic", "synthetic"]),
        metadata=(
            {"label_semantics": "profile_proxy"},
            {"label_semantics": "geometric_direct_path_oracle"},
        ),
        label_semantics="los_binary",
    )
    assert _label_semantics_by_source(dataset) == (
        "synthetic=geometric_direct_path_oracle|profile_proxy"
    )


def test_sparse_multipath_rejects_nonfinite_estimator_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import experiments.runner.features as features

    fit = SimpleNamespace(
        model_order=1,
        residual_sum_squares=0.0,
        gains=np.asarray([1.0 + 0j]),
        delays_s=np.asarray([np.nan]),
    )
    monkeypatch.setattr(features, "fit_sparse_multipath", lambda *_args, **_kwargs: fit)
    with pytest.raises(ValueError, match="estimator produced nonfinite features"):
        _sparse_multipath_features(_cfr(samples=1))


def test_stable_hash_rejects_nonregular_open_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.runner.features as features

    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    monkeypatch.setattr(
        features.os,
        "fstat",
        lambda _descriptor: SimpleNamespace(st_mode=0o040755, st_nlink=1),
    )
    with pytest.raises(ValueError, match="single-link regular file"):
        _stable_single_link_sha256(model)


def test_stable_hash_rejects_mutation_while_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.runner.features as features

    model = tmp_path / "model.bin"
    model.write_bytes(b"model")
    real_fstat = os.fstat
    calls = 0

    def mutate_second_snapshot(descriptor: int) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        actual = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_mode=actual.st_mode,
            st_nlink=actual.st_nlink,
            st_size=actual.st_size,
            st_mtime_ns=actual.st_mtime_ns + (1 if calls == 2 else 0),
            st_ctime_ns=actual.st_ctime_ns,
        )

    monkeypatch.setattr(features.os, "fstat", mutate_second_snapshot)
    with pytest.raises(ValueError, match="changed during hashing"):
        _stable_single_link_sha256(model)


def test_stable_input_rejects_read_that_exceeds_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.runner.features as features

    source = tmp_path / "input.bin"
    source.write_bytes(b"x")
    monkeypatch.setattr(features.os, "read", lambda _descriptor, _size: b"xxx")
    with pytest.raises(ValueError, match="changed or exceeded its byte limit"):
        _stable_input_bytes(source, max_bytes=2)
