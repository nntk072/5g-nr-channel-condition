"""Channel split registry builder: input, output and stable-file guards on both build paths."""

from __future__ import annotations

import json
import os
import warnings
import zipfile
from pathlib import Path

import pytest
from datasets.split_policy import SplitPolicy
from scripts import build_channel_split_registry as builder
from scripts.build_channel_split_registry import (
    _parse_dataset,
    _stable_file_bytes,
    _stable_file_sha256,
    _validate_npz_size,
    _write_exclusive,
    build_candidate_split_registry,
    main,
)
from tests.test_scripts.test_build_channel_split_registry import _fixture


@pytest.fixture(params=["direct", "cached"])
def build_mode(request, tmp_path: Path):
    """Return keyword arguments selecting the direct or the content-addressed build path."""
    if request.param == "cached":
        return {"build_cache_root": tmp_path / "dag-cache"}
    return {}


def _build(datasets, policy_path, tmp_path, build_mode, **overrides):
    kwargs = {
        "policy_path": policy_path,
        "spatial_buffer_m": 0.0,
        "registry_path": tmp_path / "out" / "registry.json",
        "report_path": tmp_path / "out" / "report.json",
        **build_mode,
    }
    kwargs.update(overrides)
    return build_candidate_split_registry(datasets, **kwargs)


@pytest.mark.parametrize(
    ("datasets_factory", "message"),
    [
        (lambda paths: [], "at least one named ChannelDataset input is required"),
        (lambda paths: [("scene", paths[0]), ("scene", paths[1])], "duplicate dataset name: scene"),
        (
            lambda paths: [("a", paths[0]), ("b", paths[0])],
            "duplicate dataset path",
        ),
        (lambda paths: [("only-name",)], "datasets must contain \\(name, path\\) pairs"),
        (lambda paths: [(" padded", paths[0])], "dataset names must be nonempty canonical strings"),
    ],
    ids=["empty", "duplicate-name", "duplicate-path", "not-a-pair", "noncanonical-name"],
)
def test_dataset_roster_is_validated_before_any_read(
    tmp_path: Path, build_mode, datasets_factory, message
) -> None:
    datasets, policy_path = _fixture(tmp_path)
    paths = [path for _name, path in datasets[:2]]
    with pytest.raises(ValueError, match=message):
        _build(datasets_factory(paths), policy_path, tmp_path, build_mode)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"max_dataset_bytes": True}, TypeError, "max_dataset_bytes must be an integer"),
        ({"max_dataset_bytes": 0}, ValueError, "max_dataset_bytes must be positive"),
        (
            {"registry_path": "same.json", "report_path": "same.json"},
            ValueError,
            "must be different",
        ),
        ({"spatial_buffer_m": float("nan")}, ValueError, "spatial_buffer_m must be finite"),
    ],
    ids=["bool-limit", "zero-limit", "same-output", "nan-buffer"],
)
def test_build_options_are_rejected_on_both_paths(
    tmp_path: Path, build_mode, overrides, error, message
) -> None:
    datasets, policy_path = _fixture(tmp_path)
    if overrides.get("registry_path") == "same.json":
        overrides = {
            **overrides,
            "registry_path": tmp_path / "same.json",
            "report_path": tmp_path / "same.json",
        }
    with pytest.raises(error, match=message):
        _build(datasets, policy_path, tmp_path, build_mode, **overrides)


def test_existing_outputs_and_policy_reuse_are_refused(tmp_path: Path, build_mode) -> None:
    datasets, policy_path = _fixture(tmp_path)
    registry = tmp_path / "out" / "registry.json"
    registry.parent.mkdir()
    registry.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(datasets, policy_path, tmp_path, build_mode)

    registry.unlink()
    if "build_cache_root" in build_mode:
        message = "cannot also be a ChannelDataset input"
    else:
        message = "ChannelDataset 'policy' is not a valid NPZ archive"
    with pytest.raises(ValueError, match=message):
        _build([("policy", policy_path), *datasets[1:]], policy_path, tmp_path, build_mode)


@pytest.mark.parametrize(
    ("policy_text", "message"),
    [
        ('{"schema": 1, "schema": 2}', "strict UTF-8 JSON without duplicate keys"),
        ("[1, 2, 3]", "split policy must be a JSON object"),
        ("not json", "strict UTF-8 JSON without duplicate keys"),
    ],
    ids=["duplicate-key", "array", "invalid-json"],
)
def test_split_policy_must_be_a_strict_json_object(
    tmp_path: Path, build_mode, policy_text, message
) -> None:
    datasets, _policy_path = _fixture(tmp_path)
    bad_policy = tmp_path / "bad-policy.json"
    bad_policy.write_text(policy_text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _build(datasets, bad_policy, tmp_path, build_mode)


def test_split_policy_rejects_invalid_utf8_bytes(tmp_path: Path, build_mode) -> None:
    datasets, _policy_path = _fixture(tmp_path)
    bad_policy = tmp_path / "bad-utf8.json"
    bad_policy.write_bytes(b'{"schema": "\xff"}')
    with pytest.raises(ValueError, match="strict UTF-8 JSON without duplicate keys"):
        _build(datasets, bad_policy, tmp_path, build_mode)


def test_policy_and_inputs_must_be_single_link_regular_files(tmp_path: Path, build_mode) -> None:
    datasets, policy_path = _fixture(tmp_path)
    hard_link = tmp_path / "policy-hardlink.json"
    os.link(policy_path, hard_link)
    with pytest.raises(ValueError, match="split policy must be a single-link regular file"):
        _build(datasets, hard_link, tmp_path, build_mode)

    symlink = tmp_path / "policy-symlink.json"
    symlink.symlink_to(policy_path)
    with pytest.raises(ValueError, match="split policy must be a single-link regular file"):
        _build(datasets, symlink, tmp_path, build_mode)


def test_stable_file_readers_enforce_their_byte_limits(tmp_path: Path) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"0123456789")
    with pytest.raises(ValueError, match="payload exceeds the 4-byte limit"):
        _stable_file_bytes(payload, label="payload", max_bytes=4)
    with pytest.raises(ValueError, match="payload exceeds the 4-byte limit"):
        _stable_file_sha256(payload, label="payload", max_bytes=4)
    assert _stable_file_bytes(payload, label="payload", max_bytes=10) == b"0123456789"


def test_stable_file_readers_reject_drift_during_open_and_read(tmp_path: Path, monkeypatch) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"stable")
    original_identity = builder._stat_identity

    with monkeypatch.context() as patched:
        calls = 0

        def opening_drift(metadata):
            nonlocal calls
            calls += 1
            identity = original_identity(metadata)
            return identity if calls == 1 else (*identity[:-1], identity[-1] + 1)

        patched.setattr(builder, "_stat_identity", opening_drift)
        with pytest.raises(ValueError, match="changed while opening"):
            _stable_file_bytes(payload, label="payload", max_bytes=20)

    original_read = os.read
    changed = False

    def mutate_after_read(descriptor, size):
        nonlocal changed
        chunk = original_read(descriptor, size)
        if not changed:
            changed = True
            payload.write_bytes(b"changed")
        return chunk

    monkeypatch.setattr(os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while reading"):
        _stable_file_bytes(payload, label="payload", max_bytes=20)

    payload.write_bytes(b"stable")
    with monkeypatch.context() as patched:
        calls = 0

        def hashing_open_drift(metadata):
            nonlocal calls
            calls += 1
            identity = original_identity(metadata)
            return identity if calls == 1 else (*identity[:-1], identity[-1] + 1)

        patched.setattr(builder, "_stat_identity", hashing_open_drift)
        with pytest.raises(ValueError, match="changed while opening"):
            _stable_file_sha256(payload, label="payload", max_bytes=20)

    changed = False

    def mutate_during_hash(descriptor, size):
        nonlocal changed
        chunk = original_read(descriptor, size)
        if not changed:
            changed = True
            payload.write_bytes(b"changed")
        return chunk

    monkeypatch.setattr(os, "read", mutate_during_hash)
    with pytest.raises(ValueError, match="changed while hashing"):
        _stable_file_sha256(payload, label="payload", max_bytes=20)


def test_stable_file_readers_reject_overflow_during_streaming(tmp_path: Path, monkeypatch) -> None:
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"1234")
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"12345")
    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        _stable_file_bytes(payload, label="payload", max_bytes=4)
    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        _stable_file_sha256(payload, label="payload", max_bytes=4)


def test_stable_readers_refuse_non_regular_inputs(tmp_path: Path) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()
    with pytest.raises(ValueError, match="single-link regular file"):
        _stable_file_bytes(directory, label="folder", max_bytes=8)
    with pytest.raises(ValueError, match="single-link regular file"):
        _stable_file_sha256(directory, label="folder", max_bytes=8)


def _npz_with(entries: list[tuple[str, bytes]]) -> bytes:
    import io

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return buffer.getvalue()


def test_npz_archive_guards_reject_each_hostile_shape() -> None:
    with pytest.raises(ValueError, match="is not a valid NPZ archive"):
        _validate_npz_size(b"not a zip", label="dataset", max_bytes=1024)

    too_many = _npz_with([(f"x{i}.npy", b"") for i in range(33)])
    with pytest.raises(ValueError, match="contains too many NPZ entries"):
        _validate_npz_size(too_many, label="dataset", max_bytes=1 << 20)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        duplicated = _npz_with([("h.npy", b"a"), ("h.npy", b"b")])
    with pytest.raises(ValueError, match="contains duplicate NPZ entries"):
        _validate_npz_size(duplicated, label="dataset", max_bytes=1 << 20)

    large = _npz_with([("h.npy", b"x" * 64)])
    with pytest.raises(ValueError, match="exceeds the 16-byte uncompressed limit"):
        _validate_npz_size(large, label="dataset", max_bytes=16)
    _validate_npz_size(large, label="dataset", max_bytes=64)


def test_failed_exclusive_write_leaves_no_partial_artifact(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "nested" / "artifact.json"

    def failing_fsync(_descriptor):
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk full"):
        _write_exclusive(target, b"{}")
    assert not target.exists()

    monkeypatch.undo()
    _write_exclusive(target, b"{}")
    assert target.read_bytes() == b"{}"
    with pytest.raises(FileExistsError):
        _write_exclusive(target, b"{}")


@pytest.mark.parametrize("failure", [RuntimeError("freeze failed"), "digest-mismatch"])
def test_direct_builder_handles_registry_freeze_failures(
    tmp_path: Path, monkeypatch, failure
) -> None:
    datasets, policy_path = _fixture(tmp_path)

    def fail_freeze(*_args, **_kwargs):
        if isinstance(failure, Exception):
            raise failure
        return "0" * 64

    monkeypatch.setattr(builder, "freeze_split_policy_registry", fail_freeze)
    with pytest.raises((RuntimeError, ValueError)):
        _build(datasets, policy_path, tmp_path, {})
    assert not (tmp_path / "out" / "registry.json").exists()
    if failure == "digest-mismatch":
        assert (tmp_path / "out" / "report.json").exists()
    else:
        assert not (tmp_path / "out" / "report.json").exists()


def test_cached_builder_rejects_inputs_that_drift_during_node_build(
    tmp_path: Path, monkeypatch
) -> None:
    datasets, policy_path = _fixture(tmp_path)

    def drifted_build(_datasets, *, registry_path, report_path, **_kwargs):
        builder._write_exclusive(report_path, b"{}")
        builder._write_exclusive(registry_path, b"{}")
        return {
            "inputs": [{"name": "changed", "sha256": "0" * 64}],
            "policy_file_sha256": "1" * 64,
        }

    monkeypatch.setattr(builder, "build_candidate_split_registry", drifted_build)
    with pytest.raises(ValueError, match="changed after DAG identity was frozen"):
        _build(
            datasets,
            policy_path,
            tmp_path,
            {"build_cache_root": tmp_path / "cache"},
        )


def test_cached_builder_rejects_output_tampering_after_resolution(
    tmp_path: Path, monkeypatch
) -> None:
    datasets, policy_path = _fixture(tmp_path)
    original_resolve = builder.resolve_build_artifact

    def tamper_after_resolution(root, result, name):
        path = original_resolve(root, result, name)
        if name == "build-report.json":
            path.chmod(0o644)
            path.write_bytes(path.read_bytes() + b" ")
        return path

    monkeypatch.setattr(builder, "resolve_build_artifact", tamper_after_resolution)
    with pytest.raises(ValueError, match="changed after resolution"):
        _build(
            datasets,
            policy_path,
            tmp_path,
            {"build_cache_root": tmp_path / "cache"},
        )


def test_cached_builder_rejects_report_without_its_contract(tmp_path: Path, monkeypatch) -> None:
    import hashlib

    from artifacts.build_dag import BuildNodeResult

    datasets, policy_path = _fixture(tmp_path)
    report_path = tmp_path / "cached-report.json"
    registry_path = tmp_path / "cached-registry.json"
    report_path.write_bytes(b"{}")
    registry_path.write_bytes(b"{}")
    report_bytes = report_path.read_bytes()
    registry_bytes = registry_path.read_bytes()
    artifacts = (
        {
            "path": "build-report.json",
            "sha256": hashlib.sha256(report_bytes).hexdigest(),
            "size_bytes": len(report_bytes),
        },
        {
            "path": "split-registry.json",
            "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            "size_bytes": len(registry_bytes),
        },
    )
    monkeypatch.setattr(
        builder,
        "run_build_node",
        lambda *_args, **_kwargs: BuildNodeResult(
            "a" * 64, tmp_path / "unused-manifest.json", artifacts, True
        ),
    )
    monkeypatch.setattr(
        builder,
        "resolve_build_artifact",
        lambda _root, _result, name: {
            "build-report.json": report_path,
            "split-registry.json": registry_path,
        }[name],
    )
    with pytest.raises(ValueError, match="cached split-registry outputs do not match"):
        _build(
            datasets,
            policy_path,
            tmp_path,
            {"build_cache_root": tmp_path / "cache"},
        )


def test_cached_builder_removes_report_when_registry_publication_fails(
    tmp_path: Path, monkeypatch
) -> None:
    datasets, policy_path = _fixture(tmp_path)
    original_write = builder._write_exclusive
    calls = 0

    def fail_registry_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("registry publication failed")
        return original_write(path, payload)

    monkeypatch.setattr(builder, "_write_exclusive", fail_registry_write)
    with pytest.raises(OSError, match="registry publication failed"):
        _build(
            datasets,
            policy_path,
            tmp_path,
            {"build_cache_root": tmp_path / "cache"},
        )
    assert calls == 3
    assert not (tmp_path / "out" / "report.json").exists()
    assert not (tmp_path / "out" / "registry.json").exists()


def test_cached_builder_preserves_outputs_when_report_publication_fails(
    tmp_path: Path, monkeypatch
) -> None:
    datasets, policy_path = _fixture(tmp_path)
    original_write = builder._write_exclusive
    calls = 0

    def fail_report_write(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("report publication failed")
        return original_write(path, payload)

    monkeypatch.setattr(builder, "_write_exclusive", fail_report_write)
    with pytest.raises(OSError, match="report publication failed"):
        _build(
            datasets,
            policy_path,
            tmp_path,
            {"build_cache_root": tmp_path / "cache"},
        )
    assert calls == 2
    assert not (tmp_path / "out" / "report.json").exists()
    assert not (tmp_path / "out" / "registry.json").exists()


def test_parse_dataset_accepts_only_name_equals_path() -> None:
    assert _parse_dataset("scene=data/a.npz") == ("scene", Path("data/a.npz"))
    for value in ("scene", "=data/a.npz", "scene="):
        with pytest.raises(builder.argparse.ArgumentTypeError, match="NAME=PATH"):
            _parse_dataset(value)


def test_cli_builds_and_prints_a_summary(tmp_path: Path, monkeypatch, capsys) -> None:
    datasets, policy_path = _fixture(tmp_path)
    argv = ["build", "--policy", str(policy_path), "--spatial-buffer-m", "0"]
    for name, path in datasets:
        argv += ["--dataset", f"{name}={path}"]
    argv += [
        "--registry-output",
        str(tmp_path / "cli" / "registry.json"),
        "--report-output",
        str(tmp_path / "cli" / "report.json"),
        "--build-cache-root",
        str(tmp_path / "cli" / "cache"),
    ]

    assert main(argv[1:]) == 0

    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"status"' in printed and '"registry_file_sha256"' in printed
    assert '"build_node"' in printed
    assert (tmp_path / "cli" / "registry.json").exists()


def test_cli_omits_build_node_when_report_has_no_cached_node(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        builder,
        "build_candidate_split_registry",
        lambda *_args, **_kwargs: {
            "status": "PASS",
            "registry_sha256": "registry-digest",
            "registry_file_sha256": "file-digest",
        },
    )

    assert (
        main(
            [
                "--dataset",
                "scene=scene.npz",
                "--policy",
                "policy.json",
                "--spatial-buffer-m",
                "0",
                "--registry-output",
                str(tmp_path / "registry.json"),
                "--report-output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )

    summary = json.loads(capsys.readouterr().out)
    assert summary == {
        "status": "PASS",
        "registry_sha256": "registry-digest",
        "registry_file_sha256": "file-digest",
        "report": str(tmp_path / "report.json"),
    }


def test_cli_maps_build_errors_to_exit_status_two(tmp_path: Path, capsys) -> None:
    datasets, _policy_path = _fixture(tmp_path)
    argv = [
        "--policy",
        str(tmp_path / "missing-policy.json"),
        "--spatial-buffer-m",
        "0",
        "--registry-output",
        str(tmp_path / "r.json"),
        "--report-output",
        str(tmp_path / "p.json"),
    ]
    for name, path in datasets[:1]:
        argv += ["--dataset", f"{name}={path}"]

    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_policy_loader_returns_the_policy_and_its_file_digest(tmp_path: Path) -> None:
    policy = SplitPolicy.release_v2(seed="guard-check")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy.to_dict()), encoding="utf-8")
    loaded, digest = builder._load_policy(path)
    assert loaded.to_dict() == policy.to_dict()
    assert len(digest) == 64
