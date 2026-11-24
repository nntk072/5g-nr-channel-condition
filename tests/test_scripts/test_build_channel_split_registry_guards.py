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
    ]

    assert main(argv[1:]) == 0

    printed = capsys.readouterr().out.strip().splitlines()[-1]
    assert '"status"' in printed and '"registry_file_sha256"' in printed
    assert (tmp_path / "cli" / "registry.json").exists()


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
