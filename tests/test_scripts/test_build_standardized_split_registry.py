from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from datasets.split_policy import (
    SplitPolicy,
    read_split_policy_registry,
)
from datasets.standardized import (
    StandardizedChannelRecord,
    StandardizedChannelRun,
    StandardizedFamily,
    write_standardized_run,
)
from scripts.build_standardized_split_registry import (
    _load_policy,
    _load_run,
    _parse_run,
    _read_regular_bytes,
    build_standardized_split_registry,
    main,
)


def _run(run_id: str, *, seed: int, draw_count: int = 8) -> StandardizedChannelRun:
    offsets = (np.arange(8) - 4) * 15_000.0
    h_path = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex64)
    template = StandardizedChannelRecord(
        sample_id=f"{run_id}-sample-0",
        run_id=run_id,
        sample_index=0,
        family=StandardizedFamily.TDL,
        profile_model="A",
        profile_los_component=False,
        h_path=h_path,
        tau_s=np.zeros((1, 1, 1), dtype=np.float64),
        cfr=np.broadcast_to(h_path[..., 0, None], (1, 1, 1, 1, 1, 8)).copy(),
        subcarrier_offsets_hz=offsets,
        metadata={
            "profile_model": "A",
            "profile_los_component": False,
            "delay_profile_semantics": "scalable_tr_38_901_profile",
            "delay_reference": "profile_excess_delay",
            "seed": seed,
            "spec_version": "19.2",
            "carrier_frequency_hz": 3.5e9,
            "subcarrier_spacing_hz": 15_000.0,
            "num_subcarriers": 8,
            "sampled_bandwidth_hz": 120_000.0,
            "num_time_steps": 1,
            "rx_antenna_count": 1,
            "tx_antenna_count": 1,
            "rx_corr_mat_real": None,
            "rx_corr_mat_imag": None,
        },
    )
    return StandardizedChannelRun(
        run_id,
        tuple(
            replace(
                template,
                sample_id=f"{run_id}-sample-{index:04d}",
                sample_index=index,
            )
            for index in range(draw_count)
        ),
    )


def _write_policy(path, *, seed: str) -> None:
    path.write_text(
        json.dumps(SplitPolicy.release_v2(seed=seed).to_dict(), sort_keys=True),
        encoding="utf-8",
    )


def test_standardized_split_builder_reads_immutable_runs_and_publishes_registry(tmp_path, capsys):
    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("standardized-batch", seed=101))
    manifest = store / stored.relative_path
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="stored-standardized-builder")
    output = tmp_path / "release-candidate"

    assert (
        main(
            [
                "--run",
                f"batch-a={manifest}",
                "--policy",
                str(policy),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)

    saved_report = json.loads((output / "build-report.json").read_text(encoding="utf-8"))
    assert saved_report["status"] == "CANDIDATE_NOT_EXTERNALLY_REVIEWED"
    assert saved_report["qualification_status"] == "NOT_QUALIFIED"
    assert saved_report["inputs"][0]["manifest_sha256"]
    assert saved_report["inputs"][0]["record_sha256"]
    assert cli_result["registry_sha256"] == saved_report["registry_sha256"]
    assert cli_result["qualification_status"] == "NOT_QUALIFIED"
    registry = read_split_policy_registry(output / "split-registry.json")
    assert len(registry.subjects) == 8
    assert set(registry.report()["splits"]) == {"train", "validation", "calibration", "test"}
    assert (
        saved_report["registry_file_sha256"]
        == hashlib.sha256((output / "split-registry.json").read_bytes()).hexdigest()
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_standardized_split_registry(
            [("batch-a", manifest)], policy_path=policy, output_dir=output
        )


def test_standardized_split_builder_rejects_reused_seed_index_without_outputs(tmp_path):
    store = tmp_path / "store"
    first = write_standardized_run(store, _run("run-a", seed=202, draw_count=1))
    second = write_standardized_run(store, _run("run-b", seed=202, draw_count=1))
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="reused-standardized-index")
    output = tmp_path / "rejected"

    with pytest.raises(ValueError, match="partition_id values must uniquely identify"):
        build_standardized_split_registry(
            [("first", store / first.relative_path), ("second", store / second.relative_path)],
            policy_path=policy,
            output_dir=output,
        )

    assert not output.exists()


def test_split_policy_loader_rejects_duplicate_keys_nonobjects_and_malformed_json(tmp_path):
    policy = tmp_path / "policy.json"
    for payload, message in (
        (b'{"schema": 1, "schema": 2}', "duplicate JSON object key"),
        (b"[]", "root must be an object"),
        (b"{", "not bounded canonical JSON"),
        (b"\xff", "not bounded canonical JSON"),
    ):
        policy.write_bytes(payload)
        with pytest.raises(ValueError, match=message):
            _load_policy(policy)


def test_regular_input_reader_rejects_links_and_size_overflow(tmp_path):
    target = tmp_path / "target.json"
    target.write_bytes(b"{}")
    hardlink = tmp_path / "hardlink.json"
    hardlink.hardlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        _read_regular_bytes(hardlink, label="policy")

    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        _read_regular_bytes(symlink, label="policy")

    hardlink.unlink()
    with pytest.raises(ValueError, match="1-byte limit"):
        _read_regular_bytes(target, label="policy", max_bytes=1)


def test_regular_input_reader_detects_file_identity_change_on_open(tmp_path, monkeypatch):
    import scripts.build_standardized_split_registry as builder

    target = tmp_path / "policy.json"
    target.write_bytes(b"{}")
    actual_fstat = os.fstat

    def changed_fstat(descriptor):
        current = actual_fstat(descriptor)
        return SimpleNamespace(
            st_dev=current.st_dev,
            st_ino=current.st_ino,
            st_size=current.st_size + 1,
            st_mtime_ns=current.st_mtime_ns,
        )

    monkeypatch.setattr(builder.os, "fstat", changed_fstat)
    with pytest.raises(ValueError, match="changed while opening"):
        _read_regular_bytes(target, label="policy")


def test_regular_input_reader_detects_concurrent_growth_and_replacement(tmp_path, monkeypatch):
    import scripts.build_standardized_split_registry as builder

    target = tmp_path / "policy.json"
    target.write_bytes(b"x")
    actual_read = os.read

    def grow_before_read(descriptor, count):
        target.write_bytes(b"expanded")
        monkeypatch.setattr(builder.os, "read", actual_read)
        return actual_read(descriptor, count)

    monkeypatch.setattr(builder.os, "read", grow_before_read)
    with pytest.raises(ValueError, match="1-byte limit"):
        _read_regular_bytes(target, label="policy", max_bytes=1)

    target.write_bytes(b"{}")
    actual_read = os.read
    changed = False

    def replace_during_read(descriptor, count):
        nonlocal changed
        chunk = actual_read(descriptor, count)
        if chunk and not changed:
            changed = True
            target.write_bytes(b'{"changed": true}')
        return chunk

    monkeypatch.setattr(builder.os, "read", replace_during_read)
    with pytest.raises(ValueError, match="changed while reading"):
        _read_regular_bytes(target, label="policy")


def test_run_loader_rejects_noncanonical_names_and_invalid_manifests(tmp_path):
    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("manifest-validation", seed=303, draw_count=1))
    manifest = store / stored.relative_path
    valid = json.loads(manifest.read_text(encoding="utf-8"))

    for name in ("", " padded"):
        with pytest.raises(ValueError, match="canonical nonempty"):
            _load_run(name, manifest)

    invalid_documents = (
        ({**valid, "extra": True}, "invalid field roster"),
        ({**valid, "schema": "unsupported"}, "unsupported schema"),
        ({**valid, "records": []}, "has no records"),
        ({**valid, "records": [{}]}, "invalid record reference"),
        ({**valid, "records": [{**valid["records"][0], "sample_id": 4}]}, "invalid sample roster"),
    )
    for document, message in invalid_documents:
        manifest.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            _load_run("valid-name", manifest)

    manifest.write_text('{"schema": "one", "schema": "two"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        _load_run("valid-name", manifest)

    manifest.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="is not valid JSON"):
        _load_run("valid-name", manifest)


def test_run_loader_detects_manifest_change_after_loading(tmp_path, monkeypatch):
    import scripts.build_standardized_split_registry as builder

    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("manifest-race", seed=707, draw_count=1))
    manifest = store / stored.relative_path
    actual_read = builder._read_regular_bytes
    reads = 0

    def changed_second_read(path, *, label, max_bytes=builder._MAX_INPUT_BYTES):
        nonlocal reads
        reads += 1
        payload = actual_read(path, label=label, max_bytes=max_bytes)
        return payload if reads == 1 else payload + b" "

    monkeypatch.setattr(builder, "_read_regular_bytes", changed_second_read)
    with pytest.raises(ValueError, match="changed while loading its records"):
        _load_run("batch", manifest)


def test_run_loader_rejects_manifest_outside_standardized_manifest_tree(tmp_path):
    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("manifest-location", seed=404, draw_count=1))
    valid_manifest = store / stored.relative_path
    outside = tmp_path / "manifest.json"
    outside.write_bytes(valid_manifest.read_bytes())

    with pytest.raises(ValueError, match=r"under manifests/runs/\*\.json"):
        _load_run("misplaced", outside)


def test_split_builder_rejects_empty_inputs_and_duplicate_names(tmp_path):
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="duplicate-names")
    output = tmp_path / "release"
    with pytest.raises(ValueError, match="at least one --run"):
        build_standardized_split_registry([], policy_path=policy, output_dir=output)

    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("duplicate-name-run", seed=505, draw_count=8))
    manifest = store / stored.relative_path
    with pytest.raises(ValueError, match="input names must be unique"):
        build_standardized_split_registry(
            [("same", manifest), ("same", manifest)],
            policy_path=policy,
            output_dir=output,
        )
    assert not output.exists()


def test_split_builder_cleans_staging_directory_when_registry_freeze_fails(tmp_path, monkeypatch):
    import scripts.build_standardized_split_registry as builder

    store = tmp_path / "store"
    stored = write_standardized_run(store, _run("staging-failure", seed=606, draw_count=8))
    manifest = store / stored.relative_path
    policy = tmp_path / "policy.json"
    _write_policy(policy, seed="staging-cleanup")
    output = tmp_path / "release"
    monkeypatch.setattr(builder, "freeze_split_policy_registry", lambda *_args: "wrong-digest")

    with pytest.raises(RuntimeError, match="frozen registry bytes differ"):
        build_standardized_split_registry(
            [("batch", manifest)], policy_path=policy, output_dir=output
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".release.staging-*"))


@pytest.mark.parametrize("value", ["missing-separator", "=manifest.json", "name="])
def test_run_argument_parser_rejects_incomplete_mapping(value):
    with pytest.raises(
        argparse.ArgumentTypeError, match="NAME=/STORE_ROOT/manifests/runs/RUN.json"
    ):
        _parse_run(value)


def test_split_builder_cli_reports_input_errors(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--run",
                "batch=/missing/store/manifests/runs/run.json",
                "--policy",
                str(tmp_path / "missing-policy.json"),
                "--output-dir",
                str(tmp_path / "release"),
            ]
        )

    assert exit_info.value.code == 2
    assert "error:" in capsys.readouterr().err
