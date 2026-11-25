from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import scripts.materialize_benchmark_partitions as command


def _arguments(tmp_path: Path) -> dict[str, object]:
    paths = [tmp_path / f"input-{index}" for index in range(7)]
    return {
        "registry_path": paths[0],
        "registry_signature": paths[1],
        "registry_allowed_signers": paths[2],
        "registry_signer": "registry@example.com",
        "declaration_path": paths[3],
        "declaration_signature": paths[4],
        "declaration_allowed_signers": paths[5],
        "declaration_signer": "declaration@example.com",
        "datasets": [("zeta", paths[6]), ("alpha", tmp_path / "dataset-alpha")],
        "output": tmp_path / "manifest.json",
    }


def _fake_attestation(namespace: str, payload: bytes = b'{"schema":"fixture"}'):
    return SimpleNamespace(
        artifact_sha256="a" * 64,
        signature_sha256="b" * 64,
        allowed_signers_sha256="c" * 64,
        signer_id="fixture@example.com",
        namespace=namespace,
        payload=payload,
    )


def _stub_signed_inputs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    declaration_payload: bytes = b'{"schema":"fixture"}',
    manifest: dict[str, object] | None = None,
) -> None:
    monkeypatch.setattr(
        command,
        "verify_signed_split_policy_registry_attestation",
        lambda *args, **kwargs: ({"schema": "registry"}, _fake_attestation("registry")),
    )
    monkeypatch.setattr(
        command,
        "verify_openssh_signed_artifact",
        lambda *args, **kwargs: _fake_attestation("declaration", declaration_payload),
    )
    monkeypatch.setattr(command.jsonschema, "validate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(command, "_stable_file_bytes", lambda *args, **kwargs: b"dataset")
    monkeypatch.setattr(command, "_validate_npz_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(command.ChannelDataset, "load_bytes", lambda _payload: object())
    monkeypatch.setattr(
        command,
        "materialize_benchmark_manifest",
        lambda *_args, **_kwargs: {"status": "CANDIDATE"} if manifest is None else manifest,
    )


def test_build_manifest_verifies_inputs_and_publishes_exclusively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    registry = {"schema": "registry"}
    registry_attestation = _fake_attestation("registry-namespace")
    declaration_attestation = _fake_attestation("declaration-namespace")
    monkeypatch.setattr(
        command,
        "verify_signed_split_policy_registry_attestation",
        lambda *args, **kwargs: (registry, registry_attestation),
    )
    monkeypatch.setattr(
        command,
        "verify_openssh_signed_artifact",
        lambda *args, **kwargs: declaration_attestation,
    )
    monkeypatch.setattr(command, "_stable_file_bytes", lambda *args, **kwargs: b"dataset")
    monkeypatch.setattr(command, "_validate_npz_size", lambda *args, **kwargs: None)
    monkeypatch.setattr(command.ChannelDataset, "load_bytes", lambda _payload: object())
    monkeypatch.setattr(
        command,
        "materialize_benchmark_manifest",
        lambda _registry, declaration, datasets, **kwargs: {
            "status": "CANDIDATE_NOT_SCIENTIFICALLY_QUALIFIED",
            "declaration": declaration,
            "dataset_names": [name for name, _digest, _dataset in datasets],
        },
    )
    monkeypatch.setattr(command.jsonschema, "validate", lambda *_args, **_kwargs: None)

    manifest = command.build_benchmark_partition_manifest(**arguments)  # type: ignore[arg-type]

    assert manifest["dataset_names"] == ["alpha", "zeta"]
    assert arguments["output"].read_text().endswith("\n")  # type: ignore[union-attr]
    declaration_call = declaration_attestation
    assert declaration_call.namespace == "declaration-namespace"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        ({"max_dataset_bytes": True}, ValueError, "max_dataset_bytes must be a positive integer"),
        ({"max_dataset_bytes": 0}, ValueError, "max_dataset_bytes must be a positive integer"),
        ({"datasets": []}, ValueError, "at least one named dataset is required"),
    ],
)
def test_build_manifest_rejects_invalid_budget_or_empty_inputs(
    tmp_path: Path, overrides: dict[str, object], error: type[Exception], message: str
) -> None:
    arguments = _arguments(tmp_path)
    arguments.update(overrides)

    with pytest.raises(error, match=message):
        command.build_benchmark_partition_manifest(**arguments)  # type: ignore[arg-type]


def test_build_manifest_rejects_aliasing_inputs_and_existing_output(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments["datasets"] = [("one", arguments["registry_path"])]
    with pytest.raises(ValueError, match="input paths must be distinct"):
        command.build_benchmark_partition_manifest(**arguments)  # type: ignore[arg-type]

    arguments = _arguments(tmp_path)
    output = arguments["output"]
    output.touch()  # type: ignore[union-attr]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        command.build_benchmark_partition_manifest(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize("payload", [b"not JSON", b"\xff"])
def test_build_manifest_rejects_non_json_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> None:
    _stub_signed_inputs(monkeypatch, declaration_payload=payload)
    with pytest.raises(ValueError, match="not canonical UTF-8 JSON"):
        command.build_benchmark_partition_manifest(**_arguments(tmp_path))  # type: ignore[arg-type]


def test_build_manifest_rejects_oversized_declaration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_signed_inputs(
        monkeypatch,
        declaration_payload=b" " * (command._MAX_COHORT_DECLARATION_BYTES + 1),
    )
    with pytest.raises(ValueError, match="exceeds the 16-MiB limit"):
        command.build_benchmark_partition_manifest(**_arguments(tmp_path))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("validation_call", "message"),
    [
        (1, "cohort declaration violates its schema"),
        (2, "partition manifest violates its schema"),
    ],
)
def test_build_manifest_surfaces_schema_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_call: int,
    message: str,
) -> None:
    _stub_signed_inputs(monkeypatch)
    calls = 0

    def reject_schema(_instance: object, _schema: object) -> None:
        nonlocal calls
        calls += 1
        if calls == validation_call:
            raise command.jsonschema.ValidationError("fixture violation")

    monkeypatch.setattr(command.jsonschema, "validate", reject_schema)
    with pytest.raises(ValueError, match=message):
        command.build_benchmark_partition_manifest(**_arguments(tmp_path))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    ["dataset", "=path", "name=", "name=path=with=equals"],
)
def test_dataset_argument_requires_nonempty_name_and_path(value: str) -> None:
    if value == "name=path=with=equals":
        assert command._dataset_arg(value) == ("name", Path("path=with=equals"))
    else:
        with pytest.raises(command.argparse.ArgumentTypeError, match="NAME=PATH"):
            command._dataset_arg(value)


def test_main_prints_manifest_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        command,
        "build_benchmark_partition_manifest",
        lambda **_kwargs: {
            "status": "CANDIDATE",
            "manifest_sha256": "d" * 64,
            "complete_six_cohort_coverage": False,
            "cohorts": {"train": {"rows": 2}},
        },
    )
    result = command.main(
        [
            "--registry",
            "registry.json",
            "--registry-signature",
            "registry.sig",
            "--registry-allowed-signers",
            "registry.allowed",
            "--registry-signer",
            "registry@example.com",
            "--declaration",
            "declaration.json",
            "--declaration-signature",
            "declaration.sig",
            "--declaration-allowed-signers",
            "declaration.allowed",
            "--declaration-signer",
            "declaration@example.com",
            "--dataset",
            "train=data.npz",
            "--output",
            "out.json",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "CANDIDATE",
        "manifest_sha256": "d" * 64,
        "complete_six_cohort_coverage": False,
        "cohorts": {"train": {"rows": 2}},
    }


def test_main_converts_build_errors_to_cli_exit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        command,
        "build_benchmark_partition_manifest",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad signed input")),
    )
    with pytest.raises(SystemExit, match="2") as error:
        command.main(
            [
                "--registry",
                "r",
                "--registry-signature",
                "rs",
                "--registry-allowed-signers",
                "ra",
                "--registry-signer",
                "r@example.com",
                "--declaration",
                "d",
                "--declaration-signature",
                "ds",
                "--declaration-allowed-signers",
                "da",
                "--declaration-signer",
                "d@example.com",
                "--dataset",
                "train=data.npz",
                "--output",
                "out.json",
            ]
        )
    assert error.value.code == 2
    assert "bad signed input" in capsys.readouterr().err
