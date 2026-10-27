from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from integrations import runtime_inventory


def test_runtime_inventory_hashes_lock_inputs_and_labels_claims_unqualified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "pyproject.toml"
    lock = tmp_path / "uv.lock"
    project.write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
    lock.write_text("version = 1\n", encoding="utf-8")

    def missing_distribution(_name: str) -> None:
        raise runtime_inventory.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(runtime_inventory.importlib.metadata, "distribution", missing_distribution)
    report = runtime_inventory.build_runtime_inventory(tmp_path)

    assert report["schema"] == "umcd.runtime_inventory.v1"
    assert report["status"] == "SELF_REPORTED_INTEGRITY_ONLY"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["source"]["pyproject.toml"] == hashlib.sha256(project.read_bytes()).hexdigest()
    assert report["source"]["uv.lock"] == hashlib.sha256(lock.read_bytes()).hexdigest()
    assert report["accelerator"] == {"probed": False, "status": "NOT_PROBED"}
    assert all(not item["installed"] for item in report["distributions"])


def test_sionna_inventory_binds_installed_model_table_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "site-packages"
    model = root / "sionna/phy/channel/tr38901/models/v19_2/TDL-A.json"
    record = root / "sionna_no_rt-2.2.0.dist-info/RECORD"
    wheel = root / "sionna_no_rt-2.2.0.dist-info/WHEEL"
    direct_url = root / "sionna_no_rt-2.2.0.dist-info/direct_url.json"
    for path, contents in (
        (model, b'{"profile":"TDL-A"}\n'),
        (record, b"TDL-A.json,sha256=fixture,10\n"),
        (wheel, b"Wheel-Version: 1.0\n"),
        (direct_url, b'{"url":"https://example.invalid/wheel.whl"}\n'),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)

    class Distribution:
        version = "2.2.0"
        metadata = {"Name": "sionna-no-rt"}
        files = [
            Path("sionna/phy/channel/tr38901/models/v19_2/TDL-A.json"),
            Path("sionna_no_rt-2.2.0.dist-info/RECORD"),
            Path("sionna_no_rt-2.2.0.dist-info/WHEEL"),
            Path("sionna_no_rt-2.2.0.dist-info/direct_url.json"),
        ]

        @staticmethod
        def locate_file(entry: Path) -> Path:
            return root / entry

    def get_distribution(name: str) -> Distribution:
        if name != "sionna-no-rt":
            raise runtime_inventory.importlib.metadata.PackageNotFoundError
        return Distribution()

    monkeypatch.setattr(runtime_inventory.importlib.metadata, "distribution", get_distribution)
    report = runtime_inventory._distribution_identity("sionna-no-rt")
    assert report["installed"] is True
    assert report["version"] == "2.2.0"
    assert report["identity_completeness"] == "COMPLETE"
    assert report["model_table_count"] == 1
    assert report["model_tables"] == [
        {
            "path": "sionna/phy/channel/tr38901/models/v19_2/TDL-A.json",
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    ]
    assert report["identity_files"] == {
        "RECORD": hashlib.sha256(record.read_bytes()).hexdigest(),
        "WHEEL": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "direct_url.json": hashlib.sha256(direct_url.read_bytes()).hexdigest(),
    }


def test_runtime_inventory_rejects_symlink_identity_files(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("bytes", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="regular file"):
        runtime_inventory._sha256(link)


def test_runtime_inventory_rejects_fifo_identity_files_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "identity.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ValueError, match="regular file"):
        runtime_inventory._sha256(fifo)


def test_runtime_inventory_requires_strict_probe_flag(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="Boolean"):
        runtime_inventory.build_runtime_inventory(tmp_path, probe_accelerator=1)  # type: ignore[arg-type]


def test_runtime_inventory_cli_does_not_replace_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "inventory.json"
    output.write_bytes(b"user-owned bytes")
    monkeypatch.setattr(
        runtime_inventory,
        "build_runtime_inventory",
        lambda *_args, **_kwargs: pytest.fail(
            "existing output should be rejected before inventory collection"
        ),
    )

    with pytest.raises(FileExistsError):
        runtime_inventory.main(["--repository-root", str(tmp_path), "--output", str(output)])
    assert output.read_bytes() == b"user-owned bytes"
