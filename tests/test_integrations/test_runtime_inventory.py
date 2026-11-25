from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from integrations import runtime_compatibility, runtime_inventory


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

    assert report["schema"] == "umcd.runtime_inventory.v2"
    assert report["status"] == "SELF_REPORTED_INTEGRITY_ONLY"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["api_compatibility"]["status"] == "NOT_PROBED"
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
    with pytest.raises(TypeError, match="probe_loaded_native_libraries"):
        runtime_inventory.build_runtime_inventory(
            tmp_path,
            probe_loaded_native_libraries=1,  # type: ignore[arg-type]
        )


def test_loaded_native_library_inventory_hashes_mapped_files_and_reports_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "native library.so"
    library.write_bytes(b"mapped native bytes")
    maps = tmp_path / "maps"
    escaped = str(library).replace(" ", r"\040")
    mapping = f"7f000000-7f001000 r-xp 00000000 08:01 123 {escaped}\n"
    maps.write_text(mapping + mapping, encoding="utf-8")

    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["status"] == "COMPLETE"
    assert report["total_hashed_bytes"] == library.stat().st_size
    assert report["libraries"] == [
        {
            "path": str(library),
            "size_bytes": library.stat().st_size,
            "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        }
    ]

    monkeypatch.setattr(runtime_inventory, "_MAX_NATIVE_LIBRARY_BYTES", 3)
    limited = runtime_inventory._loaded_native_library_identity(maps)
    assert limited["status"] == "INCOMPLETE"
    assert limited["libraries"] == []
    assert limited["unhashed"][0]["reason"] == "per_library_byte_limit_exceeded"


def test_loaded_native_library_inventory_is_versioned_and_schema_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {
        "pyproject.toml": None,
        "uv.lock": None,
        "runtime_inventory_sources": {
            "integrations/runtime_inventory.py": None,
            "integrations/runtime_compatibility.py": None,
            "scripts/report_runtime_inventory.py": None,
        },
        "git_commit": None,
        "tracked_worktree_clean": None,
    }
    monkeypatch.setattr(runtime_inventory, "_source_identity", lambda _root: source)
    monkeypatch.setattr(
        runtime_inventory,
        "_distribution_identity",
        lambda name: {"name": name, "installed": False},
    )
    native = {
        "status": "COMPLETE",
        "capture_method": "linux_proc_self_maps_sha256",
        "reason": None,
        "limits": {
            "max_library_bytes": 1024 * 1024 * 1024,
            "max_total_bytes": 8 * 1024 * 1024 * 1024,
        },
        "libraries": [{"path": "/lib/libfixture.so", "size_bytes": 4, "sha256": "a" * 64}],
        "unhashed": [],
        "total_hashed_bytes": 4,
    }
    monkeypatch.setattr(runtime_inventory, "_loaded_native_library_identity", lambda: native)
    report = runtime_inventory.build_runtime_inventory(tmp_path, probe_loaded_native_libraries=True)

    assert report["schema"] == "umcd.runtime_inventory.v3"
    assert report["loaded_native_libraries"] == native
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/runtime-inventory.v3.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(report)


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


def _map_line(path: str) -> str:
    return f"7f000000-7f001000 r-xp 00000000 08:01 123 {path}\n"


def test_runtime_inventory_source_identity_handles_missing_files_and_git_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subprocess import CalledProcessError

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(CalledProcessError(1, "git")),
    )
    source = runtime_inventory._source_identity(tmp_path)
    assert source["pyproject.toml"] is None
    assert source["uv.lock"] is None
    assert source["git_commit"] is None
    assert source["tracked_worktree_clean"] is None
    assert all(value is None for value in source["runtime_inventory_sources"].values())


def test_runtime_inventory_source_identity_records_git_revision_and_cleanliness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = iter((SimpleNamespace(stdout="abc123\n"), SimpleNamespace(stdout="")))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: next(responses))
    source = runtime_inventory._source_identity(tmp_path)
    assert source["git_commit"] == "abc123"
    assert source["tracked_worktree_clean"] is True


def test_distribution_identity_reports_incomplete_metadata_and_missing_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Distribution:
        version = "1.0"
        metadata = {}
        files = [Path("demo-1.dist-info/RECORD"), Path("demo-1.dist-info/WHEEL")]

        @staticmethod
        def locate_file(entry: Path) -> Path:
            return tmp_path / entry

    monkeypatch.setattr(
        runtime_inventory.importlib.metadata, "distribution", lambda _name: Distribution()
    )
    report = runtime_inventory._distribution_identity("demo")
    assert report["name"] == "demo"
    assert report["identity_completeness"] == "INCOMPLETE"
    assert report["identity_files"] == {}
    assert report["model_table_count"] == 0


def test_native_inventory_reports_maps_unavailable_and_oversized(tmp_path, monkeypatch):
    missing = runtime_inventory._loaded_native_library_identity(tmp_path / "missing-maps")
    assert missing["status"] == "UNAVAILABLE"
    assert missing["reason"] == "process_maps_unavailable"

    maps = tmp_path / "large-maps"
    maps.write_bytes(b"x" * 5)
    monkeypatch.setattr(runtime_inventory, "_MAX_PROC_MAPS_BYTES", 4)
    oversized = runtime_inventory._loaded_native_library_identity(maps)
    assert oversized["status"] == "INCOMPLETE"
    assert oversized["reason"] == "process_maps_exceed_byte_limit"


def test_native_inventory_reports_unavailable_nonregular_and_total_limit_paths(
    tmp_path, monkeypatch
):
    unavailable = tmp_path / "gone library.so"
    nonregular = tmp_path / "pipe library.so"
    os.mkfifo(nonregular)
    maps = tmp_path / "maps"
    maps.write_text(_map_line(str(unavailable)) + _map_line(str(nonregular)))
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["status"] == "INCOMPLETE"
    assert {item["reason"] for item in report["unhashed"]} == {
        "mapped_file_unavailable",
        "mapped_file_not_regular",
    }

    first = tmp_path / "a.so"
    second = tmp_path / "b.so"
    first.write_bytes(b"1234")
    second.write_bytes(b"5678")
    maps.write_text(_map_line(str(first)) + _map_line(str(second)))
    monkeypatch.setattr(runtime_inventory, "_MAX_NATIVE_LIBRARY_TOTAL_BYTES", 6)
    total_limited = runtime_inventory._loaded_native_library_identity(maps)
    assert total_limited["status"] == "INCOMPLETE"
    assert total_limited["total_hashed_bytes"] == 4
    assert total_limited["unhashed"][0]["reason"] == "total_byte_limit_exceeded"


def test_native_inventory_skips_malformed_and_non_library_map_entries(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text("malformed\n7f000000-7f001000 r-xp 0 08:01 123 /tmp/file.txt\n")
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["status"] == "COMPLETE"
    assert report["libraries"] == []
    assert report["unhashed"] == []


def test_identity_hash_enforces_file_size_descriptor_and_stability_guards(tmp_path, monkeypatch):
    path = tmp_path / "identity"
    path.write_bytes(b"a")
    with pytest.raises(ValueError, match="bounded size limit"):
        runtime_inventory._sha256(path, max_bytes=0)

    before = path.stat()
    monkeypatch.setattr(
        runtime_inventory.os,
        "fstat",
        lambda _descriptor: SimpleNamespace(
            st_mode=stat.S_IFREG | 0o600,
            st_nlink=before.st_nlink,
            st_size=before.st_size,
            st_dev=before.st_dev,
            st_ino=before.st_ino + 1,
        ),
    )
    with pytest.raises(ValueError, match="changed or violates its file contract"):
        runtime_inventory._sha256(path)
    monkeypatch.undo()

    real_read = runtime_inventory.os.read
    read_count = 0

    def oversized_read(descriptor, _size):
        nonlocal read_count
        read_count += 1
        return b"x" * 4 if read_count == 1 else b""

    monkeypatch.setattr(runtime_inventory.os, "read", oversized_read)
    with pytest.raises(ValueError, match="bounded size limit"):
        runtime_inventory._sha256(path, max_bytes=2)
    monkeypatch.setattr(runtime_inventory.os, "read", real_read)

    mutated = False

    def mutate_during_read(descriptor, size):
        nonlocal mutated
        data = real_read(descriptor, size)
        if data and not mutated:
            with path.open("ab") as stream:
                stream.write(b"changed")
            mutated = True
        return data

    monkeypatch.setattr(runtime_inventory.os, "read", mutate_during_read)
    with pytest.raises(ValueError, match="changed while hashing"):
        runtime_inventory._sha256(path)


def test_native_inventory_rejects_changed_file_snapshot_after_hash(tmp_path, monkeypatch):
    library = tmp_path / "changing.so"
    library.write_bytes(b"library")
    maps = tmp_path / "maps"
    maps.write_text(_map_line(str(library)))
    real_lstat = runtime_inventory.os.lstat
    calls = 0

    def changed_lstat(path):
        nonlocal calls
        result = real_lstat(path)
        if str(path) == str(library):
            calls += 1
            if calls == 4:
                return SimpleNamespace(
                    st_dev=result.st_dev,
                    st_ino=result.st_ino,
                    st_mode=result.st_mode,
                    st_nlink=result.st_nlink,
                    st_size=result.st_size + 1,
                    st_mtime_ns=result.st_mtime_ns,
                    st_ctime_ns=result.st_ctime_ns,
                )
        return result

    monkeypatch.setattr(runtime_inventory.os, "lstat", changed_lstat)
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["unhashed"][0]["reason"] == "mapped_file_changed_or_unhashable"


def test_native_inventory_marks_deleted_and_hash_race_mappings_unhashed(tmp_path, monkeypatch):
    library = tmp_path / "deleted.so"
    library.write_bytes(b"library")
    maps = tmp_path / "maps"
    maps.write_text(_map_line(f"{library} (deleted)"))
    library.unlink()
    report = runtime_inventory._loaded_native_library_identity(maps)
    assert report["unhashed"][0]["reason"] == "mapped_file_unavailable"

    library.write_bytes(b"library")
    maps.write_text(_map_line(str(library)))
    monkeypatch.setattr(
        runtime_inventory,
        "_sha256",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("fixture race")),
    )
    changed = runtime_inventory._loaded_native_library_identity(maps)
    assert changed["unhashed"][0]["reason"] == "mapped_file_changed_or_unhashable"


def test_accelerator_and_api_probes_are_opt_in_and_versioned(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert runtime_inventory._accelerator_identity(True) == {
        "probed": True,
        "status": "TORCH_UNAVAILABLE",
    }

    from integrations import runtime_compatibility

    monkeypatch.setattr(
        runtime_compatibility,
        "build_runtime_compatibility",
        lambda *, probe_rt: {"probe_rt": probe_rt},
    )
    report = runtime_inventory.build_runtime_inventory(
        tmp_path,
        probe_apis=True,
        probe_rt_apis=True,
    )
    assert report["api_compatibility"] == {"probe_rt": True}
    with pytest.raises(TypeError, match="API probe options"):
        runtime_inventory.build_runtime_inventory(tmp_path, probe_apis=1)
    with pytest.raises(ValueError, match="requires probe_apis"):
        runtime_inventory.build_runtime_inventory(tmp_path, probe_rt_apis=True)
    not_directory = tmp_path / "not-a-directory"
    not_directory.write_text("file")
    with pytest.raises(ValueError, match="must be a directory"):
        runtime_inventory.build_runtime_inventory(not_directory)


@pytest.mark.parametrize(
    ("available", "device_count", "expected_status"),
    [(False, 0, "CUDA_UNAVAILABLE"), (True, 1, "CUDA_AVAILABLE")],
)
def test_accelerator_probe_reports_cuda_and_device_identity(
    monkeypatch, available, device_count, expected_status
):
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "fixture-torch"
    fake_torch.version = SimpleNamespace(cuda="fixture-cuda")
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: device_count,
        get_device_properties=lambda _index: SimpleNamespace(
            name="fixture-gpu", major=8, minor=9, total_memory=1024
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = runtime_inventory._accelerator_identity(True)
    assert result["status"] == expected_status
    assert result["torch_version"] == "fixture-torch"
    assert len(result["devices"]) == device_count


def test_runtime_inventory_cli_writes_new_output_and_stdout(tmp_path, monkeypatch, capsys):
    report = {"status": "fixture"}
    monkeypatch.setattr(runtime_inventory, "build_runtime_inventory", lambda *_a, **_k: report)
    output = tmp_path / "inventory.json"
    assert runtime_inventory.main(["--repository-root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text()) == report

    monkeypatch.setattr(runtime_inventory.sys, "stdout", SimpleNamespace(buffer=SimpleNamespace(write=lambda data: captured.append(data))))
    captured = []
    assert runtime_inventory.main(["--repository-root", str(tmp_path)]) == 0
    assert json.loads(captured[0]) == report


def test_compatibility_matrix_checks_exact_distributions_and_exported_symbols() -> None:
    expected_symbols = {
        **runtime_compatibility._PHY_APIS,
        **runtime_compatibility._RT_APIS,
    }
    imported: list[str] = []

    def fake_importer(module_name: str) -> SimpleNamespace:
        imported.append(module_name)
        return SimpleNamespace(**{symbol: object() for symbol in expected_symbols[module_name]})

    expected_versions = {
        name: expected
        for requirements in runtime_compatibility._DISTRIBUTION_REQUIREMENTS.values()
        for name, expected in requirements
    }
    report = runtime_compatibility.build_runtime_compatibility(
        probe_rt=True,
        importer=fake_importer,
        version_provider=expected_versions.__getitem__,
    )

    assert report["status"] == "COMPATIBLE"
    assert report["phy"]["status"] == "COMPATIBLE"
    assert report["rt"]["status"] == "COMPATIBLE"
    assert set(imported) == set(expected_symbols)
    assert all(row["status"] == "AVAILABLE" for row in report["rt"]["modules"])


def test_compatibility_matrix_rejects_missing_exports_and_wrong_pinned_version() -> None:
    def importer(_module_name: str) -> SimpleNamespace:
        return SimpleNamespace()

    def version_provider(name: str) -> str:
        return "2.1.0" if name == "sionna-no-rt" else "2.2.0"

    report = runtime_compatibility.build_runtime_compatibility(
        importer=importer,
        version_provider=version_provider,
    )

    assert report["status"] == "PHY_INCOMPATIBLE"
    assert report["phy"]["status"] == "INCOMPATIBLE"
    assert report["phy"]["modules"][0]["status"] == "MISSING_SYMBOLS"
    assert report["rt"]["status"] == "NOT_PROBED"
