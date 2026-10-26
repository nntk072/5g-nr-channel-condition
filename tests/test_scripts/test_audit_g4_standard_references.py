from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_g4_standard_references.py"
_SPEC = importlib.util.spec_from_file_location("audit_g4_standard_references", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_g4_standard_references = _MODULE.audit_g4_standard_references
main = _MODULE.main


def _manifest_for(reference: bytes) -> dict[str, object]:
    return {
        "schema": "loslab.g4_standard_reference_manifest.v1",
        "snapshot_date": "2026-10-26",
        "redistribution_policy": "reference_identity_only; do_not_redistribute_etsi_pdf",
        "sources": [
            {
                "source_id": "tr-138901-v19.2.0",
                "standard": "3GPP TR 38.901",
                "version": "19.2.0",
                "filename": "tr_138901v190200p.pdf",
                "source_url": "https://www.etsi.org/deliver/reference.pdf",
                "sha256": hashlib.sha256(reference).hexdigest(),
                "size_bytes": len(reference),
                "reference_scope": ["CDL parameters"],
                "limitations": "Identity only; no conformance claim.",
            }
        ],
        "status": "SOURCE_IDENTITIES_PINNED_NOT_QUALIFIED",
    }


def _write_manifest(path: Path, manifest: object) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _fixture(tmp_path: Path, content: bytes = b"reference fixture") -> tuple[Path, Path]:
    root = tmp_path / "references"
    root.mkdir()
    (root / "tr_138901v190200p.pdf").write_bytes(content)
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, _manifest_for(content))
    return manifest, root


def test_exact_reference_bytes_pass_but_report_remains_nonqualifying(tmp_path: Path) -> None:
    manifest, root = _fixture(tmp_path)

    report = audit_g4_standard_references(manifest, root)

    assert report["status"] == "PASS_REFERENCE_SOURCE_BYTES_ONLY"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["source_count"] == 1
    assert report["sources"][0]["status"] == "PASS_EXACT_REFERENCE_BYTES"
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "g4-standard-reference-audit.v1.schema.json"
    )
    jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    ).validate(report)


def test_reference_tampering_and_missing_bytes_are_rejected(tmp_path: Path) -> None:
    manifest, root = _fixture(tmp_path)
    (root / "tr_138901v190200p.pdf").write_bytes(b"Reference fixture")

    try:
        audit_g4_standard_references(manifest, root)
    except ValueError as error:
        assert "do not match" in str(error)
    else:
        raise AssertionError("tampered reference should fail byte-count admission")

    (root / "tr_138901v190200p.pdf").write_bytes(b"reference fixture")
    (root / "tr_138901v190200p.pdf").unlink()
    try:
        audit_g4_standard_references(manifest, root)
    except OSError:
        pass
    else:
        raise AssertionError("missing reference should fail closed")


def test_reference_symlink_and_hardlink_are_rejected(tmp_path: Path) -> None:
    manifest, root = _fixture(tmp_path)
    target = tmp_path / "target.pdf"
    target.write_bytes(b"reference fixture")
    reference = root / "tr_138901v190200p.pdf"
    reference.unlink()
    reference.symlink_to(target)
    try:
        audit_g4_standard_references(manifest, root)
    except OSError:
        pass
    else:
        raise AssertionError("symlink reference should fail closed")

    reference.unlink()
    os.link(target, reference)
    try:
        audit_g4_standard_references(manifest, root)
    except ValueError as error:
        assert "single-link" in str(error)
    else:
        raise AssertionError("hard-linked reference should fail closed")


def test_manifest_rejects_duplicate_keys_and_unknown_fields(tmp_path: Path) -> None:
    _, root = _fixture(tmp_path)
    manifest = tmp_path / "duplicate.json"
    manifest.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
    try:
        audit_g4_standard_references(manifest, root)
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate keys should fail closed")

    valid = _manifest_for(b"reference fixture")
    valid["unexpected"] = True
    _write_manifest(manifest, valid)
    try:
        audit_g4_standard_references(manifest, root)
    except ValueError as error:
        assert "field roster" in str(error)
    else:
        raise AssertionError("unknown manifest fields should fail closed")


def test_manifest_symlink_and_oversize_are_rejected(tmp_path: Path, monkeypatch: object) -> None:
    _, root = _fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    target = tmp_path / "manifest-target.json"
    _write_manifest(target, _manifest_for(b"reference fixture"))
    manifest.unlink()
    manifest.symlink_to(target)
    try:
        audit_g4_standard_references(manifest, root)
    except OSError:
        pass
    else:
        raise AssertionError("manifest symlink should fail closed")

    manifest.unlink()
    manifest.write_bytes(b"{}" * 32)
    monkeypatch.setattr(_MODULE, "_MAX_MANIFEST_BYTES", 16)  # type: ignore[attr-defined]
    try:
        audit_g4_standard_references(manifest, root)
    except ValueError as error:
        assert "limit" in str(error)
    else:
        raise AssertionError("oversized manifest should fail closed")


def test_cli_emits_the_nonqualifying_source_identity_report(tmp_path: Path, capsys: object) -> None:
    manifest, root = _fixture(tmp_path)

    assert main([str(manifest), str(root)]) == 0
    output = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert output["qualification_status"] == "NOT_QUALIFIED"
    assert output["source_count"] == 1
