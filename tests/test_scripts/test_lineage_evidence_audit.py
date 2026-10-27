from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from datasets.lineage import LineageStatusClaim
from datasets.provenance import UMCDRunProvenance
from loslab.__main__ import main as loslab_main
from loslab.lineage_audit import audit_provenance_file


def _provenance(claims: tuple[LineageStatusClaim, ...]) -> UMCDRunProvenance:
    return UMCDRunProvenance(
        record_id="lineage-audit-test",
        subject_layer="oracle_paths",
        source_family="test-family",
        producer="lineage-audit-test",
        package_source_sha256="a" * 64,
        software_versions={"python": "3.12"},
        backend={"backend": "cpu"},
        scene_asset_sha256={},
        seed=1,
        physical_config={},
        solver_config={},
        material_config={},
        lineage_claims=claims,
    )


def _write_inputs(
    root: Path, provenance: UMCDRunProvenance, payload: bytes
) -> tuple[Path, Path, Path]:
    digest = hashlib.sha256(payload).hexdigest()
    provenance_path = root / "provenance.json"
    provenance_path.write_text(json.dumps(provenance.to_dict()), encoding="utf-8")
    path_map_path = root / "paths.json"
    path_map_path.write_text(
        json.dumps(
            {
                "schema": "umcd.lineage_evidence_path_map.v1",
                "paths": {digest: "reports/regression.json"},
            }
        ),
        encoding="utf-8",
    )
    evidence_root = root / "evidence"
    (evidence_root / "reports").mkdir(parents=True)
    (evidence_root / "reports/regression.json").write_bytes(payload)
    return provenance_path, path_map_path, evidence_root


def test_lineage_evidence_cli_workflow_binds_report_and_never_overwrites(tmp_path):
    payload = b'{"status":"tests-passed"}\n'
    digest = hashlib.sha256(payload).hexdigest()
    provenance = _provenance(
        (
            LineageStatusClaim(
                status="synthetic_debug",
                scope="debug-task",
                evidence_sha256={"regression_report_sha256": digest},
            ),
        )
    )
    provenance_path, path_map_path, evidence_root = _write_inputs(tmp_path, provenance, payload)
    output = tmp_path / "reports" / "audit.json"

    report = audit_provenance_file(provenance_path, path_map_path, evidence_root, output)

    assert report["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert report["scientific_status"] == "NOT_QUALIFIED"
    assert report["provenance_record_sha256"] == provenance.to_dict()["record_sha256"]
    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        audit_provenance_file(provenance_path, path_map_path, evidence_root, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_lineage_evidence_cli_rejects_tampered_payload_without_publishing_report(tmp_path):
    expected = b'{"status":"tests-passed"}\n'
    digest = hashlib.sha256(expected).hexdigest()
    provenance = _provenance(
        (
            LineageStatusClaim(
                status="synthetic_debug",
                scope="debug-task",
                evidence_sha256={"regression_report_sha256": digest},
            ),
        )
    )
    provenance_path, path_map_path, evidence_root = _write_inputs(tmp_path, provenance, b"tampered")
    path_map_path.write_text(
        json.dumps(
            {
                "schema": "umcd.lineage_evidence_path_map.v1",
                "paths": {digest: "reports/regression.json"},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "audit.json"

    with pytest.raises(ValueError, match="does not match digest"):
        audit_provenance_file(provenance_path, path_map_path, evidence_root, output)
    assert not output.exists()


def test_lineage_evidence_cli_rejects_provenance_without_v2_claims(tmp_path):
    provenance = _provenance(())
    legacy = UMCDRunProvenance(
        record_id=provenance.record_id,
        subject_layer=provenance.subject_layer,
        source_family=provenance.source_family,
        producer=provenance.producer,
        package_source_sha256=provenance.package_source_sha256,
        software_versions=provenance.software_versions,
        backend=provenance.backend,
        scene_asset_sha256=provenance.scene_asset_sha256,
        seed=provenance.seed,
        physical_config=provenance.physical_config,
        solver_config=provenance.solver_config,
        material_config=provenance.material_config,
    )
    provenance_path = tmp_path / "legacy.json"
    provenance_path.write_text(json.dumps(legacy.to_dict()), encoding="utf-8")
    path_map_path = tmp_path / "paths.json"
    path_map_path.write_text(
        json.dumps({"schema": "umcd.lineage_evidence_path_map.v1", "paths": {}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no lineage claim roster"):
        audit_provenance_file(provenance_path, path_map_path, tmp_path, tmp_path / "audit.json")


def test_loslab_command_exposes_lineage_evidence_audit(tmp_path, monkeypatch, capsys):
    payload = b"report\n"
    digest = hashlib.sha256(payload).hexdigest()
    provenance = _provenance(
        (
            LineageStatusClaim(
                status="synthetic_debug",
                scope="debug-task",
                evidence_sha256={"regression_report_sha256": digest},
            ),
        )
    )
    provenance_path, path_map_path, evidence_root = _write_inputs(tmp_path, provenance, payload)
    output = tmp_path / "cli-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loslab",
            "lineage-evidence-audit",
            "--provenance",
            str(provenance_path),
            "--evidence-path-map",
            str(path_map_path),
            "--evidence-root",
            str(evidence_root),
            "--output",
            str(output),
        ],
    )

    loslab_main()

    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "EVIDENCE_BYTES_VERIFIED"
    assert json.loads(output.read_text(encoding="utf-8")) == printed
