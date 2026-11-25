from __future__ import annotations

import json

import pytest
from scripts import export_public_release


def test_public_release_cli_forwards_external_approval_inputs(monkeypatch, tmp_path, capsys):
    source = tmp_path / "source"
    manifest = source / "release.json"
    archive = tmp_path / "release.zip"
    signature = source / "release.json.sig"
    allowed_signers = tmp_path / "reviewers.allowed_signers"
    expected = tmp_path / "published.zip"
    calls = []

    def fake_export(source_root, manifest_path, archive_path, **kwargs):
        calls.append((source_root, manifest_path, archive_path, kwargs))
        return expected

    monkeypatch.setattr(export_public_release, "export_public_release", fake_export)
    result = export_public_release.main(
        [
            str(source),
            str(manifest),
            str(archive),
            "--signature",
            str(signature),
            "--trusted-allowed-signers",
            str(allowed_signers),
            "--signer-id",
            "reviewer@example.org",
        ]
    )

    assert result == 0
    assert calls == [
        (
            source,
            manifest,
            archive,
            {
                "manifest_signature_path": signature,
                "trusted_allowed_signers_path": allowed_signers,
                "signer_id": "reviewer@example.org",
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {"archive": str(expected)}


def test_public_release_cli_reports_export_errors(tmp_path, monkeypatch, capsys) -> None:
    def fail_export(*_args, **_kwargs):
        raise ValueError("signature verification failed")

    monkeypatch.setattr(export_public_release, "export_public_release", fail_export)
    args = [
        str(tmp_path / "source"),
        str(tmp_path / "release.json"),
        str(tmp_path / "release.zip"),
        "--signature",
        str(tmp_path / "release.sig"),
        "--trusted-allowed-signers",
        str(tmp_path / "reviewers.allowed_signers"),
        "--signer-id",
        "reviewer@example.org",
    ]

    with pytest.raises(SystemExit) as error:
        export_public_release.main(args)

    assert error.value.code == 2
    assert "signature verification failed" in capsys.readouterr().err
