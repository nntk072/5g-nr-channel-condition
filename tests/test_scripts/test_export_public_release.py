from __future__ import annotations

import json

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
