"""CDL reference bytes: frozen digest, source contract, and real-scalar inputs."""

from __future__ import annotations

import hashlib
import json

import pytest
from evaluation.covariance.cdl import shared


def test_reference_bytes_must_match_the_frozen_digest(tmp_path, monkeypatch) -> None:
    reference = tmp_path / "reference.json"
    reference.write_bytes(b"{}")
    monkeypatch.setattr(shared, "_REFERENCE_PATH", reference)
    with pytest.raises(ValueError, match="do not match their frozen digest"):
        shared._reference_document()


def test_reference_source_contract_is_checked_after_the_digest(tmp_path, monkeypatch) -> None:
    payload = json.dumps({"schema": "other", "source_id": "x", "source_sha256": "0" * 64}).encode()
    reference = tmp_path / "reference.json"
    reference.write_bytes(payload)
    monkeypatch.setattr(shared, "_REFERENCE_PATH", reference)
    monkeypatch.setattr(shared, "_REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())
    with pytest.raises(ValueError, match="source contract is invalid"):
        shared._reference_document()


def test_finite_real_accepts_numbers_and_refuses_others() -> None:
    assert shared._finite_real(2, label="carrier") == 2.0
    with pytest.raises(TypeError, match="carrier must be a real scalar"):
        shared._finite_real(True, label="carrier")
    with pytest.raises(TypeError, match="must be a real scalar"):
        shared._finite_real("3.5", label="carrier")
    with pytest.raises(ValueError, match="must be finite"):
        shared._finite_real(float("inf"), label="carrier")
