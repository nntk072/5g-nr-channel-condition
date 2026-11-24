"""CDL covariance references: only specular and diffuse components are accepted."""

from __future__ import annotations

import copy

import pytest
from evaluation.covariance.cdl import joint_tx_rx, spatiotemporal
from evaluation.covariance.cdl.shared import _reference_document


def _document_with_component(kind: str) -> dict:
    document = copy.deepcopy(_reference_document())
    profile = next(row for row in document["profiles"] if row["profile_model"] == "A")
    profile["components"][0]["component"] = kind
    return document


@pytest.mark.parametrize("kind", ["multipath", "unknown"])
def test_spatiotemporal_reference_rejects_unknown_component_kinds(monkeypatch, kind) -> None:
    monkeypatch.setattr(
        spatiotemporal, "_reference_document", lambda: _document_with_component(kind)
    )
    with pytest.raises(ValueError, match="unsupported CDL component kind"):
        spatiotemporal.cdl_spatiotemporal_covariance_reference("A", (0.0, 0.0, 0.0))


def test_joint_tx_rx_reference_rejects_unknown_component_kinds(monkeypatch) -> None:
    monkeypatch.setattr(
        joint_tx_rx, "_reference_document", lambda: _document_with_component("unknown")
    )
    with pytest.raises(ValueError, match="unsupported CDL component kind"):
        joint_tx_rx.cdl_joint_tx_rx_spatiotemporal_covariance_reference("A", (0.0, 0.0, 0.0))
