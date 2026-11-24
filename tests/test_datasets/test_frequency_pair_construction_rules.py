"""Paired frequency views: pairing contracts, provenance binding and content-addressed manifests."""

from __future__ import annotations

import dataclasses

import pytest
from datasets.frequency_pair import StoredPairedFrequencyManifest
from tests.test_datasets.test_frequency_pair import _pair, _provenance, _view


def _replace(**changes):
    return dataclasses.replace(_pair(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param(
            {"lower_view": _view("view-28", 3.5e9), "upper_view": _view("view-28", 28e9)},
            "distinct view IDs",
            id="same-view-id",
        ),
        pytest.param(
            {
                "upper_view": _view("view-28b", 3.5e9),
                "upper_provenance": _provenance("run-28b", 3.5e9),
            },
            "distinct carrier frequencies",
            id="same-carrier",
        ),
        pytest.param(
            {"upper_view": dataclasses.replace(_view("view-28", 28e9), link_id="gNB-2-to-UE-1")},
            "share link, time, port, delay, and power contracts",
            id="link",
        ),
        pytest.param(
            {"upper_view": dataclasses.replace(_view("view-28", 28e9), rx_port_ids=("UE-H",))},
            "share link, time, port, delay, and power contracts",
            id="rx-ports",
        ),
        pytest.param(
            {
                "lower_provenance": dataclasses.replace(
                    _provenance("run-3p5", 3.5e9), source_family="F04-CDL-STATIC"
                )
            },
            "require RT solve provenance",
            id="non-rt-family",
        ),
        pytest.param(
            {
                "upper_provenance": dataclasses.replace(
                    _provenance("run-28", 28e9), scene_asset_sha256={}
                )
            },
            "must bind source scene assets",
            id="unbound-assets",
        ),
        pytest.param(
            {
                "upper_provenance": dataclasses.replace(
                    _provenance("run-28", 28e9), scene_asset_sha256={"scene.xml": "d" * 64}
                )
            },
            "same source scene asset digests",
            id="different-assets",
        ),
        pytest.param(
            {"upper_provenance": _provenance("run-28", 28e9, geometry_sha256="d" * 64)},
            "does not bind the paired scene geometry",
            id="other-geometry",
        ),
        pytest.param(
            {"upper_provenance": _provenance("run-28", 3.5e9)},
            "does not match its run-provenance",
            id="provenance-carrier",
        ),
    ],
)
def test_pair_rejects_each_broken_pairing_contract(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _replace(**changes)


def test_pair_requires_typed_views_and_provenance() -> None:
    with pytest.raises(TypeError, match="PhysicalChannelView values"):
        _replace(upper_view={"view_id": "not-a-view"})
    with pytest.raises(TypeError, match="requires its UMCDRunProvenance"):
        _replace(lower_provenance={"record_id": "not-a-record"})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lower_bundle_sha256", "E" * 64),
        ("upper_bundle_sha256", "not-a-digest"),
    ],
    ids=["lower-case", "short"],
)
def test_bundle_digests_are_lowercase_hex(field, value) -> None:
    with pytest.raises(ValueError, match="must be a lowercase SHA-256 digest"):
        _replace(**{field: value})


def test_pair_identity_is_stable_and_bound_to_its_views() -> None:
    first = _pair().to_dict()
    second = _pair().to_dict()
    assert first == second
    changed = _replace(
        upper_view=_view("view-28-alt", 28e9),
        upper_provenance=_provenance("run-28-alt", 28e9),
    ).to_dict()
    assert changed != first


def test_stored_manifest_pointer_must_be_content_addressed() -> None:
    digest = "ab" * 32
    good = f"manifests/paired_frequency_views/ab/{digest}.json"
    pointer = StoredPairedFrequencyManifest(pair_sha256=digest, relative_path=good)
    assert pointer.relative_path == good

    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        StoredPairedFrequencyManifest(pair_sha256="AB" * 32, relative_path=good)
    with pytest.raises(ValueError, match="must be content-addressed"):
        StoredPairedFrequencyManifest(
            pair_sha256=digest, relative_path="manifests/paired_frequency_views/xx/other.json"
        )
