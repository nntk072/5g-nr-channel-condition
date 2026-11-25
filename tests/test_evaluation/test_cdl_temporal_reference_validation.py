import hashlib
import json

import evaluation.covariance.cdl.temporal_reference as temporal
import pytest


def _profile(**updates):
    profile = {
        "profile_model": "A",
        "los": False,
        "cluster_count": 1,
        "cluster_spreads_deg": {"asa": 10.0, "zsa": 5.0},
        "components": [
            {
                "component": "diffuse",
                "cluster_index": 1,
                "power_db": 0.0,
                "angles_deg": {"aoa": 20.0, "zoa": 80.0},
            }
        ],
    }
    profile.update(updates)
    return profile


def _reference_bytes(**updates):
    reference = {
        "schema": "loslab.g4_cdl_tap_angle_reference.v1",
        "source_id": "tr-138901-v19.2.0",
        "source_sha256": temporal._SOURCE_SHA256,
        "profiles": [_profile()],
    }
    reference.update(updates)
    return json.dumps(reference).encode()


def _install_reference_bytes(monkeypatch, payload):
    class ReferenceFile:
        def read_bytes(self):
            return payload

    monkeypatch.setattr(temporal, "_REFERENCE_PATH", ReferenceFile())
    monkeypatch.setattr(temporal, "_REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize(
    ("model", "carrier", "speed", "lag"),
    [
        ("A", "3.5e9", 1.0, 0.1),
        ("A", 3.5e9, True, 0.1),
        ("A", float("inf"), 1.0, 0.1),
        ("A", 3.5e9, 1.0, -0.1),
    ],
)
def test_cdl_temporal_reference_rejects_non_real_and_out_of_domain_inputs(
    model, carrier, speed, lag
):
    with pytest.raises(ValueError):
        temporal.cdl_nlos_temporal_correlation_reference(
            model, carrier_frequency_hz=carrier, speed_mps=speed, lag_s=lag
        )


def test_pinned_temporal_reference_verifies_digest_and_json(monkeypatch):
    payload = b"{}"
    monkeypatch.setattr(
        temporal, "_REFERENCE_PATH", type("Ref", (), {"read_bytes": lambda self: payload})()
    )
    monkeypatch.setattr(temporal, "_REFERENCE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="frozen digest"):
        temporal._load_pinned_cdl_profile("A")

    invalid_utf8 = b"\xff"
    _install_reference_bytes(monkeypatch, invalid_utf8)
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        temporal._load_pinned_cdl_profile("A")

    malformed_json = b"{"
    _install_reference_bytes(monkeypatch, malformed_json)
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        temporal._load_pinned_cdl_profile("A")


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ({"schema": "wrong"}, "not bound"),
        ({"profiles": {}}, "profile list"),
        ({"profiles": []}, "one CDL-A profile"),
        ({"profiles": [_profile(), _profile()]}, "one CDL-A profile"),
        ({"profiles": [{**_profile(), "los": True}]}, "must be NLOS"),
    ],
)
def test_pinned_temporal_reference_rejects_unbound_or_ambiguous_profiles(
    monkeypatch, reference, message
):
    payload = _reference_bytes(**reference)
    _install_reference_bytes(monkeypatch, payload)
    with pytest.raises(ValueError, match=message):
        temporal._load_pinned_cdl_profile("A")


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"cluster_spreads_deg": None}, "malformed clusters"),
        ({"components": []}, "malformed clusters"),
        ({"cluster_spreads_deg": {"asa": "bad"}}, "cluster ASA"),
        ({"cluster_spreads_deg": {"asa": 10.0, "zsa": float("nan")}}, "cluster ZSA"),
        ({"components": [None]}, "only diffuse components"),
        ({"components": [{"component": "los"}]}, "only diffuse components"),
        (
            {"components": [{"component": "diffuse", "cluster_index": True}]},
            "invalid cluster index",
        ),
        ({"components": [{"component": "diffuse", "cluster_index": 0}]}, "invalid cluster index"),
        (
            {"components": [{"component": "diffuse", "cluster_index": 1, "angles_deg": []}]},
            "malformed cluster angles",
        ),
        (
            {
                "components": [
                    {
                        "component": "diffuse",
                        "cluster_index": 1,
                        "power_db": 0.0,
                        "angles_deg": {"aoa": 0.0, "zoa": 0.0},
                    },
                    {
                        "component": "diffuse",
                        "cluster_index": 1,
                        "power_db": 0.0,
                        "angles_deg": {"aoa": 0.0, "zoa": 0.0},
                    },
                ]
            },
            "duplicates cluster",
        ),
        (
            {
                "components": [
                    {
                        "component": "diffuse",
                        "cluster_index": 1,
                        "power_db": -4000.0,
                        "angles_deg": {"aoa": 0.0, "zoa": 0.0},
                    }
                ]
            },
            "invalid linear cluster power",
        ),
        ({"cluster_count": 2}, "roster differs"),
        ({"cluster_count": True}, "roster differs"),
    ],
)
def test_cdl_temporal_reference_rejects_malformed_cluster_rows(monkeypatch, profile, message):
    row = _profile(**profile)
    monkeypatch.setattr(temporal, "_load_pinned_cdl_profile", lambda model: row)
    with pytest.raises(ValueError, match=message):
        temporal.cdl_nlos_temporal_correlation_reference(
            "A", carrier_frequency_hz=3.5e9, speed_mps=1.0, lag_s=0.001
        )


def test_cdl_temporal_reference_rejects_nonfinite_numerical_result(monkeypatch):
    monkeypatch.setattr(temporal, "_load_pinned_cdl_profile", lambda model: _profile())
    monkeypatch.setattr(temporal.math, "cos", lambda phase: float("inf"))
    monkeypatch.setattr(temporal.math, "sin", lambda phase: 0.0)
    with pytest.raises(ValueError, match="non-finite value"):
        temporal.cdl_nlos_temporal_correlation_reference(
            "A", carrier_frequency_hz=3.5e9, speed_mps=1.0, lag_s=0.001
        )
