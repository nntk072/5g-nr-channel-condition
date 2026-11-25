"""ETSI-derived CDL spatial covariance: reference identities, sampled claims and claim audit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from statistics import NormalDist

import evaluation.covariance.cdl.four_element as four_element
import numpy as np
import pytest
from evaluation.covariance.cdl.four_element import (
    _CLAIM_NAMES,
    audit_cdl_spatial_covariance_claims,
    cdl_spatial_covariance_claims,
    cdl_spatial_covariance_reference,
)


@pytest.mark.parametrize("model", ["A", "B", "C", "D", "E"])
def test_reference_is_a_normalized_hermitian_positive_semidefinite_covariance(model) -> None:
    covariance, digest = cdl_spatial_covariance_reference(model)
    assert covariance.shape == (4, 4) and covariance.dtype == np.complex128
    assert len(digest) == 64
    # Powers are normalized to one, and every element has the same unit power.
    assert np.allclose(np.diag(covariance).real, 1.0, atol=1e-12)
    assert np.allclose(np.diag(covariance).imag, 0.0, atol=1e-12)
    assert np.allclose(covariance, covariance.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > -1e-10


def test_los_profiles_add_a_coherent_component_that_nlos_profiles_do_not_have() -> None:
    los_matrix, _ = cdl_spatial_covariance_reference("D")
    nlos_matrix, _ = cdl_spatial_covariance_reference("A")
    assert not np.allclose(los_matrix, nlos_matrix)
    # The LOS component raises the magnitude of the strongest off-diagonal element.
    assert abs(los_matrix[0, 3]) > 0.0


def test_unsupported_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported CDL profile"):
        cdl_spatial_covariance_reference("F")
    with pytest.raises(ValueError, match="unsupported CDL profile"):
        cdl_spatial_covariance_reference("a")


def test_tampered_reference_bytes_fail_the_frozen_digest(tmp_path, monkeypatch) -> None:
    tampered = tmp_path / "reference.json"
    tampered.write_bytes(four_element.REFERENCE_PATH.read_bytes() + b" ")
    monkeypatch.setattr(four_element, "REFERENCE_PATH", tampered)
    with pytest.raises(ValueError, match="frozen digest"):
        cdl_spatial_covariance_reference("A")


def _patch_reference_document(tmp_path, monkeypatch, mutate):
    document = json.loads(four_element.REFERENCE_PATH.read_bytes())
    mutate(document)
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    path = tmp_path / "reference.json"
    path.write_bytes(payload)
    monkeypatch.setattr(four_element, "REFERENCE_PATH", path)
    monkeypatch.setattr(four_element, "REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())


def test_reference_rejects_wrong_frozen_source_identity(tmp_path, monkeypatch):
    _patch_reference_document(
        tmp_path,
        monkeypatch,
        lambda document: document.update(source_sha256="0" * 64),
    )

    with pytest.raises(ValueError, match="source digest differs"):
        cdl_spatial_covariance_reference("A")


@pytest.mark.parametrize(
    ("model", "mutate", "message"),
    [
        (
            "D",
            lambda profile: profile["components"].reverse(),
            "single specular component first",
        ),
        (
            "D",
            lambda profile: profile["components"][1].update(component="other"),
            "diffuse component roster is inconsistent",
        ),
        (
            "A",
            lambda profile: profile["components"][0].update(component="specular"),
            "NLOS profile must contain only diffuse components",
        ),
        ("A", lambda profile: profile.update(cluster_count=0), "cluster count differs"),
        (
            "A",
            lambda profile: profile["components"][0]["angles_deg"].update(aoa=float("nan")),
            "covariance reference must be finite",
        ),
    ],
    ids=["los-first", "los-roster", "nlos-specular", "cluster-count", "nonfinite"],
)
def test_reference_rejects_inconsistent_profile_rows(tmp_path, monkeypatch, model, mutate, message):
    def update(document):
        profile = next(row for row in document["profiles"] if row["profile_model"] == model)
        mutate(profile)

    _patch_reference_document(tmp_path, monkeypatch, update)

    with pytest.raises(ValueError, match=message):
        cdl_spatial_covariance_reference(model)


def _draws(reference: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    factor = np.linalg.cholesky(reference + 1e-12 * np.eye(4))
    noise = (rng.standard_normal((count, 4)) + 1j * rng.standard_normal((count, 4))) / np.sqrt(2)
    return noise @ factor.T


def test_sampled_covariance_claims_bracket_the_reference_with_bounded_width() -> None:
    reference, _ = cdl_spatial_covariance_reference("B")
    claims = cdl_spatial_covariance_claims(_draws(reference, 20000, seed=5), reference)
    assert [claim["name"] for claim in claims] == list(_CLAIM_NAMES)
    assert len(claims) == 16
    assert all(claim["contained"] for claim in claims)
    assert all(claim["precision_pass"] for claim in claims)
    assert all(claim["passed"] for claim in claims)
    diagonal = next(claim for claim in claims if claim["name"] == "diag_0")
    assert diagonal["reference"] == pytest.approx(1.0)
    z_value = NormalDist().inv_cdf(1.0 - 0.01 / 32.0)
    assert diagonal["half_width"] == pytest.approx(diagonal["standard_error"] * z_value, rel=1e-12)


def test_claim_family_assertion_guards_the_frozen_coordinate_roster(monkeypatch):
    monkeypatch.setattr(four_element, "range", lambda *_args: iter(()), raising=False)

    with pytest.raises(AssertionError, match="must contain 16 coordinates"):
        cdl_spatial_covariance_claims(np.ones((2, 4), dtype=complex), np.eye(4, dtype=complex))


def test_wrong_reference_fails_containment_and_is_counted() -> None:
    reference, _ = cdl_spatial_covariance_reference("B")
    wrong = reference.copy()
    wrong[0, 1] += 0.5
    claims = cdl_spatial_covariance_claims(_draws(reference, 20000, seed=9), wrong)
    failed = [claim for claim in claims if not claim["passed"]]
    assert failed and all(claim["name"] in {"re_01", "im_01"} for claim in failed)
    audit = audit_cdl_spatial_covariance_claims(claims)
    assert audit["claims_failed"] == len(failed)
    assert audit["status"] == "FAIL"


def test_precision_bound_is_enforced_independently_of_containment() -> None:
    reference, _ = cdl_spatial_covariance_reference("A")
    claims = cdl_spatial_covariance_claims(
        _draws(reference, 400, seed=2), reference, maximum_half_width=1e-6
    )
    assert all(claim["contained"] or claim["precision_pass"] is False for claim in claims)
    assert not any(claim["precision_pass"] for claim in claims)
    assert not any(claim["passed"] for claim in claims)


@pytest.mark.parametrize(
    ("samples", "reference", "kwargs", "message"),
    [
        (np.ones((5, 4)), np.eye(4, dtype=complex), {}, "complex \\[draw, 4\\]"),
        (np.ones((5, 3), dtype=complex), np.eye(4, dtype=complex), {}, "complex \\[draw, 4\\]"),
        (np.ones((1, 4), dtype=complex), np.eye(4, dtype=complex), {}, "at least two finite draws"),
        (
            np.full((5, 4), np.nan, dtype=complex),
            np.eye(4, dtype=complex),
            {},
            "at least two finite draws",
        ),
        (np.ones((5, 4), dtype=complex), np.ones((4, 4)), {}, "finite complex 4x4"),
        (np.ones((5, 4), dtype=complex), np.ones((3, 3), dtype=complex), {}, "finite complex 4x4"),
        (
            np.ones((5, 4), dtype=complex),
            np.eye(4, dtype=complex),
            {"familywise_alpha": 0.0},
            "familywise_alpha",
        ),
        (
            np.ones((5, 4), dtype=complex),
            np.eye(4, dtype=complex),
            {"familywise_alpha": True},
            "familywise_alpha",
        ),
        (
            np.ones((5, 4), dtype=complex),
            np.eye(4, dtype=complex),
            {"maximum_half_width": 0.0},
            "maximum_half_width",
        ),
        (
            np.ones((5, 4), dtype=complex),
            np.eye(4, dtype=complex),
            {"maximum_half_width": True},
            "maximum_half_width",
        ),
    ],
)
def test_claim_inputs_are_validated(samples, reference, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        cdl_spatial_covariance_claims(samples, reference, **kwargs)


def test_claim_names_are_the_frozen_sixteen_coordinates() -> None:
    assert len(_CLAIM_NAMES) == 16
    assert _CLAIM_NAMES[:4] == ("diag_0", "diag_1", "diag_2", "diag_3")
    assert _CLAIM_NAMES[4:6] == ("re_01", "im_01")


def _valid_claims() -> list[dict]:
    reference, _ = cdl_spatial_covariance_reference("C")
    return cdl_spatial_covariance_claims(_draws(reference, 20000, seed=13), reference)


def test_audit_accepts_recomputed_claims_and_reports_the_family_status() -> None:
    audit = audit_cdl_spatial_covariance_claims(_valid_claims())
    assert audit["claims_passed"] == 16 and audit["claims_failed"] == 0
    assert audit["status"] == "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"
    assert audit["maximum_absolute_error"] >= 0.0
    assert 0.0 < audit["maximum_half_width"] < 0.05


def _claim_mutation(index: int, **changes) -> list[dict]:
    claims = copy.deepcopy(_valid_claims())
    claims[index].update(changes)
    return claims


@pytest.mark.parametrize(
    ("claims_factory", "message"),
    [
        (lambda: "not-a-list", "exactly 16 claims"),
        (lambda: _valid_claims()[:-1], "exactly 16 claims"),
        (lambda: list(reversed(_valid_claims())), "names or order differ"),
        (lambda: _claim_mutation(0, extra=1), "fields differ"),
        (lambda: _claim_mutation(0, estimate=float("nan")), "finite and real"),
        (lambda: _claim_mutation(0, estimate=True), "finite and real"),
        (lambda: _claim_mutation(0, standard_error=-1.0), "nonnegative"),
        (lambda: _claim_mutation(0, contained=1), "Boolean"),
        (lambda: _claim_mutation(0, interval=[1.0]), "interval or error arithmetic"),
        (lambda: _claim_mutation(0, absolute_error=99.0), "interval or error arithmetic"),
        (lambda: _claim_mutation(0, passed=not _valid_claims()[0]["passed"]), "decision differs"),
    ],
)
def test_audit_rejects_tampered_claim_reports(claims_factory, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_cdl_spatial_covariance_claims(claims_factory())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"familywise_alpha": 1.0}, "familywise_alpha"),
        ({"familywise_alpha": True}, "familywise_alpha"),
        ({"maximum_half_width": -0.1}, "maximum_half_width"),
        ({"maximum_half_width": True}, "maximum_half_width"),
    ],
)
def test_audit_validates_its_criteria(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        audit_cdl_spatial_covariance_claims(_valid_claims(), **kwargs)


def test_audit_reports_failures_when_a_claim_exceeds_the_width_bound() -> None:
    reference, _ = cdl_spatial_covariance_reference("C")
    narrow = cdl_spatial_covariance_claims(
        _draws(reference, 20000, seed=13), reference, maximum_half_width=1e-9
    )
    audit = audit_cdl_spatial_covariance_claims(narrow, maximum_half_width=1e-9)
    assert audit["status"] == "FAIL"
    assert audit["claims_failed"] == 16
    assert math.isfinite(audit["maximum_half_width"])
