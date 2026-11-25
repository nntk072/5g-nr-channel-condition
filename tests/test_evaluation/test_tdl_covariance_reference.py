from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from evaluation.covariance import tdl_covariance_reference as reference
from evaluation.covariance.tdl_covariance_reference import (
    fixed_tdl_first_tap_power_components,
    fixed_tdl_rayleigh_tap_indices,
    scalable_tdl_first_tap_covariance_reference,
    scalable_tdl_first_tap_power_components,
)


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_first_tap_component_powers_come_from_scalable_etsi_rows(model: str) -> None:
    diffuse, specular = scalable_tdl_first_tap_power_components(model)

    assert diffuse > 0.0
    if model in "DE":
        assert specular > 0.0
    else:
        assert specular == 0.0
    assert diffuse + specular <= 1.0


@pytest.mark.parametrize("model", ("A10", "A30", "B100", "C60", "C300", "D10", "D30"))
def test_fixed_first_tap_component_powers_come_from_annex_b_rows(model: str) -> None:
    diffuse, specular = fixed_tdl_first_tap_power_components(model)

    assert diffuse > 0.0
    assert 0.0 <= specular < 1.0
    assert (specular > 0.0) == (model in {"D10", "D30"})
    assert diffuse + specular <= 1.0


@pytest.mark.parametrize(
    ("model", "expected_count"),
    (("A10", 16), ("A30", 12), ("B100", 12), ("C60", 12), ("C300", 12), ("D10", 9), ("D30", 9)),
)
def test_fixed_rayleigh_tap_roster_excludes_only_mixed_los_taps(
    model: str, expected_count: int
) -> None:
    tap_indices = fixed_tdl_rayleigh_tap_indices(model)

    assert len(tap_indices) == expected_count
    assert tap_indices == tuple(sorted(set(tap_indices)))
    if model in {"D10", "D30"}:
        assert tap_indices == tuple(range(1, 10))
    else:
        assert tap_indices == tuple(range(expected_count))


@pytest.mark.parametrize("model", tuple("ABCDE"))
def test_first_tap_covariance_is_hermitian_psd_and_normalized(model: str) -> None:
    rho = 0.45
    phase = 0.23
    correlation = np.array(
        [
            [rho ** abs(row - column) * np.exp(1j * phase * (row - column)) for column in range(3)]
            for row in range(3)
        ],
        dtype=np.complex128,
    )
    covariance = scalable_tdl_first_tap_covariance_reference(model, correlation, 2)

    assert covariance.shape == (6, 6)
    np.testing.assert_allclose(covariance, covariance.conj().T, rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
    diffuse_power, specular_power = scalable_tdl_first_tap_power_components(model)
    los_vector = np.linalg.cholesky(correlation) @ np.ones(3)
    assert np.trace(covariance).real / 6 == pytest.approx(
        diffuse_power + specular_power * np.vdot(los_vector, los_vector).real / 3,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("model", "correlation", "num_tx_ant"),
    (
        ("F", [[1.0]], 1),
        ("A", [[1.0, 0.2], [0.1, 1.0]], 1),
        ("A", [[1.0, 1.0], [1.0, 1.0]], 1),
        ("A", [[1.0, 0.2], [0.2, 1.0]], 0),
    ),
)
def test_first_tap_covariance_rejects_invalid_inputs(
    model: str, correlation: list[list[float]], num_tx_ant: int
) -> None:
    with pytest.raises(ValueError):
        scalable_tdl_first_tap_covariance_reference(model, correlation, num_tx_ant)


@pytest.mark.parametrize(
    "correlation",
    (
        [[True]],
        [["1.0"]],
        np.asarray([[1.0]], dtype=object),
    ),
)
def test_first_tap_covariance_rejects_coercible_non_numeric_dtypes(correlation) -> None:
    with pytest.raises(ValueError, match="numeric"):
        scalable_tdl_first_tap_covariance_reference("A", correlation, 1)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param("contract", "unexpected source contract", id="contract"),
        pytest.param("profile-count", "cover all five profiles", id="profile-count"),
        pytest.param("profile-row", "profile row must be an object", id="profile-row"),
        pytest.param("profile-model", "profile roster is invalid", id="profile-model"),
        pytest.param("component-index", "component roster is invalid", id="component-index"),
        pytest.param("component-fields", "component fields are invalid", id="component-fields"),
        pytest.param("first-delay", "must begin at zero normalized delay", id="first-delay"),
        pytest.param("los-roster", "LOS component roster is inconsistent", id="los-roster"),
        pytest.param("los-pair", "LOS component must pair at zero delay", id="los-pair"),
    ],
)
def test_scalable_profile_loader_rejects_altered_reference_contracts(
    tmp_path, monkeypatch: pytest.MonkeyPatch, change: str, message: str
) -> None:
    document = json.loads(reference._REFERENCE_PATH.read_text(encoding="utf-8"))
    row = document["profiles"][0]
    component = row["components"][0]
    if change == "contract":
        document["version"] = "other"
    elif change == "profile-count":
        document["profiles"].pop()
    elif change == "profile-row":
        document["profiles"][0] = None
    elif change == "profile-model":
        row["profile_model"] = "F"
    elif change == "component-index":
        component["component_index"] = 2
    elif change == "component-fields":
        component["power_db"] = float("nan")
    elif change == "first-delay":
        component["normalized_delay"] = 1.0
    elif change == "los-roster":
        for component in document["profiles"][3]["components"]:
            component["fading_distribution"] = "Rayleigh"
    elif change == "los-pair":
        document["profiles"][3]["components"][1]["normalized_delay"] = 1.0
    payload = json.dumps(document, allow_nan=True).encode("utf-8")
    path = tmp_path / "scalable-reference.json"
    path.write_bytes(payload)
    monkeypatch.setattr(reference, "_REFERENCE_PATH", path)
    monkeypatch.setattr(reference, "_REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())
    reference._load_profiles.cache_clear()
    try:
        with pytest.raises(ValueError, match=message):
            reference._load_profiles()
    finally:
        reference._load_profiles.cache_clear()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        pytest.param("contract", "unexpected source contract", id="contract"),
        pytest.param("profile-count", "cover all seven profiles", id="profile-count"),
        pytest.param("profile-row", "profile row must be an object", id="profile-row"),
        pytest.param("profile-model", "profile roster is invalid", id="profile-model"),
        pytest.param("component-index", "component roster is invalid", id="component-index"),
        pytest.param("component-fields", "component fields are invalid", id="component-fields"),
    ],
)
def test_fixed_profile_loader_rejects_altered_reference_contracts(
    tmp_path, monkeypatch: pytest.MonkeyPatch, change: str, message: str
) -> None:
    document = json.loads(reference._FIXED_REFERENCE_PATH.read_text(encoding="utf-8"))
    row = document["profiles"][0]
    component = row["components"][0]
    if change == "contract":
        document["version"] = "other"
    elif change == "profile-count":
        document["profiles"].pop()
    elif change == "profile-row":
        document["profiles"][0] = None
    elif change == "profile-model":
        row["profile_model"] = "other"
    elif change == "component-index":
        component["component_index"] = 2
    elif change == "component-fields":
        component["tap_index"] = True

    payload = json.dumps(document).encode("utf-8")
    path = tmp_path / "fixed-reference.json"
    path.write_bytes(payload)
    monkeypatch.setattr(reference, "_FIXED_REFERENCE_PATH", path)
    monkeypatch.setattr(reference, "_FIXED_REFERENCE_SHA256", hashlib.sha256(payload).hexdigest())
    reference._load_fixed_profiles.cache_clear()
    try:
        with pytest.raises(ValueError, match=message):
            reference._load_fixed_profiles()
    finally:
        reference._load_fixed_profiles.cache_clear()


@pytest.mark.parametrize(
    ("correlation", "message"),
    [
        pytest.param([], "finite square Hermitian", id="not-matrix"),
        pytest.param([[1.0, 0.0]], "finite square Hermitian", id="not-square"),
        pytest.param([[float("inf")]], "finite square Hermitian", id="nonfinite"),
        pytest.param([[0.5]], "unit diagonal", id="diagonal"),
        pytest.param([[1.0, 1.0], [1.0, 1.0]], "positive definite", id="singular"),
    ],
)
def test_first_tap_covariance_checks_matrix_structure(correlation, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        scalable_tdl_first_tap_covariance_reference("A", correlation, 1)
