"""Input and defensive-branch checks for the handheld-UT covariance reference."""

from __future__ import annotations

from copy import deepcopy

import evaluation.covariance.cdl.handheld_ut as handheld_ut
import pytest


@pytest.mark.parametrize(
    ("geometry", "kwargs"),
    [
        (handheld_ut.handheld_ut_single_v_geometry, {"device_depth_m": 0.0}),
        (handheld_ut.handheld_ut_single_v_geometry, {"device_width_m": -0.1}),
        (handheld_ut.handheld_ut_dual_geometry, {"device_depth_m": -0.1}),
        (handheld_ut.handheld_ut_dual_geometry, {"device_width_m": 0.0}),
    ],
)
def test_handheld_geometry_requires_positive_device_dimensions(geometry, kwargs) -> None:
    with pytest.raises(ValueError, match="device dimensions must be positive"):
        geometry((1,), **kwargs)


def test_handheld_location_and_normalization_guards() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        handheld_ut.handheld_ut_single_v_geometry(())
    with pytest.raises(ValueError, match="must be nonzero"):
        handheld_ut._normalize((0.0, 0.0, 0.0))


def test_handheld_port_field_rejects_unknown_component() -> None:
    _, bases = handheld_ut.handheld_ut_single_v_geometry((1,))
    with pytest.raises(ValueError, match="field_component must be 0"):
        handheld_ut.handheld_ut_port_field_gcs(
            (1.0, 0.0, 0.0),
            endpoint_orientation_rad=(0.0, 0.0, 0.0),
            port_basis_device=bases[0],
            field_component=2,
        )


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (handheld_ut.cdl_handheld_ut_spatiotemporal_covariance_reference, {}),
        (
            handheld_ut.cdl_handheld_ut_dual_spatiotemporal_covariance_reference,
            {},
        ),
    ],
)
def test_handheld_covariance_rejects_invalid_model_velocity_carrier_and_lag(
    function, kwargs
) -> None:
    common = {"antenna_locations": (1,), "ut_orientation_rad": (0.0, 0.0, 0.0)}
    with pytest.raises(ValueError, match="unsupported CDL profile"):
        function("Z", (0.0, 0.0, 0.0), **common, **kwargs)
    with pytest.raises(ValueError, match="exactly three"):
        function("A", (0.0, 0.0), **common, **kwargs)
    with pytest.raises(ValueError, match="carrier must be positive"):
        function("A", (0.0, 0.0, 0.0), **common, carrier_frequency_hz=0.0, **kwargs)
    with pytest.raises(ValueError, match="carrier must be positive"):
        function("A", (0.0, 0.0, 0.0), **common, lag_s=-1.0, **kwargs)


@pytest.mark.parametrize(
    "function",
    [
        handheld_ut.cdl_handheld_ut_spatiotemporal_covariance_reference,
        handheld_ut.cdl_handheld_ut_dual_spatiotemporal_covariance_reference,
    ],
)
def test_handheld_covariance_rejects_unknown_component_kind(monkeypatch, function) -> None:
    document = deepcopy(handheld_ut._reference_document())
    profile = next(row for row in document["profiles"] if row["profile_model"] == "A")
    profile["components"][0]["component"] = "unknown"
    monkeypatch.setattr(handheld_ut, "_reference_document", lambda: document)

    with pytest.raises(ValueError, match="unsupported CDL component kind: unknown"):
        function(
            "A",
            (0.0, 0.0, 0.0),
            antenna_locations=(1,),
            ut_orientation_rad=(0.0, 0.0, 0.0),
        )
