"""Full-sphere direction transforms and solid-angle-weighted power maps."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

_TWO_PI = 2.0 * math.pi


def _contains_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "b":
            return True
        if value.dtype.kind != "O":
            return False
        return any(_contains_boolean(item) for item in value.flat)
    if isinstance(value, (list, tuple)):
        return any(_contains_boolean(item) for item in value)
    return False


def _real_array(value: Any, *, name: str) -> NDArray[np.float64]:
    if _contains_boolean(value):
        raise ValueError(f"{name} must contain real numeric values, not booleans")
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_vector(value: Any, *, name: str) -> NDArray[np.float64]:
    array = _real_array(value, name=name)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must be a finite one-dimensional array with at least two values")
    return array


def _orientation(value: Sequence[float]) -> tuple[float, float, float]:
    angles = _real_array(value, name="orientation_rad")
    if angles.ndim != 1 or angles.size != 3:
        raise ValueError("orientation_rad must contain three finite Euler angles")
    return (float(angles[0]), float(angles[1]), float(angles[2]))


def global_to_element_local_directions(
    zenith_rad: Any,
    azimuth_rad: Any,
    *,
    orientation_rad: Sequence[float],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Express global directions in a local frame using ``Rz(a) Ry(b) Rx(g)``.

    Zenith is measured from +z and azimuth from +x toward +y. The returned
    azimuth is wrapped by ``atan2`` into ``[-pi, pi]``. Inputs may have any
    equal shape and finite azimuths are accepted outside one principal turn.
    """
    zenith = _real_array(zenith_rad, name="zenith_rad")
    azimuth = _real_array(azimuth_rad, name="azimuth_rad")
    if zenith.shape != azimuth.shape:
        raise ValueError("zenith_rad and azimuth_rad must be finite arrays with equal shapes")
    if np.any((zenith < 0.0) | (zenith > math.pi)):
        raise ValueError("zenith_rad must lie in [0, pi]")

    alpha, beta, gamma = _orientation(orientation_rad)
    x_global = np.sin(zenith) * np.cos(azimuth)
    y_global = np.sin(zenith) * np.sin(azimuth)
    z_global = np.cos(zenith)
    ca, sa = math.cos(alpha), math.sin(alpha)
    cb, sb = math.cos(beta), math.sin(beta)
    cg, sg = math.cos(gamma), math.sin(gamma)
    x_local = ca * cb * x_global + sa * cb * y_global - sb * z_global
    y_local = (
        (ca * sb * sg - sa * cg) * x_global
        + (sa * sb * sg + ca * cg) * y_global
        + cb * sg * z_global
    )
    z_local = (
        (ca * sb * cg + sa * sg) * x_global
        + (sa * sb * cg - ca * sg) * y_global
        + cb * cg * z_global
    )
    local_zenith = np.arccos(np.clip(z_local, -1.0, 1.0))
    local_azimuth = np.arctan2(y_local, x_local)
    return local_zenith, local_azimuth


def spherical_cell_solid_angles(
    zenith_edges_rad: Any, azimuth_edges_rad: Any
) -> NDArray[np.float64]:
    """Return cell solid angles for a full-sphere separable angular grid.

    Zenith cells use the exact spherical area factor ``cos(theta0)-cos(theta1)``;
    azimuth cells may have unequal widths but must span exactly one full turn.
    Both edge vectors describe half-open bins, with the south pole assigned to
    the final zenith bin and the azimuth upper endpoint wrapping to the first.
    """
    zenith = _finite_vector(zenith_edges_rad, name="zenith_edges_rad")
    azimuth = _finite_vector(azimuth_edges_rad, name="azimuth_edges_rad")
    if zenith[0] != 0.0 or not math.isclose(
        zenith[-1], math.pi, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("zenith edges must span [0, pi]")
    if np.any(np.diff(zenith) <= 0.0):
        raise ValueError("zenith edges must be strictly increasing")
    if np.any(np.diff(azimuth) <= 0.0):
        raise ValueError("azimuth edges must be strictly increasing")
    if not math.isclose(
        azimuth[-1] - azimuth[0], _TWO_PI, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("azimuth edges must span exactly 2*pi")
    zenith_factors = np.cos(zenith[:-1]) - np.cos(zenith[1:])
    azimuth_widths = np.diff(azimuth)
    areas = zenith_factors[:, None] * azimuth_widths[None, :]
    if np.any(areas <= 0.0) or not math.isclose(
        float(areas.sum()), 4.0 * math.pi, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("angular grid does not form positive full-sphere cells")
    return areas


def map_spherical_power(
    zenith_rad: Any,
    azimuth_rad: Any,
    power_weights: Any,
    *,
    zenith_edges_rad: Any,
    azimuth_edges_rad: Any,
) -> dict[str, Any]:
    """Bin ray/sample powers on a full sphere and report density per steradian."""
    zenith = _real_array(zenith_rad, name="zenith_rad")
    azimuth = _real_array(azimuth_rad, name="azimuth_rad")
    weights = _real_array(power_weights, name="power_weights")
    if zenith.ndim != 1 or zenith.size == 0 or azimuth.shape != zenith.shape:
        raise ValueError("direction arrays must be equal-length nonempty vectors")
    if weights.shape != zenith.shape:
        raise ValueError("power_weights must match the direction arrays")
    if (
        np.any((zenith < 0.0) | (zenith > math.pi))
        or np.any(weights < 0.0)
    ):
        raise ValueError("directions must be finite/on-sphere and powers finite/nonnegative")
    total_power = float(weights.sum())
    if not math.isfinite(total_power) or total_power <= 0.0:
        raise ValueError("power_weights must have a finite positive total")

    zenith_edges = _finite_vector(zenith_edges_rad, name="zenith_edges_rad")
    azimuth_edges = _finite_vector(azimuth_edges_rad, name="azimuth_edges_rad")
    solid_angles = spherical_cell_solid_angles(zenith_edges, azimuth_edges)
    zenith_index = np.searchsorted(zenith_edges, zenith, side="right") - 1
    zenith_index = np.clip(zenith_index, 0, len(zenith_edges) - 2)
    azimuth_wrapped = azimuth_edges[0] + np.mod(azimuth - azimuth_edges[0], _TWO_PI)
    azimuth_index = np.searchsorted(azimuth_edges, azimuth_wrapped, side="right") - 1
    azimuth_index = np.clip(azimuth_index, 0, len(azimuth_edges) - 2)
    flat_bins = zenith_index * (len(azimuth_edges) - 1) + azimuth_index
    powers = np.bincount(flat_bins, weights=weights, minlength=solid_angles.size).reshape(
        solid_angles.shape
    )
    fractions = powers / total_power
    density = fractions / solid_angles
    return {
        "schema": "umcd.spherical_angular_power_map.v1",
        "input_direction_count": int(zenith.size),
        "total_power": total_power,
        "zenith_edges_rad": zenith_edges.tolist(),
        "azimuth_edges_rad": azimuth_edges.tolist(),
        "cell_solid_angles_sr": solid_angles.tolist(),
        "integrated_power_fractions": fractions.tolist(),
        "power_density_per_sr": density.tolist(),
        "bin_boundary_convention": (
            "half-open; south pole in last zenith bin; "
            "azimuth upper edge wraps to lower edge"
        ),
    }


def map_element_frame_spherical_power(
    global_zenith_rad: Any,
    global_azimuth_rad: Any,
    power_weights: Any,
    *,
    orientation_rad: Sequence[float],
    zenith_edges_rad: Any,
    azimuth_edges_rad: Any,
) -> dict[str, Any]:
    """Map global ray power into bins expressed in an oriented element frame.

    ``orientation_rad`` gives the element's global yaw/pitch/roll under the
    ``Rz(yaw) Ry(pitch) Rx(roll)`` convention. Directions are inverse-rotated
    before binning, so directional element patterns can consume a map in their
    local coordinate frame without callers duplicating transform semantics.
    """
    orientation = _orientation(orientation_rad)
    local_zenith, local_azimuth = global_to_element_local_directions(
        global_zenith_rad,
        global_azimuth_rad,
        orientation_rad=orientation,
    )
    result = map_spherical_power(
        local_zenith,
        local_azimuth,
        power_weights,
        zenith_edges_rad=zenith_edges_rad,
        azimuth_edges_rad=azimuth_edges_rad,
    )
    return {
        **result,
        "schema": "umcd.element_frame_spherical_power_map.v1",
        "direction_frame": "element_local",
        "orientation_rad": list(orientation),
        "orientation_convention": "Rz(yaw) Ry(pitch) Rx(roll); inverse applied to global directions",
    }
