"""Canonical spatial split identities for Sionna RT channel samples."""

from __future__ import annotations

import math
from numbers import Real
from typing import Sequence


def rt_spatial_split_subject_id(
    *,
    scene_asset_set_sha256: object,
    scene_asset_manifest_complete: object,
    scene_name: str,
    rx_position_m: Sequence[float],
    block_size_m: float,
) -> tuple[str, bool]:
    """Build a stable complete or explicitly provisional spatial parent ID.

    Complete scene hashes make the identity independent of a scene filename,
    output run, or transmitter ID. Incomplete asset closures use a visibly
    provisional scene-name identity and cannot masquerade as cross-run audited.
    Coordinates are assigned with exact binary-rational floor division so
    negative positions and points on block boundaries are deterministic.
    """
    if not isinstance(scene_name, str) or not scene_name.strip():
        raise ValueError("scene_name must be a nonempty string")
    if type(scene_asset_manifest_complete) is not bool:
        raise ValueError("scene_asset_manifest_complete must be a boolean")
    if (
        isinstance(block_size_m, bool)
        or not isinstance(block_size_m, Real)
        or not math.isfinite(float(block_size_m))
        or block_size_m <= 0
    ):
        raise ValueError("block_size_m must be finite and positive")
    if not isinstance(rx_position_m, Sequence) or len(rx_position_m) != 3:
        raise ValueError("rx_position_m must contain three finite coordinates")
    coordinates: list[float] = []
    for coordinate in rx_position_m:
        if (
            isinstance(coordinate, bool)
            or not isinstance(coordinate, Real)
            or not math.isfinite(float(coordinate))
        ):
            raise ValueError("rx_position_m must contain three finite coordinates")
        coordinates.append(float(coordinate))

    complete = scene_asset_manifest_complete is True
    if complete:
        if (
            not isinstance(scene_asset_set_sha256, str)
            or len(scene_asset_set_sha256) != 64
            or any(character not in "0123456789abcdef" for character in scene_asset_set_sha256)
        ):
            raise ValueError("complete scene assets require a lowercase SHA-256 identity")
        scene_identity = scene_asset_set_sha256
    else:
        scene_identity = f"unclosed-{scene_name.strip()}"

    block_size = float(block_size_m)

    def floor_block_index(coordinate: float) -> int:
        numerator, denominator = coordinate.as_integer_ratio()
        size_numerator, size_denominator = block_size.as_integer_ratio()
        return (numerator * size_denominator) // (denominator * size_numerator)

    block_x = floor_block_index(coordinates[0])
    block_y = floor_block_index(coordinates[1])
    return (
        f"rt:{scene_identity}:block:{block_size.hex()}:{block_x}:{block_y}",
        complete,
    )
