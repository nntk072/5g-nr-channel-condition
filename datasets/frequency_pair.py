"""Provenance-bound pairing for independently solved complex frequency views."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .channel_view import PhysicalChannelView
from .provenance import UMCDRunProvenance

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_digest(value: object) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _scene_geometry_digest(record: UMCDRunProvenance) -> str | None:
    scene_geometry = record.physical_config.get("scene_geometry")
    if isinstance(scene_geometry, Mapping):
        digest = scene_geometry.get("scene_geometry_sha256")
        return digest if isinstance(digest, str) else None
    return None


def _bound_carrier_hz(record: UMCDRunProvenance) -> float | None:
    """Read the carrier from the supported RT trajectory provenance shape."""
    config = record.physical_config
    trajectory = config.get("trajectory")
    candidate = trajectory.get("carrier_hz") if isinstance(trajectory, Mapping) else None
    if candidate is None:
        candidate = config.get("carrier_hz")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        return None
    return float(candidate)


def _view_digest(view: PhysicalChannelView) -> str:
    metadata = {
        "view_id": view.view_id,
        "link_id": view.link_id,
        "frame_times_s": list(view.frame_times_s),
        "rx_port_ids": list(view.rx_port_ids),
        "tx_port_ids": list(view.tx_port_ids),
        "carrier_hz": view.carrier_hz,
        "subcarrier_offsets_hz": view.subcarrier_offsets_hz.tolist(),
        "delay_reference": view.delay_reference.value,
        "power_scale": view.power_scale.value,
        "normalization_recipe": view.normalization_recipe,
        "phase_gauge_id": view.phase_gauge_id,
        "source_time_alignment_sha256": view.source_time_alignment_sha256,
        "axes": list(view.axes),
        "shape": list(view.channel.shape),
        "dtype": view.channel.dtype.str,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    digest.update(np.ascontiguousarray(view.channel).tobytes())
    digest.update(np.ascontiguousarray(view.valid_mask).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PairedFrequencyViews:
    """Two independently solved, same-scene complex channels at distinct bands.

    The object binds the source scene/drop and split parent to each view's
    content-addressed RT provenance. It does not claim cross-band phase
    coherence; phase metrics still require the separate gauge contract.
    """

    pair_id: str
    source_scene_id: str
    parent_episode_id: str
    split_group_id: str
    scene_geometry_sha256: str
    lower_bundle_sha256: str
    upper_bundle_sha256: str
    lower_view: PhysicalChannelView
    lower_provenance: UMCDRunProvenance
    upper_view: PhysicalChannelView
    upper_provenance: UMCDRunProvenance

    def __post_init__(self) -> None:
        for name in ("pair_id", "source_scene_id", "parent_episode_id", "split_group_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise ValueError(f"{name} must be a non-empty canonical string")
        if not isinstance(self.scene_geometry_sha256, str) or not _SHA256.fullmatch(
            self.scene_geometry_sha256
        ):
            raise ValueError("scene_geometry_sha256 must be a lowercase SHA-256 digest")
        for name in ("lower_bundle_sha256", "upper_bundle_sha256"):
            if not isinstance(getattr(self, name), str) or not _SHA256.fullmatch(
                getattr(self, name)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.lower_view, PhysicalChannelView) or not isinstance(
            self.upper_view, PhysicalChannelView
        ):
            raise TypeError("paired frequency inputs must be complex PhysicalChannelView values")
        if not isinstance(self.lower_provenance, UMCDRunProvenance) or not isinstance(
            self.upper_provenance, UMCDRunProvenance
        ):
            raise TypeError("each frequency view requires its UMCDRunProvenance")
        if self.lower_view.view_id == self.upper_view.view_id:
            raise ValueError("paired frequency views must have distinct view IDs")
        if self.lower_view.carrier_hz == self.upper_view.carrier_hz:
            raise ValueError("paired frequency views must use distinct carrier frequencies")
        if (
            self.lower_view.link_id != self.upper_view.link_id
            or self.lower_view.frame_times_s != self.upper_view.frame_times_s
            or self.lower_view.rx_port_ids != self.upper_view.rx_port_ids
            or self.lower_view.tx_port_ids != self.upper_view.tx_port_ids
            or self.lower_view.delay_reference is not self.upper_view.delay_reference
            or self.lower_view.power_scale is not self.upper_view.power_scale
            or self.lower_view.normalization_recipe != self.upper_view.normalization_recipe
        ):
            raise ValueError("paired views must share link, time, port, delay, and power contracts")
        for view, record in (
            (self.lower_view, self.lower_provenance),
            (self.upper_view, self.upper_provenance),
        ):
            if record.subject_layer != "oracle_channels":
                raise ValueError("paired channel provenance must describe oracle_channels")
            if "RT" not in record.source_family.upper():
                raise ValueError("paired complex channels require RT solve provenance")
            if _bound_carrier_hz(record) != view.carrier_hz:
                raise ValueError("view carrier does not match its run-provenance physical config")
            if _scene_geometry_digest(record) != self.scene_geometry_sha256:
                raise ValueError("run provenance does not bind the paired scene geometry")
            if not record.scene_asset_sha256:
                raise ValueError("paired run provenance must bind source scene assets")
        if dict(self.lower_provenance.scene_asset_sha256) != dict(
            self.upper_provenance.scene_asset_sha256
        ):
            raise ValueError("paired solves must use the same source scene asset digests")

    def to_dict(self) -> dict[str, object]:
        """Return a canonical pair manifest with hashes of each solve record."""
        core: dict[str, object] = {
            "schema": "umcd.paired_frequency_views.v1",
            "pair_id": self.pair_id,
            "source_scene_id": self.source_scene_id,
            "parent_episode_id": self.parent_episode_id,
            "split_group_id": self.split_group_id,
            "scene_geometry_sha256": self.scene_geometry_sha256,
            "bundle_sha256s": [self.lower_bundle_sha256, self.upper_bundle_sha256],
            "views": [
                {
                    "view_id": self.lower_view.view_id,
                    "view_content_sha256": _view_digest(self.lower_view),
                    "carrier_hz": self.lower_view.carrier_hz,
                    "provenance_record_id": self.lower_provenance.record_id,
                    "provenance_record_sha256": self.lower_provenance.to_dict()["record_sha256"],
                    "material_config_sha256": _canonical_digest(
                        dict(self.lower_provenance.material_config)
                    ),
                    "solver_config_sha256": _canonical_digest(
                        dict(self.lower_provenance.solver_config)
                    ),
                },
                {
                    "view_id": self.upper_view.view_id,
                    "view_content_sha256": _view_digest(self.upper_view),
                    "carrier_hz": self.upper_view.carrier_hz,
                    "provenance_record_id": self.upper_provenance.record_id,
                    "provenance_record_sha256": self.upper_provenance.to_dict()["record_sha256"],
                    "material_config_sha256": _canonical_digest(
                        dict(self.upper_provenance.material_config)
                    ),
                    "solver_config_sha256": _canonical_digest(
                        dict(self.upper_provenance.solver_config)
                    ),
                },
            ],
            "complex_csi_source": "independent_band_specific_propagation_solves",
            "phase_coherence_claim": "not_asserted",
        }
        return {**core, "pair_sha256": _canonical_digest(core)}


def pair_stored_frequency_bundles(
    root: str,
    *,
    pair_id: str,
    source_scene_id: str,
    parent_episode_id: str,
    lower_bundle: object,
    upper_bundle: object,
    lower_view_id: str,
    upper_view_id: str,
) -> PairedFrequencyViews:
    """Read two stored bundles and bind their exact artifact hashes into a pair.

    Source-scene and parent IDs are caller-supplied lineage claims; the helper
    independently checks the stored bundle hashes, split group, selected view,
    and unique carrier-matched oracle-channel provenance.
    """
    from .episode_store import StoredEpisodeBundle, read_episode_bundle

    if not isinstance(lower_bundle, StoredEpisodeBundle) or not isinstance(
        upper_bundle, StoredEpisodeBundle
    ):
        raise TypeError("lower_bundle and upper_bundle must be StoredEpisodeBundle references")
    if not isinstance(lower_view_id, str) or not lower_view_id.strip():
        raise ValueError("lower_view_id must be a non-empty string")
    if not isinstance(upper_view_id, str) or not upper_view_id.strip():
        raise ValueError("upper_view_id must be a non-empty string")
    if lower_bundle.bundle_sha256 == upper_bundle.bundle_sha256:
        raise ValueError("paired frequency bundles must have distinct content hashes")
    lower = read_episode_bundle(root, lower_bundle)
    upper = read_episode_bundle(root, upper_bundle)
    if lower.episode.split_group_id != upper.episode.split_group_id:
        raise ValueError("paired frequency bundles must share one frozen split group")

    def selected(bundle: object, view_id: str) -> tuple[PhysicalChannelView, UMCDRunProvenance]:
        channel_matches = [view for view in bundle.channel_views if view.view_id == view_id]
        if len(channel_matches) != 1:
            raise ValueError(f"stored bundle must contain exactly one channel view {view_id!r}")
        view = channel_matches[0]
        provenance_matches = [
            record
            for record in bundle.provenance_records
            if record.subject_layer == "oracle_channels"
            and _bound_carrier_hz(record) == view.carrier_hz
            and _scene_geometry_digest(record) is not None
        ]
        if len(provenance_matches) != 1:
            raise ValueError("stored bundle needs one carrier-matched oracle-channel provenance")
        return view, provenance_matches[0]

    lower_view, lower_provenance = selected(lower, lower_view_id)
    upper_view, upper_provenance = selected(upper, upper_view_id)
    geometry_digest = _scene_geometry_digest(lower_provenance)
    if geometry_digest is None:
        raise ValueError("lower bundle provenance has no scene geometry digest")
    return PairedFrequencyViews(
        pair_id=pair_id,
        source_scene_id=source_scene_id,
        parent_episode_id=parent_episode_id,
        split_group_id=lower.episode.split_group_id,
        scene_geometry_sha256=geometry_digest,
        lower_bundle_sha256=lower_bundle.bundle_sha256,
        upper_bundle_sha256=upper_bundle.bundle_sha256,
        lower_view=lower_view,
        lower_provenance=lower_provenance,
        upper_view=upper_view,
        upper_provenance=upper_provenance,
    )
