"""Contracts for pairing independent band-specific complex channel solves."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from datasets.channel_view import ChannelDelayReference, ChannelPowerScale, PhysicalChannelView
from datasets.frequency_pair import PairedFrequencyViews
from datasets.provenance import UMCDRunProvenance

_GEOMETRY_SHA256 = "a" * 64


def _view(view_id: str, carrier_hz: float) -> PhysicalChannelView:
    return PhysicalChannelView(
        view_id=view_id,
        link_id="gNB-1-to-UE-1",
        channel=np.ones((1, 1, 1, 3), dtype=np.complex64),
        frame_times_s=(0.0,),
        rx_port_ids=("UE-V",),
        tx_port_ids=("gNB-V",),
        carrier_hz=carrier_hz,
        subcarrier_offsets_hz=np.asarray([-15_000.0, 0.0, 15_000.0]),
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )


def _provenance(
    record_id: str, carrier_hz: float, geometry_sha256: str = _GEOMETRY_SHA256
) -> UMCDRunProvenance:
    return UMCDRunProvenance(
        record_id=record_id,
        subject_layer="oracle_channels",
        source_family="F04-RT-DYNAMIC",
        producer="Sionna RT PathSolver",
        package_source_sha256="b" * 64,
        software_versions={"sionna-rt": "2.2.0"},
        backend={"backend": "llvm"},
        scene_asset_sha256={"scene.xml": "c" * 64},
        seed=12,
        physical_config={
            "carrier_hz": carrier_hz,
            "scene_geometry": {"scene_geometry_sha256": geometry_sha256},
        },
        solver_config={"max_depth": 4},
        material_config={"concrete": {"frequency_hz": carrier_hz}},
    )


def _pair() -> PairedFrequencyViews:
    return PairedFrequencyViews(
        pair_id="paired-drop-1",
        source_scene_id="munich-block-3",
        parent_episode_id="drop-1",
        split_group_id="site-3",
        scene_geometry_sha256=_GEOMETRY_SHA256,
        lower_view=_view("view-3p5", 3.5e9),
        lower_provenance=_provenance("run-3p5", 3.5e9),
        upper_view=_view("view-28", 28e9),
        upper_provenance=_provenance("run-28", 28e9),
    )


def test_pair_manifest_binds_two_band_specific_solves_and_same_scene():
    jsonschema = pytest.importorskip("jsonschema")
    manifest = _pair().to_dict()
    schema = json.loads(
        (
            Path(__file__).parents[2] / "datasets/schema/umcd-paired-frequency-views.v1.schema.json"
        ).read_text()
    )

    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert [view["carrier_hz"] for view in manifest["views"]] == [3.5e9, 28e9]
    assert (
        manifest["views"][0]["material_config_sha256"]
        != manifest["views"][1]["material_config_sha256"]
    )
    assert manifest["phase_coherence_claim"] == "not_asserted"


def test_pair_rejects_carrier_provenance_mismatch():
    with pytest.raises(ValueError, match="carrier"):
        PairedFrequencyViews(
            pair_id="paired-drop-1",
            source_scene_id="munich-block-3",
            parent_episode_id="drop-1",
            split_group_id="site-3",
            scene_geometry_sha256=_GEOMETRY_SHA256,
            lower_view=_view("view-3p5", 3.5e9),
            lower_provenance=_provenance("run-3p5", 3.6e9),
            upper_view=_view("view-28", 28e9),
            upper_provenance=_provenance("run-28", 28e9),
        )


def test_pair_rejects_different_scene_geometry():
    with pytest.raises(ValueError, match="scene geometry"):
        PairedFrequencyViews(
            pair_id="paired-drop-1",
            source_scene_id="munich-block-3",
            parent_episode_id="drop-1",
            split_group_id="site-3",
            scene_geometry_sha256=_GEOMETRY_SHA256,
            lower_view=_view("view-3p5", 3.5e9),
            lower_provenance=_provenance("run-3p5", 3.5e9),
            upper_view=_view("view-28", 28e9),
            upper_provenance=_provenance("run-28", 28e9, "d" * 64),
        )


def test_pair_rejects_non_rt_source_family():
    pair = _pair()
    with pytest.raises(ValueError, match="RT solve provenance"):
        PairedFrequencyViews(
            pair_id=pair.pair_id,
            source_scene_id=pair.source_scene_id,
            parent_episode_id=pair.parent_episode_id,
            split_group_id=pair.split_group_id,
            scene_geometry_sha256=pair.scene_geometry_sha256,
            lower_view=pair.lower_view,
            lower_provenance=replace(pair.lower_provenance, source_family="measurement.archive"),
            upper_view=pair.upper_view,
            upper_provenance=pair.upper_provenance,
        )
