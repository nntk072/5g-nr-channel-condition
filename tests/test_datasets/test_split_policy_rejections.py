"""Split policy: each rejected leakage, footprint, subject and policy input has its stated reason."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SpatiotemporalFootprint,
    SplitParentType,
    SplitSubject,
    audit_spatiotemporal_leakage_relations,
    derive_spatiotemporal_leakage_relations,
)


def _subject(subject_id: str, *relations, parent=SplitParentType.NETWORK_EPISODE, family="f") -> SplitSubject:
    return SplitSubject(
        subject_id=subject_id,
        parent_type=parent,
        source_family=family,
        relations=tuple(relations),
    )


def test_leakage_relation_namespace_and_text_form() -> None:
    relation = LeakageRelation(LeakageNamespace.SITE, "site-a")
    assert relation.qualified_identity == f"{LeakageNamespace.SITE.value}:site-a"
    with pytest.raises(ValueError, match="unsupported leakage namespace"):
        LeakageRelation("not-a-namespace", "site-a")


def test_subject_parent_type_and_relation_types_are_checked() -> None:
    with pytest.raises(ValueError, match="unsupported split parent type"):
        SplitSubject("s", "not-a-parent", "f")
    with pytest.raises(TypeError, match="relations must contain LeakageRelation values"):
        SplitSubject("s", SplitParentType.NETWORK_EPISODE, "f", relations=("site-a",))
    relation = LeakageRelation(LeakageNamespace.SITE, "site-a")
    with pytest.raises(ValueError, match="cannot repeat a leakage relation"):
        SplitSubject("s", SplitParentType.NETWORK_EPISODE, "f", relations=(relation, relation))


def test_footprint_positions_must_be_three_dimensional() -> None:
    with pytest.raises(ValueError, match="positions_m must contain 3D points"):
        SpatiotemporalFootprint("s", positions_m=((1.0, 2.0),))


def test_footprint_frame_and_time_scope_must_accompany_their_data() -> None:
    with pytest.raises(ValueError, match="coordinate_frame_id requires at least one position"):
        SpatiotemporalFootprint("s", coordinate_frame_id="frame")
    with pytest.raises(ValueError, match="coordinate_frame_id is required when positions_m are present"):
        SpatiotemporalFootprint("s", positions_m=((0.0, 0.0, 0.0),))
    with pytest.raises(ValueError, match="time_scope_id is required"):
        SpatiotemporalFootprint(
            "s",
            positions_m=((0.0, 0.0, 0.0),),
            coordinate_frame_id="frame",
            start_time_s=0.0,
            end_time_s=1.0,
        )


def test_derivation_rejects_non_footprints_duplicates_and_bad_buffers() -> None:
    subject = _subject("a")
    footprint = SpatiotemporalFootprint("a", positions_m=((0.0, 0.0, 0.0),), coordinate_frame_id="f")
    with pytest.raises(ValueError, match="spatial_buffer_m must be finite and non-negative"):
        derive_spatiotemporal_leakage_relations([footprint], spatial_buffer_m=-1.0)
    with pytest.raises(ValueError, match="spatial_buffer_m must be finite and non-negative"):
        derive_spatiotemporal_leakage_relations([footprint], spatial_buffer_m=math.inf)
    with pytest.raises(TypeError, match="footprints must contain SpatiotemporalFootprint values"):
        derive_spatiotemporal_leakage_relations(["not-a-footprint"], spatial_buffer_m=1.0)
    with pytest.raises(ValueError, match="duplicate spatiotemporal footprint"):
        derive_spatiotemporal_leakage_relations([footprint, footprint], spatial_buffer_m=1.0)


def test_audit_requires_matching_footprint_and_subject_rosters() -> None:
    footprint = SpatiotemporalFootprint("a", positions_m=((0.0, 0.0, 0.0),), coordinate_frame_id="f")
    with pytest.raises(TypeError, match="subjects must contain SplitSubject values"):
        audit_spatiotemporal_leakage_relations([footprint], ["not-a-subject"], spatial_buffer_m=1.0)
    with pytest.raises(ValueError, match="duplicate split subject in footprint audit"):
        audit_spatiotemporal_leakage_relations([footprint], [_subject("a"), _subject("a")], spatial_buffer_m=1.0)
    with pytest.raises(ValueError, match="must cover exactly the same parents"):
        audit_spatiotemporal_leakage_relations([footprint], [_subject("b")], spatial_buffer_m=1.0)
