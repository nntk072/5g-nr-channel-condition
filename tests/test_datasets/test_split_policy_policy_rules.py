"""Split policy contract: fractions, namespaces, serialized form and component derivation."""

from __future__ import annotations

from typing import Any

import pytest
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
)
from datasets.split_policy.policy import (
    SpatiotemporalFootprint,
    _derive_registry_assignments,
    derive_spatiotemporal_leakage_relations,
    derive_split_components,
)


def _subject(
    subject_id: str, *relations: tuple[LeakageNamespace, str], **kwargs: Any
) -> SplitSubject:
    return SplitSubject(
        subject_id=subject_id,
        parent_type=SplitParentType.NETWORK_EPISODE,
        source_family=kwargs.get("family", "family-a"),
        relations=tuple(LeakageRelation(namespace, identity) for namespace, identity in relations),
    )


THREE_WAY = {"train": 0.6, "validation": 0.2, "test": 0.2}


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        pytest.param(
            {"train": 0.5, "test": 0.5}, "three-way or four-way split set", id="missing-validation"
        ),
        pytest.param(
            {"train": 0.0, "validation": 0.5, "test": 0.5}, "finite and positive", id="zero"
        ),
        pytest.param(
            {"train": 1.0, "validation": 0.5, "test": 0.5}, "finite and positive", id="one"
        ),
        pytest.param(
            {"train": float("nan"), "validation": 0.5, "test": 0.5}, "finite and positive", id="nan"
        ),
        pytest.param({"train": 0.7, "validation": 0.2, "test": 0.2}, "sum to one", id="sum"),
        pytest.param(
            {"train": True, "validation": 0.5, "test": 0.5}, "must be real numbers", id="bool"
        ),
    ],
)
def test_split_fractions_must_be_a_valid_partition(fractions, message) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        SplitPolicy(seed="seed-a", split_fractions=fractions)


def test_fraction_overflow_is_reported_as_invalid() -> None:
    huge = 10**400
    with pytest.raises(ValueError, match="finite and positive"):
        SplitPolicy(seed="seed-a", split_fractions={"train": huge, "validation": 0.5, "test": 0.5})


def test_relation_namespaces_must_be_nonempty_and_unique() -> None:
    with pytest.raises(ValueError, match="non-empty and unique"):
        SplitPolicy(seed="seed-a", split_fractions=THREE_WAY, relation_namespaces=())
    with pytest.raises(ValueError, match="non-empty and unique"):
        SplitPolicy(
            seed="seed-a",
            split_fractions=THREE_WAY,
            relation_namespaces=(LeakageNamespace.SITE, LeakageNamespace.SITE),
        )


def test_component_algorithm_is_pinned() -> None:
    with pytest.raises(ValueError, match="unsupported component assignment algorithm"):
        SplitPolicy(seed="seed-a", split_fractions=THREE_WAY, component_algorithm="other")


def test_policy_round_trips_through_its_canonical_dictionary() -> None:
    policy = SplitPolicy.release_v2(seed="seed-a")
    restored = SplitPolicy.from_dict(policy.to_dict())
    assert restored.to_dict() == policy.to_dict()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda d: d.update(schema="umcd.split_policy.v9"),
            "unsupported UMCD split policy schema",
            id="schema",
        ),
        pytest.param(
            lambda d: d.update(schema_version=3),
            "unsupported UMCD split policy schema",
            id="schema-version",
        ),
        pytest.param(lambda d: d.update(seed="changed"), "hash is invalid", id="hash"),
    ],
)
def test_serialized_policy_is_checked_before_use(mutate, message) -> None:
    payload = SplitPolicy.release_v2(seed="seed-a").to_dict()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        SplitPolicy.from_dict(payload)


def test_serialized_policy_must_be_its_canonical_form() -> None:
    import hashlib
    import json

    payload = SplitPolicy.release_v2(seed="seed-a").to_dict()
    payload["component_algorithm"] = "sha256-connected-component-v2"
    payload["relation_namespaces"] = list(reversed(payload["relation_namespaces"]))
    core = {key: value for key, value in payload.items() if key != "policy_sha256"}
    payload["policy_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="not canonical"):
        SplitPolicy.from_dict(payload)


def test_components_group_subjects_that_share_a_relation() -> None:
    subjects = [
        _subject("a", (LeakageNamespace.SITE, "site-1")),
        _subject("b", (LeakageNamespace.SITE, "site-1")),
        _subject("c"),
    ]
    components = derive_split_components(subjects)
    assert [tuple(item.subject_id for item in component) for component in components] == [
        ("a", "b"),
        ("c",),
    ]


@pytest.mark.parametrize(
    ("subjects", "error", "message"),
    [
        pytest.param([], ValueError, "at least one split subject", id="empty"),
        pytest.param(
            ["not-a-subject"], TypeError, "must contain SplitSubject values", id="wrong-type"
        ),
        pytest.param(
            [_subject("a"), _subject("a")],
            ValueError,
            "duplicate split subject ID: a",
            id="duplicate",
        ),
    ],
)
def test_components_reject_malformed_subject_sets(subjects, error, message) -> None:
    with pytest.raises(error, match=message):
        derive_split_components(subjects)


def test_registry_assignment_reconstruction_rejects_inconsistent_inputs(monkeypatch) -> None:
    import datasets.split_policy.policy as policy_module

    valid_subjects = {f"s{index}": _subject(f"s{index}") for index in range(4)}
    policy = SplitPolicy.release_v2(seed="assignment-check")
    with pytest.raises(TypeError, match="policy must be a SplitPolicy"):
        _derive_registry_assignments(valid_subjects, "invalid")
    with pytest.raises(ValueError, match="at least one split subject"):
        _derive_registry_assignments({}, policy)
    with pytest.raises(TypeError, match="must contain SplitSubject values"):
        _derive_registry_assignments({"a": object()}, policy)
    with pytest.raises(ValueError, match="mapping keys must equal subject IDs"):
        _derive_registry_assignments({"wrong": _subject("right")}, policy)

    duplicate_partition_subjects = {
        "a": SplitSubject("a", SplitParentType.NETWORK_EPISODE, "family-a", partition_id="shared"),
        "b": SplitSubject("b", SplitParentType.NETWORK_EPISODE, "family-a", partition_id="shared"),
    }
    with pytest.raises(ValueError, match="partition_id values must uniquely identify"):
        _derive_registry_assignments(duplicate_partition_subjects, policy)

    excluded_relation = _subject("excluded", (LeakageNamespace.SITE, "site-1"))
    restricted_policy = SplitPolicy(
        seed="restricted",
        split_fractions=THREE_WAY,
        relation_namespaces=(LeakageNamespace.SCENE_REALIZATION,),
    )
    with pytest.raises(ValueError, match="relations excluded by policy"):
        _derive_registry_assignments({"excluded": excluded_relation}, restricted_policy)

    monkeypatch.setattr(
        policy_module,
        "make_independent_split",
        lambda *_args, **_kwargs: type("Assignment", (), {"group_to_split": {"only": "train"}})(),
    )
    with pytest.raises(ValueError, match="omitted a required split partition"):
        _derive_registry_assignments(valid_subjects, policy)

    malformed_subject = _subject("malformed")
    object.__setattr__(malformed_subject, "partition_id", None)
    with pytest.raises(ValueError, match="missing its canonical partition ID"):
        _derive_registry_assignments({"malformed": malformed_subject}, restricted_policy)


def test_spatial_relation_lookup_handles_overflow_and_repeated_component_roots() -> None:
    footprints = (
        SpatiotemporalFootprint("a", ((1e308, 0.0, 0.0), (1e308, 0.0, 0.0)), "map"),
        SpatiotemporalFootprint("b", ((1e308, 0.0, 0.0),), "map"),
    )
    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=1e-308)

    assert relations["a"] == relations["b"]


def test_spatiotemporal_time_groups_keep_disjoint_singletons_unlinked() -> None:
    footprints = (
        SpatiotemporalFootprint("first", time_scope_id="clock", start_time_s=0, end_time_s=1),
        SpatiotemporalFootprint("second", time_scope_id="clock", start_time_s=1, end_time_s=2),
        SpatiotemporalFootprint("third", time_scope_id="clock", start_time_s=4, end_time_s=5),
    )

    relations = derive_spatiotemporal_leakage_relations(footprints, spatial_buffer_m=0.0)

    assert all(not relations[subject_id] for subject_id in ("first", "second", "third"))


def test_spatial_relation_lookup_supports_zero_buffer_and_empty_inventories() -> None:
    linked = derive_spatiotemporal_leakage_relations(
        (
            SpatiotemporalFootprint("left", ((0.0, 0.0, 0.0),), "map"),
            SpatiotemporalFootprint("right", ((0.0, 0.0, 0.0),), "map"),
        ),
        spatial_buffer_m=0.0,
    )
    singleton = derive_spatiotemporal_leakage_relations(
        (SpatiotemporalFootprint("alone", ((3.0, 4.0, 5.0),), "map"),),
        spatial_buffer_m=0.0,
    )

    assert linked["left"] == linked["right"]
    assert singleton["alone"] == ()
    assert derive_spatiotemporal_leakage_relations((), spatial_buffer_m=0.0) == {}
