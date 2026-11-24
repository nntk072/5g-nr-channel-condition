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
from datasets.split_policy.policy import derive_split_components


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
