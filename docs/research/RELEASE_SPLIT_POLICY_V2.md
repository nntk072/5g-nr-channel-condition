# UMCD release split policy v2

Release Split Policy v2 defines how statistical parent records are assigned to
train, validation, calibration, and test without leaking related observations
across partitions. It is implemented as a reusable policy and lock format; it
does not certify any current corpus as a release or close G7.

## Parent and child records

Assign splits to statistical parents. A `NetworkEpisode` is one supported
parent; links, frames, paths, receiver observations, noise realizations,
bandwidth variants, and targets derived from it inherit that parent's split.
A `StandardizedChannelRecord` represents one standardized stochastic draw;
its `sample_id` is the parent ID. Its `run_id` only identifies a generator
batch. Profile category and LOS-component flags are not leakage relations and
must not determine the split.

Other families can be represented by `SplitSubject` directly. Include only
known relations. An absent relation is an empty tuple, never a shared sentinel
such as `"N/A"`. Relation keys are `(namespace, identity)`, so (for example)
`site:x` and `trajectory_session:x` do not join. The supported namespaces are
asset family, site, scene realization, trajectory session, capture session,
counterfactual family, material realization, and equipment family. Shared
relations are closed transitively into connected components before assignment.

The network adapter conservatively maps the legacy `split_group_id` to
`asset_family`. `family_id` is reporting metadata only; provide other site,
scene, trajectory, material, or equipment relations explicitly when that
lineage is known. Standardized draws have no implicit shared relation.

## Assignment and immutable locks

`SplitPolicy.release_v2(seed=...)` creates the default four-way fractions
70/10/10/10. `SplitPolicy` can also represent the legacy three-way
train/validation/test contract. The existing `make_independent_split` API
continues to support three-way use. Component IDs and assignments use stable
SHA-256 ordering and do not depend on input record order or Python hash state.
Every partition must have at least one independent component; a small dataset
that cannot support all requested partitions is rejected instead of silently
collapsing a split.

Example:

```python
from datasets import (
    LeakageNamespace,
    LeakageRelation,
    SplitPolicy,
    audit_split_policy_registry,
    freeze_split_policy_registry,
    make_split_policy_registry,
    network_episode_subject,
    standardized_draw_subject,
)

policy = SplitPolicy.release_v2(seed="umcd-release-1")
subjects = [
    network_episode_subject(
        episode,
        relations=(LeakageRelation(LeakageNamespace.SITE, "site-west"),),
    ),
    standardized_draw_subject(standardized_record),
]
registry = make_split_policy_registry(subjects, policy=policy)
audit_split_policy_registry(registry, subjects)
lock_sha256 = freeze_split_policy_registry("release/split-registry.json", registry)
```

Persist the returned lock digest with release provenance. Existing lock files
are immutable: attempts to replace them with different policy, assignment,
parent, or lineage data fail. Reading validates canonical serialization and
the policy and registry SHA-256 digests. Re-audit exact parent coverage and
lineage before every release/export. A child uses
`registry.split_for_child(parent_id)`; do not create a new split assignment
for it.

`registry.report()` reports independent component and parent counts per split,
including counts by source family. It reports effective independent units,
not the much larger count of correlated child observations. The report does
not replace spatial-buffer, near-duplicate, target-coverage, or privacy audits.

## Scope and release status

The v2 data structures, deterministic grouping, assignment, lock, audit, and
legacy three-way compatibility have acceptance tests. A real release still
needs verified lineage adapters for every included family, corpus-wide
spatial/duplicate/target audits, a frozen registry generated from the final
release candidates, and the rest of G7. In particular, these utilities do not
approve RT production D1 generation while G3 remains open.
