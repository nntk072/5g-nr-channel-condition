# UMCD release split policy v2

Release Split Policy v2 defines how statistical parent records are assigned to
train, validation, calibration, and test without leaking related observations
across partitions. It is implemented as a reusable policy and lock format; it
does not certify any current corpus as a release or close G7.

The focused identity-gate verification currently passes: Split Policy v2 and
signing tests (82), plus legacy independent-split, ChannelDataset lineage,
experiment resolver, and signed lineage-audit tests (109). Ruff passes on the
corresponding source and test modules. This evidence covers local assignment and
submitted-lineage integrity; it does not establish external signer custody,
authoritative inventory completeness, physical independence, or release
qualification.

The legacy `evaluation.independent_splits` helper also fails closed on identity
inputs: sample IDs, group IDs, the seed, and configured field names must be
nonempty canonical strings with no surrounding whitespace. The builder,
immutable split value, and audit do not trim or stringify malformed identities.
This preserves deterministic behavior and assignment hashes for valid string
IDs while preventing whitespace aliases from silently entering a split. The
ChannelDataset lineage builder likewise requires canonical,
non-padded string dataset, parent, source-family, partition, group, relation,
scene, coordinate-frame, and time-scope IDs before deriving relations; it no
longer converts arbitrary group values with `str()` or silently strips identity
whitespace. Core split-policy constructors, standardized partition metadata,
scene-roster audits, child bindings, and signer/principal inputs also reject
padded identities instead of normalizing them. The canonical v2 registry
remains the release-policy path.

Spatial and temporal relation identities use
`umcd.spatiotemporal-leakage.v2`. Each spatial identity commits to the exact
buffer, coordinate frame, sorted subject set, and positions used to derive its
connected component. Each temporal identity commits to the named clock,
sorted subject set, and exact half-open time windows. Changing the buffer or
footprint values therefore changes the identity even when component
membership stays the same. `audit_spatiotemporal_leakage_relations` re-derives
relations from a complete footprint inventory and verifies exact agreement
with split subjects; the ChannelDataset lineage auditor includes its
input-scoped audit report and footprint digest. `bind_spatiotemporal_audit_to_registry`
attaches that report to a v3 immutable lock, including parent-roster and
footprint digests. New locks containing spatial or time relations cannot be
frozen without this binding. Verify it again with
`verify_registry_spatiotemporal_audit` when auditing the release. Legacy v2
locks remain readable. Hashes do not establish that an inventory is complete
or authoritative; an external signer must review and attest to that inventory.

## Parent and child records

Assign splits to statistical parents. A `NetworkEpisode` is one supported
parent; links, frames, paths, receiver observations, noise realizations,
bandwidth variants, and targets derived from it inherit that parent's split.
A `StandardizedChannelRecord` represents one standardized stochastic draw;
its `sample_id` is the parent ID. Its `run_id` only identifies a generator
batch. A separate opaque `partition_id` determines component identity and
assignment; the standardized adapter derives it from generation seed and
sample index, or accepts an explicit stable value. Profile/category metadata,
source-family labels, targets, and artifact IDs do not enter the partition
hash. A combined release must provide unique partition IDs; collisions reject
instead of silently joining distinct parents.

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

For RT pilot exports, a complete scene-asset closure now yields a stable
`split_group_id` from the asset-set digest, exact spatial block size, and
receiver XY cell. The key is shared across scene renames, reruns, and TX
identities, so repeated physical blocks remain one statistical parent. An
incomplete closure uses a provisional scene-name key and is not evidence of
cross-run identity; production release audits must still derive and verify
site, spatial-buffer, and time-overlap relations from complete lineage.

Older RT `ChannelDataset` files may lack the asset closure even when their
generation report identifies a Git revision and package-source fingerprint.
`scripts/reconstruct_legacy_rt_scene_assets.py` can retrospectively rebuild a
closure from that exact Git tree or from a Sionna-RT wheel whose hash is pinned
in that revision's `uv.lock`. The resulting sidecar binds each reconstruction
to the original NPZ hash, row count, scene configuration, source fingerprint,
lockfile hash, and wheel hash. To use it, pass
`--derive-legacy-rt-split-identities --legacy-scene-assets-manifest PATH` to
`scripts/audit_channel_split_lineage.py`. The migration occurs in memory; the
auditor hashes the original bytes and always returns `NOT_QUALIFIED`. A
retrospective sidecar cannot attest that the original runtime actually used
the locked wheel, establish an external inventory authority, prove corpus
completeness, or justify site independence. Use it for split-structure
preflight only.

`datasets.split_lineage.derive_channel_dataset_split_subjects` is the reusable
pre-split adapter for a supplied collection of named `ChannelDataset` runs. It
requires explicit parent type and source-family metadata, merges repeated
parent IDs, rejects conflicting identity fields, and derives spatial-buffer
and overlapping-time relations from declared footprints before returning
`SplitSubject` records. Its report is always `NOT_QUALIFIED`: it covers only
the inputs supplied to that invocation and does not prove corpus completeness,
source lineage, site relations absent from metadata, external signatures, or
release suitability. Run it over the frozen, hash-inventoried corpus and
review its source provenance and missing-footprint counts before building and
signing a release registry.

The lineage summary also reports deterministically derived leakage-connected
component counts and sizes, without assigning any train/validation/calibration/
test partitions. This exposes whether the supplied independent units can
support a proposed partition before a caller attempts to freeze a policy. The
same `derive_split_components` function supplies the component grouping used by
release-registry assignment, so the preview and frozen lock use identical
relation-union semantics.

The UMCD corpus target-coverage report also records the number of distinct
frozen leakage components contributing to each split × source-family × target
cell. Multiple episode rows in one component increase the observed element
coverage but count as one independent group. This count is input-scoped: it
does not establish that the supplied site, capture, spatial, or temporal
relations are complete or physically correct, and no minimum group threshold
is inferred from the count. The versioned schema at
`datasets/schema/umcd-corpus-audit-inventory.v2.schema.json` requires an
explicit `minimum_independent_groups` for every expected target cell, and the
auditor validates the inventory against that schema. The audit fails when
observed component support is below the frozen minimum, even when repeated
correlated rows provide many target elements. The threshold is a declared
software gate; its scientific suitability must be set by the relevant protocol
and is not established by the audit.

The reusable `evaluation.target_coverage.audit_target_coverage` API requires
each `TargetRecord` to provide `independent_group_id`; it never assumes that a
record is an independent unit merely because it has a distinct record ID. The
canonical corpus auditor passes the frozen split registry's leakage-component
ID. Other callers must supply a group identity derived from their own reviewed
lineage; this API does not authenticate caller-provided IDs.

Generic target-coverage inventory cells can additionally be stratified by
scene, band, topology, antenna configuration, and task. These dimensions are
part of the exact inventory key alongside split, source family, and target;
records in different strata cannot pool support. Legacy callers receive the
literal `unspecified` value for omitted dimensions, which cannot satisfy a
named stratum. Canonical corpus-inventory v3 freezes these dimensions. The
auditor joins scene IDs to the frozen scene relation, validates declared band
IDs against carrier frequency and frozen frequency intervals, derives
topology and antenna-configuration hashes from the episode, and joins each
target ID to an explicit task. It retains the v2 reader for legacy inventories.
An optional detached OpenSSH signature can authenticate the exact v3 inventory
bytes in the `umcd-corpus-inventory-v3@github.com` namespace against a supplied
allowed-signers file. The audit records the signature and trust-file digests.
This authenticates the signer's declaration only: independent key custody,
authoritative inventory review, physical completeness, and qualification
remain external gates.

Canonical corpus-inventory v4 adds a frozen `scene_inventory` roster mapping
each scene realization to its exact parent episode IDs. The corpus auditor
passes that roster through `audit_whole_scene_holdout` and includes its
input-scoped digest and scene-to-component/split mapping under
`checks.split_lineage.whole_scene_holdout`. The v2 and v3 readers remain
available. V4 still depends on an independently reviewed and externally
complete scene roster; a valid signature binds bytes but does not establish
that authority. V4 inventory signatures use the distinct
`umcd-corpus-inventory-v4@github.com` namespace; v3 signatures continue to use
`umcd-corpus-inventory-v3@github.com`.

For an explicit whole-scene holdout check, call
`audit_whole_scene_holdout(registry, scene_subjects, expected_scene_ids=...,
expected_subject_ids=...)`. The expected scene and parent rosters must come
from a separately reviewed inventory. The audit requires exact roster coverage,
an explicit `scene_realization:<scene-id>` relation on each scene parent, and
one connected component and split per scene. Its `PASS_INPUT_SCOPED` report
hashes the registry, expected rosters, and scene memberships and includes the
scene-to-component/split mapping only for those supplied inputs. It cannot
authenticate the inventory, prove the corpus is complete, or establish
independent sites; those remain external G7.08 release checks.

The ChannelDataset lineage adapter derives these relations for Sionna RT rows:
complete scene closures use `scene_realization:asset-set:<sha256>`, which links
the same scene assets across run names and receiver blocks. Incomplete closures
use `scene_realization:provisional-name:<scene>`, a conservative local grouping
that cannot establish cross-run physical-scene identity. A release audit must
reject provisional identities or separately resolve their lineage before
claiming whole-scene holdout.

When a signed split registry is supplied to the saved-dataset lineage audit,
the audit now fails closed if any RT parent still has a provisional scene
identity. Unsigned lineage reports may still surface provisional parents for
diagnosis, but provisional scene-name keys cannot be compared as release-lock
evidence. Resolve complete scene-asset closure and regenerate the parent roster
before signing or auditing a release split lock.

For saved NPZ files, the repeatable entry point is:

```bash
uv run --no-sync python scripts/audit_channel_split_lineage.py \
  --dataset run-a=data/run-a.npz \
  --dataset run-b=data/run-b.npz \
  --spatial-buffer-m 30 \
  --output reports/split-lineage-audit.json
```

The CLI hashes the exact bytes parsed and refuses to overwrite an existing
report. Its input inventory and derived subjects are useful review evidence,
but a complete, independently inventoried corpus and trusted provenance review
are still required before creating the signed lock.

An explicit `--derive-legacy-rt-split-identities` option supports older RT
ChannelDatasets that predate the split-identity fields. It derives identities
only in memory, only for all-RT datasets, and only when each row's receiver and
scene metadata agree with the saved complete scene-asset closure and the
dataset-level block size. The report hashes the original NPZ bytes and records
which inputs and rows were upgraded; the source files are never rewritten.
Without this option, the auditor continues to reject missing identities. This
compatibility path remains diagnostic: it cannot authenticate the declared
scene closure or certify that the supplied inputs are the complete corpus.

For Sionna RT rows, the adapter recomputes and validates the parent ID against
the scene-asset digest (or provisional scene name), exact block size, and
receiver XY grid cell. Complete identities require matching asset closure,
coordinate frame, and finite spatial position; provisional identities require
an explicitly incomplete closure and derive their cell from the recorded RX
position. Contradictory flags, malformed positions, or IDs inconsistent with
the declared physical parent fail closed. This prevents metadata from claiming
a parent different from the one encoded by its scene and receiver location.

## Assignment and immutable locks

Registry locks and signed payloads are read from stable regular-file snapshots with a 256 MiB maximum. Detached OpenSSH signatures are limited to 1 MiB, and trusted `allowed_signers` files to 4 MiB. The reader checks the file identity before and after bounded chunked reads and rejects an input that exceeds its limit. These are resource and snapshot-integrity controls; they do not establish that the supplied trust file is institutionally authoritative or that a release is qualified.

The registry reader also rejects invalid UTF-8/JSON, non-object roots or containers, non-integer schema versions, and fields outside the exact v2 or v3 roster before reconstruction. It then verifies the canonical digest and independently reconstructs assignments. A parseable, correctly signed lock is still only as authoritative as its supplied inventory and signer governance.

`SplitPolicy.release_v2(seed=...)` creates the default four-way fractions
70/10/10/10. `SplitPolicy` can also represent the legacy three-way
train/validation/test contract. The existing `make_independent_split` API
continues to support three-way use. Component IDs and assignments use stable
SHA-256 ordering and do not depend on input record order or Python hash state.
Every partition must have at least one independent component; a small dataset
that cannot support all requested partitions is rejected instead of silently
collapsing a split. The registry constructor's diagnostic reports the required
partition count, observed independent leakage-component count, and exact shortfall
(for example, three components for a four-way policy reports that one more is
needed). Counts are after connected-component reconstruction, so multiple parent
rows linked by scene, site, or spatial-buffer relations still count as one
independent unit. The diagnostic does not merge components or relax the policy
to make a small pilot fit.

Example:

```python
from datasets import (
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    SplitPolicy,
    audit_split_policy_registry,
    bind_split_children,
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
children = [
    SplitChildBinding("pusch-view-0001", standardized_record.sample_id, "pusch_view"),
]
registry = bind_split_children(registry, children)
audit_split_policy_registry(registry, subjects, children)
lock_sha256 = freeze_split_policy_registry("release/split-registry.json", registry)
```

Persist the returned lock digest with release provenance. Existing lock files
are immutable: attempts to replace them with different policy, assignment,
parent, or lineage data fail. Reading validates canonical serialization and
the policy and registry SHA-256 digests by independently rebuilding connected
components, canonical component IDs, deterministic assignments, required
partition presence, child inheritance, and report counts. Re-audit exact
parent and persisted-child coverage and lineage before every release/export.
A lock larger than the reader's 256 MiB bound is rejected before publication,
so the writer cannot create an immutable registry that this implementation
cannot read back.
A child uses `registry.split_for_child(child_id)`; its frozen binding resolves
the parent, and the child cannot supply a split label. Child IDs are disjoint
from parent IDs: registry construction rejects collisions so parent-first
consumers cannot mistake a child for a parent and bypass its binding.

The SHA-256 values provide content integrity and deterministic identity; they
do not prove who created or approved a lock. Before model tuning, sign the
exact immutable registry bytes with the dedicated OpenSSH namespace
`umcd-split-policy-v2@github.com`. For example, an authorized custodian can run:

```sh
ssh-keygen -Y sign -n umcd-split-policy-v2@github.com -f "$UMCD_SPLIT_SIGNING_KEY" \
  release/split-registry.json
```

For a saved ChannelDataset release candidate, construct the v3 registry from
the exact saved rows so the complete spatial/time footprint inventory is
bound into the lock:

```python
from datasets import (
    SplitPolicy,
    freeze_split_policy_registry,
    make_channel_dataset_split_registry,
)

registry, lineage_report = make_channel_dataset_split_registry(
    named_channel_datasets,
    spatial_buffer_m=30.0,
    policy=SplitPolicy.release_v2(seed="umcd-release-1"),
)
freeze_split_policy_registry("release/split-registry.json", registry)
```

After signing that lock, rerun the saved-file lineage audit with the signed
lock and external trust file to recheck exact parent coverage and recompute
the footprint digest from the files being audited:

```bash
uv run --no-sync python scripts/audit_channel_split_lineage.py \
  --dataset run-a=data/run-a.npz \
  --dataset run-b=data/run-b.npz \
  --spatial-buffer-m 30 \
  --split-registry release/split-registry.json \
  --split-registry-signature release/split-registry.json.sig \
  --split-registry-allowed-signers /etc/umcd/split-allowed-signers \
  --split-registry-signer-id release-approver@example.org \
  --output reports/split-lineage-audit.json
```

The report explicitly states that child data coverage is outside this
ChannelDataset audit. Both commands produce implementation/audit evidence;
they do not approve an incomplete inventory or qualify a corpus.

This creates `release/split-registry.json.sig`. Load it only after verification
against an institution-managed `allowed_signers` file that is supplied
separately from the release bundle:

```python
from datasets import verify_signed_split_policy_registry

registry = verify_signed_split_policy_registry(
    "release/split-registry.json",
    "release/split-registry.json.sig",
    trusted_allowed_signers_path="/etc/umcd/trusted_allowed_signers",
    signer_id="release-approver@example.org",
)
```

Verification snapshots the lock, signature, and trust file before invoking
`ssh-keygen`; it verifies the exact bytes before parsing the lock. The trusted
keyring must be governed outside the release bundle. Key generation, private
key custody, signer authorization, and key revocation remain institutional
responsibilities. A lock without a valid signature from the configured trust
store is not approved, and no existing retrospective model result is promoted
to a preregistered split by signing it later.

The frozen UMCD corpus inventory can declare a detached registry signature
using `external_lock_attestation.status: "provided"`, a relative
`signature_path`, and a `signer_id`. Verify it during the corpus audit by
supplying the independently governed trust file:

```bash
uv run --no-sync --locked python scripts/audit_umcd_corpus.py \
  release/inventory.json \
  --output release/corpus-audit.json \
  --trusted-allowed-signers /etc/umcd/trusted_allowed_signers
```

The audit checks that the signature covers the exact registry digest declared
in the inventory and records the signature, principal, namespace, and supplied
trust-file digests. Its `signature_valid_against_supplied_trust_file` status
only reports cryptographic verification against those bytes; it cannot prove
how the trust file was governed or establish legal approval, corpus
completeness, or release qualification. The audit remains
`qualification: not_claimed`.

`registry.report()` reports independent component and parent counts per split,
including counts by source family. It reports effective independent units,
not the much larger count of correlated child observations. The report does
not replace spatial-buffer, near-duplicate, target-coverage, or privacy audits.

`evaluation.near_duplicate_audit.audit_near_duplicates` provides a local cross-split
CFR/geometry candidate scan. It requires canonical sample, split, and geometry-hash
identities so whitespace aliases cannot create distinct split labels. Its matches are
review candidates rather than automatic leakage findings. The canonical corpus auditor
calls this scan with split assignments derived from the loaded frozen registry. A
zero-norm comparison with no independent geometry match is reported as incomplete in
corpus report v6. This does not establish geometric near-duplicate coverage or
authoritative inventory completeness; external signer trust and reviewer disposition
remain separate release gates.

The UMCD corpus inventory may also declare `nr_trajectory_inventory` rows
with `trajectory_id`, `parent_episode_id`, canonical trajectory-manifest path,
and manifest SHA-256, plus `nr_expected_coverage` cells containing
`split_id`, `family_id`, and `target_id`. The auditor derives each trajectory's
split and family from its inventoried parent episode and the frozen registry;
these values are not accepted from the trajectory row. If NR manifests exist,
their exact set must be inventoried. Expected cells with no trajectory remain
in the report with zero support. `nr_trajectory_target_support` reports
trajectory/record presence, missing records, selected and masked elements,
and implicit all-valid masks. This input-scoped count does not prove expected
inventory completeness, label validity, or scientific qualification. Its
versioned report shape is `datasets/schema/nr-target-support-audit.v1.schema.json`.
The inventory fragment is specified by
`datasets/schema/nr-trajectory-inventory-extension.v1.schema.json`.

## Candidate registry build CLI

Build an immutable candidate registry from named saved `ChannelDataset` files
and a canonical `umcd.split_policy.v2` policy file:

```bash
uv run --no-sync python -m scripts.build_channel_split_registry \\
  --dataset run-a=data/run-a.npz \\
  --dataset run-b=data/run-b.npz \\
  --policy release/split-policy.json \\
  --spatial-buffer-m 30 \\
  --registry-output release/split-registry.json \\
  --report-output release/split-registry-build-report.json
```

The builder hashes each exact input snapshot and the policy file, bounds both
compressed and uncompressed NPZ size, derives the spatial/time leakage graph,
and refuses to overwrite either output. RT-only legacy ChannelDatasets with no
split fields are migrated only when the saved scene configuration and complete,
internally consistent asset manifest are present. The report records this
migration per input in build-report v2 while retaining the hash of the original
archive bytes. Partial identities are not repaired. The report and registry
are candidate artifacts marked `NOT_QUALIFIED`. A reviewer must still verify the complete
source inventory, externally sign the final lock under the configured
namespace, and supply trusted signer policy before release audit. The build
report is not an external inventory attestation. The v1 report schema remains
available for historical reports; new builds use
`datasets/schema/channel-split-registry-build-report.v2.schema.json`.

## Scope and release status

The v2 data structures, canonical reconstruction, opaque partition identities,
explicit child bindings, deterministic grouping, lock, audit, and legacy
three-way compatibility have adversarial acceptance tests. A real release still
needs verified lineage adapters for every included family, corpus-wide
spatial/duplicate/target audits, a frozen registry generated from the final
release candidates, and the rest of G7. The standard experiment runner can now
consume a prospectively signed registry when every row carries an explicit
`split_subject_id`; it requires exact frozen-parent coverage and records the
trusted signer and artifact digests. This integration does not establish the
correctness of those IDs or corpus lineage. In particular, these utilities do
not approve RT production D1 generation while G3 remains open.

Inventory-backed dataset reads open each declared path component relative to
an already-open directory descriptor with no-follow semantics. The auditor
hashes and parses the bytes read from that descriptor, preventing a swapped
parent directory from redirecting a read after inventory path validation. This
is an input-integrity guarantee; it does not establish that the inventory is
complete, authoritative, or scientifically sufficient.

Signed child-artifact roster v2 requires a `ChannelDataset` NPZ for every
frozen child. Its rows are checked against the frozen child, kind, and parent
binding. A recognized `nr_trajectory_store_manifest_json` artifact is traversed
through its trajectory and record hashes, and every contained record must carry
the same split subject, child, and child-kind provenance. Unknown supplemental
artifact types are still byte-hash checked but receive no semantic validation;
the audit reports them separately. These checks establish the integrity and
declared lineage of listed artifacts, not completeness of the release inventory
or external authority over its contents.

Signed child-artifact roster v3 additionally names one `artifact_root` directory
below the roster file. The auditor rejects symlinks, hard links, special files,
directory changes during enumeration, excessive depth, and excessive file
counts, then requires the roster paths to equal every regular file beneath that
root. This closes omission of files inside that declared subtree. It does not
prove that the subtree itself contains every artifact required by an
authoritative release inventory; that inventory and signer custody remain
separate release gates.

Roster v4 adds the immutable UMCD episode store. A typed episode manifest is
read through the canonical store reader with a conservative 512 MiB
shard-accounting ceiling,
and its episode ID must equal the frozen split parent subject. Every episode
layer object under the roster's exact `artifact_root` must be listed as a
content object and match exactly one manifest reference, including its layer
name and content-addressed path. The v4 child audit and outer lineage report
are versioned separately; v1-v3 roster and report behavior is retained. This
checks the signed roster's declared episode descendants, not completeness of
the authoritative corpus inventory or external signer custody. The accounting
ceiling is a conservative estimate, not measured process RSS.

Roster v4 content objects are joined against every episode manifest in the
frozen child roster and retain their original single-child owner field. Roster
v5 adds `child_ids` for content objects so content-addressed deduplication can
be represented without assigning a shared object to one child. The auditor
requires this sorted, unique owner list to equal every manifest that references
the object. Episode manifests remain bound to one child and one frozen parent;
unlisted owners, missing owners, duplicate per-child references, and wrong
owner IDs fail admission. Roster v1-v4 retain their historical interpretation.
