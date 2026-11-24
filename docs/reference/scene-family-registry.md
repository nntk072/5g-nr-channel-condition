# Scene-family registry and candidate sampling

`config/scenes/scene-family-registry.v1.json` binds the Unified Multipath Plan
S00–S11 taxonomy to candidates found in the byte-pinned scene asset/material
audit. The inventory names asset-tree digests, source-record digests, source
kind, generator/site independence group, license state, and separate research
and public-release states. Its schema is
`datasets/schema/scene-family-registry.v1.schema.json`.

The present snapshot contains the three City41/42/43 procedural variants and
Etoile, Florence, and Munich. The three City scenes share one generator group
and therefore contribute one independent group. The built-in stylized
street-canyon fixture is retained as an observed
candidate, but its per-scene license was not independently recorded, so the
sampler excludes it. Etoile, Florence, and Munich are candidates for internal
research under the accepted ODbL policy; their public release remains uncleared
until the policy's provenance, attribution, third-party-rights, source-offer,
and artifact-classification requirements are met. No family or candidate is
scientifically qualified by this catalog.

Families without evidence-backed candidates remain present with an explicit
`UNSUPPORTED_NO_CANDIDATE` state and count shortfall. Candidate sampling uses a
frozen integer seed and SHA-256 ranking over independence groups; it selects at
most one candidate from each group. The report binds the canonical registry
digest and the sampler/CLI source digests and has its own schema.

```bash
uv run --no-sync --locked python -m scripts.sample_scene_families \
  --seed 7103 \
  --candidates-per-family 1 \
  --output reports/scene-family-sample.json
```

Exit code 2 means at least one family has a candidate shortfall; the report is
still written so the missing families remain visible. Existing output paths
are never overwritten. The direct `integrations.sionna.adapter.generate_raytraced()`
producer accepts `scene_family_candidate_id`; it checks the loaded asset-tree
digest and internal-use state, then writes the candidate ID, registry digest,
family, and canonical site/asset-family/exact-scene relations to each generated
row. These fields flow into the existing split-lineage builder and signed
`SplitPolicyRegistry`; they do not create a parallel split allocator. The
cached entry point binds candidate identity into its cache key and per-run UMCD
provenance. Static and trajectory UMCD producers also store leakage relations
on each episode; the immutable episode store round-trips them and
`network_episode_subject()` passes them to the same signed split policy.
`datasets/schema/umcd-episode-metadata.v1.schema.json` documents this optional
lineage field, and the episode writer validates new metadata against that
schema while the reader continues to accept earlier v1 objects without it.
Candidate selection remains opt-in on each producer. A new producer route must
resolve candidates against loaded asset bytes and preserve the resulting
relations, provenance, and run identity.

The catalog is a candidate inventory only. It does not establish full family
coverage, independent measured sites, ODbL redistribution clearance, scene
quality/calibration, or any // qualification.
