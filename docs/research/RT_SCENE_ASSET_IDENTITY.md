# RT scene asset identity

`integrations/sionna/scene_assets.py` resolves a scene XML's local references
and hashes the XML, any adjacent scene manifest, referenced mesh files,
Wavefront OBJ material libraries, and referenced MTL textures. Each file is
parsed and hashed through one stable descriptor, in bounded chunks, with
mutation checks before and after inspection. The default limits are 2 GiB per
asset and 8 GiB for the complete closure; callers can set explicit larger
limits. The remaining total-set allowance is applied to each next asset before
that file is parsed or hashed, so exceeding the closure budget does not first
consume the per-file allowance. A canonical asset-set hash is independent of
the checkout's absolute root; the absolute scene path remains available as
provenance. Unresolved
URI/data resources are listed and make the manifest `complete: false` instead
of being treated as hashed evidence. OBJ/MTL dependency lines are capped at
1 MiB so malformed or pathological text cannot force an unbounded line read.

Static RT datasets keep the full file inventory once in their generation
report and place the set hash/count/completeness flag in each sample row.
Trajectory records carry the compact identity; their generation sidecar stores
the full inventory. This keeps artifact size controlled while making scene
asset changes detectable. Runtime backend and solver configuration are stored
separately, so the scene asset hash does not pretend to identify carrier,
materials converted at runtime, solver settings, or random seeds.

The resolver follows XML `filename` references and OBJ/MTL dependencies. It
does not parse arbitrary custom plugin formats or hash runtime-generated
material state. Scene XML/assets and the Sionna RT/Mitsuba/Dr.Jit versions must
therefore be retained together for replay. A complete local hash is
reproducibility provenance, not evidence that geometry or radio materials are
physically qualified.

## Pinned ODbL scene source snapshot

Run `loslab-audit-odbl-scene-source /path/to/sionna-checkout audit.json` to
compare the Sionna root commit and RT submodule against the accepted ODbL policy,
record Git tree/blob inventory digests and the last XML history event, and
recompute the three recursive scene asset hashes. The command returns success
only when both source checkouts are clean, both commits match policy, and every
asset closure matches. It performs no scene loading or propagation generation.

The report is source-history evidence only. Git timestamps are not OSM extract
timestamps; missing OSM object IDs, conversion records, and per-file third-party
license closure stay explicitly unresolved. The report always sets
`public_release_eligible` to false. Continue to use the signed release-specific
manifest and independent rights review before any public redistribution.
