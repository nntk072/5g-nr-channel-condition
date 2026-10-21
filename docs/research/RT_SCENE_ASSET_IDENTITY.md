# RT scene asset identity

`integrations/sionna/scene_assets.py` resolves a scene XML's local references
and hashes the XML, any adjacent scene manifest, referenced mesh files,
Wavefront OBJ material libraries, and referenced MTL textures. A canonical
asset-set hash is independent of the checkout's absolute root; the absolute
scene path remains available as provenance. Unresolved URI/data resources are
listed and make the manifest `complete: false` instead of being treated as
hashed evidence.

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
