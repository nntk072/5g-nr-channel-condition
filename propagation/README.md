# propagation

**Purpose.** Geometry, visibility and physics primitives for radio propagation
in deterministic urban scenes, plus diagnostics on whether delay-path
parameters are identifiable from a finite bandwidth.

**Public API.**

- `propagation.geometry` — parameterized urban-canyon scenes written as Mitsuba
  XML with OBJ meshes.
- `propagation.visibility`, `propagation.scene_visibility` — independent
  segment-versus-triangle visibility over explicit meshes.
- `propagation.physics` — unprivileged channel observables and mathematical
  transformations.
- `propagation.identifiability` — finite-bandwidth steering diagnostics for
  delay paths.

**Invariants.**

- Visibility is computed from the mesh, never from scene metadata or from a
  ray tracer's path output. Geometric line-of-sight truth therefore stays
  independent of the solver it is used to check.
- Scene generation is deterministic for a given parameter set.
- Nothing here imports a native ray-tracing backend; Mitsuba is only referenced
  as an XML format.
