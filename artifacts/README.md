# artifacts

**Purpose.** Shared mechanics for immutable, content-addressed run artifacts:
run layout, identities, manifests with completion markers, build-DAG nodes, and
run replay provenance.

**Public API.**

- `artifacts.layout` — canonical paths for one immutable run under
  `outputs/runs/<run-id>/`.
- `artifacts.identity` — stable identities for resolved inputs and outputs.
- `artifacts.manifest` — run-manifest envelope and the `COMPLETE` marker.
- `artifacts.build_dag` — content-addressed build nodes for reproducible
  workflows.
- `artifacts.provenance` — source revision, dependency versions, and atomic JSON
  writing.
- `artifacts.run_replay` — non-qualifying replay provenance contracts.

**Invariants.**

- Run directories are write-once. A consumer must reject a run that lacks its
  `COMPLETE` marker.
- Writes are atomic: content is written to a temporary file and renamed only
  after it is complete.
- Artifact meaning stays with the package that produces it. This package owns
  only the mechanics.
