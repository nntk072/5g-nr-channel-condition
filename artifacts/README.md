# Artifact storage

`outputs/runs/<run-id>/` is the canonical location for experiment runs. Each
run records its input hashes, resolved configuration, source revision,
dependencies, result summaries, and a `COMPLETE` marker after successful
generation. Consumers should reject runs without that marker.

Shared mechanics live in the importable `artifacts/` package. Run-specific
meaning remains with the package that produces the artifact.
