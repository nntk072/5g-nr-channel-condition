# Changelog

All notable changes are listed here, newest first. Versions follow `pyproject.toml`.

## Unreleased

- Renamed every schema identity, data key, signing namespace and identifier that
  carried an earlier gate number to a concept name. Readers of the renamed
  identities reject the old strings. Eight pinned reference files under
  `config/reference/` keep their original identity strings, because their bytes
  are pinned by digest.
- Removed unreachable validation branches from the lineage and path-oracle
  modules.
- Added package README files, a module-size record, and a command reference that
  lists only installed commands.

## 0.2.0

- Removed the agent control plane, the `loslab` command and its facades from the
  published tree. Reusable workflows moved to `scripts/` and to
  `integrations/sionna/scenario_gbsm_run.py`.
- Moved research campaign drivers, plan documents and record files out of the
  published tree.
- Reorganized `evaluation/` into `scoring/`, `covariance/`, `acceptance/` and
  `audits/`.
- Reference data read by the product moved to `config/reference/`.
- Removed the digest-rebinding module; digests are compared directly.

## 0.1.0

- Initial research framework: datasets, propagation, receivers, learning,
  evaluation and Sionna integrations.
