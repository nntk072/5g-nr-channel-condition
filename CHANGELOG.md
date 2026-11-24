# Changelog

All notable changes are listed here, newest first. Versions follow `pyproject.toml`.

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
